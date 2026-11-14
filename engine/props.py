"""Player prop projections from ESPN rosters + athlete game logs."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from . import espn
from .common import book_line, norm_name
from .espn import League

MARKETS = {
    "basketball": [("PTS", "Points"), ("REB", "Rebounds"), ("AST", "Assists"), ("PRA", "Pts+Reb+Ast")],
    "football": [("PASS", "Pass Yds"), ("RUSH", "Rush Yds"), ("REC", "Rec Yds"), ("RECS", "Receptions")],
    "hockey": [("G", "Goals"), ("A", "Assists"), ("SOG", "Shots"), ("PTS", "Points")],
}
# (lean, strong) edge thresholds: projection minus last-5 average
EDGE = {
    "basketball": {"PTS": (1.5, 3.0), "REB": (0.8, 1.6), "AST": (0.8, 1.6), "PRA": (2.0, 4.0)},
    "football": {"PASS": (15, 30), "RUSH": (8, 16), "REC": (8, 16), "RECS": (0.6, 1.2)},
    "hockey": {"G": (0.08, 0.16), "A": (0.1, 0.2), "SOG": (0.4, 0.8), "PTS": (0.12, 0.25)},
}
# minimum recent average for a player to be listed in a market
MIN_AVG = {"football": {"PASS": 60, "RUSH": 12, "REC": 12, "RECS": 1.2}, "hockey": {"SOG": 0.8}}
FOOTBALL_POS = {"QB": 3, "RB": 5, "FB": 1, "WR": 7, "TE": 4}
HEADSHOT = "https://a.espncdn.com/i/headshots/{league}/players/full/{id}.png"
MAX_MIN = {"NBA": 40.0, "NCAAB": 38.0}


def candidates(lg: League, team_id: str) -> list[dict]:
    try:
        ros = espn.roster_athletes(espn.team_roster(lg, team_id))
    except Exception:
        return []
    if lg.kind == "football":
        counts: dict[str, int] = {}
        out = []
        for a in ros:
            pos = a["pos"].upper()
            if pos in FOOTBALL_POS and counts.get(pos, 0) < FOOTBALL_POS[pos]:
                counts[pos] = counts.get(pos, 0) + 1
                out.append(a)
        return out
    if lg.kind == "hockey":
        return [a for a in ros if a["pos"].upper() not in ("G", "GOALIE", "GOALTENDER")]
    return ros


def _inj_lookup(inj: list[dict], roster_status: str, aid: str, name: str) -> str:
    key = norm_name(name)
    for i in inj:
        if (i["athlete_id"] and i["athlete_id"] == aid) or i["key"] == key:
            return i["status"]
    return roster_status or ""


def _wavg(vals: list[float], w_end: float = 0.55) -> float:
    if not vals:
        return 0.0
    w = np.linspace(1.0, w_end, len(vals))
    return float(np.dot(vals, w) / w.sum())


def _market_fields(row: dict, m: str, proj: float, hist: list[float], line: float | None = None) -> None:
    line = book_line(proj) if line is None else line
    row[f"P_{m}"] = proj
    row[f"L_{m}"] = line
    row[f"A5_{m}"] = float(np.mean(hist[:5])) if hist else 0.0
    row[f"V5_{m}"] = [float(x) for x in hist[:5]]
    row[f"V10_{m}"] = [float(x) for x in hist[:10]]
    row[f"H_{m}"] = float(np.mean([x > line for x in hist[:10]]) * 100) if hist else float("nan")


def build_props(lg: League, game: dict, proj: dict, profiles: dict, base: dict, b2b: set[str],
                inj_all: list[dict], q_sit: float) -> pd.DataFrame:
    before: date = game["date"]
    rows = []
    for side, opp_side in (("away", "home"), ("home", "away")):
        team, opp = game[side], game[opp_side]
        tp, op = profiles.get(team["id"]), profiles.get(opp["id"])
        if not tp or not op:
            continue
        inj = espn.team_injuries(inj_all, team)
        players = []
        for a in candidates(lg, team["id"]):
            logs = []
            raw = espn.cached_gamelog(lg, a["id"])
            if raw:
                try:
                    logs = espn.parse_gamelog(raw, lg.kind, before)
                except Exception:
                    logs = []
            season = None
            if lg.kind == "hockey" and len(logs) < 3:  # NHL gamelog can come back empty: use season rates
                sraw = espn.cache_read_json(espn.athlete_stats_key(lg, a["id"]))
                season = espn.hockey_season_rates(sraw) if sraw else None
            if not logs and not season:
                continue
            status = _inj_lookup(inj, a["inj_status"], a["id"], a["name"])
            cls = espn.injury_class(status)
            players.append((a, logs, status, cls, season))
        team_pts = proj["pts_away"] if side == "away" else proj["pts_home"]
        is_b2b = team["id"] in b2b
        common = {"Team": team["abbr"], "Opp": opp["abbr"], "TeamId": team["id"], "Home": side == "home",
                  "TeamLogo": team.get("logo", ""), "OppName": opp.get("short") or opp["abbr"]}
        if lg.kind == "basketball":
            team_rows = _basketball(lg, players, tp, op, base, proj, is_b2b, q_sit, common)
        elif lg.kind == "football":
            team_rows = _football(lg, players, op, base, team_pts, common)
        elif lg.kind == "hockey":
            team_rows = _hockey(lg, players, op, base, is_b2b, common)
        else:
            team_rows = []
        _attach_splits(lg, team_rows, players, opp["abbr"], side == "home", is_b2b)
        rows += team_rows
    return pd.DataFrame(rows)


def stat_value(kind: str, m: str, g: dict) -> float:
    """A market's value in one game-log row (PRA and hockey points are sums)."""
    if kind == "basketball" and m == "PRA":
        return g["PTS"] + g["REB"] + g["AST"]
    if kind == "hockey" and m == "PTS":
        return g["G"] + g["A"]
    return float(g.get(m, 0.0))


def _key_absence(lg: League, players: list) -> tuple[str, set] | None:
    """Most important teammate who is OUT tonight and has played this season: (name, event ids played)."""
    best, best_val = None, 0.0
    stat = {"basketball": "PTS", "football": "REC", "hockey": "G"}.get(lg.kind, "PTS")
    for a, logs, status, cls, _ in players:
        if cls != "OUT" or len(logs) < 3:
            continue
        val = float(np.mean([r.get(stat, 0.0) for r in logs]))
        if lg.kind == "football":
            val = float(np.mean([r.get("PASS", 0) / 10 + r.get("RUSH", 0) + r.get("REC", 0) for r in logs]))
        if val > best_val:
            best, best_val = (a["name"], {r["eid"] for r in logs}), val
    return best


def _attach_splits(lg: League, team_rows: list[dict], players: list, opp_abbr: str, home_tonight: bool,
                   b2b_tonight: bool) -> None:
    """Per market: season, last 10, vs tonight's opponent, tonight's venue, back-to-backs and without the key
    absent teammate — the "each split against the line" chart — plus last-10 game labels."""
    logs_by_id = {a["id"]: logs for a, logs, *_ in players}
    absent = _key_absence(lg, players)
    for row in team_rows:
        logs = logs_by_id.get(row["Id"]) or []
        dates = sorted(r["date"] for r in logs)
        b2b_days = {d for i, d in enumerate(dates) if i and (d - dates[i - 1]).days == 1}
        row["G10"] = [{"opp": r.get("opp", ""), "date": f"{r['date'].month}/{r['date'].day}",
                       "away": bool(r.get("away"))} for r in logs[:10]]
        spl = {}
        for m, _ in MARKETS.get(lg.kind, []):
            vals = lambda rows: [stat_value(lg.kind, m, r) for r in rows]  # noqa: E731
            out = []
            if logs:
                out.append(("Season average", float(np.mean(vals(logs))), len(logs)))
                out.append(("Last 10 games", float(np.mean(vals(logs[:10]))), min(len(logs), 10)))
            vs = [r for r in logs if r.get("opp") == opp_abbr]
            if vs:
                out.append((f"vs. {opp_abbr}", float(np.mean(vals(vs))), len(vs)))
            venue = [r for r in logs if bool(r.get("away")) != home_tonight]
            if len(venue) >= 2 and len(venue) < len(logs):
                out.append(("Home games" if home_tonight else "Away games", float(np.mean(vals(venue))), len(venue)))
            b2b = [r for r in logs if r["date"] in b2b_days]
            if len(b2b) >= 2:
                out.append(("On a back-to-back", float(np.mean(vals(b2b))), len(b2b)))
            if absent and absent[0] != row["Player"]:
                name, played = absent
                first = min((r["date"] for r in logs if r["eid"] in played), default=None)
                without = [r for r in logs if r["eid"] not in played and (first is None or r["date"] >= first)]
                if len(without) >= 2:
                    out.append((f"Without {name}", float(np.mean(vals(without))), len(without)))
            spl[m] = out
        row["SPL"] = spl
        row["B2BTonight"] = b2b_tonight


def _base_row(lg: League, a: dict, status: str, cls: str, common: dict) -> dict:
    return {"Player": a["name"], "Id": a["id"], "Pos": a["pos"],
            "Headshot": a["headshot"] or HEADSHOT.format(league=lg.league, id=a["id"]),
            "Status": status or "Active", "Class": cls, **common}


def _basketball(lg, players, tp, op, base, proj, is_b2b, q_sit, common) -> list[dict]:
    team_dates = set(tp["dates"][-10:])
    info = []
    for a, logs, status, cls, _ in players:
        l10 = logs[:10]
        mins = [r["MIN"] for r in l10]
        avg_min = float(np.mean(mins)) if mins else 0.0
        if len(l10) < 2 or avg_min < 8:
            continue
        avail = len({r["date"] for r in l10} & team_dates) / len(team_dates) if team_dates else 0.0
        sit = 1.0 if cls == "OUT" else (q_sit if cls == "Q" else 0.0)
        info.append((a, l10, status, cls, avg_min, avail, sit))
    if not info:
        return []
    lost = {k: sum(i[6] * i[5] * float(np.mean([r[k] for r in i[1]])) for i in info)
            for k in ("MIN", "PTS", "REB", "AST")}
    active = [i for i in info if i[3] != "OUT"]
    denom = sum(i[4] * (1 - i[6]) for i in active) or 1.0
    team_tot = {k: sum(float(np.mean([r[k] for r in i[1]])) * (1 - i[6]) for i in active) + lost[k] or 1.0
                for k in ("PTS", "REB", "AST")}
    usg = {"PTS": 1 + 0.30 * lost["PTS"] / team_tot["PTS"], "REB": 1 + 0.20 * lost["REB"] / team_tot["REB"],
           "AST": 1 + 0.15 * lost["AST"] / team_tot["AST"]}
    poss = proj.get("poss")
    pace_f = poss / tp["pace"] if poss and tp.get("pace") else 1.0
    if op.get("pace") and base.get("pace"):
        pts_f = (op["pa"] / op["pace"]) / (base["pts"] / base["pace"])
    else:
        pts_f = op["pa"] / base["pts"]
    pts_f = max(0.85, min(1.15, pts_f))
    cap = MAX_MIN.get(lg.key, 40.0)
    rows = []
    for a, l10, status, cls, avg_min, avail, sit in active:
        pm = min(avg_min + lost["MIN"] * avg_min / denom, max(avg_min, cap))
        if is_b2b:
            pm *= 0.96
        wmin = _wavg([r["MIN"] for r in l10]) or 1.0
        rate = {k: _wavg([r[k] for r in l10]) / wmin for k in ("PTS", "REB", "AST")}
        p = {"PTS": rate["PTS"] * pm * pace_f * pts_f * usg["PTS"] * (0.98 if is_b2b else 1.0),
             "REB": rate["REB"] * pm * pace_f * usg["REB"],
             "AST": rate["AST"] * pm * pace_f * usg["AST"]}
        row = _base_row(lg, a, status, cls, common)
        row["Info"] = f"{pm:.0f} min proj"
        for k in ("PTS", "REB", "AST"):
            _market_fields(row, k, p[k], [r[k] for r in l10])
        _market_fields(row, "PRA", p["PTS"] + p["REB"] + p["AST"], [r["PTS"] + r["REB"] + r["AST"] for r in l10])
        rows.append(row)
    return rows


def _football(lg, players, op, base, team_pts, common) -> list[dict]:
    opp_f = max(0.8, min(1.2, (op["pa"] / base["pts"]) ** 0.5))
    game_f = max(0.85, min(1.15, (team_pts / base["pts"]) ** 0.3))
    rows = []
    for a, logs, status, cls, _ in players:
        if cls == "OUT":
            continue
        l8 = logs[:8]
        if not l8:
            continue
        row = _base_row(lg, a, status, cls, common)
        row["Info"] = f"{a['pos']} · {len(l8)} recent games"
        mult = {"PASS": opp_f * game_f, "REC": opp_f * game_f, "RECS": opp_f ** 0.5 * game_f,
                "RUSH": opp_f * game_f ** 0.5}
        for k in ("PASS", "RUSH", "REC", "RECS"):
            hist = [r[k] for r in l8]
            _market_fields(row, k, _wavg(hist, 0.6) * mult[k], hist)
        rows.append(row)
    return rows


def _hockey(lg, players, op, base, is_b2b, common) -> list[dict]:
    opp_f = max(0.8, min(1.25, op["pa"] / base["pts"]))
    fat = 0.97 if is_b2b else 1.0
    rows = []
    for a, logs, status, cls, season in players:
        if cls == "OUT":
            continue
        l10 = logs[:10]
        if len(l10) < 3:
            if season and season["gp"] >= 3:
                rows.append(_hockey_season_row(lg, a, status, cls, common, season, opp_f, fat))
            continue
        row = _base_row(lg, a, status, cls, common)
        toi = float(np.mean([r["TOI"] for r in l10])) if l10 else 0.0
        row["Info"] = f"{int(toi)}:{int((toi % 1) * 60):02d} TOI" if toi else f"{len(l10)} recent games"
        g = _wavg([r["G"] for r in l10]) * opp_f * fat
        ast = _wavg([r["A"] for r in l10]) * opp_f * fat
        sog = _wavg([r["SOG"] for r in l10]) * opp_f ** 0.5 * fat
        _market_fields(row, "G", g, [r["G"] for r in l10], line=0.5)
        _market_fields(row, "A", ast, [r["A"] for r in l10], line=0.5)
        _market_fields(row, "SOG", sog, [r["SOG"] for r in l10])
        _market_fields(row, "PTS", g + ast, [r["G"] + r["A"] for r in l10], line=0.5)
        row["AnyGoal"] = 1 - math.exp(-g)
        rows.append(row)
    return rows


def _hockey_season_row(lg, a, status, cls, common, season, opp_f, fat) -> dict:
    """Projection from season per-game rates when no per-game log is available (no last-5 history)."""
    row = _base_row(lg, a, status, cls, common)
    row["Info"] = f"season avg · {season['gp']:.0f} GP"
    g, ast, sog = season["G"] * opp_f * fat, season["A"] * opp_f * fat, season["SOG"] * opp_f ** 0.5 * fat
    for m, p, avg, line in (("G", g, season["G"], 0.5), ("A", ast, season["A"], 0.5),
                            ("SOG", sog, season["SOG"], None), ("PTS", g + ast, season["G"] + season["A"], 0.5)):
        _market_fields(row, m, p, [], line=line)
        row[f"A5_{m}"] = avg  # season average stands in for the last-5 average
    row["AnyGoal"] = 1 - math.exp(-g)
    return row


def lost_points(lg: League, game_team: dict, inj_all: list[dict], q_sit: float, before: date) -> float:
    """Scoring lost to injuries (basketball): season PPG of Out (and Questionable x sit-chance) players,
    from cached game logs when available, else from the team's scoreboard leaders."""
    if lg.kind != "basketball":
        return 0.0
    total = 0.0
    for i in espn.team_injuries(inj_all, game_team):
        cls = espn.injury_class(i["status"])
        w = 1.0 if cls == "OUT" else (q_sit if cls == "Q" else 0.0)
        if w == 0:
            continue
        ppg = None
        raw = espn.cached_gamelog(lg, i["athlete_id"]) if i["athlete_id"] else None
        if raw:
            try:
                logs = espn.parse_gamelog(raw, lg.kind, before)
                if logs:
                    recent = [r for r in logs if (before - r["date"]).days <= 21]
                    ppg = float(np.mean([r["PTS"] for r in logs])) if recent else 0.0  # long-term absences are priced in
            except Exception:
                ppg = None
        if ppg is None:
            for ld in game_team.get("leaders", []):
                if "point" in ld["cat"] and (ld["id"] == i["athlete_id"] or norm_name(ld["name"]) == i["key"]):
                    ppg = ld["value"]
        total += w * (ppg or 0.0)
    return total
