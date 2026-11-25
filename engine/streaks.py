"""Hit-rate streaks ("100% hit rate props"): the highest round-number line a player / team / fighter has
cleared in every one of their recent games, and how far back the run goes.

For each player and market the candidate lines are the floors of the last K games (K = 5 … all). Each is
scored by the streak length x (line / season average)^2, so a long run at a trivial line (4+ points for a
scorer) doesn't beat a solid line cleared in 8 straight. Only lines at or above a per-market floor count.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from . import espn
from .props import MARKETS, _inj_lookup, candidates, stat_value

MIN_STREAK = 5
STALE_DAYS = 30  # last game more than this long before tonight: not part of the current rotation
FLOOR = {
    "basketball": {"PTS": 5, "REB": 3, "AST": 2, "PRA": 10},
    "football": {"PASS": 125, "RUSH": 15, "REC": 15, "RECS": 2},
    "hockey": {"G": 1, "A": 1, "SOG": 2, "PTS": 1},
}
STEP = {("football", "PASS"): 5, ("football", "RUSH"): 5, ("football", "REC"): 5}


def best_streak(vals: list[float], floor: float, step: float = 1, min_k: int = MIN_STREAK) -> dict | None:
    """vals newest first. Returns {t, k, n, hits, avg} for the best 't+' line cleared in the last k games."""
    n = len(vals)
    if n < min_k:
        return None
    avg = float(np.mean(vals)) or 1e-9
    best, seen = None, set()
    for k_min in range(min_k, n + 1):
        t = float(np.floor(min(vals[:k_min]) / step) * step)
        if t < floor or t <= 0:
            break  # the floor only falls as the window grows
        if t in seen:
            continue
        seen.add(t)
        k = next((i for i, v in enumerate(vals) if v < t), n)
        score = k * (t / avg) ** 2
        if best is None or score > best["score"]:
            best = {"t": t, "k": k, "n": n, "hits": sum(v >= t for v in vals), "avg": avg, "score": score}
    return best


MIN_RATE = 0.8  # "hit-rate" lines: cleared in at least 80% of a recent window (not every game)


def best_rate_line(vals: list[float], floor: float, step: float = 1, min_rate: float = MIN_RATE,
                   min_w: int = MIN_STREAK, skip_t: float | None = None) -> dict | None:
    """Best line cleared in >= min_rate of the last w games (w >= min_w), allowing misses. Scored by
    window x rate^4 x (line / average)^2 so a higher line hit 8/10 can beat a trivial one hit 10/10.
    Only lines that aren't already a perfect streak (rate < 1) and differ from skip_t are returned."""
    n = len(vals)
    if n < min_w:
        return None
    avg = float(np.mean(vals)) or 1e-9
    best = None
    for t in sorted({float(np.floor(v / step) * step) for v in vals}):
        if t < floor or t <= 0 or t == skip_t:
            continue
        hit = [v >= t for v in vals]
        w = next((w for w in range(n, min_w - 1, -1) if sum(hit[:w]) / w >= min_rate), None)
        if w is None:
            continue
        wh = sum(hit[:w])
        rate = wh / w
        if rate >= 0.999:
            continue
        score = w * rate ** 4 * (t / avg) ** 2
        if best is None or score > best["score"]:
            best = {"t": t, "w": w, "wh": wh, "rate": rate, "k": run_length(hit), "n": n, "hits": sum(hit),
                    "avg": avg, "score": score}
    return best


def _norm(r: dict) -> dict:
    """Perfect streaks: window = streak length, all hits."""
    r.setdefault("w", r["k"])
    r.setdefault("wh", r["k"])
    r.setdefault("rate", 1.0)
    return r


def _md(iso) -> str:
    try:
        y, m, d = str(iso)[:10].split("-")
        return f"{int(m)}/{int(d)}"
    except Exception:
        return ""


def run_length(flags: list[bool]) -> int:
    return next((i for i, f in enumerate(flags) if not f), len(flags))


def player_streaks(lg, game: dict, inj_all: list[dict]) -> list[dict]:
    """Every qualifying player/market streak for one game, from cached ESPN game logs."""
    kind = lg.kind
    before: date = game["date"]
    out = []
    for side, opp_side in (("away", "home"), ("home", "away")):
        team, opp = game[side], game[opp_side]
        inj = espn.team_injuries(inj_all, team)
        for a in candidates(lg, team["id"]):
            raw = espn.cached_gamelog(lg, a["id"])
            if not raw:
                continue
            try:
                logs = espn.parse_gamelog(raw, kind, before)
            except Exception:
                continue
            if kind == "basketball":
                logs = [r for r in logs if r.get("MIN", 1) > 0]
            if len(logs) < MIN_STREAK or (before - logs[0]["date"]) > timedelta(days=STALE_DAYS):
                continue
            status = _inj_lookup(inj, a["inj_status"], a["id"], a["name"])
            cls = espn.injury_class(status)
            if cls == "OUT":
                continue
            for m, label in MARKETS.get(kind, []):
                vals = [stat_value(kind, m, r) for r in logs]
                fl, st = FLOOR.get(kind, {}).get(m, 1), STEP.get((kind, m), 1)
                perfect = best_streak(vals, fl, st)
                found = [x for x in (perfect, best_rate_line(vals, fl, st, skip_t=perfect["t"] if perfect else None))
                         if x]
                for s in found:
                    out.append(_norm({**s, "kind": "player", "m": m, "label": label, "name": a["name"], "id": a["id"],
                            "team": team["abbr"], "opp": opp["abbr"], "home": side == "home", "gid": game["id"],
                            "headshot": a.get("headshot") or "", "logo": team.get("logo", ""),
                            "status": status if cls not in ("", "ACTIVE") else "", "last": vals[:10],
                            "opps": [r.get("opp", "") for r in logs[:10]],
                            "dates": [f'{r["date"].month}/{r["date"].day}' for r in logs[:10]]}))
    return out


TEAM_FLOOR = {"basketball": (95, 5), "football": (14, 1), "hockey": (2, 1), "soccer": (1, 1)}
SOCCER_FLAGS = [("Scored 1+ goal", lambda g: g["pf"] >= 1), ("Over 1.5 match goals", lambda g: g["pf"] + g["pa"] >= 2),
                ("Over 2.5 match goals", lambda g: g["pf"] + g["pa"] >= 3),
                ("Both teams scored", lambda g: g["pf"] >= 1 and g["pa"] >= 1),
                ("Unbeaten", lambda g: g["pf"] >= g["pa"]), ("Won", lambda g: g["pf"] > g["pa"]),
                ("Clean sheet", lambda g: g["pa"] == 0)]
TEAM_MIN = {"soccer": 4}


def team_streaks(lg, game: dict, profiles: dict) -> list[dict]:
    """Team runs from the profile's game log: points/goals scored line, wins, and soccer match markets."""
    kind = lg.kind
    out = []
    unit = {"basketball": "Points", "football": "Points", "hockey": "Goals", "soccer": "Goals"}.get(kind, "Points")
    min_k = TEAM_MIN.get(kind, MIN_STREAK)
    for side, opp_side in (("away", "home"), ("home", "away")):
        team, opp = game[side], game[opp_side]
        prof = profiles.get(team["id"]) or {}
        log = prof.get("log") or []
        if len(log) < min_k:
            continue
        base = {"kind": "team", "name": team.get("short") or team["abbr"], "id": team["id"], "team": team["abbr"],
                "opp": opp["abbr"], "home": side == "home", "gid": game["id"], "headshot": "",
                "logo": team.get("logo", ""), "status": "", "n": len(log), "league": lg.key}
        if kind != "soccer":
            floor, step = TEAM_FLOOR.get(kind, (1, 1))
            s = best_streak([g["pf"] for g in log], floor, step, min_k)
            if s:
                out.append({**s, **base, "m": "TEAM", "label": f"Team {unit}",
                            "last": [g["pf"] for g in log[:10]], "opps": [],
                            "dates": [_md(g.get("date")) for g in log[:10]]})
            flags = [("Won", lambda g: g["pf"] > g["pa"])]
        else:
            flags = SOCCER_FLAGS
        for label, f in flags:
            fl = [f(g) for g in log]
            k = run_length(fl)
            if k >= min_k:
                out.append({**base, "m": "TEAMRUN", "label": label, "t": None, "k": k, "hits": sum(fl),
                            "avg": None, "score": k, "flag": label,
                            "last": [f"{g['pf']:.0f}-{g['pa']:.0f}" for g in log[:10]], "opps": [],
                            "dates": [_md(g.get("date")) for g in log[:10]]})
    return [_norm(r) for r in out]


def fighter_streaks(bouts: list[dict]) -> list[dict]:
    """UFC: current win / finish / KO / submission runs from each fighter's recent bouts (ESPN history)."""
    out = []
    for x in bouts:
        b = x["bout"]
        if b.get("state") == "post":
            continue
        for prof, card_f, opp in ((x["a"], b["fighters"][0], x["b"]), (x["b"], b["fighters"][1], x["a"])):
            rec = prof.get("recent") or []
            if len(rec) < 2:
                continue
            base = {"kind": "fighter", "name": prof["name"], "id": card_f.get("id", ""), "team": "",
                    "opp": opp["name"], "gid": b["id"], "headshot": prof.get("headshot", ""), "logo": "",
                    "status": "", "n": len(rec), "t": None, "avg": None,
                    "last": [f'{h["result"]} {h["method"]}'.strip() for h in rec], "opps": [h["opponent"] for h in rec],
                    "dates": [_md(h.get("_d")) for h in rec]}
            for label, f, min_k in (("Win streak", lambda h: h["result"] == "W", 2),
                                    ("Wins inside the distance", lambda h: h["result"] == "W" and h["code"] in ("KO", "SUB"), 2),
                                    ("Wins by KO/TKO", lambda h: h["result"] == "W" and h["code"] == "KO", 2),
                                    ("Wins by submission", lambda h: h["result"] == "W" and h["code"] == "SUB", 2),
                                    ("Fights went the distance", lambda h: h["code"] == "DEC", 3),
                                    ("Fights ended inside the distance", lambda h: h["code"] in ("KO", "SUB"), 3)):
                fl = [f(h) for h in rec]
                k = run_length(fl)
                if k >= min_k:
                    out.append({**base, "m": label, "label": label, "k": k, "hits": sum(fl), "score": k})
    return [_norm(r) for r in out]


# --------------------------------------------------------------------------
# Streak parlays
# --------------------------------------------------------------------------
def leg_prob(r: dict, proj: float | None) -> float:
    """Chance the player clears his streak line tonight: smoothed season hit rate at that line, averaged with the
    model projection's normal tail (sd from his last 10 games) when a projection exists. Capped at 97%."""
    hits, n, t = r["hits"], r["n"], r["t"]
    p_hist = (hits + 1) / (n + 2)
    if proj is None or proj != proj:
        return min(0.97, p_hist)
    vals = r.get("last") or []
    mean = sum(vals) / len(vals) if vals else proj
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) if len(vals) > 1 else 0.0
    sd = max(sd, 0.15 * max(proj, 1e-9), 0.8)
    z = (t - 0.5 - proj) / sd  # integer stats: clearing t means beating t - 0.5
    p_model = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return min(0.97, 0.5 * p_hist + 0.5 * p_model)


def build_parlay(rows: list[dict], proj: dict, legs: int, markets: set[str] | None = None, min_k: int = 5,
                 skip_questionable: bool = True, min_rate: float = 1.0, tune: dict | None = None) -> list[dict]:
    """The `legs` most likely streak legs: one per player (his likeliest line), any game tonight.
    min_k = minimum window (games), min_rate = minimum hit rate in that window (1.0 = perfect streaks only)."""
    best: dict[str, dict] = {}
    for r in rows:
        if (r["kind"] != "player" or r.get("w", r["k"]) < min_k or r.get("rate", 1.0) < min_rate - 1e-9
                or (markets and r["m"] not in markets)):
            continue
        if skip_questionable and r.get("status"):
            continue
        p = leg_prob(r, proj.get((r["id"], r["m"])))
        if tune:  # calibration from your graded results (engine.feedback)
            p = min(0.97, p * tune.get(r["m"], tune.get("_default", 1.0)))
        if r["id"] not in best or p > best[r["id"]]["p"]:
            best[r["id"]] = {**r, "p": p, "proj": proj.get((r["id"], r["m"]))}
    picks = sorted(best.values(), key=lambda x: (-x["p"], -x.get("w", x["k"])))
    return picks[:legs]


def parlay_prob(picks: list[dict]) -> float:
    out = 1.0
    for x in picks:
        out *= x["p"]
    return out
