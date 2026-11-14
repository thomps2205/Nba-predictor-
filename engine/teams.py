"""Team ratings from ESPN schedules/team stats and per-sport game projections."""

from __future__ import annotations

import math
from datetime import date, timedelta

from . import espn
from .common import NOTES, flag_active, flatten_stats, norm_cdf, pick, set_flag
from .espn import League

# Per-sport model settings
CFG = {
    "NBA": dict(hca=2.2, sigma=12.0, b2b=True, shrink=4, lg_pts=114.0, lg_pace=99.5),
    "NCAAB": dict(hca=3.2, sigma=11.0, b2b=False, shrink=3, lg_pts=73.0, lg_pace=68.0),
    "NFL": dict(hca=1.6, sigma=13.5, b2b=False, shrink=2, lg_pts=22.5),
    "NHL": dict(home_goals=0.12, b2b=True, shrink=5, lg_pts=3.05),
}
SOCCER_CFG = dict(home_goals=0.30, b2b=False, shrink=4, lg_pts=1.35)


def cfg(key: str) -> dict:
    """Model settings for a league key (every soccer competition shares SOCCER_CFG)."""
    return CFG.get(key, SOCCER_CFG)
B2B_OWN = {"NBA": 1.2, "NHL": 0.15}     # points/goals lost on the 2nd night of a back-to-back
B2B_OPP = {"NBA": 1.0, "NHL": 0.10}     # extra points/goals conceded
INJ_PTS_FACTOR = 0.15                    # team points lost per missing player-point (replacement recovers the rest)
STATS_DOWN_FLAG = "stats_nba_down.flag"
STATS_DOWN_TTL = 6 * 3600


def needed_requests(lg: League, team_id: str) -> list[tuple[str, float, callable]]:
    """(cache-key, ttl, fetcher) a team profile needs — the UI prefetches these behind a progress bar."""
    reqs = [(espn.schedule_key(lg, team_id), espn.TTL_SCHEDULE, lambda: espn.team_schedule(lg, team_id))]
    if lg.kind == "basketball":
        reqs.append((f"teamstats|{lg.key}|{team_id}", espn.TTL_TEAM_STATS, lambda: espn.team_statistics(lg, team_id)))
    return reqs


MIN_SOCCER_GAMES = 3
TARGET_SOCCER_GAMES = 6


def _soccer_form(lg: League, team_id: str, before: date, games: list[dict]) -> list[dict]:
    """Cup ties and national teams often have < 3 matches under that competition: merge in the team's results
    from ESPN's all-competitions schedule, then likely domestic leagues (or other international competitions)."""
    if len(games) >= MIN_SOCCER_GAMES:
        return games
    seen = {(g["date"], g["opp"]) for g in games}
    merged = list(games)
    for slug in ["all", *espn.SOCCER_FORM_FALLBACK.get(lg.key, [])]:
        if slug == lg.league:
            continue
        try:
            extra = espn.parse_completed_games(espn.soccer_schedule(slug, team_id), team_id, before)
        except Exception:
            continue
        for g in extra:
            if (g["date"], g["opp"]) not in seen:
                seen.add((g["date"], g["opp"]))
                merged.append(g)
        if len(merged) >= TARGET_SOCCER_GAMES:
            break
    merged.sort(key=lambda g: g["date"])
    return merged[-20:]  # recent form only


def team_profile(lg: League, team_id: str, before: date) -> dict | None:
    try:
        games = espn.parse_completed_games(espn.team_schedule(lg, team_id), team_id, before)
    except Exception as e:
        games = []
        if lg.kind != "soccer":
            NOTES[f"schedule {lg.key}"] = f"Team schedule unavailable ({e})"
            return None
    if lg.kind == "soccer":
        games = _soccer_form(lg, team_id, before, games)
    if not games:
        return None
    pf = [g["pf"] for g in games]
    pa = [g["pa"] for g in games]
    l5f, l5a = pf[-5:], pa[-5:]
    prof = {
        "gp": len(games), "pf": sum(pf) / len(pf), "pa": sum(pa) / len(pa),
        "pf_l5": sum(l5f) / len(l5f), "pa_l5": sum(l5a) / len(l5a),
        "last_date": games[-1]["date"], "dates": [g["date"] for g in games],
        "w": sum(g["pf"] > g["pa"] for g in games), "l": sum(g["pf"] < g["pa"] for g in games),
        "pace": None,
    }
    prof.update(period_rates(lg, games))
    if lg.kind == "basketball":
        try:
            prof["pace"] = espn.team_possessions(espn.team_statistics(lg, team_id), prof["gp"])
        except Exception:
            prof["pace"] = None
    return prof


def nba_api_ratings(season: str) -> dict[str, dict]:
    """Optional: stats.nba.com advanced ratings via nba_api. Strict try/except, 5s timeout, 6h skip on failure."""
    if flag_active(STATS_DOWN_FLAG, STATS_DOWN_TTL):
        return {}
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        from nba_api.stats.static import teams as static_teams

        from .common import BROWSER_UA, throttle

        headers = {
            "Host": "stats.nba.com", "User-Agent": BROWSER_UA,
            "Accept": "application/json, text/plain, */*", "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com", "x-nba-stats-origin": "stats", "x-nba-stats-token": "true",
        }
        throttle()
        d = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame",
            headers=headers, timeout=5).get_dict()
        rs = d["resultSets"][0]
        rows = [dict(zip(rs["headers"], r)) for r in rs["rowSet"]]
        by_id = {t["id"]: t for t in static_teams.get_teams()}
        out = {}
        for r in rows:
            t = by_id.get(int(r["TEAM_ID"]))
            if t:
                out[t["nickname"].lower()] = {"pace": float(r["PACE"]), "off": float(r["OFF_RATING"]),
                                              "def": float(r["DEF_RATING"])}
        return out
    except Exception as e:
        set_flag(STATS_DOWN_FLAG)
        NOTES["stats.nba.com"] = f"stats.nba.com unavailable ({type(e).__name__}); using ESPN-derived pace/ratings."
        return {}


def _adv_lookup(adv: dict, display_name: str) -> dict | None:
    dn = display_name.lower()
    return next((v for k, v in adv.items() if dn.endswith(k)), None)


# --------------------------------------------------------------------------
# Period / half splits
# --------------------------------------------------------------------------
# Regulation periods per sport and league-typical scoring shares (fallback when a game has no linescores).
DEFAULT_SHARES = {
    "NBA": [0.252, 0.249, 0.251, 0.248], "NFL": [0.18, 0.31, 0.19, 0.32], "NCAAB": [0.48, 0.52],
    "NHL": [0.31, 0.34, 0.35], "soccer": [0.44, 0.56],
}
TOTAL_SD = {"NBA": 17.0, "NCAAB": 15.0, "NFL": 13.0}
PERIOD_GAMES = 15


def _shares_key(lg: League) -> str:
    return "soccer" if lg.kind == "soccer" else lg.key


def period_rates(lg: League, games: list[dict]) -> dict:
    """Average points/goals scored and allowed per regulation period over the last PERIOD_GAMES games, from ESPN
    linescores; games without usable linescores fall back to league-typical shares of the final score."""
    shares = DEFAULT_SHARES.get(_shares_key(lg))
    if not shares or not games:
        return {}
    n = len(shares)
    recent = games[-PERIOD_GAMES:]
    pf_p, pa_p, real = [0.0] * n, [0.0] * n, 0
    for g in recent:
        lf, la = list(g.get("ls_for") or []), list(g.get("ls_against") or [])
        if lg.kind == "soccer" and lf and la:  # 2nd half = full score - 1st half (drops extra time / pens)
            lf = [lf[0], max(g["pf"] - lf[0], 0.0)]
            la = [la[0], max(g["pa"] - la[0], 0.0)]
        usable = len(lf) >= n and len(la) >= n and (
            sum(lf[:n]) + sum(la[:n]) > 0 or lg.kind in ("hockey", "soccer"))  # 0-0 periods are real data
        if usable:
            lf, la = lf[:n], la[:n]
            real += 1
        else:
            lf = [g["pf"] * x for x in shares]
            la = [g["pa"] * x for x in shares]
        for i in range(n):
            pf_p[i] += lf[i]
            pa_p[i] += la[i]
    k = len(recent)
    return {"per_for": [x / k for x in pf_p], "per_against": [x / k for x in pa_p],
            "ls_real": real, "ls_games": k}


def period_segments(lg: League) -> list[tuple[str, str, list[int] | None]]:
    """(code, label, period indices); None = full game."""
    if lg.key in ("NBA", "NFL"):
        return [("FG", "Full Game", None), ("1H", "1st Half", [0, 1]), ("2H", "2nd Half", [2, 3]),
                ("1Q", "1Q", [0]), ("2Q", "2Q", [1]), ("3Q", "3Q", [2]), ("4Q", "4Q", [3])]
    if lg.key == "NHL":
        return [("FG", "Full Game", None), ("1P", "1st Period", [0]), ("2P", "2nd Period", [1]),
                ("3P", "3rd Period", [2])]
    if lg.key == "NCAAB" or lg.kind == "soccer":
        return [("FG", "Full Game", None), ("1H", "1st Half", [0]), ("2H", "2nd Half", [1])]
    return [("FG", "Full Game", None)]


def _american(p: float) -> str:
    if p <= 0 or p >= 1:
        return "—"
    return f"-{100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def _half_line(x: float) -> float:
    return round(x - 0.5) + 0.5


def period_projection(lg: League, proj: dict, ph: dict, pa_: dict, idx: list[int]) -> dict | None:
    """Projection for a set of regulation periods. Splits the full-game projection by the home team's scoring
    share and the away team's conceding share for those periods (and vice versa), then prices the segment:
    normal margin/total for basketball & football (with push on a tie), independent Poisson per side (i.e.
    bivariate Poisson with no correlation term) for hockey & soccer."""
    if not proj or not ph or not pa_ or "per_for" not in ph or "per_for" not in pa_:
        return None

    league = DEFAULT_SHARES.get(_shares_key(lg), [])
    base_share = sum(league[i] for i in idx) if league else len(idx) / max(len(ph["per_for"]), 1)

    def share(vals, games):
        tot = sum(vals)
        team = sum(vals[i] for i in idx) / tot if tot > 0 else base_share
        w = games / (games + 5)  # small samples lean on league-typical splits
        return w * team + (1 - w) * base_share

    gh, ga = ph.get("ls_games", 0), pa_.get("ls_games", 0)
    sh_h = 0.5 * share(ph["per_for"], gh) + 0.5 * share(pa_["per_against"], ga)
    sh_a = 0.5 * share(pa_["per_for"], ga) + 0.5 * share(ph["per_against"], gh)
    lh, la = proj["pts_home"] * sh_h, proj["pts_away"] * sh_a
    out = {"pts_home": lh, "pts_away": la, "total": lh + la, "spread_home": -(lh - la),
           "share": 0.5 * (sh_h + sh_a),  # segment's share of full-game scoring (for implied market lines)
           "real": min(ph.get("ls_real", 0), pa_.get("ls_real", 0)), "games": min(ph.get("ls_games", 0),
                                                                                   pa_.get("ls_games", 0))}
    if lg.kind in ("hockey", "soccer"):
        p_h = p_d = p_a = btts = 0.0
        tot = {}
        for i in range(10):
            for j in range(10):
                p = _poisson(i, max(lh, 0.01)) * _poisson(j, max(la, 0.01))
                if i > j:
                    p_h += p
                elif i == j:
                    p_d += p
                else:
                    p_a += p
                if i and j:
                    btts += p
                tot[i + j] = tot.get(i + j, 0.0) + p
        z = p_h + p_d + p_a
        out.update(home_win=p_h / z, draw=p_d / z, away_win=p_a / z, btts=btts / z, three_way=True,
                   over={line: sum(v for k, v in tot.items() if k > line) / z for line in (0.5, 1.5, 2.5)})
        out["total_line"] = 0.5 if out["total"] < 1.0 else 1.5 if out["total"] < 2.2 else 2.5
        out["p_over"] = out["over"][out["total_line"]]
    else:
        frac = max(0.5 * (sh_h + sh_a), 0.05)
        sd_m = cfg(lg.key)["sigma"] * math.sqrt(frac)
        sd_t = TOTAL_SD.get(lg.key, 13.0) * math.sqrt(frac)
        mu = lh - la
        p_h = 1 - norm_cdf((0.5 - mu) / sd_m)
        p_a = norm_cdf((-0.5 - mu) / sd_m)
        out.update(home_win=p_h, away_win=p_a, draw=max(0.0, 1 - p_h - p_a), three_way=False)
        out["total_line"] = _half_line(out["total"])
        out["p_over"] = 1 - norm_cdf((out["total_line"] - out["total"]) / sd_t)
    two = out["home_win"] / max(out["home_win"] + out["away_win"], 1e-9)  # push / draw refunded
    out["fair_home"], out["fair_away"] = _american(two), _american(1 - two)
    out["fair_home_3w"] = _american(out["home_win"]) if out["three_way"] else out["fair_home"]
    out["fair_away_3w"] = _american(out["away_win"]) if out["three_way"] else out["fair_away"]
    out["fair_draw"] = _american(out["draw"]) if out["three_way"] else ""
    out["spread_line"] = _half_line(abs(lh - la)) if abs(lh - la) >= 0.25 else 0.0
    return out


def full_as_segment(lg: League, proj: dict) -> dict:
    """Full-game projection in the same shape as period_projection (for the breakdown drawer)."""
    three = lg.kind == "soccer"
    out = {"pts_home": proj["pts_home"], "pts_away": proj["pts_away"], "total": proj["total"],
           "spread_home": proj["spread_home"], "three_way": three}
    if three:
        out.update(home_win=proj["home_win_prob"], draw=proj["draw_prob"], away_win=proj["away_win_prob"])
    else:
        out.update(home_win=proj["home_win_prob"], draw=0.0, away_win=1 - proj["home_win_prob"])
    return out


def league_baseline(key: str, profiles: dict[str, dict]) -> dict:
    cfg = globals()["cfg"](key)
    ps = [p for p in profiles.values() if p]
    lg_pts = sum(p["pf"] for p in ps) / len(ps) if len(ps) >= 6 else cfg["lg_pts"]
    paces = [p["pace"] for p in ps if p.get("pace")]
    lg_pace = sum(paces) / len(paces) if len(paces) >= 6 else cfg.get("lg_pace")
    return {"pts": lg_pts, "pace": lg_pace}


def _blend(p: dict, k: str, lg_val: float, shrink: int) -> float:
    season = p[k]
    recent = p[f"{k}_l5"]
    v = 0.7 * season + 0.3 * recent if p["gp"] >= 6 else season
    return (p["gp"] * v + shrink * lg_val) / (p["gp"] + shrink)  # regress small samples toward league average


def _poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def project_game(lg: League, game: dict, ph: dict | None, pa_: dict | None, base: dict,
                 b2b: set[str], lost_pts: dict[str, float], nba_adv: dict | None = None) -> dict | None:
    if not ph or not pa_:
        return None
    key = lg.key
    cfg = globals()["cfg"](key)
    hid, aid = game["home"]["id"], game["away"]["id"]
    lg_pts = base["pts"]
    off_h, def_h = _blend(ph, "pf", lg_pts, cfg["shrink"]), _blend(ph, "pa", lg_pts, cfg["shrink"])
    off_a, def_a = _blend(pa_, "pf", lg_pts, cfg["shrink"]), _blend(pa_, "pa", lg_pts, cfg["shrink"])
    out: dict = {"poss": None}

    if lg.kind == "soccer":
        hg = cfg["home_goals"] if not game["neutral"] else 0.0
        lam_h = max(off_h * def_a / lg_pts + hg / 2, 0.15)
        lam_a = max(off_a * def_h / lg_pts - hg / 2, 0.15)
        p_h = p_d = p_a = btts = 0.0
        for i in range(11):
            for j in range(11):
                p = _poisson(i, lam_h) * _poisson(j, lam_a)
                if i > j:
                    p_h += p
                elif i == j:
                    p_d += p
                else:
                    p_a += p
                if i > 0 and j > 0:
                    btts += p
        tot = p_h + p_d + p_a
        out.update(pts_home=lam_h, pts_away=lam_a, total=lam_h + lam_a, spread_home=-(lam_h - lam_a),
                   home_win_prob=p_h / tot, draw_prob=p_d / tot, away_win_prob=p_a / tot, btts_prob=btts / tot)
        return out

    if lg.kind == "hockey":
        hg = cfg["home_goals"] if not game["neutral"] else 0.0
        lam_h = off_h * def_a / lg_pts + hg / 2
        lam_a = off_a * def_h / lg_pts - hg / 2
        if hid in b2b:
            lam_h -= B2B_OWN["NHL"]; lam_a += B2B_OPP["NHL"]
        if aid in b2b:
            lam_a -= B2B_OWN["NHL"]; lam_h += B2B_OPP["NHL"]
        lam_h, lam_a = max(lam_h, 1.2), max(lam_a, 1.2)
        p_h = p_a = p_t = 0.0
        p_h2 = p_a2 = 0.0
        for i in range(12):
            for j in range(12):
                p = _poisson(i, lam_h) * _poisson(j, lam_a)
                if i > j:
                    p_h += p
                    if i - j >= 2:
                        p_h2 += p
                elif j > i:
                    p_a += p
                    if j - i >= 2:
                        p_a2 += p
                else:
                    p_t += p
        out.update(pts_home=lam_h, pts_away=lam_a, total=lam_h + lam_a, spread_home=-(lam_h - lam_a),
                   home_win_prob=p_h + 0.52 * p_t, home_puckline_prob=p_h2,
                   away_puckline_prob=p_a2)
        return out

    if lg.kind == "basketball":
        pace_h, pace_a, lg_pace = ph.get("pace"), pa_.get("pace"), base.get("pace")
        if nba_adv:
            ah, aa = _adv_lookup(nba_adv, game["home"]["name"]), _adv_lookup(nba_adv, game["away"]["name"])
            if ah and aa:
                pace_h, pace_a = ah["pace"], aa["pace"]
                lg_pace = sum(v["pace"] for v in nba_adv.values()) / len(nba_adv)
                lg_rtg = sum(v["off"] for v in nba_adv.values()) / len(nba_adv)
                off_h, def_h = ah["off"] * pace_h / 100, ah["def"] * pace_h / 100
                off_a, def_a = aa["off"] * pace_a / 100, aa["def"] * pace_a / 100
                lg_pts = lg_rtg * lg_pace / 100
        if pace_h and pace_a and lg_pace:
            poss = pace_h * pace_a / lg_pace
            lg_rtg = 100 * lg_pts / lg_pace
            ortg_h, drtg_h = 100 * off_h / pace_h, 100 * def_h / pace_h
            ortg_a, drtg_a = 100 * off_a / pace_a, 100 * def_a / pace_a
            pts_h = poss * (ortg_h * drtg_a / lg_rtg) / 100
            pts_a = poss * (ortg_a * drtg_h / lg_rtg) / 100
            out["poss"] = poss
        else:
            pts_h = off_h * def_a / lg_pts
            pts_a = off_a * def_h / lg_pts
    else:  # football
        pts_h = off_h * def_a / lg_pts
        pts_a = off_a * def_h / lg_pts

    if not game["neutral"]:
        pts_h += cfg["hca"] / 2
        pts_a -= cfg["hca"] / 2
    if cfg.get("b2b"):
        if hid in b2b:
            pts_h -= B2B_OWN[key]; pts_a += B2B_OPP[key]
        if aid in b2b:
            pts_a -= B2B_OWN[key]; pts_h += B2B_OPP[key]
    pts_h -= min(INJ_PTS_FACTOR * lost_pts.get(hid, 0.0), 0.08 * pts_h)
    pts_a -= min(INJ_PTS_FACTOR * lost_pts.get(aid, 0.0), 0.08 * pts_a)
    margin = pts_h - pts_a
    out.update(pts_home=pts_h, pts_away=pts_a, total=pts_h + pts_a, spread_home=-margin,
               home_win_prob=norm_cdf(margin / cfg["sigma"]))
    return out


def back_to_back_teams(profiles: dict[str, dict], slate_date: date) -> set[str]:
    y = slate_date - timedelta(days=1)
    return {tid for tid, p in profiles.items() if p and p["last_date"] == y}


# --------------------------------------------------------------------------
# Market blend (college): few games + no strength-of-schedule -> lean on the betting line early
# --------------------------------------------------------------------------
BLEND = {"NCAAB": {"k": 12.0, "cap": 0.5}}


def market_blend(lg: League, game: dict, proj: dict, gp: int) -> dict:
    """Blend the projected scores with the market-implied scores (home = (total - spread) / 2, away = (total +
    spread) / 2). Model weight = gp / (gp + k), capped; spread, total and win probability are recomputed."""
    rule = BLEND.get(lg.key)
    o = game.get("odds") or {}
    if not rule or not proj or (o.get("spread_home") is None and o.get("total") is None):
        return proj
    w = min(gp / (gp + rule["k"]), rule["cap"]) if gp > 0 else 0.0
    m_h, m_a = proj["pts_home"], proj["pts_away"]
    total = o["total"] if o.get("total") is not None else m_h + m_a
    spread = o["spread_home"] if o.get("spread_home") is not None else -(m_h - m_a)
    k_h, k_a = (total - spread) / 2, (total + spread) / 2
    out = dict(proj)
    out["model_raw"] = {"pts_home": m_h, "pts_away": m_a, "spread_home": proj["spread_home"], "total": proj["total"]}
    out["pts_home"] = w * m_h + (1 - w) * k_h
    out["pts_away"] = w * m_a + (1 - w) * k_a
    margin = out["pts_home"] - out["pts_away"]
    out["spread_home"] = -margin
    out["total"] = out["pts_home"] + out["pts_away"]
    out["home_win_prob"] = norm_cdf(margin / cfg(lg.key)["sigma"])
    out["blend_w"] = w
    return out
