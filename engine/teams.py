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


def team_profile(lg: League, team_id: str, before: date) -> dict | None:
    try:
        games = espn.parse_completed_games(espn.team_schedule(lg, team_id), team_id, before)
    except Exception as e:
        NOTES[f"schedule {lg.key}"] = f"Team schedule unavailable ({e})"
        return None
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


def league_baseline(key: str, profiles: dict[str, dict]) -> dict:
    cfg = CFG[key]
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
    cfg = CFG[key]
    hid, aid = game["home"]["id"], game["away"]["id"]
    lg_pts = base["pts"]
    off_h, def_h = _blend(ph, "pf", lg_pts, cfg["shrink"]), _blend(ph, "pa", lg_pts, cfg["shrink"])
    off_a, def_a = _blend(pa_, "pf", lg_pts, cfg["shrink"]), _blend(pa_, "pa", lg_pts, cfg["shrink"])
    out: dict = {"poss": None}

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
