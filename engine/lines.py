"""Live player prop lines from public pick'em boards: PrizePicks first, Underdog Fantasy as fallback.

Both are public JSON feeds used by their web apps. They sometimes block cloud servers; every call is wrapped,
throttled, cached for 10 minutes, and skipped for 30 minutes after a failure, so props fall back to model lines.
"""

from __future__ import annotations

from .common import BROWSER_UA, NOTES, disk_cached, flag_active, http_json, norm_name, num, set_flag
from .espn import League

PP_LEAGUES = "https://api.prizepicks.com/leagues"
PP_PROJECTIONS = "https://api.prizepicks.com/projections"
UD_LINES = "https://api.underdogfantasy.com/beta/v5/over_under_lines"
TTL_LINES = 10 * 60
DOWN_TTL = 30 * 60

PP_HEADERS = {"User-Agent": BROWSER_UA, "Accept": "application/json", "Referer": "https://app.prizepicks.com/",
              "Origin": "https://app.prizepicks.com"}
UD_HEADERS = {"User-Agent": BROWSER_UA, "Accept": "application/json", "Referer": "https://underdogfantasy.com/",
              "Origin": "https://underdogfantasy.com"}

# Board league / sport names per app sport
BOARD_NAMES = {"NBA": {"NBA"}, "NCAAB": {"CBB", "NCAAB", "CBK"}, "NFL": {"NFL"}, "NHL": {"NHL"}}

STAT_MAP = {
    "basketball": {"points": "PTS", "rebounds": "REB", "assists": "AST", "pts+rebs+asts": "PRA",
                   "pts + rebs + asts": "PRA", "points + rebounds + assists": "PRA", "pts+reb+ast": "PRA"},
    "football": {"pass yards": "PASS", "passing yards": "PASS", "rush yards": "RUSH", "rushing yards": "RUSH",
                 "receiving yards": "REC", "rec yards": "REC", "receptions": "RECS"},
    "hockey": {"goals": "G", "assists": "A", "shots on goal": "SOG", "shots": "SOG", "points": "PTS"},
}


def _prizepicks(lg: League) -> dict[tuple[str, str], float]:
    if flag_active("pp_down.flag", DOWN_TTL):
        return {}
    try:
        leagues = disk_cached("pp_leagues", 24 * 3600, lambda: http_json(PP_LEAGUES, headers=PP_HEADERS, retries=1))
        ids = [str(d.get("id")) for d in leagues.get("data", []) or []
               if str((d.get("attributes") or {}).get("name", "")).upper() in BOARD_NAMES.get(lg.key, set())]
        out: dict[tuple[str, str], float] = {}
        for lid in ids:
            raw = disk_cached(f"pp_proj|{lid}", TTL_LINES, lambda lid=lid: http_json(
                PP_PROJECTIONS, {"league_id": lid, "per_page": "1000", "single_stat": "true"},
                headers=PP_HEADERS, retries=1))
            players = {str(i.get("id")): (i.get("attributes") or {}) for i in raw.get("included", []) or []
                       if i.get("type") in ("new_player", "player")}
            for d in raw.get("data", []) or []:
                if d.get("type") != "projection":
                    continue
                at = d.get("attributes") or {}
                if at.get("odds_type", "standard") not in ("standard", None):  # skip goblin/demon alt lines
                    continue
                rel = ((d.get("relationships") or {}).get("new_player") or {}).get("data") or {}
                pl = players.get(str(rel.get("id")), {})
                name = pl.get("display_name") or pl.get("name") or ""
                m = STAT_MAP[lg.kind].get(str(at.get("stat_type", "")).lower().strip())
                line = num(at.get("line_score"), default=float("nan"))
                if name and m and line == line:
                    out.setdefault((norm_name(name), m), line)
        return out
    except Exception as e:
        set_flag("pp_down.flag")
        NOTES["prizepicks"] = f"PrizePicks lines unavailable ({type(e).__name__}); trying Underdog."
        return {}


def _underdog(lg: League) -> dict[tuple[str, str], float]:
    if flag_active("ud_down.flag", DOWN_TTL):
        return {}
    try:
        raw = disk_cached("ud_lines", TTL_LINES, lambda: http_json(UD_LINES, headers=UD_HEADERS, retries=1))
        players = {str(p.get("id")): p for p in raw.get("players", []) or []}
        apps = {str(a.get("id")): a for a in raw.get("appearances", []) or []}
        names = BOARD_NAMES.get(lg.key, set())
        out: dict[tuple[str, str], float] = {}
        for ln in raw.get("over_under_lines", []) or []:
            if ln.get("status", "active") != "active":
                continue
            ast = ((ln.get("over_under") or {}).get("appearance_stat")) or {}
            app = apps.get(str(ast.get("appearance_id")), {})
            pl = players.get(str(app.get("player_id")), {})
            if str(pl.get("sport_id", "")).upper() not in names:
                continue
            name = f'{pl.get("first_name", "")} {pl.get("last_name", "")}'.strip()
            stat = str(ast.get("display_stat") or ast.get("stat") or "").lower().replace("_", " ").strip()
            m = STAT_MAP[lg.kind].get(stat)
            line = num(ln.get("stat_value"), default=float("nan"))
            if name and m and line == line:
                out.setdefault((norm_name(name), m), line)
        return out
    except Exception as e:
        set_flag("ud_down.flag")
        NOTES["underdog"] = f"Underdog lines unavailable ({type(e).__name__})."
        return {}


def book_lines(lg: League) -> dict[tuple[str, str], tuple[float, str]]:
    """{(normalized player name, market): (line, source)}; PrizePicks first, Underdog fills gaps."""
    if lg.kind not in STAT_MAP:
        return {}
    out = {k: (v, "PrizePicks") for k, v in _prizepicks(lg).items()}
    for k, v in _underdog(lg).items():
        out.setdefault(k, (v, "Underdog"))
    return out
