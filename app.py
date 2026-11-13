"""
NBA Game & Player Prop Predictor
================================

Live projections built only from real data:

* ESPN public JSON  - daily schedule / scores / market lines, injury report,
                      previous-day schedule (back-to-back detection)
* nba_api (stats.nba.com) - team advanced stats (Pace, OffRtg, DefRtg),
                      team opponent stats, and every player game log of the
                      season (used for last-10 windows and rosters)

Every network call goes through one throttle (0.6 s minimum spacing), sends
browser-like request headers, and is cached on disk under ./.cache with a
per-source TTL. If a source is unreachable, the last cached copy is used and
the UI says so; nothing is ever invented.

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from nba_api.live.nba.endpoints import boxscore as live_boxscore
from nba_api.stats.endpoints import leaguedashteamstats, leaguegamelog
from nba_api.stats.static import teams as nba_static_teams

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
CACHE_DIR = Path(os.environ.get("NBA_PREDICTOR_CACHE", Path(__file__).parent / ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_SECONDS = 0.6

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
CDN_SCHEDULE = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

# stats.nba.com silently drops many cloud-host requests (e.g. Streamlit Cloud), so fail fast
# and remember the failure; the app then falls back to box scores on cdn.nba.com.
NBA_STATS_TIMEOUT = 10
STATS_DOWN_TTL = 30 * 60
STATS_DOWN_FLAG = "stats_nba_down.flag"
# NBA game-id prefixes by season type
GAME_ID_PREFIX = {"Regular Season": "002", "Playoffs": "004", "PlayIn": "005"}

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
ESPN_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/",
    "Origin": "https://www.espn.com",
}
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": BROWSER_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
CDN_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

# Cache lifetimes (seconds)
TTL_SCOREBOARD_LIVE = 120
TTL_SCOREBOARD_PAST = 12 * 3600
TTL_INJURIES = 15 * 60
TTL_TEAM_STATS = 6 * 3600
TTL_GAME_LOGS = 2 * 3600
TTL_CDN_SCHEDULE = 30 * 60
TTL_FINAL_BOXSCORE = 365 * 24 * 3600  # final box scores never change

# Model constants
HOME_COURT_RTG = 1.0          # pts/100 added to home ORtg and removed from away ORtg (~2 pts total)
B2B_OFF_PENALTY = 1.2         # pts/100 lost by a team on the 2nd night of a back-to-back
B2B_DEF_PENALTY = 1.0         # pts/100 conceded extra by a team on a back-to-back
B2B_MIN_FACTOR = 0.96         # player minutes multiplier on a back-to-back
B2B_EFF_FACTOR = 0.98         # player per-minute production multiplier on a back-to-back
INJ_OFF_FACTOR = 0.15         # ORtg fraction lost per unit of team scoring share missing
INJ_DEF_PER_MIN_SHARE = 6.0   # DRtg pts/100 added per unit of team minute share missing
USG_PTS_K = 0.30              # extra per-minute scoring for teammates per unit of lost scoring share
USG_AST_K = 0.15
USG_REB_K = 0.20
MAX_PLAYER_MIN = 40.0
MARGIN_SD = 12.0              # std-dev of NBA final margin vs. projection
MIN_ROTATION_MIN = 8.0        # last-10 average minutes to count as rotation player

# ESPN abbreviations that differ from NBA.com
ESPN_TO_NBA_ABBR = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}

NBA_TEAMS = nba_static_teams.get_teams()
NBA_TEAM_BY_ABBR = {t["abbreviation"]: t for t in NBA_TEAMS}
NBA_TEAM_BY_ID = {t["id"]: t for t in NBA_TEAMS}


# --------------------------------------------------------------------------
# Networking: throttle + disk cache
# --------------------------------------------------------------------------
_throttle_lock = threading.Lock()
_last_call = [0.0]
_session = requests.Session()


def throttle() -> None:
    """Guarantee at least RATE_LIMIT_SECONDS between any two outbound requests."""
    with _throttle_lock:
        wait = RATE_LIMIT_SECONDS - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def _note_stale(key: str, age: float, err: Exception) -> None:
    msgs = st.session_state.setdefault("stale_notes", {})
    msgs[key] = f"{key}: live fetch failed ({type(err).__name__}); using cached copy {age / 60:.0f} min old."


def disk_cached(key: str, ttl: int, fetch):
    """Return fresh cached JSON for `key`, otherwise call fetch() and store it.

    On a fetch failure a stale cached copy is returned (and flagged in the UI);
    with no cached copy the error propagates.
    """
    path = _cache_path(key)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < ttl:
            with path.open() as f:
                return json.load(f)
    try:
        data = fetch()
    except Exception as err:
        if path.exists():
            _note_stale(key, time.time() - path.stat().st_mtime, err)
            with path.open() as f:
                return json.load(f)
        raise
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
    tmp.replace(path)
    return data


def clear_disk_cache() -> None:
    for pattern in ("*.json", "*.flag"):
        for p in CACHE_DIR.glob(pattern):
            p.unlink(missing_ok=True)


def espn_get(url: str, params: dict | None = None) -> dict:
    last_err: Exception | None = None
    for attempt in range(3):
        throttle()
        try:
            r = _session.get(url, params=params, headers=ESPN_HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as err:  # network / HTTP / JSON errors
            last_err = err
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ESPN request failed: {url}") from last_err


def stats_nba_down() -> bool:
    flag = CACHE_DIR / STATS_DOWN_FLAG
    return flag.exists() and time.time() - flag.stat().st_mtime < STATS_DOWN_TTL


def nba_call(endpoint_cls, **kwargs) -> dict:
    """Call a stats.nba.com endpoint. Timeouts/connection drops are not retried (they only
    repeat the wait); after one, stats.nba.com is skipped for STATS_DOWN_TTL seconds."""
    if stats_nba_down():
        raise RuntimeError("stats.nba.com did not respond recently; skipping it for now")
    last_err: Exception | None = None
    for attempt in range(2):
        throttle()
        try:
            return endpoint_cls(headers=NBA_HEADERS, timeout=NBA_STATS_TIMEOUT, **kwargs).get_dict()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            last_err = err
            break
        except Exception as err:
            last_err = err
            time.sleep(1.5)
    (CACHE_DIR / STATS_DOWN_FLAG).touch()
    raise RuntimeError(f"stats.nba.com request failed: {endpoint_cls.__name__} ({type(last_err).__name__})") from last_err


def cdn_get(url: str) -> dict:
    last_err: Exception | None = None
    for attempt in range(3):
        throttle()
        try:
            r = _session.get(url, headers=CDN_HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as err:
            last_err = err
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"cdn.nba.com request failed: {url}") from last_err


def result_set_df(payload: dict, index: int = 0) -> pd.DataFrame:
    sets = payload.get("resultSets", payload.get("resultSet"))
    if isinstance(sets, dict):
        sets = [sets]
    if not sets:
        return pd.DataFrame()
    rs = sets[index]
    return pd.DataFrame(rs.get("rowSet", []), columns=rs.get("headers", []))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def season_for(d: date) -> str:
    start = d.year if d.month >= 9 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def prev_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def norm_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    n = n.replace("-", " ").replace(".", "").replace("'", "")
    n = re.sub(r"[^a-z ]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return " ".join(n.split())


def espn_abbr_to_nba(abbr: str) -> str:
    return ESPN_TO_NBA_ABBR.get(abbr, abbr)


def team_name_to_abbr(display_name: str) -> str | None:
    """Map an ESPN display name (e.g. 'LA Clippers') to an NBA abbreviation."""
    dn = (display_name or "").lower()
    for t in NBA_TEAMS:
        if dn == t["full_name"].lower() or dn.endswith(t["nickname"].lower()):
            return t["abbreviation"]
    return None


def to_minutes(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    if isinstance(v, str) and v.startswith("PT"):  # ISO-8601 duration, e.g. PT34M12.00S
        m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?", v)
        if not m:
            return 0.0
        h, mi, se = (float(x) if x else 0.0 for x in m.groups())
        return h * 60 + mi + se / 60
    if isinstance(v, str) and ":" in v:
        m, s = v.split(":")[:2]
        return float(m) + float(s) / 60
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def book_line(x: float) -> float:
    """Nearest value ending in .5, the way sportsbooks post props."""
    return round(x - 0.5) + 0.5


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fmt_spread(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return "PK" if abs(v) < 0.05 else f"{v:+.1f}"


# --------------------------------------------------------------------------
# Data loaders (ESPN)
# --------------------------------------------------------------------------
def _parse_market(comp: dict, home_espn: str) -> tuple[float | None, float | None, str]:
    odds = (comp.get("odds") or [{}])[0] or {}
    total = odds.get("overUnder")
    details = (odds.get("details") or "").strip()
    spread_home = None
    if details:
        if details.upper() in ("EVEN", "PK", "PICK"):
            spread_home = 0.0
        else:
            m = re.match(r"([A-Za-z]+)\s+([+-]?\d+(?:\.\d+)?)", details)
            if m:
                fav, num = m.group(1).upper(), float(m.group(2))
                spread_home = num if fav == home_espn.upper() else -num
    try:
        total = float(total) if total is not None else None
    except (TypeError, ValueError):
        total = None
    return spread_home, total, details


@st.cache_data(ttl=60, show_spinner=False)
def load_schedule(d: date, today: date) -> pd.DataFrame:
    ymd = d.strftime("%Y%m%d")
    ttl = TTL_SCOREBOARD_PAST if d < today else TTL_SCOREBOARD_LIVE
    data = disk_cached(f"espn_scoreboard|{ymd}", ttl, lambda: espn_get(ESPN_SCOREBOARD, {"dates": ymd}))
    rows = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in teams or "away" not in teams:
            continue
        h, a = teams["home"], teams["away"]
        h_espn, a_espn = h["team"]["abbreviation"], a["team"]["abbreviation"]
        status = ev.get("status", {}).get("type", {})
        mkt_spread, mkt_total, mkt_details = _parse_market(comp, h_espn)
        tip = pd.to_datetime(ev.get("date"), utc=True)
        rows.append(
            {
                "game_id": ev.get("id"),
                "tip_et": tip.tz_convert(ET) if pd.notna(tip) else pd.NaT,
                "home": espn_abbr_to_nba(h_espn),
                "away": espn_abbr_to_nba(a_espn),
                "home_name": h["team"].get("displayName", h_espn),
                "away_name": a["team"].get("displayName", a_espn),
                "home_score": h.get("score"),
                "away_score": a.get("score"),
                "state": status.get("state", ""),
                "status": status.get("shortDetail", ""),
                "mkt_spread_home": mkt_spread,
                "mkt_total": mkt_total,
                "mkt_details": mkt_details,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner=False)
def load_injuries() -> pd.DataFrame:
    data = disk_cached("espn_injuries", TTL_INJURIES, lambda: espn_get(ESPN_INJURIES))
    rows = []
    for team in data.get("injuries", []):
        team_abbr = team_name_to_abbr(team.get("displayName", ""))
        for inj in team.get("injuries", []):
            ath = inj.get("athlete") or {}
            ath_team = (ath.get("team") or {}).get("abbreviation")
            abbr = espn_abbr_to_nba(ath_team) if ath_team else team_abbr
            name = ath.get("displayName") or ath.get("fullName") or ""
            if not abbr or not name:
                continue
            rows.append(
                {
                    "team": abbr,
                    "player": name,
                    "key": norm_name(name),
                    "status": inj.get("status") or (inj.get("type") or {}).get("description", ""),
                    "detail": inj.get("shortComment") or inj.get("longComment") or "",
                    "updated": inj.get("date", ""),
                }
            )
    return pd.DataFrame(rows, columns=["team", "player", "key", "status", "detail", "updated"])


def injury_class(status: str) -> str:
    s = (status or "").lower()
    if "probable" in s:
        return "ACTIVE"
    if any(k in s for k in ("out", "doubtful", "suspen", "inactive", "injured reserve")):
        return "OUT"
    if any(k in s for k in ("questionable", "day-to-day", "day to day", "gtd", "game time")):
        return "Q"
    return "ACTIVE"


# --------------------------------------------------------------------------
# Data loaders (nba_api)
# --------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_team_stats(season: str, season_type: str) -> pd.DataFrame:
    common = dict(season=season, season_type_all_star=season_type, per_mode_detailed="PerGame")
    adv = result_set_df(
        disk_cached(
            f"nba_team_adv|{season}|{season_type}",
            TTL_TEAM_STATS,
            lambda: nba_call(leaguedashteamstats.LeagueDashTeamStats, measure_type_detailed_defense="Advanced", **common),
        )
    )
    opp = result_set_df(
        disk_cached(
            f"nba_team_opp|{season}|{season_type}",
            TTL_TEAM_STATS,
            lambda: nba_call(leaguedashteamstats.LeagueDashTeamStats, measure_type_detailed_defense="Opponent", **common),
        )
    )
    if adv.empty:
        return adv
    df = adv[["TEAM_ID", "TEAM_NAME", "GP", "W", "L", "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING"]].copy()
    if not opp.empty:
        df = df.merge(opp[["TEAM_ID", "OPP_PTS", "OPP_REB", "OPP_AST"]], on="TEAM_ID", how="left")
    df["ABBR"] = df["TEAM_ID"].map(lambda i: NBA_TEAM_BY_ID.get(int(i), {}).get("abbreviation"))
    return df.dropna(subset=["ABBR"]).set_index("ABBR")


@st.cache_data(ttl=300, show_spinner=False)
def load_player_logs(season: str, season_type: str) -> pd.DataFrame:
    payload = disk_cached(
        f"nba_player_logs|{season}|{season_type}",
        TTL_GAME_LOGS,
        lambda: nba_call(
            leaguegamelog.LeagueGameLog,
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="P",
            direction="DESC",
            sorter="DATE",
        ),
    )
    df = result_set_df(payload)
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    df["MIN"] = df["MIN"].map(to_minutes)
    for c in ("PTS", "REB", "AST", "FG3M"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["KEY"] = df["PLAYER_NAME"].map(norm_name)
    return df


# --------------------------------------------------------------------------
# Fallback loaders: nba_api live box scores (cdn.nba.com)
# --------------------------------------------------------------------------
def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _compact_boxscore(payload: dict, game_date: str) -> dict:
    g = payload.get("game", payload)
    teams = []
    for side in ("homeTeam", "awayTeam"):
        t = g[side]
        s = t.get("statistics", {}) or {}
        players = []
        for p in t.get("players", []) or []:
            ps = p.get("statistics", {}) or {}
            if str(p.get("played", "1")) != "1" or to_minutes(ps.get("minutes")) <= 0:
                continue
            players.append(
                {
                    "id": p.get("personId"),
                    "name": p.get("name") or f"{p.get('firstName', '')} {p.get('familyName', '')}".strip(),
                    "min": to_minutes(ps.get("minutes")),
                    "pts": _num(ps.get("points")),
                    "reb": _num(ps.get("reboundsTotal")),
                    "ast": _num(ps.get("assists")),
                    "fg3m": _num(ps.get("threePointersMade")),
                    "pm": _num(ps.get("plusMinusPoints")),
                }
            )
        teams.append(
            {
                "side": side,
                "id": t.get("teamId"),
                "tri": t.get("teamTricode"),
                "pts": _num(t.get("score", s.get("points"))),
                "fga": _num(s.get("fieldGoalsAttempted")),
                "fta": _num(s.get("freeThrowsAttempted")),
                "oreb": _num(s.get("reboundsOffensive")),
                "tov": _num(s.get("turnoversTotal", s.get("turnovers"))),
                "reb": _num(s.get("reboundsTotal")),
                "ast": _num(s.get("assists")),
                "min": to_minutes(s.get("minutesCalculated") or s.get("minutes")) or 240.0,
                "players": players,
            }
        )
    return {"game_id": g.get("gameId"), "date": game_date, "teams": teams}


def fetch_boxscore(game_id: str, game_date: str) -> dict:
    def _fetch():
        throttle()
        payload = live_boxscore.BoxScore(game_id, headers=CDN_HEADERS, timeout=20).get_dict()
        return _compact_boxscore(payload, game_date)

    return disk_cached(f"cdn_box|{game_id}", TTL_FINAL_BOXSCORE, _fetch)


def finished_games(season: str, season_type: str, before: date) -> list[tuple[str, str]]:
    data = disk_cached("cdn_schedule", TTL_CDN_SCHEDULE, lambda: cdn_get(CDN_SCHEDULE))
    ls = data.get("leagueSchedule", {})
    if ls.get("seasonYear") and ls["seasonYear"] != season:
        raise RuntimeError(f"cdn.nba.com only has the {ls['seasonYear']} schedule (requested {season})")
    prefix = GAME_ID_PREFIX.get(season_type, "002")
    out = []
    for gd in ls.get("gameDates", []):
        for g in gd.get("games", []):
            gid = str(g.get("gameId", ""))
            if not gid.startswith(prefix) or int(g.get("gameStatus", 0) or 0) != 3:
                continue
            raw = g.get("gameDateEst") or gd.get("gameDate", "")
            try:
                gdate = (datetime.strptime(raw[:10], "%Y-%m-%d") if "-" in raw[:10]
                         else datetime.strptime(raw[:10], "%m/%d/%Y")).date()
            except ValueError:
                continue
            if gdate < before:
                out.append((gid, gdate.isoformat()))
    return out


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def build_from_boxscores(game_keys: tuple[tuple[str, str], ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team ratings (Pace/OffRtg/DefRtg/opp REB & AST) and a LeagueGameLog-style player table."""
    team_rows, player_rows = [], []
    for gid, gdate in game_keys:
        path = _cache_path(f"cdn_box|{gid}")
        if not path.exists():
            continue
        with path.open() as f:
            box = json.load(f)
        if len(box.get("teams", [])) != 2:
            continue
        a, b = box["teams"]
        poss = {t["tri"]: t["fga"] + 0.44 * t["fta"] - t["oreb"] + t["tov"] for t in (a, b)}
        game_poss = (poss[a["tri"]] + poss[b["tri"]]) / 2
        for t, o in ((a, b), (b, a)):
            win = t["pts"] > o["pts"]
            team_rows.append(
                {"ABBR": t["tri"], "TEAM_ID": t["id"], "W": int(win), "L": int(not win), "PTS": t["pts"],
                 "OPP_PTS": o["pts"], "POSS": game_poss, "MIN": t["min"], "OPP_REB": o["reb"], "OPP_AST": o["ast"]}
            )
            matchup = f"{t['tri']} vs. {o['tri']}" if t["side"] == "homeTeam" else f"{t['tri']} @ {o['tri']}"
            for p in t["players"]:
                player_rows.append(
                    {"PLAYER_ID": p["id"], "PLAYER_NAME": p["name"], "TEAM_ID": t["id"],
                     "TEAM_ABBREVIATION": t["tri"], "GAME_ID": gid, "GAME_DATE": date.fromisoformat(gdate),
                     "MATCHUP": matchup, "WL": "W" if win else "L", "MIN": p["min"], "PTS": p["pts"],
                     "REB": p["reb"], "AST": p["ast"], "FG3M": p["fg3m"], "PLUS_MINUS": p["pm"]}
                )
    tg = pd.DataFrame(team_rows)
    if tg.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg = tg.groupby("ABBR").agg(
        TEAM_ID=("TEAM_ID", "first"), GP=("W", "size"), W=("W", "sum"), L=("L", "sum"), PTS=("PTS", "sum"),
        OPP_PTS_SUM=("OPP_PTS", "sum"), POSS=("POSS", "sum"), MIN=("MIN", "sum"),
        OPP_PTS=("OPP_PTS", "mean"), OPP_REB=("OPP_REB", "mean"), OPP_AST=("OPP_AST", "mean"),
    )
    agg["PACE"] = 48 * agg["POSS"] / (agg["MIN"] / 5)
    agg["OFF_RATING"] = 100 * agg["PTS"] / agg["POSS"]
    agg["DEF_RATING"] = 100 * agg["OPP_PTS_SUM"] / agg["POSS"]
    agg["NET_RATING"] = agg["OFF_RATING"] - agg["DEF_RATING"]
    agg["TEAM_NAME"] = [NBA_TEAM_BY_ABBR.get(x, {}).get("full_name", x) for x in agg.index]
    team_stats = agg[["TEAM_ID", "TEAM_NAME", "GP", "W", "L", "PACE", "OFF_RATING", "DEF_RATING",
                      "NET_RATING", "OPP_PTS", "OPP_REB", "OPP_AST"]]
    logs = pd.DataFrame(player_rows)
    logs["KEY"] = logs["PLAYER_NAME"].map(norm_name)
    return team_stats, logs


def load_from_boxscores(season: str, season_type: str, before: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = finished_games(season, season_type, before)
    if not games:
        raise RuntimeError(f"No finished {season} {season_type} games before {before} on cdn.nba.com")
    missing = [g for g in games if not _cache_path(f"cdn_box|{g[0]}").exists()]
    failures = 0
    if missing:
        bar = st.progress(0.0)
        for i, (gid, gdate) in enumerate(missing, 1):
            try:
                fetch_boxscore(gid, gdate)
            except Exception:
                failures += 1
            bar.progress(i / len(missing), text=f"Downloading NBA box scores {i}/{len(missing)} "
                                                 "(first load only; cached afterwards)…")
        bar.empty()
    if failures > 0.2 * len(games):
        raise RuntimeError(f"{failures} of {len(games)} box scores could not be downloaded from cdn.nba.com")
    if failures:
        st.warning(f"{failures} box score(s) could not be downloaded; ratings use the other {len(games) - failures}.")
    return build_from_boxscores(tuple(games))


# --------------------------------------------------------------------------
# Player pool / injury impact
# --------------------------------------------------------------------------
def build_team_pool(logs: pd.DataFrame, team: str, injuries: pd.DataFrame, q_sit_prob: float) -> tuple[pd.DataFrame, dict]:
    """Current roster (from most recent team per player) with last-10 and season-with-team numbers."""
    latest = logs.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(1)
    pids = latest.loc[latest["TEAM_ABBREVIATION"] == team, "PLAYER_ID"].tolist()

    team_rows = logs[logs["TEAM_ABBREVIATION"] == team]
    team_games = team_rows.drop_duplicates("GAME_ID").sort_values("GAME_DATE", ascending=False)
    n_team_games = len(team_games)
    last10_team = set(team_games["GAME_ID"].head(10))
    team_ppg = team_rows.groupby("GAME_ID")["PTS"].sum().mean() if n_team_games else np.nan

    inj = injuries[injuries["team"] == team] if not injuries.empty else injuries
    inj_by_key = {r.key: r for r in inj.itertuples()}

    rows = []
    for pid in pids:
        p = logs[logs["PLAYER_ID"] == pid].sort_values("GAME_DATE", ascending=False)
        l10 = p.head(10)
        w_team = p[p["TEAM_ABBREVIATION"] == team]
        key = p["KEY"].iloc[0]
        inj_row = inj_by_key.get(key)
        status = inj_row.status if inj_row is not None else ""
        cls = injury_class(status)
        sit_prob = 1.0 if cls == "OUT" else (q_sit_prob if cls == "Q" else 0.0)
        rows.append(
            {
                "PLAYER_ID": pid,
                "PLAYER": p["PLAYER_NAME"].iloc[0],
                "KEY": key,
                "STATUS": status or "Active",
                "INJ_DETAIL": inj_row.detail if inj_row is not None else "",
                "CLASS": cls,
                "SIT_PROB": sit_prob,
                "GP_L10": len(l10),
                "L10_MIN": l10["MIN"].mean(),
                "L10_PTS": l10["PTS"].mean(),
                "L10_REB": l10["REB"].mean(),
                "L10_AST": l10["AST"].mean(),
                "SEASON_GP_TEAM": len(w_team),
                "SEASON_MIN_TEAM": w_team["MIN"].mean() if len(w_team) else 0.0,
                "SEASON_PTS_TEAM": w_team["PTS"].mean() if len(w_team) else 0.0,
                "AVAIL_SEASON": len(w_team) / n_team_games if n_team_games else 0.0,
                "AVAIL_RECENT": (w_team["GAME_ID"].isin(last10_team).sum() / len(last10_team)) if last10_team else 0.0,
            }
        )
    pool = pd.DataFrame(rows)
    meta = {"team_games": n_team_games, "team_ppg": team_ppg}
    if pool.empty:
        meta.update(lost_pts_season=0.0, lost_min_season=0.0, lost_pts_recent=0.0,
                    lost_min_recent=0.0, lost_reb_recent=0.0, lost_ast_recent=0.0)
        return pool, meta

    w = pool["SIT_PROB"]
    # Season ratings include these players in proportion to games played -> scale by availability.
    meta["lost_pts_season"] = float((w * pool["SEASON_PTS_TEAM"] * pool["AVAIL_SEASON"]).sum())
    meta["lost_min_season"] = float((w * pool["SEASON_MIN_TEAM"] * pool["AVAIL_SEASON"]).sum())
    # Teammates' last-10 numbers already reflect absences inside that window -> scale by recent availability.
    rw = w * pool["AVAIL_RECENT"]
    meta["lost_pts_recent"] = float((rw * pool["L10_PTS"]).sum())
    meta["lost_min_recent"] = float((rw * pool["L10_MIN"]).sum())
    meta["lost_reb_recent"] = float((rw * pool["L10_REB"]).sum())
    meta["lost_ast_recent"] = float((rw * pool["L10_AST"]).sum())
    return pool, meta


# --------------------------------------------------------------------------
# Game model
# --------------------------------------------------------------------------
def league_baselines(ts: pd.DataFrame) -> dict:
    base = {"pace": ts["PACE"].mean(), "rtg": ts["OFF_RATING"].mean()}
    if "OPP_REB" in ts:
        base["opp_reb_per_poss"] = (ts["OPP_REB"] / ts["PACE"]).mean()
        base["opp_ast_per_poss"] = (ts["OPP_AST"] / ts["PACE"]).mean()
    return base


def project_game(home: str, away: str, ts: pd.DataFrame, lg: dict, b2b: set, meta_h: dict, meta_a: dict) -> dict:
    H, A = ts.loc[home], ts.loc[away]
    poss = H["PACE"] * A["PACE"] / lg["pace"]
    if home in b2b:
        poss -= 0.5
    if away in b2b:
        poss -= 0.5

    def adj(team_row, meta, is_b2b):
        off, de = float(team_row["OFF_RATING"]), float(team_row["DEF_RATING"])
        ppg = meta.get("team_ppg")
        share_pts = (meta["lost_pts_season"] / ppg) if ppg and ppg > 0 else 0.0
        share_min = meta["lost_min_season"] / 240.0
        off *= 1 - INJ_OFF_FACTOR * min(share_pts, 0.6)
        de += INJ_DEF_PER_MIN_SHARE * min(share_min, 0.6)
        if is_b2b:
            off -= B2B_OFF_PENALTY
            de += B2B_DEF_PENALTY
        return off, de

    off_h, def_h = adj(H, meta_h, home in b2b)
    off_a, def_a = adj(A, meta_a, away in b2b)

    ortg_h = off_h * def_a / lg["rtg"] + HOME_COURT_RTG
    ortg_a = off_a * def_h / lg["rtg"] - HOME_COURT_RTG
    pts_h = poss * ortg_h / 100
    pts_a = poss * ortg_a / 100
    margin = pts_h - pts_a
    return {
        "poss": poss,
        "pts_home": pts_h,
        "pts_away": pts_a,
        "spread_home": -margin,
        "total": pts_h + pts_a,
        "home_win_prob": norm_cdf(margin / MARGIN_SD),
        "ortg_home": ortg_h,
        "ortg_away": ortg_a,
        "def_home_adj": def_h,
        "def_away_adj": def_a,
    }


# --------------------------------------------------------------------------
# Player prop model
# --------------------------------------------------------------------------
def project_players(
    logs: pd.DataFrame,
    pool: pd.DataFrame,
    meta: dict,
    team: str,
    opp: str,
    ts: pd.DataFrame,
    lg: dict,
    game_poss: float,
    is_b2b: bool,
) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame()
    rot = pool[(pool["GP_L10"] >= 2) & (pool["L10_MIN"] >= MIN_ROTATION_MIN) & (pool["CLASS"] != "OUT")].copy()
    if rot.empty:
        return pd.DataFrame()

    # Minutes redistribution from absent teammates
    play_prob = 1 - rot["SIT_PROB"]
    base_min = rot["L10_MIN"]
    denom = float((base_min * play_prob).sum()) or 1.0
    extra_min = meta["lost_min_recent"] * base_min / denom
    proj_min = np.minimum(base_min + extra_min, np.maximum(base_min, MAX_PLAYER_MIN))
    if is_b2b:
        proj_min = proj_min * B2B_MIN_FACTOR

    team_pts = float((rot["L10_PTS"] * play_prob).sum() + meta["lost_pts_recent"]) or 1.0
    team_reb = float((rot["L10_REB"] * play_prob).sum() + meta["lost_reb_recent"]) or 1.0
    team_ast = float((rot["L10_AST"] * play_prob).sum() + meta["lost_ast_recent"]) or 1.0
    usg_pts = 1 + USG_PTS_K * meta["lost_pts_recent"] / team_pts
    usg_reb = 1 + USG_REB_K * meta["lost_reb_recent"] / team_reb
    usg_ast = 1 + USG_AST_K * meta["lost_ast_recent"] / team_ast

    T, O = ts.loc[team], ts.loc[opp]
    pace_f = game_poss / float(T["PACE"])
    pts_f = float(O["DEF_RATING"]) / lg["rtg"]
    reb_f = (float(O["OPP_REB"]) / float(O["PACE"])) / lg["opp_reb_per_poss"] if "opp_reb_per_poss" in lg else 1.0
    ast_f = (float(O["OPP_AST"]) / float(O["PACE"])) / lg["opp_ast_per_poss"] if "opp_ast_per_poss" in lg else 1.0
    eff_b2b = B2B_EFF_FACTOR if is_b2b else 1.0

    out = []
    for (idx, r), pm in zip(rot.iterrows(), proj_min):
        g = logs[logs["PLAYER_ID"] == r["PLAYER_ID"]].sort_values("GAME_DATE", ascending=False).head(10)
        w = np.linspace(1.0, 0.55, len(g))  # recency weighting, newest first
        wmin = float((g["MIN"] * w).sum()) or 1.0
        rate = {s: float((g[s] * w).sum()) / wmin for s in ("PTS", "REB", "AST")}
        proj = {
            "PTS": rate["PTS"] * pm * pace_f * pts_f * usg_pts * eff_b2b,
            "REB": rate["REB"] * pm * pace_f * reb_f * usg_reb,
            "AST": rate["AST"] * pm * pace_f * ast_f * usg_ast * eff_b2b,
        }
        row = {
            "Player": r["PLAYER"],
            "Status": r["STATUS"],
            "L10 MIN": r["L10_MIN"],
            "Proj MIN": pm,
        }
        for s in ("PTS", "REB", "AST"):
            line = book_line(proj[s])
            row[f"L10 {s}"] = float(g[s].mean())
            row[f"Proj {s}"] = proj[s]
            row[f"{s} Line"] = line
            row[f"{s} L10 Over%"] = float((g[s] > line).mean() * 100)
        pra = proj["PTS"] + proj["REB"] + proj["AST"]
        row["Proj PRA"] = pra
        row["PRA Line"] = book_line(pra)
        row["_pid"] = r["PLAYER_ID"]
        out.append(row)
    return pd.DataFrame(out).sort_values("Proj PTS", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide")
st.title("🏀 NBA Game & Player Prop Predictor")
st.caption("Live data: ESPN (schedule, injuries, lines) + stats.nba.com via nba_api (advanced team stats, player game logs).")

now_et = datetime.now(ET)
today_et = now_et.date()

with st.sidebar:
    st.header("Settings")
    game_date = st.date_input("Slate date (ET)", value=today_et)
    season = st.text_input("Season", value=season_for(game_date))
    season_type = st.selectbox("Season type", ["Regular Season", "Playoffs", "PlayIn"], index=0)
    q_sit_prob = st.slider("Chance a Questionable / Day-to-Day player sits", 0.0, 1.0, 0.5, 0.05)
    if st.button("🔄 Refresh all data"):
        clear_disk_cache()
        st.cache_data.clear()
        st.session_state.pop("stale_notes", None)
        st.rerun()
    st.caption(f"Cache dir: `{CACHE_DIR}`  \nRequests throttled to ≥ {RATE_LIMIT_SECONDS}s apart.")

# ---- Schedule -----------------------------------------------------------
try:
    with st.spinner("Loading ESPN schedule…"):
        sched = load_schedule(game_date, today_et)
        sched_prev = load_schedule(game_date - timedelta(days=1), today_et)
except Exception as e:
    st.error(f"Could not load the ESPN schedule and no cached copy exists: {e}")
    st.stop()

if sched.empty:
    st.info(f"No NBA games scheduled on {game_date:%A, %B %d, %Y}.")
    st.stop()

b2b_teams = set(sched_prev["home"]).union(sched_prev["away"]) if not sched_prev.empty else set()

# ---- Injuries -----------------------------------------------------------
try:
    with st.spinner("Loading ESPN injury report…"):
        injuries = load_injuries()
except Exception as e:
    st.warning(f"Injury report unavailable ({e}); projections run without injury adjustments.")
    injuries = pd.DataFrame(columns=["team", "player", "key", "status", "detail", "updated"])

# ---- Team stats + player logs ------------------------------------------
stats_season = season
data_source = "stats.nba.com (nba_api)"
team_stats = logs = None
stats_err: Exception | None = None
try:
    with st.spinner("Loading team advanced stats and player game logs from stats.nba.com…"):
        team_stats = load_team_stats(season, season_type)
        logs = load_player_logs(season, season_type)
        if team_stats.empty or logs.empty or (logs["GAME_DATE"] < game_date).sum() == 0:
            stats_season = prev_season(season)
            team_stats = load_team_stats(stats_season, season_type)
            logs = load_player_logs(stats_season, season_type)
            st.info(f"No {season} {season_type} data before {game_date}; using {stats_season} stats instead.")
except Exception as e:
    stats_err = e
    team_stats = logs = None
    stats_season = season

if team_stats is None:
    data_source = "NBA box scores on cdn.nba.com (nba_api live)"
    st.info(
        "stats.nba.com is not responding (it often blocks cloud servers), so team ratings and "
        "player game logs are being built from official NBA box scores instead."
    )
    try:
        with st.spinner("Loading NBA schedule and box scores…"):
            team_stats, logs = load_from_boxscores(season, season_type, game_date)
    except Exception as e:
        st.error(f"Could not load team or player stats.\n\n- stats.nba.com: {stats_err}\n- cdn.nba.com: {e}")
        st.subheader("Schedule")
        st.dataframe(sched[["tip_et", "away", "home", "status", "mkt_details", "mkt_total"]], hide_index=True)
        st.stop()
    if team_stats.empty or logs.empty:
        st.error("No finished games found to build ratings from.")
        st.stop()

# Use only games played before the slate date (no look-ahead when viewing past slates).
logs = logs[logs["GAME_DATE"] < game_date]
lg = league_baselines(team_stats)

for note in st.session_state.get("stale_notes", {}).values():
    st.warning(note)

# ---- Project the slate ---------------------------------------------------
pools: dict[str, tuple[pd.DataFrame, dict]] = {}


def team_pool(abbr: str):
    if abbr not in pools:
        pools[abbr] = build_team_pool(logs, abbr, injuries, q_sit_prob)
    return pools[abbr]


slate_rows, projections = [], {}
for g in sched.itertuples():
    if g.home not in team_stats.index or g.away not in team_stats.index:
        continue
    pool_h, meta_h = team_pool(g.home)
    pool_a, meta_a = team_pool(g.away)
    proj = project_game(g.home, g.away, team_stats, lg, b2b_teams, meta_h, meta_a)
    projections[g.game_id] = proj
    score = f"{g.away_score}-{g.home_score}" if g.state in ("in", "post") else ""
    slate_rows.append(
        {
            "Tip (ET)": g.tip_et.strftime("%I:%M %p").lstrip("0") if pd.notna(g.tip_et) else "",
            "Matchup": f"{g.away} @ {g.home}",
            "Status": f"{g.status} {score}".strip(),
            "B2B": ", ".join(t for t in (g.away, g.home) if t in b2b_teams) or "-",
            "Proj Score": f"{proj['pts_away']:.0f}-{proj['pts_home']:.0f}",
            "Proj Spread (Home)": fmt_spread(proj["spread_home"]),
            "Market Spread": fmt_spread(g.mkt_spread_home),
            "Spread Edge": fmt_spread(g.mkt_spread_home - proj["spread_home"]) if g.mkt_spread_home is not None else "-",
            "Proj Total": round(proj["total"], 1),
            "Market Total": g.mkt_total if g.mkt_total is not None else "-",
            "Total Edge": f"{proj['total'] - g.mkt_total:+.1f}" if g.mkt_total is not None else "-",
            "Home Win %": f"{proj['home_win_prob'] * 100:.0f}%",
        }
    )

st.subheader(f"Slate — {game_date:%A, %B %d, %Y}")
st.caption(
    f"Team ratings: {stats_season} {season_type} via {data_source} · League avg pace {lg['pace']:.1f}, ORtg {lg['rtg']:.1f} · "
    "Spread edge > 0 means the model likes the home side vs. the market; total edge > 0 leans over."
)
st.dataframe(pd.DataFrame(slate_rows), hide_index=True, width="stretch")

# ---- Game detail ---------------------------------------------------------
games = [g for g in sched.itertuples() if g.game_id in projections]
if not games:
    st.stop()
labels = {g.game_id: f"{g.away_name} @ {g.home_name} — {g.status}" for g in games}
gid = st.selectbox("Game detail", list(labels), format_func=labels.get)
g = next(x for x in games if x.game_id == gid)
proj = projections[gid]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{g.home} spread", fmt_spread(proj["spread_home"]),
          help=f"Market: {g.mkt_details or 'n/a'}")
c2.metric("Total", f"{proj['total']:.1f}",
          delta=f"{proj['total'] - g.mkt_total:+.1f} vs market" if g.mkt_total is not None else None)
c3.metric(f"{g.home} win prob", f"{proj['home_win_prob'] * 100:.0f}%")
c4.metric("Projected possessions", f"{proj['poss']:.1f}")

tab_game, tab_props, tab_logs = st.tabs(["Team factors & injuries", "Player props", "Last-10 game logs"])

with tab_game:
    rows = []
    for side, abbr in (("Away", g.away), ("Home", g.home)):
        r = team_stats.loc[abbr]
        meta = team_pool(abbr)[1]
        rows.append(
            {
                "Side": side,
                "Team": abbr,
                "Record": f"{int(r['W'])}-{int(r['L'])}",
                "Pace": round(r["PACE"], 1),
                "OffRtg": round(r["OFF_RATING"], 1),
                "DefRtg": round(r["DEF_RATING"], 1),
                "NetRtg": round(r["NET_RATING"], 1),
                "Back-to-back": "Yes" if abbr in b2b_teams else "No",
                "Injury-weighted PPG missing": round(meta["lost_pts_season"], 1),
                "Adj. matchup ORtg": round(proj["ortg_home" if side == "Home" else "ortg_away"], 1),
                "Proj PTS": round(proj["pts_home" if side == "Home" else "pts_away"], 1),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    ic1, ic2 = st.columns(2)
    for col, abbr in ((ic1, g.away), (ic2, g.home)):
        with col:
            st.markdown(f"**{abbr} injury report (ESPN)**")
            ti = injuries[injuries["team"] == abbr] if not injuries.empty else injuries
            if ti.empty:
                st.caption("No players listed.")
            else:
                st.dataframe(ti[["player", "status", "detail"]], hide_index=True, width="stretch")

with tab_props:
    for abbr, opp in ((g.away, g.home), (g.home, g.away)):
        pool, meta = team_pool(abbr)
        props = project_players(logs, pool, meta, abbr, opp, team_stats, lg, proj["poss"], abbr in b2b_teams)
        st.markdown(f"#### {abbr} vs {opp}" + ("  ·  back-to-back" if abbr in b2b_teams else ""))
        if meta["lost_min_recent"] > 0:
            st.caption(
                f"Absent-teammate adjustment: {meta['lost_min_recent']:.0f} min, "
                f"{meta['lost_pts_recent']:.1f} pts, {meta['lost_reb_recent']:.1f} reb, "
                f"{meta['lost_ast_recent']:.1f} ast per game redistributed."
            )
        if props.empty:
            st.caption("No rotation players with recent game logs.")
            continue
        show = props.drop(columns=["_pid"])
        num_cols = [c for c in show.columns if c not in ("Player", "Status")]
        st.dataframe(
            show.style.format({c: "{:.1f}" for c in num_cols}).format(
                {c: "{:.0f}%" for c in num_cols if c.endswith("Over%")}
            ),
            hide_index=True,
            width="stretch",
        )

with tab_logs:
    opts = []
    for abbr in (g.away, g.home):
        pool, _ = team_pool(abbr)
        if not pool.empty:
            opts += [(abbr, r.PLAYER_ID, r.PLAYER) for r in pool.sort_values("L10_MIN", ascending=False).itertuples()]
    if opts:
        sel = st.selectbox("Player", opts, format_func=lambda o: f"{o[2]} ({o[0]})")
        pl = logs[logs["PLAYER_ID"] == sel[1]].sort_values("GAME_DATE", ascending=False).head(10)
        cols = [c for c in ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "FG3M", "PLUS_MINUS"] if c in pl]
        st.dataframe(pl[cols], hide_index=True, width="stretch")

with st.expander("Methodology"):
    st.markdown(
        f"""
**Game model**
- Possessions = Pace_home × Pace_away / League pace (−0.5 per team on a back-to-back).
- Matchup ORtg = Team OffRtg × Opp DefRtg / League rating, ±{HOME_COURT_RTG} per 100 for home court.
- Injuries: each Out (or Questionable × sit-probability) player's season scoring, weighted by the share of
  team games they played, lowers team OffRtg by {INJ_OFF_FACTOR:.0%} of the missing scoring share and raises
  DefRtg by {INJ_DEF_PER_MIN_SHARE} × missing minute share (season ratings only include them in games they played).
- Back-to-back: −{B2B_OFF_PENALTY} OffRtg, +{B2B_DEF_PENALTY} DefRtg (detected from the previous day's ESPN schedule).
- Spread = −(home pts − away pts); win probability uses a normal margin with σ = {MARGIN_SD}.

**Player props**
- Recency-weighted last-10 per-minute PTS/REB/AST × projected minutes.
- Minutes: last-10 average plus a proportional share of absent teammates' recent minutes (capped at {MAX_PLAYER_MIN:.0f}),
  ×{B2B_MIN_FACTOR} on a back-to-back.
- Pace factor = game possessions / team pace. Opponent factor: PTS by opp DefRtg; REB/AST by opp
  rebounds/assists allowed per possession, each relative to league average.
- Teammate usage bump from missing scoring/rebounding/assists of absent teammates.
- Lines are rounded to the nearest .5; "L10 Over%" is how often the player cleared that line in their last 10.
"""
    )

st.caption(f"Rendered {now_et:%Y-%m-%d %I:%M %p} ET. Projections are estimates, not betting advice.")
