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
import html as htmlmod
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
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
CDN_DOWN_FLAG = "cdn_nba_down.flag"
# ESPN season.type codes by season type
ESPN_SEASON_TYPE = {"Regular Season": 2, "Playoffs": 3, "PlayIn": 5}

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


def flag_active(name: str, ttl: int = STATS_DOWN_TTL) -> bool:
    flag = CACHE_DIR / name
    return flag.exists() and time.time() - flag.stat().st_mtime < ttl


def cdn_get(url: str) -> dict:
    if flag_active(CDN_DOWN_FLAG):
        raise RuntimeError("cdn.nba.com did not respond recently; skipping it for now")
    last_err: Exception | None = None
    for attempt in range(2):
        throttle()
        try:
            r = _session.get(url, headers=CDN_HEADERS, timeout=NBA_STATS_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            last_err = err
            break
        except Exception as err:
            last_err = err
            time.sleep(1.5)
    (CACHE_DIR / CDN_DOWN_FLAG).touch()
    raise RuntimeError(f"cdn.nba.com request failed ({type(last_err).__name__}: {last_err})") from last_err


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
                "home_short": h["team"].get("shortDisplayName") or h["team"].get("name") or h_espn,
                "away_short": a["team"].get("shortDisplayName") or a["team"].get("name") or a_espn,
                "home_logo": h["team"].get("logo", ""),
                "away_logo": a["team"].get("logo", ""),
                "home_record": ((h.get("records") or [{}])[0] or {}).get("summary", ""),
                "away_record": ((a.get("records") or [{}])[0] or {}).get("summary", ""),
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
# Fallback loaders: box scores from cdn.nba.com (nba_api live) or ESPN summaries
# Both are reduced to the same compact per-game format, then aggregated into
# team ratings and a LeagueGameLog-style player table.
# --------------------------------------------------------------------------
def _num(v) -> float:
    if isinstance(v, dict):
        v = v.get("value", v.get("displayValue"))
    if isinstance(v, str):
        v = v.strip().replace("+", "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _made_att(v) -> tuple[float, float]:
    """'40-88' -> (40, 88)."""
    if isinstance(v, str) and "-" in v.lstrip("-"):
        m, _, a = v.partition("-")
        return _num(m), _num(a)
    return 0.0, 0.0


def _parse_date(raw: str) -> date | None:
    raw = (raw or "").strip()
    for fmt, n in (("%Y-%m-%d", 10), ("%m/%d/%Y", 10), ("%Y%m%d", 8)):
        try:
            return datetime.strptime(raw[:n], fmt).date()
        except ValueError:
            continue
    return None


def _season_window(season: str, season_type: str) -> tuple[date, date]:
    start_year = int(season[:4])
    if season_type == "Regular Season":
        return date(start_year, 9, 15), date(start_year + 1, 4, 30)
    return date(start_year + 1, 4, 1), date(start_year + 1, 7, 1)


def _box_key(src: str, gid: str) -> str:
    return f"{src}_box|{gid}"


# ---- cdn.nba.com ---------------------------------------------------------
def _compact_cdn_boxscore(payload: dict, game_date: str) -> dict:
    g = payload.get("game", payload)
    teams = []
    for side in ("homeTeam", "awayTeam"):
        t = g.get(side) or {}
        s = t.get("statistics", {}) or {}
        players = []
        for p in t.get("players", []) or []:
            ps = p.get("statistics", {}) or {}
            mins = to_minutes(ps.get("minutes"))
            if str(p.get("played", "1")) != "1" or mins <= 0:
                continue
            players.append(
                {
                    "id": p.get("personId"),
                    "name": p.get("name") or f"{p.get('firstName', '')} {p.get('familyName', '')}".strip(),
                    "min": mins,
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


def cdn_finished_games(season: str, season_type: str, before: date) -> list[tuple[str, str]]:
    data = disk_cached("cdn_schedule", TTL_CDN_SCHEDULE, lambda: cdn_get(CDN_SCHEDULE))
    ls = data.get("leagueSchedule") or data.get("leagueschedule") or {}
    prefix = GAME_ID_PREFIX.get(season_type, "002")
    lo, hi = _season_window(season, season_type)
    out = []
    for gd in ls.get("gameDates", []) or []:
        for g in gd.get("games", []) or []:
            gid = str(g.get("gameId", ""))
            try:
                status = int(g.get("gameStatus") or 0)
            except (TypeError, ValueError):
                status = 0
            if not gid.startswith(prefix) or status != 3:
                continue
            gdate = None
            for k in ("gameDateEst", "gameDateTimeEst", "gameDate"):
                gdate = _parse_date(g.get(k, ""))
                if gdate:
                    break
            gdate = gdate or _parse_date(gd.get("gameDate", ""))
            if gdate and lo <= gdate < min(before, hi):
                out.append((gid, gdate.isoformat()))
    return out


def cdn_fetch_boxscore(game_id: str, game_date: str) -> None:
    def _fetch():
        if flag_active(CDN_DOWN_FLAG):
            raise RuntimeError("cdn.nba.com marked down")
        throttle()
        payload = live_boxscore.BoxScore(game_id, headers=CDN_HEADERS, timeout=NBA_STATS_TIMEOUT).get_dict()
        return _compact_cdn_boxscore(payload, game_date)

    disk_cached(_box_key("cdn", game_id), TTL_FINAL_BOXSCORE, _fetch)


# ---- ESPN ----------------------------------------------------------------
def _espn_events(payload: dict) -> list[dict]:
    return payload.get("events", []) or []


def espn_finished_games(season: str, season_type: str, before: date) -> list[tuple[str, str]]:
    """Completed games from ESPN's scoreboard: one date-range request, per-day as a fallback."""
    lo, hi = _season_window(season, season_type)
    end = min(before - timedelta(days=1), hi)
    if end < lo:
        return []
    want_type = ESPN_SEASON_TYPE.get(season_type)

    def keep(ev) -> tuple[str, str] | None:
        st_type = ((ev.get("status") or {}).get("type") or {})
        if st_type.get("state") != "post" and not st_type.get("completed"):
            return None
        ev_type = (ev.get("season") or {}).get("type")
        if want_type and ev_type and int(ev_type) != want_type:
            return None
        tip = pd.to_datetime(ev.get("date"), utc=True, errors="coerce")
        if pd.isna(tip):
            return None
        gdate = tip.tz_convert(ET).date()
        return (str(ev.get("id")), gdate.isoformat()) if lo <= gdate < before else None

    games: dict[str, str] = {}
    rng = f"{lo:%Y%m%d}-{end:%Y%m%d}"
    try:
        data = disk_cached(f"espn_scoreboard_range|{rng}", TTL_CDN_SCHEDULE,
                           lambda: espn_get(ESPN_SCOREBOARD, {"dates": rng, "limit": 1500}))
        for ev in _espn_events(data):
            k = keep(ev)
            if k:
                games[k[0]] = k[1]
    except Exception:
        pass
    if not games:  # range query unsupported or empty -> walk the days
        d = lo
        while d <= end:
            ymd = d.strftime("%Y%m%d")
            try:
                data = disk_cached(f"espn_scoreboard|{ymd}", TTL_SCOREBOARD_PAST,
                                   lambda ymd=ymd: espn_get(ESPN_SCOREBOARD, {"dates": ymd}))
                for ev in _espn_events(data):
                    k = keep(ev)
                    if k:
                        games[k[0]] = k[1]
            except Exception:
                pass
            d += timedelta(days=1)
    return sorted(games.items(), key=lambda x: x[1])


def _col_index(block: dict, *cands: str) -> int | None:
    for arr_name in ("keys", "names", "labels"):
        arr = [str(x).lower() for x in (block.get(arr_name) or [])]
        for c in cands:
            if c.lower() in arr:
                return arr.index(c.lower())
    return None


def _compact_espn_summary(payload: dict, game_date: str) -> dict:
    comp = ((payload.get("header") or {}).get("competitions") or [{}])[0]
    competitors = comp.get("competitors", []) or []
    periods = max([len(c.get("linescores") or []) for c in competitors] + [4])
    team_min = 240.0 + 25.0 * max(0, periods - 4)

    teams: dict[str, dict] = {}
    for c in competitors:
        tm = c.get("team") or {}
        tid = str(tm.get("id") or c.get("id"))
        tri = espn_abbr_to_nba(tm.get("abbreviation", ""))
        teams[tid] = {
            "side": "homeTeam" if c.get("homeAway") == "home" else "awayTeam",
            "id": NBA_TEAM_BY_ABBR.get(tri, {}).get("id", tid),
            "tri": tri,
            "pts": _num(c.get("score")),
            "fga": 0.0, "fta": 0.0, "oreb": 0.0, "tov": 0.0, "reb": 0.0, "ast": 0.0,
            "min": team_min,
            "players": [],
        }

    box = payload.get("boxscore") or {}
    for t in box.get("teams", []) or []:
        tid = str((t.get("team") or {}).get("id"))
        if tid not in teams:
            continue
        stats = {}
        for s in t.get("statistics", []) or []:
            for k in (s.get("name"), s.get("abbreviation"), s.get("label")):
                if k:
                    stats[str(k).lower()] = s.get("displayValue")
        T = teams[tid]
        T["fga"] = _made_att(stats.get("fieldgoalsmade-fieldgoalsattempted") or stats.get("fg"))[1]
        T["fta"] = _made_att(stats.get("freethrowsmade-freethrowsattempted") or stats.get("ft"))[1]
        T["oreb"] = _num(stats.get("offensiverebounds") or stats.get("oreb") or stats.get("offensive rebounds"))
        T["tov"] = _num(stats.get("totalturnovers") or stats.get("turnovers") or stats.get("to"))
        T["reb"] = _num(stats.get("totalrebounds") or stats.get("rebounds") or stats.get("reb"))
        T["ast"] = _num(stats.get("assists") or stats.get("ast"))

    for pt in box.get("players", []) or []:
        tid = str((pt.get("team") or {}).get("id"))
        if tid not in teams:
            continue
        T = teams[tid]
        sums = {"fga": 0.0, "fta": 0.0, "oreb": 0.0, "tov": 0.0, "reb": 0.0, "ast": 0.0}
        for blk in pt.get("statistics", []) or []:
            ix = {
                "min": _col_index(blk, "minutes", "min"),
                "pts": _col_index(blk, "points", "pts"),
                "reb": _col_index(blk, "rebounds", "totalRebounds", "reb"),
                "ast": _col_index(blk, "assists", "ast"),
                "fg": _col_index(blk, "fieldGoalsMade-fieldGoalsAttempted", "fg"),
                "fg3": _col_index(blk, "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "3pt"),
                "ft": _col_index(blk, "freeThrowsMade-freeThrowsAttempted", "ft"),
                "oreb": _col_index(blk, "offensiveRebounds", "oreb"),
                "tov": _col_index(blk, "turnovers", "to"),
                "pm": _col_index(blk, "plusMinus", "+/-"),
            }

            def val(stats, key):
                i = ix[key]
                return stats[i] if i is not None and i < len(stats) else None

            for ath in blk.get("athletes", []) or []:
                stats = ath.get("stats") or []
                if ath.get("didNotPlay") or not stats:
                    continue
                mins = to_minutes(val(stats, "min"))
                if mins <= 0:
                    continue
                a = ath.get("athlete") or {}
                T["players"].append(
                    {
                        "id": a.get("id"),
                        "name": a.get("displayName") or a.get("shortName") or "",
                        "min": mins,
                        "pts": _num(val(stats, "pts")),
                        "reb": _num(val(stats, "reb")),
                        "ast": _num(val(stats, "ast")),
                        "fg3m": _made_att(val(stats, "fg3"))[0],
                        "pm": _num(val(stats, "pm")),
                    }
                )
                sums["fga"] += _made_att(val(stats, "fg"))[1]
                sums["fta"] += _made_att(val(stats, "ft"))[1]
                sums["oreb"] += _num(val(stats, "oreb"))
                sums["tov"] += _num(val(stats, "tov"))
                sums["reb"] += _num(val(stats, "reb"))
                sums["ast"] += _num(val(stats, "ast"))
        for k, v in sums.items():  # team totals missing -> use player sums
            if not T[k]:
                T[k] = v
        if not T["pts"]:
            T["pts"] = sum(p["pts"] for p in T["players"])
    return {"game_id": str((payload.get("header") or {}).get("id", "")), "date": game_date,
            "teams": list(teams.values())}


def espn_fetch_boxscore(event_id: str, game_date: str) -> None:
    def _fetch():
        return _compact_espn_summary(espn_get(ESPN_SUMMARY, {"event": event_id}), game_date)

    disk_cached(_box_key("espn", event_id), TTL_FINAL_BOXSCORE, _fetch)


BOX_SOURCES = {
    "cdn": ("NBA box scores on cdn.nba.com (nba_api live)", cdn_finished_games, cdn_fetch_boxscore),
    "espn": ("ESPN game box scores", espn_finished_games, espn_fetch_boxscore),
}


# ---- aggregation -----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def build_from_boxscores(src: str, game_keys: tuple[tuple[str, str], ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team ratings (Pace/OffRtg/DefRtg/opp REB & AST) and a LeagueGameLog-style player table."""
    team_rows, player_rows = [], []
    for gid, gdate in game_keys:
        path = _cache_path(_box_key(src, gid))
        if not path.exists():
            continue
        try:
            with path.open() as f:
                box = json.load(f)
            a, b = box["teams"]
            if not a.get("tri") or not b.get("tri") or a["tri"] == b["tri"]:
                continue
            poss_a = a["fga"] + 0.44 * a["fta"] - a["oreb"] + a["tov"]
            poss_b = b["fga"] + 0.44 * b["fta"] - b["oreb"] + b["tov"]
            game_poss = (poss_a + poss_b) / 2
            if game_poss < 60 or a["pts"] <= 0 or b["pts"] <= 0:  # incomplete box score
                continue
        except Exception:
            continue
        for t, o in ((a, b), (b, a)):
            win = t["pts"] > o["pts"]
            team_rows.append(
                {"ABBR": t["tri"], "TEAM_ID": t["id"], "W": int(win), "L": int(not win), "PTS": t["pts"],
                 "OPP_PTS": o["pts"], "POSS": game_poss, "MIN": t["min"] or 240.0,
                 "OPP_REB": o["reb"], "OPP_AST": o["ast"]}
            )
            matchup = f"{t['tri']} vs. {o['tri']}" if t["side"] == "homeTeam" else f"{t['tri']} @ {o['tri']}"
            for p in t["players"]:
                if not p.get("id") or not p.get("name"):
                    continue
                player_rows.append(
                    {"PLAYER_ID": str(p["id"]), "PLAYER_NAME": p["name"], "TEAM_ID": t["id"],
                     "TEAM_ABBREVIATION": t["tri"], "GAME_ID": gid, "GAME_DATE": date.fromisoformat(gdate),
                     "MATCHUP": matchup, "WL": "W" if win else "L", "MIN": p["min"], "PTS": p["pts"],
                     "REB": p["reb"], "AST": p["ast"], "FG3M": p["fg3m"], "PLUS_MINUS": p["pm"]}
                )
    tg = pd.DataFrame(team_rows)
    if tg.empty or not player_rows:
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


def load_from_boxscores(src: str, season: str, season_type: str, before: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    label, list_games, fetch_one = BOX_SOURCES[src]
    games = list_games(season, season_type, before)
    if not games:
        raise RuntimeError(f"no finished {season} {season_type} games found before {before}")
    missing = [g for g in games if not _cache_path(_box_key(src, g[0])).exists()]
    failures = 0
    if missing:
        bar = st.progress(0.0)
        for i, (gid, gdate) in enumerate(missing, 1):
            try:
                fetch_one(gid, gdate)
            except Exception:
                failures += 1
                # Source clearly unreachable: stop early instead of timing out on every game.
                if i >= 5 and failures == i:
                    break
            bar.progress(i / len(missing), text=f"Loading {label}: {i}/{len(missing)} "
                                                 "(first load only — cached afterwards)")
        bar.empty()
    have = sum(_cache_path(_box_key(src, g[0])).exists() for g in games)
    if have < max(1, 0.5 * len(games)):
        raise RuntimeError(f"only {have} of {len(games)} box scores could be downloaded")
    team_stats, logs = build_from_boxscores(src, tuple(games))
    if team_stats.empty or logs.empty or len(team_stats) < 2:
        raise RuntimeError("box scores did not contain usable team/player stats")
    return team_stats, logs


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
        g5 = g.head(5)
        row = {
            "Player": r["PLAYER"],
            "Team": team,
            "Opp": opp,
            "Class": r["CLASS"],
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
            row[f"L5 {s}"] = float(g5[s].mean())
            row[f"L5 {s} vals"] = [float(x) for x in g5[s]]
        pra = proj["PTS"] + proj["REB"] + proj["AST"]
        row["Proj PRA"] = pra
        row["PRA Line"] = book_line(pra)
        g5_pra = g5["PTS"] + g5["REB"] + g5["AST"]
        row["L5 PRA"] = float(g5_pra.mean())
        row["L5 PRA vals"] = [float(x) for x in g5_pra]
        row["PRA L10 Over%"] = float(((g["PTS"] + g["REB"] + g["AST"]) > row["PRA Line"]).mean() * 100)
        row["_pid"] = r["PLAYER_ID"]
        out.append(row)
    return pd.DataFrame(out).sort_values("Proj PTS", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# UI  (sportsbook-style dark cards; every block degrades gracefully)
# --------------------------------------------------------------------------
st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="centered",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
:root{--bg:#0b0f14;--card:#121821;--card2:#18202b;--line:#232d3a;--text:#e8edf3;--muted:#8b97a7;
--green:#22c55e;--green-bg:rgba(34,197,94,.14);--red:#f43f5e;--red-bg:rgba(244,63,94,.14);
--amber:#f59e0b;--amber-bg:rgba(245,158,11,.15);--blue:#38bdf8;--blue-bg:rgba(56,189,248,.14)}
.stApp{background:var(--bg)}
.block-container{padding:1.2rem .8rem 3rem;max-width:760px}
#MainMenu,footer{visibility:hidden}
h1,h2,h3{letter-spacing:-.01em}
.hero{display:flex;justify-content:space-between;align-items:flex-end;margin:.2rem 0 .8rem}
.brand{font-size:1.45rem;font-weight:800;color:var(--text)}
.brand span{color:var(--green)}
.hero-sub{color:var(--muted);font-size:.82rem;text-align:right;line-height:1.3}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--card);border:1px solid var(--line);
border-radius:14px;padding:4px}
.stTabs [data-baseweb="tab"]{flex:1;justify-content:center;border-radius:10px;padding:.55rem .2rem;
color:var(--muted);font-weight:600;height:auto}
.stTabs [data-baseweb="tab"] p{font-size:.86rem}
.stTabs [aria-selected="true"]{background:var(--card2);color:var(--text)}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px 14px 12px;
margin:0 0 12px;color:var(--text)}
.ghead{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:8px}
.gtime{color:var(--muted);font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.tags{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.pill{display:inline-block;font-size:.66rem;font-weight:700;padding:3px 7px;border-radius:999px;
letter-spacing:.03em;text-transform:uppercase;white-space:nowrap}
.pill.live{background:var(--red-bg);color:var(--red)}
.pill.final{background:var(--card2);color:var(--muted)}
.pill.b2b{background:var(--amber-bg);color:var(--amber)}
.pill.out{background:var(--red-bg);color:var(--red)}
.pill.q{background:var(--amber-bg);color:var(--amber)}
.pill.ok{background:var(--green-bg);color:var(--green)}
.pill.info{background:var(--blue-bg);color:var(--blue)}
.trow{display:flex;align-items:center;gap:10px;padding:6px 0}
.logo{width:34px;height:34px;object-fit:contain;flex:none}
.logo-fb{width:34px;height:34px;border-radius:50%;background:var(--card2);display:flex;align-items:center;
justify-content:center;font-size:.62rem;font-weight:800;color:var(--muted);flex:none}
.tinfo{flex:1;min-width:0}
.tabbr{font-weight:800;font-size:1rem}
.tabbr .rec{color:var(--muted);font-weight:500;font-size:.75rem;margin-left:4px}
.tname{color:var(--muted);font-size:.75rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tscore{font-weight:800;font-size:1.3rem;min-width:34px;text-align:right}
.tproj{text-align:right;min-width:62px}
.tproj .lbl,.lbl{display:block;color:var(--muted);font-size:.6rem;font-weight:700;letter-spacing:.06em;
text-transform:uppercase}
.tproj .num{font-weight:800;font-size:1.15rem;color:var(--green)}
.winbar{display:flex;height:6px;border-radius:999px;overflow:hidden;background:var(--card2);margin:8px 0 4px}
.winbar .a{background:#64748b}.winbar .h{background:var(--green)}
.winlbl{display:flex;justify-content:space-between;color:var(--muted);font-size:.7rem;margin-bottom:10px}
.mkts{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.mkt{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:9px 8px;text-align:center}
.mkt .val{font-weight:800;font-size:1rem;margin:2px 0}
.mkt .sub{color:var(--muted);font-size:.68rem;margin-bottom:6px;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.badge{display:inline-block;font-size:.64rem;font-weight:800;padding:4px 8px;border-radius:8px;
letter-spacing:.02em;white-space:nowrap}
.badge.strong{background:var(--green);color:#04210f}
.badge.lean{background:var(--green-bg);color:var(--green)}
.badge.under-strong{background:var(--red);color:#2a0710}
.badge.under-lean{background:var(--red-bg);color:var(--red)}
.badge.muted{background:transparent;color:var(--muted);border:1px solid var(--line)}
.note{color:var(--muted);font-size:.74rem;margin:2px 2px 12px;line-height:1.45}
.phead{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
.pname{font-weight:800;font-size:1rem}
.psub{color:var(--muted);font-size:.74rem;margin-top:2px}
.pgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px}
.pbox{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:7px 4px;text-align:center}
.pbox .num{font-weight:800;font-size:1.05rem;margin-top:1px}
.pbox.hl{border-color:rgba(34,197,94,.45)}
.pbox.hl .num{color:var(--green)}
.l5{display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin-bottom:8px}
.chip{font-size:.72rem;font-weight:700;padding:3px 7px;border-radius:7px;min-width:26px;text-align:center}
.chip.o{background:var(--green-bg);color:var(--green)}
.chip.u{background:var(--red-bg);color:var(--red)}
.mini{display:flex;justify-content:space-between;color:var(--muted);font-size:.72rem;border-top:1px solid var(--line);
padding-top:8px}
.mini b{color:var(--text)}
.ihead{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.irow{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:8px 0;
border-top:1px solid var(--line)}
.irow .who{min-width:0}
.irow .nm{font-weight:700;font-size:.9rem}
.irow .dt{color:var(--muted);font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:56vw}
.empty{color:var(--muted);font-size:.8rem;padding:6px 0}
div[data-testid="stExpander"]{background:var(--card);border:1px solid var(--line);border-radius:14px}
@media (max-width:420px){.mkt .val{font-size:.92rem}.pbox .num{font-size:.95rem}.brand{font-size:1.25rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

esc = htmlmod.escape


def H(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def logo_html(url: str, abbr: str) -> str:
    if url:
        return f'<img class="logo" src="{esc(url)}" alt="{esc(abbr)}">'
    return f'<div class="logo-fb">{esc(abbr)}</div>'


def edge_badge(edge: float | None, lean: float, strong: float, pos: str, neg: str, neg_is_red: bool = False) -> str:
    if edge is None or (isinstance(edge, float) and np.isnan(edge)):
        return '<span class="badge muted">NO LINE</span>'
    a = abs(edge)
    if a < lean:
        return '<span class="badge muted">NO EDGE</span>'
    level = "strong" if a >= strong else "lean"
    cls = f"under-{level}" if (edge < 0 and neg_is_red) else level
    label = pos if edge > 0 else neg
    return f'<span class="badge {cls}">{esc(label)} · {a:.1f}</span>'


def fav_line(spread_home: float, home: str, away: str) -> str:
    if abs(spread_home) < 0.25:
        return "PK"
    return f"{home} -{abs(spread_home):.1f}" if spread_home < 0 else f"{away} -{abs(spread_home):.1f}"


def status_pill(cls: str, status: str) -> str:
    if cls == "OUT":
        return f'<span class="pill out">{esc(status or "Out")}</span>'
    if cls == "Q":
        return f'<span class="pill q">{esc(status or "Questionable")}</span>'
    return ""


now_et = datetime.now(ET)
today_et = now_et.date()

# ---- Settings (expander works better than a sidebar on phones) ----------
with st.expander("⚙️  Settings", expanded=False):
    game_date = st.date_input("Slate date (ET)", value=today_et)
    c1, c2 = st.columns(2)
    season = c1.text_input("Season", value=season_for(game_date))
    season_type = c2.selectbox("Season type", ["Regular Season", "Playoffs", "PlayIn"], index=0)
    q_sit_prob = st.slider("Chance a Questionable / Day-to-Day player sits", 0.0, 1.0, 0.5, 0.05)
    if st.button("🔄 Refresh all data", width="stretch"):
        clear_disk_cache()
        st.cache_data.clear()
        st.session_state.pop("stale_notes", None)
        st.rerun()

# ---- Schedule (ESPN) -----------------------------------------------------
try:
    with st.spinner("Loading today's games…"):
        sched = load_schedule(game_date, today_et)
except Exception as e:
    H(f'<div class="card"><b>Couldn\'t load the ESPN schedule.</b><div class="note">{esc(str(e))}</div></div>')
    st.stop()
try:
    sched_prev = load_schedule(game_date - timedelta(days=1), today_et)
except Exception:
    sched_prev = pd.DataFrame(columns=["home", "away"])

H(f'<div class="hero"><div class="brand">🏀 NBA <span>Predictor</span></div>'
  f'<div class="hero-sub">{game_date:%a, %b %d}<br>{len(sched)} game{"s" if len(sched) != 1 else ""}</div></div>')

if sched.empty:
    H(f'<div class="card"><b>No NBA games on {game_date:%A, %B %d}.</b>'
      '<div class="note">Pick another date in Settings.</div></div>')
    st.stop()

b2b_teams = set(sched_prev["home"]).union(sched_prev["away"]) if not sched_prev.empty else set()

# ---- Injuries (ESPN) -----------------------------------------------------
injury_note = ""
try:
    injuries = load_injuries()
except Exception as e:
    injuries = pd.DataFrame(columns=["team", "player", "key", "status", "detail", "updated"])
    injury_note = f"Injury report unavailable ({e}); projections run without injury adjustments."

# ---- Team ratings + player logs: stats.nba.com -> cdn.nba.com -> ESPN ----
source_errors: dict[str, str] = {}
team_stats = logs = None
data_source = ""
stats_season = season

try:
    with st.spinner("Loading team ratings from stats.nba.com…"):
        ts_ = load_team_stats(season, season_type)
        lg_ = load_player_logs(season, season_type)
        if ts_.empty or lg_.empty or not (lg_["GAME_DATE"] < game_date).any():
            stats_season = prev_season(season)
            ts_ = load_team_stats(stats_season, season_type)
            lg_ = load_player_logs(stats_season, season_type)
        if not ts_.empty and not lg_.empty:
            team_stats, logs, data_source = ts_, lg_, "stats.nba.com (nba_api)"
except Exception as e:
    source_errors["stats.nba.com"] = str(e)
    stats_season = season

if team_stats is None:
    for src in ("cdn", "espn"):
        label = BOX_SOURCES[src][0]
        try:
            with st.spinner(f"Loading {label}…"):
                team_stats, logs = load_from_boxscores(src, season, season_type, game_date)
            data_source = label
            break
        except Exception as e:
            source_errors[label] = str(e)
            team_stats = logs = None

projections: dict[str, dict] = {}
pools: dict[str, tuple[pd.DataFrame, dict]] = {}
lg: dict = {}
if team_stats is not None:
    try:
        logs = logs[logs["GAME_DATE"] < game_date]  # no look-ahead on past slates
        lg = league_baselines(team_stats)
    except Exception as e:
        source_errors["model"] = str(e)
        team_stats = None


def team_pool(abbr: str):
    if abbr not in pools:
        try:
            pools[abbr] = build_team_pool(logs, abbr, injuries, q_sit_prob)
        except Exception:
            pools[abbr] = (pd.DataFrame(), {"team_games": 0, "team_ppg": np.nan, "lost_pts_season": 0.0,
                                            "lost_min_season": 0.0, "lost_pts_recent": 0.0, "lost_min_recent": 0.0,
                                            "lost_reb_recent": 0.0, "lost_ast_recent": 0.0})
    return pools[abbr]


if team_stats is not None:
    for g in sched.itertuples():
        try:
            if g.home in team_stats.index and g.away in team_stats.index:
                projections[g.game_id] = project_game(g.home, g.away, team_stats, lg, b2b_teams,
                                                      team_pool(g.home)[1], team_pool(g.away)[1])
        except Exception as e:
            source_errors[f"{g.away}@{g.home}"] = str(e)

tab_picks, tab_props, tab_inj = st.tabs(["🎯 Game Picks", "📊 Player Props", "🩹 Injury Report"])

# ---- Game Picks ----------------------------------------------------------
with tab_picks:
    if team_stats is None:
        H('<div class="card"><b>Projections unavailable right now.</b><div class="note">'
          + "<br>".join(f"{esc(k)}: {esc(v)}" for k, v in source_errors.items())
          + "<br>Matchups and market lines are shown below; tap Refresh in Settings to retry.</div></div>")
    else:
        H('<div class="note">Model line vs. market. Spread badge names the side the model likes; '
          'total badge shows over/under lean. Number = points of edge.</div>')

    for g in sched.itertuples():
        p = projections.get(g.game_id)
        state = g.state
        if state == "in":
            st_pill = f'<span class="pill live">● {esc(g.status)}</span>'
        elif state == "post":
            st_pill = f'<span class="pill final">{esc(g.status)}</span>'
        else:
            st_pill = ""
        tags = "".join(f'<span class="pill b2b">{esc(t)} B2B</span>' for t in (g.away, g.home) if t in b2b_teams)
        tip = g.tip_et.strftime("%-I:%M %p ET") if pd.notna(g.tip_et) else ""

        def trow(abbr, short, rec, logo, score, proj_pts):
            sc = f'<div class="tscore">{esc(str(score))}</div>' if state in ("in", "post") and score not in (None, "") else ""
            pj = (f'<div class="tproj"><span class="lbl">Proj</span><span class="num">{proj_pts:.0f}</span></div>'
                  if proj_pts is not None else "")
            return (f'<div class="trow">{logo_html(logo, abbr)}<div class="tinfo"><div class="tabbr">{esc(abbr)}'
                    f'<span class="rec">{esc(rec or "")}</span></div><div class="tname">{esc(short)}</div></div>'
                    f'{sc}{pj}</div>')

        body = (f'<div class="card"><div class="ghead"><div class="gtime">{esc(tip)}</div>'
                f'<div class="tags">{tags}{st_pill}</div></div>'
                + trow(g.away, g.away_short, g.away_record, g.away_logo, g.away_score, p["pts_away"] if p else None)
                + trow(g.home, g.home_short, g.home_record, g.home_logo, g.home_score, p["pts_home"] if p else None))
        if p:
            hp = p["home_win_prob"] * 100
            body += (f'<div class="winbar"><div class="a" style="width:{100 - hp:.0f}%"></div>'
                     f'<div class="h" style="width:{hp:.0f}%"></div></div>'
                     f'<div class="winlbl"><span>{esc(g.away)} {100 - hp:.0f}%</span>'
                     f'<span>{esc(g.home)} {hp:.0f}%</span></div>')
            mkt_s, mkt_t = g.mkt_spread_home, g.mkt_total
            has_s = mkt_s is not None and not (isinstance(mkt_s, float) and np.isnan(mkt_s))
            has_t = mkt_t is not None and not (isinstance(mkt_t, float) and np.isnan(mkt_t))
            s_edge = (mkt_s - p["spread_home"]) if has_s else None
            s_badge = edge_badge(s_edge, 1.5, 3.0,
                                 f"{g.home} {fmt_spread(mkt_s)}" if has_s else "",
                                 f"{g.away} {fmt_spread(-mkt_s)}" if has_s else "")
            t_edge = (p["total"] - mkt_t) if has_t else None
            t_badge = edge_badge(t_edge, 2.0, 5.0, f"OVER {mkt_t}" if has_t else "",
                                 f"UNDER {mkt_t}" if has_t else "", neg_is_red=True)
            fav = g.home if p["home_win_prob"] >= 0.5 else g.away
            body += (
                '<div class="mkts">'
                f'<div class="mkt"><span class="lbl">Spread</span><div class="val">'
                f'{esc(fav_line(p["spread_home"], g.home, g.away))}</div>'
                f'<div class="sub">Mkt {esc(g.mkt_details or "—")}</div>{s_badge}</div>'
                f'<div class="mkt"><span class="lbl">Total</span><div class="val">{p["total"]:.1f}</div>'
                f'<div class="sub">Mkt {esc(str(mkt_t) if has_t else "—")}</div>{t_badge}</div>'
                f'<div class="mkt"><span class="lbl">Win</span><div class="val">{esc(fav)} '
                f'{max(hp, 100 - hp):.0f}%</div><div class="sub">{p["poss"]:.0f} poss</div>'
                f'<span class="badge muted">MODEL</span></div>'
                '</div>'
            )
        elif team_stats is not None:
            body += '<div class="empty">No projection: team ratings missing for this matchup.</div>'
        body += "</div>"
        H(body)

# ---- Player Props --------------------------------------------------------
PROP_EDGE = {"PTS": (1.5, 3.0), "REB": (0.8, 1.6), "AST": (0.8, 1.6), "PRA": (2.0, 4.0)}

with tab_props:
    games = [g for g in sched.itertuples() if g.game_id in projections]
    if not games:
        H('<div class="card"><b>Player props need team ratings and game logs, which are unavailable right now.</b>'
          '<div class="note">Tap Refresh in Settings to retry.</div></div>')
    else:
        labels = {g.game_id: f"{g.away} @ {g.home}" + (f"  ·  {g.tip_et:%-I:%M %p}" if pd.notna(g.tip_et) else "")
                  for g in games}
        gid = st.selectbox("Game", list(labels), format_func=labels.get, label_visibility="collapsed")
        g = next(x for x in games if x.game_id == gid)
        stat = st.radio("Market", ["PTS", "REB", "AST", "PRA"], horizontal=True, label_visibility="collapsed")
        side = st.radio("Team", ["Both", g.away, g.home], horizontal=True, label_visibility="collapsed")
        H('<div class="note">Edge = projection vs. last-5 average (free data sources publish no sportsbook '
          'prop lines). Chips show the last 5 games, green when above the model line. '
          'Hit = share of last 10 games over the model line.</div>')

        frames = []
        for abbr, opp in ((g.away, g.home), (g.home, g.away)):
            if side not in ("Both", abbr):
                continue
            try:
                pool, meta = team_pool(abbr)
                df = project_players(logs, pool, meta, abbr, opp, team_stats, lg, projections[gid]["poss"],
                                     abbr in b2b_teams)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                H(f'<div class="card"><div class="empty">Could not project {esc(abbr)} players: {esc(str(e))}</div></div>')
        props = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if props.empty:
            H('<div class="card"><div class="empty">No rotation players with recent game logs.</div></div>')
        else:
            props["_edge"] = props[f"Proj {stat}"] - props[f"L5 {stat}"]
            props = props.reindex(props["_edge"].abs().sort_values(ascending=False).index)
            lean, strong = PROP_EDGE[stat]
            for r in props.to_dict("records"):
                line = r[f"{stat} Line"]
                chips = "".join(f'<span class="chip {"o" if v > line else "u"}">{v:.0f}</span>'
                                for v in r[f"L5 {stat} vals"])
                badge = edge_badge(r["_edge"], lean, strong, "▲ OVER", "▼ UNDER", neg_is_red=True)
                pill = status_pill(r["Class"], r["Status"])
                b2b = '<span class="pill b2b">B2B</span>' if r["Team"] in b2b_teams else ""
                H(
                    f'<div class="card"><div class="phead"><div><div class="pname">{esc(r["Player"])}</div>'
                    f'<div class="psub">{esc(r["Team"])} vs {esc(r["Opp"])} · {r["Proj MIN"]:.0f} min proj '
                    f'{pill} {b2b}</div></div>{badge}</div>'
                    '<div class="pgrid">'
                    f'<div class="pbox hl"><span class="lbl">Proj {stat}</span><div class="num">{r[f"Proj {stat}"]:.1f}</div></div>'
                    f'<div class="pbox"><span class="lbl">Model line</span><div class="num">{line:.1f}</div></div>'
                    f'<div class="pbox"><span class="lbl">L5 avg</span><div class="num">{r[f"L5 {stat}"]:.1f}</div></div>'
                    f'<div class="pbox"><span class="lbl">L10 hit</span><div class="num">{r[f"{stat} L10 Over%"]:.0f}%</div></div>'
                    '</div>'
                    f'<div class="l5"><span class="lbl" style="margin-right:4px">Last 5</span>{chips}</div>'
                    '<div class="mini">'
                    f'<span>PTS <b>{r["Proj PTS"]:.1f}</b> <small>(L5 {r["L5 PTS"]:.1f})</small></span>'
                    f'<span>REB <b>{r["Proj REB"]:.1f}</b> <small>(L5 {r["L5 REB"]:.1f})</small></span>'
                    f'<span>AST <b>{r["Proj AST"]:.1f}</b> <small>(L5 {r["L5 AST"]:.1f})</small></span>'
                    '</div></div>'
                )

# ---- Injury Report -------------------------------------------------------
CLASS_ORDER = {"OUT": 0, "Q": 1, "ACTIVE": 2}

with tab_inj:
    if injury_note:
        H(f'<div class="note">{esc(injury_note)}</div>')
    H(f'<div class="note">ESPN injury report · Questionable/Day-to-Day players are treated as sitting '
      f'{q_sit_prob:.0%} of the time in projections (change in Settings).</div>')
    for g in sched.itertuples():
        for abbr, short, logo in ((g.away, g.away_short, g.away_logo), (g.home, g.home_short, g.home_logo)):
          try:
            ti = injuries[injuries["team"] == abbr].copy() if not injuries.empty else pd.DataFrame()
            ppg = {}
            if team_stats is not None:
                pool = team_pool(abbr)[0]
                if not pool.empty:
                    ppg = dict(zip(pool["KEY"], pool["SEASON_PTS_TEAM"]))
            n_out = 0 if ti.empty else int((ti["status"].map(injury_class) == "OUT").sum())
            head_pill = (f'<span class="pill out">{n_out} out</span>' if n_out
                         else '<span class="pill ok">Healthy</span>' if ti.empty else "")
            rows = ""
            if not ti.empty:
                ti["cls"] = ti["status"].map(injury_class)
                ti = ti.sort_values("cls", key=lambda s: s.map(CLASS_ORDER))
                for r in ti.itertuples():
                    cls = r.cls
                    pill_cls = {"OUT": "out", "Q": "q"}.get(cls, "ok")
                    extra = f" · {ppg[r.key]:.1f} PPG" if r.key in ppg and ppg[r.key] > 0 else ""
                    rows += (f'<div class="irow"><div class="who"><div class="nm">{esc(r.player)}'
                             f'<small style="color:var(--muted);font-weight:500">{esc(extra)}</small></div>'
                             f'<div class="dt">{esc(r.detail or "")}</div></div>'
                             f'<span class="pill {pill_cls}">{esc(r.status or "")}</span></div>')
            else:
                rows = '<div class="empty">No players listed.</div>'
            H(f'<div class="card"><div class="ihead">{logo_html(logo, abbr)}<div class="tinfo">'
              f'<div class="tabbr">{esc(abbr)}</div><div class="tname">{esc(short)} · vs '
              f'{esc(g.home if abbr == g.away else g.away)}</div></div>{head_pill}</div>{rows}</div>')
          except Exception as e:
            H(f'<div class="card"><div class="empty">{esc(abbr)} injury report unavailable: {esc(str(e))}</div></div>')

# ---- Footer --------------------------------------------------------------
notes = list(st.session_state.get("stale_notes", {}).values())
if data_source:
    src_line = f"Ratings & game logs: {data_source}"
    if stats_season != season and data_source.startswith("stats"):
        src_line += f" ({stats_season} season — no {season} games yet)"
    notes.insert(0, src_line)
for k, v in source_errors.items():
    if team_stats is not None:
        notes.append(f"{k} skipped: {v}")
notes.append(f"Updated {now_et:%b %d, %-I:%M %p} ET · Estimates only, not betting advice.")
H('<div class="note" style="margin-top:8px">' + "<br>".join(esc(n) for n in notes) + '</div>')

with st.expander("How the model works"):
    st.markdown(
        f"""
**Games** — Possessions = Pace_home × Pace_away / league pace. Matchup rating = OffRtg × opponent DefRtg /
league average, ±{HOME_COURT_RTG} per 100 for home court. Missing players (Out, or Questionable × sit chance)
lower OffRtg and raise DefRtg in proportion to their scoring and minutes. Back-to-backs cost
{B2B_OFF_PENALTY} OffRtg and {B2B_DEF_PENALTY} DefRtg. Win probability uses a normal margin, σ = {MARGIN_SD}.

**Props** — Recency-weighted last-10 per-minute PTS/REB/AST × projected minutes (absent teammates' minutes
redistributed, capped at {MAX_PLAYER_MIN:.0f}), adjusted for game pace, opponent DefRtg (PTS) and opponent
rebounds/assists allowed per possession, a usage bump when teammates are out, and back-to-back fatigue.

**Data** — ESPN: schedule, scores, market lines, injuries. Team ratings and player logs: stats.nba.com via
nba_api, falling back to NBA box scores on cdn.nba.com and then ESPN box scores when a source is blocked.
"""
    )
