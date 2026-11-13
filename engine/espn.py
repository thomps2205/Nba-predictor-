"""ESPN public JSON endpoints (site.api / site.web.api) and tolerant parsers for team sports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from .common import (cache_fresh, cache_read, disk_cached, http_json, made_att, norm_name, num, to_et,
                     to_et_date, to_minutes)

SITE = "https://site.api.espn.com/apis/site/v2/sports"
WEB = "https://site.web.api.espn.com/apis/common/v3/sports"

TTL_SCOREBOARD_LIVE = 120
TTL_SCOREBOARD_PAST = 12 * 3600
TTL_SCHEDULE = 3 * 3600
TTL_TEAM_STATS = 6 * 3600
TTL_ROSTER = 6 * 3600
TTL_GAMELOG = 4 * 3600
TTL_INJURIES = 15 * 60


@dataclass(frozen=True)
class League:
    key: str            # NBA / NCAAB / NFL / NHL / UFC
    sport: str          # ESPN sport slug
    league: str         # ESPN league slug
    kind: str           # basketball / football / hockey / mma
    label: str
    has_injuries: bool = True


LEAGUES = {
    "NBA": League("NBA", "basketball", "nba", "basketball", "NBA"),
    "NCAAB": League("NCAAB", "basketball", "mens-college-basketball", "basketball", "College Basketball", False),
    "NFL": League("NFL", "football", "nfl", "football", "NFL"),
    "NHL": League("NHL", "hockey", "nhl", "hockey", "NHL"),
    "UFC": League("UFC", "mma", "ufc", "mma", "UFC", False),
}


def _base(lg: League) -> str:
    return f"{SITE}/{lg.sport}/{lg.league}"


# --------------------------------------------------------------------------
# Scoreboard
# --------------------------------------------------------------------------
def scoreboard(lg: League, d: date | None, today: date, days: int = 1) -> dict:
    """Scoreboard for a date (or a `days`-long range starting at d; None = ESPN's current default)."""
    params: dict = {}
    if lg.key == "NCAAB":
        params.update(groups="50", limit="500")
    elif lg.kind in ("football", "hockey", "basketball"):
        params["limit"] = "300"
    if d is not None:
        end = d + timedelta(days=days - 1)
        params["dates"] = d.strftime("%Y%m%d") if days == 1 else f"{d:%Y%m%d}-{end:%Y%m%d}"
        ttl = TTL_SCOREBOARD_PAST if end < today else TTL_SCOREBOARD_LIVE
    else:
        ttl = TTL_SCOREBOARD_LIVE
    key = f"scoreboard|{lg.key}|{params.get('dates', 'current')}"
    return disk_cached(key, ttl, lambda: http_json(f"{_base(lg)}/scoreboard", params))


def _rank(c: dict) -> int | None:
    r = (c.get("curatedRank") or {}).get("current")
    try:
        r = int(r)
    except (TypeError, ValueError):
        return None
    return r if 1 <= r <= 25 else None


def _odds(comp: dict, home_abbr: str, away_abbr: str) -> dict:
    o = (comp.get("odds") or [{}])[0] or {}
    out = {"details": (o.get("details") or "").strip(), "total": None, "spread_home": None,
           "ml_home": None, "ml_away": None}
    t = num(o.get("overUnder"), default=float("nan"))
    out["total"] = None if t != t or t <= 0 else t
    d = out["details"]
    if d.upper() in ("EVEN", "PK", "PICK"):
        out["spread_home"] = 0.0
    else:
        m = re.match(r"([A-Za-z0-9&.]+)\s+([+-]?\d+(?:\.\d+)?)$", d)
        if m and abs(float(m.group(2))) < 60:  # spreads only; "-150" style details are moneylines
            fav, val = m.group(1).upper(), float(m.group(2))
            if fav == home_abbr.upper():
                out["spread_home"] = val
            elif fav == away_abbr.upper():
                out["spread_home"] = -val
    for side in ("home", "away"):
        ml = (o.get(f"{side}TeamOdds") or {}).get("moneyLine")
        if ml is None:
            ml = (((o.get("moneyline") or {}).get(side) or {}).get("close") or {}).get("odds")
        if ml is not None and abs(num(ml)) >= 100:
            out[f"ml_{side}"] = num(ml)
    return out


def parse_games(lg: League, raw: dict) -> list[dict]:
    games = []
    for ev in raw.get("events", []) or []:
        try:
            comp = (ev.get("competitions") or [{}])[0]
            sides = {c.get("homeAway"): c for c in comp.get("competitors", []) or []}
            if "home" not in sides or "away" not in sides:
                continue
            st = ((ev.get("status") or comp.get("status") or {}).get("type") or {})
            teams = {}
            for k in ("home", "away"):
                c = sides[k]
                t = c.get("team") or {}
                leaders = []
                for cat in c.get("leaders", []) or []:
                    for ld in (cat.get("leaders") or [])[:1]:
                        a = ld.get("athlete") or {}
                        leaders.append({"cat": (cat.get("name") or "").lower(), "id": str(a.get("id", "")),
                                        "name": a.get("displayName", ""), "value": num(ld.get("value"))})
                teams[k] = {
                    "id": str(t.get("id") or c.get("id") or ""),
                    "abbr": t.get("abbreviation") or t.get("shortDisplayName") or "?",
                    "name": t.get("displayName") or t.get("name") or "",
                    "short": t.get("shortDisplayName") or t.get("name") or t.get("abbreviation") or "",
                    "logo": t.get("logo") or ((t.get("logos") or [{}])[0] or {}).get("href", ""),
                    "record": ((c.get("records") or [{}])[0] or {}).get("summary", ""),
                    "rank": _rank(c),
                    "score": c.get("score"),
                    "leaders": leaders,
                }
            games.append({
                "id": str(ev.get("id")),
                "tip": to_et(ev.get("date")),
                "date": to_et_date(ev.get("date")),
                "state": st.get("state", ""),
                "status": st.get("shortDetail") or st.get("detail") or "",
                "neutral": bool(comp.get("neutralSite")),
                "home": teams["home"],
                "away": teams["away"],
                "odds": _odds(comp, teams["home"]["abbr"], teams["away"]["abbr"]),
                "week": (ev.get("week") or {}).get("number"),
            })
        except Exception:
            continue
    games.sort(key=lambda g: (g["tip"] is None, g["tip"] or 0))
    return games


# --------------------------------------------------------------------------
# Teams
# --------------------------------------------------------------------------
def team_schedule(lg: League, team_id: str) -> dict:
    return disk_cached(f"schedule|{lg.key}|{team_id}", TTL_SCHEDULE,
                       lambda: http_json(f"{_base(lg)}/teams/{team_id}/schedule"))


def team_statistics(lg: League, team_id: str) -> dict:
    return disk_cached(f"teamstats|{lg.key}|{team_id}", TTL_TEAM_STATS,
                       lambda: http_json(f"{_base(lg)}/teams/{team_id}/statistics"))


def team_roster(lg: League, team_id: str) -> dict:
    return disk_cached(f"roster|{lg.key}|{team_id}", TTL_ROSTER,
                       lambda: http_json(f"{_base(lg)}/teams/{team_id}/roster"))


def schedule_key(lg: League, team_id: str) -> str:
    return f"schedule|{lg.key}|{team_id}"


def parse_completed_games(raw: dict, team_id: str, before: date) -> list[dict]:
    """Completed regular/post-season games before `before`, oldest first."""
    out = []
    for ev in raw.get("events", []) or []:
        try:
            st_type = (ev.get("seasonType") or {}).get("type") or (ev.get("season") or {}).get("type")
            if str(st_type) == "1":  # preseason
                continue
            comp = (ev.get("competitions") or [{}])[0]
            st = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
            if not (st.get("completed") or st.get("state") == "post"):
                continue
            d = to_et_date(ev.get("date") or comp.get("date"))
            if d is None or d >= before:
                continue
            cs = comp.get("competitors", []) or []
            me = next((c for c in cs if str(c.get("id") or (c.get("team") or {}).get("id")) == str(team_id)), None)
            op = next((c for c in cs if c is not me), None)
            if me is None or op is None:
                continue
            pf, pa = num(me.get("score")), num(op.get("score"))
            if pf <= 0 and pa <= 0:
                continue
            out.append({"date": d, "pf": pf, "pa": pa, "home": me.get("homeAway") == "home",
                        "opp": str(op.get("id") or (op.get("team") or {}).get("id"))})
        except Exception:
            continue
    out.sort(key=lambda g: g["date"])
    return out


def roster_athletes(raw: dict) -> list[dict]:
    """Flatten basketball (flat) and football/hockey (grouped) roster payloads."""
    items = []
    for a in raw.get("athletes", []) or []:
        if isinstance(a, dict) and isinstance(a.get("items"), list):
            items.extend(a["items"])
        elif isinstance(a, dict):
            items.append(a)
    out = []
    for a in items:
        aid = str(a.get("id") or "")
        if not aid:
            continue
        inj = (a.get("injuries") or [{}])[0] or {}
        pos = a.get("position") or {}
        out.append({
            "id": aid,
            "name": a.get("displayName") or a.get("fullName") or "",
            "pos": (pos.get("abbreviation") or pos.get("name") or "") if isinstance(pos, dict) else str(pos),
            "headshot": (a.get("headshot") or {}).get("href", ""),
            "inj_status": inj.get("status") or (inj.get("type") or {}).get("description", "") if inj else "",
            "jersey": a.get("jersey", ""),
        })
    return out


# --------------------------------------------------------------------------
# Athlete game logs
# --------------------------------------------------------------------------
def gamelog_key(lg: League, athlete_id: str) -> str:
    return f"gamelog|{lg.key}|{athlete_id}"


def athlete_gamelog(lg: League, athlete_id: str) -> dict:
    return disk_cached(gamelog_key(lg, athlete_id), TTL_GAMELOG,
                       lambda: http_json(f"{WEB}/{lg.sport}/{lg.league}/athletes/{athlete_id}/gamelog"))


def gamelog_cached(lg: League, athlete_id: str) -> bool:
    return cache_fresh(gamelog_key(lg, athlete_id), TTL_GAMELOG)


STAT_COLS = {
    "basketball": {"MIN": ["minutes", "min"], "PTS": ["points", "pts"],
                   "REB": ["totalrebounds", "rebounds", "reb"], "AST": ["assists", "ast"]},
    "football": {"PASS": ["passingyards"], "RUSH": ["rushingyards"], "REC": ["receivingyards"],
                 "RECS": ["receptions"], "ATT": ["passingattempts"], "CAR": ["rushingattempts"]},
    "hockey": {"G": ["goals", "g"], "A": ["assists", "a"], "SOG": ["shotstotal", "shots", "shotsongoal", "sog"],
               "TOI": ["timeonice", "timeonicepergame", "toi"]},
}


def parse_gamelog(raw: dict, kind: str, before: date) -> list[dict]:
    names = [str(n).lower() for n in raw.get("names") or []]
    labels = [str(n).lower() for n in raw.get("labels") or []]
    cols = {}
    for col, cands in STAT_COLS[kind].items():
        idx = next((names.index(c) for c in cands if c in names), None)
        if idx is None and kind == "basketball":  # labels are unambiguous for basketball only
            idx = next((labels.index(c) for c in cands if c in labels), None)
        cols[col] = idx
    meta = raw.get("events") or {}
    rows: dict[str, dict] = {}
    for stype in raw.get("seasonTypes") or []:
        if "preseason" in str(stype.get("displayName", "")).lower():
            continue
        for cat in stype.get("categories") or []:
            if cat.get("type") not in (None, "event"):
                continue
            for ev in cat.get("events") or []:
                eid = str(ev.get("eventId", ""))
                m = meta.get(eid) or {}
                d = to_et_date(m.get("gameDate"))
                if not eid or d is None or d >= before:
                    continue
                stats = ev.get("stats") or []
                row = {"eid": eid, "date": d, "opp": (m.get("opponent") or {}).get("abbreviation", "")}
                for col, i in cols.items():
                    v = stats[i] if i is not None and i < len(stats) else None
                    row[col] = to_minutes(v) if col in ("MIN", "TOI") else (
                        made_att(v)[0] if isinstance(v, str) and "-" in v.lstrip("-") else num(v))
                rows[eid] = row
    return sorted(rows.values(), key=lambda r: r["date"], reverse=True)


# --------------------------------------------------------------------------
# Injuries
# --------------------------------------------------------------------------
def injuries(lg: League) -> list[dict]:
    if not lg.has_injuries:
        return []
    raw = disk_cached(f"injuries|{lg.key}", TTL_INJURIES, lambda: http_json(f"{_base(lg)}/injuries"))
    out = []
    for team in raw.get("injuries", []) or []:
        tname = team.get("displayName", "")
        tid = str(team.get("id", ""))
        for inj in team.get("injuries", []) or []:
            a = inj.get("athlete") or {}
            aid = str(a.get("id") or "")
            if not aid:
                for link in a.get("links", []) or []:
                    m = re.search(r"/id/(\d+)", link.get("href", ""))
                    if m:
                        aid = m.group(1)
                        break
            name = a.get("displayName") or a.get("fullName") or ""
            if not name:
                continue
            out.append({
                "team_name": tname, "team_id": tid,
                "team_abbr": (a.get("team") or {}).get("abbreviation", ""),
                "player": name, "key": norm_name(name), "athlete_id": aid,
                "status": inj.get("status") or (inj.get("type") or {}).get("description", ""),
                "detail": inj.get("shortComment") or inj.get("longComment") or "",
                "headshot": (a.get("headshot") or {}).get("href", ""),
                "pos": ((a.get("position") or {}).get("abbreviation", "")),
            })
    return out


def injury_class(status: str) -> str:
    s = (status or "").lower()
    if "probable" in s:
        return "ACTIVE"
    if s in ("ir", "pup", "nfi") or any(k in s for k in ("out", "doubtful", "suspen", "inactive", "injured reserve")):
        return "OUT"
    if any(k in s for k in ("questionable", "day-to-day", "day to day", "gtd", "game time")):
        return "Q"
    return "ACTIVE"


def team_injuries(all_inj: list[dict], team: dict) -> list[dict]:
    """Injuries for a scoreboard team: match by team id, display name, or abbreviation."""
    tid, name, abbr = team["id"], team["name"].lower(), team["abbr"].upper()
    return [i for i in all_inj
            if (i["team_id"] and i["team_id"] == tid) or i["team_name"].lower() == name
            or (i["team_abbr"] and i["team_abbr"].upper() == abbr)]


def cached_gamelog(lg: League, athlete_id: str):
    return cache_read(gamelog_key(lg, athlete_id))
