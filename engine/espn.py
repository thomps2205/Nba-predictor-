"""ESPN public JSON endpoints (site.api / site.web.api) and tolerant parsers for team sports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from .common import (LEAGUE_LOGOS, cache_fresh, cache_read, disk_cached, http_json, made_att, norm_name, num, to_et,
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

# Soccer: global roster grouped by region. (key, ESPN slug, display name, type: league / cup / intl)
SOCCER_REGIONS = [
    ("UEFA Club Competitions", [
        ("UCL", "uefa.champions", "Champions League", "cup"), ("UEL", "uefa.europa", "Europa League", "cup"),
        ("UECL", "uefa.europa.conf", "Conference League", "cup")]),
    ("England", [
        ("EPL", "eng.1", "Premier League", "league"), ("CHAMP", "eng.2", "EFL Championship", "league"),
        ("FACUP", "eng.fa", "FA Cup", "cup"), ("EFLCUP", "eng.league_cup", "Carabao Cup", "cup")]),
    ("Europe — Leagues & Cups", [
        ("LALIGA", "esp.1", "LaLiga", "league"), ("COPAREY", "esp.copa_del_rey", "Copa del Rey", "cup"),
        ("BUNDES", "ger.1", "Bundesliga", "league"), ("BUNDES2", "ger.2", "2. Bundesliga", "league"),
        ("DFBPOKAL", "ger.dfb_pokal", "DFB-Pokal", "cup"), ("SERIEA", "ita.1", "Serie A", "league"),
        ("COPPAIT", "ita.coppa_italia", "Coppa Italia", "cup"), ("LIGUE1", "fra.1", "Ligue 1", "league"),
        ("EREDIV", "ned.1", "Eredivisie", "league"), ("PRIMEIRA", "por.1", "Primeira Liga", "league"),
        ("SPFL", "sco.1", "Scottish Premiership", "league"), ("SUPERLIG", "tur.1", "Süper Lig", "league")]),
    ("Americas & Global", [
        ("MLS", "usa.1", "MLS", "league"), ("USL", "usa.usl.1", "USL Championship", "league"),
        ("NWSL", "usa.nwsl", "NWSL", "league"), ("LIGAMX", "mex.1", "Liga MX", "league"),
        ("BRASIL", "bra.1", "Brasileirão Série A", "league"), ("ARG", "arg.1", "Liga Profesional", "league"),
        ("CCC", "concacaf.champions", "CONCACAF Champions Cup", "cup"),
        ("LIBERT", "conmebol.libertadores", "Copa Libertadores", "cup"),
        ("SAUDI", "ksa.1", "Saudi Pro League", "league"), ("J1", "jpn.1", "J1 League", "league")]),
    ("International", [
        ("WC", "fifa.world", "FIFA World Cup", "intl"), ("WCQ_UEFA", "fifa.worldq.uefa", "WC Qualifying — UEFA", "intl"),
        ("WCQ_CONMEBOL", "fifa.worldq.conmebol", "WC Qualifying — CONMEBOL", "intl"),
        ("WCQ_CONCACAF", "fifa.worldq.concacaf", "WC Qualifying — CONCACAF", "intl"),
        ("NATIONS", "uefa.nations", "UEFA Nations League", "intl"),
        ("FRIENDLY", "fifa.friendly", "International Friendlies", "intl")]),
]
SOCCER_LEAGUES = [(k, slug, name) for _, items in SOCCER_REGIONS for k, slug, name, _ in items]
SOCCER_TYPE = {k: t for _, items in SOCCER_REGIONS for k, _, _, t in items}
# "All Featured Leagues": the top active competitions queried together
SOCCER_FEATURED = ["UCL", "UEL", "UECL", "EPL", "CHAMP", "FACUP", "EFLCUP", "LALIGA", "BUNDES", "SERIEA", "LIGUE1",
                   "MLS", "LIGAMX", "WCQ_UEFA", "WCQ_CONMEBOL", "WCQ_CONCACAF", "NATIONS", "FRIENDLY"]
SOCCER_DEFAULT = ["EPL", "LALIGA", "BUNDES", "SERIEA", "LIGUE1", "UCL", "MLS"]  # MIX uses these
for _k, _slug, _name in SOCCER_LEAGUES:
    LEAGUES[_k] = League(_k, "soccer", _slug, "soccer", _name, False)

# Where to borrow recent form when a cup / international slug has < 3 completed matches for a team.
_EURO_TOP = ["eng.1", "esp.1", "ger.1", "ita.1", "fra.1", "por.1", "ned.1", "sco.1", "tur.1", "eng.2", "ger.2"]
_INTL = ["fifa.friendly", "fifa.worldq.uefa", "fifa.worldq.conmebol", "fifa.worldq.concacaf", "uefa.nations",
         "fifa.world"]
SOCCER_FORM_FALLBACK = {
    "FACUP": ["eng.1", "eng.2"], "EFLCUP": ["eng.1", "eng.2"], "CHAMP": ["eng.1"], "COPAREY": ["esp.1"],
    "DFBPOKAL": ["ger.1", "ger.2"], "COPPAIT": ["ita.1"], "UCL": _EURO_TOP, "UEL": _EURO_TOP, "UECL": _EURO_TOP,
    "CCC": ["usa.1", "mex.1"], "LIBERT": ["bra.1", "arg.1"],
    **{k: _INTL for _, items in SOCCER_REGIONS for k, _, _, t in items if t == "intl"},
}
TTL_SOCCER_SCOREBOARD = 10 * 60  # many competitions per view: refresh every 10 min (live scores still update)


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
    if d is not None:
        end = d + timedelta(days=days - 1)
        params["dates"] = d.strftime("%Y%m%d") if days == 1 else f"{d:%Y%m%d}-{end:%Y%m%d}"
        ttl = TTL_SCOREBOARD_PAST if end < today else TTL_SCOREBOARD_LIVE
    else:
        ttl = TTL_SCOREBOARD_LIVE
    if lg.kind == "soccer" and ttl == TTL_SCOREBOARD_LIVE:
        ttl = TTL_SOCCER_SCOREBOARD
    key = f"scoreboard|{lg.key}|{params.get('dates', 'current')}"
    raw = disk_cached(key, ttl, lambda: http_json(f"{_base(lg)}/scoreboard", params))
    try:
        logos = ((raw.get("leagues") or [{}])[0] or {}).get("logos") or []
        href = next((l.get("href") for l in logos if "dark" not in (l.get("rel") or [])), None) or \
            (logos[0].get("href") if logos else None)  # standard logo: it sits on a light tile
        if href:
            LEAGUE_LOGOS[lg.key] = href
    except Exception:
        pass
    return raw


def week_games(lg: League, start: date, days: int, today: date) -> list[dict]:
    """All games in [start, start+days). ESPN rejects some date-range queries (NFL), so fall back to the
    default current-week scoreboard, then to one single-day request per day. Never raises."""
    end = start + timedelta(days=days)
    in_window = lambda g: g["date"] is not None and start <= g["date"] < end  # noqa: E731
    try:
        games = [g for g in parse_games(lg, scoreboard(lg, start, today, days)) if in_window(g)]
        if games:
            return games
    except Exception:
        pass
    if start <= today < end:
        try:
            games = [g for g in parse_games(lg, scoreboard(lg, None, today)) if in_window(g)]
            if games:
                return games
        except Exception:
            pass
    merged: dict[str, dict] = {}
    for i in range(days):
        try:
            for g in parse_games(lg, scoreboard(lg, start + timedelta(days=i), today)):
                merged[g["id"]] = g
        except Exception:
            continue
    return sorted(merged.values(), key=lambda g: (g["tip"] is None, g["tip"] or 0))


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
           "ml_home": None, "ml_away": None, "ml_draw": None}
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
    ml = (o.get("drawOdds") or {}).get("moneyLine")  # soccer three-way market
    if ml is None:
        ml = (((o.get("moneyline") or {}).get("draw") or {}).get("close") or {}).get("odds")
    if ml is not None and abs(num(ml)) >= 100:
        out["ml_draw"] = num(ml)
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


def _team_stat_map(raw: dict) -> dict[str, float]:
    """results.stats.categories[].stats[] -> {name/abbreviation (lower-case): value}. First occurrence wins,
    scanning general -> offensive -> defensive -> anything else (defensive FGA etc. must not shadow ours)."""
    cats = (((raw.get("results") or {}).get("stats") or {}).get("categories")) or \
           ((raw.get("statistics") or {}).get("splits") or {}).get("categories") or []
    order = {"general": 0, "offensive": 1, "defensive": 2}
    out: dict[str, float] = {}
    for cat in sorted(cats, key=lambda c: order.get(str(c.get("name", "")).lower(), 3)):
        for s in cat.get("stats", []) or []:
            v = s.get("value")
            v = num(v if v is not None else s.get("displayValue"), default=float("nan"))
            if v != v:
                continue
            for k in (s.get("name"), s.get("abbreviation")):
                if isinstance(k, str) and k.lower() not in out:
                    out[k.lower()] = v
    return out


def team_possessions(raw: dict, games_from_schedule: int = 0) -> float | None:
    """Possessions per game (Dean Oliver): FGA + 0.44*FTA - OREB + TOV, from per-game averages when ESPN
    provides them, otherwise season totals divided by games played."""
    s = _team_stat_map(raw)

    def get(*keys):
        for k in keys:
            if k in s:
                return s[k]
        return None

    avg = [get("avgfieldgoalsattempted"), get("avgfreethrowsattempted"), get("avgoffensiverebounds"),
           get("avgturnovers", "avgtotalturnovers")]
    if all(v is not None and v > 0 for v in avg):
        p = avg[0] + 0.44 * avg[1] - avg[2] + avg[3]
        if 50 < p < 120:
            return p
    tot = [get("fieldgoalsattempted", "fga"), get("freethrowsattempted", "fta"),
           get("offensiverebounds", "oreb"), get("turnovers", "totalturnovers", "to", "tov")]
    if not all(v is not None and v >= 0 for v in tot) or not tot[0]:
        return None
    raw_poss = tot[0] + 0.44 * tot[1] - tot[2] + tot[3]
    gp = get("gamesplayed", "gp") or games_from_schedule
    if gp and 50 < raw_poss / gp < 120:
        return raw_poss / gp
    if 50 < raw_poss < 120:  # values were already per game
        return raw_poss
    return None


def soccer_schedule(slug: str, team_id: str) -> dict:
    """A club / national team's fixtures under any soccer slug (incl. 'all' for every competition)."""
    return disk_cached(f"schedule|SOC:{slug}|{team_id}", TTL_SCHEDULE,
                       lambda: http_json(f"{SITE}/soccer/{slug}/teams/{team_id}/schedule", retries=1))


def schedule_key(lg: League, team_id: str) -> str:
    return f"schedule|{lg.key}|{team_id}"


def _linescores(c: dict) -> list[float]:
    """Points / goals per period from competitors[].linescores ([{"value": 28.0}, ...])."""
    out = []
    for x in c.get("linescores") or []:
        if isinstance(x, dict):
            out.append(num(x.get("value", x.get("displayValue"))))
        else:
            out.append(num(x))
    return out


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
                        "opp": str(op.get("id") or (op.get("team") or {}).get("id")),
                        "ls_for": _linescores(me), "ls_against": _linescores(op)})
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


def athlete_stats_key(lg: League, athlete_id: str) -> str:
    return f"athstats|{lg.key}|{athlete_id}"


def athlete_stats(lg: League, athlete_id: str) -> dict:
    """Season stats (v3). Used when an NHL game log comes back empty."""
    return disk_cached(athlete_stats_key(lg, athlete_id), TTL_ROSTER,
                       lambda: http_json(f"{WEB}/{lg.sport}/{lg.league}/athletes/{athlete_id}/stats"))


def hockey_season_rates(raw: dict) -> dict | None:
    """Per-game goals / assists / shots for the latest season in a v3 athlete stats payload:
    categories[] with names[] and statistics[] rows ({season: {year}, stats: [...]})."""
    vals: dict[str, float] = {}
    for cat in raw.get("categories") or []:
        names = [str(n).lower() for n in cat.get("names") or []]
        rows = [r for r in cat.get("statistics") or [] if isinstance(r, dict) and r.get("stats")]
        if not names or not rows:
            continue
        row = max(rows, key=lambda r: num(((r.get("season") or {}).get("year")), 0))
        for n, v in zip(names, row["stats"]):
            vals.setdefault(n, num(v, default=float("nan")))
    if not vals:
        from .common import flatten_stats
        vals = flatten_stats(raw)

    def get(*keys):
        for k in keys:
            v = vals.get(k)
            if v is not None and v == v:
                return v
        return None

    gp = get("gamesplayed", "games", "gp")
    if not gp or gp <= 0:
        return None
    g, a, s = get("goals", "g"), get("assists", "a"), get("shotstotal", "shots", "shotsongoal", "sog")
    if g is None and a is None and s is None:
        return None
    return {"gp": gp, "G": (g or 0) / gp, "A": (a or 0) / gp, "SOG": (s or 0) / gp}


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
                row = {"eid": eid, "date": d, "opp": (m.get("opponent") or {}).get("abbreviation", ""),
                       "away": str(m.get("atVs", "")).strip() == "@"}
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


def cache_read_json(key: str):
    return cache_read(key)


def needs_hockey_fallback(lg: League, athlete_id: str, before: date) -> bool:
    raw = cached_gamelog(lg, athlete_id)
    try:
        return len(parse_gamelog(raw, "hockey", before)) < 3 if raw else True
    except Exception:
        return True


def cached_gamelog(lg: League, athlete_id: str):
    return cache_read(gamelog_key(lg, athlete_id))
