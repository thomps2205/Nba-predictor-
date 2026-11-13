"""Latest player news from ESPN's public news feeds (league-wide and per-team)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from .common import ET, disk_cached, http_json, norm_name
from .espn import SITE, League

TTL_NEWS = 15 * 60


def _fetch(lg: League, team_id: str | None) -> dict:
    params = {"limit": "50"}
    if team_id:
        params["team"] = team_id
    return http_json(f"{SITE}/{lg.sport}/{lg.league}/news", params, retries=1)


def articles(lg: League, team_ids: tuple[str, ...] = ()) -> list[dict]:
    """League feed plus each team's feed, de-duplicated, newest first. Failures just shrink the list."""
    raw_feeds = []
    for tid in (None, *team_ids):
        try:
            raw_feeds.append(disk_cached(f"news|{lg.key}|{tid or 'all'}", TTL_NEWS, lambda tid=tid: _fetch(lg, tid)))
        except Exception:
            continue
    seen, out = set(), []
    for raw in raw_feeds:
        for a in raw.get("articles", []) or []:
            key = a.get("id") or a.get("headline")
            if not key or key in seen:
                continue
            seen.add(key)
            ids, names = set(), set()
            for c in a.get("categories", []) or []:
                if c.get("type") == "athlete":
                    aid = c.get("athleteId") or (c.get("athlete") or {}).get("id")
                    if aid:
                        ids.add(str(aid))
                    nm = c.get("description") or (c.get("athlete") or {}).get("description")
                    if nm:
                        names.add(norm_name(nm))
            ts = pd.to_datetime(a.get("published") or a.get("lastModified"), utc=True, errors="coerce")
            out.append({
                "headline": a.get("headline") or a.get("title") or "",
                "description": a.get("description") or "",
                "published": ts.tz_convert(ET).to_pydatetime() if not pd.isna(ts) else None,
                "link": (((a.get("links") or {}).get("web") or {}).get("href")) or "",
                "athlete_ids": ids, "athlete_names": names,
            })
    out.sort(key=lambda x: x["published"] or datetime.min.replace(tzinfo=ET), reverse=True)
    return out


def for_player(arts: list[dict], athlete_id: str, name: str, days: int = 5) -> dict | None:
    """Most recent article tagged with this athlete (by id, else by name) within `days`."""
    cutoff = datetime.now(ET) - timedelta(days=days)
    key = norm_name(name)
    for a in arts:
        if a["published"] and a["published"] < cutoff:
            continue
        if (athlete_id and athlete_id in a["athlete_ids"]) or (key and key in a["athlete_names"]):
            return a
    return None


def age(dt: datetime | None) -> str:
    if not dt:
        return ""
    mins = int((datetime.now(ET) - dt).total_seconds() // 60)
    if mins < 60:
        return f"{max(mins, 1)}m ago"
    if mins < 48 * 60:
        return f"{mins // 60}h ago"
    return f"{mins // 1440}d ago"
