"""Grade player props on finished games: final box score (ESPN game summary) vs the pregame pick."""

from __future__ import annotations

from .common import cache_fresh, cache_read, cache_write, made_att, num, to_minutes
from .espn import SITE, League, http_json

TTL_FINAL_BOX = 30 * 24 * 3600  # a final box score never changes

KEY_ALIASES = {
    "basketball": {"PTS": ["points", "pts"], "REB": ["rebounds", "totalrebounds", "reb"], "AST": ["assists", "ast"]},
    "football": {"PASS": ["passingyards"], "RUSH": ["rushingyards"], "REC": ["receivingyards"],
                 "RECS": ["receptions"]},
    "hockey": {"G": ["goals", "g"], "A": ["assists", "a"], "SOG": ["shotstotal", "shots", "shotsongoal", "s", "sog"]},
}
# football labels repeat ("YDS"), so only basketball / hockey fall back to labels
LABEL_OK = {"basketball", "hockey"}


def _fetch_box(lg: League, event_id: str) -> dict:
    raw = http_json(f"{SITE}/{lg.sport}/{lg.league}/summary", {"event": event_id}, retries=1)
    out: dict[str, dict] = {}
    st = (((raw.get("header") or {}).get("competitions") or [{}])[0].get("status") or {}).get("type") or {}
    out["_final"] = bool(st.get("completed")) or st.get("state") == "post"
    for team in (raw.get("boxscore") or {}).get("players", []) or []:
        abbr = (team.get("team") or {}).get("abbreviation", "")
        for blk in team.get("statistics", []) or []:
            keys = [str(k).lower() for k in (blk.get("keys") or blk.get("names") or [])]
            labels = [str(k).lower() for k in (blk.get("labels") or blk.get("names") or [])]
            for ath in blk.get("athletes", []) or []:
                aid = str((ath.get("athlete") or {}).get("id", ""))
                stats = ath.get("stats") or []
                if not aid or not stats:
                    continue
                row = out.setdefault(aid, {"_dnp": bool(ath.get("didNotPlay")),
                                           "_name": (ath.get("athlete") or {}).get("displayName", ""),
                                           "_team": abbr})
                for i, v in enumerate(stats):
                    for name in {keys[i] if i < len(keys) else None, labels[i] if i < len(labels) else None}:
                        if name and name not in row:
                            if name in ("minutes", "min", "timeonice", "toi"):
                                row[name] = to_minutes(v)
                            elif isinstance(v, str) and "-" in v.lstrip("-") and "/" not in v:
                                row[name] = made_att(v)[0]
                            else:
                                row[name] = num(v, default=float("nan"))
                row["_labels_ok"] = True
    return out


TTL_LIVE_BOX = 120  # in-progress games: refresh every 2 minutes


def box_stats(lg: League, event_id: str) -> dict[str, dict]:
    """{athlete_id: {stat key: value}} from the box score, plus "_final" (game over). A final box score is
    cached for 30 days; a live one for 2 minutes. {} if unavailable."""
    key = f"box2|{lg.key}|{event_id}"
    cached = cache_read(key)
    if cached is not None and (cached.get("_final") and cache_fresh(key, TTL_FINAL_BOX)
                               or cache_fresh(key, TTL_LIVE_BOX)):
        return cached
    try:
        out = _fetch_box(lg, event_id)
    except Exception:
        return cached or {}
    cache_write(key, out)
    return out


def played(row: dict | None) -> bool:
    """Box-score row for someone who actually got on the floor / field / ice tonight."""
    if not row or row.get("_dnp"):
        return False
    mins = next((row[k] for k in ("minutes", "min", "timeonice", "toi") if k in row), None)
    return mins is None or (mins == mins and mins > 0)


def actual_value(kind: str, m: str, row: dict | None) -> float | None:
    if not row or row.get("_dnp"):
        return None
    if kind == "basketball" and m == "PRA":
        parts = [actual_value(kind, x, row) for x in ("PTS", "REB", "AST")]
        return None if any(p is None for p in parts) else sum(parts)
    if kind == "hockey" and m == "PTS":
        g, a = actual_value(kind, "G", row), actual_value(kind, "A", row)
        return None if g is None or a is None else g + a
    for k in KEY_ALIASES.get(kind, {}).get(m, []):
        v = row.get(k)
        if v is not None and v == v:
            return float(v)
    return 0.0 if kind == "football" and row else None  # football: no row in that category = 0 yards


def result(side: str, line: float, actual: float | None) -> str | None:
    if actual is None:
        return None
    if actual == line:
        return "PUSH"
    won = actual > line if side == "Over" else actual < line
    return "WON" if won else "LOST"
