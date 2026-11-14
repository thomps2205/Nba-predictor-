"""UFC fight cards (ESPN mma/ufc scoreboard), fighter profiles, win-probability and method-of-victory model."""

from __future__ import annotations

import math
import re
from datetime import date, timedelta

from .common import (cache_fresh, cache_read, cache_write, disk_cached, flatten_stats, http_json, inches, ml_to_prob, num, pick, to_et,
                     to_et_date)
from .espn import LEAGUES, WEB, scoreboard

CORE = "https://sports.core.api.espn.com/v2/sports/mma"
SEGMENT_RANK = {"main card": 0, "prelims": 1, "early prelims": 2}

TTL_FIGHTER = 24 * 3600
UFC = LEAGUES["UFC"]

# Finish-method base rates by weight class (UFC historical shares: KO/TKO, Submission, Decision)
METHOD_PRIOR = {
    "heavyweight": (0.48, 0.17, 0.35), "light heavyweight": (0.42, 0.18, 0.40), "middleweight": (0.36, 0.20, 0.44),
    "welterweight": (0.30, 0.20, 0.50), "lightweight": (0.26, 0.22, 0.52), "featherweight": (0.25, 0.18, 0.57),
    "bantamweight": (0.23, 0.15, 0.62), "flyweight": (0.18, 0.18, 0.64), "strawweight": (0.12, 0.20, 0.68),
    "women": (0.15, 0.20, 0.65), "default": (0.28, 0.19, 0.53),
}


def _prior(weight_class: str) -> tuple[float, float, float]:
    wc = (weight_class or "").lower()
    if "women" in wc:
        base = METHOD_PRIOR["women"]
        if "bantam" in wc or "feather" in wc:
            return (0.20, 0.18, 0.62)
        return base
    for k in ("light heavyweight", "heavyweight", "middleweight", "welterweight", "lightweight", "featherweight",
              "bantamweight", "flyweight", "strawweight"):
        if k in wc:
            return METHOD_PRIOR[k]
    return METHOD_PRIOR["default"]


def _record(s: str) -> tuple[int, int, int]:
    parts = [int(num(x)) for x in str(s or "").replace(" ", "").split("-")[:3]]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


# --------------------------------------------------------------------------
# Fight card
# --------------------------------------------------------------------------
def fight_cards(today: date) -> list[dict]:
    """Upcoming / live UFC events (falls back to the next 60 days, then the last 14 days)."""
    events: dict[str, dict] = {}
    for d, days in ((None, 1), (today, 60), (today - timedelta(days=14), 15)):
        try:
            raw = scoreboard(UFC, d, today, days)
        except Exception:
            continue
        for ev in raw.get("events", []) or []:
            card = _parse_event(ev)
            if card and card["bouts"]:
                events[card["id"]] = card
        upcoming = [e for e in events.values() if e["state"] != "post"]
        if upcoming:
            break
    cards = sorted(events.values(), key=lambda e: (e["state"] == "post", e["date"] or date.max))
    return cards


def _parse_event(ev: dict) -> dict | None:
    comps = ev.get("competitions") or []
    bouts = []
    for idx, c in enumerate(comps):
        try:
            cs = c.get("competitors") or []
            if len(cs) != 2:
                continue
            fighters = []
            for f in sorted(cs, key=lambda x: num(x.get("order"), 9)):
                a = f.get("athlete") or {}
                fighters.append({
                    "id": str(a.get("id") or f.get("id") or ""),
                    "name": a.get("fullName") or a.get("displayName") or "TBA",
                    "short": a.get("shortName") or a.get("displayName") or "",
                    "record": ((f.get("records") or a.get("records") or [{}])[0] or {}).get("summary", ""),
                    "headshot": (a.get("headshot") or {}).get("href", "") if isinstance(a.get("headshot"), dict) else "",
                    "flag": (a.get("flag") or {}).get("href", ""),
                    "country": (a.get("flag") or {}).get("alt", ""),
                    "winner": f.get("winner"),
                    "ml": _fighter_ml(c, f, len(fighters)),
                })
            st = ((c.get("status") or {}).get("type") or {})
            seg_name, wc = _segment_and_weight(c)
            rounds = ((c.get("format") or {}).get("regulation") or {}).get("periods")
            result = (c.get("status") or {}).get("result") or {}
            bouts.append({
                "id": str(c.get("id") or idx), "idx": idx, "start": to_et(c.get("date") or c.get("startDate")),
                "segment": seg_name, "weight_class": wc, "rounds": int(num(rounds)) if rounds else None,
                "state": st.get("state", ""), "status": st.get("shortDetail") or st.get("detail") or "",
                "result": (result.get("displayName") or result.get("name") or st.get("detail") or "")
                if st.get("state") == "post" else "",
                "fighters": fighters,
            })
        except Exception:
            continue
    if not bouts:
        return None
    # Card order: segment (Main Card > Prelims > Early Prelims), then competition id descending — ESPN gives
    # the main event / co-main the highest ids. Start times alone are unreliable (shared block timestamps).
    def cid(b):
        return int(b["id"]) if str(b["id"]).isdigit() else -b["idx"]

    bouts.sort(key=lambda b: (SEGMENT_RANK.get(b["segment"].lower(), 0 if not b["segment"] else 3), -cid(b)))
    if not any(b["segment"] for b in bouts):
        for i, b in enumerate(bouts):
            b["segment"] = "Main Card" if i < 5 else "Prelims"
    # Main event: highest-id 5-round bout on the main card if there is one, else the top bout.
    main = next((b for b in bouts if b["rounds"] == 5 and SEGMENT_RANK.get(b["segment"].lower(), 0) == 0), bouts[0])
    bouts.remove(main)
    bouts.insert(0, main)
    for b in bouts:
        if b["rounds"] is None:
            b["rounds"] = 5 if b is main else 3
    main["main_event"] = True
    st = ((ev.get("status") or {}).get("type") or {})
    states = {b["state"] for b in bouts}
    venue = (((comps[0] if comps else {}).get("venue") or {}).get("fullName", "")) if comps else ""
    return {"id": str(ev.get("id")), "name": ev.get("name") or ev.get("shortName") or "UFC",
            "date": to_et_date(ev.get("date")), "start": to_et(ev.get("date")), "venue": venue,
            "state": "in" if "in" in states else (st.get("state") or ("post" if states == {"post"} else "pre")),
            "bouts": bouts}


_SEG_RE = ("early prelim", "prelim", "main card")
_WEIGHT_WORDS = ("weight", "women", "catch")


def _segment_and_weight(c: dict) -> tuple[str, str]:
    """Card segment and weight class from cardSegment / type.abbreviation / type.text / notes[].headline."""
    texts = []
    seg = c.get("cardSegment") or {}
    texts += [seg.get("description"), seg.get("name")]
    typ = c.get("type") or {}
    texts += [typ.get("abbreviation"), typ.get("text"), typ.get("name")]
    for n in c.get("notes") or []:
        if isinstance(n, dict):
            texts.append(n.get("headline") or n.get("text"))
    texts.append(c.get("note"))
    segment = weight = ""
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            continue
        tl = t.lower()
        if not segment:
            if "early prelim" in tl:
                segment = "Early Prelims"
            elif "prelim" in tl:
                segment = "Prelims"
            elif "main card" in tl or tl == "main":
                segment = "Main Card"
        if not weight and any(w in tl for w in _WEIGHT_WORDS):
            weight = t.split(" - ")[0].strip()
    return segment, weight


def _fighter_ml(comp: dict, f: dict, order: int):
    for src in (f.get("odds"), (f.get("athlete") or {}).get("odds")):
        if isinstance(src, dict) and src.get("moneyLine") is not None:
            return num(src["moneyLine"])
    o = (comp.get("odds") or [{}])[0] or {}
    for key in (("homeAthleteOdds", "awayAthleteOdds"), ("homeTeamOdds", "awayTeamOdds")):
        side = o.get(key[0] if f.get("homeAway") == "home" or (not f.get("homeAway") and order == 0) else key[1])
        if isinstance(side, dict) and side.get("moneyLine") is not None:
            return num(side["moneyLine"])
    return None


TTL_ML_FALLBACK = 30 * 60


def _core_bout_odds(event_id: str, comp_id: str) -> list[dict]:
    raw = http_json(f"{CORE}/leagues/ufc/events/{event_id}/competitions/{comp_id}/odds", retries=1)
    items = []
    for it in (raw.get("items") or [])[:4]:
        if isinstance(it, dict) and "$ref" in it and len(it) == 1:
            try:
                it = http_json(it["$ref"], retries=1)
            except Exception:
                continue
        items.append(it)
    return items


def _ml_from_core(items: list[dict], fighters: list[dict]) -> tuple | None:
    for it in items:
        sides = {}
        for side in ("homeAthleteOdds", "awayAthleteOdds", "homeTeamOdds", "awayTeamOdds"):
            o = it.get(side)
            if not isinstance(o, dict):
                continue
            ml = o.get("moneyLine")
            if ml is None:
                ml = ((o.get("current") or {}).get("moneyLine") or {}).get("american")
            if ml in (None, "", "OFF", "EVEN"):
                ml = 100 if ml == "EVEN" else None
            if ml is None:
                continue
            ref = ((o.get("athlete") or o.get("team") or {}).get("$ref") or "")
            aid = ref.rstrip("/").split("/")[-1].split("?")[0] if ref else ""
            sides["home" if side.startswith("home") else "away"] = (aid, num(ml))
        if len(sides) < 2:
            continue
        ids = [f["id"] for f in fighters]
        by_id = {aid: ml for aid, ml in sides.values() if aid}
        if all(i in by_id for i in ids):
            return by_id[ids[0]], by_id[ids[1]]
        return sides["home"][1], sides["away"][1]  # same home = first-listed convention as _fighter_ml
    return None


def ensure_moneylines(card: dict) -> None:
    """Every bout gets actionable odds. Order: ESPN scoreboard (saved per bout as the closing line) ->
    saved pregame line -> ESPN core competition odds -> model fair odds (flagged 'model fair', applied later)."""
    for b in card["bouts"]:
        fa, fb = b["fighters"]
        key = f"ufcml|{card['id']}|{b['id']}"
        if fa.get("ml") is not None and fb.get("ml") is not None:
            b["ml_src"] = "ESPN"
            if b["state"] == "pre":
                cache_write(key, [fa["ml"], fb["ml"]])
            continue
        saved = cache_read(key)
        if saved and len(saved) == 2 and None not in saved:
            fa["ml"], fb["ml"] = saved
            b["ml_src"] = "saved closing line" if b["state"] != "pre" else "saved line"
            continue
        try:
            items = disk_cached(f"ufcmlcore|{card['id']}|{b['id']}", TTL_ML_FALLBACK,
                                lambda: _core_bout_odds(card["id"], b["id"]))
            got = _ml_from_core(items, b["fighters"])
        except Exception:
            got = None
        if got:
            fa["ml"], fb["ml"] = got
            b["ml_src"] = "ESPN core odds"
            cache_write(key, list(got))
        else:
            b["ml_src"] = "model fair"


# --------------------------------------------------------------------------
# Fighter profiles
# --------------------------------------------------------------------------
SITE_MMA = "https://site.api.espn.com/apis/site/v2/sports/mma"

# Approximate UFC divisional career averages, used only when ESPN has no stats for a fighter (debuts / nulls).
DIV_AVG = {
    "heavyweight": dict(slpm=3.6, sapm=3.5, str_acc=48, str_def=52, td_avg=0.9, td_acc=35, td_def=62, sub_avg=0.3),
    "light heavyweight": dict(slpm=3.8, sapm=3.7, str_acc=47, str_def=53, td_avg=1.1, td_acc=36, td_def=62, sub_avg=0.4),
    "middleweight": dict(slpm=4.0, sapm=3.8, str_acc=47, str_def=55, td_avg=1.3, td_acc=37, td_def=64, sub_avg=0.5),
    "welterweight": dict(slpm=4.0, sapm=3.7, str_acc=46, str_def=56, td_avg=1.6, td_acc=38, td_def=64, sub_avg=0.5),
    "lightweight": dict(slpm=4.2, sapm=3.8, str_acc=45, str_def=56, td_avg=1.7, td_acc=37, td_def=63, sub_avg=0.6),
    "featherweight": dict(slpm=4.4, sapm=4.0, str_acc=44, str_def=56, td_avg=1.5, td_acc=36, td_def=62, sub_avg=0.6),
    "bantamweight": dict(slpm=4.6, sapm=4.1, str_acc=44, str_def=57, td_avg=1.6, td_acc=37, td_def=61, sub_avg=0.5),
    "flyweight": dict(slpm=4.3, sapm=3.8, str_acc=45, str_def=57, td_avg=1.7, td_acc=36, td_def=61, sub_avg=0.5),
    "women": dict(slpm=4.5, sapm=4.0, str_acc=45, str_def=56, td_avg=1.5, td_acc=38, td_def=60, sub_avg=0.4),
    "default": dict(slpm=4.1, sapm=3.8, str_acc=46, str_def=55, td_avg=1.4, td_acc=37, td_def=62, sub_avg=0.5),
}


def div_avg(weight_class: str) -> dict:
    wc = (weight_class or "").lower()
    if "women" in wc:
        return DIV_AVG["women"]
    for k in ("light heavyweight", "heavyweight", "middleweight", "welterweight", "lightweight", "featherweight",
              "bantamweight", "flyweight"):
        if k in wc:
            return DIV_AVG[k]
    return DIV_AVG["default"]


def fighter_keys(fid: str) -> list[str]:
    return [f"fighter|{fid}", f"fighterstats|{fid}", f"fighterhist2|{fid}", f"fighterrec|{fid}"]


def fighter_cached(fid: str) -> bool:
    return cache_fresh(f"fighter|{fid}", TTL_FIGHTER)


def _fetch_athlete(fid: str) -> dict:
    last = None
    for url in (f"{CORE}/athletes/{fid}", f"{WEB}/mma/athletes/{fid}", f"{WEB}/mma/ufc/athletes/{fid}"):
        try:
            return http_json(url, retries=1)
        except Exception as e:
            last = e
    raise RuntimeError(f"fighter {fid} unavailable") from last


def _fetch_stats(fid: str) -> dict:
    for url in (f"{CORE}/athletes/{fid}/statistics", f"{WEB}/mma/athletes/{fid}/stats"):
        try:
            d = http_json(url, retries=1)
            if d:
                return d
        except Exception:
            continue
    return {}


TTL_PAST_EVENT = 30 * 24 * 3600   # finished events never change
EVENTLOG_BOUTS = 5                 # past bouts resolved per fighter (career totals come from /records)


def _https(u: str) -> str:
    return u.replace("http://", "https://", 1) if u.startswith("http://") else u


def _ref(obj) -> str:
    return _https(obj.get("$ref", "")) if isinstance(obj, dict) else (_https(obj) if isinstance(obj, str) else "")


def _ids_from(ref: str) -> tuple[str, str]:
    e = re.search(r"/events/(\d+)", ref or "")
    c = re.search(r"/competitions/(\d+)", ref or "")
    return (e.group(1) if e else ""), (c.group(1) if c else "")


def _find_comp(obj, fid: str, cid: str = ""):
    """The competition dict (anywhere in an event payload) whose competitors include fighter fid."""
    if isinstance(obj, dict):
        cs = obj.get("competitors")
        if isinstance(cs, list) and cs:
            ids = {str(c.get("id") or (c.get("athlete") or {}).get("id") or "") for c in cs if isinstance(c, dict)}
            refs = " ".join(_ref(c.get("athlete")) for c in cs if isinstance(c, dict))
            if fid in ids or f"/athletes/{fid}" in refs or (cid and str(obj.get("id")) == cid):
                return obj
        for v in obj.values():
            r = _find_comp(v, fid, cid)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_comp(v, fid, cid)
            if r is not None:
                return r
    return None


def _athlete_name(c: dict) -> str:
    a = c.get("athlete") or {}
    name = a.get("displayName") or a.get("fullName") or c.get("displayName") or ""
    if name or not _ref(a):
        return name
    oid = _ref(a).rstrip("/").split("/")[-1].split("?")[0]
    try:
        d = disk_cached(f"fighter|{oid}", TTL_FIGHTER, lambda: http_json(_ref(a), retries=1))
        d = d.get("athlete", d)
        return d.get("displayName") or d.get("fullName") or ""
    except Exception:
        return ""


def _bout_from_comp(comp: dict, fid: str, event_name: str, event_date: str, ufc: bool) -> dict | None:
    cs = [c for c in comp.get("competitors") or [] if isinstance(c, dict)]
    if len(cs) != 2:
        return None

    def is_me(c):
        return str(c.get("id") or (c.get("athlete") or {}).get("id") or "") == fid or \
            _ref(c.get("athlete")).rstrip("/").split("/")[-1].split("?")[0] == fid

    me = next((c for c in cs if is_me(c)), None)
    if me is None:
        return None
    opp = cs[1] if me is cs[0] else cs[0]
    st = comp.get("status") or {}
    if isinstance(st, dict) and "$ref" in st and len(st) == 1:
        try:
            st = http_json(_ref(st), retries=1)
        except Exception:
            st = {}
    typ = st.get("type") or {}
    if typ.get("completed") is False or typ.get("state") in ("pre", "in"):
        return None
    if me.get("winner") is True:
        res = "W"
    elif opp.get("winner") is True:
        res = "L"
    else:
        res = "D" if typ.get("completed") or typ.get("state") == "post" else ""
    if not res:
        return None
    result = st.get("result") or {}
    method_raw = (_txt(result.get("description")) or _txt(result.get("displayName")) or _txt(result.get("name"))
                  or _txt(typ.get("detail")) or _txt(typ.get("shortDetail")) or _txt(typ.get("description")))
    when = str(comp.get("date") or comp.get("startDate") or event_date or "")
    d = to_et_date(when) if when else None
    return {"result": res, "method": _method_label(method_raw), "code": _method_code(method_raw),
            "detail": method_raw, "round": int(num(st.get("period"), 0)),
            "time": str(st.get("displayClock") or st.get("clock") or ""), "opponent": _athlete_name(opp),
            "event": event_name, "date": f"{d:%b %d, %Y}" if d else "", "_d": d.isoformat() if d else "", "ufc": ufc}


def _event_payload(eid: str, ev_ref: str) -> dict:
    """Whole past event: common/v3 fightcenter (names inline), else the core event ref. Cached 30 days and
    shared by every fighter who was on that card."""
    def fetch():
        for url in (f"{WEB}/mma/ufc/fightcenter/{eid}", ev_ref):
            if not url:
                continue
            try:
                d = http_json(url, retries=1)
                if d:
                    return d
            except Exception:
                continue
        return {}
    return disk_cached(f"mmaevent|{eid}", TTL_PAST_EVENT, fetch)


def _resolve_item(it: dict, fid: str) -> dict | None:
    ev_ref, comp_ref = _ref(it.get("event")), _ref(it.get("competition"))
    eid, cid = _ids_from(ev_ref or comp_ref)
    if not cid:
        cid = _ids_from(comp_ref)[1]
    ufc = "/leagues/ufc/" in (ev_ref + comp_ref) or "ufc" in _txt(it.get("league")).lower()
    comp, ev_name, ev_date = None, "", ""
    if eid:
        try:
            ev = _event_payload(eid, ev_ref)
        except Exception:
            ev = {}
        evo = ev.get("event") or ev.get("header") or ev
        ev_name = _txt(evo.get("name") or evo.get("shortName")) if isinstance(evo, dict) else ""
        ev_date = str(evo.get("date") or "") if isinstance(evo, dict) else ""
        comp = _find_comp(ev, fid, cid)
    if comp is None and comp_ref:
        try:
            comp = disk_cached(f"mmacomp|{eid}|{cid}", TTL_PAST_EVENT, lambda: http_json(comp_ref, retries=1))
        except Exception:
            comp = None
    if not isinstance(comp, dict):
        return None
    ufc = ufc or "ufc" in ev_name.lower()
    return _bout_from_comp(comp, fid, ev_name, ev_date, ufc)


def _eventlog_history(fid: str) -> dict:
    """Core eventlog -> newest past bouts, each resolved from its event (fightcenter / core competition)."""
    raw, items = None, []
    for url in (f"{CORE}/athletes/{fid}/eventlog", f"{CORE}/leagues/ufc/athletes/{fid}/eventlog"):
        try:
            raw = http_json(url, retries=1)
        except Exception:
            continue
        evs = raw.get("events") or {}
        items = evs.get("items") if isinstance(evs, dict) else (evs if isinstance(evs, list) else [])
        pages = int(num((evs if isinstance(evs, dict) else {}).get("pageCount") or raw.get("pageCount"), 1))
        if pages > 1:  # newest bouts may sit on the last page
            try:
                last = http_json(url, {"page": pages}, retries=1)
                le = last.get("events") or {}
                items = (items or []) + list((le.get("items") if isinstance(le, dict) else le) or [])
            except Exception:
                pass
        if items:
            break
    if not items:
        return {}
    full = []
    for it in items:
        if isinstance(it, dict) and "$ref" in it and len(it) == 1:
            it = {"event": {"$ref": it["$ref"]}}
        if isinstance(it, dict) and it.get("played") is not False:
            full.append(it)
    if not full:
        return {}
    allrefs = [(_ref(it.get("event")) + _ref(it.get("competition"))) for it in full]
    ufc_count = sum("/leagues/ufc/" in r for r in allrefs)
    # orient newest-first: resolve both ends, keep the order whose head is newer
    head, tail = _resolve_item(full[0], fid), (_resolve_item(full[-1], fid) if len(full) > 1 else None)
    if head and tail and tail["_d"] > head["_d"]:
        full.reverse()
        head, tail = tail, head
    bouts = [b for b in (head,) if b]
    for it in full[1:]:
        if len(bouts) >= EVENTLOG_BOUTS:
            break
        b = tail if (it is full[-1] and tail) else _resolve_item(it, fid)
        if b:
            bouts.append(b)
    bouts.sort(key=lambda h: h["_d"], reverse=True)
    return {"bouts": bouts, "count": len(full), "ufc_count": ufc_count or None, "source": "ESPN eventlog"}


def _fetch_history(fid: str) -> dict:
    """Fight history: core eventlog (+ fightcenter per event) first, then fight-history / gamelog feeds."""
    try:
        d = _eventlog_history(fid)
        if d.get("bouts"):
            return d
    except Exception:
        pass
    for url in (f"{WEB}/mma/athletes/{fid}/fighthistory", f"{WEB}/mma/athletes/{fid}/gamelog",
                f"{SITE_MMA}/athletes/{fid}/gamelog", f"{WEB}/mma/ufc/athletes/{fid}/gamelog"):
        try:
            d = http_json(url, retries=1)
        except Exception:
            continue
        if parse_history(d):
            return d
    return {}


def _fetch_records(fid: str) -> dict:
    """Career record splits (tkos / submissions / decisions / lossesByKo ...) from core /records."""
    try:
        raw = http_json(f"{CORE}/athletes/{fid}/records", retries=1)
    except Exception:
        return {}
    items = []
    for it in raw.get("items") or []:
        if isinstance(it, dict) and "$ref" in it and len(it) == 1:
            try:
                it = http_json(_ref(it), retries=1)
            except Exception:
                continue
        items.append(it)
    return {"items": items}


def prefetch_fighter(fid: str) -> None:
    disk_cached(f"fighter|{fid}", TTL_FIGHTER, lambda: _fetch_athlete(fid))
    disk_cached(f"fighterstats|{fid}", TTL_FIGHTER, lambda: _fetch_stats(fid))
    disk_cached(f"fighterhist2|{fid}", TTL_FIGHTER, lambda: _fetch_history(fid))
    disk_cached(f"fighterrec|{fid}", TTL_FIGHTER, lambda: _fetch_records(fid))


def _find_record(obj) -> str:
    """First 'W-L-D' display string under a wins-losses-draws stat or a 'record' key."""
    if isinstance(obj, dict):
        name = str(obj.get("name") or obj.get("abbreviation") or "").lower()
        dv = obj.get("displayValue") or obj.get("summary")
        if isinstance(dv, str) and ("wins-losses" in name or name in ("record", "overall")) and "-" in dv:
            return dv
        for v in obj.values():
            r = _find_record(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_record(v)
            if r:
                return r
    return ""


# ---- fight history -------------------------------------------------------
def _txt(v) -> str:
    if isinstance(v, dict):
        return str(v.get("displayName") or v.get("shortDisplayName") or v.get("text") or v.get("description")
                   or v.get("name") or v.get("abbreviation") or "")
    return "" if v is None else str(v)


def _method_code(m: str) -> str:
    u = (m or "").upper()
    if "DQ" in u or "DISQUAL" in u or "KO" in u or "KNOCKOUT" in u:
        return "KO"
    if "SUB" in u:
        return "SUB"
    if "DEC" in u or "UNANIMOUS" in u or "SPLIT" in u or "MAJORITY" in u:
        return "DEC"
    return ""


def _method_label(m: str) -> str:
    c, u = _method_code(m), (m or "").upper()
    if c == "KO":
        return "DQ" if "DQ" in u or "DISQUAL" in u else "KO/TKO"
    if c == "SUB":
        return "SUB"
    if c == "DEC":
        return "S-DEC" if "SPLIT" in u or "S-DEC" in u else "M-DEC" if "MAJORITY" in u or "M-DEC" in u else "U-DEC"
    return (m or "").strip()[:14]


_COLS = {"method": ("method", "decision", "result", "res", "dec", "wintype", "howwon"),
         "round": ("round", "rnd", "rd", "r", "period"), "time": ("time", "clock", "displayclock")}


def parse_history(raw: dict) -> list[dict]:
    """Bouts (newest first) from ESPN gamelog-style payloads: top-level events {id: meta} + rows in
    seasonTypes[].categories[].events[] aligned to names/labels; also accepts a plain events list."""
    if not isinstance(raw, dict):
        return []
    if isinstance(raw.get("bouts"), list):  # already normalized (core eventlog path)
        return raw["bouts"]
    evs = raw.get("events") or raw.get("fights") or raw.get("fightHistory") or {}
    if isinstance(evs, list):
        evs = {str(e.get("id") or e.get("eventId") or i): e for i, e in enumerate(evs) if isinstance(e, dict)}
    if not isinstance(evs, dict) or not evs:
        return []
    names = [re.sub(r"[^a-z]", "", str(x).lower()) for x in (raw.get("names") or raw.get("labels") or [])]
    labels = [re.sub(r"[^a-z]", "", str(x).lower()) for x in (raw.get("labels") or [])]

    def col(kind):
        for cand in _COLS[kind]:
            for arr in (names, labels):
                if cand in arr:
                    return arr.index(cand)
        return None

    ci = {k: col(k) for k in _COLS}
    rows = {}
    for stp in raw.get("seasonTypes") or []:
        for cat in stp.get("categories") or []:
            for e in cat.get("events") or []:
                if isinstance(e, dict) and e.get("eventId") is not None:
                    rows[str(e["eventId"])] = e.get("stats") or []
    out = []
    for eid, e in evs.items():
        if not isinstance(e, dict):
            continue
        st = rows.get(str(eid), [])

        def cell(kind):
            i = ci[kind]
            return st[i] if i is not None and i < len(st) else None

        res = _txt(e.get("gameResult") or e.get("result") or e.get("outcome")).strip().upper()[:1]
        if res not in ("W", "L", "D", "N"):
            w = e.get("winner")
            res = "W" if w is True else "L" if w is False else ""
        if not res:
            continue
        method_raw = (cell("method") or _txt(e.get("method")) or _txt(e.get("decision")) or _txt(e.get("winMethod"))
                      or _txt(e.get("resultDescription")) or _txt((e.get("status") or {}).get("result"))
                      or _txt(e.get("eventNote")) or "")
        rnd = cell("round") or e.get("round") or e.get("period") or (e.get("status") or {}).get("period")
        tm = cell("time") or e.get("time") or e.get("clock") or (e.get("status") or {}).get("displayClock") or ""
        opp = e.get("opponent") or e.get("opponentAthlete") or {}
        when = str(e.get("gameDate") or e.get("date") or "")
        d = to_et_date(when) if when else None
        league = _txt(e.get("league") or e.get("leagueName") or e.get("leagueAbbreviation"))
        out.append({"result": res, "method": _method_label(method_raw), "code": _method_code(method_raw),
                    "detail": method_raw if _method_label(method_raw) != method_raw else "",
                    "round": int(num(rnd, 0)), "time": str(tm), "opponent": _txt(opp),
                    "event": _txt(e.get("eventName") or e.get("name") or e.get("shortName")),
                    "date": f"{d:%b %d, %Y}" if d else "", "_d": d.isoformat() if d else "",
                    "ufc": "ufc" in (league + " " + _txt(e.get("eventName") or e.get("name"))).lower()})
    out.sort(key=lambda h: h["_d"], reverse=True)
    return out


def _norm_stats(s: dict) -> dict:
    return {re.sub(r"[^a-z0-9]", "", k): v for k, v in s.items()}


def fighter_profile(fid: str, card_fighter: dict, weight_class: str = "") -> dict:
    prof = {"name": card_fighter["name"], "headshot": card_fighter.get("headshot", ""),
            "flag": card_fighter.get("flag", ""), "country": card_fighter.get("country", ""),
            "record": _record(card_fighter.get("record", "")), "age": None, "height": None, "reach": None,
            "stance": "", "weight_class": weight_class, "stats": {}, "est": [], "methods": None,
            "methods_est": False, "detail_ok": False, "finish_rounds": [], "recent": [], "losses_by": {},
            "ufc_bouts": None, "bouts_listed": 0, "src": []}
    got = {}
    if fid:
        for key, fn in (("bio", _fetch_athlete), ("stats", _fetch_stats), ("hist", _fetch_history),
                        ("rec", _fetch_records)):
            ck = {"bio": f"fighter|{fid}", "stats": f"fighterstats|{fid}", "hist": f"fighterhist2|{fid}",
                  "rec": f"fighterrec|{fid}"}[key]
            try:
                got[key] = disk_cached(ck, TTL_FIGHTER, lambda fn=fn: fn(fid)) or {}
            except Exception:
                got[key] = {}
    raw, raw_stats, raw_hist, raw_rec = (got.get(k, {}) for k in ("bio", "stats", "hist", "rec"))
    a = raw.get("athlete", raw) if isinstance(raw, dict) else {}
    if a:
        prof["detail_ok"] = True
        prof["src"].append("ESPN bio")
        prof["age"] = num(a.get("age"), default=0) or None
        prof["height"] = inches(a.get("displayHeight")) or inches(a.get("height"))
        prof["reach"] = inches(a.get("displayReach")) or inches(a.get("reach"))
        st = a.get("stance")
        prof["stance"] = (st.get("text") or st.get("name") or "") if isinstance(st, dict) else (st or "")
        wc = a.get("weightClass")
        prof["weight_class"] = prof["weight_class"] or (
            (wc.get("text") or wc.get("shortName") or "") if isinstance(wc, dict) else (wc or ""))
        hs = a.get("headshot")
        if isinstance(hs, dict) and hs.get("href"):
            prof["headshot"] = hs["href"]
        fl = a.get("flag")
        if isinstance(fl, dict) and fl.get("href"):
            prof["flag"] = fl["href"]
            prof["country"] = fl.get("alt") or prof["country"]
        prof["country"] = prof["country"] or a.get("citizenship", "")
    s = _norm_stats(flatten_stats(raw_stats))
    rec = _norm_stats(flatten_stats(raw_rec))
    for src in (rec, _norm_stats(flatten_stats(raw))):
        for k, v in src.items():
            s.setdefault(k, v)
    if not any(prof["record"]):
        summary = _find_record(raw) or _find_record(raw_rec) or _find_record(raw_stats)
        if summary:
            prof["record"] = _record(summary)
    if prof["reach"] is None:
        prof["reach"] = pick(s, "reach")
    stats = {
        "slpm": pick(s, "sigstrikeslandedpermin", "slpm", "sigstrikeslandedperminute", "strikelpm",
                     "strikeslandedperminute", "strikeslandedpermin"),
        "str_acc": pick(s, "sigstrikeaccuracy", "sigstracc", "stracc", "strikeaccuracy", "sigstrikesaccuracy",
                        "strikingaccuracy"),
        "sapm": pick(s, "sigstrikesabsorbedpermin", "sapm", "sigstrikesabsorbedperminute", "strikesabsorbedpermin",
                     "strikesabsorbedperminute"),
        "str_def": pick(s, "sigstrikedefense", "strdef", "strikedefense", "strikingdefense"),
        "td_avg": pick(s, "takedownavg", "tdavg", "takedownsaverage", "takedownaverage", "takedownsper15min",
                       "takedownsper15"),
        "td_acc": pick(s, "takedownaccuracy", "tdacc", "takedownsaccuracy"),
        "td_def": pick(s, "takedowndefense", "tddef", "takedownsdefense"),
        "sub_avg": pick(s, "submissionavg", "subavg", "submissionsaverage", "submissionaverage", "subattemptsper15"),
    }
    # derive rates from raw counts when ESPN only gives totals
    landed, att = pick(s, "sigstrikeslanded", "ssl", "sigstrlanded"), pick(s, "sigstrikesattempted", "ssa")
    if stats["str_acc"] is None and landed and att:
        stats["str_acc"] = landed / att * 100
    tdl, tda = pick(s, "takedownslanded", "tdl"), pick(s, "takedownsattempted", "tda")
    if stats["td_acc"] is None and tdl is not None and tda:
        stats["td_acc"] = tdl / tda * 100
    for k in ("str_acc", "str_def", "td_acc", "td_def"):  # fractions (0.52) -> percent
        if stats[k] is not None and 0 < stats[k] <= 1.0:
            stats[k] *= 100
    prof["stats"] = {k: v for k, v in stats.items() if v is not None}
    if prof["stats"]:
        prof["src"].append("ESPN career stats")

    # fight history -> method splits, finish rounds, recent form
    hist = parse_history(raw_hist)
    if hist:
        prof["src"].append("ESPN fight history")
        ufc_only = [h for h in hist if h["ufc"]]
        prof["recent"] = (ufc_only or hist)[:5]
        prof["bouts_listed"] = int(raw_hist.get("count") or len(hist)) if isinstance(raw_hist, dict) else len(hist)
        prof["ufc_bouts"] = (raw_hist.get("ufc_count") if isinstance(raw_hist, dict) and raw_hist.get("ufc_count")
                             else (len(ufc_only) if ufc_only else None))
        wins = [h for h in hist if h["result"] == "W"]
        ko = sum(h["code"] == "KO" for h in wins)
        sub = sum(h["code"] == "SUB" for h in wins)
        dec = sum(h["code"] == "DEC" for h in wins)
        if ko + sub + dec:
            prof["methods"] = (ko, sub, dec)
        prof["finish_rounds"] = [h["round"] for h in wins if h["code"] in ("KO", "SUB") and h["round"] > 0]
        losses = [h for h in hist if h["result"] == "L"]
        prof["losses_by"] = {c: sum(h["code"] == c for h in losses) for c in ("KO", "SUB", "DEC")}
    if rec:
        prof["src"].append("ESPN records")
    lko = pick(rec, "lossesbyko", "lossesbytko", "lossesbykotko", "kolosses", "tkolosses")
    lsub = pick(rec, "lossesbysubmission", "lossesbysub", "sublosses", "submissionlosses")
    ldec = pick(rec, "lossesbydecision", "lossesbydec", "declosses", "decisionlosses")
    if lko is not None or lsub is not None:
        prof["losses_by"] = {"KO": int(lko or 0), "SUB": int(lsub or 0),
                             "DEC": int(ldec) if ldec is not None else prof["losses_by"].get("DEC", 0)}
    # career method totals (records / stats) win when they cover more fights than the listed history
    ko = pick(s, "tkowins", "kowins", "tkos", "knockouts", "kotko", "tko", "kotkowins")
    sub = pick(s, "submissionwins", "subwins", "submissions", "sub")
    dec = pick(s, "decisionwins", "decwins", "decisions", "dec")
    if ko is not None and sub is not None and dec is not None and (ko + sub + dec) > sum(prof["methods"] or (0,)):
        prof["methods"] = (ko, sub, dec)

    # fallback: divisional averages for anything ESPN left null (debuts)
    avg = div_avg(prof["weight_class"])
    for k, v in avg.items():
        if k not in prof["stats"]:
            prof["stats"][k] = float(v)
            prof["est"].append(k)
    if prof["methods"] is None:
        k_, s_, d_ = _prior(prof["weight_class"])
        prof["methods"] = (k_ * 10, s_ * 10, d_ * 10)
        prof["methods_est"] = True
    return prof


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def predict_bout(bout: dict, fa: dict, fb: dict) -> dict:
    """Win probability for fighter A (first listed) plus method-of-victory split for both."""
    factors = []
    wa, la, _ = fa["record"]
    wb, lb, _ = fb["record"]
    wp_a, wp_b = (wa + 3) / (wa + la + 6), (wb + 3) / (wb + lb + 6)
    z = 2.4 * (wp_a - wp_b)
    factors.append(("Record (smoothed win %)", 2.4 * (wp_a - wp_b)))
    exp = 0.10 * math.log((wa + la + 1) / (wb + lb + 1))
    z += exp
    factors.append(("Experience", exp))
    if fa["reach"] and fb["reach"]:
        r = 0.035 * max(-8, min(8, fa["reach"] - fb["reach"]))
        z += r
        factors.append(("Reach", r))
    if fa["age"] and fb["age"]:
        ag = -0.045 * max(-10, min(10, fa["age"] - fb["age"]))
        if max(fa["age"], fb["age"]) >= 35:
            ag *= 1.4
        z += ag
        factors.append(("Age", ag))
    sa, sb = fa["stats"], fb["stats"]
    if all(k in sa and k in sb for k in ("slpm", "sapm")):
        st = 0.22 * ((sa["slpm"] - sa["sapm"]) - (sb["slpm"] - sb["sapm"]))
        st = max(-0.8, min(0.8, st))
        z += st
        factors.append(("Striking differential", st))
    if "td_avg" in sa and "td_avg" in sb:
        td_def_b = (sb.get("td_def", 60) or 60) / 100
        td_def_a = (sa.get("td_def", 60) or 60) / 100
        gr = 0.12 * (sa["td_avg"] * (1 - td_def_b) - sb["td_avg"] * (1 - td_def_a)) * 2
        gr = max(-0.5, min(0.5, gr))
        z += gr
        factors.append(("Grappling", gr))
    p_model = _sig(z)
    p_a = p_model
    ma, mb = ml_to_prob(bout["fighters"][0].get("ml")), ml_to_prob(bout["fighters"][1].get("ml"))
    market = None
    if ma and mb:
        market = ma / (ma + mb)  # remove the vig
        p_a = 0.5 * p_model + 0.5 * market
    wc = bout.get("weight_class") or fa.get("weight_class") or fb.get("weight_class") or ""
    prior = _prior(wc)

    def methods(win: dict, lose: dict) -> tuple[float, float, float]:
        ko, sub, dec = prior
        if win["methods"]:
            k, s, d = win["methods"]
            t = k + s + d
            ko, sub, dec = 0.5 * ko + 0.5 * k / t, 0.5 * sub + 0.5 * s / t, 0.5 * dec + 0.5 * d / t
        if win["stats"].get("sub_avg") is not None:
            sub *= 0.8 + 0.25 * min(win["stats"]["sub_avg"], 2.0)
        if bout.get("rounds") == 5:
            dec *= 0.9
        t = ko + sub + dec
        return ko / t, sub / t, dec / t

    m_a, m_b = methods(fa, fb), methods(fb, fa)
    outcomes = [(p_a * m_a[0], 0, "KO/TKO"), (p_a * m_a[1], 0, "Submission"), (p_a * m_a[2], 0, "Decision"),
                ((1 - p_a) * m_b[0], 1, "KO/TKO"), ((1 - p_a) * m_b[1], 1, "Submission"),
                ((1 - p_a) * m_b[2], 1, "Decision")]
    winner = 0 if p_a >= 0.5 else 1
    best = max((o for o in outcomes if o[1] == winner), key=lambda o: o[0])
    return {"p_a": p_a, "p_model": p_model, "market": market, "methods_a": m_a, "methods_b": m_b,
            "pick": winner, "pick_method": best[2], "pick_method_p": best[0] / (p_a if winner == 0 else 1 - p_a),
            "factors": factors, "weight_class": wc}
