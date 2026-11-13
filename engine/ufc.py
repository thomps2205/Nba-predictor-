"""UFC fight cards (ESPN mma/ufc scoreboard), fighter profiles, win-probability and method-of-victory model."""

from __future__ import annotations

import math
from datetime import date, timedelta

from .common import (cache_fresh, disk_cached, flatten_stats, http_json, inches, ml_to_prob, num, pick, to_et,
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


# --------------------------------------------------------------------------
# Fighter profiles
# --------------------------------------------------------------------------
def fighter_keys(fid: str) -> list[str]:
    return [f"fighter|{fid}", f"fighterstats|{fid}"]


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
            return http_json(url, retries=1)
        except Exception:
            continue
    return {}


def prefetch_fighter(fid: str) -> None:
    disk_cached(f"fighter|{fid}", TTL_FIGHTER, lambda: _fetch_athlete(fid))
    disk_cached(f"fighterstats|{fid}", TTL_FIGHTER, lambda: _fetch_stats(fid))


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


def fighter_profile(fid: str, card_fighter: dict, ufcs: dict | None = None) -> dict:
    prof = {"name": card_fighter["name"], "headshot": card_fighter.get("headshot", ""),
            "flag": card_fighter.get("flag", ""), "country": card_fighter.get("country", ""),
            "record": _record(card_fighter.get("record", "")), "age": None, "height": None, "reach": None,
            "stance": "", "weight_class": "", "stats": {}, "methods": None, "detail_ok": False}
    raw, raw_stats = {}, {}
    if fid:
        try:
            raw = disk_cached(f"fighter|{fid}", TTL_FIGHTER, lambda: _fetch_athlete(fid))
        except Exception:
            raw = {}
        try:
            raw_stats = disk_cached(f"fighterstats|{fid}", TTL_FIGHTER, lambda: _fetch_stats(fid))
        except Exception:
            raw_stats = {}
    a = raw.get("athlete", raw) if isinstance(raw, dict) else {}
    if a:
        prof["detail_ok"] = True
        prof["age"] = num(a.get("age"), default=0) or None
        prof["height"] = inches(a.get("displayHeight")) or inches(a.get("height"))
        prof["reach"] = inches(a.get("displayReach")) or inches(a.get("reach"))
        st = a.get("stance")
        prof["stance"] = (st.get("text") or st.get("name") or "") if isinstance(st, dict) else (st or "")
        wc = a.get("weightClass")
        prof["weight_class"] = (wc.get("text") or wc.get("shortName") or "") if isinstance(wc, dict) else (wc or "")
        hs = a.get("headshot")
        if isinstance(hs, dict) and hs.get("href"):
            prof["headshot"] = hs["href"]
        fl = a.get("flag")
        if isinstance(fl, dict) and fl.get("href"):
            prof["flag"] = fl["href"]
            prof["country"] = fl.get("alt") or prof["country"]
        prof["country"] = prof["country"] or a.get("citizenship", "")
    s = flatten_stats(raw)
    flatten_stats(raw_stats, s)
    if not any(prof["record"]):
        summary = _find_record(raw) or _find_record(raw_stats)
        if summary:
            prof["record"] = _record(summary)
    if prof["reach"] is None:
        prof["reach"] = pick(s, "reach")
    stats = {
        "slpm": pick(s, "sigstrikeslandedpermin", "slpm", "sigstrikeslandedperminute", "strikelpm"),
        "str_acc": pick(s, "sigstrikeaccuracy", "sigstracc", "stracc", "strikeaccuracy"),
        "sapm": pick(s, "sigstrikesabsorbedpermin", "sapm", "sigstrikesabsorbedperminute"),
        "str_def": pick(s, "sigstrikedefense", "strdef", "strikedefense"),
        "td_avg": pick(s, "takedownavg", "tdavg", "takedownsaverage"),
        "td_acc": pick(s, "takedownaccuracy", "tdacc"),
        "td_def": pick(s, "takedowndefense", "tddef"),
        "sub_avg": pick(s, "submissionavg", "subavg", "submissionsaverage"),
    }
    for k in ("str_acc", "str_def", "td_acc", "td_def"):  # fractions (0.52) -> percent
        if stats[k] is not None and 0 < stats[k] <= 1.0:
            stats[k] *= 100
    prof["stats"] = {k: v for k, v in stats.items() if v is not None}
    ko = pick(s, "tkowins", "kowins", "tkos", "knockouts", "ko/tko")
    sub = pick(s, "submissionwins", "subwins", "submissions")
    dec = pick(s, "decisionwins", "decwins", "decisions")
    if ko is not None and sub is not None and dec is not None and (ko + sub + dec) > 0:
        prof["methods"] = (ko, sub, dec)
    prof["ufcstats"] = False
    if ufcs:  # UFCStats.com: official career stats + win methods override ESPN; physicals fill gaps
        prof["ufcstats"] = True
        for k in ("height", "reach", "age"):
            if not prof[k] and ufcs.get(k):
                prof[k] = ufcs[k]
        if not prof["stance"] and ufcs.get("stance"):
            prof["stance"] = ufcs["stance"]
        if not any(prof["record"]) and ufcs.get("record"):
            prof["record"] = tuple(ufcs["record"])
        prof["stats"].update({k: v for k, v in (ufcs.get("stats") or {}).items() if v is not None})
        if ufcs.get("methods"):
            prof["methods"] = tuple(ufcs["methods"])
        prof["detail_ok"] = True
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
