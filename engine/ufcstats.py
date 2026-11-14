"""Official UFC fighter stats scraped from UFCStats.com (search page + fighter-details page).

Gives career SLpM / Str. Acc. / SApM / Str. Def / TD Avg / TD Acc / TD Def / Sub. Avg, physicals, and win
methods counted from the fight-history table. Cached 7 days per fighter; any failure returns None.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from .common import BROWSER_UA, cache_fresh, disk_cached, flag_active, http_text, inches, norm_name, num, set_flag

BASE = "http://ufcstats.com"
TTL = 7 * 24 * 3600
HEADERS = {"User-Agent": BROWSER_UA, "Accept": "text/html,application/xhtml+xml"}
DOWN_FLAG = "ufcstats_down.flag"


def key(name: str) -> str:
    return f"ufcstats2|{norm_name(name)}"  # v2: adds fight history (rounds, recent bouts)


def is_cached(name: str) -> bool:
    return cache_fresh(key(name), TTL)


def _soup(html: str):
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "html.parser")


def _find_url(name: str) -> str | None:
    target = norm_name(name)
    parts = target.split()
    if not parts:
        return None
    html = http_text(f"{BASE}/statistics/fighters/search", {"query": parts[-1], "page": "all"}, headers=HEADERS)
    best = None
    for tr in _soup(html).select("tr.b-statistics__table-row"):
        tds = tr.find_all("td")
        if len(tds) < 2 or not tds[0].find("a"):
            continue
        full = norm_name(f"{tds[0].get_text(strip=True)} {tds[1].get_text(strip=True)}")
        href = tds[0].find("a").get("href")
        if full == target:
            return href
        if best is None and full.split() and full.split()[-1] == parts[-1] and full[:1] == target[:1]:
            best = href  # same last name + first initial
    return best


def _parse_details(html: str) -> dict:
    soup = _soup(html)
    out: dict = {"stats": {}}
    rec = soup.select_one("span.b-content__title-record")
    if rec:
        m = re.search(r"(\d+)-(\d+)-(\d+)", rec.get_text())
        if m:
            out["record"] = tuple(int(x) for x in m.groups())
    info = {}
    for li in soup.select("li.b-list__box-list-item"):
        txt = " ".join(li.get_text(" ", strip=True).split())
        if ":" in txt:
            k, _, v = txt.partition(":")
            info[k.strip().lower().rstrip(".")] = v.strip()
    out["height"] = inches(info.get("height", "").replace("--", ""))
    out["reach"] = inches(info.get("reach", "").replace("--", ""))
    out["stance"] = info.get("stance", "") if info.get("stance", "--") != "--" else ""
    dob = info.get("dob", "")
    try:
        born = datetime.strptime(dob, "%b %d, %Y").date()
        today = date.today()
        out["age"] = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except ValueError:
        out["age"] = None
    stat_keys = {"slpm": "slpm", "str_acc": "str. acc", "sapm": "sapm", "str_def": "str. def",
                 "td_avg": "td avg", "td_acc": "td acc", "td_def": "td def", "sub_avg": "sub. avg"}
    for k, label in stat_keys.items():
        v = info.get(label)
        if v not in (None, "", "--"):
            out["stats"][k] = num(v)
    ko = sub = dec = 0
    finish_rounds, history = [], []

    def ptexts(td):
        ps = [p.get_text(" ", strip=True) for p in td.find_all("p")]
        return [x for x in ps if x] or [td.get_text(" ", strip=True)]

    for tr in soup.select("tr.b-fight-details__table-row"):
        tds = tr.find_all("td")
        if len(tds) < 8:
            continue
        result = tds[0].get_text(" ", strip=True).lower()
        if result.startswith("next") or not result:
            continue
        method_full = ptexts(tds[7])
        method = (method_full[0] if method_full else "").upper()
        rnd = int(num(tds[8].get_text(strip=True), 0)) if len(tds) > 8 else 0
        names = ptexts(tds[1])
        event = ptexts(tds[6]) if len(tds) > 6 else []
        history.append({"result": result[:1].upper(), "method": method, "detail": method_full[1] if len(method_full) > 1 else "",
                        "round": rnd, "time": tds[9].get_text(strip=True) if len(tds) > 9 else "",
                        "opponent": names[1] if len(names) > 1 else "", "event": event[0] if event else "",
                        "date": event[1] if len(event) > 1 else ""})
        if not result.startswith("win"):
            continue
        if "KO" in method or "DQ" in method:
            ko += 1
            finish_rounds.append(rnd)
        elif "SUB" in method:
            sub += 1
            finish_rounds.append(rnd)
        elif "DEC" in method:
            dec += 1
    if ko + sub + dec:
        out["methods"] = (ko, sub, dec)
    out["finish_rounds"] = [r for r in finish_rounds if r > 0]
    out["recent"] = history[:5]  # newest first on UFCStats
    out["losses_by"] = {"KO": sum(1 for h in history if h["result"] == "L" and ("KO" in h["method"] or "DQ" in h["method"])),
                        "SUB": sum(1 for h in history if h["result"] == "L" and "SUB" in h["method"]),
                        "DEC": sum(1 for h in history if h["result"] == "L" and "DEC" in h["method"])}
    out["ufc_bouts"] = len(history)
    return out


def _lookup(name: str) -> dict:
    url = _find_url(name)
    if not url:
        return {"found": False}
    data = _parse_details(http_text(url, headers=HEADERS))
    data.update(found=True, url=url)
    return data


def fetch(name: str) -> dict | None:
    """Cached UFCStats profile for a fighter name, or None if unavailable."""
    if flag_active(DOWN_FLAG, 30 * 60) and not is_cached(name):
        return None
    try:
        d = disk_cached(key(name), TTL, lambda: _lookup(name))
    except Exception:
        set_flag(DOWN_FLAG)
        return None
    return d if d.get("found") else None


def cached(name: str) -> dict | None:
    from .common import cache_read

    d = cache_read(key(name))
    return d if d and d.get("found") else None
