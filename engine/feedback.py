"""Your real results -> builder tuning.

Legs you track (from the streak parlay builder, a betslip screenshot, or entered by hand) are saved, graded
from ESPN box scores once their games are final, and compared with the hit chance the builder gave them. The
gap becomes a per-market calibration factor that the parlay builder applies to future legs.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from datetime import date

from .common import CACHE_DIR, norm_name

STORE = CACHE_DIR / "feedback.json"
SHRINK = 15          # pseudo-legs at the builder's own estimate (more of your results = more weight)
MIN_N = 5            # graded legs in a market before it gets its own factor
FACTOR_RANGE = (0.70, 1.10)


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
def load() -> list[dict]:
    try:
        return json.loads(STORE.read_text()) if STORE.exists() else []
    except Exception:
        return []


def save(rows: list[dict]) -> None:
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1))
    tmp.replace(STORE)


def _key(r: dict) -> str:
    return (f'{r.get("date")}|{r.get("sport")}|{r.get("pid") or norm_name(r.get("name", ""))}|{r.get("m")}|'
            f'{r.get("t")}|{r.get("side", "over")}|{r.get("source", "")}')


def add(legs: list[dict], source: str, day: str | None = None) -> int:
    """legs: {sport, gid, pid, name, m, label, t, p, tier}. Returns how many were new."""
    rows = load()
    have = {_key(r) for r in rows}
    n = 0
    for l in legs:
        r = {"date": day or date.today().isoformat(), "result": None, "actual": None, "source": source,
             "ts": time.time(), **l}
        if _key(r) not in have:
            rows.append(r)
            have.add(_key(r))
            n += 1
    save(rows)
    return n


def find(pid, m: str, t: float, day: str | None = None) -> int | None:
    day = day or date.today().isoformat()
    rows = load()
    return next((i for i, r in enumerate(rows) if str(r.get("pid")) == str(pid) and r.get("m") == m
                 and r.get("t") == t and r.get("date") == day), None)


def set_result(idx: int, result: str | None) -> None:
    rows = load()
    if 0 <= idx < len(rows):
        rows[idx]["result"] = result
        rows[idx]["graded_by"] = "you"
        save(rows)


def remove(idx: int) -> None:
    rows = load()
    if 0 <= idx < len(rows):
        rows.pop(idx)
        save(rows)


# --------------------------------------------------------------------------
# Auto-grading from ESPN box scores
# --------------------------------------------------------------------------
def grade_pending(leagues: dict, box_stats, actual_value) -> int:
    """Grade every tracked leg whose game has a final box score. Returns how many were graded now."""
    rows = load()
    n = 0
    for r in rows:
        if r.get("result") or not r.get("gid") or not r.get("pid") or r.get("sport") not in leagues:
            continue
        lg = leagues[r["sport"]]
        try:
            box = box_stats(lg, r["gid"])
        except Exception:
            box = {}
        if not box or not box.get("_final"):
            continue  # only grade finished games
        val = actual_value(lg.kind, r["m"], box.get(str(r["pid"])))
        if val is None:
            r["result"] = "void"  # didn't play
        else:
            r["actual"] = val
            if r.get("side") == "under":
                line = float(r.get("line", r["t"]))
                r["result"] = "push" if val == line else ("hit" if val < line else "miss")
            else:
                r["result"] = "hit" if val >= r["t"] else "miss"
        r["graded_by"] = "box score"
        n += 1
    if n:
        save(rows)
    return n


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibration(rows: list[dict] | None = None) -> dict:
    """{"_all": stats, (sport, m): stats} with stats = n, hits, pred (mean predicted), actual, factor."""
    rows = load() if rows is None else rows
    graded = [r for r in rows if r.get("result") in ("hit", "miss") and r.get("p")
              and not str(r.get("source", "")).startswith("ai:")]  # calibrate the builder on YOUR legs
    groups: dict = {"_all": graded}
    for r in graded:
        groups.setdefault((r.get("sport"), r.get("m")), []).append(r)
    out = {}
    for k, g in groups.items():
        if not g:
            continue
        n = len(g)
        hits = sum(r["result"] == "hit" for r in g)
        pred = sum(r["p"] for r in g) / n
        cal = (hits + SHRINK * pred) / (n + SHRINK)
        f = min(FACTOR_RANGE[1], max(FACTOR_RANGE[0], cal / pred)) if pred else 1.0
        out[k] = {"n": n, "hits": hits, "pred": pred, "actual": hits / n, "factor": f}
    return out


def factors(sport: str, cal: dict | None = None) -> dict[str, float]:
    """Per-market multipliers for one sport (markets with < MIN_N graded legs use the overall factor)."""
    cal = calibration() if cal is None else cal
    base = cal.get("_all", {}).get("factor", 1.0) if cal.get("_all", {}).get("n", 0) >= MIN_N else 1.0
    out = {"_default": base}
    for k, v in cal.items():
        if isinstance(k, tuple) and k[0] == sport and v["n"] >= MIN_N:
            out[k[1]] = v["factor"]
    return out


# --------------------------------------------------------------------------
# Betslip screenshots (OCR)
# --------------------------------------------------------------------------
MARKET_WORDS = [
    ("PRA", r"pts\s*\+\s*reb\s*\+\s*ast|points\s*\+\s*rebounds\s*\+\s*assists|\bpra\b"),
    ("PASS", r"pass(ing)?\s*y(ar)?ds"), ("RUSH", r"rush(ing)?\s*y(ar)?ds"), ("REC", r"rec(eiving)?\s*y(ar)?ds"),
    ("RECS", r"receptions"), ("SOG", r"shots(\s*on\s*goal)?"), ("REB", r"rebounds?|\breb\b"),
    ("AST", r"assists?|\bast\b"), ("G", r"\bgoals?\b"), ("PTS", r"points?|\bpts\b"),
]


def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    return shutil.which("tesseract") is not None


def _prep(img):
    """Dark sportsbook screenshots: grayscale, invert if dark, upscale — tesseract reads dark-on-light best."""
    from PIL import ImageOps

    g = ImageOps.grayscale(img)
    if sum(g.resize((32, 32)).getdata()) / 1024 < 110:
        g = ImageOps.invert(g)
    w, h = g.size
    if w < 1400:
        g = g.resize((w * 2, h * 2))
    return g


def _status_color(img, top: int, bottom: int, scale: float) -> str | None:
    """Look left of a leg's text line for the sportsbook's green check or red x."""
    rgb = img.convert("RGB")
    w, _ = rgb.size
    y0, y1 = int(top / scale), int(bottom / scale)
    green = red = 0
    for y in range(max(0, y0 - 6), y1 + 6, 2):
        for x in range(0, int(w * 0.22), 2):
            try:
                r, g, b = rgb.getpixel((x, y))
            except Exception:
                continue
            if g > 120 and g > r * 1.4 and g > b * 1.2:
                green += 1
            elif r > 150 and r > g * 1.6 and r > b * 1.4:
                red += 1
    if max(green, red) < 6:
        return None
    return "hit" if green > red else "miss"


def read_slip(img, players: list[dict]) -> list[dict]:
    """OCR a betslip image and match legs to tonight's players.
    players: [{name, pid, sport, gid}]. Returns [{name, pid, sport, gid, m, t, status, text}]."""
    import pytesseract

    pre = _prep(img)
    scale = pre.size[0] / img.size[0]
    data = pytesseract.image_to_data(pre, output_type=pytesseract.Output.DICT)
    lines: dict = {}
    for i, word in enumerate(data["text"]):
        if not str(word).strip():
            continue
        k = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        ln = lines.setdefault(k, {"words": [], "top": data["top"][i], "bottom": data["top"][i] + data["height"][i]})
        ln["words"].append(word)
        ln["top"] = min(ln["top"], data["top"][i])
        ln["bottom"] = max(ln["bottom"], data["top"][i] + data["height"][i])
    by_last: dict = {}
    for p in players:
        parts = norm_name(p["name"]).split()
        if parts:
            by_last.setdefault(parts[-1], []).append(p)
    out = []
    for ln in sorted(lines.values(), key=lambda l: l["top"]):
        text = " ".join(ln["words"])
        low = norm_name(text)
        who = None
        for last, cands in by_last.items():
            if re.search(rf"\b{re.escape(last)}\b", low):
                who = next((c for c in cands if norm_name(c["name"]).split()[0][:1] in low), cands[0])
                break
        m = next((code for code, pat in MARKET_WORDS if re.search(pat, text, re.I)), None)
        num = re.search(r"(\d+(?:\.\d)?)\s*\+", text) or re.search(r"over\s*(\d+(?:\.\d)?)", text, re.I)
        if not (who and m and num):
            continue
        t = float(num.group(1))
        if "+" not in text[num.start(): num.end()]:
            t = t + 0.5  # "Over 6.5" = 7+
        out.append({**who, "m": m, "t": t, "status": _status_color(img, ln["top"], ln["bottom"], scale),
                    "text": text})
    return out


# --------------------------------------------------------------------------
# Imported AI / tipster picks
# --------------------------------------------------------------------------
_SIDE_OVER = re.compile(r"\bover\b|\bo\s?\d|\bmore\b|\bhigher\b", re.I)
_SIDE_UNDER = re.compile(r"\bunder\b|\bu\s?\d|\bless\b|\blower\b", re.I)


def parse_picks(text: str, players: list[dict]) -> list[dict]:
    """Free text from any AI pick source -> legs. One pick per line, any common wording:
    "Jalen Brunson Over 24.5 Points", "Brunson o24.5 pts", "J. Brunson 25+ PTS", "Under 8.5 Reb - Bam Adebayo".
    players: [{name, pid, sport, gid}]. Returns [{..player, m, side, t, line, text}]."""
    by_last: dict = {}
    for p in players:
        parts = norm_name(p["name"]).split()
        if parts:
            by_last.setdefault(parts[-1], []).append(p)
    out, seen = [], set()
    for raw in re.split(r"[\n;]+", text or ""):
        line_txt = raw.strip()
        if not line_txt:
            continue
        low = norm_name(line_txt)
        who = None
        for last, cands in by_last.items():
            if re.search(rf"\b{re.escape(last)}\b", low):
                toks = low.split()
                first = [c for c in cands if norm_name(c["name"]).split()[0] in toks] or \
                        [c for c in cands if norm_name(c["name"])[:1] in toks]
                who = (first or cands)[0]
                break
        m = next((code for code, pat in MARKET_WORDS if re.search(pat, line_txt, re.I)), None)
        if not (who and m):
            continue
        plus = re.search(r"(\d+(?:\.\d+)?)\s*\+", line_txt)
        num = plus or re.search(r"(?:over|under|o|u)\s*(\d+(?:\.\d+)?)", line_txt, re.I) or \
            re.search(r"(\d+(?:\.\d+)?)", re.sub(r"[+-]\d{3,}", " ", line_txt))
        if not num:
            continue
        val = float(num.group(1))
        under = not plus and bool(_SIDE_UNDER.search(line_txt)) and not _SIDE_OVER.search(line_txt)
        if plus:
            side, t, line = "over", val, val - 0.5
        elif under:
            side, t, line = "under", val, val
        else:  # over L -> clears at floor(L) + 1
            side, line = "over", val
            t = float(int(val) + 1)  # "over 24.5" and "over 24" both need 25+
        key = (who["pid"], m, side, line)
        if key in seen:
            continue
        seen.add(key)
        out.append({**who, "m": m, "side": side, "t": t, "line": line, "text": line_txt})
    return out


def source_stats(rows: list[dict] | None = None) -> dict:
    """{source name: {n, hits, rate}} for imported AI picks (graded only)."""
    rows = load() if rows is None else rows
    out: dict = {}
    for r in rows:
        src = str(r.get("source", ""))
        if not src.startswith("ai:") or r.get("result") not in ("hit", "miss"):
            continue
        d = out.setdefault(src[3:], {"n": 0, "hits": 0})
        d["n"] += 1
        d["hits"] += r["result"] == "hit"
    for d in out.values():
        d["rate"] = d["hits"] / d["n"] if d["n"] else 0.0
    return out


def agreement(rows: list[dict], pid: str, m: str, t: float, day: str) -> list[str]:
    """Imported AI sources that picked this player/market on the over at a line no higher than t, today."""
    srcs = set()
    for r in rows:
        if (str(r.get("source", "")).startswith("ai:") and str(r.get("pid")) == str(pid) and r.get("m") == m
                and r.get("side", "over") == "over" and r.get("date") == day and float(r.get("t", 99)) <= t + 0.5):
            srcs.add(str(r["source"])[3:])
    return sorted(srcs)
