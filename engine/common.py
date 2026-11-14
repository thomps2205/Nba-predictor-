"""HTTP (throttle + retries), disk cache, and small parsing helpers shared by every sport."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
CACHE_DIR = Path(os.environ.get("NBA_PREDICTOR_CACHE", Path(__file__).resolve().parent.parent / ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_SECONDS = 0.6
HTTP_TIMEOUT = 12

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

# Notes surfaced in the UI footer: stale-cache fallbacks and endpoint failures.
NOTES: dict[str, str] = {}

# League logos: ESPN's CDN defaults, replaced by the URL ESPN's own scoreboard reports (leagues[].logos).
LEAGUE_LOGOS: dict[str, str] = {
    "NBA": "https://a.espncdn.com/i/teamlogos/leagues/500/nba.png",
    "NCAAB": "https://a.espncdn.com/i/teamlogos/ncaa/500/ncaa.png",
    "NFL": "https://a.espncdn.com/i/teamlogos/leagues/500/nfl.png",
    "NHL": "https://a.espncdn.com/i/teamlogos/leagues/500/nhl.png",
    "UFC": "https://a.espncdn.com/i/teamlogos/leagues/500/ufc.png",
}

_lock = threading.Lock()
_last_call = [0.0]
_session = requests.Session()


def throttle() -> None:
    """At least RATE_LIMIT_SECONDS between any two outbound requests."""
    with _lock:
        wait = RATE_LIMIT_SECONDS - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------
def cache_path(key: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def cache_fresh(key: str, ttl: float) -> bool:
    p = cache_path(key)
    return p.exists() and time.time() - p.stat().st_mtime < ttl


def cache_read(key: str):
    p = cache_path(key)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except Exception:
        return None


def _cache_write(key: str, data) -> None:
    p = cache_path(key)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
    tmp.replace(p)


def disk_cached(key: str, ttl: float, fetch):
    """Fresh cached JSON, else fetch() and store. On failure fall back to a stale copy, else raise."""
    if cache_fresh(key, ttl):
        data = cache_read(key)
        if data is not None:
            return data
    try:
        data = fetch()
    except Exception as err:
        stale = cache_read(key)
        if stale is not None:
            age = (time.time() - cache_path(key).stat().st_mtime) / 60
            NOTES[key.split("|")[0]] = f"Live fetch failed ({type(err).__name__}); showing data cached {age:.0f} min ago."
            return stale
        raise
    _cache_write(key, data)
    return data


def cache_write(key: str, data) -> None:
    _cache_write(key, data)


def clear_disk_cache() -> None:
    for pattern in ("*.json", "*.flag", "*.tmp"):
        for p in CACHE_DIR.glob(pattern):
            p.unlink(missing_ok=True)


def flag_active(name: str, ttl: float) -> bool:
    p = CACHE_DIR / name
    return p.exists() and time.time() - p.stat().st_mtime < ttl


def set_flag(name: str) -> None:
    (CACHE_DIR / name).touch()


def http_json(url: str, params: dict | None = None, headers: dict | None = None,
              timeout: float = HTTP_TIMEOUT, retries: int = 2) -> dict:
    """GET JSON with throttle + retries. Timeouts/connection drops are retried once at most."""
    last: Exception | None = None
    for attempt in range(retries):
        throttle()
        try:
            r = _session.get(url, params=params, headers=headers or ESPN_HEADERS, timeout=timeout)
            if r.status_code == 404:
                raise FileNotFoundError(f"404 {url}")
            r.raise_for_status()
            return r.json()
        except FileNotFoundError:
            raise
        except Exception as err:
            last = err
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"{type(last).__name__}: {url}") from last


def http_text(url: str, params: dict | None = None, headers: dict | None = None,
              timeout: float = HTTP_TIMEOUT, retries: int = 2) -> str:
    """GET an HTML page with the same throttle/retry rules as http_json."""
    last: Exception | None = None
    for attempt in range(retries):
        throttle()
        try:
            r = _session.get(url, params=params, headers=headers or {"User-Agent": BROWSER_UA}, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as err:
            last = err
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"{type(last).__name__}: {url}") from last


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------
def num(v, default: float = 0.0) -> float:
    if isinstance(v, dict):
        v = v.get("value", v.get("displayValue"))
    if isinstance(v, str):
        v = v.strip().replace("+", "").replace(",", "").rstrip("%")
        if v in ("", "-", "--"):
            return default
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def made_att(v) -> tuple[float, float]:
    if isinstance(v, str) and "-" in v.lstrip("-"):
        m, _, a = v.partition("-")
        return num(m), num(a)
    return 0.0, 0.0


def to_minutes(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("PT"):
            m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?", v)
            if not m:
                return 0.0
            h, mi, se = (float(x) if x else 0.0 for x in m.groups())
            return h * 60 + mi + se / 60
        if ":" in v:
            parts = v.split(":")
            try:
                return float(parts[0]) + float(parts[1]) / 60
            except ValueError:
                return 0.0
    return num(v)


def norm_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    n = n.replace("-", " ").replace(".", "").replace("'", "")
    n = re.sub(r"[^a-z ]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return " ".join(n.split())


def to_et_date(raw) -> date | None:
    if not raw:
        return None
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.tz_convert(ET).date()


def to_et(raw):
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    return ts.tz_convert(ET) if not pd.isna(ts) else None


def book_line(x: float) -> float:
    """Nearest value ending in .5, like a sportsbook prop."""
    return round(x - 0.5) + 0.5


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fmt_spread(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return "PK" if abs(v) < 0.05 else f"{v:+.1f}"


def ml_to_prob(ml) -> float | None:
    ml = num(ml, default=float("nan"))
    if math.isnan(ml) or ml == 0:
        return None
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def flatten_stats(obj, out: dict | None = None) -> dict[str, float]:
    """Collect every {name|abbreviation: value} pair anywhere in an ESPN stats payload (lower-cased keys)."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        val = obj.get("value", obj.get("displayValue"))
        if val is not None and not isinstance(val, (dict, list)):
            for k in ("name", "abbreviation"):
                key = obj.get(k)
                if isinstance(key, str) and key.lower() not in out:
                    out[key.lower()] = num(val, default=float("nan"))
        for v in obj.values():
            if isinstance(v, (dict, list)):
                flatten_stats(v, out)
    elif isinstance(obj, list):
        for v in obj:
            flatten_stats(v, out)
    return out


def pick(stats: dict[str, float], *names: str) -> float | None:
    for n in names:
        v = stats.get(n.lower())
        if v is not None and not math.isnan(v):
            return v
    return None


def inches(v) -> float | None:
    """6' 4" / 76 / '84"' / '84.0' -> inches."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else None
    s = str(v)
    m = re.match(r"\s*(\d+)'\s*(\d+(?:\.\d+)?)?", s)
    if m:
        return float(m.group(1)) * 12 + float(m.group(2) or 0)
    f = num(s.replace('"', ""), default=float("nan"))
    return f if not math.isnan(f) and f > 0 else None
