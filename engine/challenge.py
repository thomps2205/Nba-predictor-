"""Bankroll rollover challenge ("$50 to $10k"): planning math, the daily challenge pick, and a saved log.

Every bet rolls (part of) the bankroll into the next one, so the challenge only survives if every bet wins.
The planner is honest about that: success chance = product of each bet's hit probability.
"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import date
from itertools import combinations

from .common import CACHE_DIR
from .parlays import Leg, dec_to_american

STORE = CACHE_DIR / "challenges"
STORE.mkdir(parents=True, exist_ok=True)
MAX_LEGS = 2          # challenge bets: a single or a 2-leg parlay (like the posted examples)
MIN_DEC, MAX_DEC = 1.20, 3.00


# --------------------------------------------------------------------------
# Planning math
# --------------------------------------------------------------------------
def per_bet_multiplier(start: float, target: float, bets: int, rollover: float = 1.0) -> float:
    """Decimal odds every bet needs so start grows to target in `bets` wins (only `rollover` of the bankroll
    is staked each time)."""
    if start <= 0 or target <= start or bets <= 0:
        return 1.0
    growth = (target / start) ** (1 / bets)          # bankroll multiple per bet
    return 1 + (growth - 1) / max(rollover, 1e-6)     # stake share s: B -> B (1 + s (d - 1))


def bets_needed(start: float, target: float, dec: float, rollover: float = 1.0) -> int:
    g = 1 + rollover * (dec - 1)
    if start <= 0 or target <= start or g <= 1:
        return 0
    return math.ceil(math.log(target / start) / math.log(g))


def american(dec: float) -> str:
    return dec_to_american(dec) if dec > 1.0001 else "—"


def ladder(start: float, target: float, dec: float, rollover: float = 1.0, max_rows: int = 60) -> list[dict]:
    rows, b, k = [], start, 0
    while b < target and k < max_rows:
        k += 1
        stake = b * rollover
        win = stake * (dec - 1)
        rows.append({"n": k, "bank": b, "stake": stake, "win": win, "after": b + win})
        b += win
    return rows


# --------------------------------------------------------------------------
# Daily pick: the most likely single / 2-leg parlay paying at least the needed odds
# --------------------------------------------------------------------------
def candidates(legs: list[Leg], need_dec: float, max_dec: float = MAX_DEC) -> list[dict]:
    """Singles and 2-leg parlays (different games) priced >= need_dec, ranked by hit chance, then value.
    Only legs the model rates at or above the market (edge >= 0) are used when there are enough of them."""
    pool = [l for l in legs if l.dec > 1.01 and 0 < l.prob < 1]
    good = [l for l in pool if l.edge >= 0]
    if len(good) >= 4:
        pool = good
    pool = sorted(pool, key=lambda l: -l.prob)[:60]
    need = max(need_dec, MIN_DEC)
    out = []
    for l in pool:
        if need <= l.dec <= max_dec:
            out.append({"legs": [l], "prob": l.prob, "dec": l.dec})
    for a, b in combinations(pool, 2):
        if a.game_id == b.game_id:
            continue
        d = a.dec * b.dec
        if need <= d <= max_dec:
            out.append({"legs": [a, b], "prob": a.prob * b.prob, "dec": d})
    for c in out:
        c["ev"] = c["prob"] * c["dec"] - 1
        c["implied"] = 1 / c["dec"]
    out.sort(key=lambda c: (-c["prob"], -c["ev"]))
    seen, uniq = set(), []
    for c in out:  # don't repeat the same game across the top options
        games = {l.game_id for l in c["legs"]}
        if games & seen and len(uniq) >= 1:
            continue
        uniq.append(c)
        seen |= games
        if len(uniq) == 3:
            break
    return uniq


def leg_text(l: Leg) -> str:
    return f"{l.pick} ({l.market}, {l.sport})"


# --------------------------------------------------------------------------
# Saved challenges (JSON per challenge name)
# --------------------------------------------------------------------------
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:40]


def path(name: str):
    return STORE / f"{slug(name)}.json"


def load(name: str) -> dict | None:
    p = path(name)
    if not slug(name) or not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save(ch: dict) -> None:
    p = path(ch["name"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(ch, indent=1))
    tmp.replace(p)


def new(name: str, start: float, target: float, days: int, rollover: float) -> dict:
    return {"name": name.strip(), "start": float(start), "target": float(target), "days": int(days),
            "rollover": float(rollover), "created": date.today().isoformat(), "bets": [], "v": 1}


def bankroll(ch: dict) -> float:
    b = ch["start"]
    for bet in ch["bets"]:
        if bet["result"] == "won":
            b += bet["stake"] * (bet["dec"] - 1)
        elif bet["result"] == "lost":
            b -= bet["stake"]
    return b


def pending(ch: dict) -> dict | None:
    return next((b for b in ch["bets"] if b["result"] == "pending"), None)


def status(ch: dict) -> str:
    b = bankroll(ch)
    if b >= ch["target"]:
        return "complete"
    if any(x["result"] == "lost" for x in ch["bets"]) and b < 1:
        return "busted"
    return "active"


def record(ch: dict, desc: str, dec: float, prob: float | None, legs: list[str]) -> dict:
    stake = round(bankroll(ch) * ch["rollover"], 2)
    bet = {"n": len(ch["bets"]) + 1, "date": date.today().isoformat(), "desc": desc, "legs": legs,
           "dec": float(dec), "prob": prob, "stake": stake, "result": "pending", "ts": time.time()}
    ch["bets"].append(bet)
    return bet


def settle(ch: dict, result: str) -> None:
    b = pending(ch)
    if b:
        b["result"] = result


def bets_left(ch: dict) -> int:
    """Remaining planned bets (pushes don't use up a day's bet). At least 1 while the challenge is active."""
    used = sum(1 for b in ch["bets"] if b["result"] in ("won", "lost", "pending"))
    return max(1, ch["days"] - used)


def need_dec_now(ch: dict) -> float:
    """Odds today's bet needs so the remaining bets (one per remaining day) still reach the target."""
    left = bets_left(ch) or 10
    return per_bet_multiplier(bankroll(ch), ch["target"], left, ch["rollover"])


def survival(probs: list[float]) -> float:
    out = 1.0
    for p in probs:
        out *= p
    return out
