"""UFC round betting, method props and exact-outcome probabilities, priced against an implied market.

Model: P(A wins) from the fight model; how A wins splits into KO/TKO-DQ / Submission / Decision (fighter's own
win methods blended with weight-class rates, see ufc.predict_bout); finishes spread over rounds using the
rounds of the fighter's own UFC stoppages blended with UFC-wide round priors.

Market: ESPN's free feed has no method / round prices, so the reference is market-implied: the vig-free
moneyline x weight-class outcome rates, with a typical 15% prop hold added. A fighter-specific edge over that
earns 🟢 BET.
"""

from __future__ import annotations

from .common import ml_to_prob
from .decide import american, verdict
from .ufc import _prior

ROUND_PRIOR = {3: [0.47, 0.31, 0.22], 5: [0.38, 0.26, 0.17, 0.11, 0.08]}  # share of finishes by round (UFC)
DEC_SPLIT = {"Unanimous": 0.66, "Split": 0.28, "Majority": 0.06}           # UFC decisions by type
PROP_HOLD = 1.15     # method / round props carry ~15% hold
MIN_PROP_P = 0.06    # ignore extreme longshots


def _round_dist(prof: dict, rounds: int) -> list[float]:
    prior = ROUND_PRIOR[5 if rounds == 5 else 3]
    fr = [r for r in (prof.get("finish_rounds") or []) if 1 <= r <= rounds]
    if not fr:
        return prior
    own = [fr.count(i + 1) / len(fr) for i in range(rounds)]
    w = len(fr) / (len(fr) + 4)
    d = [w * o + (1 - w) * p for o, p in zip(own, prior)]
    t = sum(d)
    return [x / t for x in d]


def breakdown(bout: dict, fa: dict, fb: dict, pred: dict) -> dict:
    rounds = bout.get("rounds") or 3
    p_a = pred["p_a"]
    ma, mb = pred["methods_a"], pred["methods_b"]  # (KO, SUB, DEC) shares of each fighter's wins
    ra, rb = _round_dist(fa, rounds), _round_dist(fb, rounds)
    fin_a, fin_b = p_a * (ma[0] + ma[1]), (1 - p_a) * (mb[0] + mb[1])
    out = {
        "rounds": rounds,
        "method": {
            "KO/TKO/DQ": (p_a * ma[0], (1 - p_a) * mb[0]),
            "Submission": (p_a * ma[1], (1 - p_a) * mb[1]),
            "Decision": (p_a * ma[2], (1 - p_a) * mb[2]),
        },
        "round": {f"Round {i + 1}": (fin_a * ra[i], fin_b * rb[i]) for i in range(rounds)},
        "distance": p_a * ma[2] + (1 - p_a) * mb[2],
    }
    out["inside"] = 1 - out["distance"]
    out["dec_types"] = {k: out["distance"] * v for k, v in DEC_SPLIT.items()}

    # implied market: vig-free moneyline x weight-class outcome rates
    mla, mlb = ml_to_prob(bout["fighters"][0].get("ml")), ml_to_prob(bout["fighters"][1].get("ml"))
    q = (mla / (mla + mlb)) if mla and mlb else None
    out["market_q"] = q
    pr = _prior(pred.get("weight_class", ""))
    rp = ROUND_PRIOR[5 if rounds == 5 else 3]
    imp = {}
    if q is not None:
        for i, k in enumerate(("KO/TKO/DQ", "Submission", "Decision")):
            imp[k] = (q * pr[i], (1 - q) * pr[i])
        for i in range(rounds):
            imp[f"Round {i + 1}"] = (q * (pr[0] + pr[1]) * rp[i], (1 - q) * (pr[0] + pr[1]) * rp[i])
    out["implied"] = imp

    def judge(p: float, imp_p: float | None, name: str, what: str) -> dict:
        if imp_p is None:
            return verdict(False, f"No moneyline to anchor the market — fair {american(p)}", f"{name} {what}", p)
        be = min(imp_p * PROP_HOLD, 0.95)
        if p >= MIN_PROP_P and p - be >= 0.02:
            return verdict(True, f"{p * 100:.0f}% vs {be * 100:.0f}% implied (ML × class rates + hold)",
                           f"{name} {what}", p)
        return verdict(False, f"{p * 100:.0f}% vs {be * 100:.0f}% implied — thin edge", f"{name} {what}", p)

    la, lb = fa["name"].split()[-1], fb["name"].split()[-1]
    rows = []
    for k in ("KO/TKO/DQ", "Submission", "Decision", *[f"Round {i + 1}" for i in range(rounds)]):
        pa_, pb_ = out["method"].get(k) or out["round"][k]
        ia, ib = imp.get(k, (None, None))
        what = f"by {k}" if not k.startswith("Round") else f"in {k}"
        rows.append({"outcome": k, "a": pa_, "b": pb_, "fair_a": american(pa_), "fair_b": american(pb_),
                     "v_a": judge(pa_, ia, la, what), "v_b": judge(pb_, ib, lb, what)})
    out["rows"] = rows
    bets = [r[s] for r in rows for s in ("v_a", "v_b") if r[s]["verdict"] == "BET"]
    out["best"] = max(bets, key=lambda v: v["p"]) if bets else None
    return out
