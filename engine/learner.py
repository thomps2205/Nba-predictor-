"""A small learning model for prop legs, trained on graded legs: yours (tracked parlays, slips, hand-entered)
and picks imported from other AI / tipster sources.

Every leg is described by the same handful of features, computed from the player's game log *before* that
game (so the result never leaks in): season and last-10 hit rate at the line, how high the line sits versus his
average, the current run, tonight's projection margin, the market, over/under, and how many imported AI sources
agree. A ridge-regularised logistic regression learns how those turn into real hits. It only switches on when it
beats the builder's own estimate on held-out legs (5-fold cross-validation), and even then it is blended in.
"""

from __future__ import annotations

import math

import numpy as np

MARKET_SET = ["PTS", "REB", "AST", "PRA", "SOG", "PASS", "RUSH", "REC", "RECS", "G", "A"]
FEATURES = ["hit rate (season)", "hit rate (last 10)", "line vs average", "current run", "projection margin",
            "has projection", "under", "AI sources agreeing", "imported pick"] + [f"market {m}" for m in MARKET_SET]
MIN_TRAIN = 60       # graded legs with features before the model may switch on
MAX_WEIGHT = 0.7     # never more than 70% model / 30% builder
L2 = 0.02          # ridge penalty per standardized feature


def feats_from_vals(vals: list[float], t: float, side: str, m: str, proj: float | None = None,
                    agree: int = 0, external: bool = False, line: float | None = None) -> list[float] | None:
    """vals = the player's stat in his games before this one, newest first. Over legs: hit = value >= t.
    Under legs: hit = value < line."""
    if len(vals) < 3:
        return None
    under = side == "under"
    hit = [(v < line) if under else (v >= t) for v in vals]
    n = len(vals)
    hr = (sum(hit) + 1) / (n + 2)
    hr10 = (sum(hit[:10]) + 1) / (min(n, 10) + 2)
    avg = float(np.mean(vals)) or 1e-6
    ref = line if under else t
    ratio = (avg / max(ref, 1e-6)) if under else (ref / avg)
    run = next((i for i, h in enumerate(hit) if not h), n)
    sd = float(np.std(vals[:10], ddof=1)) if n > 2 else 1.0
    sd = max(sd, 0.15 * max(avg, 1e-6), 0.8)
    if proj is not None and proj == proj:
        margin = ((ref - proj) if under else (proj - (t - 0.5))) / sd
        has_p = 1.0
    else:
        margin, has_p = 0.0, 0.0
    onehot = [1.0 if m == x else 0.0 for x in MARKET_SET]
    return [hr, hr10, min(ratio, 3.0), math.log1p(run), max(-3.0, min(3.0, margin)), has_p, float(under),
            float(min(agree, 3)), float(external)] + onehot


def base_p(f: list[float]) -> float:
    """The builder-equivalent estimate a feature row implies: its smoothed season hit rate."""
    return min(0.97, f[0])


# --------------------------------------------------------------------------
# Logistic regression (numpy)
# --------------------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.02, 0.98)
    return np.log(p / (1 - p))


def _fit(X: np.ndarray, y: np.ndarray, l2: float = L2, iters: int = 500) -> dict:
    """Logistic regression on top of the builder: logit(p) = logit(builder) + b + z.w, so with no evidence it
    reproduces the builder exactly and only learns corrections (ridge penalty keeps them small)."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    off = _logit(np.array([base_p(list(f)) for f in X]))
    w, b = np.zeros(Z.shape[1]), 0.0
    lr = 0.2
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * (g.mean() + l2 * b)
    return {"w": w.tolist(), "b": float(b), "mu": mu.tolist(), "sd": sd.tolist()}


def _predict(model: dict, X: np.ndarray) -> np.ndarray:
    Z = (X - np.array(model["mu"])) / np.array(model["sd"])
    off = _logit(np.array([base_p(list(f)) for f in X]))
    return 1 / (1 + np.exp(-(off + Z @ np.array(model["w"]) + model["b"])))


def train(rows: list[dict]) -> dict:
    """rows: graded legs with "feat" and result hit/miss. Returns a status dict (and the model when active)."""
    data = [(r["feat"], 1.0 if r["result"] == "hit" else 0.0) for r in rows
            if r.get("feat") and r.get("result") in ("hit", "miss") and len(r["feat"]) == len(FEATURES)]
    n = len(data)
    out = {"n": n, "need": MIN_TRAIN, "active": False, "weight": 0.0}
    if n < MIN_TRAIN or len({y for _, y in data}) < 2:
        return out
    X = np.array([f for f, _ in data], dtype=float)
    y = np.array([v for _, v in data], dtype=float)
    # 5-fold CV: model vs the builder-equivalent estimate (Brier score, lower is better)
    idx = np.arange(n)
    rng = np.random.default_rng(7)
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    pm, pb = np.zeros(n), np.array([base_p(list(f)) for f in X])
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        if len(set(y[tr])) < 2:
            pm[te] = pb[te]
            continue
        pm[te] = _predict(_fit(X[tr], y[tr]), X[te])
    brier_m = float(np.mean((pm - y) ** 2))
    brier_b = float(np.mean((pb - y) ** 2))
    model = _fit(X, y)
    gain = (brier_b - brier_m) / brier_b if brier_b > 0 else 0.0
    weight = 0.0 if gain <= 0 else min(MAX_WEIGHT, n / 300) * min(1.0, gain / 0.05)
    std_w = np.array(model["w"])
    top = sorted(zip(FEATURES, std_w), key=lambda kv: -abs(kv[1]))[:5]
    out.update(active=weight > 0, weight=round(weight, 3), brier_model=brier_m, brier_base=brier_b,
               gain=gain, model=model, top=[(k, float(v)) for k, v in top],
               hit_rate=float(y.mean()))
    return out


def predict(status: dict, feat: list[float] | None, p_builder: float) -> float:
    """Blend the learned chance into the builder's chance (unchanged when the model isn't active)."""
    if not status.get("active") or not feat or len(feat) != len(FEATURES):
        return p_builder
    p_m = float(_predict(status["model"], np.array([feat], dtype=float))[0])
    w = status["weight"]
    return min(0.97, max(0.01, (1 - w) * p_builder + w * p_m))
