"""Alternate lines (alt ladders) and the automated BET / PASS decision engine.

BET  — the model's edge clears the market's safety threshold (points/goals/yards edge AND hit probability at
       least 2 percentage points above break-even), or the last-10 record backs a probability edge.
PASS — the edge is thin, the line sits right on the projection (trap zone / key number), the sample is small,
       variance is high, or the player is questionable.

Break-evens: -110 for game spreads/totals, the posted moneyline for moneylines, -115 for pick'em props.
Alt lines have no posted prices in any free feed, so they get the model's fair odds and a stricter rule
(model and last-10 both clearly above 60%).
"""

from __future__ import annotations

import math

import numpy as np

from .common import book_line, norm_cdf

BE_STD = 110 / 210          # -110
BE_PROP = 115 / 215         # -115
PP_EDGE = 0.02              # probability must clear break-even by 2 percentage points

COUNT_MARKETS = {("hockey", "G"), ("hockey", "A"), ("hockey", "PTS"), ("hockey", "SOG"), ("football", "RECS")}
PROP_SD_FLOOR = {"PTS": 4.5, "REB": 2.2, "AST": 1.8, "PRA": 6.0, "PASS": 45.0, "RUSH": 18.0, "REC": 18.0}

# Minimum projection-vs-line gap (in the market's units) for a BET.
PROP_EDGE = {
    "basketball": {"PTS": 2.0, "REB": 1.0, "AST": 1.0, "PRA": 3.0},
    "football": {"PASS": 15.0, "RUSH": 8.0, "REC": 8.0, "RECS": 0.8},
    "hockey": {"G": 0.15, "A": 0.15, "SOG": 0.6, "PTS": 0.2},
}
# Alt tiers: points-type markets +-2/4/6, count markets +-1/2/3. Yards use +-10/20/30 (a 2-yard alt is not a
# real ladder); hockey goals/assists/points ladder 0.5 -> 3.5 because they start at 0.5.
ALT_STEPS = {
    ("basketball", "PTS"): (2, 4, 6), ("basketball", "PRA"): (2, 4, 6),
    ("basketball", "REB"): (1, 2, 3), ("basketball", "AST"): (1, 2, 3),
    ("football", "PASS"): (10, 20, 30), ("football", "RUSH"): (10, 20, 30), ("football", "REC"): (10, 20, 30),
    ("football", "RECS"): (1, 2, 3),
    ("hockey", "SOG"): (1, 2, 3), ("hockey", "G"): (1, 2, 3), ("hockey", "A"): (1, 2, 3), ("hockey", "PTS"): (1, 2, 3),
}
UNIT = {"PTS": "pts", "REB": "reb", "AST": "ast", "PRA": "PRA", "PASS": "yds", "RUSH": "yds", "REC": "yds",
        "RECS": "rec", "G": "goals", "A": "ast", "SOG": "shots"}
LADDER_UP_ONLY = {("hockey", "G"), ("hockey", "A"), ("hockey", "PTS")}  # these start at 0.5: only + tiers
GAME_EDGE = {"basketball": 2.0, "football": 2.0, "hockey": 0.3, "soccer": 0.3}  # spread / total gap for a BET
GAME_ALT_STEPS = {"basketball": (2, 4, 6), "football": (2, 4, 6), "hockey": (1, 2, 3), "soccer": (1, 2, 3)}
TOTAL_SD = {"NBA": 17.0, "NCAAB": 15.0, "NFL": 13.0}
MARGIN_SD = {"NBA": 12.0, "NCAAB": 11.0, "NFL": 13.5}
NFL_KEY_NUMBERS = {3.0, 7.0}


def american(p: float) -> str:
    if p <= 0.001 or p >= 0.999:
        return "—"
    return f"-{100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def _poisson_cdf(k: int, lam: float) -> float:
    return sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(max(k, -1) + 1))


def prop_p_over(kind: str, m: str, proj: float, hist: list[float], line: float) -> float:
    """P(stat > line): Poisson for count markets, normal (sd from the last 10, floored) otherwise."""
    if (kind, m) in COUNT_MARKETS:
        return 1 - _poisson_cdf(int(math.floor(line)), max(proj, 0.01))
    sd = max(float(np.std(hist)) if len(hist) >= 4 else 0.0, PROP_SD_FLOOR.get(m, 0.25 * max(proj, 1)),
             0.2 * max(proj, 1))
    return 1 - norm_cdf((line - proj) / sd)


def _hit(hist: list[float], line: float, over: bool) -> float | None:
    if not hist:
        return None
    return float(np.mean([(x > line) if over else (x < line) for x in hist[:10]]))


def verdict(bet: bool, reason: str, pick: str = "", p: float | None = None, edge: str = "") -> dict:
    return {"verdict": "BET" if bet else "PASS", "reason": reason, "pick": pick, "p": p, "edge": edge}


def _be(american: float | None, default: float) -> float:
    if american is None:
        return default
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def no_line_verdict(kind: str, m: str, label: str, proj: float, line: float, hist: list[float],
                    season: float | None) -> dict:
    """No sportsbook line posted: show the model's lean against a season-average line instead of BET/PASS."""
    over = proj > line
    p_over = prop_p_over(kind, m, proj, hist, line)
    p = p_over if over else 1 - p_over
    gap = abs(proj - line)
    thr = PROP_EDGE.get(kind, {}).get(m, 1.0)
    unit = UNIT.get(m, "")
    base = f"season avg {season:.1f}" if season is not None else "recent average"
    if gap < 0.5 * thr:
        reason = f"No sportsbook line — no clear lean (proj {proj:.1f} vs {base})"
        pick = ""
    else:
        reason = f"No sportsbook line — model leans {'Over' if over else 'Under'} ({gap:.1f} {unit} vs {base})"
        pick = f"Lean {'Over' if over else 'Under'} {line:g} {label}"
    return {"verdict": "NO LINE", "reason": reason, "pick": pick, "p": p, "edge": f"{gap:.1f} {unit}",
            "lean": gap >= 0.5 * thr}


# --------------------------------------------------------------------------
# Player props
# --------------------------------------------------------------------------
def prop_decision(kind: str, m: str, label: str, proj: float, line: float, hist: list[float], posted: bool,
                  status_cls: str = "ACTIVE", over_price: float | None = None,
                  under_price: float | None = None) -> dict:
    over = proj > line
    p_over = prop_p_over(kind, m, proj, hist, line)
    p = p_over if over else 1 - p_over
    side = "Over" if over else "Under"
    pick = f"{side} {line:g} {label}"
    gap = abs(proj - line)
    thr = PROP_EDGE.get(kind, {}).get(m, 1.0)
    hit = _hit(hist, line, over)
    be = _be(over_price if over else under_price, BE_PROP)  # real price when ESPN lists one
    pp = p - be
    unit = UNIT.get(m, "")
    edge_txt = f"{gap:.1f} {unit} edge · {p * 100:.0f}% to hit"
    if status_cls == "Q":
        return verdict(False, "Questionable status", pick, p, edge_txt)
    if len(hist) < 5:
        return verdict(False, f"Small sample ({len(hist)} games)", pick, p, edge_txt)
    if side == "Under" and (kind, m) in COUNT_MARKETS and line < 1:
        return verdict(False, "Trap zone: unders at 0.5 are priced as heavy favorites", pick, p, edge_txt)
    sd = float(np.std(hist)) if len(hist) >= 4 else 0.0
    if (kind, m) not in COUNT_MARKETS and proj > 0 and sd / max(proj, 1e-9) > 0.6:
        return verdict(False, "High variance in recent games", pick, p, edge_txt)
    if gap < 0.5 * thr:
        return verdict(False, "Trap zone: line sits on the projection", pick, p, edge_txt)
    src = "" if posted else " (implied line)"
    if gap >= thr and pp >= PP_EDGE:
        return verdict(True, f"{edge_txt}{src}", pick, p, edge_txt)
    if hit is not None and hit >= 0.7 and pp >= PP_EDGE:
        return verdict(True, f"{hit * 100:.0f}% L10 hit rate · {p * 100:.0f}% to hit{src}", pick, p, edge_txt)
    return verdict(False, f"Thin edge ({gap:.1f} {unit}, needs {thr:g})", pick, p, edge_txt)


ALT_MIN_P, ALT_MAX_P = 0.62, 0.85  # alt BET band: likely enough, but not so likely the price is worthless


def prop_alt_ladder(kind: str, m: str, proj: float, hist: list[float], main_line: float) -> list[dict]:
    """Main line plus three alt tiers each way. Each row: line, P(over), fair odds, L10 over-hit, BET/PASS."""
    steps = ALT_STEPS.get((kind, m), (1, 2, 3))
    lines = sorted({main_line + s * d for s in (-1, 1) for d in steps} | {main_line})
    rows = []
    for ln in lines:
        if ln <= 0:
            continue
        p = prop_p_over(kind, m, proj, hist, ln)
        hit = _hit(hist, ln, True)
        bet = ALT_MIN_P <= p <= ALT_MAX_P and hit is not None and hit >= 0.6 and len(hist) >= 5
        rows.append({"line": ln, "offset": ln - main_line, "p_over": p, "p": p, "fair": american(p), "hit": hit,
                     "verdict": "BET" if bet else "PASS", "main": ln == main_line,
                     "why": ("price too short" if p > ALT_MAX_P else "")})
    return rows


def shift_line(kind: str, m: str, main_line: float, tier: int) -> float:
    """Alt line for a tier (-3..+3) relative to the main line."""
    if tier == 0:
        return main_line
    steps = ALT_STEPS.get((kind, m), (1, 2, 3))
    ln = main_line + (1 if tier > 0 else -1) * steps[abs(tier) - 1]
    return ln if ln > 0 else main_line


# --------------------------------------------------------------------------
# Game lines
# --------------------------------------------------------------------------
def _poisson_grid(lh: float, la: float, n: int = 12):
    for i in range(n):
        pi = math.exp(-lh) * lh ** i / math.factorial(i)
        for j in range(n):
            yield i, j, pi * math.exp(-la) * la ** j / math.factorial(j)


def cover_prob(lg_key: str, kind: str, p: dict, home_line: float) -> float:
    """P(home covers home_line), i.e. home margin + home_line > 0 (pushes split)."""
    if kind in ("hockey", "soccer"):
        win = push = 0.0
        for i, j, q in _poisson_grid(p["pts_home"], p["pts_away"]):
            v = i - j + home_line
            if v > 0:
                win += q
            elif v == 0:
                push += q
        return win + push / 2
    mu = p["pts_home"] - p["pts_away"]
    return 1 - norm_cdf((-home_line - mu) / MARGIN_SD.get(lg_key, 12.0))


def over_prob(lg_key: str, kind: str, mean_total: float, line: float, lh: float = 0, la: float = 0,
              sd: float | None = None) -> float:
    if kind in ("hockey", "soccer"):
        return sum(q for i, j, q in _poisson_grid(lh, la) if i + j > line)
    return 1 - norm_cdf((line - mean_total) / (sd or TOTAL_SD.get(lg_key, 13.0)))


def game_decision(lg_key: str, kind: str, g: dict, p: dict, team_gp: int) -> dict:
    """Best BET among spread / total / moneyline (3-way for soccer) vs ESPN's market, else PASS with a reason."""
    if not p:
        return verdict(False, "No projection")
    o, h, a = g["odds"], g["home"], g["away"]
    thr = GAME_EDGE.get(kind, 2.0)
    cands = []  # (bet?, pp_edge, pick, p, edge_txt, pass_reason)
    if o.get("spread_home") is not None and kind not in ("hockey", "soccer"):
        s = o["spread_home"]
        p_h = cover_prob(lg_key, kind, p, s)
        model_h = p["pts_home"] - p["pts_away"]
        gap_h = model_h + s  # >0: home beats the number
        side_home = gap_h > 0
        pc = p_h if side_home else 1 - p_h
        pick = f'{h["abbr"]} {s:+g}' if side_home else f'{a["abbr"]} {-s:+g}'
        reason = "Thin edge"
        if lg_key == "NFL" and abs(s) in NFL_KEY_NUMBERS and abs(gap_h) < 3:
            reason = f"Trap zone: key number {abs(s):g}"
        bet = abs(gap_h) >= thr and pc - BE_STD >= PP_EDGE and not reason.startswith("Trap")
        cands.append((bet, pc - BE_STD, pick, pc, f"{abs(gap_h):.1f} pts vs the spread · {pc * 100:.0f}% to cover",
                      reason))
    if o.get("total") is not None:
        t = o["total"]
        po = over_prob(lg_key, kind, p["total"], t, p["pts_home"], p["pts_away"])
        over = p["total"] > t
        pc = po if over else 1 - po
        gap = abs(p["total"] - t)
        unit = "goals" if kind in ("hockey", "soccer") else "pts"
        bet = gap >= thr and pc - BE_STD >= PP_EDGE
        cands.append((bet, pc - BE_STD, f'{"Over" if over else "Under"} {t:g}', pc,
                      f"{gap:.1f} {unit} vs the total · {pc * 100:.0f}% to hit", "Thin edge"))
    mls = [("home", h["abbr"], o.get("ml_home")), ("away", a["abbr"], o.get("ml_away"))]
    if kind == "soccer":
        mls.append(("draw", "Draw", o.get("ml_draw")))
    probs = {"home": p.get("home_win_prob"), "away": p.get("away_win_prob", 1 - p.get("home_win_prob", 0.5)),
             "draw": p.get("draw_prob")}
    for side, name, ml in mls:
        if ml is None or probs.get(side) is None:
            continue
        be = 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)
        pc = probs[side]
        bet = pc - be >= PP_EDGE and pc >= 0.35
        cands.append((bet, pc - be, f"{name} ML" if side != "draw" else "Draw", pc,
                      f"{pc * 100:.0f}% vs {be * 100:.0f}% break-even at {ml:+.0f}", "Thin edge"))
    if not cands:
        return verdict(False, "No market line posted")
    if team_gp < 5:
        best = max(cands, key=lambda c: c[1])
        return verdict(False, f"Small sample ({team_gp} games)", best[2], best[3], best[4])
    bets = [c for c in cands if c[0]]
    if bets:
        best = max(bets, key=lambda c: c[1])
        return verdict(True, best[4], best[2], best[3], best[4])
    best = max(cands, key=lambda c: c[1])
    return verdict(False, f"{best[5]} ({best[4]})", best[2], best[3], best[4])


def game_alt_ladders(lg_key: str, kind: str, g: dict, p: dict) -> dict:
    """Alt spread (home side), alt total and team totals around the market (or projected) numbers."""
    steps = GAME_ALT_STEPS.get(kind, (2, 4, 6))
    o = g["odds"]
    goals = kind in ("hockey", "soccer")
    out = {}
    base_spread = o.get("spread_home") if o.get("spread_home") is not None else -round(
        (p["pts_home"] - p["pts_away"]) * 2) / 2
    if goals:
        base_spread = -1.5 if p["pts_home"] >= p["pts_away"] else 1.5
    sp_rows = []
    for ln in sorted({base_spread + s * d for s in (-1, 1) for d in steps} | {base_spread}, reverse=True):
        if goals and abs(ln) < 0.5:
            continue
        pc = cover_prob(lg_key, kind, p, ln)
        sp_rows.append({"line": ln, "p": pc, "fair": american(pc), "main": ln == base_spread})
    out["spread"] = sp_rows
    base_total = o.get("total") if o.get("total") is not None else book_line(p["total"])
    if goals:
        tot_lines = [x + 0.5 for x in range(0, 7)]
    else:
        tot_lines = sorted({base_total + s * d for s in (-1, 1) for d in steps} | {base_total})
    out["total"] = [{"line": ln, "p": over_prob(lg_key, kind, p["total"], ln, p["pts_home"], p["pts_away"]),
                     "main": ln == base_total} for ln in tot_lines]
    for r in out["total"]:
        r["fair"] = american(r["p"])
    tt = {}
    for side, lam in (("home", p["pts_home"]), ("away", p["pts_away"])):
        if goals:
            lines = [0.5, 1.5, 2.5, 3.5]
            rows = [{"line": ln, "p": 1 - _poisson_cdf(int(ln), lam), "main": ln == 1.5} for ln in lines]
        else:
            main = book_line(lam)
            sd = TOTAL_SD.get(lg_key, 13.0) / math.sqrt(2)
            rows = [{"line": ln, "p": 1 - norm_cdf((ln - lam) / sd), "main": ln == main}
                    for ln in sorted({main + s * d for s in (-1, 1) for d in steps} | {main})]
        for r in rows:
            r["fair"] = american(r["p"])
        tt[side] = rows
    out["team_totals"] = tt
    return out


# --------------------------------------------------------------------------
# UFC
# --------------------------------------------------------------------------
def fight_decision(name: str, p: float, ml: float | None, model_only: float) -> dict:
    if ml is None:
        return verdict(False, "No moneyline posted", f"{name} to win", p)
    be = 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)
    pp = p - be
    if pp >= PP_EDGE and p >= 0.35 and abs(model_only - p) < 0.25:
        return verdict(True, f"{p * 100:.0f}% vs {be * 100:.0f}% break-even at {ml:+.0f}", f"{name} to win", p,
                       f"{pp * 100:+.0f}pp")
    reason = "High model/market disagreement" if abs(model_only - p) >= 0.25 else "Thin edge"
    return verdict(False, reason, f"{name} to win", p, f"{pp * 100:+.0f}pp")


# --------------------------------------------------------------------------
# Quarter / half / period: implied market lines from the full-game line
# --------------------------------------------------------------------------
def implied_period_lines(odds: dict, share: float) -> dict:
    """ESPN only posts full-game lines. Scale them by the segment's share of full-game scoring (from both
    teams' real period splits): e.g. FG total 224.0 x 0.51 = implied 1H total 114.2."""
    out = {"total": None, "spread_home": None, "share": share}
    if odds.get("total") is not None:
        out["total"] = odds["total"] * share
    if odds.get("spread_home") is not None:
        out["spread_home"] = odds["spread_home"] * share
    return out


def period_decision(lg_key: str, kind: str, g: dict, q: dict, imp: dict, seg: str) -> dict:
    """BET / PASS for a segment against its implied market line (priced at the standard -110)."""
    share = max(imp.get("share") or 0.25, 0.1)
    h, a = g["home"]["abbr"], g["away"]["abbr"]
    thr = GAME_EDGE.get(kind, 2.0) * math.sqrt(share)
    goals = kind in ("hockey", "soccer")
    cands = []
    if imp.get("spread_home") is not None and not goals:
        s = imp["spread_home"]
        mu = q["pts_home"] - q["pts_away"]
        sd = MARGIN_SD.get(lg_key, 12.0) * math.sqrt(share)
        p_h = 1 - norm_cdf((-s - mu) / sd)
        gap = mu + s
        home = gap > 0
        pc = p_h if home else 1 - p_h
        pick = f"{seg} {h} {s:+.1f}" if home else f"{seg} {a} {-s:+.1f}"
        cands.append((abs(gap) >= thr and pc - BE_STD >= PP_EDGE, pc - BE_STD, pick, pc,
                      f"+{abs(gap):.1f} pts vs implied spread · {pc * 100:.0f}% to cover"))
    if imp.get("total") is not None:
        t = imp["total"]
        if goals:
            po = sum(qq for i, j, qq in _poisson_grid(q["pts_home"], q["pts_away"]) if i + j > t)
        else:
            sd = TOTAL_SD.get(lg_key, 13.0) * math.sqrt(share)
            po = 1 - norm_cdf((t - q["total"]) / sd)
        over = q["total"] > t
        pc = po if over else 1 - po
        gap = abs(q["total"] - t)
        unit = "goals" if goals else "pts"
        cands.append((gap >= (0.3 * math.sqrt(share) if goals else thr) and pc - BE_STD >= PP_EDGE, pc - BE_STD,
                      f'{seg} {"Over" if over else "Under"} {t:.1f}', pc,
                      f"+{gap:.1f} {unit} vs implied total · {pc * 100:.0f}% to hit"))
    if not cands:
        return verdict(False, f"No full-game line to derive {seg} markets from")
    bets = [c for c in cands if c[0]]
    best = max(bets or cands, key=lambda c: c[1])
    if bets:
        return verdict(True, best[4] + " (implied line, -110)", best[2], best[3], best[4])
    return verdict(False, f"Thin edge ({best[4]})", best[2], best[3], best[4])


# --------------------------------------------------------------------------
# Implied consensus prop line (no sportsbook / DFS line posted)
# --------------------------------------------------------------------------
def implied_prop_line(kind: str, m: str, hist: list[float], opp_factor: float = 1.0) -> float | None:
    """What a book would hang without a posted line: 50% last-5 + 50% last-10 average, scaled by the opponent's
    defense for that stat (points / goals allowed vs league; neutral where ESPN has no allowance data),
    rounded to a .5 line. Needs 3+ games."""
    if len(hist) < 3:
        return None
    base = 0.5 * float(np.mean(hist[:5])) + 0.5 * float(np.mean(hist[:10]))
    adj = base * (opp_factor or 1.0)
    if (kind, m) in {("hockey", "G"), ("hockey", "A"), ("hockey", "PTS")}:
        return 0.5 if adj < 1.0 else 1.5
    return max(0.5, round(adj - 0.5) + 0.5)
