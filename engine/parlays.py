"""Parlay builder: turn model projections into priced legs, then search leg combinations.

Leg probabilities come from the same models the app shows:
  * spreads / totals: normal margin / total around the projection (Poisson grid for NHL),
  * moneylines: model win probability (NHL incl. overtime, UFC fight model),
  * player props: normal (or Poisson for count stats) around the projection vs. the posted PrizePicks / Underdog line.
Prices: ESPN moneylines where listed; spreads/totals at the standard -110; pick'em prop legs at -115.
Legs are treated as independent, so at most one leg per game is allowed in a parlay.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .common import norm_cdf

STD_DEC = 1.909          # -110
PROP_DEC = 1.87          # ~ -115 pick'em equivalent
TOTAL_SD = {"NBA": 17.0, "NCAAB": 15.0, "NFL": 13.0}
MARGIN_SD = {"NBA": 12.0, "NCAAB": 11.0, "NFL": 13.5}
COUNT_MARKETS = {("hockey", "G"), ("hockey", "A"), ("hockey", "PTS"), ("hockey", "SOG"), ("football", "RECS")}
PROP_SD_FLOOR = {"PTS": 4.5, "REB": 2.2, "AST": 1.8, "PRA": 6.0, "PASS": 45.0, "RUSH": 18.0, "REC": 18.0}

MODEL_WEIGHT = 0.6      # leg probability = 60% model + 40% market (no-vig)
MIN_PROB = 0.55
MIN_EDGE = 0.02
MAX_POOL = 16


@dataclass
class Leg:
    sport: str
    game_id: str
    matchup: str
    pick: str          # e.g. "BOS -4.5", "Over 226.5", "LeBron James Over 24.5 PTS"
    market: str        # Spread / Total / ML / Prop / Fight
    prob: float
    dec: float
    when: str = ""
    note: str = ""
    mkt: float | None = None   # market's no-vig probability for this side (None = already blended)
    logo: str = ""             # picked team's logo / fighter photo
    logo2: str = ""            # second logo (totals show both teams)
    fallback: str = ""         # text shown if the image can't load
    photo: bool = False        # round photo (UFC) instead of a logo tile
    model: float = field(init=False)
    implied: float = field(init=False)
    edge: float = field(init=False)

    def __post_init__(self):
        self.model = self.prob
        if self.mkt is not None:  # calibrate: raw model edges are overconfident, lean partly on the market
            self.prob = MODEL_WEIGHT * self.prob + (1 - MODEL_WEIGHT) * self.mkt
        self.implied = 1 / self.dec
        self.edge = self.prob - self.implied


def american_to_dec(ml: float | None) -> float | None:
    if ml is None or ml == 0 or ml != ml:
        return None
    return 1 + (ml / 100 if ml > 0 else 100 / -ml)


def dec_to_american(dec: float) -> str:
    if dec <= 1:
        return "—"
    return f"+{(dec - 1) * 100:.0f}" if dec >= 2 else f"-{100 / (dec - 1):.0f}"


def _poisson_cdf(k: int, lam: float) -> float:
    return sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))


def _hockey_probs(lh: float, la: float, line_total: float | None) -> tuple[float, float | None]:
    p_over, p_home = 0.0, 0.0
    tie = 0.0
    for i in range(14):
        for j in range(14):
            p = math.exp(-lh) * lh ** i / math.factorial(i) * math.exp(-la) * la ** j / math.factorial(j)
            if line_total is not None and i + j > line_total:
                p_over += p
            if i > j:
                p_home += p
            elif i == j:
                tie += p
    return p_home + 0.52 * tie, (p_over if line_total is not None else None)


def _name(t: dict) -> str:
    """Friendly team name for a pick: short display name (e.g. 'Hawks', 'Duke'), else the abbreviation."""
    return t.get("short") or t["abbr"]


def game_legs(sport: str, kind: str, games: list[dict], projections: dict) -> list[Leg]:
    legs: list[Leg] = []
    for g in games:
        p = projections.get(g["id"])
        if not p or g["state"] != "pre":
            continue
        h, a, o = g["home"], g["away"], g["odds"]
        matchup = f'{_name(a)} @ {_name(h)}'
        when = g["tip"].strftime("%a %-I:%M %p") if g["tip"] is not None else ""

        def leg(pick, market, prob, dec, mkt, team=None):
            if team is not None:
                return Leg(sport, g["id"], matchup, pick, market, prob, dec, when, mkt=mkt,
                           logo=team.get("logo", ""), fallback=team["abbr"])
            return Leg(sport, g["id"], matchup, pick, market, prob, dec, when, mkt=mkt,
                       logo=a.get("logo", ""), logo2=h.get("logo", ""), fallback=f'{a["abbr"]}/{h["abbr"]}')

        cands: list[Leg] = []
        if kind == "hockey":
            win_h, p_over = _hockey_probs(p["pts_home"], p["pts_away"], o["total"])
            if p_over is not None:
                cands += [leg(f'Over {o["total"]} goals', "Total", p_over, STD_DEC, 0.5),
                          leg(f'Under {o["total"]} goals', "Total", 1 - p_over, STD_DEC, 0.5)]
        else:
            win_h = p["home_win_prob"]
            mu = -p["spread_home"]  # projected home margin
            if o["spread_home"] is not None:
                p_cover_h = 1 - norm_cdf((-o["spread_home"] - mu) / MARGIN_SD[sport])
                cands += [leg(f'{_name(h)} {o["spread_home"]:+.1f}', "Spread", p_cover_h, STD_DEC, 0.5, h),
                          leg(f'{_name(a)} {-o["spread_home"]:+.1f}', "Spread", 1 - p_cover_h, STD_DEC, 0.5, a)]
            if o["total"] is not None:
                p_over = 1 - norm_cdf((o["total"] - p["total"]) / TOTAL_SD[sport])
                cands += [leg(f'Over {o["total"]}', "Total", p_over, STD_DEC, 0.5),
                          leg(f'Under {o["total"]}', "Total", 1 - p_over, STD_DEC, 0.5)]
        dh, da = american_to_dec(o["ml_home"]), american_to_dec(o["ml_away"])
        if dh and da:
            mh = (1 / dh) / (1 / dh + 1 / da)  # no-vig
            cands += [leg(f'{_name(h)} to win', "Moneyline", win_h, dh, mh, h),
                      leg(f'{_name(a)} to win', "Moneyline", 1 - win_h, da, 1 - mh, a)]
        legs += cands
    return legs


COUNT_LINE = {("hockey", "G"): 0.5, ("hockey", "A"): 0.5, ("hockey", "PTS"): 0.5}
MIN_AVG_LEG = {"PASS": 120, "RUSH": 25, "REC": 25, "RECS": 2.0, "SOG": 1.5, "PTS": 5.0, "REB": 2.5, "AST": 1.5,
               "PRA": 10.0}


def _proxy_line(kind: str, m: str, hist: list[float]) -> float | None:
    """Stand-in line when no book line is posted: the player's last-10 average rounded to a .5 (books set
    their lines near recent form). Needs 5+ games and a meaningful role in the market."""
    if len(hist) < 5:
        return None
    if (kind, m) in COUNT_LINE:
        return COUNT_LINE[(kind, m)]
    avg = float(np.mean(hist))
    if avg < MIN_AVG_LEG.get(m, 0) and not (kind == "hockey" and m == "SOG"):
        return None
    return round(avg - 0.5) + 0.5


def prop_legs(sport: str, kind: str, game: dict, df, book: dict, norm_name, labels: dict) -> list[Leg]:
    """Player-prop legs. Uses the PrizePicks / Underdog line when posted; otherwise the player's last-10 average
    as a stand-in line (so props still mix into parlays when the pick'em boards block cloud servers)."""
    legs: list[Leg] = []
    if df is None or getattr(df, "empty", True) or game["state"] != "pre":
        return legs
    matchup = f'{_name(game["away"])} @ {_name(game["home"])}'
    when = game["tip"].strftime("%a %-I:%M %p") if game["tip"] is not None else ""
    by_abbr = {game[s_]["abbr"]: game[s_] for s_ in ("home", "away")}
    for r in df.to_dict("records"):
        team = by_abbr.get(r.get("Team"), {})
        for m in labels:
            if f"P_{m}" not in r:
                continue
            b = book.get((norm_name(r["Player"]), m))
            if b:
                line, src = b
                src_txt = f"{src} line"
            else:
                line = _proxy_line(kind, m, r.get(f"V10_{m}") or [])
                if line is None:
                    continue
                src_txt = "L10-avg line"
            proj = r[f"P_{m}"]
            if (kind, m) in COUNT_MARKETS:
                p_over = 1 - _poisson_cdf(int(math.floor(line)), max(proj, 0.01))
            else:
                hist = r.get(f"V10_{m}") or []
                sd = max(float(np.std(hist)) if len(hist) >= 4 else 0.0, PROP_SD_FLOOR.get(m, 0.25 * proj), 0.2 * proj)
                p_over = 1 - norm_cdf((line - proj) / sd)
            side, prob = ("Over", p_over) if p_over >= 0.5 else ("Under", 1 - p_over)
            if side == "Under" and (kind, m) in COUNT_LINE:
                continue  # "under 0.5 goals" is a heavy favorite at real books, never a -115 pick
            legs.append(Leg(sport, game["id"], matchup, f'{r["Player"]} {side} {line:g} {labels[m]}', "Prop", prob,
                            PROP_DEC, when, note=f"{src_txt} · model {proj:.1f}", mkt=0.5,
                            logo=team.get("logo", ""), fallback=r.get("Team", "")))
    return legs


def ufc_legs(card: dict, bouts: list[dict]) -> list[Leg]:
    legs: list[Leg] = []
    for x in bouts:
        b, pred = x["bout"], x["pred"]
        if b["state"] != "pre":
            continue
        fa, fb = b["fighters"]
        da, db = american_to_dec(fa.get("ml")), american_to_dec(fb.get("ml"))
        matchup = f'{x["a"]["name"]} vs {x["b"]["name"]}'
        when = card["start"].strftime("%a %-I:%M %p") if card.get("start") is not None else ""
        def initials(n):
            return "".join(w[0] for w in n.split()[:2]).upper()

        if da:
            legs.append(Leg("UFC", b["id"], matchup, f'{x["a"]["name"]} to win', "Fight", pred["p_a"], da, when,
                            logo=x["a"].get("headshot", ""), fallback=initials(x["a"]["name"]), photo=True))
        if db:
            legs.append(Leg("UFC", b["id"], matchup, f'{x["b"]["name"]} to win', "Fight", 1 - pred["p_a"], db, when,
                            logo=x["b"].get("headshot", ""), fallback=initials(x["b"]["name"]), photo=True))
    return legs


def value_legs(legs: list[Leg]) -> list[Leg]:
    """Legs worth considering: likely enough and priced below the model probability. Best leg per game first."""
    good = sorted((l for l in legs if l.prob >= MIN_PROB and l.edge >= MIN_EDGE), key=lambda l: -l.edge)
    return good


def _combos(pool: list[Leg], n: int, mixed: bool, all_sports: bool = False, props_mix: bool = False) -> list[dict]:
    out = []
    for c in itertools.combinations(pool, n):
        if len({(l.sport, l.game_id) for l in c}) < n:  # one leg per game: legs stay independent
            continue
        if props_mix:  # at least one player prop AND at least one game line
            k = sum(l.market == "Prop" for l in c)
            if k == 0 or k == n:
                continue
        n_sports = len({l.sport for l in c})
        if (mixed and n_sports < 2) or (all_sports and n_sports < n):
            continue
        p = math.prod(l.prob for l in c)
        dec = math.prod(l.dec for l in c)
        out.append({"legs": list(c), "prob": p, "dec": dec, "ev": p * dec - 1, "n_sports": n_sports})
    return out


def build(legs: list[Leg], sizes=(2, 3, 4), mixed: bool = False) -> list[dict]:
    """Best parlays, chosen in order (safest 2-leg by hit probability; for mixed, a 3-leg with every leg from a
    different sport; best-value 3-leg and longshot 4-leg by EV). No game is reused across parlays, so the same
    spread / total / pick never appears on two cards."""
    ranked = value_legs(legs)
    cap = MAX_POOL * (2 if mixed else 1)
    pool = ranked[:cap]
    extra = [l for l in ranked[cap:] if l.market == "Prop"][:6]  # make sure props get a seat in the search
    if not any(l.market == "Prop" for l in pool):
        pool += extra
    used: set[tuple[str, str]] = set()
    pre = "Mixed " if mixed else ""
    plan = [(2, f"{pre}safest 2-leg".capitalize(), "SAFE", "prob", False)]
    if any(l.market == "Prop" for l in pool) and any(l.market != "Prop" for l in pool):
        plan.append((3, "Props + game lines", "PROPS", "ev", "props"))
    if mixed:
        plan.append((3, "One leg per sport (3 sports)", "MIXED", "ev", True))
    plan += [(3, f"{pre}best value 3-leg".capitalize(), "VALUE", "ev", False),
             (4, f"{pre}longshot 4-leg".capitalize(), "LONGSHOT", "ev", False)]
    out = []
    for n, title, tag, key, all_sports in plan:
        if n not in sizes:
            continue
        avail = [l for l in pool if (l.sport, l.game_id) not in used]
        combos = _combos(avail, n, mixed, all_sports is True, props_mix=all_sports == "props")
        if not combos:
            continue
        if key == "prob":
            best = max(combos, key=lambda x: (x["ev"] > 0, x["prob"]))
        else:
            best = max(combos, key=lambda x: x["ev"])
        out.append({"title": title, "tag": tag, **best})
        used |= {(l.sport, l.game_id) for l in best["legs"]}
    return out
