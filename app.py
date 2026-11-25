"""
Sportsbook-style multi-sport predictor: NBA, College Basketball, NFL, NHL, UFC.

Data: ESPN public JSON (site.api.espn.com scoreboards, team schedules/stats/rosters, injuries;
site.web.api.espn.com athlete game logs and fighter profiles). stats.nba.com via nba_api is an optional
NBA enrichment wrapped in a strict try/except. Every request is throttled (0.6 s), sent with browser
headers, and cached on disk (./.cache). Nothing is mocked: if a source is down the app says so.

Run:  streamlit run app.py
"""

from __future__ import annotations

import importlib
import json
import math
import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import engine.common
import engine.espn
import engine.lines
import engine.news
import engine.parlays
import engine.props
import engine.teams
import engine.ufc
import engine.ufc_props
import engine.streaks
import engine.challenge
import engine.feedback
import engine.learner
import engine.decide
import engine.report
import engine.grade
import engine.ui

_ENGINE = [engine.common, engine.decide, engine.espn, engine.teams, engine.props, engine.streaks, engine.lines, engine.news, engine.ufc,
           engine.ufc_props, engine.parlays, engine.challenge, engine.feedback, engine.learner, engine.ui, engine.report, engine.grade]  # dependency order


@st.cache_resource(show_spinner=False)
def _engine_mtimes() -> dict:
    return {}


def _refresh_engine() -> None:
    """Streamlit Cloud re-runs app.py after a push but keeps already-imported engine modules in memory.
    Reload them (in dependency order) on first run of this code and whenever an engine file changes."""
    if os.environ.get("SHARPBOARD_NO_RELOAD"):  # tests patch engine functions; keep them
        return
    seen = _engine_mtimes()
    now = {m.__file__: os.path.getmtime(m.__file__) for m in _ENGINE}
    if seen != now:  # first run of this code in the process, or a file changed since the last reload
        for m in _ENGINE:
            importlib.reload(m)
        seen.clear()
        seen.update(now)


_refresh_engine()

from engine import decide, grade, report, espn, lines as linesmod, news as newsmod, parlays, props as propmod, teams, streaks as streakmod, challenge as chmod, feedback as fbmod, learner as lrn, ufc, ufc_props  # noqa: E402
from engine.common import ET, NOTES, cache_fresh, clear_disk_cache, fmt_spread, ml_to_prob, norm_name  # noqa: E402
from engine.espn import LEAGUES, League  # noqa: E402
from engine.ui import (CSS, H, avatar_html, edge_badge, empty_card, esc, ladder_html,  # noqa: E402
                       league_logo_html, logo_html, note, status_pill, verdict_html)

st.set_page_config(page_title="Sharp Board", page_icon="🏆", layout="centered", initial_sidebar_state="collapsed")
H(CSS)
H(report.REPORT_CSS)

NOTES.clear()  # notes describe this run only (engine code adds stale-cache / endpoint warnings as it goes)
now_et = datetime.now(ET)
today_et = now_et.date()
SPORTS = ["NBA", "NCAAB", "NFL", "NHL", "SOCCER", "UFC", "MIX", "CHAL"]
SPORT_ICON = {"NBA": "🏀", "NCAAB": "🏀", "NFL": "🏈", "NHL": "🏒", "SOCCER": "⚽", "UFC": "🥊", "MIX": "🎟", "CHAL": "🎯"}
PAGE_SIZE = 12


def bucket(minutes: int = 10) -> int:
    """Changes every N minutes so cached computations refresh as new data lands."""
    return int(time.time() // (minutes * 60))


PREFETCH_WORKERS = 8
ALL_TABS = os.environ.get("SHARPBOARD_ALL_TABS") == "1"  # tests: render lazy tabs without clicking them


def prefetch(reqs: list[tuple], label: str) -> int:
    """Fetch every not-yet-cached request in parallel behind one progress bar. Returns the failure count.
    Requests run on a small worker pool; per-host spacing lives in engine.common.throttle."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    todo = [r for r in reqs if not cache_fresh(r[0], r[1])]
    if not todo:
        return 0
    bar = st.progress(0.0, text=f"{label} 0/{len(todo)}")
    fails = done = 0
    with ThreadPoolExecutor(max_workers=min(PREFETCH_WORKERS, len(todo))) as ex:
        futs = [ex.submit(fetch) for _, _, fetch in todo]
        for fut in as_completed(futs):
            done += 1
            if fut.exception() is not None:
                fails += 1
            if done >= 6 and fails == done:  # endpoint clearly down: drop the rest instead of timing out on each
                for f in futs:
                    f.cancel()
                break
            bar.progress(done / len(todo), text=f"{label} {done}/{len(todo)} (cached afterwards)")
    bar.empty()
    return fails + (len(todo) - done if fails == done and done >= 6 else 0)


@st.cache_resource(show_spinner=False)
def _warm_registry() -> dict:
    return {}


def warm_in_background(key: str, jobs: list) -> None:
    """Fill the disk cache for jobs (callables) on a background thread, once per key per server process.
    Used to get the slate's other games ready while the user looks at the first one."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    reg = _warm_registry()
    if key in reg or not jobs:
        return
    reg[key] = time.time()

    def run():
        with ThreadPoolExecutor(max_workers=4) as ex:
            for f in [ex.submit(j) for j in jobs]:
                try:
                    f.result()
                except Exception:
                    pass

    threading.Thread(target=run, daemon=True).start()


# --------------------------------------------------------------------------
# Header: sport switcher + settings
# --------------------------------------------------------------------------
SPORT_NAME = {"NBA": "NBA", "NCAAB": "College Hoops", "NFL": "NFL", "NHL": "NHL", "SOCCER": "Soccer", "UFC": "UFC",
              "MIX": "Mixed Parlays", "CHAL": "Bankroll Challenge"}
BRAND_MARK = ('<svg class="mark" viewBox="0 0 40 40" fill="none"><rect x="1" y="1" width="38" height="38" rx="11" '
              'fill="#0f1a12" stroke="#3ddc84" stroke-width="1.5"/><path d="M9 27l7-8 5 5 10-12" stroke="#3ddc84" '
              'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><circle cx="31" cy="12" r="2.6" '
              'fill="#3ddc84"/></svg>')
with st.container(key="topbar"):
    c_brand, c_sport = st.columns([1.15, 1])
    with c_brand:
        H(f'<div class="topbar">{BRAND_MARK}<div class="bn">Sharp<span>Board</span></div></div>')
    with c_sport:
        sport = st.selectbox("Sport", SPORTS, key="sport_pick", label_visibility="collapsed",
                             format_func=lambda k: f"{SPORT_ICON[k]}  {SPORT_NAME[k]}")
if sport is None:
    sport = st.session_state.get("last_sport", "NBA")
st.session_state["last_sport"] = sport
lg = LEAGUES.get(sport)  # None for the mixed-parlay view

with st.expander("⚙️  Settings", expanded=False):
    if sport == "UFC":
        st.caption("UFC shows the current or next fight card automatically.")
        sel_date = today_et
    elif sport in ("MIX", "CHAL"):
        st.caption("Mixed parlays and the challenge use today's NBA / NHL / Top 25 college games, top-league soccer, NFL games in "
                   "the next 3 days, and this week's UFC card.")
        sel_date = today_et
    else:
        sel_date = st.date_input("Week containing" if sport == "NFL" else "Slate date (ET)", value=today_et,
                                 key=f"date_{sport}")
    q_sit = st.slider("Chance a Questionable / Day-to-Day player sits", 0.0, 1.0, 0.5, 0.05, key="q_sit")
    if st.button("🔄 Refresh all data", width="stretch"):
        clear_disk_cache()
        st.cache_data.clear()
        st.cache_resource.clear()
        NOTES.clear()
        st.rerun()


def hero(sub: str) -> None:
    icon = SPORT_ICON[sport] if sport in ("MIX", "SOCCER", "CHAL") else league_logo_html(sport, 36)
    title = {"CHAL": "Bankroll <span>Challenge</span>"}.get(sport, f"{esc(sport)} <span>Picks</span>")
    H(f'<div class="hero"><div class="brand">{icon}{title}</div>'
      f'<div class="hero-sub">{sub}</div></div>')


# --------------------------------------------------------------------------
# Team sports
# --------------------------------------------------------------------------
def nfl_week_window(d: date) -> tuple[date, int]:
    start = d - timedelta(days=(d.weekday() - 1) % 7)  # back to Tuesday
    return start, 7


@st.cache_data(ttl=600, show_spinner=False, max_entries=48)
def compute_slate(lg_key: str, game_ids: tuple, q: float, inj_sig: tuple, tick: int,
                  _games: list[dict], _inj: list[dict], odds_sig: tuple = ()) -> dict:
    lg_ = LEAGUES[lg_key]
    profiles: dict[str, dict | None] = {}
    for g in _games:
        for side in ("home", "away"):
            tid = g[side]["id"]
            if tid not in profiles:
                profiles[tid] = teams.team_profile(lg_, tid, g["date"] or today_et)
    base = teams.league_baseline(lg_key, profiles)
    slate_day = min((g["date"] for g in _games if g["date"]), default=today_et)
    b2b = teams.back_to_back_teams(profiles, slate_day) if teams.cfg(lg_key).get("b2b") else set()
    lost = {}
    for g in _games:
        for side in ("home", "away"):
            lost[g[side]["id"]] = propmod.lost_points(lg_, g[side], _inj, q, g["date"] or today_et)
    nba_adv = None
    if lg_key == "NBA":
        y = slate_day.year if slate_day.month >= 9 else slate_day.year - 1
        nba_adv = teams.nba_api_ratings(f"{y}-{(y + 1) % 100:02d}") or None
    projections = {}
    for g in _games:
        try:
            p = teams.project_game(lg_, g, profiles.get(g["home"]["id"]), profiles.get(g["away"]["id"]), base,
                                   b2b, lost, nba_adv)
            if p:
                projections[g["id"]] = p
        except Exception as e:
            NOTES[f"model {g['id']}"] = f"Projection failed for {g['away']['abbr']} @ {g['home']['abbr']}: {e}"
    if lg_key in teams.BLEND:  # college: blend with the betting line (model weight grows with games played)
        for g in _games:
            if g["id"] in projections:
                gp = min((profiles.get(g["home"]["id"]) or {}).get("gp", 0), (profiles.get(g["away"]["id"]) or {}).get("gp", 0))
                projections[g["id"]] = teams.market_blend(lg_, g, projections[g["id"]], gp)
    return {"profiles": profiles, "base": base, "b2b": b2b, "projections": projections,
            "nba_adv": bool(nba_adv)}


@st.cache_data(ttl=300, show_spinner=False, max_entries=8)
def cached_lines(lg_key: str, tick: int) -> dict:
    return linesmod.book_lines(LEAGUES[lg_key])


def _odds_api_key() -> str:
    try:
        return str(st.secrets.get("ODDS_API_KEY", "") or os.environ.get("ODDS_API_KEY", ""))
    except Exception:
        return os.environ.get("ODDS_API_KEY", "")


@st.cache_data(ttl=1800, show_spinner=False, max_entries=64)
def cached_odds_api(lg_key: str, home_name: str, away_name: str, tick: int) -> dict:
    return linesmod.odds_api_props(LEAGUES[lg_key], home_name, away_name, None, _odds_api_key())


@st.cache_data(ttl=300, show_spinner=False, max_entries=64)
def cached_espn_props(lg_key: str, game_id: str, tick: int) -> dict:
    return linesmod.espn_props(LEAGUES[lg_key], game_id)


@st.cache_data(ttl=300, show_spinner=False, max_entries=32)
def cached_news(lg_key: str, team_ids: tuple, tick: int) -> list[dict]:
    return newsmod.articles(LEAGUES[lg_key], team_ids)


@st.cache_data(ttl=600, show_spinner=False, max_entries=64)
def compute_props(lg_key: str, game_id: str, q: float, tick: int, _game: dict, _proj: dict,
                  _profiles: dict, _base: dict, _b2b: set, _inj: list[dict]) -> pd.DataFrame:
    return propmod.build_props(LEAGUES[lg_key], _game, _proj, _profiles, _base, _b2b, _inj, q)


def matchup_card(g: dict, p: dict | None, b2b: set[str], kind: str, badge: str = "") -> str:
    h, a = g["home"], g["away"]
    st_pill = f'<span class="pill comp">{esc(badge)}</span>' if badge else ""
    if g["state"] == "in":
        st_pill += f'<span class="pill live">● {esc(g["status"])}</span>'
    elif g["state"] == "post":
        st_pill += f'<span class="pill final">{esc(g["status"])}</span>'
    tags = "".join(f'<span class="pill b2b">{esc(t["abbr"])} B2B</span>' for t in (a, h) if t["id"] in b2b)
    if g["neutral"]:
        tags += '<span class="pill info">Neutral</span>'
    if p and p.get("blend_w") is not None:
        tags += f'<span class="pill info">{p["blend_w"] * 100:.0f}% model · {100 - p["blend_w"] * 100:.0f}% mkt</span>'
    tip = g["tip"].strftime("%a %-I:%M %p ET") if g["tip"] is not None else ""
    dec = 1 if kind in ("hockey", "soccer") else 0

    def trow(t, proj_pts):
        rank = f'<span class="rank">#{t["rank"]}</span>' if t.get("rank") else ""
        sc = (f'<div class="tscore">{esc(str(t["score"]))}</div>'
              if g["state"] in ("in", "post") and t["score"] not in (None, "") else "")
        pj = (f'<div class="tproj"><span class="lbl">Proj</span><span class="num">{proj_pts:.{dec}f}</span></div>'
              if proj_pts is not None else "")
        return (f'<div class="trow">{logo_html(t["logo"], t["abbr"])}<div class="tinfo"><div class="tabbr">{rank}'
                f'{esc(t["abbr"])}<span class="rec">{esc(t["record"] or "")}</span></div>'
                f'<div class="tname">{esc(t["short"] or t["name"])}</div></div>{sc}{pj}</div>')

    body = (f'<div class="card"><div class="ghead"><div class="gtime">{esc(tip)}</div>'
            f'<div class="tags">{tags}{st_pill}</div></div>'
            + trow(a, p["pts_away"] if p else None) + trow(h, p["pts_home"] if p else None))
    if not p:
        return body + '<div class="empty">Not enough completed games yet for a projection.</div></div>'
    if kind == "soccer":
        return body + soccer_markets(g, p) + "</div>"
    hp = p["home_win_prob"] * 100
    fav_cls_a, fav_cls_h = ("a", "h fav") if hp >= 50 else ("a fav", "h")
    body += (f'<div class="winbar"><div class="{fav_cls_a}" style="width:{100 - hp:.0f}%"></div>'
             f'<div class="{fav_cls_h}" style="width:{hp:.0f}%"></div></div>'
             f'<div class="winlbl"><span>{esc(a["abbr"])} {100 - hp:.0f}%</span><span>{esc(h["abbr"])} {hp:.0f}%</span></div>')
    o = g["odds"]
    mkl = "Closing Mkt" if o.get("closing") else "Mkt"
    fav = h if hp >= 50 else a
    if kind == "hockey":
        # P(favorite wins by 2+ in regulation), from the Poisson grid
        pl_p = p["home_puckline_prob"] if fav is h else p["away_puckline_prob"]
        box1 = (f'<div class="mkt"><span class="lbl">Puck line</span><div class="val">{esc(fav["abbr"])} -1.5</div>'
                f'<div class="sub">Model {pl_p * 100:.0f}%</div><span class="badge muted">MODEL</span></div>')
        t_lean, t_strong = 0.3, 0.6
    else:
        s_edge = (o["spread_home"] - p["spread_home"]) if o["spread_home"] is not None else None
        s_badge = edge_badge(s_edge, 1.5, 3.0,
                             f'{h["abbr"]} {fmt_spread(o["spread_home"])}' if s_edge is not None else "",
                             f'{a["abbr"]} {fmt_spread(-o["spread_home"])}' if s_edge is not None else "")
        sp = p["spread_home"]
        fav_txt = "PK" if abs(sp) < 0.25 else (f'{h["abbr"]} -{abs(sp):.1f}' if sp < 0 else f'{a["abbr"]} -{abs(sp):.1f}')
        box1 = (f'<div class="mkt"><span class="lbl">Spread</span><div class="val">{esc(fav_txt)}</div>'
                f'<div class="sub">{mkl} {esc(o["details"] or (fmt_spread(o["spread_home"]) if o["spread_home"] is not None else "—"))}</div>{s_badge}</div>')
        t_lean, t_strong = (2.0, 5.0) if kind == "basketball" else (1.5, 3.5)
    t_edge = (p["total"] - o["total"]) if o["total"] is not None else None
    t_badge = edge_badge(t_edge, t_lean, t_strong, f'OVER {o["total"]}' if o["total"] else "",
                         f'UNDER {o["total"]}' if o["total"] else "", neg_is_red=True)
    box2 = (f'<div class="mkt"><span class="lbl">Total</span><div class="val">{p["total"]:.1f}</div>'
            f'<div class="sub">{mkl} {esc(str(o["total"]) if o["total"] is not None else "—")}</div>{t_badge}</div>')
    mh, ma = ml_to_prob(o["ml_home"]), ml_to_prob(o["ml_away"])
    if mh and ma:
        mkt_h = mh / (mh + ma)  # vig-free
        ml_badge = edge_badge((p["home_win_prob"] - mkt_h) * 100, 3, 7, f'{h["abbr"]} ML', f'{a["abbr"]} ML',
                              fmt="{:.0f}%")
        ml_sub = f'{mkl} {fav["abbr"]} {(mkt_h if fav is h else 1 - mkt_h) * 100:.0f}%'
    else:
        ml_badge = '<span class="badge muted">MODEL</span>'
        ml_sub = f'{p["poss"]:.0f} poss' if p.get("poss") else "Win probability"
    box3 = (f'<div class="mkt"><span class="lbl">Win</span><div class="val">{esc(fav["abbr"])} '
            f'{max(hp, 100 - hp):.0f}%</div><div class="sub">{esc(ml_sub)}</div>{ml_badge}</div>')
    return body + f'<div class="mkts">{box1}{box2}{box3}</div></div>'


# --------------------------------------------------------------------------
# Period / half markets
# --------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def period_drawer(lg_: League, g: dict, p: dict | None, ph: dict | None, pa_: dict | None) -> str:
    """Collapsible table: every period / half at a glance (projected score, spread, total, win %)."""
    if not p or not ph or not pa_ or "per_for" not in ph or "per_for" not in pa_:
        return ""
    segs = teams.period_segments(lg_)
    if len(segs) < 2:
        return ""
    h, a = g["home"]["abbr"], g["away"]["abbr"]
    goals = lg_.kind in ("hockey", "soccer")
    d = 2 if goals else 1
    rows = []
    for code, label, idx in segs:
        q = teams.full_as_segment(lg_, p) if idx is None else teams.period_projection(lg_, p, ph, pa_, idx)
        if not q:
            continue
        margin = q["pts_home"] - q["pts_away"]
        fav = h if margin >= 0 else a
        spread = "PK" if abs(margin) < 0.25 else f"{fav} -{abs(margin):.{d}f}"
        if q.get("three_way"):
            win = f'{_pct(q["home_win"])} · D {_pct(q["draw"])}'
        else:
            wh = q["home_win"] / max(q["home_win"] + q["away_win"], 1e-9)
            win = f"{h if wh >= .5 else a} {_pct(max(wh, 1 - wh))}"
        rows.append(f'<div class="pt{" fg" if idx is None else ""}"><span class="sg">{esc(code)}</span>'
                    f'<span>{q["pts_away"]:.{d}f}</span><span>{q["pts_home"]:.{d}f}</span>'
                    f'<span>{esc(spread)}</span><span>{q["total"]:.{d}f}</span><span>{esc(win)}</span></div>')
    real = min(ph.get("ls_real", 0), pa_.get("ls_real", 0))
    n = min(ph.get("ls_games", 0), pa_.get("ls_games", 0))
    src = (f"Splits from {real} of the last {n} games with ESPN period scores"
           + ("; the rest use league-typical splits." if real < n else "."))
    win_head = f"{h} win · draw" if lg_.kind in ("hockey", "soccer") else "Win"
    return (f'<details class="pdrawer"><summary>Period &amp; Half Breakdown</summary><div class="ptab">'
            f'<div class="pt head"><span></span><span>{esc(a)}</span><span>{esc(h)}</span><span>Spread</span>'
            f'<span>Total</span><span>{esc(win_head)}</span></div>{"".join(rows)}</div>'
            f'<div class="pt-note">{esc(src)}</div></details>')


def period_card(g: dict, q: dict, code: str, label: str, kind: str, badge: str = "", imp: dict | None = None) -> str:
    """Matchup card for one period / half: projected score, moneyline (with draw/push), spread, total."""
    h, a = g["home"], g["away"]
    goals = kind in ("hockey", "soccer")
    d = 2 if goals else 1
    tags = f'<span class="pill comp">{esc(badge)}</span>' if badge else ""
    tags += f'<span class="pill info">{esc(label)}</span>'
    tip = g["tip"].strftime("%a %-I:%M %p ET") if g["tip"] is not None else ""

    def trow(t, pts):
        return (f'<div class="trow">{logo_html(t["logo"], t["abbr"])}<div class="tinfo"><div class="tabbr">'
                f'{esc(t["abbr"])}<span class="rec">{esc(t["record"] or "")}</span></div>'
                f'<div class="tname">{esc(t["short"] or t["name"])}</div></div>'
                f'<div class="tproj"><span class="lbl">Proj {esc(code)}</span><span class="num">{pts:.{d}f}</span></div></div>')

    ph, pd_, pa = q["home_win"], q["draw"], q["away_win"]
    top = max(ph, pd_, pa) if q["three_way"] else max(ph, pa)
    if q["three_way"]:
        bar = (f'<div class="winbar"><div class="a{" fav" if pa == top else ""}" style="width:{pa * 100:.0f}%"></div>'
               f'<div class="d{" fav" if pd_ == top else ""}" style="width:{pd_ * 100:.0f}%"></div>'
               f'<div class="h{" fav" if ph == top else ""}" style="width:{ph * 100:.0f}%"></div></div>'
               f'<div class="winlbl"><span>{esc(a["abbr"])} {_pct(pa)}</span><span>Draw {_pct(pd_)}</span>'
               f'<span>{esc(h["abbr"])} {_pct(ph)}</span></div>')
    else:
        bar = (f'<div class="winbar"><div class="a{" fav" if pa == top else ""}" style="width:{pa * 100:.0f}%"></div>'
               f'<div class="d" style="width:{pd_ * 100:.0f}%"></div>'
               f'<div class="h{" fav" if ph == top else ""}" style="width:{ph * 100:.0f}%"></div></div>'
               f'<div class="winlbl"><span>{esc(a["abbr"])} {_pct(pa)}</span><span>Push {pd_ * 100:.1f}%</span>'
               f'<span>{esc(h["abbr"])} {_pct(ph)}</span></div>')
    if q["three_way"]:
        best = max((("home", ph, h["abbr"], q["fair_home_3w"]), ("draw", pd_, "Draw", q["fair_draw"]),
                    ("away", pa, a["abbr"], q["fair_away_3w"])), key=lambda x: x[1])
        ml = (f'<div class="mkt"><span class="lbl">{esc(code)} 3-way</span><div class="val">{esc(best[2])} '
              f'{_pct(best[1])}</div><div class="sub">Fair {esc(best[3])}</div>'
              f'<span class="badge muted">D {esc(q["fair_draw"])}</span></div>')
    else:
        home_fav = ph >= pa
        fav, fair, p2 = (h, q["fair_home"], ph) if home_fav else (a, q["fair_away"], pa)
        ml = (f'<div class="mkt"><span class="lbl">{esc(code)} Moneyline</span><div class="val">{esc(fav["abbr"])} '
              f'{_pct(p2)}</div><div class="sub">Fair {esc(fair)}</div>'
              f'<span class="badge muted">Push {pd_ * 100:.1f}%</span></div>')
    margin = q["pts_home"] - q["pts_away"]
    fav_t = h if margin >= 0 else a
    imp = imp or {}
    root = math.sqrt(max(imp.get("share") or 0.25, 0.1))
    mk = "Implied closing" if g["odds"].get("closing") else "Implied Mkt"
    it, isp = imp.get("total"), imp.get("spread_home")
    if it is not None:
        t_sub = f"{mk} {it:.1f}"
        t_badge = edge_badge(q["total"] - it, (0.15 if goals else 2.0) * root, (0.3 if goals else 5.0) * root,
                             f"OVER {it:.1f}", f"UNDER {it:.1f}", neg_is_red=True, fmt="{:.1f}")
    else:
        t_sub, t_badge = f"{mk} —", '<span class="badge muted">NO FG LINE</span>'
    if goals:
        box2 = (f'<div class="mkt"><span class="lbl">{esc(code)} Goals</span><div class="val">{q["total"]:.2f}</div>'
                f'<div class="sub">{esc(t_sub)} · O0.5 {_pct(q["over"][0.5])}</div>{t_badge}</div>')
        if kind == "soccer":
            box3 = (f'<div class="mkt"><span class="lbl">{esc(code)} BTTS</span><div class="val">Yes {_pct(q["btts"])}'
                    f'</div><div class="sub">Both teams score</div><span class="badge muted">MODEL</span></div>')
        else:
            box3 = (f'<div class="mkt"><span class="lbl">{esc(code)} Total</span><div class="val">O/U {q["total_line"]:g}'
                    f'</div><div class="sub">Over {_pct(q["p_over"])}</div><span class="badge muted">MODEL</span></div>')
        boxes = ml + box2 + box3
    else:
        sp = "PK" if q["spread_line"] == 0 else f'{fav_t["abbr"]} -{q["spread_line"]:g}'
        if isp is not None:
            s_txt = (f'{h["abbr"]} {isp:+.1f}' if isp <= 0 else f'{a["abbr"]} {-isp:+.1f}')
            s_sub = f"{mk} {s_txt}"
            s_badge = edge_badge(isp - q["spread_home"], 1.5 * root, 3.0 * root, f'{h["abbr"]} {isp:+.1f}',
                                 f'{a["abbr"]} {-isp:+.1f}', fmt="{:.1f}")
        else:
            s_sub, s_badge = f"{mk} —", '<span class="badge muted">NO FG LINE</span>'
        box2 = (f'<div class="mkt"><span class="lbl">{esc(code)} Spread</span><div class="val">{esc(sp)}</div>'
                f'<div class="sub">{esc(s_sub)}</div>{s_badge}</div>')
        box3 = (f'<div class="mkt"><span class="lbl">{esc(code)} Total</span><div class="val">{q["total"]:.1f}'
                f'</div><div class="sub">{esc(t_sub)}</div>{t_badge}</div>')
        boxes = ml + box2 + box3
    return (f'<div class="card"><div class="ghead"><div class="gtime">{esc(tip)}</div><div class="tags">{tags}</div></div>'
            + trow(a, q["pts_away"]) + trow(h, q["pts_home"]) + bar + f'<div class="mkts">{boxes}</div></div>')


def game_alt_html(g: dict, lad: dict, kind: str) -> str:
    h, a = g["home"]["abbr"], g["away"]["abbr"]
    goals = kind in ("hockey", "soccer")
    title = "Puck line / goal line" if kind == "hockey" else ("Goal line" if kind == "soccer" else "Alt spread")
    parts = [ladder_html(f"{title} ({h} side)", lad["spread"], lambda x: f'{esc(h)} {x["line"]:+g}',
                         ("Line", "Covers", "Fair", "")),
             ladder_html("Alt total (over)", lad["total"], lambda x: f'O {x["line"]:g}', ("Line", "Over", "Fair", ""))]
    for side, abbr in (("away", a), ("home", h)):
        parts.append(ladder_html(f"{abbr} team total (over)", lad["team_totals"][side],
                                 lambda x: f'O {x["line"]:g}', ("Line", "Over", "Fair", "")))
    unit = "goals" if goals else "points"
    return (f'<details class="pdrawer"><summary>Alt lines · spread, total, team totals</summary>{"".join(parts)}'
            f'<div class="pt-note">Model probabilities and fair odds in {unit}; ESPN posts only the main line. '
            'Main line highlighted.</div></details>')


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def graded_props(lg_key: str, gid: str, q: float, tick: int, _game: dict, _slate: dict, _inj: list) -> list[dict]:
    """Pregame picks for a finished game (projection from games before tip-off, posted or implied line)
    graded against the final box score."""
    lg_ = LEAGUES[lg_key]
    kind = lg_.kind
    if kind not in propmod.MARKETS or gid not in _slate["projections"]:
        return []
    df = compute_props(lg_key, gid, q, tick, _game, _slate["projections"][gid], _slate["profiles"], _slate["base"],
                       _slate["b2b"], _inj)
    box = grade.box_stats(lg_, gid)
    if df.empty or not box:
        return []
    book = cached_espn_props(lg_key, gid, tick)
    out = []
    for r in df.to_dict("records"):
        row = box.get(str(r["Id"]))
        for m, label in propmod.MARKETS[kind]:
            if f"P_{m}" not in r:
                continue
            floor = propmod.MIN_AVG.get(kind, {}).get(m)
            if floor is not None and max(r.get(f"A5_{m}") or 0, r[f"P_{m}"]) < floor:
                continue
            info = prop_line_info(r, m, label, kind, linesmod.lookup(book, r["Id"], r["Player"], m), 0)
            v = info["verdict"]
            if v["verdict"] == "NO LINE":
                continue
            side = "Over" if r[f"P_{m}"] > info["line"] else "Under"
            act = grade.actual_value(kind, m, row)
            res = grade.result(side, info["line"], act)
            if res is None:
                continue
            out.append({"player": r["Player"], "id": r["Id"], "team": r["Team"], "label": label, "m": m,
                        "side": side, "line": info["line"], "src": info["src"], "verdict": v["verdict"],
                        "actual": act, "result": res, "proj": r[f"P_{m}"]})
    return out


def results_drawer(rows: list[dict]) -> str:
    if not rows:
        return ""
    bets = [r for r in rows if r["verdict"] == "BET"]
    passes = [r for r in rows if r["verdict"] != "BET"]

    def rec(rs):
        w, l_, p_ = (sum(r["result"] == k for r in rs) for k in ("WON", "LOST", "PUSH"))
        return f"{w}-{l_}" + (f"-{p_}" if p_ else "")

    def line_html(r):
        chip = {"WON": ("won", "✅ WON"), "LOST": ("lost", "❌ LOST"), "PUSH": ("push", "➖ PUSH")}[r["result"]]
        tag = '<span class="vpill bet sm">BET</span>' if r["verdict"] == "BET" else '<span class="vpill pass sm">PASS</span>'
        act = f'{r["actual"]:.0f}' if float(r["actual"]).is_integer() else f'{r["actual"]:.1f}'
        return (f'<div class="resrow"><div class="rn"><b>{esc(r["player"])}</b> {tag}<small>{esc(r["side"])} '
                f'{r["line"]:g} {esc(r["label"])} · proj {r["proj"]:.1f}</small></div>'
                f'<div class="ra">{act}</div><span class="res {chip[0]}">{chip[1]}</span></div>')

    head = f"Prop results · BETs {rec(bets)}" if bets else "Prop results · no BET picks"
    body = "".join(line_html(r) for r in bets)
    if passes:
        body += (f'<div class="pt-note">PASS picks went {rec(passes)} — top {min(len(passes), 8)} shown</div>'
                 + "".join(line_html(r) for r in sorted(passes, key=lambda r: -abs(r["proj"] - r["line"]))[:8]))
    return (f'<details class="pdrawer"><summary>{esc(head)}</summary>{body}'
            '<div class="pt-note">Graded against the final box score. Picks are rebuilt as they stood pregame: the '
            'projection only uses games before tip-off; the line is the posted one if still available, else the '
            'implied line.</div></details>')


@st.fragment
def matchups_fragment(games: list[dict], slate_for, b2b: set, kind: str, soccer: bool = False) -> None:
    """Market period pill bar + matchup cards (each with a Period & Half Breakdown drawer)."""
    if not games:
        return
    segs = teams.period_segments(LEAGUES[games[0].get("league", sport)])
    labels = {c: l for c, l, _ in segs}
    idx_of = {c: i for c, _, i in segs}
    code = "FG"
    if len(segs) > 1:
        code = st.segmented_control("Market period", list(labels), default="FG", key=f"period_{sport}",
                                    format_func=labels.get, label_visibility="collapsed", width="stretch") or "FG"
    finished = [g for g in games if g["state"] == "post"]
    do_grade = False
    if finished and kind in propmod.MARKETS and code == "FG":
        do_grade = st.toggle(f"Show prop results on finished games (first {GRADE_CAP}, loads player logs)",
                             value=True, key=f"grade_{sport}")
    graded_n = 0
    if code != "FG":
        note(f"{labels[code]} markets: each team's real scoring and conceding splits for that segment (ESPN period "
             "scores, last 15 games) applied to the full-game projection. Model prices only — ESPN doesn't list "
             "period odds.")
    html, cur = [], None
    for g in games:
        lk = g.get("league", sport)
        lg_ = LEAGUES[lk]
        sl = slate_for(g)
        if soccer and lk != cur:
            cur = lk
            kind_tag = {"cup": "Cup", "intl": "International"}.get(espn.SOCCER_TYPE.get(lk, ""), "")
            html.append(f'<div class="seg-head">{league_logo_html(lk, 22)} {esc(lg_.label)}'
                        + (f' <span class="pill info">{kind_tag}</span>' if kind_tag else "") + "</div>")
        p = sl["projections"].get(g["id"])
        ph, pa_ = sl["profiles"].get(g["home"]["id"]), sl["profiles"].get(g["away"]["id"])
        badge = lg_.label if soccer else ""
        card = None
        imp, q = None, None
        if code != "FG" and p:
            q = teams.period_projection(lg_, p, ph, pa_, idx_of[code])
            if q:
                imp = decide.implied_period_lines(g["odds"], q["share"])
                card = period_card(g, q, code, labels[code], kind, badge, imp)
        extra = ""
        if card is None:
            card = matchup_card(g, p, b2b, kind, badge=badge)
            if p and g["state"] in ("pre", "in"):
                gp = min((ph or {}).get("gp", 0), (pa_ or {}).get("gp", 0))
                v = decide.game_decision(lg_.key, kind, g, p, gp)
                if g["state"] == "in":
                    v["reason"] = "Pregame call vs closing line · " + v["reason"]
                extra += verdict_html(v)
                extra += game_alt_html(g, decide.game_alt_ladders(lg_.key, kind, g, p), kind)
        elif q and imp is not None and g["state"] in ("pre", "in"):
            v = decide.period_decision(lg_.key, kind, g, q, imp, code)
            if g["state"] == "in":
                v["reason"] = "Pregame call vs implied closing line · " + v["reason"]
            extra += verdict_html(v)
        if do_grade and g["state"] == "post" and p and graded_n < GRADE_CAP:
            graded_n += 1
            try:
                prefetch_players(lg_, g)
                try:
                    inj_ = espn.injuries(lg_)
                except Exception:
                    inj_ = []
                extra += results_drawer(graded_props(lg_.key, g["id"], float(q_sit), bucket(60), g, sl, inj_))
            except Exception as e:
                NOTES[f"grade {g['id']}"] = f"Couldn't grade props for {g['away']['abbr']} @ {g['home']['abbr']} ({e})."
        extra += period_drawer(lg_, g, p, ph, pa_)
        if extra and card.endswith("</div>"):
            card = card[:-6] + extra + "</div>"
        html.append(card)
    H("".join(html))


def soccer_markets(g: dict, p: dict) -> str:
    """Three-way bar (away / draw / home) + Result, Total goals and Both-teams-score boxes."""
    h, a, o = g["home"], g["away"], g["odds"]
    mkl = "Closing Mkt" if o.get("closing") else "Mkt"
    ph, pd_, pa = p["home_win_prob"], p["draw_prob"], p["away_win_prob"]
    top = max(ph, pd_, pa)
    bar = (f'<div class="winbar"><div class="a{" fav" if pa == top else ""}" style="width:{pa * 100:.0f}%"></div>'
           f'<div class="d{" fav" if pd_ == top else ""}" style="width:{pd_ * 100:.0f}%"></div>'
           f'<div class="h{" fav" if ph == top else ""}" style="width:{ph * 100:.0f}%"></div></div>'
           f'<div class="winlbl"><span>{esc(a["abbr"])} {pa * 100:.0f}%</span><span>Draw {pd_ * 100:.0f}%</span>'
           f'<span>{esc(h["abbr"])} {ph * 100:.0f}%</span></div>')
    outcomes = [(h["abbr"], ph, o.get("ml_home")), ("Draw", pd_, o.get("ml_draw")), (a["abbr"], pa, o.get("ml_away"))]
    pick = max(outcomes, key=lambda x: x[1])
    decs = [parlays.american_to_dec(x[2]) for x in outcomes]
    if all(decs):
        inv = sum(1 / d for d in decs)
        mkt = [(1 / d) / inv for d in decs]  # vig-free three-way
        edges = [x[1] - m for x, m in zip(outcomes, mkt)]
        i = max(range(3), key=lambda k: edges[k])
        lbl = "Draw" if outcomes[i][0] == "Draw" else f"{outcomes[i][0]} win"
        r_badge = edge_badge(edges[i] * 100, 3, 7, lbl, lbl, fmt="{:.0f}%")
        r_sub = f"{mkl} {pick[0]} {mkt[outcomes.index(pick)] * 100:.0f}%"
    else:
        r_badge, r_sub = '<span class="badge muted">MODEL</span>', "Home · Draw · Away"
    t_edge = (p["total"] - o["total"]) if o["total"] is not None else None
    t_badge = edge_badge(t_edge, 0.25, 0.5, f'OVER {o["total"]}' if o["total"] else "",
                         f'UNDER {o["total"]}' if o["total"] else "", neg_is_red=True, fmt="{:.2f}")
    return (bar + '<div class="mkts">'
            f'<div class="mkt"><span class="lbl">Result</span><div class="val">{esc(pick[0])} {pick[1] * 100:.0f}%</div>'
            f'<div class="sub">{esc(r_sub)}</div>{r_badge}</div>'
            f'<div class="mkt"><span class="lbl">Goals</span><div class="val">{p["total"]:.2f}</div>'
            f'<div class="sub">{mkl} {esc(str(o["total"]) if o["total"] is not None else "—")}</div>{t_badge}</div>'
            f'<div class="mkt"><span class="lbl">Both score</span><div class="val">Yes {p["btts_prob"] * 100:.0f}%</div>'
            f'<div class="sub">BTTS</div><span class="badge muted">MODEL</span></div></div>')


def soccer_window(mode: str, d: date) -> tuple[date, int, str]:
    if mode == "Next 3 Days":
        return d, 3, f"{d:%a %b %d} – {d + timedelta(days=2):%a %b %d}"
    if mode == "Weekend":
        start = d if d.weekday() >= 4 else d + timedelta(days=4 - d.weekday())  # Friday (or today if Fri-Sun)
        days = 6 - start.weekday() + 1                                          # through Sunday
        return start, days, f"Weekend · {start:%a %b %d} – {start + timedelta(days=days - 1):%a %b %d}"
    return d, 1, f"{d:%a, %b %d}"


def soccer_fixtures(keys: list[str], start: date, days: int) -> tuple[list[dict], list[str]]:
    games, failed = [], []
    for k in keys:
        lg_ = LEAGUES[k]
        try:
            gs = (espn.parse_games(lg_, espn.scoreboard(lg_, start, today_et)) if days == 1
                  else espn.week_games(lg_, start, days, today_et))
        except Exception:
            failed.append(lg_.label)
            continue
        for g in gs:
            g["league"] = k
            games.append(g)
    return games, failed


def render_soccer() -> None:
    keys = [k for k, _, _ in espn.SOCCER_LEAGUES]
    order = {k: i for i, k in enumerate(keys)}
    mode = st.segmented_control("Leagues", ["All Featured", "Choose leagues"], default="All Featured",
                                key="soc_mode", label_visibility="collapsed", width="stretch") or "All Featured"
    if mode == "Choose leagues":
        picked: list[str] = []
        n_sel = sum(len(st.session_state.get(f"soc_r{i}") or []) for i in range(len(espn.SOCCER_REGIONS)))
        with st.expander(f"🌍  Leagues & cups ({n_sel or len(espn.SOCCER_DEFAULT)} selected)", expanded=not n_sel):
            for i, (region, items) in enumerate(espn.SOCCER_REGIONS):
                H(f'<div class="region-h">{esc(region)}</div>')
                rk = [k for k, _, _, _ in items]
                sel = st.pills(region, rk, selection_mode="multi", key=f"soc_r{i}",
                               default=[k for k in rk if k in espn.SOCCER_DEFAULT],
                               format_func=lambda k: LEAGUES[k].label, label_visibility="collapsed")
                picked += sel or []
        picked = picked or espn.SOCCER_DEFAULT
    else:
        picked = espn.SOCCER_FEATURED
    window = st.segmented_control("Window", ["Today", "Next 3 Days", "Weekend"], default="Today",
                                  key="soc_window", label_visibility="collapsed", width="stretch") or "Today"
    start, days, label = soccer_window(window, sel_date)
    with st.spinner(f"Loading fixtures from {len(picked)} competitions…"):
        games, failed = soccer_fixtures(picked, start, days)
        if not games and window == "Today":  # empty day: look 4 days ahead automatically
            start, days = sel_date, 4
            games, failed = soccer_fixtures(picked, start, days)
            if games:
                label = f"Next 4 days · {start:%a %b %d} – {start + timedelta(days=3):%a %b %d}"
                NOTES["soccer window"] = "No matches today in these competitions — showing the next 4 days."
    if failed:
        NOTES["soccer"] = f"No ESPN fixtures feed for: {', '.join(failed)}."
    games.sort(key=lambda g: (order[g["league"]], g["tip"] is None, g["tip"] or 0))
    n_comp = len({g["league"] for g in games})
    hero(f'{label}<br>{len(games)} match{"es" if len(games) != 1 else ""} · {n_comp} '
         f'competition{"s" if n_comp != 1 else ""}')
    if not games:
        empty_card("No matches in these competitions for this window.",
                   "Try Next 3 Days / Weekend, add competitions, or pick another date in Settings.")
        return
    if len(games) > PAGE_SIZE:
        pages = (len(games) + PAGE_SIZE - 1) // PAGE_SIZE
        n = len(games)
        page = st.selectbox("Page", list(range(1, pages + 1)), key="page_SOCCER", label_visibility="collapsed",
                            format_func=lambda i: f"Matches {(i - 1) * PAGE_SIZE + 1}–{min(i * PAGE_SIZE, n)} of {n}")
        games = games[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    for g in games:
        espn.ensure_odds(LEAGUES[g["league"]], g)
    slates = {}
    for k in sorted({g["league"] for g in games}, key=order.get):
        slates[k], _ = load_team_data(k, [g for g in games if g["league"] == k])

    tab_m, tab_s, tab_p = st.tabs(["🎯 Matchups", "🔥 Streaks", "🎟 Parlays"])
    with tab_s:
        rows = []
        for g in games:
            rows += streakmod.team_streaks(LEAGUES[g["league"]], g, slates[g["league"]]["profiles"])
        note("Current runs from each team's recent results (all competitions ESPN lists): match-total overs, "
             "both teams scoring, scoring, unbeaten, wins and clean sheets. 4+ straight matches to qualify.")
        render_streaks(rows, {g["id"]: g for g in games}, {}, "SOCCER")
    with tab_m:
        note("Poisson goal model from each team's goals for/against (season + last 5, home edge). Cup ties and "
             "national teams with little history in that competition borrow the team's recent results from other "
             "competitions. Bar = away · draw · home; badges compare with ESPN's three-way odds and goal total.")
        matchups_fragment(games, lambda g: slates[g["league"]], set(), "soccer", soccer=True)
    with tab_p:
        legs = []
        for k, sl in slates.items():
            legs += parlays.game_legs(k, "soccer", [g for g in games if g["league"] == k and not_started(g)],
                                      sl["projections"])
        note("Soccer legs: match result (home / draw / away, priced from ESPN's three-way odds) and goal totals at "
             "-110. Hit chances blend the model (60%) with the market (40%). One leg per match. Not betting advice.")
        render_parlays(legs)


def prop_edge(r: dict, m: str, book: tuple | None) -> float:
    """Projection minus the sportsbook line when one exists, else minus the last-5 average."""
    return r[f"P_{m}"] - (book[0] if book else r[f"A5_{m}"])


def news_html(a: dict | None) -> str:
    if not a:
        return ""
    head = esc(a["headline"])
    if a.get("link"):
        head = f'<a href="{esc(a["link"])}" target="_blank" rel="noopener">{head}</a>'
    return f'<div class="news">📰 {head} <span>{esc(newsmod.age(a["published"]))}</span></div>'


def prop_line_info(r: dict, m: str, label: str, kind: str, book: tuple | None, tier: int) -> dict:
    """Main line (posted PrizePicks/Underdog line, else last-10-average estimate, else model line), the alt line
    for the selected tier, the BET/PASS verdict for the shown line, and the full alt ladder."""
    hist = r.get(f"V10_{m}") or []
    proj = r[f"P_{m}"]
    over_p = book[2] if book and len(book) > 2 else None
    under_p = book[3] if book and len(book) > 3 else None
    season = next((s_[1] for s_ in (r.get("SPL") or {}).get(m, []) if s_[0] == "Season average"), None)
    implied = decide.implied_prop_line(kind, m, hist, r.get(f"OF_{m}", 1.0))
    if book:
        main, src, posted = book[0], book[1], True
    elif implied is not None:  # no sportsbook / DFS line: implied consensus line (L5/L10 x opponent defense)
        main, src, posted = implied, "Implied line", False
    else:
        main, src, posted = r[f"L_{m}"], "Model line", False
    ladder = decide.prop_alt_ladder(kind, m, proj, hist, main)
    line = decide.shift_line(kind, m, main, tier)
    if line == main and (posted or implied is not None):
        v = decide.prop_decision(kind, m, label, proj, line, hist, posted, r.get("Class", "ACTIVE"), over_p, under_p)
        shown = src if over_p is None else f"{src} {'O' if proj > line else 'U'} {(over_p if proj > line else under_p) or 0:+.0f}"
        if not posted:
            of = r.get(f"OF_{m}", 1.0)
            v["reason"] += (f" · line = L5/L10 avg × opp defense {of:.2f}" if abs(of - 1) > 0.005
                            else " · line = L5/L10 avg")
    elif line == main:
        v = decide.no_line_verdict(kind, m, label, proj, line, hist, season)
        shown = src
    else:
        row = next((x for x in ladder if abs(x["line"] - line) < 1e-9), None)
        p_over = decide.prop_p_over(kind, m, proj, hist, line)
        hit = row["hit"] if row else None
        bet = bool(row and row["verdict"] == "BET")
        why = (f"Alt line · model {p_over * 100:.0f}%"
               + (f" · L10 {hit * 100:.0f}%" if hit is not None else "") + f" · fair {decide.american(p_over)}")
        if not bet:
            why += " — price too short" if p_over > decide.ALT_MAX_P else " — not enough cushion"
        v = decide.verdict(bet, why, f"Over {line:g} {label}", p_over)
        shown = f"Alt {line - main:+g}"
    return {"main": main, "line": line, "label": shown, "posted": posted, "verdict": v, "ladder": ladder,
            "src": src}


def alt_drawer(info: dict, label: str) -> str:
    ladder = ladder_html(f'Alt {label} ladder · main {info["main"]:g} ({info["src"]})', info["ladder"],
                         lambda x: f'O {x["line"]:g}' + (f' <small>{x["offset"]:+g}</small>' if x["offset"] else ""),
                         ("Line", "Model", "Fair", ""))
    return (f'<details class="pdrawer"><summary>Alt lines ({len(info["ladder"])})</summary>{ladder}'
            '<div class="pt-note">Alt prices aren\'t in any free feed: fair odds are the model\'s. BET needs the '
            'model at 62–85% (higher means the price is too short to be worth it) and the last 10 games ≥ 60% '
            'over that line.</div></details>')


def prop_card(r: dict, m: str, label: str, kind: str, b2b_ids: set[str], info: dict,
              news_item: dict | None = None) -> str:
    lean, strong = propmod.EDGE[kind][m]
    proj = r[f"P_{m}"]
    line = info["line"]
    hist = r.get(f"V10_{m}") or []
    hit_v = float(sum(v > line for v in hist) / len(hist) * 100) if hist else float("nan")
    edge = proj - line
    small = kind == "hockey" or m == "RECS"
    f1 = "{:.2f}" if kind == "hockey" and m in ("G", "A", "PTS") else "{:.1f}"
    chips = "".join(f'<span class="chip {"o" if v > line else "u"}">{v:.0f}</span>' for v in r[f"V5_{m}"])
    badge = edge_badge(edge, lean, strong, "▲ OVER", "▼ UNDER", neg_is_red=True, fmt="{:.2f}" if small else "{:.1f}")
    extra = ""
    if kind == "hockey" and m == "G" and r.get("AnyGoal") is not None:
        extra = f'<span class="pill info">Anytime goal {r["AnyGoal"] * 100:.0f}%</span>'
    b2b = '<span class="pill b2b">B2B</span>' if r["TeamId"] in b2b_ids else ""
    hit = "—" if hit_v != hit_v else f"{hit_v:.0f}%"
    l5_label = "Season" if "season avg" in str(r.get("Info", "")) else "L5 avg"
    others = [mm for mm, _ in propmod.MARKETS[kind] if mm != m][:3]
    mini = "".join(
        f'<span>{esc(mm)} <b>{r[f"P_{mm}"]:{".2f" if kind == "hockey" and mm != "SOG" else ".1f"}}</b> '
        f'<small>(L5 {r[f"A5_{mm}"]:.1f})</small></span>' for mm in others)
    alt = alt_drawer(info, label)
    return (
        f'<div class="card"><div class="phead">{avatar_html(r["Headshot"], r["Player"])}'
        f'<div class="pmeta"><div class="pname">{esc(r["Player"])}</div>'
        f'<div class="psub">{esc(r["Team"])} vs {esc(r["Opp"])} · {esc(r["Info"])} '
        f'{status_pill(r["Class"], r["Status"])} {b2b} {extra}</div></div>{badge}</div>'
        '<div class="pgrid">'
        f'<div class="pbox hl"><span class="lbl">Proj {esc(label)}</span><div class="num">{f1.format(proj)}</div></div>'
        f'<div class="pbox{" book" if info["posted"] and info["line"] == info["main"] else ""}">'
        f'<span class="lbl">{esc(info["label"])}</span><div class="num">{line:g}</div></div>'
        f'<div class="pbox"><span class="lbl">{l5_label}</span><div class="num">{r[f"A5_{m}"]:.{2 if small else 1}f}</div></div>'
        f'<div class="pbox"><span class="lbl">L10 hit</span><div class="num">{hit}</div></div>'
        '</div>'
        f'<div class="l5"><span class="lbl" style="margin-right:4px">Last 5</span>{chips or "—"}</div>'
        f'{verdict_html(info["verdict"])}'
        f'<div class="mini">{mini}</div>{alt}{news_html(news_item)}</div>'
    )


def fetch_games(sk: str, d: date) -> list[dict]:
    lg_ = LEAGUES[sk]
    if sk == "NFL":
        start, days = nfl_week_window(d)
        return espn.week_games(lg_, start, days, today_et)
    return espn.parse_games(lg_, espn.scoreboard(lg_, d, today_et))


def load_team_data(sk: str, games: list[dict]) -> tuple[dict, list[dict]]:
    """Injuries + team profiles (prefetched behind a progress bar) + cached slate projections."""
    lg_ = LEAGUES[sk]
    try:
        inj = espn.injuries(lg_)
    except Exception as e:
        inj = []
        NOTES[f"injuries {sk}"] = f"{sk} injury report unavailable ({e})."
    reqs, seen = [], set()
    for g in games:
        for side in ("home", "away"):
            tid = g[side]["id"]
            if tid and tid not in seen:
                seen.add(tid)
                reqs += teams.needed_requests(lg_, tid)
    fails = prefetch(reqs, f"Loading {lg_.label} team form")
    if fails and lg_.kind != "soccer":
        NOTES[f"teams {sk}"] = (f"{fails} {sk} team request(s) failed; affected teams use a simpler points-only "
                                "model or show no projection.")
    inj_sig = tuple((i["team_id"], i["key"], i["status"]) for i in inj)
    with st.spinner(f"Crunching {sk} projections…"):
        odds_sig = tuple((g["id"], g["odds"].get("spread_home"), g["odds"].get("total")) for g in games)
        slate = compute_slate(sk, tuple(g["id"] for g in games), float(q_sit), inj_sig, bucket(), games, inj, odds_sig)
    return slate, inj


def _player_jobs(lg_: League, g: dict) -> None:
    """Background version of prefetch_players (no Streamlit calls)."""
    for side in ("away", "home"):
        try:
            espn.team_roster(lg_, g[side]["id"])
        except Exception:
            continue
        for a_ in propmod.candidates(lg_, g[side]["id"]):
            if not espn.gamelog_cached(lg_, a_["id"]):
                try:
                    espn.athlete_gamelog(lg_, a_["id"])
                except Exception:
                    pass


def prefetch_players(lg_: League, g: dict) -> None:
    """Rosters, every prop candidate's game log, and NHL season-stat fallbacks for one game (cached)."""
    prefetch([(f"roster|{lg_.key}|{g[s]['id']}", espn.TTL_ROSTER, lambda s=s: espn.team_roster(lg_, g[s]["id"]))
              for s in ("away", "home")], "Loading rosters")
    cands = [a for s in ("away", "home") for a in propmod.candidates(lg_, g[s]["id"])]
    prefetch([(espn.gamelog_key(lg_, a["id"]), espn.TTL_GAMELOG, lambda a=a: espn.athlete_gamelog(lg_, a["id"]))
              for a in cands], "Loading player game logs")
    if lg_.kind == "hockey":
        prefetch([(espn.athlete_stats_key(lg_, a["id"]), espn.TTL_ROSTER,
                   lambda a=a: espn.athlete_stats(lg_, a["id"]))
                  for a in cands if espn.needs_hockey_fallback(lg_, a["id"], g["date"])], "Loading season stats")


def render_team_sport() -> None:
    try:
        with st.spinner("Loading schedule…"):
            games = fetch_games(sport, sel_date)
    except Exception as e:
        hero(f"{sel_date:%a, %b %d}")
        empty_card("Couldn't load the ESPN schedule.", str(e))
        return
    when = f"Week of {nfl_week_window(sel_date)[0]:%b %d}" if sport == "NFL" else f"{sel_date:%a, %b %d}"

    if sport == "NCAAB":
        scope = st.segmented_control("Scope", ["Top 25", "All D1"], default="Top 25", key="ncaab_scope",
                                     label_visibility="collapsed", width="stretch") or "Top 25"
        if scope == "Top 25":
            ranked = [g for g in games if g["home"]["rank"] or g["away"]["rank"]]
            if ranked or not games:
                games = ranked
            else:
                note("No Top 25 teams play on this date — showing all D1 games.")
    hero(f'{when}<br>{len(games)} game{"s" if len(games) != 1 else ""}')
    if not games:
        empty_card(f"No {lg.label} games on this date.", "Pick another date in Settings.")
        return

    if len(games) > PAGE_SIZE:
        pages = (len(games) + PAGE_SIZE - 1) // PAGE_SIZE
        n = len(games)
        page = st.selectbox("Page", list(range(1, pages + 1)), key=f"page_{sport}", label_visibility="collapsed",
                            format_func=lambda i: f"Games {(i - 1) * PAGE_SIZE + 1}–{min(i * PAGE_SIZE, n)} of {n}")
        games = games[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    for g in games:  # live / final games: recover the pregame (closing) line
        espn.ensure_odds(lg, g)
    slate, inj = load_team_data(sport, games)
    projections, b2b = slate["projections"], slate["b2b"]

    tabs = st.tabs(["🎯 Matchups", "📊 Player Props", "🔥 Streaks", "🎟 Parlays"]
                   + (["🩹 Injuries"] if lg.has_injuries else []), key=f"tabs_{sport}", on_change="rerun")
    with tabs[0]:
        note("Model line vs. market. Spread/ML badges name the side the model likes; total badges show the "
             "over/under lean. Number = edge in points (win-probability points for ML).")
        matchups_fragment(games, lambda g: slate, b2b, lg.kind)
    with tabs[1]:
        props_fragment(games, slate, inj)
    with tabs[2]:
        if tabs[2].open or ALL_TABS:  # lazy: game logs for the whole slate load only when the tab is opened
            streaks_fragment(games, slate, inj)
    with tabs[3]:
        parlay_fragment(games, slate, inj)
    if lg.has_injuries:
        with tabs[4]:
            render_injuries(games, inj)


@st.cache_data(ttl=600, show_spinner=False, max_entries=96)
def streaks_for_game(lg_key: str, gid: str, tick: int, _g: dict, _inj: list[dict], _profiles: dict) -> list[dict]:
    lg_ = LEAGUES[lg_key]
    return streakmod.player_streaks(lg_, _g, _inj) + streakmod.team_streaks(lg_, _g, _profiles)


@st.fragment
def streaks_fragment(games: list[dict], slate: dict, inj: list[dict]) -> None:
    pool = [g for g in games if g["state"] != "post"] or games
    labels = {"ALL": f"All games ({len(pool)})"}
    labels.update({g["id"]: f'{g["away"]["abbr"]} @ {g["home"]["abbr"]}'
                   + (f'  ·  {g["tip"]:%a %-I:%M %p}' if g["tip"] is not None else "") for g in pool})
    pick = st.selectbox("Streak games", list(labels), format_func=labels.get, key=f"stk_game_{sport}",
                        label_visibility="collapsed")
    chosen = pool if pick == "ALL" else [g for g in pool if g["id"] == pick]
    rost = [(f"roster|{lg.key}|{g[s]['id']}", espn.TTL_ROSTER, lambda g=g, s=s: espn.team_roster(lg, g[s]["id"]))
            for g in chosen for s in ("away", "home")]
    prefetch(rost, "Loading rosters")
    cands = {a["id"]: a for g in chosen for s in ("away", "home") for a in propmod.candidates(lg, g[s]["id"])}
    prefetch([(espn.gamelog_key(lg, aid), espn.TTL_GAMELOG, lambda aid=aid: espn.athlete_gamelog(lg, aid))
              for aid in cands], "Loading player game logs")
    rows, proj = [], {}
    for g in chosen:
        rows += streaks_for_game(lg.key, g["id"], bucket(), g, inj, slate["profiles"])
        if g["id"] in slate["projections"]:
            try:
                df = compute_props(lg.key, g["id"], float(q_sit), bucket(), g, slate["projections"][g["id"]],
                                   slate["profiles"], slate["base"], slate["b2b"], inj)
                for r in df.to_dict("records"):
                    for mk, _ in propmod.MARKETS[lg.kind]:
                        proj[(r["Id"], mk)] = r.get(f"P_{mk}")
            except Exception:
                pass
    note("Lines this player or team has cleared in every one of their recent games (X/X straight), plus higher "
         "lines cleared in 80%+ of recent games (e.g. 8/10) — filter with 100% / 90%+ / 80%+. Each line is the best "
         "round number for the run: long enough to matter, high enough to be worth playing. Proj = tonight's model projection — ⚠ means the model sees it falling short. "
         "Streaks end; this is history, not a guarantee.")
    view = st.segmented_control("Streak view", ["🔥 Streaks", "🧩 Parlay builder"], default="🔥 Streaks",
                                key=f"stk_view_{sport}", label_visibility="collapsed", width="stretch") or "🔥 Streaks"
    if view == "🧩 Parlay builder":
        streak_parlay(rows, proj, {g["id"]: g for g in games})
        return
    render_streaks(rows, {g["id"]: g for g in chosen}, proj, sport,
                   ctx={"games": {g["id"]: g for g in chosen}, "slate": slate, "inj": inj, "rows": rows})


def streak_parlay(rows: list[dict], proj: dict, games_by_id: dict, slate_games: list[dict] | None = None) -> None:
    """Build a parlay of any size from players on streaks in the selected games."""
    players = {r["id"] for r in rows if r["kind"] == "player"}
    if not players:
        empty_card("No player streaks in these games yet.")
        return
    mk = [(m, lab) for m, lab in propmod.MARKETS.get(lg.kind, []) if any(r["m"] == m for r in rows)]
    labs = dict(mk)
    pick_m = st.pills("Markets", [m for m, _ in mk], default=[m for m, _ in mk], selection_mode="multi",
                      key=f"spl_m_{sport}", format_func=labs.get, label_visibility="collapsed") or [m for m, _ in mk]
    rsel = st.segmented_control("Leg hit rate", ["100% only", "90%+", "80%+"], default="100% only",
                                key=f"spl_r_{sport}", label_visibility="collapsed", width="stretch") or "100% only"
    c1, c2 = st.columns(2)
    min_k = c1.number_input("Min games in the run", 5, 30, 5, 1, key=f"spl_k_{sport}")
    skip_q = c2.toggle("Skip Questionable / DTD", True, key=f"spl_q_{sport}")
    tune = fbmod.factors(lg.key)
    fb_rows = fbmod.load()
    status = learner_status((len(fb_rows), sum(1 for r in fb_rows if r.get("result") in ("hit", "miss"))))
    today_iso = today_et.isoformat()

    def agree_of(r):
        return fbmod.agreement(fb_rows, r["id"], r["m"], r["t"], today_iso)

    def feat_of(r):
        return lrn.feats_from_vals(r.get("vals") or r["last"], r["t"], "over", r["m"], proj.get((r["id"], r["m"])),
                                   len(agree_of(r)))

    def learn(r, p):
        return lrn.predict(status, feat_of(r), p)

    pool = streakmod.build_parlay(rows, proj, 10_000, set(pick_m), int(min_k), skip_q,
                                  {"100% only": 0.999, "90%+": 0.9, "80%+": 0.8}[rsel], tune, learn)
    if len(pool) < 2:
        empty_card("Not enough streak legs with these filters.", "Lower the minimum streak or add markets.")
        return
    n = 2 if len(pool) == 2 else st.slider(f"Legs (up to {len(pool)} players on streaks)", 2, len(pool),
                                           min(5, len(pool)), 1, key=f"spl_n_{sport}_{len(pool)}")
    picks = pool[:n]
    p_all = streakmod.parlay_prob(picks)
    n_games = len({x["gid"] for x in picks})
    fair = decide.american(p_all) if 0.001 < p_all < 0.999 else "—"
    H(f'<div class="card"><div class="ghead"><div class="pl-title">{n}-leg streak parlay'
      f'<div class="psub">{n_games} game{"s" if n_games != 1 else ""} · one leg per player · safest lines first</div>'
      f'</div><span class="badge {"strong" if p_all >= 0.3 else "lean" if p_all >= 0.1 else "muted"}">'
      f'{p_all * 100:.1f}% to hit</span></div>'
      f'<div class="ch-stats"><div><span class="num">{p_all * 100:.1f}%</span><span class="lbl">model hit chance</span></div>'
      f'<div><span class="num">{fair}</span><span class="lbl">fair odds</span></div>'
      f'<div><span class="num">{min(x["p"] for x in picks) * 100:.0f}%</span><span class="lbl">weakest leg</span></div></div>'
      + "".join(
          f'<div class="spl-leg">{avatar_html(x["headshot"], x["name"], 36)}<div class="spl-main">'
          f'<div class="spl-pick">{esc(_short_name(x["name"]))} · <span class="ln">{_fmt_line(x["t"])}+ {esc(x["label"])}</span></div>'
          f'<div class="spl-sub">{esc(x["team"])} {"vs" if x.get("home") else "@"} {esc(x["opp"])} · '
          + (f'{x["k"]}/{x["k"]} straight' if x.get("rate", 1) >= 0.999 else f'{x["wh"]}/{x["w"]} recent ({x["rate"] * 100:.0f}%)')
          + f' · season {x["hits"]}/{x["n"]}'
          + (f' · proj {x["proj"]:.1f}' if x.get("proj") == x.get("proj") and x.get("proj") is not None else "")
          + (f' · <span class="ok">🤖 {esc(", ".join(agree_of(x)))}</span>' if agree_of(x) else "")
          + f'</div></div><div class="spl-p"><span class="num">{x["p"] * 100:.0f}%</span>'
            f'<span class="lbl">hit</span></div></div>' for x in picks)
      + '<div class="pt-note">Each leg\'s chance blends how often the player cleared that line this season with '
        'tonight\'s model projection (capped at 97%), and the parlay multiplies them. Streak lines are alt lines, '
        'usually priced -300 to -1000 each, so compare the book\'s total price with the fair odds above. Legs in '
        'the same game are treated as independent. Not betting advice.</div></div>')
    if st.button(f"📌 Track this {n}-leg parlay (auto-grades when the games finish)", key=f"spl_track_{sport}",
                 width="stretch"):
        added = fbmod.add([{"sport": lg.key, "gid": x["gid"], "pid": x["id"], "name": x["name"], "m": x["m"],
                            "label": x["label"], "t": x["t"], "p": round(x["p"], 4), "agree": len(agree_of(x)),
                            "feat": feat_of(x), "game_date": today_iso,
                            "tier": "100%" if x.get("rate", 1) >= 0.999 else "80%+"} for x in picks], "builder",
                          today_iso)
        st.toast(f"Tracking {added} new leg{'s' if added != 1 else ''} — results tune the builder.")
    feedback_section(rows, games_by_id)


def played_tonight(games: list[dict]) -> dict:
    """{athlete id: {name, pid, sport, gid, team}} for everyone who has played in tonight's started games."""
    out = {}
    for g in games:
        if g.get("state") not in ("in", "post"):
            continue
        box = grade.box_stats(lg, g["id"])
        for aid, row in box.items():
            if aid.startswith("_") or not grade.played(row):
                continue
            out[aid] = {"name": row.get("_name") or aid, "pid": aid, "sport": lg.key, "gid": g["id"],
                        "team": row.get("_team", ""), "final": bool(box.get("_final"))}
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["team"], kv[1]["name"])))


def feedback_section(rows: list[dict], games_by_id: dict | None = None) -> None:
    """Your tracked legs: auto-graded from box scores, betslip screenshots, and the calibration they produce."""
    from datetime import date as _date

    H('<div class="sub-h">📈 Your results · tunes the builder</div>')
    if st.session_state.get("fb_graded_at") != bucket(10):
        st.session_state["fb_graded_at"] = bucket(10)
        fbmod.grade_pending(LEAGUES, grade.box_stats, grade.actual_value)
    if not st.session_state.get(f"fb_pfix_{lg.key}"):
        st.session_state[f"fb_pfix_{lg.key}"] = True
        with st.spinner("Checking tracked legs…"):
            fix_placeholder_p()
    all_rows = fbmod.load()
    cal = fbmod.calibration(all_rows)
    tot = cal.get("_all")
    mk_lab = dict(propmod.MARKETS.get(lg.kind, []))
    if tot:
        f = fbmod.factors(lg.key, cal)
        per = "".join(
            f'<div class="factor"><span>{esc(mk_lab.get(k[1], k[1]))} · {v["n"]} legs</span><span class="v">'
            f'said {v["pred"] * 100:.0f}% · hit {v["actual"] * 100:.0f}% → ×{v["factor"]:.2f}</span></div>'
            for k, v in cal.items() if isinstance(k, tuple) and k[0] == lg.key)
        H(f'<div class="card"><div class="ch-stats"><div><span class="num">{tot["n"]}</span>'
          f'<span class="lbl">legs graded</span></div><div><span class="num">{tot["pred"] * 100:.0f}%</span>'
          f'<span class="lbl">builder said</span></div><div><span class="num">{tot["actual"] * 100:.0f}%</span>'
          f'<span class="lbl">actually hit</span></div></div>{per}'
          f'<div class="pt-note">The builder now multiplies each {esc(lg.key)} leg by '
          f'×{f.get("_default", 1):.2f} (or its market\'s factor once that market has {fbmod.MIN_N}+ graded legs). '
          f'Factors lean on your results more as they add up, and stay between ×{fbmod.FACTOR_RANGE[0]:.2f} '
          f'and ×{fbmod.FACTOR_RANGE[1]:.2f}.</div></div>')
    else:
        note("Track a parlay or upload a betslip — once legs are graded, the builder recalibrates its hit chances "
             "to how your legs actually land.")

    learner_card()
    import_ai_picks(games_by_id or {})
    with st.expander("📷 Upload a betslip screenshot"):
        up = st.file_uploader("Betslip image", type=["png", "jpg", "jpeg", "webp"], key=f"fb_up_{sport}",
                              label_visibility="collapsed")
        recent = list((games_by_id or {}).values())
        try:  # last night's games too (after midnight the slate date has already rolled over)
            seen = {g["id"] for g in recent}
            recent += [g for g in fetch_games(sport, sel_date - timedelta(days=1)) if g["id"] not in seen]
        except Exception:
            pass
        players = played_tonight(recent)
        if not players:  # nobody has played yet: fall back to tonight's rosters so entry is never blocked
            players = {r["id"]: {"name": r["name"], "pid": r["id"], "sport": lg.key, "gid": r["gid"],
                                 "team": r["team"], "final": False, "pre": True}
                       for r in rows if r["kind"] == "player"}
        if up is not None:
            from PIL import Image, ImageOps

            found = []
            try:
                img = ImageOps.exif_transpose(Image.open(up))
                st.image(img, width=240)
            except Exception as e:
                img = None
                st.warning(f"Couldn't open that image ({type(e).__name__}) — try a PNG or JPG screenshot.")
            if img is not None and fbmod.ocr_available():
                with st.spinner("Reading the betslip…"):
                    try:
                        found = fbmod.read_slip(img, list(players.values()))
                    except Exception as e:
                        st.caption(f"Couldn't read the image ({type(e).__name__}).")
                if not found:
                    st.info("Couldn't match any legs on this slip to last night's or tonight's players — "
                            "add them below (it takes a few seconds per leg).")
            elif img is not None:
                st.info("Screenshot reading isn't switched on yet on this server. If the app was just updated, "
                        "open Manage app → ⋮ → Reboot app once so Streamlit installs the OCR package. Meanwhile, "
                        "add the legs below.")
            if found:
                st.caption(f"Found {len(found)} leg{'s' if len(found) != 1 else ''} — check the results, then save.")
                pick = []
                for i, x in enumerate(found):
                    opts = ["hit", "miss", "not settled"]
                    d = opts.index(x["status"]) if x["status"] in opts else 2
                    res = st.segmented_control(f'{x["name"]} · {_fmt_line(x["t"])}+ {mk_lab.get(x["m"], x["m"])}',
                                               opts, default=opts[d], key=f"fb_ocr_{i}") or opts[d]
                    pick.append((x, res))
                if st.button("Save these legs", key="fb_ocr_save", width="stretch"):
                    legs = []
                    for x, res in pick:
                        legs.append({"sport": x["sport"], "gid": x["gid"], "pid": x["pid"], "name": x["name"],
                                     "m": x["m"], "label": mk_lab.get(x["m"], x["m"]), "t": x["t"],
                                     "p": _builder_p(rows, x), "tier": "slip"})
                    fbmod.add(legs, "screenshot")
                    for x, res in pick:
                        idx = fbmod.find(x["pid"], x["m"], x["t"])
                        if res in ("hit", "miss") and idx is not None:
                            fbmod.set_result(idx, res)
                    st.rerun()
        manual_leg_form(rows, mk_lab)

    mine = [(i, r) for i, r in enumerate(all_rows) if r.get("sport") == lg.key]
    if mine:
        open_ = [(i, r) for i, r in mine if not r.get("result")]
        done = [(i, r) for i, r in mine if r.get("result")][-12:]
        with st.expander(f"Tracked legs · {len(open_)} waiting · {len([1 for _, r in mine if r.get('result') in ('hit', 'miss')])} graded"):
            for i, r in reversed(open_ + done):
                badge = {"hit": "✅", "miss": "❌", "void": "➖"}.get(r.get("result"), "⏳")
                act = f' · scored {r["actual"]:g}' if r.get("actual") is not None else ""
                st.markdown(f'{badge} **{esc(r["name"])}** {_fmt_line(r["t"])}+ {esc(r.get("label", r["m"]))} · '
                            f'{r.get("game_date") or r["date"]} · '
                            + (f'said {r["p"] * 100:.0f}%' if r.get("p") else "no estimate")
                            + f'{act}')
                if not r.get("result"):
                    b1, b2, b3 = st.columns(3)
                    if b1.button("✅ Hit", key=f"fb_hit_{i}", width="stretch"):
                        fbmod.set_result(i, "hit"); st.rerun()
                    if b2.button("❌ Miss", key=f"fb_miss_{i}", width="stretch"):
                        fbmod.set_result(i, "miss"); st.rerun()
                    if b3.button("🗑", key=f"fb_del_{i}", width="stretch"):
                        fbmod.remove(i); st.rerun()
            st.download_button("⬇ Backup results (.json)", json.dumps(all_rows, indent=1), "sharpboard-results.json",
                               "application/json", key=f"fb_dl_{sport}", width="stretch")
    up2 = st.file_uploader("Restore results backup", type=["json"], key=f"fb_restore_{sport}")
    if up2 is not None:
        try:
            fbmod.save(json.loads(up2.getvalue()))
            st.toast("Results restored.")
        except Exception as e:
            st.error(f"Not a results backup ({e}).")


def learner_card() -> None:
    rows_ = fbmod.load()
    stt = learner_status((len(rows_), sum(1 for r in rows_ if r.get("result") in ("hit", "miss"))))
    mine = sum(1 for r in rows_ if r.get("feat") and r.get("result") in ("hit", "miss")
               and not str(r.get("source", "")).startswith("ai:"))
    ext = stt["n"] - mine
    if not stt.get("brier_model"):
        body = (f'<div class="pt-note" style="color:var(--text)">Learning: <b>{stt["n"]}/{stt["need"]}</b> graded legs '
                f'({mine} yours · {ext} imported AI picks). It switches on once it has {stt["need"]} and beats the '
                f'builder on legs it hasn\'t seen. Importing AI picks fills this up fastest.</div>'
                f'<div class="ch-bar"><i style="width:{min(100, 100 * stt["n"] / stt["need"]):.0f}%"></i></div>')
    else:
        verdict = (f'<b>Active</b> · blended in at {stt["weight"] * 100:.0f}%' if stt["active"] else
                   "<b>Not active yet</b> — it doesn't beat the builder on held-out legs, so picks are unchanged")
        top = "".join(
            f'<div class="factor"><span>{esc(k)}</span><span class="v {"pos" if v > 0 else "neg"}">'
            f'{"↑ more hits" if v > 0 else "↓ fewer hits"}</span></div>' for k, v in stt.get("top", []))
        body = (f'<div class="pt-note" style="color:var(--text)">{verdict}. Trained on {stt["n"]} graded legs '
                f'({mine} yours · {ext} imported). Accuracy on unseen legs (Brier, lower is better): model '
                f'{stt["brier_model"]:.3f} vs builder {stt["brier_base"]:.3f}.</div>'
                f'<div class="sub-h">What it learned matters most</div>{top}')
    H(f'<div class="card"><div class="sub-h" style="margin-top:0">🧠 Learning model</div>{body}</div>')
    srcs = fbmod.source_stats(rows_)
    if srcs:
        H('<div class="card"><div class="sub-h" style="margin-top:0">🤖 AI sources you\'ve imported</div>' + "".join(
            f'<div class="factor"><span>{esc(k)} · {v["n"]} graded</span><span class="v">'
            f'{v["rate"] * 100:.0f}% hit</span></div>' for k, v in sorted(srcs.items(), key=lambda kv: -kv[1]["rate"]))
          + '<div class="pt-note">When imported sources back a builder leg it shows 🤖 with their names, and the model '
            'learns how much that agreement is worth.</div></div>')


def import_ai_picks(games_by_id: dict) -> None:
    """Paste picks from any AI / tipster source: matched to tonight's players, auto-graded, used for learning."""
    with st.expander("📥 Import AI picks (any site, tweet or Discord)"):
        src = st.text_input("Source name", key=f"ai_src_{sport}", placeholder="e.g. PropAI, Dimers, @someguy")
        txt = st.text_area("Paste the picks — one per line", key=f"ai_txt_{sport}", height=140,
                           placeholder="Jalen Brunson Over 24.5 Points\nUnder 8.5 Rebounds - Bam Adebayo\n"
                                       "J. Williams 5+ AST")
        up = st.file_uploader("…or a screenshot of their picks", type=["png", "jpg", "jpeg", "webp"],
                              key=f"ai_img_{sport}")
        if up is not None and fbmod.ocr_available():
            try:
                import pytesseract
                from PIL import Image, ImageOps

                txt = (txt or "") + "\n" + pytesseract.image_to_string(fbmod._prep(ImageOps.exif_transpose(Image.open(up))))
            except Exception as e:
                st.caption(f"Couldn't read the screenshot ({type(e).__name__}).")
        elif up is not None:
            st.caption("Screenshot reading isn't switched on on this server yet — paste the text instead.")
        if not (txt or "").strip():
            st.caption("Picks are graded automatically from the box scores, and each source's hit rate is tracked. "
                       "Every graded pick also trains the 🧠 learning model.")
            return
        players = {}
        for g in games_by_id.values():
            players.update(game_players(g)[0])
        found = fbmod.parse_picks(txt, list(players.values()))
        if not found:
            st.caption("No picks matched tonight's players — check the names / markets (Points, Rebounds, "
                       "Assists, PRA, 3s aren't supported yet).")
            return
        mk_lab = dict(propmod.MARKETS.get(lg.kind, []))
        st.caption(f"Matched {len(found)} pick{'s' if len(found) != 1 else ''}:")
        for x in found:
            st.markdown(f'• **{esc(x["name"])}** {x["side"].title()} {x["line"]:g} {esc(mk_lab.get(x["m"], x["m"]))}')
        if st.button(f"Save {len(found)} picks from {src or 'this source'}", key=f"ai_save_{sport}",
                     disabled=not src.strip(), width="stretch"):
            legs = []
            for x in found:
                leg = {"sport": lg.key, "gid": x["gid"], "pid": x["pid"], "name": x["name"], "m": x["m"],
                       "label": mk_lab.get(x["m"], x["m"]), "t": x["t"], "side": x["side"], "line": x["line"],
                       "tier": "ai", "game_date": today_et.isoformat(), "source": f"ai:{src.strip()}"}
                leg["feat"] = leg_feat(leg, today_et)
                leg["p"] = lrn.base_p(leg["feat"]) if leg["feat"] else None
                legs.append(leg)
            n = fbmod.add(legs, f"ai:{src.strip()}", today_et.isoformat())
            st.toast(f"Saved {n} picks from {src.strip()}.")
            st.rerun()
        if not src.strip():
            st.caption("Name the source to save them.")


def game_players(g: dict) -> tuple[dict, str]:
    """Players for one game: who actually played (box score), else both rosters. Returns (players, source)."""
    box = grade.box_stats(lg, g["id"]) if g.get("state") in ("in", "post") else {}
    out = {}
    for aid, row in box.items():
        if not aid.startswith("_") and grade.played(row):
            out[aid] = {"name": row.get("_name") or aid, "pid": aid, "sport": lg.key, "gid": g["id"],
                        "team": row.get("_team", "")}
    if out:
        return dict(sorted(out.items(), key=lambda kv: (kv[1]["team"], kv[1]["name"]))), (
            "box score" if box.get("_final") else "live box score")
    for side in ("away", "home"):
        try:
            for a in espn.roster_athletes(espn.team_roster(lg, g[side]["id"])):
                out[a["id"]] = {"name": a["name"], "pid": a["id"], "sport": lg.key, "gid": g["id"],
                                "team": g[side]["abbr"]}
        except Exception:
            continue
    src = "rosters (box score not available yet)" if g.get("state") in ("in", "post") else "rosters (not started)"
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["team"], kv[1]["name"]))), src


def manual_leg_form(rows: list[dict], mk_lab: dict) -> None:
    """Add a leg by hand: date -> game -> player (one game at a time, so no player goes missing)."""
    st.caption("Or add a leg by hand — pick the date, the game, then the player:")
    try:
        today_games = fetch_games(sport, today_et)
    except Exception:
        today_games = []
    started = any(g["state"] in ("in", "post") for g in today_games)
    d = st.date_input("Game date", value=today_et if started else today_et - timedelta(days=1),
                      max_value=today_et, key=f"fb_man_d_{sport}")
    try:
        day_games = today_games if d == today_et else fetch_games(sport, d)
    except Exception as e:
        day_games = []
        st.caption(f"Couldn't load that day's games ({type(e).__name__}).")
    if not day_games:
        st.caption(f"No {lg.label} games on {d:%b %d}.")
        return
    status = {"post": "Final", "in": "Live"}
    gid = st.selectbox("Game", [g["id"] for g in day_games], key=f"fb_man_g_{sport}_{d}",
                       format_func=lambda i: next(
                           f'{g["away"]["abbr"]} @ {g["home"]["abbr"]} · '
                           + status.get(g["state"], f'{g["tip"]:%-I:%M %p}' if g["tip"] is not None else "")
                           for g in day_games if g["id"] == i))
    g = next(x for x in day_games if x["id"] == gid)
    with st.spinner("Loading players…"):
        players, src = game_players(g)
    if not players:
        st.caption("No players found for this game yet.")
        return
    st.caption(f"{len(players)} players · from the {src}")
    c1, c2 = st.columns(2)
    pid = c1.selectbox("Player", list(players), key=f"fb_man_p_{sport}_{gid}",
                       format_func=lambda i: f'{players[i]["name"]} · {players[i]["team"]}')
    m = c2.selectbox("Market", list(mk_lab), format_func=mk_lab.get, key=f"fb_man_m_{sport}")
    c3, c4 = st.columns(2)
    t = c3.number_input("Line (N+)", 0.5, 600.0, 5.0, 0.5, key=f"fb_man_t_{sport}")
    res = c4.segmented_control("Result", ["hit", "miss", "not settled"], default="not settled",
                               key=f"fb_man_r_{sport}") or "not settled"
    if st.button("Add leg", key=f"fb_man_add_{sport}", width="stretch"):
        x = {**players[pid], "m": m, "t": float(t)}
        fbmod.add([{"sport": x["sport"], "gid": x["gid"], "pid": pid, "name": x["name"], "m": m,
                    "label": mk_lab[m], "t": float(t), "p": _builder_p(rows, x, d), "tier": "manual", "p_src": "history",
                    "game_date": d.isoformat()}], "manual")
        idx = fbmod.find(pid, m, float(t))
        if res in ("hit", "miss") and idx is not None:
            fbmod.set_result(idx, res)
        st.toast(f"Added {players[pid]['name']} {_fmt_line(float(t))}+ {mk_lab[m]}.")
        st.rerun()


def _builder_p(rows: list[dict], x: dict, game_date: date | None = None) -> float | None:
    """What the builder would have said for a leg before its game: the matching streak row's chance when the leg
    is on tonight's board, else the player's hit rate at that line in his games before that date (same smoothing
    as the builder). None if no game log is available."""
    r = next((r for r in rows if r["kind"] == "player" and r["id"] == x["pid"] and r["m"] == x["m"]
              and r.get("t") == x["t"]), None)
    if r is not None and game_date in (None, today_et):
        return round(streakmod.leg_prob(r, None), 4)
    return hist_p(x["pid"], x["m"], x["t"], game_date or today_et, x.get("gid"))


def hist_vals(pid: str, m: str, before: date, gid: str | None = None) -> list[float] | None:
    """The player's stat in every game before the leg's game, newest first (ESPN season log, cached). With gid
    the cutoff is that exact game, so its own result never leaks in."""
    try:
        raw = espn.cached_gamelog(lg, pid) or espn.athlete_gamelog(lg, pid)
        logs = espn.parse_gamelog(raw, lg.kind, today_et + timedelta(days=1))
    except Exception:
        return None
    hit = next((g for g in logs if gid and str(g.get("eid")) == str(gid)), None)
    cutoff = hit["date"] if hit else before
    logs = [g for g in logs if g["date"] < cutoff]
    if lg.kind == "basketball":
        logs = [g for g in logs if g.get("MIN", 1) > 0]
    return [propmod.stat_value(lg.kind, m, g) for g in logs] if len(logs) >= 3 else None


def hist_p(pid: str, m: str, t: float, before: date, gid: str | None = None) -> float | None:
    """Smoothed hit rate of `t+` before the leg's game."""
    vals = hist_vals(pid, m, before, gid)
    if not vals:
        return None
    return round(min(0.97, (sum(v >= t for v in vals) + 1) / (len(vals) + 2)), 4)


def leg_feat(r: dict, before: date | None = None) -> list[float] | None:
    """Learner features for a saved leg, from the player's log before its game."""
    d = before or (date.fromisoformat(r["game_date"]) if r.get("game_date") else
                   date.fromisoformat(r.get("date", today_et.isoformat())))
    vals = hist_vals(str(r.get("pid")), r["m"], d, r.get("gid"))
    if not vals:
        return None
    return lrn.feats_from_vals(vals, float(r["t"]), r.get("side", "over"), r["m"],
                               agree=int(r.get("agree", 0)), external=str(r.get("source", "")).startswith("ai:"),
                               line=r.get("line"))


@st.cache_data(ttl=600, show_spinner=False)
def learner_status(sig: tuple) -> dict:
    """Train on every graded leg with features (yours + imported AI picks). sig busts the cache on new results."""
    return lrn.train(fbmod.load())


def fix_placeholder_p() -> int:
    """Older hand-entered / screenshot legs were saved with a flat 60% placeholder: replace it with the real
    pre-game hit rate (once per leg). Also back-fills learner features on legs saved before the model existed."""
    rows_ = fbmod.load()
    n = 0
    for r in rows_:
        if r.get("sport") == lg.key and "feat" not in r and r.get("m") and r.get("pid"):
            r["feat"] = leg_feat(r, (date.fromisoformat(r["date"]) - timedelta(days=1)) if not r.get("gid") else None)
            n += 1
        if r.get("sport") != lg.key or r.get("p_src") or r.get("source") not in ("manual", "screenshot"):
            continue
        if abs((r.get("p") or 0) - 0.6) > 1e-9:
            r["p_src"] = "saved"
            continue
        try:
            d = date.fromisoformat(r.get("game_date") or r.get("date"))
        except Exception:
            d = today_et
        p = hist_p(str(r.get("pid")), r["m"], float(r["t"]), d - timedelta(days=1), r.get("gid"))
        r["p"], r["p_src"] = (p, "history") if p is not None else (None, "none")
        n += 1
    if n:
        fbmod.save(rows_)
    return n


STK_COLORS = ["#3ddc84", "#a855f7", "#f59e0b", "#3b82f6", "#f43f5e", "#14b8a6", "#eab308", "#ec4899"]
STK_PER_SECTION = 15


def _short_name(name: str) -> str:
    parts = name.split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else name


def _fmt_line(t: float) -> str:
    return f"{t:.0f}" if float(t).is_integer() else f"{t:g}"


def streak_row(r: dict, g: dict | None, proj: dict) -> str:
    vis = avatar_html(r["headshot"] or r["logo"], r["name"], 40)
    if r["kind"] == "player":
        pick = f'{esc(_short_name(r["name"]))} · <span class="ln">{_fmt_line(r["t"])}+ {esc(r["label"])}</span>'
    elif r["m"] == "TEAM":
        pick = f'{esc(r["name"])} · <span class="ln">{_fmt_line(r["t"])}+ {esc(r["label"])}</span>'
    else:
        pick = f'{esc(_short_name(r["name"]) if r["kind"] == "fighter" else r["name"])} · ' \
               f'<span class="ln">{esc(r["label"])}</span>'
    where = (f'{r["team"]} {"vs" if r.get("home") else "@"} {r["opp"]}' if r["kind"] != "fighter"
             else f'vs {r["opp"]}')
    bits = [esc(where)]
    if r.get("avg") is not None:
        bits.append(f'avg {r["avg"]:.1f}')
    if r.get("hits") is not None and r.get("n"):
        bits.append(f'{"season" if r["kind"] != "fighter" else "last"} {r["hits"]}/{r["n"]}')
    p = proj.get((r["id"], r["m"])) if r["kind"] == "player" else None
    if p is not None and p == p:
        ok = p >= r["t"]
        bits.append(f'proj {p:.1f} <span class="{"ok" if ok else "warn"}">{"✓" if ok else "⚠"}</span>')
    if r.get("status"):
        bits.append(f'<span class="warn">{esc(r["status"])}</span>')
    if r.get("t") is not None:
        vals = list(reversed(r["last"]))           # oldest -> newest, left to right
        opps = list(reversed((r.get("opps") or [])[:len(r["last"])]))
        top = max(max(vals), r["t"]) or 1
        bar_h = 40
        line_px = bar_h * r["t"] / top
        d = 0 if all(float(v).is_integer() for v in vals) else 1
        cols = "".join(
            f'<div class="c"><span class="v{"" if v >= r["t"] else " miss"}">{v:.{d}f}</span>'
            f'<i class="{"" if v >= r["t"] else "miss"}" style="height:{max(3, bar_h * v / top):.0f}px"></i></div>'
            for v in vals)
        dates = list(reversed((r.get("dates") or [])[:len(r["last"])]))
        if len(dates) != len(vals):
            dates = [""] * len(vals)
        if len(opps) != len(vals):
            opps = [""] * len(vals)
        labels = "".join(f'<span>{esc(o[:4])}<small>{esc(d_)}</small></span>' for o, d_ in zip(opps, dates)) \
            if (any(opps) or any(dates)) else ""
        n = len(vals)
        hits10 = sum(v >= r["t"] for v in vals)
        chart = (f'<div class="stk-chart"><div class="stk-cap"><span>Last {n} games · oldest → newest</span>'
                 f'<span><b>{hits10}/{n}</b> at {_fmt_line(r["t"])}+</span></div>'
                 f'<div class="stk-bars" style="grid-template-columns:repeat({n},1fr)">{cols}'
                 f'<div class="ln" style="bottom:{line_px:.0f}px"></div></div>'
                 + (f'<div class="stk-opps" style="grid-template-columns:repeat({n},1fr)">{labels}</div>' if labels else "")
                 + '<div class="stk-leg"><i></i>cleared <i class="miss"></i>missed '
                   f'<span class="dash"></span>{_fmt_line(r["t"])}+ line</div></div>')
    else:
        hit = r["k"]
        chart = ('<div class="stk-chart"><div class="stk-cap"><span>Recent results · newest first</span>'
                 f'<span><b>{hit}</b> in a row</span></div><div class="stk-chips">' + "".join(
                     f'<span class="{"" if i < hit else "miss"}">{esc(str(v))}'
                     + (f'<small>{esc((r.get("dates") or [""] * 9)[i])}</small>' if r.get("dates") else "") + '</span>'
                     for i, v in enumerate(r["last"][:8])) + "</div></div>")
    return (f'<div class="stk-item"><div class="stk-row">{vis}<div class="stk-main"><div class="stk-pick">{pick}</div>'
            f'<div class="stk-sub">{" · ".join(bits)}</div></div>'
            f'<div class="stk-k{"" if r.get("rate", 1) >= 0.999 else " rate"}"><span class="num">{r.get("wh", r["k"])}/'
            f'{r.get("w", r["k"])}</span><span class="lbl">'
            f'{"straight" if r.get("rate", 1) >= 0.999 else f"{r['rate'] * 100:.0f}% hit"}'
            f'</span></div></div>'
            f'{chart}</div>')


def render_streaks(rows: list[dict], games_by_id: dict, proj: dict, sport_key: str, ctx: dict | None = None) -> None:
    if not rows:
        empty_card("No active streaks yet.", "Streaks need 5+ recent games (4 for soccer, 2 bouts for UFC).")
        return
    kind_ = LEAGUES[sport_key].kind if sport_key in LEAGUES else ""
    mk_rank = {lab: i for i, (_, lab) in enumerate(propmod.MARKETS.get(kind_, []))}
    first: dict[str, int] = {}
    for i, r in enumerate(rows):
        first.setdefault(r["label"], i)
    order = sorted(first, key=lambda lab: (lab not in mk_rank, mk_rank.get(lab, 0), first[lab]))
    pick = st.pills("Streak market", ["All"] + order, default="All", key=f"stk_mkt_{sport_key}",
                    label_visibility="collapsed") or "All"
    if any(r.get("rate", 1) < 0.999 for r in rows):
        rsel = st.segmented_control("Hit rate", ["100% only", "90%+", "80%+"], default="80%+",
                                    key=f"stk_rate_{sport_key}", label_visibility="collapsed",
                                    width="stretch") or "80%+"
        floor_r = {"100% only": 0.999, "90%+": 0.9, "80%+": 0.8}[rsel]
        rows = [r for r in rows if r.get("rate", 1) >= floor_r - 1e-9]
    html = []
    for i, label in enumerate(order):
        if pick != "All" and label != pick:
            continue
        sec = sorted((r for r in rows if r["label"] == label),
                     key=lambda r: (-r.get("rate", 1), -r.get("w", r["k"]),
                                    -(r["t"] or 0) / max(r.get("avg") or 1, 1e-9)))
        shown = sec if pick != "All" else sec[:STK_PER_SECTION]
        head = (f'<div class="stk-h"><span class="dot" style="background:{STK_COLORS[i % len(STK_COLORS)]}">'
                f'</span>{esc(label)}<span class="cnt">· {len(sec)}</span></div>')
        if ctx is None:  # not clickable (soccer / UFC): one static card
            html.append(head + '<div class="card">' + "".join(
                streak_row(r, games_by_id.get(r["gid"]), proj) for r in shown) + "</div>")
            continue
        H("".join(html) + head)
        html = []
        for r in shown:
            key = f'stkrow_{sport_key}_{i}_{r["kind"]}_{r["id"]}_{r["m"]}_{_fmt_line(r["t"]) if r.get("t") is not None else ""}'
            with st.container(key=key):
                H(streak_row(r, games_by_id.get(r["gid"]), proj))
                if r["kind"] == "player" and st.button("Open player profile", key=f"btn_{key}"):
                    player_dialog(r["id"], r["gid"], r["m"], ctx)
    if html:
        H("".join(html))


GLOG_COLS = {"basketball": [("MIN", "MIN"), ("PTS", "PTS"), ("REB", "REB"), ("AST", "AST"), ("PRA", "PRA")],
             "football": [("PASS", "Pass"), ("RUSH", "Rush"), ("REC", "RecYd"), ("RECS", "Rec")],
             "hockey": [("G", "G"), ("A", "A"), ("PTS", "PTS"), ("SOG", "SOG")]}


def book_for(g: dict) -> dict:
    return {**cached_lines(lg.key, bucket(5)), **cached_odds_api(lg.key, g["home"]["name"], g["away"]["name"],
                                                                  bucket(30)),
            **cached_espn_props(lg.key, g["id"], bucket(5))}


@st.dialog("Player profile", width="large")
def player_dialog(pid: str, gid: str, m0: str, ctx: dict) -> None:
    """Full player card: tonight's projections and lines for every market, active streaks, the prop report
    for the chosen market, and the recent game log."""
    kind = lg.kind
    g, slate, inj = ctx["games"][gid], ctx["slate"], ctx["inj"]
    markets = propmod.MARKETS[kind]
    mlabels = dict(markets)
    row = None
    if gid in slate["projections"]:
        try:
            df = compute_props(lg.key, gid, float(q_sit), bucket(), g, slate["projections"][gid], slate["profiles"],
                               slate["base"], slate["b2b"], inj)
            hit = df[df["Id"] == pid] if not df.empty else df
            row = hit.to_dict("records")[0] if len(hit) else None
        except Exception:
            row = None
    logs = []
    raw = espn.cached_gamelog(lg, pid)
    if raw:
        try:
            logs = espn.parse_gamelog(raw, kind, g["date"])
        except Exception:
            logs = []
    if kind == "basketball":
        logs = [x for x in logs if x.get("MIN", 1) > 0]
    mine = [x for x in ctx["rows"] if x["kind"] == "player" and x["id"] == pid]
    base = mine[0] if mine else {}
    name = row["Player"] if row else base.get("name", "Player")
    team = row["Team"] if row else base.get("team", "")
    opp = row["Opp"] if row else base.get("opp", "")
    shot = (row or {}).get("Headshot") or base.get("headshot", "")
    status = (row or {}).get("Status") or base.get("status") or "Active"
    pos = (row or {}).get("Pos", "")
    H(f'<div class="card"><div class="ev-head"><div style="display:flex;gap:12px;align-items:center">'
      f'{avatar_html(shot, name, 64)}<div><div class="ev-name">{esc(name)}</div>'
      f'<div class="psub">{esc(" · ".join(x for x in (pos, team, f"vs {opp}" if opp else "") if x))}</div>'
      f'<div class="psub">{len(logs)} games logged · {esc(status)}</div></div></div></div></div>')

    # tonight: every market at a glance
    book = book_for(g)
    trs = []
    infos = {}
    for m, label in markets:
        vals = [propmod.stat_value(kind, m, x) for x in logs]
        avg = float(np.mean(vals)) if vals else None
        l5 = float(np.mean(vals[:5])) if vals else None
        stk = next((x for x in mine if x["m"] == m), None)
        if row is not None and row.get(f"P_{m}") == row.get(f"P_{m}"):
            info = prop_line_info(row, m, label, kind, linesmod.lookup(book, pid, name, m), 0)
            infos[m] = info
            v = info["verdict"]["verdict"]
            side = "O" if row[f"P_{m}"] > info["line"] else "U"
            pill = (f'<span class="vpill {"bet" if v == "BET" else "pass"} sm">{v}</span>'
                    if v in ("BET", "PASS") else "")
            proj_txt, line_txt = f'{row[f"P_{m}"]:.1f}', f'{side} {info["line"]:g}'
        else:
            pill, proj_txt, line_txt = "", "—", "—"
        trs.append(f'<div class="pp-r"><span class="sg" title="{esc(label)}">{esc(m)}</span><span>{proj_txt}</span>'
                   f'<span>{esc(line_txt)}</span><span>{"—" if l5 is None else f"{l5:.1f}"}</span>'
                   f'<span>{"—" if avg is None else f"{avg:.1f}"}</span>'
                   f'<span class="stk">{(_fmt_line(stk["t"]) + "+ <small>" + str(stk.get("wh", stk["k"])) + "/" + str(stk.get("w", stk["k"])) + "</small>") if stk else "—"}</span>'
                   f'<span>{pill}</span></div>')
    H('<div class="card"><div class="sub-h" style="margin-top:0">Tonight</div><div class="ptab pp">'
      '<div class="pp-r head"><span>Mkt</span><span>Proj</span><span>Line</span><span>L5</span>'
      '<span>Szn</span><span>Streak</span><span></span></div>' + "".join(trs) + "</div>"
      '<div class="pt-note">Line = sportsbook / DFS line, else the implied line (L5/L10 × opponent defense). '
      'O/U = the side the model takes.</div></div>')

    if row is not None and infos:
        opts = [m for m, _ in markets if m in infos]
        m = st.segmented_control("Report market", opts, default=m0 if m0 in opts else opts[0],
                                 key=f"pp_m_{pid}", format_func=lambda k: mlabels[k],
                                 label_visibility="collapsed", width="stretch") or (m0 if m0 in opts else opts[0])
        info = infos[m]
        arts = cached_news(lg.key, (g["away"]["id"], g["home"]["id"]), bucket(5))
        H(report.report(SPORT_NAME.get(sport, sport), row, m, mlabels[m], kind, info,
                        report.confidence(row, m, info), verdict_html(info["verdict"]), alt_drawer(info, mlabels[m]),
                        news_html(newsmod.for_player(arts, pid, name))))
    elif not logs:
        empty_card("No game log on ESPN for this player yet.")

    if logs:
        cols = GLOG_COLS.get(kind, [])
        lines_ = {}
        for x in mine:  # perfect-streak line first
            lines_.setdefault(x["m"], x["t"])
        tmpl = f"grid-template-columns:52px 52px repeat({len(cols)},1fr)"
        head = "".join(f"<span>{esc(c[1])}</span>" for c in cols)
        body = []
        for x in logs[:15]:
            cells = []
            for c, _ in cols:
                v = propmod.stat_value(kind, c, x) if c != "MIN" else x.get("MIN", 0)
                t = lines_.get(c)
                cls = "" if t is None else (" hit" if v >= t else " miss")
                cells.append(f'<span class="v{cls}">{v:.0f}</span>')
            body.append(f'<div class="gl-r" style="{tmpl}"><span>{x["date"].month}/{x["date"].day}</span>'
                        f'<span>{"@" if x.get("away") else "vs"} {esc(x.get("opp", ""))}</span>{"".join(cells)}</div>')
        H(f'<div class="card"><div class="sub-h" style="margin-top:0">Game log (last {min(15, len(logs))})</div>'
          f'<div class="ptab"><div class="gl-r head" style="{tmpl}"><span>Date</span><span>Opp</span>{head}</div>'
          + "".join(body) + '</div><div class="pt-note">Green / red = cleared or missed the streak line for '
          'that stat.</div></div>')


@st.fragment
def props_fragment(games: list[dict], slate: dict, inj: list[dict]) -> None:
    """Fragment: switching game / market / team re-renders only this block."""
    kind = lg.kind
    playable = [g for g in games if g["id"] in slate["projections"]]
    if not playable:
        empty_card("Player props need team projections, which aren't available for these games yet.")
        return
    labels = {g["id"]: f'{g["away"]["abbr"]} @ {g["home"]["abbr"]}'
              + (f'  ·  {g["tip"]:%a %-I:%M %p}' if g["tip"] is not None else "") for g in playable}
    gid = st.selectbox("Game", list(labels), format_func=labels.get, key=f"prop_game_{sport}",
                       label_visibility="collapsed")
    g = next(x for x in playable if x["id"] == gid)

    prefetch_players(lg, g)
    others = [x for x in playable if x["id"] != gid and x["state"] != "post"]
    warm_in_background(f"players|{lg.key}|{sel_date}|{bucket(30)}",
                       [lambda x=x: _player_jobs(lg, x) for x in others])

    df = compute_props(lg.key, gid, float(q_sit), bucket(), g, slate["projections"][gid], slate["profiles"],
                       slate["base"], slate["b2b"], inj)
    all_players = df[["Id", "Player", "Team"]].to_dict("records") if not df.empty else []
    with st.spinner("Checking sportsbook lines and news…"):
        book = {**cached_lines(lg.key, bucket(5)), **cached_odds_api(lg.key, g["home"]["name"], g["away"]["name"],
                                                                      bucket(30)),
                **cached_espn_props(lg.key, gid, bucket(5))}
        arts = cached_news(lg.key, (g["away"]["id"], g["home"]["id"]), bucket(5))
    markets = propmod.MARKETS[kind]
    mlabels = dict(markets)
    m = st.segmented_control("Market", [k for k, _ in markets], default=markets[0][0], key=f"mkt_{sport}",
                             format_func=lambda k: mlabels[k], label_visibility="collapsed",
                             width="stretch") or markets[0][0]
    side = st.segmented_control("Team", ["Both", g["away"]["abbr"], g["home"]["abbr"]], default="Both",
                                key=f"side_{sport}_{gid}", label_visibility="collapsed", width="stretch") or "Both"
    steps = decide.ALT_STEPS.get((kind, m), (1, 2, 3))
    tier_lbl = {t: ("Main line" if t == 0 else f"Alt {'+' if t > 0 else '−'}{steps[abs(t) - 1]:g}")
                for t in (-3, -2, -1, 0, 1, 2, 3)}
    up_only = (kind, m) in decide.LADDER_UP_ONLY
    tier = st.segmented_control("Alt line", list(tier_lbl), default=0, key=f"tier_{sport}",
                                format_func=tier_lbl.get, label_visibility="collapsed", width="stretch")
    tier = 0 if tier is None else tier
    if up_only and tier < 0:  # goals / assists / points ladders start at 0.5
        note(f"{mlabels[m]} lines start at 0.5, so only the + alt tiers apply — showing the main line.")
        tier = 0
    n_book = sum(1 for k in book if k[1] == m)
    note((f"{n_book} real {mlabels[m]} lines found (ESPN sportsbook feed, The Odds API, PrizePicks / Underdog). "
          if n_book else "No sportsbook or DFS lines posted for this game yet. ")
         + "Players without one get an implied line: last-5 / last-10 average × the opponent's defense for that "
           "stat, rounded to .5. Every card is judged 🟢 BET / 🟡 PASS on projection vs. line"
         + ("" if _odds_api_key() else " · add an ODDS_API_KEY in Streamlit secrets for real sportsbook lines")
         + ".")
    if df.empty:
        empty_card("No players with recent game logs for this matchup yet.")
        return
    if side != "Both":
        df = df[df["Team"] == side]
    floor = propmod.MIN_AVG.get(kind, {}).get(m)
    if floor is not None:
        df = df[df[f"A5_{m}"].fillna(0).ge(floor) | df[f"P_{m}"].ge(floor)]
    if df.empty:
        empty_card("No players qualify for this market.")
        return
    recs = df.to_dict("records")
    for r in recs:
        r["_info"] = prop_line_info(r, m, mlabels[m], kind, linesmod.lookup(book, r["Id"], r["Player"], m), tier)
        r["_e"] = abs(r[f"P_{m}"] - r["_info"]["line"])
    # BET cards first, then posted lines, then biggest edges
    box = grade.box_stats(lg, gid) if g["state"] == "post" else {}
    for r in recs:
        r["_conf"] = report.confidence(r, m, r["_info"])
        if box:
            act = grade.actual_value(kind, m, box.get(str(r["Id"])))
            side = "Over" if r[f"P_{m}"] > r["_info"]["line"] else "Under"
            r["_actual"], r["_result"] = act, grade.result(side, r["_info"]["line"], act)
    rank = {"BET": 0, "PASS": 1, "NO LINE": 2}
    recs.sort(key=lambda r: (rank.get(r["_info"]["verdict"]["verdict"], 3),
                             not r["_info"]["verdict"].get("lean", True), -r["_conf"], -r["_e"]))
    cnt = {k: sum(r["_info"]["verdict"]["verdict"] == k for r in recs) for k in rank}
    H(f'<div class="note"><b>{cnt["BET"]}</b> BET · <b>{cnt["PASS"]}</b> PASS'
      + (f' · <b>{cnt["NO LINE"]}</b> too few games' if cnt["NO LINE"] else "") + f' on {esc(mlabels[m])}'
      + (f" at {esc(tier_lbl[tier])}" if tier else "") + " · tap a player below for the full report</div>")
    H('<div class="pklist">' + "".join(report.pick_row(r, m, mlabels[m], r["_info"], r["_conf"]) for r in recs)
      + "</div>")
    by_id = {r["Id"]: r for r in recs}
    names = {x["Id"]: f'{x["Player"]} · {x["Team"]}' for x in all_players}  # stable across markets / filters
    order = list(by_id) + [i for i in names if i not in by_id]
    pid = st.selectbox("Player report", order, key=f"rep_{sport}_{gid}", label_visibility="collapsed",
                       format_func=lambda i: f"📋  {names.get(i, i)}")
    if pid not in by_id:
        empty_card(f"No {mlabels[m]} line for {names.get(pid, 'this player')}.",
                   "Pick another player or switch markets.")
        return
    r = by_id[pid]
    H(report.report(SPORT_NAME.get(sport, sport), r, m, mlabels[m], kind, r["_info"], r["_conf"],
                    verdict_html(r["_info"]["verdict"]), alt_drawer(r["_info"], mlabels[m]),
                    news_html(newsmod.for_player(arts, r["Id"], r["Player"]))))


def render_injuries(games: list[dict], inj: list[dict]) -> None:
    arts = cached_news(lg.key, (), bucket(5))  # league feed only: one request keeps this tab instant
    note(f"ESPN injury report · Questionable / Day-to-Day players count as sitting {q_sit:.0%} of the time "
         "in projections (change in Settings).")
    order = {"OUT": 0, "Q": 1, "ACTIVE": 2}
    html = []
    for g in games:
        for t, o in ((g["away"], g["home"]), (g["home"], g["away"])):
            rows = sorted(espn.team_injuries(inj, t), key=lambda i: order[espn.injury_class(i["status"])])
            n_out = sum(espn.injury_class(i["status"]) == "OUT" for i in rows)
            head = (f'<span class="pill out">{n_out} out</span>' if n_out
                    else '<span class="pill ok">Healthy</span>' if not rows else "")
            body = "".join(
                f'<div class="irow"><div class="who"><div class="nm">{esc(i["player"])}'
                f'<small style="color:var(--muted);font-weight:500"> {esc(i["pos"])}</small></div>'
                f'<div class="dt">{esc(i["detail"])}</div>'
                f'{news_html(newsmod.for_player(arts, i["athlete_id"], i["player"], 7))}</div>'
                f'<span class="pill {dict(OUT="out", Q="q").get(espn.injury_class(i["status"]), "ok")}">'
                f'{esc(i["status"])}</span></div>' for i in rows) or '<div class="empty">No players listed.</div>'
            html.append(f'<div class="card"><div class="ihead">{logo_html(t["logo"], t["abbr"])}<div class="tinfo">'
                        f'<div class="tabbr">{esc(t["abbr"])}</div><div class="tname">{esc(t["short"])} · vs '
                        f'{esc(o["abbr"])}</div></div>{head}</div>{body}</div>')
    H("".join(html))


# --------------------------------------------------------------------------
# UFC
# --------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False, max_entries=8)
def compute_card(event_id: str, tick: int, _card: dict) -> list[dict]:
    out = []
    for b in _card["bouts"]:
        fa_, fb_ = b["fighters"]
        try:
            pa_ = ufc.fighter_profile(fa_["id"], fa_, b.get("weight_class", ""))
            pb_ = ufc.fighter_profile(fb_["id"], fb_, b.get("weight_class", ""))
            pred = ufc.predict_bout(b, pa_, pb_)
        except Exception as e:
            NOTES[f"bout {b['id']}"] = f"Prediction failed for {fa_['name']} vs {fb_['name']}: {e}"
            continue
        try:
            props_ = ufc_props.breakdown(b, pa_, pb_, pred)
        except Exception as e:
            NOTES[f"bout props {b['id']}"] = f"Round/method props failed for {fa_['name']} vs {fb_['name']}: {e}"
            props_ = None
        out.append({"bout": b, "a": pa_, "b": pb_, "pred": pred, "props": props_})
    return out


def _rec_txt(r: tuple) -> str:
    return f"{r[0]}-{r[1]}" + (f"-{r[2]}" if r[2] else "")


def _last(name: str) -> str:
    return name.split()[-1] if name.split() else name


def fighter_row(f: dict, prof: dict, p: float, is_pick: bool, won) -> str:
    flag = f'<img class="flag" src="{esc(prof["flag"])}" alt="">' if prof.get("flag") else ""
    ml = (f' · {f["ml"]:+.0f}' if f.get("ml") is not None else f' · fair {decide.american(p)} (model)')
    tag = ' <span class="pill ok">W</span>' if won else ""
    return (f'<div class="frow">{avatar_html(prof["headshot"], prof["name"], 46)}'
            f'<div class="tinfo"><div class="fname{" win" if won else ""}">{flag}{esc(prof["name"])}{tag}</div>'
            f'<div class="fsub">{esc(_rec_txt(prof["record"]))}{esc(ml)}'
            f'{" · " + esc(prof["stance"]) if prof.get("stance") else ""}</div></div>'
            f'<div class="fprob"><span class="lbl">Win</span><span class="num" '
            f'style="color:{"var(--green)" if is_pick else "var(--text)"}">{p * 100:.0f}%</span></div></div>')


def bout_card(x: dict) -> str:
    b, pa_, pb_, pred = x["bout"], x["a"], x["b"], x["pred"]
    fa_, fb_ = b["fighters"]
    p = pred["p_a"]
    st_pill = (f'<span class="pill live">● {esc(b["status"])}</span>' if b["state"] == "in" else
               f'<span class="pill final">{esc(b["result"] or b["status"])}</span>' if b["state"] == "post" else "")
    me = '<span class="pill b2b">Main Event</span>' if b.get("main_event") else ""
    head = (f'<div class="ghead"><div class="gtime">{esc(b["weight_class"] or "Bout")} · {b["rounds"]} rds</div>'
            f'<div class="tags">{me}{st_pill}</div></div>')
    pick = pa_ if pred["pick"] == 0 else pb_
    pick_p = p if pred["pick"] == 0 else 1 - p
    conf = "strong" if pick_p >= 0.68 else "lean"
    pick_html = (f'<div class="pickrow"><span class="badge {conf}">PICK · {esc(_last(pick["name"]))} by '
                 f'{esc(pred["pick_method"])}</span><span class="note" style="margin:0">'
                 f'{pick_p * pred["pick_method_p"] * 100:.0f}% exact outcome'
                 + (f' · mkt {pred["market"] * 100:.0f}%' if pred["market"] else "") + '</span></div>')
    bar = (f'<div class="winbar"><div class="r" style="width:{p * 100:.0f}%"></div>'
           f'<div class="b" style="width:{(1 - p) * 100:.0f}%"></div></div>')
    src = b.get("ml_src", "")
    src_html = (f'<div class="ml-src">Odds · {esc(src)}' + (" — no sportsbook line found, fair odds from the model"
                if src == "model fair" else "") + '</div>') if src and src != "ESPN" else ""
    if b["state"] == "pre":
        fp, f_ = (fa_, pa_) if pred["pick"] == 0 else (fb_, pb_)
        model_only = pred["p_model"] if pred["pick"] == 0 else 1 - pred["p_model"]
        v_html = verdict_html(decide.fight_decision(_last(f_["name"]), pick_p, fp.get("ml"), model_only))
    else:
        v_html = ""
    return (f'<div class="card">{head}' + fighter_row(fa_, pa_, p, pred["pick"] == 0, fa_.get("winner"))
            + bar + fighter_row(fb_, pb_, 1 - p, pred["pick"] == 1, fb_.get("winner")) + src_html + pick_html
            + v_html + props_drawer(x) + backing_drawer(x) + '</div>')


def props_table(x: dict) -> str:
    pr = x.get("props")
    if not pr:
        return ""
    la, lb = _last(x["a"]["name"]), _last(x["b"]["name"])

    def cell(p, fair, v):
        return (f'<span><b>{p * 100:.1f}%</b> <small>{esc(fair)}</small> '
                f'{(f'<span class="vpill {"bet" if v["verdict"] == "BET" else "pass"} sm">'
                  f'{"🟢 BET" if v["verdict"] == "BET" else "🟡 PASS"}</span>') if p >= 0.005 else ""}</span>')

    rows = []
    for r in pr["rows"]:
        if r["outcome"] == "Round 1":
            rows.append('<div class="ut sec"><span>Round moneylines (finish in round)</span></div>')
        rows.append(f'<div class="ut"><span class="sg">{esc(r["outcome"])}</span>'
                    f'{cell(r["a"], r["fair_a"], r["v_a"])}{cell(r["b"], r["fair_b"], r["v_b"])}</div>')
    dec = " · ".join(f'{k} {v * 100:.0f}% ({decide.american(v)})' for k, v in pr["dec_types"].items())
    best = pr.get("best")
    best_html = (f'<div class="verdict bet"><span class="vpill bet">🟢 BET</span><span class="vtxt"><b>'
                 f'{esc(best["pick"])}</b> · {esc(best["reason"])}</span></div>') if best else ""
    mkt = ("Market = vig-free moneyline × weight-class method/round rates + ~15% prop hold (ESPN posts no "
           "method or round prices)." if pr["market_q"] is not None else
           "No moneyline found, so no market anchor — fair odds shown, every outcome is PASS.")
    return (f'{best_html}<div class="ptab"><div class="ut head"><span>Outcome</span><span>{esc(la)} · fair</span>'
            f'<span>{esc(lb)} · fair</span></div>{"".join(rows)}</div>'
            f'<div class="pt-note"><b>Goes the distance</b> {pr["distance"] * 100:.0f}% '
            f'({decide.american(pr["distance"])}) · <b>Inside the distance</b> {pr["inside"] * 100:.0f}% '
            f'({decide.american(pr["inside"])})</div>'
            f'<div class="pt-note"><b>Decision type</b> · {esc(dec)}</div><div class="pt-note">{esc(mkt)}</div>')


def props_drawer(x: dict) -> str:
    t = props_table(x)
    return f'<details class="pdrawer"><summary>Round &amp; Method Props</summary>{t}</details>' if t else ""


def backing_drawer(x: dict) -> str:
    a, b = x["a"], x["b"]

    def f(v, unit="", d=0):
        return "—" if v is None else f"{v:.{d}f}{unit}"

    def row(lab, va, vb, better=None, unit="", d=0, ea=False, eb=False):
        adv_a = adv_b = False
        if better and va is not None and vb is not None and va != vb:
            adv_a = (va > vb) if better == "high" else (va < vb)
            adv_b = not adv_a
        ta = ("≈" if ea and va is not None else "") + f(va, unit, d)
        tb = ("≈" if eb and vb is not None else "") + f(vb, unit, d)
        return (f'<div class="l{" adv" if adv_a else ""}{" est" if ea else ""}">{esc(ta)}</div>'
                f'<div class="m">{esc(lab)}</div>'
                f'<div class="r{" adv" if adv_b else ""}{" est" if eb else ""}">{esc(tb)}</div>')

    sa, sb = a["stats"], b["stats"]
    xa, xb = set(a.get("est") or []), set(b.get("est") or [])

    def E(*keys):
        return {"ea": any(k in xa for k in keys), "eb": any(k in xb for k in keys)}

    def diff(s):
        return s["slpm"] - s["sapm"] if "slpm" in s and "sapm" in s else None

    def wins_split(prof):
        m = prof.get("methods")
        if not m or not sum(m):
            return None
        t = sum(m)
        return [x_ / t * 100 for x_ in m]

    wa, wb = wins_split(a), wins_split(b)
    me = {"ea": bool(a.get("methods_est")), "eb": bool(b.get("methods_est"))}
    sec = lambda t: f'<div class="sub-h">{esc(t)}</div>'  # noqa: E731
    strike = (row("Sig. strikes landed / min", sa.get("slpm"), sb.get("slpm"), "high", d=2, **E("slpm"))
              + row("Sig. strikes absorbed / min", sa.get("sapm"), sb.get("sapm"), "low", d=2, **E("sapm"))
              + row("Striking differential", diff(sa), diff(sb), "high", d=2, **E("slpm", "sapm"))
              + row("Striking accuracy", sa.get("str_acc"), sb.get("str_acc"), "high", "%", **E("str_acc"))
              + row("Striking defense", sa.get("str_def"), sb.get("str_def"), "high", "%", **E("str_def")))
    grap = (row("Takedowns / 15 min", sa.get("td_avg"), sb.get("td_avg"), "high", d=2, **E("td_avg"))
            + row("Takedown accuracy", sa.get("td_acc"), sb.get("td_acc"), "high", "%", **E("td_acc"))
            + row("Takedown defense", sa.get("td_def"), sb.get("td_def"), "high", "%", **E("td_def"))
            + row("Submissions / 15 min", sa.get("sub_avg"), sb.get("sub_avg"), "high", d=2, **E("sub_avg")))
    lb_a, lb_b = a.get("losses_by") or {}, b.get("losses_by") or {}
    fin = (row("Wins by KO/TKO", wa[0] if wa else None, wb[0] if wb else None, None, "%", **me)
           + row("Wins by submission", wa[1] if wa else None, wb[1] if wb else None, None, "%", **me)
           + row("Wins by decision", wa[2] if wa else None, wb[2] if wb else None, None, "%", **me)
           + row("Stoppage rate (wins)", (wa[0] + wa[1]) if wa else None, (wb[0] + wb[1]) if wb else None, "high", "%", **me)
           + row("Losses by KO/TKO", lb_a.get("KO"), lb_b.get("KO"), "low")
           + row("Losses by submission", lb_a.get("SUB"), lb_b.get("SUB"), "low")
           + row("UFC bouts (experience)", a.get("ufc_bouts"), b.get("ufc_bouts"), "high")
           + row("Pro bouts listed", a.get("bouts_listed") or None, b.get("bouts_listed") or None, "high"))

    def form(prof):
        rec = prof.get("recent") or []
        if not rec:
            return (f'<div class="note">{esc(prof["name"])}: no fight history on ESPN yet (likely a UFC debut) — '
                    f'method splits use {esc(prof.get("weight_class") or "division")} averages.</div>')
        wl = "".join(h["result"] for h in rec)
        items = "".join(
            f'<div class="form-row"><span class="res {"w" if h["result"] == "W" else "l" if h["result"] == "L" else ""}">'
            f'{esc(h["result"])}</span><span class="fo">vs {esc(h["opponent"] or "—")}</span>'
            f'<span class="fm">{esc(h["method"] or "—")}{" R" + str(h["round"]) if h["round"] else ""}'
            f'{" " + esc(h["time"]) if h.get("time") else ""}</span><span class="fd">{esc(h["date"])}</span></div>'
            for h in rec)
        return (f'<div class="form-h">{esc(prof["name"])} · last {len(rec)}: <b>{esc(wl)}</b> · '
                f'{sum(h["result"] == "W" for h in rec)}-{sum(h["result"] == "L" for h in rec)}</div>{items}')

    est_note = ('<div class="pt-note">≈ = ESPN has no figure for this fighter (debut or null stat); a '
                'weight-class divisional average is shown and used instead.</div>') if (xa or xb or me["ea"] or me["eb"]) else ""
    head = (f'<div class="tot bk"><div class="l"><b>{esc(_last(a["name"]))}</b></div><div class="m"></div>'
            f'<div class="r"><b>{esc(_last(b["name"]))}</b></div></div>')
    body = (head + sec("Striking efficiency") + f'<div class="tot bk">{strike}</div>'
            + sec("Grappling & control") + f'<div class="tot bk">{grap}</div>'
            + sec("Durability & finish rate") + f'<div class="tot bk">{fin}</div>'
            + est_note + sec("Recent form (last 5 bouts, newest first)") + form(a) + form(b)
            + '<div class="pt-note">Strength of schedule: opponents are listed above; full opponent-quality '
              'ratings would need every opponent\'s page, so experience (UFC bouts) stands in.</div>')
    return f'<details class="pdrawer"><summary>Data Backing &amp; Analytics</summary>{body}</details>'


def render_ufc() -> None:
    try:
        with st.spinner("Loading UFC fight cards…"):
            cards = ufc.fight_cards(today_et)
    except Exception as e:
        hero("UFC")
        empty_card("Couldn't load the UFC schedule from ESPN.", str(e))
        return
    if not cards:
        hero("No events")
        empty_card("No UFC events found in the next 60 days.")
        return
    if len(cards) > 1:
        names = {c["id"]: c["name"] + (f' · {c["date"]:%b %d}' if c["date"] else "") for c in cards}
        cid = st.selectbox("Event", list(names), format_func=names.get, label_visibility="collapsed", key="ufc_event")
        card = next(c for c in cards if c["id"] == cid)
    else:
        card = cards[0]
    hero((f'{card["date"]:%a, %b %d}<br>' if card["date"] else "") + f'{len(card["bouts"])} bouts')

    bouts = load_card(card)
    render_card_tabs(card, bouts)


def load_card(card: dict) -> list[dict]:
    ids = [f["id"] for b in card["bouts"] for f in b["fighters"] if f["id"]]
    fails = prefetch([(f"fighter|{fid}", ufc.TTL_FIGHTER, lambda fid=fid: ufc.prefetch_fighter(fid)) for fid in ids],
                     "Loading fighter profiles")
    if fails:
        NOTES["ufc fighters"] = f"{fails} fighter profile(s) unavailable; those bouts lean on records."
    try:
        ufc.ensure_moneylines(card)
    except Exception as e:
        NOTES["ufc odds"] = f"Moneyline fallback failed: {e}"
    with st.spinner("Projecting bouts…"):
        return compute_card(card["id"], bucket(), card)


def render_card_tabs(card: dict, bouts: list[dict]) -> None:
    status = {"in": '<span class="pill live">● LIVE</span>', "post": '<span class="pill final">Final</span>'}.get(
        card["state"], '<span class="pill info">Upcoming</span>')
    start = card["start"].strftime("%a %b %-d · %-I:%M %p ET") if card["start"] is not None else ""
    H(f'<div class="card"><div class="ev-head"><div><div class="ev-name">{esc(card["name"])}</div>'
      f'<div class="psub">{esc(start)}{" · " + esc(card["venue"]) if card["venue"] else ""}</div></div>'
      f'{status}</div></div>')

    tab_card, tab_bd, tab_st, tab_pl = st.tabs(["🥊 Fight Card", "🔍 Fight Breakdowns", "🔥 Streaks", "🎟 Parlays"])
    with tab_st:
        note("Current runs from each fighter's recent bouts on ESPN: win streaks, finishes, KOs, submissions, and "
             "fights that went (or didn't go) the distance.")
        render_streaks(streakmod.fighter_streaks(bouts), {}, {}, "UFC")
    with tab_card:
        note("Win % blends the model (record, experience, reach, age, striking and grappling rates) with the "
             "betting market when ESPN lists odds. Method = most likely way the pick wins.")
        html, seg = [], None
        for x in bouts:
            s = "Main Event" if x["bout"].get("main_event") else x["bout"]["segment"]
            if s != seg:
                html.append(f'<div class="seg-head">{esc(s)}</div>')
                seg = s
            html.append(bout_card(x))
        H("".join(html) if html else '<div class="card"><div class="empty">No bouts listed yet.</div></div>')
    with tab_bd:
        breakdown_fragment(bouts)
    with tab_pl:
        legs = parlays.ufc_legs(card, bouts)
        note("Fight legs need a moneyline from ESPN; bouts without odds are left out.")
        render_parlays(legs)


@st.fragment
def breakdown_fragment(bouts: list[dict]) -> None:
    if not bouts:
        empty_card("No bouts to break down yet.")
        return
    labels = {x["bout"]["id"]: f'{x["a"]["name"]} vs {x["b"]["name"]}' for x in bouts}
    bid = st.selectbox("Bout", list(labels), format_func=labels.get, label_visibility="collapsed", key="ufc_bout")
    x = next(y for y in bouts if y["bout"]["id"] == bid)
    a, b, pred = x["a"], x["b"], x["pred"]

    def fmt(v, unit="", d=0):
        return "—" if v is None else f"{v:.{d}f}{unit}"

    def ht(v):
        return "—" if not v else f"{int(v // 12)}' {int(round(v % 12))}\""

    rows = [
        ("Record", _rec_txt(a["record"]), _rec_txt(b["record"]), None),
        ("Age", fmt(a["age"]), fmt(b["age"]), (a["age"], b["age"], "low")),
        ("Height", ht(a["height"]), ht(b["height"]), (a["height"], b["height"], "high")),
        ("Reach", fmt(a["reach"], '"'), fmt(b["reach"], '"'), (a["reach"], b["reach"], "high")),
        ("Stance", a["stance"] or "—", b["stance"] or "—", None),
    ]
    for k, lab, d, better in (("slpm", "Sig. strikes landed / min", 2, "high"), ("str_acc", "Striking accuracy", 0, "high"),
                              ("sapm", "Sig. strikes absorbed / min", 2, "low"), ("str_def", "Striking defense", 0, "high"),
                              ("td_avg", "Takedowns / 15 min", 2, "high"), ("td_acc", "Takedown accuracy", 0, "high"),
                              ("td_def", "Takedown defense", 0, "high"), ("sub_avg", "Submissions / 15 min", 2, "high")):
        va, vb = a["stats"].get(k), b["stats"].get(k)
        if va is None and vb is None:
            continue
        unit = "%" if ("acc" in k or "def" in k) else ""
        ea, eb = k in (a.get("est") or []), k in (b.get("est") or [])
        rows.append((lab, ("≈" if ea else "") + fmt(va, unit, d), ("≈" if eb else "") + fmt(vb, unit, d),
                     (va, vb, better)))

    def cell(txt, mine, other, better, left):
        adv = (mine is not None and other is not None and mine != other
               and ((mine > other) if better == "high" else (mine < other)))
        return f'<div class="{"l" if left else "r"}{" adv" if adv else ""}">{esc(txt)}</div>'

    tot = "".join(
        (cell(la, cmp[0], cmp[1], cmp[2], True) if cmp else f'<div class="l">{esc(la)}</div>')
        + f'<div class="m">{esc(lab)}</div>'
        + (cell(lb, cmp[1], cmp[0], cmp[2], False) if cmp else f'<div class="r">{esc(lb)}</div>')
        for lab, la, lb, cmp in rows)
    p = pred["p_a"]
    head = (f'<div class="ev-head"><div style="display:flex;gap:10px;align-items:center">'
            f'{avatar_html(a["headshot"], a["name"], 48)}<div><div class="fname">{esc(a["name"])}</div>'
            f'<div class="fsub" style="color:#ef4444">Red · {p * 100:.0f}%</div></div></div>'
            f'<div style="display:flex;gap:10px;align-items:center;text-align:right"><div>'
            f'<div class="fname">{esc(b["name"])}</div><div class="fsub" style="color:#3b82f6">Blue · '
            f'{(1 - p) * 100:.0f}%</div></div>{avatar_html(b["headshot"], b["name"], 48)}</div></div>')
    bar = (f'<div class="winbar"><div class="r" style="width:{p * 100:.0f}%"></div>'
           f'<div class="b" style="width:{(1 - p) * 100:.0f}%"></div></div>')

    def mbars(name, pw, ms, color):
        return (f'<div class="sub-h">{esc(name)} wins by…</div>' + "".join(
            f'<div class="mbar"><span class="lab">{lab}</span><div class="track"><div class="fill {color}" '
            f'style="width:{pw * mp * 100:.0f}%"></div></div><span class="pct">{pw * mp * 100:.0f}%</span></div>'
            for lab, mp in zip(("KO/TKO", "Submission", "Decision"), ms)))

    def who(v):
        return esc(_last(a["name"])) if v > 0.005 else esc(_last(b["name"])) if v < -0.005 else ""

    factors = "".join(
        f'<div class="factor"><span>{esc(n)}</span><span class="v {"pos" if v > 0.005 else "neg" if v < -0.005 else ""}">'
        f'{"+" if v > 0 else ""}{v:.2f} {who(v)}</span></div>' for n, v in pred["factors"])
    mkt = (f'<div class="factor"><span>Betting market (vig-free)</span><span class="v">{esc(_last(a["name"]))} '
           f'{pred["market"] * 100:.0f}%</span></div>' if pred["market"] else "")
    arts = cached_news("UFC", (), bucket(5))
    fnews = "".join(news_html(newsmod.for_player(arts, f["id"], f["name"], 10))
                    for f in x["bout"]["fighters"])
    src = [f'{esc(f["name"])}: {esc(", ".join(pf.get("src") or []) or "card record only")}'
           + (" · ≈ division averages for missing stats" if pf.get("est") else "")
           for f, pf in zip(x["bout"]["fighters"], (a, b))]
    src_note = f'<div class="note" style="margin-top:8px">Stats source · {" · ".join(src)}</div>'
    detail_note = "" if (a["detail_ok"] and b["detail_ok"]) else (
        '<div class="note" style="margin-top:8px">Some fighter details are missing from ESPN; '
        'the projection leans on records for those fighters.</div>')
    H(f'<div class="card">{head}{bar}<div class="sub-h">Tale of the tape</div><div class="tot">{tot}</div></div>'
      f'<div class="card"><div class="sub-h" style="margin-top:0">Method of victory</div>'
      f'{mbars(a["name"], p, pred["methods_a"], "r")}{mbars(b["name"], 1 - p, pred["methods_b"], "b")}</div>'
      + (f'<div class="card"><div class="sub-h" style="margin-top:0">Round &amp; method props</div>'
         f'{props_table(x)}</div>' if x.get("props") else "")
      + f'<div class="card"><div class="sub-h" style="margin-top:0">Model factors (log-odds, + favors '
      f'{esc(_last(a["name"]))})</div>{factors}{mkt}<div class="factor"><span>Model only</span>'
      f'<span class="v">{esc(_last(a["name"]))} {pred["p_model"] * 100:.0f}%</span></div>{detail_note}{src_note}</div>'
      + (f'<div class="card"><div class="sub-h" style="margin-top:0">Latest news</div>{fnews}</div>' if fnews else ""))


# --------------------------------------------------------------------------
# Parlays
# --------------------------------------------------------------------------


def leg_visual(l: parlays.Leg) -> str:
    """Team logo (both teams for totals, round photo for UFC) with a small league badge in the corner."""
    def img(url, cls=""):
        return f'<div class="im {cls}" style="background-image:url(\'{esc(url)}\')"></div>' if url else ""

    inner = (img(l.logo, "a") + img(l.logo2, "b")) if l.logo2 else img(l.logo)
    cls = "tm two" if l.logo2 else ("tm photo" if l.photo else "tm")
    return (f'<div class="lg-vis"><div class="{cls}"><span>{esc(l.fallback[:7])}</span>{inner}</div>'
            f'<div class="lgb">{league_logo_html(l.sport, 20)}</div></div>')


def leg_html(l: parlays.Leg, show_edge: bool = False) -> str:
    note_ = f'<div class="lg-note">{esc(l.note)}</div>' if l.note else ""
    edge = (f'<span class="lg-edge">+{l.edge * 100:.1f}% edge at {parlays.dec_to_american(l.dec)}</span>'
            if show_edge else "")
    when = f'<div class="lg-when">🕒 {esc(l.when)} · {esc(l.sport)}</div>' if l.when else ""
    return (f'<div class="leg">{leg_visual(l)}'
            f'<div class="lg-main"><div class="lg-pick">{esc(l.pick)}</div>'
            f'<div class="lg-sub"><span class="mk">{esc(l.market)}</span>{esc(l.matchup)}</div>'
            f'{when}{note_}{edge}</div>'
            f'<div class="lg-p"><span class="lbl">Hit</span><span class="num">{l.prob * 100:.0f}%</span></div></div>')


def parlay_card(pl: dict) -> str:
    ev = pl["ev"] * 100
    tag_cls = {"SAFE": "strong", "VALUE": "lean", "LONGSHOT": "muted", "MIXED": "strong",
               "PROPS": "strong"}.get(pl["tag"], "lean")
    sports = sorted({l.sport for l in pl["legs"]})
    return (f'<div class="card"><div class="ghead"><div class="pl-title">{esc(pl["title"])}'
            f'<div class="psub">{len(pl["legs"])} legs · {esc(" + ".join(sports))}</div></div>'
            f'<span class="badge {tag_cls}">{esc(pl["tag"])}</span></div>'
            + "".join(leg_html(l) for l in pl["legs"])
            + '<div class="pgrid" style="margin:10px 0 0">'
            f'<div class="pbox hl"><span class="lbl">Hit chance</span><div class="num">{pl["prob"] * 100:.0f}%</div></div>'
            f'<div class="pbox"><span class="lbl">Est. odds</span><div class="num">{parlays.dec_to_american(pl["dec"])}</div></div>'
            f'<div class="pbox"><span class="lbl">Fair odds</span><div class="num">{parlays.dec_to_american(1 / pl["prob"])}</div></div>'
            f'<div class="pbox"><span class="lbl">EV</span><div class="num" style="color:'
            f'{"var(--green)" if ev > 0 else "var(--red)"}">{ev:+.0f}%</div></div></div></div>')


def render_parlays(legs: list[parlays.Leg], mixed: bool = False) -> None:
    pool = parlays.value_legs(legs)
    if not pool:
        empty_card("No value legs right now.",
                   f"Checked {len(legs)} priced legs; none clear {parlays.MIN_PROB:.0%} hit chance with a "
                   f"{parlays.MIN_EDGE:.0%}+ edge over the price. Check back when more lines are posted.")
        return
    built = parlays.build(legs, mixed=mixed)
    if not built:
        empty_card("Not enough independent value legs for a parlay yet.",
                   "Parlays use at most one leg per game" + (" and need legs from 2+ sports." if mixed else "."))
    H("".join(parlay_card(p) for p in built))
    H('<div class="seg-head">Top value legs</div>'
      + '<div class="card">' + "".join(leg_html(l, show_edge=True) for l in pool[:10]) + "</div>")


def gather_prop_legs(sk: str, games: list[dict], slate: dict, inj: list[dict], load_all: bool,
                     auto_games: int = 4) -> list:
    """Prop legs from the soonest `auto_games` upcoming games (player logs loaded behind a progress bar),
    every game already loaded, or every game when load_all is on."""
    lg_ = LEAGUES[sk]
    try:
        book = cached_lines(sk, bucket(5))
    except Exception:
        book = {}
    labels = dict(propmod.MARKETS[lg_.kind])
    legs = []
    upcoming = sorted((g for g in games if g["id"] in slate["projections"] and g["state"] == "pre"),
                      key=lambda g: (g["tip"] is None, g["tip"] or 0))
    for i, g in enumerate(upcoming):
        if load_all or i < auto_games:
            prefetch_players(lg_, g)
        elif not all(cache_fresh(f"roster|{sk}|{g[s]['id']}", espn.TTL_ROSTER) for s in ("away", "home")):
            continue  # props for this game haven't been loaded yet
        try:
            df = compute_props(sk, g["id"], float(q_sit), bucket(), g, slate["projections"][g["id"]],
                               slate["profiles"], slate["base"], slate["b2b"], inj)
            book_g = {**book, **cached_espn_props(sk, g["id"], bucket(5))}
            legs += parlays.prop_legs(sk, lg_.kind, g, df, book_g, norm_name, labels)
        except Exception as e:
            NOTES[f"prop legs {sk}"] = f"{sk} prop legs skipped for a game ({type(e).__name__})."
    return legs


@st.fragment
def not_started(g: dict) -> bool:
    """Pregame and tip-off still in the future (ESPN can lag a few minutes flipping the state)."""
    if g["state"] != "pre":
        return False
    return g["tip"] is None or g["tip"].to_pydatetime() > datetime.now(ET)


def upcoming_games(sk: str, days: int = 3, cap: int = 16) -> list[dict]:
    """Games that haven't started in the next `days` days (today included), soonest first."""
    lg_ = LEAGUES[sk]
    games = espn.week_games(lg_, today_et, days, today_et)
    if sk == "NCAAB":  # D1 has hundreds of games: keep parlays to ranked matchups
        games = [g for g in games if g["home"]["rank"] or g["away"]["rank"]]
    games = [g for g in games if not_started(g)]
    games.sort(key=lambda g: (g["tip"] is None, g["tip"] or 0))
    return games[:cap]


GRADE_CAP = 6
PARLAY_WINDOWS = ["Not started (this slate)", "Next 3 days"]


@st.fragment
def parlay_fragment(games: list[dict], slate: dict, inj: list[dict]) -> None:
    pending = [g for g in games if not_started(g)]
    default = PARLAY_WINDOWS[0] if pending else PARLAY_WINDOWS[1]
    win = st.segmented_control("Parlay window", PARLAY_WINDOWS, default=default, key=f"plwin_{sport}",
                               label_visibility="collapsed", width="stretch") or default
    if win == PARLAY_WINDOWS[1]:
        with st.spinner("Loading upcoming games…"):
            gs = upcoming_games(sport)
        if not pending and gs:
            note("Every game on this slate has started — building parlays from upcoming games instead.")
        if gs:
            sl, inj2 = load_team_data(sport, gs)
        else:
            sl, inj2 = {"projections": {}, "profiles": {}, "base": {}, "b2b": set()}, []
    else:
        gs, sl, inj2 = pending, slate, inj
    if not gs:
        empty_card("No games left to start in this window.",
                   "Switch to Next 3 days to build parlays from upcoming games.")
        return
    first, last = min((g["tip"] for g in gs if g["tip"] is not None), default=None), \
        max((g["tip"] for g in gs if g["tip"] is not None), default=None)
    span = (f"{first:%a %-I:%M %p} – {last:%a %-I:%M %p} ET" if first is not None and last is not None else "")
    H(f'<div class="note"><b>{len(gs)}</b> game{"s" if len(gs) != 1 else ""} not started yet'
      + (f" · {esc(span)}" if span else "") + "</div>")
    load_all = st.toggle("Scan player props in every game (default: the next 4 games; slower the first time)",
                         key=f"pl_props_{sport}")
    legs = parlays.game_legs(sport, lg.kind, gs, sl["projections"])
    legs += gather_prop_legs(sport, gs, sl, inj2, load_all)
    note("Only games that haven't started. Each leg's hit chance blends the model (60%) with the market (40%). "
         "Prices: ESPN moneylines, -110 spreads/totals, real prop prices when posted (else -115). Props without a "
         "posted line use the implied line. One leg per game; real parlay odds may differ. Not betting advice.")
    render_parlays(legs)


def render_mixed() -> None:
    hero(f"{today_et:%a, %b %d}<br>All sports")
    mix_win = st.segmented_control("Parlay window", ["Today", "Next 3 days"], default="Today", key="mix_window",
                                   label_visibility="collapsed", width="stretch") or "Today"
    load_all = st.toggle("Scan player props in every game (default: next 2 games per sport; slower first time)",
                         key="mix_props")
    legs, coverage = all_sports_board(mix_win, load_all)
    board_card(coverage)
    note("Mixed parlays combine the best value game lines and player props across sports (one leg per game, "
         "legs from 2+ sports; props use the posted PrizePicks / Underdog line or the player's last-10 average). "
         "Hit chances blend each sport's model (60%) with the market (40%); prices are ESPN moneylines, -110 spreads/totals and -115 "
         "pick'em props. Not betting advice.")
    render_parlays(legs, mixed=True)


def board_card(coverage: list[str]) -> None:
    H('<div class="card"><div class="sub-h" style="margin-top:0">Today\'s board</div>'
      + "".join(f'<div class="factor" style="align-items:center"><span style="display:flex;align-items:center;'
                f'gap:10px">{league_logo_html(c.split(":")[0], 28)}<b>{esc(c.split(":")[0])}</b></span>'
                f'<span class="v">{esc(c.split(":", 1)[1].strip())}</span></div>' for c in coverage) + "</div>")


def all_sports_board(mix_win: str, load_all: bool) -> tuple[list, list[str]]:
    """Every not-started game line, top player prop and UFC fight leg across sports (shared by Mixed Parlays
    and the Bankroll Challenge)."""
    legs, coverage = [], []
    for sk in ("NBA", "NCAAB", "NFL", "NHL"):
        try:
            with st.spinner(f"Loading {sk}…"):
                if mix_win == "Next 3 days":
                    games = upcoming_games(sk, 3, PAGE_SIZE)
                else:
                    games = fetch_games(sk, today_et)
                    if sk == "NFL":
                        games = [g for g in games if g["date"] and 0 <= (g["date"] - today_et).days <= 3]
                    elif sk == "NCAAB":
                        games = [g for g in games if g["home"]["rank"] or g["away"]["rank"]]
                    games = [g for g in games if not_started(g)][:PAGE_SIZE]
            if not games:
                coverage.append(f"{sk}: no upcoming games")
                continue
            slate, inj = load_team_data(sk, games)
            sl = parlays.game_legs(sk, LEAGUES[sk].kind, games, slate["projections"])
            sl += gather_prop_legs(sk, games, slate, inj, load_all, auto_games=2)
            legs += sl
            coverage.append(f"{sk}: {len(games)} game{'s' if len(games) != 1 else ''} · {len(sl)} legs")
        except Exception as e:
            coverage.append(f"{sk}: unavailable ({type(e).__name__})")
    try:
        soc_games, soc_legs = 0, 0
        for k in espn.SOCCER_DEFAULT:
            try:
                gs = ([g for g in espn.week_games(LEAGUES[k], today_et, 3, today_et) if not_started(g)]
                  if mix_win == "Next 3 days" else
                  [g for g in fetch_games(k, today_et) if not_started(g)])[:PAGE_SIZE]
            except Exception:
                continue
            if not gs:
                continue
            sl, _ = load_team_data(k, gs)
            lk = parlays.game_legs(k, "soccer", gs, sl["projections"])
            legs += lk
            soc_games, soc_legs = soc_games + len(gs), soc_legs + len(lk)
        coverage.append(f"Soccer: {soc_games} match{'es' if soc_games != 1 else ''} · {soc_legs} legs"
                        if soc_games else ("Soccer: no upcoming matches in the top leagues" if mix_win != "Today"
                                           else "Soccer: no matches left today in the top leagues"))
    except Exception as e:
        coverage.append(f"Soccer: unavailable ({type(e).__name__})")
    try:
        cards = ufc.fight_cards(today_et)
        card = next((c for c in cards if c["state"] != "post" and c["date"]
                     and (c["date"] - today_et).days <= 7), None)
        if card:
            ul = parlays.ufc_legs(card, load_card(card))
            legs += ul
            coverage.append(f"UFC: {card['name']} · {len(ul)} legs")
        else:
            coverage.append("UFC: no card this week")
    except Exception as e:
        coverage.append(f"UFC: unavailable ({type(e).__name__})")
    return legs, coverage


def _money(x: float) -> str:
    return f"${x:,.2f}" if x < 1000 else f"${x:,.0f}"


def render_challenge() -> None:
    hero(f"{today_et:%a, %b %d}<br>Bankroll challenge")
    note("Roll a small bankroll into a target, one bet a day — like the \"$50 to $10k\" challenges. Every bet "
         "rides the bankroll, so one loss ends it: the app shows the real odds of finishing at every step.")
    if "ch_name_next" in st.session_state:  # a restart plan was just created: open it
        st.session_state["ch_name"] = st.session_state.pop("ch_name_next")
    name = st.text_input("Challenge name", key="ch_name", placeholder="Name your challenge (e.g. xmas-25k)",
                         label_visibility="collapsed").strip()
    ch = chmod.load(name) if name else None

    if ch is None:
        H('<div class="sub-h">Set up a challenge</div>')
        c1, c2 = st.columns(2)
        start = c1.number_input("Starting bankroll ($)", 1.0, 1e6, 50.0, 5.0, key="ch_start")
        target = c2.number_input("Target ($)", 2.0, 1e7, 10000.0, 500.0, key="ch_target")
        c3, c4 = st.columns(2)
        days = c3.number_input("Bets / days", 1, 100, 32, 1, key="ch_days")
        roll = c4.slider("Rollover (% of bankroll per bet)", 25, 100, 100, 5, key="ch_roll") / 100
        challenge_math(start, target, int(days), roll)
        up = st.file_uploader("Restore a saved challenge (.json backup)", type=["json"], key="ch_up")
        if up is not None:
            try:
                data = json.loads(up.getvalue())
                chmod.save(data)
                st.session_state["ch_name"] = data["name"]
                st.rerun()
            except Exception as e:
                st.error(f"That file isn't a challenge backup ({e}).")
        if st.button("Start challenge", type="primary", disabled=not name or target <= start, key="ch_create",
                     width="stretch"):
            chmod.save(chmod.new(name, start, target, int(days), roll))
            st.rerun()
        if not name:
            st.caption("Give the challenge a name first — it's how you reopen it later.")
        return

    bank, stat = chmod.bankroll(ch), chmod.status(ch)
    progress_card(ch, bank, stat)
    pend = chmod.pending(ch)
    if stat == "busted":
        restart_card(ch)
    elif stat == "complete":
        H('<div class="card"><div class="verdict bet"><span class="vpill bet">🏆 DONE</span><span class="vtxt">'
          f'<b>Target hit: {_money(bank)}</b> in {sum(b["result"] == "won" for b in ch["bets"])} wins.</span></div></div>')
    elif pend:
        H(f'<div class="sub-h">Bet {pend["n"]} — waiting on the result</div>' + bet_card(pend))
        with st.expander(f"✏️ Change stake (now {_money(pend['stake'])} of {_money(bank)})",
                         expanded=pend["stake"] < bank - 0.005):
            new_stake = st.number_input("Stake ($)", 0.01, float(max(bank, 0.01)), float(pend["stake"]), 0.25,
                                        key=f"ch_stake_{pend['n']}", format="%.2f")
            s1, s2 = st.columns(2)
            if s1.button(f"💰 Full bankroll ({_money(bank)})", key="ch_full", width="stretch"):
                chmod.set_stake(ch, bank); chmod.save(ch); st.rerun()
            if s2.button("Save stake", key="ch_stake_save", width="stretch",
                         disabled=abs(new_stake - pend["stake"]) < 0.005):
                chmod.set_stake(ch, new_stake); chmod.save(ch); st.rerun()
        c1, c2, c3 = st.columns(3)
        if c1.button("✅ Won", key="ch_won", width="stretch"):
            chmod.settle(ch, "won"); chmod.save(ch); st.rerun()
        if c2.button("❌ Lost", key="ch_lost", width="stretch"):
            chmod.settle(ch, "lost"); chmod.save(ch); st.rerun()
        if c3.button("↩ Push", key="ch_push", width="stretch"):
            chmod.settle(ch, "push"); chmod.save(ch); st.rerun()
    else:
        todays_bet(ch, bank)
    history_card(ch)
    with st.expander("⚙️ Challenge settings"):
        r_new = st.slider("Rollover (% of bankroll per bet)", 25, 100, int(round(ch["rollover"] * 100)), 5,
                          key="ch_roll_edit") / 100
        t_new = st.number_input("Target ($)", float(ch["start"]) + 1, 1e7, float(ch["target"]), 500.0, key="ch_t_edit")
        d_new = st.number_input("Bets / days", 1, 100, int(ch["days"]), 1, key="ch_d_edit")
        if st.button("Save settings", key="ch_set_save", width="stretch"):
            ch["rollover"], ch["target"], ch["days"] = r_new, float(t_new), int(d_new)
            chmod.save(ch)
            st.rerun()
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Backup (.json)", json.dumps(ch, indent=1), file_name=f"{chmod.slug(ch['name'])}.json",
                       mime="application/json", key="ch_dl", width="stretch")
    if c2.button("🗑 Delete challenge", key="ch_del", width="stretch"):
        st.session_state["ch_del_confirm"] = True
    if st.session_state.get("ch_del_confirm"):
        st.warning(f"Delete \"{ch['name']}\" and its history?")
        d1, d2 = st.columns(2)
        if d1.button("Yes, delete", key="ch_del_yes", width="stretch"):
            chmod.path(ch["name"]).unlink(missing_ok=True)
            st.session_state["ch_del_confirm"] = False
            st.rerun()
        if d2.button("Keep it", key="ch_del_no", width="stretch"):
            st.session_state["ch_del_confirm"] = False
            st.rerun()
    note("Challenges are saved on the app server under their name. Streamlit Cloud can wipe that storage when the "
         "app restarts or redeploys — tap Backup after each bet and restore it from the setup screen if needed. "
         "Gamble responsibly: only stake what you can afford to lose (1-800-GAMBLER).")


def restart_card(ch: dict) -> None:
    lost = next((b for b in reversed(ch["bets"]) if b["result"] == "lost"), {})
    H(f'<div class="card"><div class="sub-h" style="margin-top:0">Challenge over — what now?</div>'
      f'<div class="pt-note" style="font-size:.78rem;color:var(--text)">{esc(chmod.loss_note(lost))}</div></div>'
      '<div class="sub-h">Easier ways to get the win</div>')
    for i, pl in enumerate(chmod.restart_plans(ch)):
        finish = (f' · finish chance ≈ {_pct_small(pl["finish"])} at fair odds' if pl["finish"] is not None
                  else " · survives losses (each one halves the stake base)")
        H(f'<div class="card"><div class="ghead"><div class="pl-title">{esc(pl["title"])}'
          f'<div class="psub">{_money(pl["start"])} → {_money(pl["target"])} · {pl["days"]} bets · '
          f'{pl["rollover"] * 100:.0f}% rollover</div></div>'
          f'<span class="badge strong">{chmod.american(pl["need"])} / bet</span></div>'
          f'<div class="pt-note" style="color:var(--text)">{esc(pl["why"])}</div>'
          f'<div class="pt-note">Each bet needs to hit about {pl["p_bet"] * 100:.0f}% of the time'
          f'{finish}.</div></div>')
        if st.button(f"Start: {pl['title'].lower()}", key=f"ch_restart_{i}", width="stretch"):
            name = chmod.unique_name(ch["name"])
            chmod.save(chmod.new(name, pl["start"], pl["target"], pl["days"], pl["rollover"]))
            st.session_state["ch_name_next"] = name
            st.rerun()
    note("Over many tries, the sportsbook's margin works against every plan — smaller targets and cushions "
         "just make each run last longer and finish more often. Only play with money you're fine losing.")


def challenge_math(start: float, target: float, days: int, roll: float) -> None:
    need = chmod.per_bet_multiplier(start, target, days, roll)
    fair_p = 1 / need
    allwin = fair_p ** days
    edge_p = min(0.97, fair_p * 1.08) ** days  # every bet beating its price by 8%
    vig_p = (fair_p * 0.955) ** days            # ~4.5% hold per bet (-110 both ways)
    rows = chmod.ladder(start, target, need, roll, max_rows=days)
    lad = "".join(f'<div class="ch-l"><span>#{r["n"]}</span><span>{_money(r["bank"])}</span>'
                  f'<span>{_money(r["stake"])}</span><span class="g">{_money(r["after"])}</span></div>'
                  for r in rows[:6])
    more = f'<div class="pt-note">… {len(rows) - 6} more bets to {_money(target)}</div>' if len(rows) > 6 else ""
    H(f'<div class="card"><div class="sub-h" style="margin-top:0">The math</div>'
      f'<div class="ch-stats"><div><span class="num">{target / start:,.0f}×</span><span class="lbl">growth needed</span></div>'
      f'<div><span class="num">{chmod.american(need)}</span><span class="lbl">odds every bet</span></div>'
      f'<div><span class="num">{fair_p * 100:.0f}%</span><span class="lbl">hit rate those odds imply</span></div></div>'
      f'<div class="verdict pass" style="margin-top:10px"><span class="vpill pass">⚠ ODDS</span><span class="vtxt">'
      f'Chance to win all {days}: <b>{_pct_small(vig_p)}</b> with a normal sportsbook margin · '
      f'<b>{_pct_small(allwin)}</b> at fair odds (always ≈ start ÷ target) · <b>{_pct_small(edge_p)}</b> if every '
      f'pick beats its price by 8%.</span></div>'
      f'<div class="ptab" style="margin-top:10px"><div class="ch-l head"><span>Bet</span><span>Bankroll</span>'
      f'<span>Stake</span><span>If it wins</span></div>{lad}</div>{more}'
      f'<div class="pt-note">Fewer, bigger-priced bets or more, safer ones — the finish chance barely moves, '
      f'because the sportsbook margin compounds on every bet.</div></div>')


def _pct_small(p: float) -> str:
    if p >= 0.01:
        return f"{p * 100:.1f}%"
    return f"{p * 100:.2f}% (1 in {1 / p:,.0f})" if p > 0 else "0%"


def progress_card(ch: dict, bank: float, stat: str) -> None:
    start, target = ch["start"], ch["target"]
    frac = 0.0 if bank <= start else min(1.0, math.log(bank / start) / math.log(target / start))
    wins = sum(b["result"] == "won" for b in ch["bets"])
    left = chmod.bets_left(ch)
    need = chmod.need_dec_now(ch)
    probs = [b["prob"] for b in ch["bets"] if b["result"] == "won" and b.get("prob")]
    H(f'<div class="card"><div class="ev-head"><div><div class="ev-name">{esc(ch["name"])}</div>'
      f'<div class="psub">{_money(start)} → {_money(target)} · started {esc(ch["created"])}</div></div>'
      f'<span class="pill {"ok" if stat == "complete" else "out" if stat == "busted" else "info"}">{stat}</span></div>'
      f'<div class="ch-stats"><div><span class="num">{_money(bank)}</span><span class="lbl">bankroll</span></div>'
      f'<div><span class="num">{wins}</span><span class="lbl">wins in a row</span></div>'
      f'<div><span class="num">{left if stat == "active" else "—"}</span><span class="lbl">bets left</span></div></div>'
      f'<div class="ch-bar"><i style="width:{frac * 100:.0f}%"></i></div>'
      f'<div class="pt-note">{frac * 100:.0f}% of the way (log scale)'
      + (f' · next bet needs {chmod.american(need)} or longer to stay on pace' if stat == "active" else "")
      + (f' · the wins so far had a {_pct_small(chmod.survival(probs))} chance together' if probs else "")
      + '</div></div>')


def bet_card(b: dict) -> str:
    legs = "".join(f'<div class="ch-leg">• {esc(x)}</div>' for x in b.get("legs") or [])
    hp = f' · {b["prob"] * 100:.0f}% to hit' if b.get("prob") else ""
    return (f'<div class="card"><div class="ghead"><div class="pl-title">{esc(b["desc"])}'
            f'<div class="psub">{chmod.american(b["dec"])}{hp} · {esc(b["date"])}</div></div>'
            f'<span class="badge {"strong" if b["result"] == "won" else "muted" if b["result"] != "lost" else "lean"}">'
            f'{esc(b["result"].upper())}</span></div>{legs}'
            f'<div class="factor"><span>Stake</span><span class="v">{_money(b["stake"])}</span></div>'
            f'<div class="factor"><span>Pays (if it wins)</span><span class="v">{_money(b["stake"] * b["dec"])}</span></div></div>')


def todays_bet(ch: dict, bank: float) -> None:
    need = chmod.need_dec_now(ch)
    full = st.toggle(f"Stake the full bankroll ({_money(bank)})", value=ch["rollover"] >= 0.999,
                     key=f"ch_fullnext_{int(round(ch['rollover'] * 100))}")
    stake = bank if full else st.number_input(
        "Stake for this bet ($)", 0.01, float(max(bank, 0.01)), float(round(bank * ch["rollover"], 2)), 0.25,
        key="ch_next_stake", format="%.2f")
    H(f'<div class="sub-h">Today\'s challenge bet · {_money(stake)} stake · needs {chmod.american(need)} or longer</div>')
    if not st.toggle("Scan today's board for the challenge bet", key="ch_scan",
                     help="Loads every sport's games, props and fights (a few seconds the first time)."):
        note("Turn on the scan to see today's best single or 2-leg parlay at the price this step needs.")
    else:
        legs, coverage = all_sports_board("Today", False)
        opts = chmod.candidates(legs, need)
        if not opts:
            empty_card("Nothing on today's board pays enough at a decent hit chance.",
                       "Skip a day (the target odds adjust), or log your own bet below.")
        for i, o in enumerate(opts):
            title = ["Best pick", "Alternative", "Alternative"][i]
            H(f'<div class="card"><div class="ghead"><div class="pl-title">{title} · {chmod.american(o["dec"])}'
              f'<div class="psub">{len(o["legs"])} leg{"s" if len(o["legs"]) > 1 else ""} · {o["prob"] * 100:.0f}% to hit '
              f'(price implies {o["implied"] * 100:.0f}%) · pays {_money(stake * o["dec"])}</div></div>'
              f'<span class="badge {"strong" if o["ev"] >= 0 else "muted"}">{"+EV" if o["ev"] >= 0 else "−EV"}</span></div>'
              + "".join(leg_html(l) for l in o["legs"]) + "</div>")
            if st.button(f"Lock in: {title.lower()} ({chmod.american(o['dec'])})", key=f"ch_lock_{i}", width="stretch"):
                desc = " + ".join(l.pick for l in o["legs"])
                chmod.record(ch, desc, o["dec"], o["prob"], [chmod.leg_text(l) for l in o["legs"]], stake)
                chmod.save(ch)
                st.rerun()
        with st.expander("Today's board"):
            board_card(coverage)
    with st.expander("✍️ Log a bet you placed yourself"):
        desc = st.text_input("Bet", key="ch_own_desc", placeholder="e.g. Odunze 60+ rec yds + Chiefs ML")
        odds = st.number_input("American odds", -2000, 2000, -127, 1, key="ch_own_odds")
        dec = (1 + odds / 100) if odds > 0 else (1 + 100 / abs(odds)) if odds != 0 else 0
        if st.button("Add bet", key="ch_own_add", disabled=not desc or abs(odds) < 100, width="stretch"):
            chmod.record(ch, desc, dec, None, [], stake)
            chmod.save(ch)
            st.rerun()


def history_card(ch: dict) -> None:
    done = [b for b in ch["bets"] if b["result"] != "pending"]
    if not done:
        return
    rows = "".join(
        f'<div class="ch-h"><span class="res {"w" if b["result"] == "won" else "l" if b["result"] == "lost" else ""}">'
        f'{b["result"][0].upper()}</span><span class="d">{esc(b["desc"])}<small>{esc(b["date"])} · '
        f'{chmod.american(b["dec"])}</small></span><span class="m">{_money(b["stake"])}</span></div>'
        for b in reversed(done))
    H(f'<div class="card"><div class="sub-h" style="margin-top:0">History</div>{rows}</div>')


# --------------------------------------------------------------------------
# Route
# --------------------------------------------------------------------------
try:
    if sport == "UFC":
        render_ufc()
    elif sport == "MIX":
        render_mixed()
    elif sport == "CHAL":
        render_challenge()
    elif sport == "SOCCER":
        render_soccer()
    else:
        render_team_sport()
except Exception as e:  # last line of defence: never a crashed page
    empty_card("Something went wrong rendering this view.", f"{type(e).__name__}: {e}. Tap Refresh in Settings.")

notes = list(NOTES.values())
notes.append(f"Data: ESPN public APIs{' + stats.nba.com (optional)' if sport == 'NBA' else ''} · Updated "
             f"{now_et:%b %d, %-I:%M %p} ET · Estimates only, not betting advice.")
H('<div class="note" style="margin-top:10px">' + "<br>".join(esc(n) for n in notes) + "</div>")

with st.expander("How the models work"):
    st.markdown("""
**Basketball (NBA / NCAAB)** — Team scoring for/against from ESPN schedules (season blended 70/30 with last 5,
regressed toward league average early in the season). With possessions from ESPN team stats (or stats.nba.com
when reachable), points = possessions × offensive rating × opponent defensive rating / league rating. Home court,
neutral sites, back-to-backs (NBA) and injured scorers adjust the line. Props: recency-weighted last-10 per-minute
production × projected minutes (absent teammates' minutes redistributed), pace- and opponent-adjusted.

**NFL** — Points for/against model with home field. Props: weighted last-8 yards / receptions × opponent points
allowed and the team's implied total.

**NHL** — Poisson goal model (goals for × opponent goals against / league average, home ice, back-to-backs);
win % includes overtime, plus puck-line probability. Props: weighted last-10 goals / assists / shots × opponent
goals allowed; anytime-goal probability = 1 − e^(−projected goals).

**Soccer** — Poisson goal model per team (goals for × opponent goals against / league average, season blended
with last 5, home edge, neutral venues). Gives home / draw / away, projected score, total goals and both-teams-to-
score. Cup ties and national teams with fewer than 3 matches in that competition borrow recent results from
ESPN's all-competitions schedule, then domestic leagues (or other international competitions).

**UFC** — Logistic model on smoothed win %, experience, reach, age, striking differential and grappling rates,
blended 50/50 with the vig-free betting line when ESPN lists odds. Method of victory blends weight-class finish
rates with each fighter's own KO / submission / decision wins (5-round fights shift toward finishes).
""")
