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
import math
import os
import time
from datetime import date, datetime, timedelta

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
import engine.ufcstats
import engine.decide
import engine.report
import engine.ui

_ENGINE = [engine.common, engine.decide, engine.espn, engine.teams, engine.props, engine.lines, engine.news, engine.ufc,
           engine.ufcstats, engine.parlays, engine.ui, engine.report]  # dependency order


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

from engine import decide, report, espn, lines as linesmod, news as newsmod, parlays, props as propmod, teams, ufc, ufcstats  # noqa: E402
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
SPORTS = ["NBA", "NCAAB", "NFL", "NHL", "SOCCER", "UFC", "MIX"]
SPORT_ICON = {"NBA": "🏀", "NCAAB": "🏀", "NFL": "🏈", "NHL": "🏒", "SOCCER": "⚽", "UFC": "🥊", "MIX": "🎟"}
PAGE_SIZE = 12


def bucket(minutes: int = 10) -> int:
    """Changes every N minutes so cached computations refresh as new data lands."""
    return int(time.time() // (minutes * 60))


def prefetch(reqs: list[tuple], label: str) -> int:
    """Fetch every not-yet-cached request behind one progress bar. Returns the failure count."""
    todo = [r for r in reqs if not cache_fresh(r[0], r[1])]
    if not todo:
        return 0
    bar = st.progress(0.0, text=f"{label} 0/{len(todo)}")
    fails = 0
    for i, (_, _, fetch) in enumerate(todo, 1):
        try:
            fetch()
        except Exception:
            fails += 1
            if i >= 6 and fails == i:  # endpoint clearly down: stop instead of timing out on each
                break
        bar.progress(i / len(todo), text=f"{label} {i}/{len(todo)} (cached afterwards)")
    bar.empty()
    return fails


# --------------------------------------------------------------------------
# Header: sport switcher + settings
# --------------------------------------------------------------------------
SPORT_NAME = {"NBA": "NBA", "NCAAB": "College Hoops", "NFL": "NFL", "NHL": "NHL", "SOCCER": "Soccer", "UFC": "UFC",
              "MIX": "Mixed Parlays"}
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
    elif sport == "MIX":
        st.caption("Mixed parlays use today's NBA / NHL / Top 25 college games, top-league soccer, NFL games in "
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
    icon = SPORT_ICON[sport] if sport in ("MIX", "SOCCER") else league_logo_html(sport, 36)
    H(f'<div class="hero"><div class="brand">{icon}{esc(sport)} <span>Picks</span></div>'
      f'<div class="hero-sub">{sub}</div></div>')


# --------------------------------------------------------------------------
# Team sports
# --------------------------------------------------------------------------
def nfl_week_window(d: date) -> tuple[date, int]:
    start = d - timedelta(days=(d.weekday() - 1) % 7)  # back to Tuesday
    return start, 7


@st.cache_data(ttl=600, show_spinner=False, max_entries=48)
def compute_slate(lg_key: str, game_ids: tuple, q: float, inj_sig: tuple, tick: int,
                  _games: list[dict], _inj: list[dict]) -> dict:
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

    tab_m, tab_p = st.tabs(["🎯 Matchups", "🎟 Parlays"])
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
        slate = compute_slate(sk, tuple(g["id"] for g in games), float(q_sit), inj_sig, bucket(), games, inj)
    return slate, inj


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

    tabs = st.tabs(["🎯 Matchups", "📊 Player Props", "🎟 Parlays"] + (["🩹 Injuries"] if lg.has_injuries else []))
    with tabs[0]:
        note("Model line vs. market. Spread/ML badges name the side the model likes; total badges show the "
             "over/under lean. Number = edge in points (win-probability points for ML).")
        matchups_fragment(games, lambda g: slate, b2b, lg.kind)
    with tabs[1]:
        props_fragment(games, slate, inj)
    with tabs[2]:
        parlay_fragment(games, slate, inj)
    if lg.has_injuries:
        with tabs[3]:
            render_injuries(games, inj)


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
    for r in recs:
        r["_conf"] = report.confidence(r, m, r["_info"])
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
            pa_ = ufc.fighter_profile(fa_["id"], fa_, ufcstats.cached(fa_["name"]))
            pb_ = ufc.fighter_profile(fb_["id"], fb_, ufcstats.cached(fb_["name"]))
            pred = ufc.predict_bout(b, pa_, pb_)
        except Exception as e:
            NOTES[f"bout {b['id']}"] = f"Prediction failed for {fa_['name']} vs {fb_['name']}: {e}"
            continue
        out.append({"bout": b, "a": pa_, "b": pb_, "pred": pred})
    return out


def _rec_txt(r: tuple) -> str:
    return f"{r[0]}-{r[1]}" + (f"-{r[2]}" if r[2] else "")


def _last(name: str) -> str:
    return name.split()[-1] if name.split() else name


def fighter_row(f: dict, prof: dict, p: float, is_pick: bool, won) -> str:
    flag = f'<img class="flag" src="{esc(prof["flag"])}" alt="">' if prof.get("flag") else ""
    ml = f' · {f["ml"]:+.0f}' if f.get("ml") else ""
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
    if b["state"] == "pre":
        fp, f_ = (fa_, pa_) if pred["pick"] == 0 else (fb_, pb_)
        model_only = pred["p_model"] if pred["pick"] == 0 else 1 - pred["p_model"]
        v_html = verdict_html(decide.fight_decision(_last(f_["name"]), pick_p, fp.get("ml"), model_only))
    else:
        v_html = ""
    return (f'<div class="card">{head}' + fighter_row(fa_, pa_, p, pred["pick"] == 0, fa_.get("winner"))
            + bar + fighter_row(fb_, pb_, 1 - p, pred["pick"] == 1, fb_.get("winner")) + pick_html + v_html
            + '</div>')


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
    names = [f["name"] for b in card["bouts"] for f in b["fighters"] if f["name"] and f["name"] != "TBA"]
    prefetch([(ufcstats.key(n), ufcstats.TTL, lambda n=n: ufcstats.fetch(n)) for n in names],
             "Loading UFCStats.com fighter stats")
    missing = sum(ufcstats.cached(n) is None for n in names)
    if missing:
        NOTES["ufcstats"] = f"UFCStats.com had no match for {missing} of {len(names)} fighters (debuts or unreachable)."
    with st.spinner("Projecting bouts…"):
        return compute_card(card["id"], bucket(), card)


def render_card_tabs(card: dict, bouts: list[dict]) -> None:
    status = {"in": '<span class="pill live">● LIVE</span>', "post": '<span class="pill final">Final</span>'}.get(
        card["state"], '<span class="pill info">Upcoming</span>')
    start = card["start"].strftime("%a %b %-d · %-I:%M %p ET") if card["start"] is not None else ""
    H(f'<div class="card"><div class="ev-head"><div><div class="ev-name">{esc(card["name"])}</div>'
      f'<div class="psub">{esc(start)}{" · " + esc(card["venue"]) if card["venue"] else ""}</div></div>'
      f'{status}</div></div>')

    tab_card, tab_bd, tab_pl = st.tabs(["🥊 Fight Card", "🔍 Fight Breakdowns", "🎟 Parlays"])
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
        rows.append((lab, fmt(va, unit, d), fmt(vb, unit, d), (va, vb, better)))

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
    src = [f'{esc(f["name"])}: {"UFCStats.com + ESPN" if pf.get("ufcstats") else "ESPN"}'
           for f, pf in zip(x["bout"]["fighters"], (a, b))]
    src_note = f'<div class="note" style="margin-top:8px">Stats source · {" · ".join(src)}</div>'
    detail_note = "" if (a["detail_ok"] and b["detail_ok"]) else (
        '<div class="note" style="margin-top:8px">Some fighter details are missing from ESPN; '
        'the projection leans on records for those fighters.</div>')
    H(f'<div class="card">{head}{bar}<div class="sub-h">Tale of the tape</div><div class="tot">{tot}</div></div>'
      f'<div class="card"><div class="sub-h" style="margin-top:0">Method of victory</div>'
      f'{mbars(a["name"], p, pred["methods_a"], "r")}{mbars(b["name"], 1 - p, pred["methods_b"], "b")}</div>'
      f'<div class="card"><div class="sub-h" style="margin-top:0">Model factors (log-odds, + favors '
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
    H('<div class="card"><div class="sub-h" style="margin-top:0">Today\'s board</div>'
      + "".join(f'<div class="factor" style="align-items:center"><span style="display:flex;align-items:center;'
                f'gap:10px">{league_logo_html(c.split(":")[0], 28)}<b>{esc(c.split(":")[0])}</b></span>'
                f'<span class="v">{esc(c.split(":", 1)[1].strip())}</span></div>' for c in coverage) + "</div>")
    note("Mixed parlays combine the best value game lines and player props across sports (one leg per game, "
         "legs from 2+ sports; props use the posted PrizePicks / Underdog line or the player's last-10 average). "
         "Hit chances blend each sport's model (60%) with the market (40%); prices are ESPN moneylines, -110 spreads/totals and -115 "
         "pick'em props. Not betting advice.")
    render_parlays(legs, mixed=True)


# --------------------------------------------------------------------------
# Route
# --------------------------------------------------------------------------
try:
    if sport == "UFC":
        render_ufc()
    elif sport == "MIX":
        render_mixed()
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
