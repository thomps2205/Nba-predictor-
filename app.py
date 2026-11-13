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
import engine.ui

_ENGINE = [engine.common, engine.espn, engine.teams, engine.props, engine.lines, engine.news, engine.ufc,
           engine.ufcstats, engine.parlays, engine.ui]  # dependency order


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

from engine import espn, lines as linesmod, news as newsmod, parlays, props as propmod, teams, ufc, ufcstats  # noqa: E402
from engine.common import ET, NOTES, cache_fresh, clear_disk_cache, fmt_spread, ml_to_prob, norm_name  # noqa: E402
from engine.espn import LEAGUES, League  # noqa: E402
from engine.ui import (CSS, H, avatar_html, edge_badge, empty_card, esc, league_logo_html, logo_html,  # noqa: E402
                       note, status_pill)

st.set_page_config(page_title="Sharp Board", page_icon="🏆", layout="centered", initial_sidebar_state="collapsed")
H(CSS)

NOTES.clear()  # notes describe this run only (engine code adds stale-cache / endpoint warnings as it goes)
now_et = datetime.now(ET)
today_et = now_et.date()
SPORTS = ["NBA", "NCAAB", "NFL", "NHL", "UFC", "MIX"]
SPORT_ICON = {"NBA": "🏀", "NCAAB": "🏀", "NFL": "🏈", "NHL": "🏒", "UFC": "🥊", "MIX": "🎟"}
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
sport = st.segmented_control("Sport", SPORTS, default="NBA", key="sport_pick",
                             label_visibility="collapsed", width="stretch")
if sport is None:  # tapping the active button deselects it; keep the last sport
    sport = st.session_state.get("last_sport", "NBA")
st.session_state["last_sport"] = sport
lg = LEAGUES.get(sport)  # None for the mixed-parlay view

with st.expander("⚙️  Settings", expanded=False):
    if sport == "UFC":
        st.caption("UFC shows the current or next fight card automatically.")
        sel_date = today_et
    elif sport == "MIX":
        st.caption("Mixed parlays use today's NBA / NHL / Top 25 college games, NFL games in the next 3 days, "
                   "and this week's UFC card.")
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
    icon = SPORT_ICON[sport] if sport == "MIX" else league_logo_html(sport, 36)
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
    b2b = teams.back_to_back_teams(profiles, slate_day) if teams.CFG[lg_key].get("b2b") else set()
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


@st.cache_data(ttl=300, show_spinner=False, max_entries=32)
def cached_news(lg_key: str, team_ids: tuple, tick: int) -> list[dict]:
    return newsmod.articles(LEAGUES[lg_key], team_ids)


@st.cache_data(ttl=600, show_spinner=False, max_entries=64)
def compute_props(lg_key: str, game_id: str, q: float, tick: int, _game: dict, _proj: dict,
                  _profiles: dict, _base: dict, _b2b: set, _inj: list[dict]) -> pd.DataFrame:
    return propmod.build_props(LEAGUES[lg_key], _game, _proj, _profiles, _base, _b2b, _inj, q)


def matchup_card(g: dict, p: dict | None, b2b: set[str], kind: str) -> str:
    h, a = g["home"], g["away"]
    st_pill = ""
    if g["state"] == "in":
        st_pill = f'<span class="pill live">● {esc(g["status"])}</span>'
    elif g["state"] == "post":
        st_pill = f'<span class="pill final">{esc(g["status"])}</span>'
    tags = "".join(f'<span class="pill b2b">{esc(t["abbr"])} B2B</span>' for t in (a, h) if t["id"] in b2b)
    if g["neutral"]:
        tags += '<span class="pill info">Neutral</span>'
    tip = g["tip"].strftime("%a %-I:%M %p ET") if g["tip"] is not None else ""
    dec = 1 if kind == "hockey" else 0

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
    hp = p["home_win_prob"] * 100
    fav_cls_a, fav_cls_h = ("a", "h fav") if hp >= 50 else ("a fav", "h")
    body += (f'<div class="winbar"><div class="{fav_cls_a}" style="width:{100 - hp:.0f}%"></div>'
             f'<div class="{fav_cls_h}" style="width:{hp:.0f}%"></div></div>'
             f'<div class="winlbl"><span>{esc(a["abbr"])} {100 - hp:.0f}%</span><span>{esc(h["abbr"])} {hp:.0f}%</span></div>')
    o = g["odds"]
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
                f'<div class="sub">Mkt {esc(o["details"] or "—")}</div>{s_badge}</div>')
        t_lean, t_strong = (2.0, 5.0) if kind == "basketball" else (1.5, 3.5)
    t_edge = (p["total"] - o["total"]) if o["total"] is not None else None
    t_badge = edge_badge(t_edge, t_lean, t_strong, f'OVER {o["total"]}' if o["total"] else "",
                         f'UNDER {o["total"]}' if o["total"] else "", neg_is_red=True)
    box2 = (f'<div class="mkt"><span class="lbl">Total</span><div class="val">{p["total"]:.1f}</div>'
            f'<div class="sub">Mkt {esc(str(o["total"]) if o["total"] is not None else "—")}</div>{t_badge}</div>')
    mh, ma = ml_to_prob(o["ml_home"]), ml_to_prob(o["ml_away"])
    if mh and ma:
        mkt_h = mh / (mh + ma)  # vig-free
        ml_badge = edge_badge((p["home_win_prob"] - mkt_h) * 100, 3, 7, f'{h["abbr"]} ML', f'{a["abbr"]} ML',
                              fmt="{:.0f}%")
        ml_sub = f'Mkt {fav["abbr"]} {(mkt_h if fav is h else 1 - mkt_h) * 100:.0f}%'
    else:
        ml_badge = '<span class="badge muted">MODEL</span>'
        ml_sub = f'{p["poss"]:.0f} poss' if p.get("poss") else "Win probability"
    box3 = (f'<div class="mkt"><span class="lbl">Win</span><div class="val">{esc(fav["abbr"])} '
            f'{max(hp, 100 - hp):.0f}%</div><div class="sub">{esc(ml_sub)}</div>{ml_badge}</div>')
    return body + f'<div class="mkts">{box1}{box2}{box3}</div></div>'


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


def prop_card(r: dict, m: str, label: str, kind: str, b2b_ids: set[str], book: tuple | None = None,
              news_item: dict | None = None) -> str:
    lean, strong = propmod.EDGE[kind][m]
    proj = r[f"P_{m}"]
    if book:
        line, line_label = book[0], book[1]
        hist = r.get(f"V10_{m}") or []
        hit_v = float(sum(v > line for v in hist) / len(hist) * 100) if hist else float("nan")
    else:
        line, line_label, hit_v = r[f"L_{m}"], "Model line", r[f"H_{m}"]
    edge = prop_edge(r, m, book)
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
    return (
        f'<div class="card"><div class="phead">{avatar_html(r["Headshot"], r["Player"])}'
        f'<div class="pmeta"><div class="pname">{esc(r["Player"])}</div>'
        f'<div class="psub">{esc(r["Team"])} vs {esc(r["Opp"])} · {esc(r["Info"])} '
        f'{status_pill(r["Class"], r["Status"])} {b2b} {extra}</div></div>{badge}</div>'
        '<div class="pgrid">'
        f'<div class="pbox hl"><span class="lbl">Proj {esc(label)}</span><div class="num">{f1.format(proj)}</div></div>'
        f'<div class="pbox{" book" if book else ""}"><span class="lbl">{esc(line_label)}</span>'
        f'<div class="num">{line:.1f}</div></div>'
        f'<div class="pbox"><span class="lbl">{l5_label}</span><div class="num">{r[f"A5_{m}"]:.{2 if small else 1}f}</div></div>'
        f'<div class="pbox"><span class="lbl">L10 hit</span><div class="num">{hit}</div></div>'
        '</div>'
        f'<div class="l5"><span class="lbl" style="margin-right:4px">Last 5</span>{chips or "—"}</div>'
        f'<div class="mini">{mini}</div>{news_html(news_item)}</div>'
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
    fails = prefetch(reqs, f"Loading {sk} team stats")
    if fails:
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

    slate, inj = load_team_data(sport, games)
    projections, b2b = slate["projections"], slate["b2b"]

    tabs = st.tabs(["🎯 Matchups", "📊 Player Props", "🎟 Parlays"] + (["🩹 Injuries"] if lg.has_injuries else []))
    with tabs[0]:
        note("Model line vs. market. Spread/ML badges name the side the model likes; total badges show the "
             "over/under lean. Number = edge in points (win-probability points for ML).")
        H("".join(matchup_card(g, projections.get(g["id"]), b2b, lg.kind) for g in games))
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
    with st.spinner("Checking sportsbook lines and news…"):
        book = cached_lines(lg.key, bucket(5))
        arts = cached_news(lg.key, (g["away"]["id"], g["home"]["id"]), bucket(5))
    markets = propmod.MARKETS[kind]
    mlabels = dict(markets)
    m = st.segmented_control("Market", [k for k, _ in markets], default=markets[0][0], key=f"mkt_{sport}",
                             format_func=lambda k: mlabels[k], label_visibility="collapsed",
                             width="stretch") or markets[0][0]
    side = st.segmented_control("Team", ["Both", g["away"]["abbr"], g["home"]["abbr"]], default="Both",
                                key=f"side_{sport}_{gid}", label_visibility="collapsed", width="stretch") or "Both"
    n_book = sum(1 for (_, mm) in book if mm == m)
    note((f"Edge = projection vs. the live PrizePicks / Underdog line where one is posted ({n_book} {mlabels[m]} "
          "lines on the board), otherwise vs. the last-5 average. " if n_book else
          "No PrizePicks / Underdog lines found right now, so edge = projection vs. last-5 average. ")
         + "Chips = last 5 games, green when over the line. Hit = share of last 10 games over the line.")
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
        r["_book"] = book.get((norm_name(r["Player"]), m))
        r["_e"] = abs(prop_edge(r, m, r["_book"]))
    recs.sort(key=lambda r: (r["_book"] is None, -r["_e"]))  # posted lines first, biggest edges on top
    H("".join(prop_card(r, m, mlabels[m], kind, slate["b2b"], r["_book"],
                        newsmod.for_player(arts, r["Id"], r["Player"])) for r in recs))


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
    return (f'<div class="card">{head}' + fighter_row(fa_, pa_, p, pred["pick"] == 0, fa_.get("winner"))
            + bar + fighter_row(fb_, pb_, 1 - p, pred["pick"] == 1, fb_.get("winner")) + pick_html + '</div>')


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
            legs += parlays.prop_legs(sk, lg_.kind, g, df, book, norm_name, labels)
        except Exception as e:
            NOTES[f"prop legs {sk}"] = f"{sk} prop legs skipped for a game ({type(e).__name__})."
    return legs


@st.fragment
def parlay_fragment(games: list[dict], slate: dict, inj: list[dict]) -> None:
    load_all = st.toggle("Scan player props in every game (default: the next 4 games; slower the first time)",
                         key=f"pl_props_{sport}")
    legs = parlays.game_legs(sport, lg.kind, games, slate["projections"])
    legs += gather_prop_legs(sport, games, slate, inj, load_all)
    note("Each leg's hit chance blends the model (60%) with the market (40%). Prices: ESPN moneylines, -110 "
         "spreads/totals, -115 props. Props use the PrizePicks / Underdog line when posted, otherwise the player's "
         "last-10 average as the line. One leg per game; real parlay odds may differ. Not betting advice.")
    render_parlays(legs)


def render_mixed() -> None:
    hero(f"{today_et:%a, %b %d}<br>All sports")
    load_all = st.toggle("Scan player props in every game (default: next 2 games per sport; slower first time)",
                         key="mix_props")
    legs, coverage = [], []
    for sk in ("NBA", "NCAAB", "NFL", "NHL"):
        try:
            with st.spinner(f"Loading {sk}…"):
                games = fetch_games(sk, today_et)
            if sk == "NFL":
                games = [g for g in games if g["date"] and 0 <= (g["date"] - today_et).days <= 3]
            elif sk == "NCAAB":
                games = [g for g in games if g["home"]["rank"] or g["away"]["rank"]]
            games = [g for g in games if g["state"] == "pre"][:PAGE_SIZE]
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

**UFC** — Logistic model on smoothed win %, experience, reach, age, striking differential and grappling rates,
blended 50/50 with the vig-free betting line when ESPN lists odds. Method of victory blends weight-class finish
rates with each fighter's own KO / submission / decision wins (5-round fights shift toward finishes).
""")
