"""Shared HTML/CSS for the sportsbook-style cards."""

from __future__ import annotations

import html as _html
import math

import streamlit as st

CSS = """
<style>
:root{--bg:#0b0f14;--card:#121821;--card2:#18202b;--line:#232d3a;--text:#e8edf3;--muted:#8b97a7;
--green:#22c55e;--green-bg:rgba(34,197,94,.14);--red:#f43f5e;--red-bg:rgba(244,63,94,.14);
--amber:#f59e0b;--amber-bg:rgba(245,158,11,.15);--blue:#38bdf8;--blue-bg:rgba(56,189,248,.14)}
.stApp{background:var(--bg)}
.block-container{padding:4.2rem .8rem 3rem;max-width:760px}
/* keep the sport switcher reachable below Streamlit's toolbar */
.stMainBlockContainer{padding-top:4.2rem}
#MainMenu,footer{visibility:hidden}
h1,h2,h3{letter-spacing:-.01em}
.hero{display:flex;justify-content:space-between;align-items:flex-end;margin:.2rem 0 .8rem}
.brand{font-size:1.45rem;font-weight:800;color:var(--text)}
.brand span{color:var(--green)}
.hero-sub{color:var(--muted);font-size:.82rem;text-align:right;line-height:1.3}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--card);border:1px solid var(--line);
border-radius:14px;padding:4px}
.stTabs [data-baseweb="tab"]{flex:1;justify-content:center;border-radius:10px;padding:.55rem .2rem;
color:var(--muted);font-weight:600;height:auto}
.stTabs [data-baseweb="tab"] p{font-size:.86rem}
.stTabs [aria-selected="true"]{background:var(--card2);color:var(--text)}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px 14px 12px;
margin:0 0 12px;color:var(--text)}
.ghead{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:8px}
.gtime{color:var(--muted);font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.tags{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.pill{display:inline-block;font-size:.66rem;font-weight:700;padding:3px 7px;border-radius:999px;
letter-spacing:.03em;text-transform:uppercase;white-space:nowrap}
.pill.live{background:var(--red-bg);color:var(--red)}
.pill.final{background:var(--card2);color:var(--muted)}
.pill.b2b{background:var(--amber-bg);color:var(--amber)}
.pill.out{background:var(--red-bg);color:var(--red)}
.pill.q{background:var(--amber-bg);color:var(--amber)}
.pill.ok{background:var(--green-bg);color:var(--green)}
.pill.info{background:var(--blue-bg);color:var(--blue)}
.pill.comp{background:var(--card2);color:var(--text);border:1px solid var(--line);text-transform:none;letter-spacing:0}
.pdrawer{margin-top:12px;border-top:1px solid var(--line);padding-top:8px}
.pdrawer summary{cursor:pointer;list-style:none;font-size:.8rem;font-weight:800;color:var(--text);padding:4px 0;
display:flex;align-items:center;gap:6px}
.pdrawer summary::-webkit-details-marker{display:none}
.pdrawer summary:before{content:"▸";color:var(--green);transition:transform .15s}
.pdrawer[open] summary:before{transform:rotate(90deg)}
.ptab{margin-top:6px;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.pt{display:grid;grid-template-columns:34px 1fr 1fr 1.35fr 1fr 1.35fr;gap:4px;padding:7px 8px;font-size:.76rem;
align-items:center;border-top:1px solid var(--line);font-variant-numeric:tabular-nums}
.pt span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pt.head{border-top:0;background:var(--card2);color:var(--muted);font-size:.62rem;font-weight:800;
text-transform:uppercase;letter-spacing:.05em}
.pt.fg{background:rgba(34,197,94,.06)}
.pt .sg{font-weight:800;color:var(--green)}
.pt-note{color:var(--muted);font-size:.68rem;margin-top:6px}
.verdict{display:flex;align-items:center;gap:8px;margin:10px 0 2px;padding:8px 10px;border-radius:12px;
font-size:.8rem;line-height:1.35}
.verdict.bet{background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.45)}
.verdict.pass{background:rgba(245,158,11,.10);border:1px solid rgba(245,158,11,.40)}
.verdict .vtxt{color:var(--text);min-width:0}
.vpill{display:inline-block;flex:none;font-size:.72rem;font-weight:900;padding:4px 9px;border-radius:999px;
letter-spacing:.04em;white-space:nowrap}
.vpill.bet{background:var(--green);color:#04210f}
.vpill.pass{background:var(--amber);color:#2a1a02}
.vpill.sm{font-size:.6rem;padding:2px 6px}
.lad-title{font-size:.7rem;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;
margin:10px 0 5px}
.ladtab{border:1px solid var(--line);border-radius:10px;overflow:hidden}
.lad{display:grid;grid-template-columns:1.25fr 1.35fr .9fr 1.1fr;gap:4px;padding:6px 8px;font-size:.76rem;
align-items:center;border-top:1px solid var(--line);font-variant-numeric:tabular-nums}
.lad small{color:var(--muted);font-size:.66rem}
.lad.head{border-top:0;background:var(--card2);color:var(--muted);font-size:.6rem;font-weight:800;
text-transform:uppercase;letter-spacing:.05em}
.lad.main{background:rgba(56,189,248,.08)}
.lad span:last-child{text-align:right}
.lg-main-tag{font-size:.6rem;font-weight:800;color:var(--blue)}
.region-h{color:var(--muted);font-size:.68rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;margin:10px 2px 4px}
.trow{display:flex;align-items:center;gap:10px;padding:6px 0}
.logo{width:34px;height:34px;object-fit:contain;flex:none}
.logo-fb{width:34px;height:34px;border-radius:50%;background:var(--card2);display:flex;align-items:center;
justify-content:center;font-size:.62rem;font-weight:800;color:var(--muted);flex:none}
.tinfo{flex:1;min-width:0}
.tabbr{font-weight:800;font-size:1rem}
.tabbr .rec{color:var(--muted);font-weight:500;font-size:.75rem;margin-left:4px}
.tname{color:var(--muted);font-size:.75rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tscore{font-weight:800;font-size:1.3rem;min-width:34px;text-align:right}
.tproj{text-align:right;min-width:62px}
.tproj .lbl,.lbl{display:block;color:var(--muted);font-size:.6rem;font-weight:700;letter-spacing:.06em;
text-transform:uppercase}
.tproj .num{font-weight:800;font-size:1.15rem;color:var(--green)}
.winbar{display:flex;height:6px;border-radius:999px;overflow:hidden;background:var(--card2);margin:8px 0 4px}
.winbar .a,.winbar .h{background:#64748b}.winbar .d{background:#3b4656}.winbar .fav{background:var(--green)}
.winlbl{display:flex;justify-content:space-between;color:var(--muted);font-size:.7rem;margin-bottom:10px}
.mkts{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.mkt{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:9px 8px;text-align:center}
.mkt .val{font-weight:800;font-size:1rem;margin:2px 0}
.mkt .sub{color:var(--muted);font-size:.68rem;margin-bottom:6px;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.badge{display:inline-block;font-size:.64rem;font-weight:800;padding:4px 8px;border-radius:8px;
letter-spacing:.02em;white-space:nowrap}
.badge.strong{background:var(--green);color:#04210f}
.badge.lean{background:var(--green-bg);color:var(--green)}
.badge.under-strong{background:var(--red);color:#2a0710}
.badge.under-lean{background:var(--red-bg);color:var(--red)}
.badge.muted{background:transparent;color:var(--muted);border:1px solid var(--line)}
.note{color:var(--muted);font-size:.74rem;margin:2px 2px 12px;line-height:1.45}
.phead{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.pname{font-weight:800;font-size:1rem;line-height:1.2}
.pmeta{flex:1;min-width:0}
.avatar{position:relative;width:54px;height:54px;border-radius:50%;flex:none;overflow:hidden;
background:linear-gradient(145deg,#1f2a37,#121821);border:2px solid var(--line);display:flex;
align-items:center;justify-content:center}
.avatar span{color:var(--muted);font-weight:800;font-size:.85rem}
.av-img{position:absolute;inset:0;background-size:cover;background-position:center 15%;background-repeat:no-repeat}
div[data-testid="stButtonGroup"] button{border-radius:10px}
.psub{color:var(--muted);font-size:.74rem;margin-top:2px}
.pgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px}
.pbox{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:7px 4px;text-align:center}
.pbox .num{font-weight:800;font-size:1.05rem;margin-top:1px}
.pbox.hl{border-color:rgba(34,197,94,.45)}
.pbox.hl .num{color:var(--green)}
.l5{display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin-bottom:8px}
.chip{font-size:.72rem;font-weight:700;padding:3px 7px;border-radius:7px;min-width:26px;text-align:center}
.chip.o{background:var(--green-bg);color:var(--green)}
.chip.u{background:var(--red-bg);color:var(--red)}
.mini{display:flex;justify-content:space-between;color:var(--muted);font-size:.72rem;border-top:1px solid var(--line);
padding-top:8px}
.mini b{color:var(--text)}
.ihead{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.irow{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:8px 0;
border-top:1px solid var(--line)}
.irow .who{min-width:0}
.irow .nm{font-weight:700;font-size:.9rem}
.irow .dt{color:var(--muted);font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:56vw}
.empty{color:var(--muted);font-size:.8rem;padding:6px 0}
div[data-testid="stExpander"]{background:var(--card);border:1px solid var(--line);border-radius:14px}
@media (max-width:420px){.mkt .val{font-size:.92rem}.pbox .num{font-size:.95rem}.brand{font-size:1.25rem}}
.rank{color:var(--amber);font-size:.72rem;font-weight:800;margin-right:5px}
.seg-head .lglogo{display:inline-flex}
.seg-head{color:var(--muted);font-size:.72rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;
margin:16px 4px 8px;display:flex;align-items:center;gap:8px}
.seg-head:after{content:"";flex:1;height:1px;background:var(--line)}
.frow{display:flex;align-items:center;gap:10px;padding:6px 0}
.flag{width:18px;height:12px;border-radius:2px;object-fit:cover;vertical-align:middle;margin-right:5px}
.fname{font-weight:800;font-size:.98rem;line-height:1.2}
.fname.win{color:var(--green)}
.fsub{color:var(--muted);font-size:.74rem;margin-top:2px}
.fprob{text-align:right;min-width:58px}
.fprob .num{font-weight:800;font-size:1.1rem}
.winbar .r{background:#ef4444}.winbar .b{background:#3b82f6}
.pickrow{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-top:4px;flex-wrap:wrap}
.tot{display:grid;grid-template-columns:1fr auto 1fr;gap:9px 10px;align-items:center;margin-top:6px}
.tot .l{text-align:right;font-weight:700;font-size:.9rem}
.tot .r{font-weight:700;font-size:.9rem}
.tot .m{color:var(--muted);font-size:.62rem;text-transform:uppercase;letter-spacing:.07em;text-align:center}
.tot .adv{color:var(--green)}
.mbar{display:flex;align-items:center;gap:8px;margin:5px 0}
.mbar .lab{width:92px;font-size:.74rem;color:var(--muted)}
.mbar .track{flex:1;height:8px;background:var(--card2);border-radius:999px;overflow:hidden}
.mbar .fill{height:100%;background:var(--green);border-radius:999px}
.mbar .fill.b{background:#3b82f6}.mbar .fill.r{background:#ef4444}
.mbar .pct{width:42px;text-align:right;font-size:.76rem;font-weight:700}
.sub-h{font-weight:800;font-size:.8rem;margin:12px 0 4px;color:var(--text)}
.ev-head{display:flex;justify-content:space-between;align-items:center;gap:10px}
.ev-name{font-weight:800;font-size:1.05rem}
.factor{display:flex;justify-content:space-between;font-size:.76rem;padding:5px 0;border-top:1px solid var(--line)}
.factor .v{font-weight:700}
.factor .v.pos{color:var(--green)}.factor .v.neg{color:var(--red)}
div[data-testid="stButtonGroup"]{gap:6px}
.news{margin-top:9px;padding-top:8px;border-top:1px solid var(--line);font-size:.76rem;line-height:1.35;color:var(--text)}
.news a{color:var(--text);text-decoration:none}
.news span{color:var(--muted);white-space:nowrap}
.pbox.book{border-color:rgba(56,189,248,.5)}
.pl-title{font-weight:800;font-size:1.12rem;color:var(--text);text-transform:none;letter-spacing:0}
.leg{display:flex;align-items:flex-start;gap:12px;padding:12px 0;border-top:1px solid var(--line)}
.lglogo{position:relative;flex:none;border-radius:9px;background:#f4f6f8;display:flex;align-items:center;
justify-content:center;overflow:hidden}
.lglogo span{font-size:.95rem}
.lglogo .lg-img{position:absolute;inset:3px;background-size:contain;background-position:center;background-repeat:no-repeat}
.lg-vis{position:relative;width:48px;height:48px;flex:none;margin:2px 4px 0 0}
.lg-vis .tm{position:relative;width:48px;height:48px;border-radius:13px;background:#f4f6f8;display:flex;
align-items:center;justify-content:center;overflow:hidden}
.lg-vis .tm span{font-size:.6rem;font-weight:800;color:#475569;text-align:center;line-height:1.1;padding:0 2px}
.lg-vis .tm .im{position:absolute;inset:5px;background-size:contain;background-position:center;background-repeat:no-repeat}
.lg-vis .tm.two .im.a{inset:3px 17px 17px 3px}
.lg-vis .tm.two .im.b{inset:17px 3px 3px 17px}
.lg-vis .tm.photo{border-radius:50%;background:linear-gradient(145deg,#1f2a37,#121821);border:2px solid var(--line)}
.lg-vis .tm.photo span{color:var(--muted)}
.lg-vis .tm.photo .im{inset:0;background-size:cover;background-position:center 15%}
.lg-vis .lgb{position:absolute;right:-7px;bottom:-7px;border:2px solid var(--card);border-radius:8px;line-height:0}
.lg-vis .lgb .lglogo{border-radius:6px}
.lg-vis .lgb .lglogo span{font-size:.6rem}
.lg-when{color:var(--muted);font-size:.78rem;margin-top:3px}
.lg-main{flex:1;min-width:0}
.lg-pick{font-weight:800;font-size:1.08rem;line-height:1.25;color:var(--text)}
.lg-sub{color:#c3ccd8;font-size:.85rem;line-height:1.4;margin-top:5px}
.lg-sub .mk{display:inline-block;font-size:.62rem;font-weight:800;padding:2px 7px;border-radius:6px;
background:var(--card2);border:1px solid var(--line);color:var(--text);margin-right:6px;letter-spacing:.04em;
text-transform:uppercase;vertical-align:1px}
.lg-note{color:var(--blue);font-size:.74rem;margin-top:3px}
.lg-edge{display:inline-block;margin-top:5px;font-size:.7rem;font-weight:800;color:var(--green);
background:var(--green-bg);padding:2px 7px;border-radius:6px}
.lg-p{text-align:right;min-width:52px}
.lg-p .num{display:block;font-weight:800;font-size:1.25rem;color:var(--green);line-height:1.1}
.brand .lglogo{display:inline-flex;vertical-align:-6px;margin-right:6px}
.pbox.book .lbl{color:var(--blue)}
</style>
"""

esc = _html.escape


def H(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def logo_html(url: str, abbr: str) -> str:
    if url:
        return f'<img class="logo" src="{esc(url)}" alt="{esc(abbr)}">'
    return f'<div class="logo-fb">{esc(abbr[:4])}</div>'


def avatar_html(url: str, name: str, size: int = 54) -> str:
    initials = "".join(w[0] for w in name.split()[:2] if w).upper()
    img = f'<div class="av-img" style="background-image:url(\'{esc(url)}\')"></div>' if url else ""
    style = f' style="width:{size}px;height:{size}px"' if size != 54 else ""
    return f'<div class="avatar"{style}><span>{esc(initials)}</span>{img}</div>'


LEAGUE_EMOJI = {"NBA": "🏀", "NCAAB": "🏀", "NFL": "🏈", "NHL": "🏒", "UFC": "🥊", "Soccer": "⚽"}


def league_logo_html(sport: str, size: int = 34) -> str:
    """League logo on a light tile (logos are dark-on-light); the emoji shows if the image can't load."""
    from .common import LEAGUE_LOGOS

    url = LEAGUE_LOGOS.get(sport, "")
    img = f'<div class="lg-img" style="background-image:url(\'{esc(url)}\')"></div>' if url else ""
    return (f'<div class="lglogo" style="width:{size}px;height:{size}px" title="{esc(sport)}">'
            f'<span>{LEAGUE_EMOJI.get(sport, "⚽")}</span>{img}</div>')


def edge_badge(edge, lean: float, strong: float, pos: str, neg: str, neg_is_red: bool = False, fmt: str = "{:.1f}") -> str:
    if edge is None or (isinstance(edge, float) and math.isnan(edge)):
        return '<span class="badge muted">NO LINE</span>'
    a = abs(edge)
    if a < lean:
        return '<span class="badge muted">NO EDGE</span>'
    level = "strong" if a >= strong else "lean"
    cls = f"under-{level}" if (edge < 0 and neg_is_red) else level
    label = pos if edge > 0 else neg
    return f'<span class="badge {cls}">{esc(label)} · {fmt.format(a)}</span>'


def status_pill(cls: str, status: str) -> str:
    if cls == "OUT":
        return f'<span class="pill out">{esc(status or "Out")}</span>'
    if cls == "Q":
        return f'<span class="pill q">{esc(status or "Questionable")}</span>'
    return ""


def verdict_html(v: dict | None, compact: bool = False) -> str:
    """Prominent BET / PASS strip: 🟢 BET · pick · reason   or   🟡 PASS · reason."""
    if not v:
        return ""
    bet = v["verdict"] == "BET"
    icon, cls = ("🟢", "bet") if bet else ("🟡", "pass")
    pick = f'<b>{esc(v["pick"])}</b>' if v.get("pick") else ""
    if compact:
        return f'<span class="vpill {cls}">{icon} {v["verdict"]}</span>'
    return (f'<div class="verdict {cls}"><span class="vpill {cls}">{icon} {v["verdict"]}</span>'
            f'<span class="vtxt">{pick}{" · " if pick else ""}{esc(v["reason"])}</span></div>')


def ladder_html(title: str, rows: list[dict], fmt_line, cols: tuple[str, ...] = ("Line", "Hit %", "Fair", "")) -> str:
    """One alt-line table. rows: line, p, fair, optional hit, optional verdict, main flag."""
    head = "".join(f"<span>{esc(c)}</span>" for c in cols)
    body = []
    for r in rows:
        v = r.get("verdict")
        tag = (f'<span class="vpill {"bet" if v == "BET" else "pass"} sm">{"🟢" if v == "BET" else "🟡"} {v}</span>'
               if v else (f'<span class="lg-main-tag">MAIN</span>' if r.get("main") else ""))
        hit = f'{r["hit"] * 100:.0f}%' if r.get("hit") is not None else ""
        cells = [fmt_line(r), f'{r["p"] * 100:.0f}%' + (f' <small>L10 {hit}</small>' if hit else ""), r["fair"], tag]
        body.append(f'<div class="lad{" main" if r.get("main") else ""}">'
                    + "".join(f"<span>{c}</span>" for c in cells) + "</div>")
    return (f'<div class="lad-title">{esc(title)}</div><div class="ladtab"><div class="lad head">{head}</div>'
            + "".join(body) + "</div>")


def note(text: str) -> None:
    H(f'<div class="note">{esc(text)}</div>')


def empty_card(title: str, text: str = "") -> None:
    H(f'<div class="card"><b>{esc(title)}</b>' + (f'<div class="note">{esc(text)}</div>' if text else "") + "</div>")
