"""Player report UI: line header, last-10 bar chart, splits-vs-line chart, and the model report
(recommendation, confidence ring, projection/edge bar, summary and FORM / MATCHUP / RISK notes).

Every number comes from the player's ESPN game logs and the projection model — nothing is generated text
beyond templates filled with those numbers.
"""

from __future__ import annotations

import html as _html

import numpy as np

esc = _html.escape

REPORT_CSS = """
<style>
.rp{background:var(--card);border:1px solid var(--line);border-radius:22px;margin:0 0 14px;overflow:hidden}
.rp-sec{padding:16px 16px;border-top:1px solid var(--line)}
.rp-sec:first-child{border-top:0}
.crumb{color:var(--muted);font-size:.86rem}
.crumb b{color:var(--text);font-weight:600;margin-left:6px}
.crumb span{margin:0 8px;opacity:.6}
.rp-head{display:flex;gap:14px;align-items:center}
.rp-shot{position:relative;width:64px;height:64px;border-radius:16px;overflow:hidden;flex:none;
background:linear-gradient(160deg,#1c1d22,#0e0f12);border:1px solid var(--line);display:flex;align-items:center;
justify-content:center;color:var(--muted);font-weight:800}
.rp-shot .im{position:absolute;inset:0;background-size:cover;background-position:center 18%}
.rp-name{font-weight:700;font-size:1.15rem;color:var(--text)}
.rp-sub{color:var(--muted);font-size:.9rem;margin-top:3px}
.rp-team{margin-left:auto;align-self:flex-start;color:var(--gold);font-weight:700;font-size:.9rem;letter-spacing:.03em}
.rp-line{display:flex;align-items:flex-end;justify-content:space-between;gap:10px;margin-top:16px}
.rp-line .lbl2{color:var(--muted);font-size:.85rem}
.rp-big{font-size:3.1rem;font-weight:800;line-height:1;color:var(--text);letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.rp-big small{font-size:.8rem;font-weight:600;color:var(--blue);margin-left:6px;letter-spacing:0}
.rp-stats{display:flex;gap:14px;text-align:center}
.rp-stats .k{color:var(--muted);font-size:.72rem}
.rp-stats .v{font-size:1.25rem;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
.pos{color:var(--green)}.neg{color:var(--pink)}.amb{color:var(--amber)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
/* last-10 bars */
.bars{position:relative;background:#08090b;border:1px solid var(--line);border-radius:16px;padding:14px 8px 8px;margin-top:14px}
.bars-plot{position:relative;height:150px;display:flex;align-items:flex-end;gap:5px;margin:0 2px}
.bar{flex:1;position:relative;height:100%;display:flex;align-items:flex-end}
.bar .fill{width:100%;border-radius:6px 6px 3px 3px;background:linear-gradient(180deg,#3ddc84,#1f9d5a)}
.bar.u .fill{background:linear-gradient(180deg,#f43f5e,#b4234a)}
.bar .val{position:absolute;left:0;right:0;text-align:center;font-size:.68rem;font-weight:700;color:var(--green)}
.bar.u .val{color:var(--pink)}
.bars .lineh{position:absolute;left:2px;right:2px;border-top:2px dashed #4f8cff;pointer-events:none}
.bars .linetag{position:absolute;right:-2px;transform:translateY(-50%);background:#2b6ef3;color:#fff;font-size:.66rem;
font-weight:700;padding:2px 6px;border-radius:6px}
.bars-lab{display:flex;gap:5px;margin:7px 2px 0}
.bars-lab span{flex:1;text-align:center;font-size:.58rem;line-height:1.25;color:var(--text);font-family:ui-monospace,Menlo,monospace;
white-space:nowrap;overflow:hidden}
.bars-lab small{display:block;color:var(--muted);font-size:.54rem}
/* splits vs line */
.spl-h{font-size:.98rem;color:var(--text)}
.spl-h span{color:var(--muted)}
.spl-sub{color:var(--muted);font-size:.78rem;margin-top:3px;font-family:ui-monospace,Menlo,monospace}
.spl-axis{display:grid;grid-template-columns:1fr 1.25fr 56px;gap:6px;margin:14px 0 2px}
.spl-axis .mid{position:relative;height:22px}
.spl-axis .u,.spl-axis .o{position:absolute;top:3px;font-size:.62rem;font-weight:700;letter-spacing:.06em}
.spl-axis .u{right:calc(50% + 30px);color:var(--pink)}.spl-axis .o{left:calc(50% + 30px);color:var(--green)}
.spl-axis .tag{position:absolute;left:50%;transform:translateX(-50%);background:#fff;color:#000;font-size:.7rem;font-weight:700;
padding:1px 7px;border-radius:6px;font-family:ui-monospace,Menlo,monospace;z-index:1}
.spl-row .trk:before{content:"";position:absolute;left:50%;top:-9px;bottom:-9px;border-left:2px dashed #3a3b40}
.spl-row{display:grid;grid-template-columns:1fr 1.25fr 56px;align-items:center;gap:6px;padding:9px 0}
.spl-row .nm{font-size:.86rem;color:var(--text);line-height:1.2}
.spl-row .nm small{display:block;color:var(--muted);font-size:.66rem}
.spl-row .trk{position:relative;height:26px}
.spl-row .b{position:absolute;top:2px;bottom:2px;border-radius:7px;background:linear-gradient(90deg,#1f9d5a,#3ddc84)}
.spl-row .b.n{background:linear-gradient(90deg,#f43f5e,#b4234a)}
.spl-row .vv{text-align:right;font-weight:700;font-size:.95rem;font-variant-numeric:tabular-nums}
.spl-row .vv small{display:block;font-size:.66rem;font-family:ui-monospace,Menlo,monospace}
/* model report */
.mr-h{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:.9rem}
.mr-h b{color:var(--text);font-weight:600}
.mr-top{display:flex;justify-content:space-between;align-items:center;margin-top:12px}
.mr-rec .k{color:var(--muted);font-size:.8rem}
.mr-rec .v{font-size:2.05rem;font-weight:800;line-height:1.1;margin-top:3px}
.mr-rec .v span{color:var(--text)}
.ring{width:64px;height:64px;border-radius:50%;display:grid;place-items:center;flex:none}
.ring .in{width:52px;height:52px;border-radius:50%;background:var(--card);display:grid;place-items:center;
font-weight:800;font-size:1.05rem;color:var(--text)}
.ring.sm{width:42px;height:42px}.ring.sm .in{width:34px;height:34px;font-size:.78rem}
.mr-conf{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:.8rem}
.mr-proj{display:flex;justify-content:space-between;margin-top:14px;font-size:.92rem;color:var(--muted)}
.mr-proj b{color:var(--text)}
.ebar{position:relative;height:9px;background:#18191d;border-radius:999px;margin-top:8px}
.ebar .f{position:absolute;top:0;bottom:0;border-radius:999px;background:linear-gradient(90deg,#1f9d5a,#3ddc84)}
.ebar .f.n{background:linear-gradient(90deg,#b4234a,#f43f5e)}
.ebar .mk{position:absolute;top:-5px;bottom:-5px;width:2px;background:#cfd3da}
.mr-sum{margin-top:14px;padding-left:12px;border-left:3px solid var(--green);font-size:.95rem;line-height:1.45;color:var(--text)}
.mr-sum.n{border-left-color:var(--pink)}
.mr-pt{display:flex;gap:10px;padding:11px 0;border-top:1px solid var(--line);font-size:.9rem;line-height:1.4;color:var(--text)}
.mr-pt:first-of-type{border-top:0}
.mr-pt i{flex:none;width:8px;height:8px;border-radius:50%;margin-top:7px;background:var(--green)}
.mr-pt i.r{background:var(--pink)}
.mr-pt em{font-style:normal;color:var(--muted);font-size:.72rem;letter-spacing:.08em;margin-right:6px}
/* pick list */
.pk{display:flex;align-items:center;gap:11px;padding:11px 12px;border-top:1px solid var(--line)}
.pk:first-child{border-top:0}
.pk .shot{width:42px;height:42px;border-radius:11px;position:relative;overflow:hidden;flex:none;background:#15161a;
display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:.7rem;font-weight:800}
.pk .shot .im{position:absolute;inset:0;background-size:cover;background-position:center 18%}
.pk .mid{flex:1;min-width:0}
.pk .nm{font-weight:700;font-size:.95rem;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pk .rec{font-size:.8rem;margin-top:2px}
.pk .ed{font-size:.72rem;font-family:ui-monospace,Menlo,monospace;text-align:right;min-width:62px}
.pklist{background:var(--card);border:1px solid var(--line);border-radius:18px;overflow:hidden;margin-bottom:12px}
</style>
"""


def _shot(url: str, name: str, cls: str = "rp-shot") -> str:
    ini = "".join(w[0] for w in name.split()[:2] if w).upper()
    im = f'<div class="im" style="background-image:url(\'{esc(url)}\')"></div>' if url else ""
    return f'<div class="{cls}">{esc(ini)}{im}</div>'


def ring(conf: int, small: bool = False) -> str:
    col = "#3ddc84" if conf >= 60 else "#f5b83d" if conf >= 45 else "#f43f5e"
    return (f'<div class="ring{" sm" if small else ""}" style="background:conic-gradient({col} {conf * 3.6:.0f}deg,'
            f'#202227 0)"><div class="in">{conf}</div></div>')


def splits_for(r: dict, m: str) -> list[tuple[str, float, int]]:
    return list((r.get("SPL") or {}).get(m) or [])


def confidence(r: dict, m: str, info: dict) -> int:
    """0-100: hit probability for the recommended side, last-10 consistency and how many splits agree."""
    v = info["verdict"]
    line = info["line"]
    over = r[f"P_{m}"] > line
    p = v.get("p") or 0.5
    hist = r.get(f"V10_{m}") or []
    hit = float(np.mean([(x > line) if over else (x < line) for x in hist])) if hist else 0.5
    spl = splits_for(r, m)
    agree = float(np.mean([(s[1] > line) if over else (s[1] < line) for s in spl])) if spl else 0.5
    conf = 100 * (0.45 * p + 0.30 * hit + 0.25 * agree)
    if r.get("Class") == "Q":
        conf -= 10
    if v.get("verdict") != "BET":
        conf = min(conf, 55)  # a PASS never reads as a high-conviction pick
    return int(max(1, min(99, round(conf))))


def pick_row(r: dict, m: str, label: str, info: dict, conf: int) -> str:
    line = info["line"]
    over = r[f"P_{m}"] > line
    edge = r[f"P_{m}"] - line
    side = "Over" if over else "Under"
    vb = info["verdict"]["verdict"]
    pill = f'<span class="vpill {"bet" if vb == "BET" else "pass"} sm">{"🟢" if vb == "BET" else "🟡"} {vb}</span>'
    return (f'<div class="pk">{_shot(r["Headshot"], r["Player"], "shot")}<div class="mid">'
            f'<div class="nm">{esc(r["Player"])}</div><div class="rec"><b class="{"pos" if over else "neg"}">{side}</b> '
            f'{line:g} {esc(label)} {pill}</div></div>'
            f'<div class="ed {"pos" if vb == "BET" else "amb"}">+{abs(edge):.1f} edge</div>{ring(conf, True)}</div>')


def _fmt(v: float, small: bool) -> str:
    return f"{v:.2f}" if small else f"{v:.1f}"


def report(sport_label: str, r: dict, m: str, label: str, kind: str, info: dict, conf: int,
           verdict_html: str, alt_html: str, news_html: str) -> str:
    line = info["line"]
    proj = r[f"P_{m}"]
    small = kind == "hockey" and m in ("G", "A", "PTS")
    hist = list(r.get(f"V10_{m}") or [])
    spl = splits_for(r, m)
    season = next((s[1] for s in spl if s[0] == "Season average"), None)
    vs = next((s for s in spl if s[0].startswith("vs.")), None)
    over = proj > line
    hit = float(np.mean([x > line for x in hist])) if hist else None
    opp_name = r.get("OppName") or r["Opp"]

    # --- header + line
    hit_cls = "pos" if (hit or 0) >= 0.5 else "neg"
    stats = (f'<div class="rp-stats"><div><div class="k">L10 hit rate</div><div class="v {hit_cls}">'
             f'{"—" if hit is None else f"{hit * 100:.0f}%"}</div></div>'
             f'<div><div class="k">Season</div><div class="v {"pos" if (season or 0) > line else "neg"}">'
             f'{"—" if season is None else _fmt(season, small)}</div></div>'
             f'<div><div class="k">vs {esc(r["Opp"])}</div><div class="v {"pos" if vs and vs[1] > line else "neg" if vs else ""}">'
             f'{_fmt(vs[1], small) if vs else "—"}</div></div></div>')
    head = (f'<div class="rp-sec"><div class="crumb">{esc(sport_label)}<span>/</span><b>{esc(r["Player"])}</b></div></div>'
            f'<div class="rp-sec"><div class="rp-head">{_shot(r["Headshot"], r["Player"])}<div>'
            f'<div class="rp-name">{esc(r["Player"])}</div><div class="rp-sub">{esc(label)} · vs. {esc(opp_name)}</div></div>'
            f'<div class="rp-team">{esc(r["Team"])}</div></div>'
            f'<div class="rp-line"><div><div class="lbl2">Line · {esc(info["label"])}</div>'
            f'<div class="rp-big">{line:g}</div></div>{stats}</div>')

    # --- last-10 bars (oldest -> newest)
    g10 = list(r.get("G10") or [])[: len(hist)]
    vals = list(reversed(hist))
    labs = list(reversed(g10))
    if vals:
        top = max(vals + [line]) * 1.18 or 1
        cols = []
        for v in vals:
            h = max(v / top * 100, 2)
            u = v <= line
            cols.append(f'<div class="bar{" u" if u else ""}"><div class="fill" style="height:{h:.0f}%"></div>'
                        f'<div class="val" style="bottom:calc({h:.0f}% + 3px)">{v:.0f}</div></div>')
        lab = "".join(f'<span>{"@" if x.get("away") else ""}{esc(x.get("opp", ""))}<small>{esc(x.get("date", ""))}</small></span>'
                      for x in labs) if labs else ""
        lp = line / top * 100
        bars = (f'<div class="bars"><div class="bars-plot">{"".join(cols)}'
                f'<div class="lineh" style="bottom:{lp:.0f}%"></div>'
                f'<div class="linetag" style="bottom:{lp:.0f}%;transform:translateY(50%)">{line:g}</div></div>'
                f'<div class="bars-lab">{lab}</div></div>')
    else:
        bars = '<div class="bars"><div class="empty">No game log yet this season.</div></div>'
    head += bars + "</div>"

    # --- splits vs line
    if spl:
        span = max([abs(s[1] - line) for s in spl] + [1.0])
        rows = []
        for name, v, n in spl:
            d = v - line
            w = abs(d) / span * 48
            pos = (f"left:50%;width:{w:.1f}%" if d >= 0 else f"right:50%;width:{w:.1f}%")
            rows.append(f'<div class="spl-row"><div class="nm">{esc(name)}<small>{n} game{"s" if n != 1 else ""}</small></div>'
                        f'<div class="trk"><div class="b{" n" if d < 0 else ""}" style="{pos}"></div></div>'
                        f'<div class="vv">{_fmt(v, small)}<small class="{"pos" if d >= 0 else "neg"}">{d:+.1f}</small></div></div>')
        splits = (f'<div class="rp-sec"><div class="spl-h">{esc(r["Player"])} <span>· {esc(label)} vs. {esc(opp_name)}</span></div>'
                  f'<div class="spl-sub">Each split against the line</div>'
                  f'<div class="spl-axis"><span></span><div class="mid"><span class="u">UNDER</span><span class="tag">{line:g}</span>'
                  f'<span class="o">OVER</span></div><span></span></div>'
                  f'<div class="spl">{"".join(rows)}</div></div>')
    else:
        splits = ""

    # --- model report
    side = "Over" if over else "Under"
    edge = proj - line
    lo, hi = min(line, proj), max(line, proj)
    pad = max(abs(edge), 1.0) * 1.6
    a, b = lo - pad, hi + pad
    pct = lambda x: (x - a) / (b - a) * 100  # noqa: E731
    fill = f'left:{pct(lo):.1f}%;width:{max(pct(hi) - pct(lo), 1):.1f}%'
    ebar = (f'<div class="ebar"><div class="f{"" if over else " n"}" style="{fill}"></div>'
            f'<div class="mk" style="left:{pct(line):.1f}%"></div></div>')
    k_over = sum(1 for s in spl if (s[1] > line) == over)
    summary = (f"Projects {_fmt(proj, small)} against a {line:g} line"
               + (f", and {k_over} of {len(spl)} splits sit on the {side.lower()} side." if spl else "."))
    pts = []
    if hist:
        n = len(hist)
        if over:
            pts.append(("FORM", f"Cleared {line:g} in {sum(x > line for x in hist)} of the last {n}, averaging "
                                f"{_fmt(float(np.mean(hist)), small)}.", ""))
        else:
            pts.append(("FORM", f"Stayed under {line:g} in {sum(x < line for x in hist)} of the last {n}, averaging "
                                f"{_fmt(float(np.mean(hist)), small)}.", ""))
    if vs and season is not None:
        pts.append(("MATCHUP", f"Averages {_fmt(vs[1], small)} against {opp_name} ({vs[2]} game"
                               f"{'s' if vs[2] != 1 else ''}), {vs[1] - season:+.1f} vs the season mark.",
                    "" if (vs[1] > line) == over else "r"))
    elif season is not None:
        pts.append(("MATCHUP", f"No meeting with {opp_name} yet; the projection sits {proj - season:+.1f} off the "
                               "season average after pace, opponent and minutes adjustments.", ""))
    risky = [s for s in spl if (s[1] > line) != over]
    side_hits = sum((x > line) if over else (x < line) for x in hist)
    if r.get("Class") == "Q":
        pts.append(("RISK", f"Listed {r.get('Status', 'questionable')} — check the final injury report.", "r"))
    elif hist and side_hits < len(hist) / 2:
        pts.append(("RISK", f"Recent form disagrees: only {side_hits} of the last {len(hist)} went "
                            f"{'over' if over else 'under'} {line:g}.", "r"))
    elif risky:
        worst = min(risky, key=lambda s: (s[1] - line) * (1 if over else -1))
        pts.append(("RISK", f"The {worst[0].lower()} split is the soft spot, at {_fmt(worst[1], small)}.", "r"))
    elif spl:
        weakest = min(spl, key=lambda s: (s[1] - line) * (1 if over else -1))
        pts.append(("RISK", f"Thinnest cushion: {weakest[0].lower()} at {_fmt(weakest[1], small)}.", "r"))
    bullets = "".join(f'<div class="mr-pt"><i class="{c}"></i><div><em>{k}</em>{esc(t)}</div></div>' for k, t, c in pts)
    model = (f'<div class="rp-sec"><div class="mr-h"><span>✦ <b>Model report</b></span>'
             f'<span class="mono">{esc(info["src"])}</span></div>'
             f'<div class="mr-top"><div class="mr-rec"><div class="k">Recommendation</div>'
             f'<div class="v {"pos" if over else "neg"}">{side} <span>{line:g}</span></div></div>'
             f'<div class="mr-conf">Confidence {ring(conf)}</div></div>'
             f'<div class="mr-proj"><span>Projects <b>{_fmt(proj, small)}</b></span>'
             f'<span class="mono pos">+{abs(edge):.1f} edge</span></div>{ebar}'
             f'<div class="mr-sum{"" if over else " n"}">{esc(summary)}</div>{bullets}{verdict_html}'
             f'{alt_html}{news_html}</div>')
    return f'<div class="rp">{head}{splits}{model}</div>'
