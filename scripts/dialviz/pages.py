"""Page renderers. Each takes the assembled model and returns {relative path: html}."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from . import svg, zenanalyze_src
from .htmlkit import blob_url, chip, esc, filterbar, md_block, md_inline, page, src_cite, table

GATE_STATE_ORDER = ["done", "ready", "blocked", "fail", "pass", "na"]


# --------------------------------------------------------------------------- shared pieces

def _quote(q: dict, show_chip: bool = False) -> str:
    """A verbatim excerpt with its citation."""
    if q["kind"] == "code":
        return (f'<pre class="code"><code>{esc(q["text"])}</code></pre>'
                f'<div class="small">{src_cite(q["path"], q["line"], f"{q["path"]}:{q["line"]}–{q["end"]}")}</div>')
    if q["kind"] == "symbol":
        return f'{src_cite(q["path"], q["line"], q["label"] + " — " + q["path"] + ":" + str(q["line"]))}'
    head = ""
    if show_chip and q.get("status") in ("pass", "fail"):
        head = chip(q["status"]) + " "
    label = ""
    if q["kind"] == "cell":
        label = f'<strong>{md_inline(q["row_key"], q["path"])}</strong>' + (f' · {esc(q["column"])}' if q.get("column") else "") + ": "
    note = f'<span class="cite muted">{esc(q["note"])}</span>' if q.get("note") else ""
    text = re.sub(r"^[-*]\s+", "", q["text"])
    body = md_inline(text, q["path"]) if "\n" not in text else md_block(text, q["path"])
    return (f'<blockquote class="src">{head}{label}{body}{note}'
            f'<span class="cite">{src_cite(q["path"], q["line"])}</span></blockquote>')


def _gate_chip(g: dict) -> str:
    out = chip(g["status"], tip=re.sub(r"\*\*", "", g["status_text"])[:300])
    if g["rule_pending"]:
        out += " " + chip("pending")
    return out


def _gate_link(g: dict, depth: int = 0) -> str:
    return f'<a href="{"../" * depth}gate/{esc(g["id"])}.html">{esc(g["name"])}</a>'


def _footer(m: dict) -> str:
    c = m["commit"][:12] if m["commit"] else "unknown"
    c += " + working-copy changes" if m.get("dirty") else ""
    return (f'Built {esc(m["built_utc"])} from zensim <span class="mono">{esc(c)}</span> · {len(m["sources"])} source files · '
            f'every number on this site is read from a committed source; nothing here is a new measurement. '
            f'<a href="{{BASE}}sources.html">Sources and coverage</a>')


def _render(m, *, title, current, body, depth=0, crumbs=()):
    return page(title=title, current=current, body=body, depth=depth, crumbs=crumbs,
                build_info=_footer(m).replace("{BASE}", "../" * depth))


def _gates_by_id(m):
    return {g["id"]: g for g in m["gates"]["gates"]}


def _prop_status(p: dict, gates: dict) -> tuple[str, list[str]]:
    """Summary chips for a property: production gate states, then measured state chips."""
    chips = []
    for gid in p.get("gates", []):
        g = gates[gid]
        chips.append(chip(g["status"], f'{g["name"]}: {g["status"]}', tip=re.sub(r"\*\*", "", g["status_text"])[:300]))
    meas = [q["status"] for q in p.get("state", []) if q.get("status") in ("pass", "fail")]
    if meas:
        worst = "fail" if "fail" in meas else "pass"
        chips.append(chip(worst, f"latest measurement: {worst}"))
    if p.get("rule_pending_note"):
        chips.append(chip("pending"))
    return (gates[p["gates"][0]]["status"] if p.get("gates") else "na"), chips


# --------------------------------------------------------------------------- charts from data

def chart_nearid(m) -> str:
    rows = m["nearid"]["rows"]
    items = [{"label": f'{r["model"]} highest nonidentical', "value": r["highest"],
              "tip": f'{r["model"]}\nhighest nonidentical {r["highest"]:.4f}\nper-ref gap {r["gap_lo"]:.3f}–{r["gap_hi"]:.3f}'}
             for r in rows]
    items += [{"label": f'{r["model"]} one-pixel ±1 min', "value": r["onepx_lo"], "color": "var(--s2)",
               "tip": f'{r["model"]}\none-pixel ±1 range {r["onepx_lo"]:.4f}–{r["onepx_hi"]:.4f}'} for r in rows]
    ch = svg.hbars(items, xmin=85, xmax=100, thresholds=[(99.0, "N1/N2 99.0"), (97.5, "identity band 97.5")],
                   label_w=200, title="Near-identity served scores")
    lad = svg.hbars([{"label": r["model"], "value": r["ladders"],
                      "tip": f'{r["model"]}\n{r["ladders"]}/{r["ladders_of"]} nonincreasing ladders'} for r in rows],
                    xmin=0, xmax=rows[0]["ladders_of"], thresholds=[(122, "N3 ≥ 122")], label_w=200,
                    title="Nonincreasing ladders")
    return (f'<h3>Near-identity panel (NEARID, 24 references × 27 rungs)</h3>{ch}'
            + svg.legend([("var(--s1)", "highest nonidentical score"), ("var(--s2)", "lowest one-pixel ±1 score")])
            + f'<h3>Nonincreasing ladders (of {rows[0]["ladders_of"]})</h3>{lad}'
            + f'<p class="small muted">Values from {src_cite(m["nearid"]["path"], m["nearid"]["line"])}. '
              f'seed0 is the frozen production seed-0 model; A and B are the named profiles.</p>')


def _gaddr_matrix(m, rows: list[str], reading: str = "native") -> str:
    head = ["Scorer"] + rows
    body = []
    for s in m["peers"]["scorers"]:
        g = s[reading] if reading in s and s[reading] else s["native"]
        cells = [f'<span class="mono small">{esc(s["name"])}</span>']
        for r in rows:
            st = g["states"].get(r, "not_measured")
            k = {"pass": "pass", "fail": "fail"}.get(st, "na")
            label = {"pass": "pass", "fail": "fail"}.get(st, "nm")
            cells.append(f'<span class="chip st-{k}" data-tip="{esc(s["name"])} {r}: {esc(st)}">{label}</span>')
        body.append(cells)
    return table(head, body, sortable=False, cls="matrix")


def chart_gaddr_matrix(m) -> str:
    rows = ["A1", "A2", "A3", "A4", "A5", "A6", "A7r", "C1", "C2", "C3", "C4", "C5", "C6"]
    return (f'<h3>G-ADDR states per scorer (native reading)</h3>'
            f'<p class="small">Instrument: {esc(m["peers"]["instrument"])}. A1–A6 are report-only mentor value pins; '
            f'A7r and C1–C6 gate. “nm” = not measured. Source {src_cite(m["peers"]["path"])}.</p>'
            + _gaddr_matrix(m, rows))


def chart_gaddr_c(m) -> str:
    return (f'<h3>C1–C6 contract rows per scorer</h3><p class="small">Source {src_cite(m["peers"]["path"])}; '
            f'“nm” = not measured (no registered peer probe for C3/C4).</p>'
            + _gaddr_matrix(m, ["C1", "C2", "C3", "C4", "C5", "C6"]))


def _heat_class(v: float) -> str:
    return "h" + str(min(5, max(0, int(v * 5 + 0.0001)))) if v is not None else "h0"


def chart_a7r(m) -> str:
    codecs = ["avif-rav1e", "avif-svt", "jpeg", "jxl", "webp"]
    mentor = next(s for s in m["peers"]["scorers"] if s["name"] == "peer_ssim2_registered")["native"]["a7r"]
    body = []
    for s in m["peers"]["scorers"]:
        a = s["native"]["a7r"]
        cells = [f'<span class="mono small">{esc(s["name"])}</span>', chip({"pass": "pass", "fail": "fail"}.get(s["native"]["states"]["A7r"], "na"))]
        for c in codecs:
            v = a.get(c)
            below = v is not None and v < mentor[c] - 1e-12
            cls = _heat_class(v) + (" below" if below else "")
            cells.append(f'<td class="heat {cls}" data-v="{v}" data-tip="{esc(s["name"])} {c}\nA7r {v:.3f} (mentor {mentor[c]:.3f})">{v:.2f}</td>')
        body.append(cells)
    head = "".join(f"<th>{h}</th>" for h in ["Scorer", "A7r"] + codecs)
    rows_html = "".join("<tr>" + "".join(c if c.startswith("<td") else f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
    return (f'<h3>A7r floor representability per codec</h3>'
            f'<p class="small">Share of a codec\'s ladders whose lowest settings stay distinguishable. Darker = higher. '
            f'A red outline marks a value below the mentor (peer_ssim2_registered) on the same cells. Source {src_cite(m["peers"]["path"])}.</p>'
            f'<div class="tablewrap"><table class="sortable"><thead><tr>{head}</tr></thead><tbody>{rows_html}</tbody></table></div>')


def chart_gaddr_bars(m) -> str:
    items = [{"label": s["name"], "value": s["native"]["strict_backwards_pooled"],
              "tip": f'{s["name"]}\nstrict backwards (pooled) {s["native"]["strict_backwards_pooled"]}'} for s in m["peers"]["scorers"]]
    return (f'<h3>Strict-backwards rate on the ladder instrument (pooled)</h3>'
            + svg.hbars(items, xmin=0, label_w=230, title="strict backwards pooled")
            + f'<p class="small muted">Lower is better. Recorded per scorer in {src_cite(m["peers"]["path"])} (gaddr.native.strict_backwards_pooled).</p>')


def chart_steerfix(m) -> str:
    rows = m["steerfix"]["rows"]
    items = [{"label": r["case"], "value": r["m2"], "state": "fail" if r["m2"] < 0.99 else "pass",
              "tip": f'{r["case"]} ({r["verdict"]})\nM2 {r["m2"]:.4f}\n{r["reason"]}'} for r in rows]
    items3 = [{"label": r["case"], "value": r["m3f"], "state": "fail" if r["m3f"] < 0.70 else "pass",
               "tip": f'{r["case"]} ({r["verdict"]})\nM3f {r["m3f"]:.4f}'} for r in rows]
    cnt = Counter(r["verdict"] for r in rows)
    return (f'<h3>The seven failing G-STEER cases, with the pre-floor replay diagnostic</h3>'
            f'<p class="small">Verdicts: ' + ", ".join(f"{v} {n}" for v, n in sorted(cnt.items())) +
            f'. These are diagnostic values with the spline floor removed, not served scores. Source {src_cite(m["steerfix"]["path"], m["steerfix"]["line"])}.</p>'
            f'<div class="two"><div><h3>M2 (bar 0.99)</h3>{svg.dots_ci(items, thresholds=[(0.99, "M2 ≥ .99")], label_w=120, title="M2")}</div>'
            f'<div><h3>M3f (bar 0.70)</h3>{svg.dots_ci(items3, thresholds=[(0.70, "M3f ≥ .70")], label_w=120, title="M3f", xmin=0.6, xmax=1.0)}</div></div>'
            + svg.legend([("var(--good)", "meets the bar"), ("var(--crit)", "below the bar")]))


def chart_speedq3(m) -> str:
    out = ['<h3>SPEEDQ3: Rev5 vs Rev4 cells by tier (pointwise paired 95% CIs)</h3>']
    for r in m["speedq3"]["rows"]:
        out.append(f'<div class="small"><strong>{esc(r["tier"])}</strong> — faster {r["faster"]}, slower {r["slower"]}, '
                   f'inconclusive {r["inconclusive"]}</div>')
        out.append(svg.status_strip({"pass": r["faster"], "fail": r["slower"], "na": r["inconclusive"]},
                                    title=f'{r["tier"]} speed cells'))
    out.append(svg.legend([("var(--good)", "faster"), ("var(--crit)", "slower"), ("var(--na)", "inconclusive")]))
    out.append(f'<p class="small muted">Source {src_cite(m["speedq3"]["path"], m["speedq3"]["line"])}.</p>')
    return "".join(out)


def chart_integrity(m) -> str:
    ig = m["integrity"]
    if not ig:
        return ('<p class="muted">The ZCTH v4 TRAIN-refit gate record (SHIPPATH6 <code>GATES.json</code>) lives outside the '
                'repository and was not supplied to this build. Not shown.</p>')
    rows = [[esc(k.replace("_", " ")), chip("pass" if v else "fail")] for k, v in ig["gates"].items()]
    s = ig["summary"]
    rates = []
    for k in ("detection", "head_fp", "honest_score_lowered", "composed_below_q20", "base_below_q20"):
        if isinstance(s.get(k), dict):
            rates.append({"label": k.replace("_", " "), "value": s[k]["rate"],
                          "tip": f'{k}\n{s[k]["count"]} of {s[k]["n"]} = {s[k]["rate"]:.4f}'})
    return (f'<h3>ZCTH v4 TRAIN refit gates (not EVAL qualification)</h3><p class="small">Scope: {esc(ig["scope"])}. '
            f'Source <span class="mono">{esc(ig["path"])}</span> (outside the repository).</p>'
            + table(["Gate", "State"], rows, sortable=False)
            + svg.hbars(rates, xmin=0, xmax=1, label_w=180, thresholds=[(0.95, "detection ≥ .95"), (0.01, "≤ 1%")],
                        title="integrity rates"))


def chart_hdr_e27(m) -> str:
    e27 = m["results"].get("E27")
    if not e27:
        return ""
    gc = e27.get("gate_checks") or {}
    checks = sorted({k for v in gc.values() for k in v})
    head = ["Check"] + list(gc)
    body = [[esc(c.replace("_", " "))] + [chip("pass" if gc[a].get(c) else "fail") for a in gc] for c in checks]
    return (f'<h3>E27 HDR teacher arms: registered checks</h3><p class="small">Neither arm passes; adopted: '
            f'{esc(e27.get("adopted"))}. Source {src_cite(e27["path"])}.</p>' + table(head, body, sortable=False))


def chart_v40_hdr(m) -> str:
    v = m["v40_hdr"]
    items = [{"label": f'{r["arm"]} {r["endpoint"].replace("_reference", "-ref")} {r["teacher"]}', "value": r["delta"],
              "lo": r["delta"] - 2 * r["se"], "hi": r["delta"] + 2 * r["se"],
              "state": "pass" if r["delta"] - 2 * r["se"] >= 0 else None,
              "tip": f'{r["arm"]} {r["endpoint"]} {r["teacher"]}\nΔ {r["delta"]:+.5f} ± {r["se"]:.5f} SE (n={r["n"]})'} for r in v["rows"]]
    passes = ", ".join(f'{a}: HDR {"pass" if p["hdr"] else "fail"}, SDR {"as good" if p["sdr"] else "not as good"}'
                       for a, p in v["passes"].items())
    return (f'<h3>E29 HDR arms (V40): teacher-agreement deltas vs control, ±2 SE</h3>'
            f'<p class="small">{esc(passes)}. Green: interval wholly above zero. Research selection only; not a human HDR '
            f'qualification. Source {src_cite(v["path"], v["line"])}.</p>'
            + svg.dots_ci(items, label_w=220, title="V40 HDR deltas"))


def chart_borda(m) -> str:
    v = m["v40_hdr"]
    colors = ["var(--s1)", "var(--s2)", "var(--s3)"]
    rows = [(arm, vals, colors[i % 3]) for i, (arm, vals) in enumerate(v["borda"].items())]
    return (f'<h3>Borda panel (report only): signed SROCC per fold × seed cell</h3>'
            f'<p class="small">All {sum(len(r[1]) for r in rows)} recorded cells, one point each. Descriptive; no rule reads it. '
            f'Source {src_cite(v["borda_path"])}.</p>' + svg.strips(rows, title="Borda panel")
            + svg.legend([(c, a) for a, _, c in rows]))


def chart_experiments_overview(m) -> str:
    return ('<p>Human-rank evidence used for research decisions is drawn on the '
            '<a href="{BASE}experiments.html">experiments page</a> (E21 as-good arms against their rule thresholds).</p>')


CHARTS = {"nearid": chart_nearid, "gaddr_matrix": chart_gaddr_matrix, "gaddr_c": chart_gaddr_c, "a7r": chart_a7r,
          "gaddr_bars": chart_gaddr_bars, "steerfix": chart_steerfix, "speedq3": chart_speedq3, "integrity": chart_integrity,
          "hdr_e27": chart_hdr_e27, "experiments_overview": chart_experiments_overview, "v40_hdr": chart_v40_hdr,
          "borda": chart_borda}


def arm_chart(arms: list[dict], mean_floor: float, source_floor: float) -> str:
    """Per-arm signed delta ± 2 SE and per-source deltas against the E21 floors."""
    items = []
    for a in arms:
        if a["signed"] is None:
            continue
        se = a["se"] or 0
        st = "pass" if a.get("as_good") else ("fail" if a.get("as_good") is False else None)
        items.append({"label": f'{a["experiment"]} {a["arm"]}', "value": a["signed"], "lo": a["signed"] - 2 * se,
                      "hi": a["signed"] + 2 * se, "state": st,
                      "tip": f'{a["experiment"]} {a["arm"]}\nsigned Δ {a["signed"]:+.5f} ± {se:.5f} (SE)\n'
                             f'as good: {a.get("as_good")}' + (f'\n{a["verdict"]}: {a["reason"]}' if a.get("verdict") else "")})
    # Axis fixed around the decision region (5× the mean floor either side); arms beyond it are drawn as labelled edge markers.
    span = abs(mean_floor) * 5
    return svg.dots_ci(items, thresholds=[(mean_floor, f"mean ≥ {mean_floor}")], label_w=170, title="signed delta",
                       xmin=-span, xmax=span)


def source_chart(arm: dict, source_floor: float) -> str:
    items = []
    for src, v in arm["per_source"].items():
        se = v.get("se")
        items.append({"label": src, "value": v["delta"], "lo": v["delta"] - 2 * se if se else None,
                      "hi": v["delta"] + 2 * se if se else None,
                      "state": "pass" if v["delta"] >= source_floor else "fail",
                      "tip": f'{src}\nΔ {v["delta"]:+.5f}' + (f' ± {se:.5f} SE, n={v.get("n")}' if se else "")})
    return svg.dots_ci(items, thresholds=[(source_floor, f"each source ≥ {source_floor}")], label_w=110, title="per source")


def seed_chart(arm: dict, mean_floor: float) -> str:
    if not arm["seed_deltas"]:
        return ""
    items = [{"label": f"seed {i}", "value": v, "tip": f"seed {i}\nΔ {v:+.5f}"} for i, v in enumerate(arm["seed_deltas"])]
    return svg.dots_ci(items, thresholds=[(mean_floor, f"{mean_floor}")], label_w=110, title="seed deltas", row=20)


# --------------------------------------------------------------------------- pages

def page_index(m) -> str:
    gates = _gates_by_id(m)
    g = m["gates"]
    st = Counter(x["status"] for x in g["gates"])
    pend = sum(1 for x in g["gates"] if x["rule_pending"])
    bugs = Counter(b["status"] for b in m["bugs"])
    exps = [a for r in m["results"].values() for a in r["arms"]]
    body = [f'<h1>How zensim is evaluated</h1>',
            f'<p class="lede">zensim is a one-number image-quality dial: users pick a target score and codecs hit it. '
            f'This site walks through what the dial must do, what it must not do, the gates that check each, how '
            f'statistics combine into verdicts, and the features the models read. Every value is quoted or read from a '
            f'committed source and cited.</p>',
            f'<div class="headline">{md_inline(g["headline"], g["path"])} '
            f'<span class="small">{src_cite(g["path"], g["headline_line"])}</span></div>',
            '<div class="statrow">',
            f'<div class="stat"><div class="v">{len(g["gates"])}</div><div class="l">release gates · {st.get("blocked", 0)} blocked, '
            f'{st.get("ready", 0)} ready, {st.get("done", 0)} done; {pend} with a rule pending</div></div>',
            f'<div class="stat"><div class="v">{len(m["wanted"])}</div><div class="l">wanted properties</div></div>',
            f'<div class="stat"><div class="v">{bugs.get("open", 0)}</div><div class="l">open Known Bugs of {len(m["bugs"])} recorded</div></div>',
            f'<div class="stat"><div class="v">{sum(1 for a in exps if a.get("as_good"))}/{sum(1 for a in exps if a.get("as_good") is not None)}</div>'
            f'<div class="l">recorded experiment arms judged as good (E21 rule)</div></div>',
            f'<div class="stat"><div class="v">{m["features"]["full_width"]}</div><div class="l">zensim feature slots · '
            f'{len(m["za"]["features"])} zenanalyze features</div></div>',
            '</div>',
            '<h2>The dial contract</h2>',
            '<p>Each card is one property the product needs. Chips show the production gate state from the release '
            'gate map and the latest recorded measurement on the frozen seed-0 model, where one exists.</p><div class="grid">']
    for p in m["wanted"]:
        _, chips = _prop_status(p, gates)
        body.append(f'<div class="card"><h3><a href="property/{p["id"]}.html">{esc(p["title"])}</a></h3>'
                    f'<p>{esc(p["definition"][:180])}{"…" if len(p["definition"]) > 180 else ""}</p>'
                    f'<div class="chips">{"".join(chips)}</div></div>')
    body.append('</div>')
    # gate matrix
    body.append('<h2>Release-gate matrix</h2><p>Rows are the release gates. Columns show the production state, whether an '
                'owner decision or rule is still missing, and which instruments check the same gate: the terminal-read '
                'authorization list, <code>freeze_check --qualify</code>, and the September 8 production contract.</p>')
    body.append(svg.status_strip({k: st.get(k, 0) for k in GATE_STATE_ORDER}, title="release gate states"))
    body.append(svg.legend([("var(--warn)", "blocked"), ("var(--accent)", "ready"), ("var(--good)", "done")]))
    rows = []
    for x in g["gates"]:
        cw = m["crosswalk"][x["id"]]
        rows.append([_gate_link(x), _gate_chip(x),
                     ("✓ " + esc(", ".join(cw["terminal"]))) if cw["terminal"] else '<span class="muted">–</span>',
                     ("✓ " + esc(str(len(cw["freeze"])) + " check" + ("s" if len(cw["freeze"]) > 1 else ""))) if cw["freeze"] else '<span class="muted">–</span>',
                     esc(", ".join(cw["contract"])) if cw["contract"] else '<span class="muted">–</span>'])
    body.append(table(["Gate", "Production state", "Terminal authorization", "freeze_check --qualify", "Sept 8 contract row"], rows,
                      tid="gate-matrix"))
    body.append(f'<p class="small muted">{md_inline(g["qualification_note"] or "", g["path"])}</p>')
    body.append('<h2>Peers through the same gates</h2>' + chart_gaddr_matrix(m))
    body.append('<h2>Sections</h2><div class="grid">')
    for href, t, d in [("gates.html", "Gates and defects", "Every release gate, the defects each catches, and open Known Bugs."),
                       ("evaluation.html", "Evaluation and verdicts", "Statistics, the E21 as-good rule, W2 guards, Borda, selection vs qualification."),
                       ("experiments.html", "Experiments", "Registered experiments, rules and recorded arm outcomes drawn against their thresholds."),
                       ("splits.html", "Data roles", "TRAIN/SELECT/VAL/TEST/TERMINAL, the dataset registry and the exposure ledger."),
                       ("features.html", "zensim features", "feature_defs families, slot layout, formula revisions and named sets."),
                       ("zenanalyze.html", "zenanalyze features", "The analyzer catalogue and which picker artifacts pin which features.")]:
        body.append(f'<div class="card"><h3><a href="{href}">{esc(t)}</a></h3><p>{esc(d)}</p></div>')
    body.append('</div>')
    return _render(m, title="Overview", current="index.html", body="\n".join(body))


def page_properties(m) -> dict:
    gates = _gates_by_id(m)
    out = {}
    body = ['<h1>Properties we want from the dial</h1>',
            '<p class="lede">One score must behave like a dial for every codec and image. These are the properties that '
            'make it one. Open a property for its exact rule, owning code and current measured state.</p>']
    rows = []
    for p in m["wanted"]:
        _, chips = _prop_status(p, gates)
        rows.append([f'<a href="property/{p["id"]}.html">{esc(p["title"])}</a>', f'<span class="small">{esc(p["definition"])}</span>',
                     f'<div class="chips">{"".join(chips)}</div>'])
    body.append(table(["Property", "Definition", "State"], rows, tid="props"))
    out["properties.html"] = _render(m, title="Wanted properties", current="properties.html", body="\n".join(body))
    for p in m["wanted"]:
        out[f'property/{p["id"]}.html'] = _render(m, title=p["title"], current="properties.html", depth=1,
                                                  crumbs=[("Wanted properties", "properties.html"), (p["title"], None)],
                                                  body=_property_body(m, p, gates))
    return out


def _property_body(m, p, gates) -> str:
    _, chips = _prop_status(p, gates)
    b = [f'<h1>{esc(p["title"])}</h1><div class="chips">{"".join(chips)}</div>',
         f'<div class="panel"><dl class="kv"><dt>Definition</dt><dd>{esc(p["definition"])}</dd>'
         f'<dt>Why it matters</dt><dd>{esc(p["why"])}</dd>'
         f'<dt>Gates</dt><dd>{", ".join(_gate_link(gates[g], 1) + " " + _gate_chip(gates[g]) for g in p.get("gates", []))}</dd>'
         f'<dt>Owning code</dt><dd>{"<br>".join(_quote(o) for o in p["owners"])}</dd></dl></div>']
    if p.get("rule_pending_note"):
        b.append(f'<p>{chip("pending")} {esc(p["rule_pending_note"])}</p>')
    b.append('<h2>Exact rule</h2>' + "".join(_quote(q) for q in p["rules"]))
    b.append('<h2>Current measured state</h2>'
             '<p class="small muted">Chips beside a quote are derived from its text by a fixed pattern recorded in the '
             'catalogue; the quote itself is the evidence.</p>' + "".join(_quote(q, True) for q in p["state"]))
    charts = [CHARTS[c](m).replace("{BASE}", "../") for c in p.get("charts", [])]
    if charts:
        b.append('<h2>Recorded values</h2>' + "".join(f'<div class="panel">{c}</div>' for c in charts))
    return "\n".join(b)


def page_gates(m) -> dict:
    gates = _gates_by_id(m)
    out = {}
    g = m["gates"]
    body = ['<h1>Gates, and the properties we do not want</h1>',
            f'<p class="lede">The release gate map is the controlling list of what a production model must pass. '
            f'Each row below links to its full rule. The second half lists the defects the gates exist to catch, '
            f'with recorded instances.</p>',
            f'<div class="headline">{md_inline(g["headline"], g["path"])} <span class="small">{src_cite(g["path"], g["headline_line"])}</span></div>',
            '<h2 id="release">Release gates</h2>',
            filterbar("release-gates", "Filter gates", [(1, "State", ["blocked", "ready", "done"])])]
    rows = []
    for x in g["gates"]:
        comps = " ".join(chip(k, f"{k} ×{len(v)}") for k, v in x["components"].items() if v and k != x["status"])
        rows.append([_gate_link(x), esc(x["status"]), _gate_chip(x) + (" " + comps if comps else ""),
                     f'<span class="small">{md_inline(_short(x["pass_rule"], 260), g["path"])}</span>'])
    body.append(table(["Gate", "State", "Production status", "Pass rule (abridged; open the gate for the full text)"], rows,
                      tid="release-gates"))
    body.append('<h2 id="unwanted">Properties we do not want</h2>')
    for p in m["unwanted"]:
        hits = p["bug_hits"]
        hit_c = Counter(b["status"] for b in hits)
        catches = ", ".join(_gate_link(gates[c]) + " " + _gate_chip(gates[c]) for c in p["catches"])
        links = " ".join(f'<a href="{l}">more</a>' for l in p.get("links", []))
        bug_rows = [[esc(b["date"]), chip(b["status"]), f'<span class="small">{md_inline(b["title"], "CLAUDE.md")}</span>',
                     src_cite("CLAUDE.md", b["line"])] for b in hits]
        body.append(f'<div class="panel" id="{p["id"]}"><h3>{esc(p["title"])}</h3><p>{esc(p["definition"])} {links}</p>'
                    f'<dl class="kv"><dt>Caught by</dt><dd>{catches}</dd>'
                    f'<dt>Known Bugs naming it</dt><dd>{len(hits)} ({hit_c.get("open", 0)} open, {hit_c.get("fixed", 0)} fixed)</dd></dl>'
                    + "".join(_quote(q, True) for q in p["evidence"])
                    + (f'<details><summary>{len(hits)} matching Known Bugs entries</summary>'
                       + table(["Date", "State", "Entry", "Source"], bug_rows) + '</details>' if hits else "")
                    + '</div>')
    body.append('<h2 id="bugs">All Known Bugs</h2><p class="small">From the zensim <code>CLAUDE.md</code> Known Bugs section. '
                'State is read from the entry heading: OPEN, or FIXED/RESOLVED/AMENDED/SUPERSEDED, or NOT A BUG (info).</p>')
    body.append(filterbar("bugs", "Filter entries", [(1, "State", ["open", "fixed", "info"])]))
    body.append(table(["Date", "State", "Entry", "Source"],
                      [[esc(b["date"]), chip(b["status"]),
                        f'<span class="small">{md_inline(b["title"], "CLAUDE.md")}</span>', src_cite("CLAUDE.md", b["line"])]
                       for b in m["bugs"]], tid="bugs"))
    out["gates.html"] = _render(m, title="Gates and defects", current="gates.html", body="\n".join(body))
    for x in g["gates"]:
        out[f'gate/{x["id"]}.html'] = _render(m, title=x["name"], current="gates.html", depth=1,
                                              crumbs=[("Gates", "gates.html"), (x["name"], None)], body=_gate_body(m, x))
    return out


def _short(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s)
    if len(s) <= n:
        return s
    cut = s[:n]
    # never end inside a code span or bold run
    if cut.count("`") % 2:
        cut = cut[:cut.rfind("`")]
    if cut.count("**") % 2:
        cut = cut[:cut.rfind("**")]
    return cut.rstrip() + " …"


def _gate_body(m, x) -> str:
    path = m["gates"]["path"]
    cw = m["crosswalk"][x["id"]]
    props = [p for p in m["wanted"] if x["id"] in p.get("gates", [])]
    unw = [p for p in m["unwanted"] if x["id"] in p.get("catches", [])]
    b = [f'<h1>{esc(x["name"])}</h1><div class="chips">{_gate_chip(x)}</div>',
         f'<p class="small">Row {src_cite(path, x["line"])} of the release gate map.</p>',
         '<div class="panel"><dl class="kv">',
         f'<dt>Owner command</dt><dd>{md_inline(x["owner_command"], path)}</dd>',
         f'<dt>Required inputs and pass rule</dt><dd>{md_inline(x["pass_rule"], path)}</dd>',
         f'<dt>Status for production</dt><dd>{md_inline(x["status_text"], path)}</dd>',
         f'<dt>Protected reads</dt><dd>{md_inline(x["protected_reads"], path)}</dd>',
         '</dl></div>', '<h2>State, as classified</h2><dl class="kv">']
    for k, v in x["components"].items():
        if v:
            b.append(f'<dt>{chip(k)}</dt><dd>{"<br>".join(md_inline(c, path) for c in v)}</dd>')
    b.append(f'<dt>Rule pending</dt><dd>{"yes — " + "; ".join(esc(o) for o in x["owner_decisions"]) if x["owner_decisions"] else ("yes" if x["rule_pending"] else "no")}</dd></dl>')
    b.append('<p class="small muted">Classification rule: a bold status clause containing “blocked” is blocked, one '
             'starting “ready” is ready, one starting “done” is done; an “Owner decision:” clause or a stated absence of '
             'a numerical rule marks the rule as pending. The text above is the source.</p>')
    b.append('<h2>Same gate in other instruments</h2><dl class="kv">'
             f'<dt>Terminal-read authorization (<code>GATES</code>)</dt><dd>{esc(", ".join(cw["terminal"]) or "—")}</dd>'
             f'<dt><code>freeze_check --qualify</code></dt><dd>{esc(", ".join(cw["freeze"]) or "—")}</dd>'
             f'<dt>September 8 contract</dt><dd>{esc(", ".join(cw["contract"]) or "—")}</dd></dl>')
    if props or unw:
        b.append('<h2>Related properties</h2><ul>' + "".join(f'<li><a href="../property/{p["id"]}.html">{esc(p["title"])}</a></li>' for p in props)
                 + "".join(f'<li><a href="../gates.html#{p["id"]}">{esc(p["title"])}</a> (unwanted)</li>' for p in unw) + '</ul>')
    return "\n".join(b)


def page_evaluation(m) -> str:
    ev = m["evaluation"]
    fz = m["freeze"]
    sc = m["scorecard"]
    floors_rows = [[f'<span class="mono">{esc(f["name"])}</span>', (f'{f["value"]:g}', f["value"]), f'<span class="small">{esc(f["comment"])}</span>']
                   for f in fz["floors"]]
    b = ['<h1>How evaluation turns numbers into verdicts</h1>',
         '<p class="lede">A verdict is never one statistic. It is a fixed chain: data roles decide what may be read; a '
         'registered statistic is computed per source and seed; guards combine those into an as-good decision; research '
         'selection picks among candidates; product qualification checks one frozen candidate against every gate; and '
         'only then may a terminal set be read, once.</p>',
         '<div class="panel"><ol>'
         '<li><strong>Data roles.</strong> TRAIN fits, transforms and checkpoint choices; SELECT and VAL inform research '
         'choices; frozen candidates may be assessed on EVAL or published TEST with exposure recorded; TERMINAL and T0 sets '
         'are read once or never. <a href="splits.html">Data roles</a>.</li>'
         '<li><strong>Per-source statistics.</strong> Signed SROCC per (source, seed) through the Rust <code>panel</code> owner; '
         'companions: KROCC, logistic PLCC, raw Pearson, PWRC, outlier ratio, Z-RMSE, within-reference and per-type means.</li>'
         '<li><strong>Pooling.</strong> Sources are averaged with equal weight inside each seed; the ten seed units give the '
         'mean and SE (sd/√10). Arms are always paired with a control on the same seeds and folds.</li>'
         '<li><strong>Guards.</strong> The E21 as-good rule: mean Δ ≥ −0.002, every source Δ ≥ −0.005, and the worst-three '
         'distortion-type mean (W2) above −2 SE. Adoption rules add a significance test where registered.</li>'
         '<li><strong>Selection vs qualification.</strong> <code>freeze_check --select</code> compares many research candidates; '
         '<code>--qualify</code> checks one frozen composition and is not satisfied by a selection.</li>'
         '<li><strong>Terminal read.</strong> Allowed once, after every gate passes, with a committed pre-read pin.</li></ol></div>',
         '<h2 id="stats">Statistics owners</h2>',
         '<p>Every statistic comes from one owner. The <code>panel</code> binary documents which functions it wraps:</p>',
         _quote(ev["panel_table"]),
         '<p class="small">Note: <code>zensim-validate/src/panel.rs</code> is now a re-export of <code>zenstats::panel</code> '
         '(zenmetrics); the file:line column in the excerpt above predates that move.</p>',
         '<h2 id="e21">The E21 as-good rule</h2>',
         '<p>Registered with E21 and reused unchanged by E24, E25, E27, E28 and the V40 studies. Its thresholds are '
         'constants in the registering script:</p>', _quote(ev["e21_floors"]),
         '<p>The shared implementation used by E24/E25:</p>', _quote(ev["as_good"]),
         '<p>The V40 form for E29/E31/E32 (ten equal-four-source seed units, paired two-source W2):</p>', _quote(ev["sdr_decision"]),
         '<p>The worst-case companions (W1 reference p10, W2 type minimum and worst-three mean, W3 Z-RMSE and outlier ratio, '
         'W4 negative share):</p>', _quote(ev["worst_case"]),
         '<p>E33 restates the rule for its arms:</p>', _quote(ev["e33_rule"]),
         '<h2 id="adoption">Adoption rules</h2>',
         '<p>As-good keeps an arm eligible; adoption needs more. E28 also required a pooled recipe signal on both KROCC and '
         'raw Pearson (Δ > 2 SE):</p>', _quote(ev["e28_decision"]),
         '<h2 id="borda">Borda / JOD consensus (E29)</h2>',
         '<p>E29\'s consensus target averages the two HDR teachers\' ranks and rescales to [0, 1]:</p>', _quote(ev["consensus"]),
         f'<div class="panel">{chart_borda(m)}</div>',
         '<h2 id="selection">Research selection versus product qualification</h2>', _quote(ev["selection_vs_qual"]),
         f'<p><code>freeze_check --qualify</code> emits {len(fz["gates"])} checks ({src_cite(fz["path"], fz["line"])}): '
         + ", ".join(f"<code>{esc(x)}</code>" for x in fz["gates"]) + '.</p>',
         f'<p>Terminal authorization requires {len(m["terminal"]["gates"])} gates to be recorded as passing '
         f'({src_cite(m["terminal"]["path"], m["terminal"]["line"])}): ' + ", ".join(f"<code>{esc(x)}</code>" for x in m["terminal"]["gates"]) + '.</p>',
         '<h3>Research-selection floors (F1–F8)</h3>', '<p>Registered floors used by <code>--select</code>; they are not product bars.</p>',
         table(["Constant", "Value", "Registration note"], floors_rows, numeric_cols=(1,)),
         '<h2 id="contract">The September 8 production contract</h2>',
         f'<p class="small">{src_cite(sc["path"], sc["contract_line"])}</p>',
         table(["Requirement", "Release bar / measurement"],
               [[esc(r["requirement"]), f'<span class="small">{md_inline(r["bar"], sc["path"])}</span>'] for r in sc["contract"]], sortable=False),
         _quote({"kind": "quote", "path": sc["path"], "line": sc["disposition_line"], "text": sc["disposition"], "status": "info"}),
         '<h2 id="exam">The five-gate exam (July scorecard)</h2>',
         table(["Gate", "Question", "Instrument", "Pass bar (SDR)"],
               [[esc(r["gate"]), md_inline(r["question"], sc["path"]), f'<span class="small">{md_inline(r["instrument"], sc["path"])}</span>',
                 md_inline(r["bar"], sc["path"])] for r in sc["exam"]], sortable=False)]
    return _render(m, title="Evaluation and verdicts", current="evaluation.html", body="\n".join(b))


def _e21_floors(m) -> tuple[float, float]:
    t = m["evaluation"]["e21_floors"]["text"]
    a, b = re.findall(r"-?\d+\.\d+", t)[:2]
    return float(a), float(b)


def page_experiments(m) -> dict:
    out = {}
    mean_floor, source_floor = _e21_floors(m)
    res = m["results"]
    all_arms = [a for k in sorted(res, key=lambda k: int(re.sub(r"\D", "", k) or 0)) for a in res[k]["arms"]]
    b = ['<h1>Registered experiments</h1>',
         '<p class="lede">Two numbering schemes exist: the Rev4 program (E1–E6, September 23) and the featpot design log '
         '(E1–E33). Each experiment registers its rule before any cell runs. Outcomes below are read from the recorded '
         'decision and result files.</p>',
         f'<h2>Recorded arms against the E21 rule</h2><p>Signed Δ versus the paired control, ±2 SE whiskers, with the '
         f'mean floor ({mean_floor}) dashed. Green: judged as good; red: not as good; blue: report-only (no rule).</p>',
         f'<div class="panel">{arm_chart(all_arms, mean_floor, source_floor)}</div>']
    rows = []
    for e in sorted(m["experiments"], key=lambda e: (e["scheme"], int(re.sub(r"\D", "", e["id"]) or 0), e["id"])):
        r = res.get(e["id"]) if e["scheme"] == "featpot" else None
        outcome = '<span class="muted">no recorded result file</span>'
        if r:
            outcome = " ".join(chip("pass" if a.get("as_good") else ("fail" if a.get("as_good") is False else "info"),
                                    f'{a["arm"]}: ' + ("as good" if a.get("as_good") else ("not as good" if a.get("as_good") is False else "report")))
                               for a in r["arms"])
        rows.append([esc(e["label"]), esc(e["scheme"]), esc(e.get("date") or "—"),
                     f'<a href="experiment/{esc(e["scheme"])}-{esc(e["id"])}.html">{esc(_short(e["title"], 140))}</a>', outcome])
        out[f'experiment/{e["scheme"]}-{e["id"]}.html'] = _render(
            m, title=f'{e["label"]}', current="experiments.html", depth=1,
            crumbs=[("Experiments", "experiments.html"), (e["label"], None)], body=_experiment_body(m, e, r, mean_floor, source_floor))
    b.append('<h2>All registered experiments</h2>' + filterbar("exps", "Filter experiments", [(1, "Scheme", ["rev4", "featpot"])]))
    b.append(table(["ID", "Scheme", "Registered", "Question", "Recorded outcome"], rows, tid="exps"))
    out["experiments.html"] = _render(m, title="Experiments", current="experiments.html", body="\n".join(b))
    return out


def _experiment_body(m, e, r, mean_floor, source_floor) -> str:
    b = [f'<h1>{esc(e["label"])} — {esc(_short(e["title"], 200))}</h1>',
         f'<p class="small">Registration: {src_cite(e["registration"], e.get("line"))}'
         + (f' · status: {esc(e["status_text"])}' if e.get("status_text") else "") + '</p>']
    if e.get("docstring"):
        b.append(f'<div class="panel">{md_block(e["docstring"][:3000], e["registration"])}</div>')
    if e.get("rule_text"):
        b.append(f'<h2>Decision rule</h2><div class="panel">{md_block(e["rule_text"][:4000], e["registration"])}'
                 f'<div class="small">{src_cite(e["registration"], e.get("rule_line") or e.get("line"))}</div></div>')
    if r:
        adopted = r.get("adopted")
        b.append(f'<h2>Recorded outcome</h2><p class="small">Source {src_cite(r["path"])}'
                 + (f' · adopted: <strong>{esc(adopted if adopted is not None else "none")}</strong>'
                    + (f' (from {src_cite(r["adopted_from"])})' if r.get("adopted_from") else "") if "adopted" in r else "") + '</p>')
        if e["id"] == "E29":
            b.append(f'<div class="panel">{chart_v40_hdr(m)}</div><div class="panel">{chart_borda(m)}</div>')
        b.append(f'<div class="panel">{arm_chart(r["arms"], mean_floor, source_floor)}</div>')
        for a in r["arms"]:
            facts = [("signed Δ", f'{a["signed"]:+.6f}' if a["signed"] is not None else "—"),
                     ("SE", f'{a["se"]:.6f}' if a["se"] is not None else "—"),
                     ("W2", f'{a["w2"]:+.6f} ± {a["w2_se"]:.6f}' if a["w2"] is not None and a["w2_se"] is not None else "—"),
                     ("as good", "—" if a.get("as_good") is None else str(a["as_good"])),
                     ("passes", "—" if a.get("passes") is None else str(a["passes"]))]
            if a.get("verdict"):
                facts.append(("verdict", f'{a["verdict"]} — {a["reason"]}'))
            b.append(f'<div class="panel"><h3>{esc(a["arm"])}</h3>'
                     + (f'<p class="mono small">{esc(a["spec"])}</p>' if a.get("spec") else "")
                     + '<dl class="kv">' + "".join(f'<dt>{esc(k)}</dt><dd>{esc(v)}</dd>' for k, v in facts) + '</dl>'
                     + f'<h3>Per source</h3>{source_chart(a, source_floor)}'
                     + (f'<h3>Seed deltas</h3>{seed_chart(a, mean_floor)}' if a["seed_deltas"] else "")
                     + '</div>')
    return "\n".join(b)


def page_splits(m) -> str:
    sp = m["splits"]
    path = sp["path"]
    tiers = Counter()
    for d in sp["datasets"]:
        t = re.sub(r"\*\*|\(.*", "", d["tier"]).strip()
        tiers[t[:40]] += 1
    by_month = Counter(l["date"][:7] for l in sp["ledger"])
    days = sorted({l["date"] for l in sp["ledger"]})
    day_c = Counter(l["date"] for l in sp["ledger"])
    b = ['<h1>Data roles and exposure</h1>',
         '<p class="lede">What each dataset may be used for, and every recorded read of protected data. The rules are '
         'quoted from <code>docs/DATA_SPLITS.md</code>; there is no machine-readable split registry, so this page reads '
         'the document\'s own registry table and ledger headings.</p>',
         _quote(m["evaluation"]["split_clarification"]),
         '<h2>Policy sections</h2><div class="toc">'
         + "".join(f'<a href="{esc(blob_url(path, p["line"]))}">'
                   f'{"&nbsp;&nbsp;" * (p["level"] - 2)}{md_inline(p["title"])}</a>' for p in sp["policy"]) + '</div>',
         f'<h2 id="registry">Per-dataset registry ({len(sp["datasets"])} datasets)</h2>',
         f'<p class="small">{src_cite(path, sp["registry_line"])}. Tier: T0 protected holdout, T1 integrity guard, T2 training, '
         'T3 instrument (DATA_SPLITS §1).</p>',
         filterbar("datasets", "Filter datasets"),
         table(["Dataset", "Tier", "Our split", "Leakage status"],
               [[md_inline(d["full"], path), f'<span class="small">{md_inline(d["tier"], path)}</span>',
                 f'<span class="small">{md_inline(_short(d["split"], 420), path)}</span>',
                 f'<span class="small">{md_inline(_short(d["leakage"], 240), path)}</span>'] for d in sp["datasets"]], tid="datasets"),
         f'<h2 id="ledger">Exposure ledger timeline ({len(sp["ledger"])} entries)</h2>',
         '<p>Each ledger entry is a dated heading in DATA_SPLITS.md. Bars count entries per day.</p>',
         svg.hbars([{"label": d, "value": day_c[d], "tip": f"{d}\n{day_c[d]} entries"} for d in days], xmin=0, label_w=110,
                   row=16, title="ledger entries per day"),
         filterbar("ledger", "Filter ledger", [(1, "Kind", sorted({l["kind"] for l in sp["ledger"]}))]),
         table(["Date", "Kind", "Entry"],
               [[esc(l["date"]), esc(l["kind"]),
                 f'<a href="{esc(blob_url(path, l["line"]))}">{md_inline(l["title"])}</a>']
                for l in sp["ledger"]], tid="ledger")]
    return _render(m, title="Data roles", current="splits.html", body="\n".join(b))


def page_features(m) -> dict:
    fd = m["features"]
    fs = m["featuresets"]
    ns = m["named_sets"]
    out = {}
    slots = fd["slots"]
    form_c = Counter(s["form"] for s in slots)
    by = set(ns["by_v2fy"])
    tips = {s["id"]: f'#{s["id"]} {s["name"]}\n{s["statistic"]}, {s["form"]}' for s in slots}
    fam_rows = []
    for bl in fd["blocks"]:
        fc = Counter(s["form"] for s in slots if bl["lo"] <= s["id"] <= bl["hi"])
        nby = sum(1 for i in range(bl["lo"], bl["hi"] + 1) if i in by)
        fam_rows.append([f'<a href="feature/{esc(bl["family"])}.html">{esc(bl["family"])}</a>',
                         (f'{bl["lo"]}–{bl["hi"]}', bl["lo"]), (str(bl["width"]), bl["width"]), (str(len(bl["signals"])), len(bl["signals"])),
                         esc(bl["replication"]), esc(", ".join(f"{k} {v}" for k, v in fc.most_common())),
                         (str(nby), nby), src_cite(fd["path"], bl["line"])])
        out[f'feature/{bl["family"]}.html'] = _render(m, title=f'{bl["family"]} features', current="features.html", depth=1,
                                                      crumbs=[("zensim features", "features.html"), (bl["family"], None)],
                                                      body=_family_body(m, bl, by))
    rev_rows = [[esc(r["name"]), f'<span class="small">{md_inline(r["doc"])}</span>',
                 esc(", ".join(r.get("eras_added") or []) or "—")] for r in fd["revisions"]]
    defect_rows = [[esc(v["id"]), (str(sum(1 for s in slots if s["defect"] == k)), sum(1 for s in slots if s["defect"] == k)),
                    f'<span class="small">{esc(_short(v["note"], 300))}</span>'] for k, v in fd["defects"].items()]
    # family strip drawn to scale over the full width
    fam_strip = _family_strip(fd)
    b = ['<h1>zensim features and feature sets</h1>',
         f'<p class="lede">The feature registry <code>feature_defs</code> declares {len(fd["blocks"])} families, '
         f'{sum(len(bl["signals"]) for bl in fd["blocks"])} signals and {fd["full_width"]} slots at {fd["n_scales"]} scales. '
         f'A slot is one signal at one scale and channel; a model reads a named subset of slots.</p>',
         '<div class="statrow">' + "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{esc(k)} slots</div></div>'
                                           for k, v in form_c.most_common()) + '</div>',
         '<p class="small">Form: <strong>difference</strong> slots are exactly 0 on identical inputs; <strong>reference_only</strong> '
         'slots depend on the reference alone and are nonzero on identity; <strong>undeclared</strong> slots have no registered form. '
         f'Source {src_cite(fd["path"])}; the reader evaluates the Rust constructors and is cross-checked against the registry JSON '
         'and the E33 fx1 table.</p>',
         '<h2>Layout, drawn to scale</h2>', fam_strip,
         '<h2 id="families">Families</h2>',
         table(["Family", "Slot IDs", "Width", "Signals", "Replication", "Forms", "In by_v2fy", "Source"], fam_rows,
               tid="families", numeric_cols=(1, 2, 3, 6)),
         '<h2 id="named">Named sets</h2>']
    for s in ns["sets"]:
        on = set(s["ids"])
        b.append(f'<div class="panel"><h3>{esc(s["name"])}</h3><p class="small">{esc(s["note"])} '
                 f'{len(on)} IDs: <span class="mono">{esc(_compress(s["ids"]))}</span>. Source {src_cite(s["source"])}.</p>'
                 + svg.id_layout(on, 720, cols=72, title=s["name"], tips=tips) + '</div>')
    b.append(f'<p class="small">Maps show slots 0–719 (basic through v2), where every named set lives; highlighted cells are members. '
             f'E33 arm C also adds 410 derived products d × fragility(cell(d)), not slots.</p>')
    b.append('<h2 id="revisions">Formula revisions</h2>'
             '<p>A formula revision fixes the arithmetic every slot is computed with. Tables and models from different revisions '
             'never mix.</p>' + table(["Revision", "Meaning", "Eras (cumulative)"], rev_rows, sortable=False))
    b.append('<h2 id="defects">Registered defects</h2>' + table(["Defect", "Slots carrying it", "Note"], defect_rows, numeric_cols=(1,)))
    set_rows = []
    for s in fs["sets"]:
        set_rows.append([f'<span class="mono small">{esc(s["id"])}</span>', esc(s["role"]), esc(s["kind"]),
                         (str(s["n_slots"]) if s["n_slots"] is not None else "—", s["n_slots"] or 0), esc(s["era"]),
                         chip("pass", "hash ok") if s["hash_ok"] else (chip("fail", "hash mismatch") if s["hash_ok"] is False else chip("na", "no slots")),
                         f'<span class="small">{md_inline(_short(s.get("note") or s.get("legacy_name") or "", 160))}</span>'])
    b.append(f'<h2 id="registry">Feature-set registry ({len(fs["sets"])} sets)</h2>'
             f'<p class="small">An ID is <code>&lt;compute&gt;@w&lt;layout&gt;/&lt;era&gt;#&lt;slots-hash8&gt;</code>. “hash ok” means the recorded '
             f'hash equals FNV-1a of the recorded slots, computed here the same way as <code>zensim::feature_set_id::slots_hash8</code>. '
             f'Source {src_cite(fs["path"])}.</p>' + filterbar("fsets", "Filter sets", [(1, "Role", ["producer", "consumer"])])
             + table(["Set ID", "Role", "Kind", "Slots", "Era", "Hash", "Note"], set_rows, tid="fsets", numeric_cols=(3,)))
    alias_rows = [[esc(k), "<br>".join(f'<span class="mono small">{esc(i)}</span>' for i in v.get("ids", [])),
                   f'<span class="small">{esc(v.get("note", ""))}</span>'] for k, v in fs["aliases"].items()]
    b.append('<h2 id="aliases">Legacy aliases</h2><p>Counts such as “944” named several different sets; they resolve only through '
             'this table.</p>' + table(["Alias", "Resolves to", "Note"], alias_rows, sortable=False))
    out["features.html"] = _render(m, title="zensim features", current="features.html", body="\n".join(b))
    return out


def _compress(ids) -> str:
    from .featuresets_src import compress
    return compress(ids)


def _family_strip(fd) -> str:
    total = fd["full_width"]
    w = 1000
    out = [f'<svg class="chart" viewBox="0 0 {w} 70" role="img" aria-label="feature layout">']
    cols = ["var(--s1)", "var(--s3)"]
    for k, bl in enumerate(fd["blocks"]):
        x0 = bl["lo"] / total * w
        x1 = (bl["hi"] + 1) / total * w
        out.append(f'<rect x="{x0:.1f}" y="8" width="{max(x1 - x0 - 1, 0.8):.1f}" height="28" rx="2" fill="{cols[k % 2]}" '
                   f'data-tip="{esc(bl["family"])}\nslots {bl["lo"]}–{bl["hi"]} ({bl["width"]})\n{bl["replication"]}"/>')
        if x1 - x0 > 34:
            out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="52" text-anchor="middle">{esc(bl["family"])}</text>')
    for t in range(0, total + 1, 200):
        out.append(f'<text x="{t / total * w:.1f}" y="66" text-anchor="middle" style="fill:var(--muted)">{t}</text>')
    out.append('</svg>')
    return "".join(out) + '<p class="small muted">Alternating colours separate adjacent families; hover a block for its range. Narrow families are labelled in the table.</p>'


def _family_body(m, bl, by) -> str:
    fd = m["features"]
    rows = [[(str(s["local"]), s["local"]), f'<span class="mono">{esc(s["name"])}</span>', esc(s["statistic"]), esc(s["form"]),
             esc(s["direction"]), esc(s["cost"]), esc(s["kernel"]), esc((fd["defects"].get(s["defect"]) or {}).get("id", "") if s["defect"] else ""),
             esc(s["ctor"])] for s in bl["signals"]]
    slots = [s for s in fd["slots"] if bl["lo"] <= s["id"] <= bl["hi"]]
    srows = [[(str(s["id"]), s["id"]), f'<span class="mono small">{esc(s["name"])}</span>', (str(s["scale"]), s["scale"]), esc(s["channel"]),
              "✓" if s["id"] in by else ""] for s in slots]
    return "\n".join([
        f'<h1>{esc(bl["family"])}</h1>',
        f'<p>Slots {bl["lo"]}–{bl["hi"]} ({bl["width"]}), {len(bl["signals"])} signals, replication <code>{esc(bl["replication"])}</code>. '
        f'Declared in {src_cite(fd["path"], bl["line"])} as <code>{esc(bl["static"])}</code>.</p>',
        '<h2>Signals</h2>', table(["Local", "Signal", "Statistic", "Form", "Direction", "Cost", "Kernel", "Defect", "Constructor"], rows, numeric_cols=(0,)),
        f'<h2>Slots ({len(slots)})</h2>', filterbar("slots", "Filter slots"),
        table(["ID", "Name", "Scale", "Channel", "by_v2fy"], srows, tid="slots", numeric_cols=(0, 2))])


def page_zenanalyze(m) -> str:
    za = m["za"]
    feats = za["features"]
    pk = m["pickers"]
    tier_names = [t for t in za["tiers"] if za["tiers"][t]]
    rows = []
    for f in feats:
        flags = []
        if f["cfg"]:
            flags.append("cfg: " + ", ".join(f["cfg"]))
        if f["deprecated"]:
            flags.append("deprecated")
        rows.append([(str(f["id"]), f["id"]), f'<span class="mono">{esc(f["name"])}</span>', esc(f["ty"]),
                     esc(", ".join(t.replace("_FEATURES", "").lower() for t in f.get("tiers", []))),
                     esc("; ".join(flags)), f'<span class="mono small">{esc((f["qualified"] or "").split("@")[-1])}</span>',
                     f'<span class="small">{md_inline(_short(f["doc"], 200))}</span>', src_cite(zenanalyze_src.FEATURE_RS, f["line"], str(f["line"]))])
    on = {f["id"] for f in feats}
    tips = {f["id"]: f'#{f["id"]} {f["name"]}' for f in feats}
    for r in za["retired_ids"]:
        tips[r] = f"#{r} retired (reserved)"
    maxid = max(on) + 1
    drift_rows = []
    totals = Counter()
    for p in pk:
        c = Counter(x["state"] for x in p["pins"])
        totals.update(c)
        drifted = [x for x in p["pins"] if x["state"] in ("hash-changed", "missing")]
        other = [x for x in p["pins"] if x["state"] == "not-a-catalogue-feature"]
        st = "fail" if drifted else ("info" if other else ("pass" if p["pins"] else "na"))
        if not p["pins"] and "dynamic" in p["kind"]:
            st = "na"
        detail = "; ".join(f'{x["pin"]} → {x["current"] or "absent"}' for x in drifted)
        if other:
            detail += ("; " if detail else "") + "not in catalogue: " + ", ".join(x["name"] for x in other[:8]) + (" …" if len(other) > 8 else "")
        drift_rows.append([f'<span class="mono small">{esc(p["name"])}</span>', esc(p["kind"]), (str(len(p["pins"])), len(p["pins"])),
                           chip(st, {"fail": "drift", "pass": "current", "info": "names not in catalogue", "na": "no static list"}[st]),
                           f'<span class="small mono">{esc(detail)}</span>', f'<span class="mono small">{esc(p["path"].split(":", 1)[1])}</span>'])
    b = ['<h1>zenanalyze features</h1>',
         f'<p class="lede">zenanalyze extracts image features that pickers use to choose codecs and settings. Its catalogue is the '
         f'<code>features_table!</code> in <code>src/feature.rs</code>: {len(feats)} features with stable numeric IDs '
         f'(crate version {esc(za["version"])}, <code>FEATURE_DEFS_VERSION</code> {za["feature_defs_version"]}). Each feature '
         f'also has a versioned identity <code>name@hex8</code> whose hash changes when its values change.</p>',
         '<div class="statrow">'
         f'<div class="stat"><div class="v">{len(feats)}</div><div class="l">features</div></div>'
         f'<div class="stat"><div class="v">{sum(1 for f in feats if "hdr" in f["cfg"])}</div><div class="l">HDR-gated (cfg hdr)</div></div>'
         f'<div class="stat"><div class="v">{sum(1 for f in feats if "experimental" in f["cfg"])}</div><div class="l">experimental-gated</div></div>'
         f'<div class="stat"><div class="v">{sum(1 for f in feats if f["deprecated"])}</div><div class="l">deprecated</div></div>'
         f'<div class="stat"><div class="v">{len(za["retired_ids"])}</div><div class="l">retired IDs reserved</div></div></div>',
         f'<h2>ID space 0–{maxid - 1}</h2><p class="small">Highlighted: assigned IDs. Hover for names; retired IDs are listed in '
         f'<code>RESERVED_RETIRED_IDS</code> ({src_cite(zenanalyze_src.FEATURE_RS, za["retired_line"])}).</p>',
         svg.id_layout(on, maxid, cols=32, title="zenanalyze ids", tips=tips),
         f'<h2 id="drift">Which pickers pin which features, and drift</h2>'
         f'<p>A picker records the features it was trained on. <strong>Drift</strong> means a pinned <code>name@hex8</code> no longer '
         f'matches the current identity (the feature\'s values changed after the bake), or a pinned name is absent from the catalogue. '
         f'Across {len(pk)} artifacts: {totals.get("current", 0)} current pins, {totals.get("hash-changed", 0)} hash-changed, '
         f'{totals.get("missing", 0)} missing, {totals.get("not-a-catalogue-feature", 0)} names not in the catalogue (retired features '
         f'or caller inputs such as <code>log_dist</code>).</p>',
         filterbar("pickers", "Filter artifacts", [(3, "State", ["drift", "current", "names not in catalogue", "no static list"])]),
         table(["Artifact", "Kind", "Pins", "State", "Drift detail", "Path (zenanalyze)"], drift_rows, tid="pickers", numeric_cols=(2,)),
         '<p class="small muted">Current identities come from <code>benchmarks/feature_qualified_names.tsv</code>, which zenanalyze '
         'keeps in sync with a golden test. Router pins come from each ZNPR file\'s <code>zentrain.feature_columns</code> metadata.</p>',
         '<h2>Tier sets</h2><p>Analyzer passes that compute groups of features together:</p>',
         table(["Set", "Features"], [[f'<span class="mono">{esc(t)}</span>', f'<span class="small mono">{esc(", ".join(za["tiers"][t]))}</span>']
                                     for t in tier_names], sortable=False),
         f'<h2>Catalogue ({len(feats)})</h2>', filterbar("zafeats", "Filter features"),
         table(["ID", "Name", "Type", "Tiers", "Flags", "Hash", "Description", "Line"], rows, tid="zafeats", numeric_cols=(0,))]
    return _render(m, title="zenanalyze features", current="zenanalyze.html", body="\n".join(b))


def page_sources(m, coverage: list[dict]) -> str:
    rows = [[f'<span class="mono small">{esc(s["path"])}</span>', f'<span class="mono small">{esc(s["sha256"][:16])}</span>',
             esc(", ".join(s["readers"])), (str(s["entities"]), s["entities"])] for s in m["sources"]]
    cov = [[esc(c["section"]), (f'{c["covered"]}/{c["total"]}', c["covered"] / c["total"] if c["total"] else 0),
            f'<span class="small">{esc(c["missing"])}</span>'] for c in coverage]
    b = ['<h1>Sources and coverage</h1>',
         f'<p>Built {esc(m["built_utc"])} from zensim commit <span class="mono">{esc(m["commit"])}</span> and zenanalyze commit '
         f'<span class="mono">{esc(m["zenanalyze_commit"])}</span>. Every file read is listed with its SHA-256 prefix at build time, '
         'the readers that parsed it, and the number of entities taken from it. The build fails if any source changes shape.</p>',
         '<h2>Coverage against the brief</h2>', table(["Section", "Covered", "Missing"], cov, numeric_cols=(1,), sortable=False),
         f'<h2>Files read ({len(rows)})</h2>', filterbar("srcs", "Filter files"),
         table(["Path", "SHA-256 (prefix)", "Readers", "Entities"], rows, tid="srcs", numeric_cols=(3,))]
    return _render(m, title="Sources", current="sources.html", body="\n".join(b))
