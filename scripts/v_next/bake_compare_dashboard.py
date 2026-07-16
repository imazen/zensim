#!/usr/bin/env python3
"""Comparative dashboard over bake_verdict `--json` panels.

Reads every `*.json` a `bake_verdict --json` run wrote in a directory and renders
one self-contained, theme-aware HTML page comparing the bakes: per-corpus SROCC
(+ full panel), delta-vs-baseline coloring, a beats-baseline scorecard, and
inline-SVG charts. No parsing of the markdown report — it consumes the
structured panel the Rust owner emits.

This is a VISUALIZATION consumer, not a re-implementation: every number comes
from `bake_verdict` (which owns the stats via `zenstats`). Python here only
lays out HTML/SVG, which has no Rust owner.

Usage:
  python3 scripts/v_next/bake_compare_dashboard.py <verdict-dir> <out.html> \
      [--baseline B] [--title "..."]
"""
import argparse
import json
import pathlib
import sys

# Held-out vs train==val: KADID/TID pairs are 100 % train==val overlap (memorization,
# not skill — docs/EVAL_PANEL_REQUIREMENT.md), so they are shown but de-emphasized.
HELDOUT = {"CID22", "AIC-3 CTC", "AIC-4 sample", "imazen-26 non-photo (held-out)",
           "HF near-lossless (held-out refs)", "KonJND-1k (full)"}
CORPUS_ORDER = ["CID22", "imazen-26 non-photo (held-out)", "HF near-lossless (held-out refs)",
                "AIC-3 CTC", "AIC-4 sample", "KonJND-1k (full)", "KADIK10k", "TID2013"]


def short(name):
    return {"imazen-26 non-photo (held-out)": "non-photo",
            "HF near-lossless (held-out refs)": "HF near-lossless",
            "KonJND-1k (full)": "KonJND", "KADIK10k": "KADID",
            "AIC-3 CTC": "AIC-3", "AIC-4 sample": "AIC-4",
            "TID2013": "TID"}.get(name, name)


def load(verdict_dir):
    bakes = {}
    for f in sorted(pathlib.Path(verdict_dir).glob("*.json")):
        d = json.loads(f.read_text())
        by = {c["display"]: c for c in d["corpora"]}
        bakes[f.stem] = {"sha": d.get("bake_sha256", "")[:12], "corpora": by,
                         "path": d.get("bake", "")}
    return bakes


def color(delta):
    """Green for improvement, red for regression, on a saturating scale."""
    if delta is None:
        return "var(--cell)"
    m = max(-1.0, min(1.0, delta / 0.03))  # ±0.03 SROCC saturates
    if m >= 0:
        return f"color-mix(in srgb, var(--good) {int(m*70)}%, var(--cell))"
    return f"color-mix(in srgb, var(--bad) {int(-m*70)}%, var(--cell))"


def svg_grouped_bars(bakes, order, baseline):
    """Per-corpus SROCC, one bar per bake, grouped by corpus."""
    names = list(bakes)
    W, H, padl, padb, top = 900, 300, 40, 68, 24
    corpora = [c for c in order if any(c in b["corpora"] for b in bakes.values())]
    gw = (W - padl) / max(1, len(corpora))
    bw = gw * 0.8 / max(1, len(names))
    palette = ["#6aa9ff", "#ff8f6a", "#7ad18f", "#d18fd1", "#e8c469", "#66c7c7"]
    bars, labels, legend = [], [], []
    def y(v): return top + (H - top - padb) * (1 - max(0, min(1, v)))
    for gi, corp in enumerate(corpora):
        gx = padl + gi * gw
        for bi, nm in enumerate(names):
            c = bakes[nm]["corpora"].get(corp)
            if not c:
                continue
            v = c["srocc"]
            x = gx + gw * 0.1 + bi * bw
            bars.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw*0.9:.1f}" '
                        f'height="{H-padb-y(v):.1f}" fill="{palette[bi%len(palette)]}" '
                        f'rx="1.5"><title>{nm} · {short(corp)} SROCC {v:.4f}</title></rect>')
        emph = "font-weight:600" if corp in HELDOUT else "opacity:.6"
        labels.append(f'<text x="{gx+gw/2:.1f}" y="{H-padb+14}" text-anchor="middle" '
                      f'class="ax" style="{emph}">{short(corp)}</text>')
    for bi, nm in enumerate(names):
        lx = padl + bi * 150
        legend.append(f'<rect x="{lx}" y="{H-20}" width="11" height="11" rx="2" '
                      f'fill="{palette[bi%len(palette)]}"/>'
                      f'<text x="{lx+16}" y="{H-10}" class="ax">{nm}</text>')
    grid = "".join(f'<line x1="{padl}" y1="{y(g):.1f}" x2="{W}" y2="{y(g):.1f}" class="grid"/>'
                   f'<text x="{padl-4}" y="{y(g)+3:.1f}" text-anchor="end" class="ax">{g:.1f}</text>'
                   for g in (0.5, 0.7, 0.9, 1.0))
    return (f'<svg viewBox="0 0 {W} {H}" class="chart" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{padl}" y="16" class="ttl">Per-corpus SROCC (bold = held-out; faint = train==val memorization)</text>'
            f'{grid}{"".join(bars)}{"".join(labels)}{"".join(legend)}</svg>')


def svg_delta(bakes, order, baseline):
    """Delta-vs-baseline SROCC per corpus, for each non-baseline bake."""
    if baseline not in bakes:
        return "<p>(no baseline for delta chart)</p>"
    base = bakes[baseline]["corpora"]
    cands = [n for n in bakes if n != baseline]
    corpora = [c for c in order if c in base]
    W, rowh, padl = 900, 26, 150
    rows = []
    yy = 24
    scale = 2000  # px per SROCC unit for the delta bars
    midx = padl + 300
    for corp in corpora:
        rows.append(f'<text x="{padl-8}" y="{yy+15}" text-anchor="end" class="ax" '
                    f'style="{"font-weight:600" if corp in HELDOUT else "opacity:.6"}">{short(corp)}</text>')
        for ci, nm in enumerate(cands):
            c = bakes[nm]["corpora"].get(corp)
            b = base.get(corp)
            if not c or not b:
                continue
            d = c["srocc"] - b["srocc"]
            w = d * scale
            x = midx if w >= 0 else midx + w
            col = "var(--good)" if d >= 0 else "var(--bad)"
            oy = yy + ci * (rowh / max(1, len(cands)))
            rows.append(f'<rect x="{x:.1f}" y="{oy:.1f}" width="{abs(w):.1f}" '
                        f'height="{rowh/len(cands)-2:.1f}" fill="{col}" rx="1">'
                        f'<title>{nm} {short(corp)} Δ{d:+.4f}</title></rect>')
        yy += rowh + 6
    axis = f'<line x1="{midx}" y1="18" x2="{midx}" y2="{yy}" class="grid"/>'
    for t in (-0.02, 0.02):
        axis += (f'<line x1="{midx+t*scale}" y1="18" x2="{midx+t*scale}" y2="{yy}" class="grid"/>'
                 f'<text x="{midx+t*scale}" y="14" text-anchor="middle" class="ax">{t:+.2f}</text>')
    return (f'<svg viewBox="0 0 {W} {yy+10}" class="chart" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="4" y="12" class="ttl">Δ SROCC vs {baseline} (right=better)</text>'
            f'{axis}{"".join(rows)}</svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdict_dir")
    ap.add_argument("out")
    ap.add_argument("--baseline", default="B")
    ap.add_argument("--title", default="zensim bake comparison")
    a = ap.parse_args()

    bakes = load(a.verdict_dir)
    if not bakes:
        sys.exit(f"no *.json verdicts in {a.verdict_dir}")
    base = a.baseline if a.baseline in bakes else None

    order = [c for c in CORPUS_ORDER if any(c in b["corpora"] for b in bakes.values())]
    order += [c for b in bakes.values() for c in b["corpora"] if c not in order]

    # scorecard: held-out corpora each bake beats the baseline on
    scorecard = {}
    if base:
        for nm, b in bakes.items():
            if nm == a.baseline:
                continue
            wins = held = 0
            for c in order:
                if c not in HELDOUT or c not in b["corpora"] or c not in bakes[a.baseline]["corpora"]:
                    continue
                held += 1
                if b["corpora"][c]["srocc"] >= bakes[a.baseline]["corpora"][c]["srocc"]:
                    wins += 1
            scorecard[nm] = (wins, held)

    # main table
    STATS = [("srocc", "SROCC"), ("plcc", "PLCC"), ("pwrc", "PWRC"),
             ("z_rmse", "Z-RMSE"), ("per_ref_mean", "per-ref"), ("per_ref_frac_negative", "%bwd")]
    thead = "<th>corpus</th>" + "".join(
        f'<th>{nm}{"" if nm!=a.baseline else " ⋆"}<br><span class="sha">{b["sha"]}</span></th>'
        for nm, b in bakes.items())
    rows = []
    for corp in order:
        emph = "heldout" if corp in HELDOUT else "trainval"
        cells = [f'<td class="corp {emph}">{short(corp)}</td>']
        for nm, b in bakes.items():
            c = b["corpora"].get(corp)
            if not c:
                cells.append('<td>—</td>'); continue
            d = None
            if base and nm != a.baseline and corp in bakes[a.baseline]["corpora"]:
                d = c["srocc"] - bakes[a.baseline]["corpora"][corp]["srocc"]
            dtxt = f'<span class="d">{d:+.4f}</span>' if d is not None else ""
            cells.append(f'<td style="background:{color(d)}">{c["srocc"]:.4f}{dtxt}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")

    # per-ref (within-image ladder) panel — the metric a CODEC DIAL cares about
    # (rank one image's distortion ladder). For near-lossless HF the pooled SROCC
    # measures cross-image *scale* (genuinely hard, less relevant); per-ref
    # measures the ladder ranking, and it can flip the verdict entirely.
    perref_corpora = [c for c in order
                      if any(b["corpora"].get(c, {}).get("per_ref_mean") is not None
                             for b in bakes.values())]
    perref_html = ""
    if perref_corpora:
        head = "<th>corpus (within-image)</th>" + "".join(f"<th>{nm}</th>" for nm in bakes)
        prrows = []
        for corp in perref_corpora:
            cells = [f'<td class="corp heldout">{short(corp)}</td>']
            for nm, b in bakes.items():
                c = b["corpora"].get(corp)
                pr = c.get("per_ref_mean") if c else None
                bw = c.get("per_ref_frac_negative") if c else None
                if pr is None:
                    cells.append("<td>—</td>"); continue
                d = None
                if base and nm != a.baseline:
                    bp = bakes[a.baseline]["corpora"].get(corp, {}).get("per_ref_mean")
                    if bp is not None:
                        d = pr - bp
                dtxt = f'<span class="d">{d:+.3f}</span>' if d is not None else ""
                bwtxt = f'<span class="d">{bw:.0%} bwd</span>' if bw is not None else ""
                cells.append(f'<td style="background:{color(d)}">{pr:+.3f}{dtxt}{bwtxt}</td>')
            prrows.append(f"<tr>{''.join(cells)}</tr>")
        perref_html = (
            '<h2 class="h2">Per-reference SROCC — the codec-dial metric (rank ONE image\'s ladder)</h2>'
            '<p class="note">For near-lossless HF the pooled table above measures cross-image '
            '<i>scale</i> (all-near-lossless, genuinely ambiguous); this measures whether each '
            'image\'s distortion ladder is ordered right — what a codec binary-searching one image '
            'actually needs. A bake can crater pooled HF yet rank every ladder near-perfectly.</p>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(prrows)}</tbody></table>')

    sc_html = " · ".join(f"<b>{nm}</b> beats {a.baseline} on {w}/{h} held-out (pooled)"
                         for nm, (w, h) in scorecard.items()) or "(baseline only)"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{a.title}</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--cell:#f5f5f7;--good:#2e9e5b;--bad:#d0433b;--grid:#ccc;--mut:#666}}
@media(prefers-color-scheme:dark){{:root{{--bg:#15171a;--fg:#e8e8ea;--cell:#22252a;--good:#3fbf72;--bad:#e2695f;--grid:#3a3f46;--mut:#9aa0a8}}}}
:root[data-theme=dark]{{--bg:#15171a;--fg:#e8e8ea;--cell:#22252a;--good:#3fbf72;--bad:#e2695f;--grid:#3a3f46;--mut:#9aa0a8}}
:root[data-theme=light]{{--bg:#fff;--fg:#1a1a1a;--cell:#f5f5f7;--good:#2e9e5b;--bad:#d0433b;--grid:#ccc;--mut:#666}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;max-width:1000px}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 18px}}
.scorecard{{background:var(--cell);border-radius:8px;padding:12px 16px;margin:12px 0 20px;font-size:15px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 24px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 9px;text-align:right;border-bottom:1px solid var(--grid)}}
th{{font-weight:600;font-size:12px;color:var(--mut);vertical-align:bottom}}
td.corp{{text-align:left;font-weight:500}} td.corp.trainval{{opacity:.55}}
td.corp.heldout::before{{content:"● ";color:var(--good)}}
.sha{{font-weight:400;font-size:10px;color:var(--mut);font-family:monospace}}
.d{{display:block;font-size:10px;color:var(--mut)}} .chart{{width:100%;height:auto;margin:8px 0 26px}}
.ttl{{fill:var(--fg);font-size:12px;font-weight:600}} .ax{{fill:var(--mut);font-size:10px}}
.grid{{stroke:var(--grid);stroke-width:1}} .note{{color:var(--mut);font-size:12px}} .h2{{font-size:15px;margin:24px 0 4px}}
</style></head><body>
<h1>{a.title}</h1>
<p class="sub">{len(bakes)} bakes · baseline <b>{a.baseline}</b> · ● = held-out corpus · faint = train==val (memorization, not skill)</p>
<div class="scorecard">{sc_html}</div>
{svg_grouped_bars(bakes, order, a.baseline)}
{svg_delta(bakes, order, a.baseline)}
<h2 class="h2">Pooled SROCC (cross-image rank)</h2>
<table><thead><tr>{thead}</tr></thead><tbody>{''.join(rows)}</tbody></table>
{perref_html}
<p class="note">Green/red cells = Δ SROCC vs {a.baseline} (±0.03 saturates). Every number from
<code>bake_verdict --json</code> (stats owned by zenstats). KADID/TID are 100% train==val
overlap — shown faint, not a skill signal. per-ref = mean within-image SROCC; %bwd = fraction
of references ranked backwards.</p>
</body></html>"""
    pathlib.Path(a.out).write_text(html)
    print(f"wrote {a.out} ({len(bakes)} bakes)")
    if base:
        for nm, (w, h) in scorecard.items():
            print(f"  {nm}: beats {a.baseline} on {w}/{h} held-out corpora")


if __name__ == "__main__":
    main()
