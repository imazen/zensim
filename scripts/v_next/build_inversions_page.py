#!/usr/bin/env python3
"""Build the D-inversions visual page: 10 worst material inversions on the
new ladder instrument, full-frame + native crop per step, labelled with q,
ssim2, D, encoded bytes. Tiles already rendered via the imazen-only
ladder_tile_gen binary (zenpng + zenresize) by gen_tiles.py."""
import html
import json
import sys
from pathlib import Path

OUT_DIR = Path("/mnt/v/output/zensim/ladder-2026-09-05/inversions")


def fmt(v, nd=2):
    return f"{v:.{nd}f}"


def bytes_fmt(n):
    return f"{n:,} B"


def render_entry(idx, e):
    img, codec, zone = e["image"], e["codec"], e["zone"]
    q0, q1 = e["q0"], e["q1"]
    d0, d1 = e["d0"], e["d1"]
    s0, s1 = e["ssim2_0"], e["ssim2_1"]
    b0, b1 = e["bytes0"], e["bytes1"]
    delta = e["delta"]
    ssim2_delta = s1 - s0
    ssim2_agrees = ssim2_delta < -0.5
    # 2026-09-05 two-reference ruling: ssim2 alone is not evidence. A step is
    # the ENCODER's only where butteraugli-pnorm3 independently agrees by its
    # own derived margin. `attribution` is attached by the caller from the
    # reference-truth table; absent, the entry is reported as NOT MEASURABLE
    # rather than silently falling back to the single-reference reading.
    attribution = e.get("attribution", "not-measurable")
    b3_delta = e.get("d_butteraugli_pnorm3")
    cx, cy, cw, ch = e["crop"]
    tiles = e["step_tiles"]

    b3_txt = "n/a" if b3_delta is None else f"{b3_delta:+.4f}"
    s2_txt = f"{ssim2_delta:+.3f}"
    if attribution == "encoder-confirmed":
        agree_badge = (
            '<span class="badge agree">ENCODER-CONFIRMED &mdash; ssim2 '
            f'({s2_txt}) AND butteraugli-pnorm3 ({b3_txt}) both call the higher '
            'setting worse</span>'
        )
    elif attribution == "dial-only":
        why = (
            "butteraugli moves the OTHER way"
            if (b3_delta is not None and b3_delta < 0)
            else "butteraugli agrees in direction but below its 0.05 margin"
            if (b3_delta is not None and ssim2_agrees)
            else "ssim2 itself is not materially backwards here"
        )
        agree_badge = (
            '<span class="badge disagree">DIAL-ONLY &mdash; not corroborated: '
            f'ssim2 {s2_txt}, butteraugli-pnorm3 {b3_txt} ({why})</span>'
        )
    else:
        agree_badge = (
            '<span class="badge unknown">NOT MEASURABLE &mdash; no reference-truth '
            'row for this pair; it stays charged to the dial</span>'
        )

    step_cells = []
    for step_label, qv, dv, sv, bv, tile in (
        ("q" + fmt(q0, 0), q0, d0, s0, b0, tiles[0]),
        ("q" + fmt(q1, 0), q1, d1, s1, b1, tiles[1]),
    ):
        step_cells.append(f"""
      <div class="step">
        <div class="step-title">{html.escape(codec)} q={fmt(qv,0)}</div>
        <img class="tile-full" src="{tile['full']}" alt="{html.escape(img)} q={fmt(qv,0)} full frame" loading="lazy">
        <img class="tile-crop" src="{tile['crop']}" alt="{html.escape(img)} q={fmt(qv,0)} crop" loading="lazy">
        <table class="stat-table">
          <tr><td>ssim2 (truth)</td><td>{fmt(sv)}</td></tr>
          <tr><td>D (shipped)</td><td>{fmt(dv)}</td></tr>
          <tr><td>encoded bytes</td><td>{bytes_fmt(bv)}</td></tr>
        </table>
      </div>""")

    return f"""
  <section class="entry" id="entry{idx}">
    <h2>#{idx+1}. {html.escape(img)} &mdash; {html.escape(codec)} &mdash; zone {html.escape(zone)}</h2>
    <div class="meta-row">
      <div class="delta-box">
        <div>D step: <b>{fmt(d0)} &rarr; {fmt(d1)}</b> (&Delta;={fmt(d1-d0)}, worst_step magnitude {fmt(delta)})</div>
        <div>ssim2 step: <b>{fmt(s0)} &rarr; {fmt(s1)}</b> (&Delta;={fmt(ssim2_delta)})</div>
        <div>bytes: {bytes_fmt(b0)} &rarr; {bytes_fmt(b1)} ({'+' if b1>=b0 else ''}{b1-b0:+,} B, {100*(b1-b0)/b0:+.1f}%)</div>
        {agree_badge}
      </div>
      <div class="ref-box">
        <div class="step-title">reference</div>
        <img class="tile-full-small" src="{e['ref_full']}" alt="{html.escape(img)} reference" loading="lazy">
        <div class="crop-note">detail crop: <code>({cx},{cy})</code> {cw}&times;{ch}px native, same window on every tile below</div>
      </div>
    </div>
    <div class="steps">
      {''.join(step_cells)}
    </div>
  </section>"""


def render_summary_table(entries):
    rows = []
    for i, e in enumerate(entries):
        ssim2_delta = e["ssim2_1"] - e["ssim2_0"]
        agrees = "yes" if ssim2_delta < -0.5 else "NO"
        rows.append(f"""      <tr>
        <td><a href="#entry{i}">#{i+1}</a></td>
        <td>{html.escape(e['image'])}</td>
        <td>{html.escape(e['codec'])}</td>
        <td>{html.escape(e['zone'])}</td>
        <td>q{fmt(e['q0'],0)}&rarr;q{fmt(e['q1'],0)}</td>
        <td>{fmt(e['delta'])}</td>
        <td>{fmt(e['d0'])}&rarr;{fmt(e['d1'])}</td>
        <td>{fmt(e['ssim2_0'])}&rarr;{fmt(e['ssim2_1'])}</td>
        <td>{fmt(ssim2_delta)}</td>
        <td class="{'yes' if agrees=='yes' else 'no'}">{agrees}</td>
      </tr>""")
    return f"""
    <table class="summary">
      <thead>
        <tr><th>#</th><th>image</th><th>codec</th><th>zone</th><th>step</th>
            <th>D worst_step</th><th>D value</th><th>ssim2 value</th><th>ssim2 &Delta;</th>
            <th>ssim2 agrees?</th></tr>
      </thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>"""


# Where the OWNER-produced attribution artifacts live. The census is written by
# `bake_verdict --encoder-inversion-census`; the per-entry butteraugli deltas
# come from the same reference-truth table.
ATTR_DIR = Path("/mnt/v/output/zensim/invtruth-2026-09-05")


def main():
    entries = json.load(open(OUT_DIR / "tiles_manifest.json"))
    # Attach the 2026-09-05 two-reference attribution. The rule's OWNER is
    # `zensim_validate::dial_addressability::encoder_inversion`; this file does
    # not re-implement it — it joins a table the owner already produced
    # (`bake_verdict --encoder-inversion-census`), so the page and the gate can
    # never disagree about which pairs are the encoder's.
    census = ATTR_DIR / "encoder_inversions_ladder_pnorm3.tsv"
    enc = set()
    if census.is_file():
        for line in census.read_text().splitlines():
            if line.startswith("#") or line.startswith("image_id"):
                continue
            f = line.split("\t")
            if len(f) >= 6:
                enc.add((f[0], f[1], round(float(f[2]), 4), round(float(f[3]), 4)))
    else:
        print(f"WARNING: {census} absent — every entry renders NOT MEASURABLE",
              file=sys.stderr)
    deltas = json.loads((ATTR_DIR / "top10_attribution_2026-09-05.json").read_text()) \
        if (ATTR_DIR / "top10_attribution_2026-09-05.json").is_file() else []
    dmap = {(d["image"], d["codec"], round(float(d["q0"]), 4), round(float(d["q1"]), 4)):
            d.get("d_butteraugli_pnorm3") for d in deltas}
    for e in entries:
        k = (e["image"], e["codec"], round(float(e["q0"]), 4), round(float(e["q1"]), 4))
        if census.is_file():
            e["attribution"] = "encoder-confirmed" if k in enc else "dial-only"
        e["d_butteraugli_pnorm3"] = dmap.get(k)
    n_agree = sum(1 for e in entries if (e["ssim2_1"] - e["ssim2_0"]) < -0.5)
    n_encoder = sum(1 for e in entries if e.get("attribution") == "encoder-confirmed")
    n_dial = sum(1 for e in entries if e.get("attribution") == "dial-only")

    body_entries = "".join(render_entry(i, e) for i, e in enumerate(entries))
    summary = render_summary_table(entries)

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>D's ten worst material inversions &mdash; new ladder instrument (2026-09-05)</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 24px;
          background: #0e1116; color: #e6e6e6; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2.5rem; border-bottom: 1px solid #333; padding-bottom: 4px; }}
  p.lede {{ color: #aab; max-width: 900px; }}
  a {{ color: #7cc4ff; }}
  code {{ background: #1c2128; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }}
  table {{ border-collapse: collapse; margin: 8px 0; font-size: 0.85rem; }}
  table.summary {{ width: 100%; }}
  th, td {{ border: 1px solid #333; padding: 4px 8px; text-align: right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
  thead th {{ background: #171b21; }}
  td.yes {{ color: #7fd88f; font-weight: 600; }}
  td.no {{ color: #ff8f7c; font-weight: 600; }}
  .meta-row {{ display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 10px; }}
  .delta-box {{ background: #171b21; border: 1px solid #2a2f38; border-radius: 6px; padding: 10px 14px; font-size: 0.88rem; line-height: 1.6; }}
  .ref-box {{ display: flex; flex-direction: column; align-items: flex-start; }}
  .tile-full-small {{ max-width: 220px; border-radius: 4px; margin: 4px 0; }}
  .crop-note {{ font-size: 0.78rem; color: #99a; max-width: 260px; }}
  .badge {{ display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; font-weight: 600; }}
  .badge.agree {{ background: #204d2a; color: #a8f0b8; }}
  .badge.disagree {{ background: #4d2020; color: #f0b8a8; }}
  .badge.unknown {{ background: #3a3a3a; color: #d8d8d8; }}
  .steps {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .step {{ background: #171b21; border: 1px solid #2a2f38; border-radius: 6px; padding: 10px; width: 340px; }}
  .step-title {{ font-size: 0.82rem; margin-bottom: 6px; font-weight: 600; }}
  .tile-full {{ width: 100%; display: block; border-radius: 3px; margin-bottom: 6px; }}
  .tile-crop {{ width: 100%; display: block; border-radius: 3px; outline: 1px dashed #445; margin-bottom: 6px; }}
  .stat-table {{ width: 100%; font-size: 0.8rem; }}
  .stat-table td {{ padding: 2px 6px; }}
  footer {{ margin-top: 3rem; font-size: 0.78rem; color: #778; border-top: 1px solid #333; padding-top: 10px; }}
  @media (prefers-color-scheme: light) {{
    body {{ background: #fafbfc; color: #1a1f27; }}
    .delta-box, .step {{ background: #f1f3f6; border-color: #d7dce2; }}
    thead th {{ background: #eceff3; }}
    code {{ background: #eceff3; }}
    .badge.agree {{ background: #d8f0dd; color: #1a6b2e; }}
    .badge.disagree {{ background: #f0d8d3; color: #8f2e1a; }}
    .badge.unknown {{ background: #e2e2e2; color: #444; }}
  }}
</style>
</head>
<body>
<h1>Shipped Profile D &mdash; ten worst material inversions on the new ladder instrument</h1>
<p class="lede">
  Report-only lane, 2026-09-05. These are shipped Profile D's (<code>zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin</code>,
  sha256 <code>921a8f67&hellip;</code>) ten deepest single-step MATERIAL inversions (&ge;0.5 dial points backwards as the
  codec's nominal quality setting rises) on the new 5-codec, floor-dense ladder instrument
  (<code>dial_grid_372col_ladder.parquet</code>, 9,593 distinct settings). All ten are internal wobbles
  within a ladder &mdash; <b>none</b> of D's inversions on this instrument make a whole ladder end lower-quality
  than it started (0/195 ladder-zones "ends backwards"); D's own <code>mono_pct</code> here is 0.9931
  (a HIGHER, i.e. better, monotonicity rate than the shipped board grid's 0.9847).
</p>
<p class="lede">
  <b>Headline finding, under the 2026-09-05 TWO-REFERENCE rule: {n_encoder} of 10 are
  ENCODER-confirmed</b> &mdash; ssim2 <i>and</i> butteraugli-pnorm3 independently call the higher setting
  worse at the same step, each by its own material margin (&minus;0.5 ssim2 points; +0.05 butteraugli
  distance, derived by equivalence to the ssim2 one). The remaining {n_dial} are <b>DIAL-ONLY</b>:
  not corroborated, so they stay charged to the dial.
</p>
<p class="lede">
  <b>ssim2 alone would have confirmed {n_agree} of 10.</b> That gap is the whole reason the rule takes two
  references: on four of these steps ssim2 reads a 7&ndash;10 point quality loss while butteraugli either
  moves the other way or moves less than its own margin. A single-reference reading credits the encoder
  with defects the second metric does not see &mdash; and, symmetrically, excuses the dial for them.
</p>
{summary}
{body_entries}
<footer>
  Full analysis + repro: <code>benchmarks/d_inversions_2026-09-05.md</code> (zensim repo).
  Tiles rendered ONLY through <code>zensim-bench/examples/ladder_tile_gen</code>
  (zenpng decode/encode + zenresize Mitchell downscale; no foreign imaging tool touches a pixel),
  crop windows are a centered native window sized to 40% of each image's short side (not hand-picked).
  Instrument: <code>benchmarks/ladder_instrument_2026-09-05.md</code>. Data: bake_verdict
  <code>--full-json</code> (D) + <code>--dial-peer-scores peer_ssim2=...</code> (mentor), cross-validated
  against a Python port of the SAME classification rule (0 mismatches vs D's own JSON on every
  (codec,zone) counter, both instruments).
</footer>
</body>
</html>"""
    (OUT_DIR / "index.html").write_text(doc)
    print(f"wrote {OUT_DIR / 'index.html'}  ({len(doc)} bytes)")
    print(f"n_agree (ssim2 alone) = {n_agree} / 10; "
          f"n_encoder (both references) = {n_encoder} / 10")


if __name__ == "__main__":
    main()
