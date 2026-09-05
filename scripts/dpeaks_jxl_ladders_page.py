#!/usr/bin/env python3
"""D-peaks JXL floor-ladder visual page builder (2026-09-05).

Builds a self-contained HTML page showing the 4 JXL dial-grid ladders that
invert under the `lam1em3` / `Dpeaks` (= `lam2em3`) peaks-block lasso fits
(benchmarks/d_peaks_jxl_floor_2026-09-05.md), and shows the `minus_f162`
leave-one-out refit (benchmarks/d_peaks_slot_ablation_2026-09-05.md) that
cures all four. Record: benchmarks/d_peaks_jxl_ladders_2026-09-05.md.

Nothing in `zensim/weights/` or `zensim/src/profile.rs` is read or written by
this script — it only reads frozen bake bytes and grid parquets that other
(already-landed) lanes produced.

Data sources (all read-only):
  - postC dial grid (image_id/codec/q/codec_param), the ORIGINAL (not the
    dummy-target copy) so `codec_param`/`param_kind` are present:
      instruments/dial_grid_372col_postC_2026-09-05.parquet
  - pixel paths. Row-order validated against the grid before this script was
    written (0 mismatches over all 4424 rows: codec_param == human_score,
    and image_id/codec/q identical to the "with_dummy_target" copy that
    `bake_dial_refit predict` was actually run against):
      ~/tmp/dial372_instruments/postC/grid_pairs.tsv
  - ssim2 truth: /mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv
  - `bake_dial_refit predict` dumps (raw + score-units), row-order == grid
    row order, in jxlfloor/work/: dship_{raw,dial}.tsv, lam1em3_{raw,dial}.tsv
    (produced by the d_peaks_jxl_floor_2026-09-05 lane) and
    minus_f162_{raw,dial}.tsv (produced by this lane, same command shape,
    against the frozen `slots/bakes/minus_f162_dial.bin`, sha256
    fcf4e4d4a0901ecf9fa7e99a39c5e2a636109abdf428c9b45180070b825c9f90).

Tile pixels are decoded/cropped/downscaled ONLY through imazen tools
(zenpng + zenresize, via zensim-bench's `ladder_tile_gen` example) per the
"IMAZEN-ONLY IMAGING/CODEC SOFTWARE" rule (~/work/zen/CLAUDE.md, 2026-09-02).
This script never decodes or resamples a pixel itself — it shells the Rust
binary per tile and composes the already-rendered PNGs into HTML/CSS. No
ImageMagick or other foreign imaging tool is used anywhere in this path.

Usage:
    cargo build --release --manifest-path zensim-bench/Cargo.toml \
        --example ladder_tile_gen --features m3-fixtures
    python3 scripts/dpeaks_jxl_ladders_page.py [--out-dir DIR] [--tile-bin PATH]
"""
from __future__ import annotations

import argparse
import csv
import html
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent

GRID = Path(
    "/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments/dial_grid_372col_postC_2026-09-05.parquet"
)
PAIRS_TSV = Path("/home/lilith/tmp/dial372_instruments/postC/grid_pairs.tsv")
SSIM2_TSV = Path("/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv")
WORK = Path("/mnt/v/output/zensim/dpeaks372-2026-09-05/jxlfloor/work")

DEFAULT_TILE_BIN = REPO_ROOT / "zensim-bench" / "target" / "release" / "examples" / "ladder_tile_gen"
DEFAULT_OUT_DIR = Path("/mnt/v/output/zensim/dpeaks372-2026-09-05/jxlfloor/ladders")

TILE_MAX = 512  # long-side cap for the full-frame downscale, per the brief

# Bake identity, for the page footer / provenance strip.
BAKE_SHAS = {
    "D (shipped)": ("zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin", "921a8f67…"),
    "lam1em3": (
        "/mnt/v/output/zensim/dpeaks372-2026-09-05/sweep/bakes/Dsweep_lam1em3_dial.bin",
        "4490e64b…",
    ),
    "minus_f162": (
        "/mnt/v/output/zensim/dpeaks372-2026-09-05/slots/bakes/minus_f162_dial.bin",
        "fcf4e4d4a090…",
    ),
}

# Manually-chosen 320x320 native detail-crop windows, picked by eye against a
# downscaled preview of each REFERENCE image (not from any codec output) so
# the same window is honest to compare across every q step. See the record
# doc for the reasoning behind each region (thin/high-frequency content:
# lens/strap edges, a shop-sign + window mullions, handwritten music
# annotation, a slur crossing dense notation).
CROPS: dict[str, tuple[int, int, int, int]] = {
    "2b79a18d1b7537e0_818x1022": (498, 0, 320, 320),
    "96a0024c685ead3f_1024sq": (704, 704, 320, 320),
    "b2e6e2b5969eaf25_1022x818": (340, 498, 320, 320),
    "f65a24b7e176eb47_1022x818": (260, 150, 320, 320),
}

TARGET_IMAGES = list(CROPS.keys())


def load_pred(path: Path) -> list[float]:
    out = []
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            out.append(float(row["pred"]))
    return out


def load_ssim2() -> dict[tuple[str, str, float], float]:
    m = {}
    with SSIM2_TSV.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            m[(row["image_id"], row["codec"], round(float(row["q"]), 6))] = float(row["pred"])
    return m


def load_pairs_tsv() -> dict[int, tuple[str, str]]:
    out = {}
    with PAIRS_TSV.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            out[int(row["row_id"])] = (row["ref_path"], row["dist_path"])
    return out


def run_tile(tile_bin: Path, mode: str, args: list[str]) -> None:
    cmd = [str(tile_bin), mode, *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stdout}\n{r.stderr}")


def fmt(v: float | None, nd: int = 4) -> str:
    if v is None:
        return "&mdash;"
    return f"{v:.{nd}f}"


def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--tile-bin", type=Path, default=DEFAULT_TILE_BIN)
    args = ap.parse_args()

    if not args.tile_bin.exists():
        sys.exit(
            f"tile binary not found: {args.tile_bin}\n"
            "build it first:\n"
            "  cargo build --release --manifest-path zensim-bench/Cargo.toml "
            "--example ladder_tile_gen --features m3-fixtures"
        )

    out_dir = args.out_dir
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    t = pq.read_table(GRID, columns=["image_id", "codec", "q", "codec_param", "param_kind"])
    image_id = t.column("image_id").to_pylist()
    codec = t.column("codec").to_pylist()
    q_col = t.column("q").to_pylist()
    cp_col = t.column("codec_param").to_pylist()

    pairs = load_pairs_tsv()
    ssim2_map = load_ssim2()

    lam1em3_raw = load_pred(WORK / "lam1em3_raw.tsv")
    lam1em3_dial = load_pred(WORK / "lam1em3_dial.tsv")
    mf162_raw = load_pred(WORK / "minus_f162_raw.tsv")
    mf162_dial = load_pred(WORK / "minus_f162_dial.tsv")
    dship_dial = load_pred(WORK / "dship_dial.tsv")
    dship_raw = load_pred(WORK / "dship_raw.tsv")

    n = len(image_id)
    ladders_html = []
    summary_rows = []

    for target in TARGET_IMAGES:
        crop_x, crop_y, crop_w, crop_h = CROPS[target]
        rows = [i for i in range(n) if image_id[i] == target and codec[i] == "jxl"]
        rows.sort(key=lambda i: q_col[i])
        assert len(rows) >= 5, f"{target}: only {len(rows)} jxl rows"

        bottom4 = rows[:4]
        q_lo, q_hi = q_col[rows[0]], q_col[rows[-1]]
        mid_target = (q_lo + q_hi) / 2.0
        rest = rows[4:]
        mid_idx = min(rest, key=lambda i: abs(q_col[i] - mid_target))

        step_defs = [
            ("q0", bottom4[0]),
            ("q8", bottom4[1]),
            ("q16", bottom4[2]),
            ("q24", bottom4[3]),
            ("mid", mid_idx),
        ]

        # Inversion pair: which adjacent bottom-4 pair has lam1em3 RAW
        # decreasing as q increases (the brief's own definition).
        inv_pair = None
        for a in range(3):
            ia, ib = bottom4[a], bottom4[a + 1]
            if lam1em3_raw[ib] < lam1em3_raw[ia]:
                inv_pair = (a, a + 1)
        assert inv_pair is not None, f"{target}: no inversion found in bottom-4 (expected one)"

        ref_path, _ = pairs[rows[0]]

        # --- tiles ---
        ref_full = tiles_dir / f"{target}__ref_full.png"
        ref_crop = tiles_dir / f"{target}__ref_crop.png"
        run_tile(args.tile_bin, "full", ["--in", ref_path, "--out", str(ref_full), "--max", str(TILE_MAX)])
        run_tile(
            args.tile_bin,
            "crop",
            ["--in", ref_path, "--out", str(ref_crop), "--x", str(crop_x), "--y", str(crop_y), "--w", str(crop_w), "--h", str(crop_h)],
        )

        cells = []
        for label, ridx in step_defs:
            _, dist_path = pairs[ridx]
            full_p = tiles_dir / f"{target}__{label}_full.png"
            crop_p = tiles_dir / f"{target}__{label}_crop.png"
            run_tile(args.tile_bin, "full", ["--in", dist_path, "--out", str(full_p), "--max", str(TILE_MAX)])
            run_tile(
                args.tile_bin,
                "crop",
                ["--in", dist_path, "--out", str(crop_p), "--x", str(crop_x), "--y", str(crop_y), "--w", str(crop_w), "--h", str(crop_h)],
            )
            qv = q_col[ridx]
            dist_val = cp_col[ridx]
            s2 = ssim2_map.get((target, "jxl", round(qv, 6)))
            cells.append(
                dict(
                    label=label,
                    row=ridx,
                    q=qv,
                    distance=dist_val,
                    full=full_p.relative_to(out_dir),
                    crop=crop_p.relative_to(out_dir),
                    ssim2=s2,
                    d_dial=dship_dial[ridx],
                    d_raw=dship_raw[ridx],
                    lam_raw=lam1em3_raw[ridx],
                    lam_dial=lam1em3_dial[ridx],
                    mf162_raw=mf162_raw[ridx],
                    mf162_dial=mf162_dial[ridx],
                )
            )

        a_label, b_label = step_defs[inv_pair[0]][0], step_defs[inv_pair[1]][0]
        summary_rows.append(
            dict(
                image=target,
                inv_pair=f"{a_label}→{b_label}",
                lam_a=lam1em3_raw[bottom4[inv_pair[0]]],
                lam_b=lam1em3_raw[bottom4[inv_pair[1]]],
                lam_dial_a=lam1em3_dial[bottom4[inv_pair[0]]],
                lam_dial_b=lam1em3_dial[bottom4[inv_pair[1]]],
                mf162_dial_a=mf162_dial[bottom4[inv_pair[0]]],
                mf162_dial_b=mf162_dial[bottom4[inv_pair[1]]],
                ssim2_a=ssim2_map.get((target, "jxl", round(q_col[bottom4[inv_pair[0]]], 6))),
                ssim2_b=ssim2_map.get((target, "jxl", round(q_col[bottom4[inv_pair[1]]], 6))),
                d_dial_a=dship_dial[bottom4[inv_pair[0]]],
                d_dial_b=dship_dial[bottom4[inv_pair[1]]],
            )
        )

        ladders_html.append(
            render_ladder(target, ref_path, ref_full.relative_to(out_dir), ref_crop.relative_to(out_dir), crop_x, crop_y, crop_w, crop_h, cells, inv_pair)
        )

    html_doc = render_page(ladders_html, summary_rows)
    (out_dir / "index.html").write_text(html_doc)
    print(f"wrote {out_dir / 'index.html'}")
    for r in summary_rows:
        print(
            f"{r['image']}: inversion {r['inv_pair']}  "
            f"lam1em3 raw {r['lam_a']:.4f}->{r['lam_b']:.4f}  "
            f"lam1em3 dial {r['lam_dial_a']:.4f}->{r['lam_dial_b']:.4f}  "
            f"minus_f162 dial {r['mf162_dial_a']:.4f}->{r['mf162_dial_b']:.4f}  "
            f"ssim2 {r['ssim2_a']:.4f}->{r['ssim2_b']:.4f}  "
            f"D {r['d_dial_a']:.4f}->{r['d_dial_b']:.4f}"
        )


def render_ladder(target, ref_path, ref_full, ref_crop, cx, cy, cw, ch, cells, inv_pair):
    cell_tiles = []
    labels_order = [c["label"] for c in cells]
    for idx, c in enumerate(cells):
        is_inv_a = idx == inv_pair[0]
        is_inv_b = idx == inv_pair[1]
        cls = "cell"
        if is_inv_a or is_inv_b:
            cls += " inv"
        title = f"q={c['q']:.0f}" if c["label"] != "mid" else f"mid (q={c['q']:.0f})"
        cell_tiles.append(
            f"""
      <div class="{cls}">
        <div class="cell-title">{html.escape(title)} <span class="dist">dist={c['distance']:.3f}</span></div>
        <img class="tile-full" src="{c['full']}" alt="{html.escape(target)} {c['label']} full frame" loading="lazy">
        <img class="tile-crop" src="{c['crop']}" alt="{html.escape(target)} {c['label']} crop" loading="lazy">
        {"<div class='arrow'>&darr; inversion starts here</div>" if is_inv_a else ""}
      </div>"""
        )

    table_rows = []
    for idx, c in enumerate(cells):
        row_cls = "inv-row" if idx in inv_pair else ""
        title = f"q={c['q']:.0f}" if c["label"] != "mid" else f"mid q={c['q']:.0f}"
        table_rows.append(
            f"""      <tr class="{row_cls}">
        <td>{html.escape(title)}</td>
        <td>{c['distance']:.3f}</td>
        <td>{fmt(c['ssim2'])}</td>
        <td>{fmt(c['d_dial'])}</td>
        <td class="lam-raw">{fmt(c['lam_raw'])}</td>
        <td class="lam-dial">{fmt(c['lam_dial'])}</td>
        <td>{fmt(c['mf162_dial'])}</td>
      </tr>"""
        )

    return f"""
  <section class="ladder">
    <h2>{html.escape(target)}</h2>
    <div class="ref-row">
      <div class="cell ref">
        <div class="cell-title">reference (source)</div>
        <img class="tile-full" src="{ref_full}" alt="{html.escape(target)} reference full frame" loading="lazy">
        <img class="tile-crop" src="{ref_crop}" alt="{html.escape(target)} reference crop" loading="lazy">
      </div>
      <div class="crop-note">detail crop origin: <code>({cx}, {cy})</code>, size <code>{cw}&times;{ch}</code> native pixels,
        same window on the reference and every JXL step below.</div>
    </div>
    <div class="cells">
      {''.join(cell_tiles)}
    </div>
    <table class="scores">
      <thead>
        <tr><th>step</th><th>distance</th><th>ssim2 truth</th><th>D (shipped)</th>
            <th>lam1em3 raw</th><th>lam1em3 dial</th><th>minus_f162 dial</th></tr>
      </thead>
      <tbody>
{chr(10).join(table_rows)}
      </tbody>
    </table>
  </section>"""


def render_page(ladders_html, summary_rows):
    summary_tr = []
    for r in summary_rows:
        summary_tr.append(
            f"""      <tr>
        <td>{html.escape(r['image'])}</td>
        <td>{html.escape(r['inv_pair'])}</td>
        <td>{r['ssim2_a']:.4f} &rarr; {r['ssim2_b']:.4f}</td>
        <td>{r['d_dial_a']:.4f} &rarr; {r['d_dial_b']:.4f}</td>
        <td class="lam-dial">{r['lam_dial_a']:.4f} &rarr; {r['lam_dial_b']:.4f}</td>
        <td>{r['mf162_dial_a']:.4f} &rarr; {r['mf162_dial_b']:.4f}</td>
      </tr>"""
        )

    bake_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(path)}</code></td><td><code>{sha}</code></td></tr>"
        for name, (path, sha) in BAKE_SHAS.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>D-peaks JXL floor ladders (2026-09-05)</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 24px;
          background: #0e1116; color: #e6e6e6; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.5rem; border-bottom: 1px solid #333; padding-bottom: 4px; }}
  p.lede {{ color: #aab; max-width: 900px; }}
  a {{ color: #7cc4ff; }}
  code {{ background: #1c2128; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 8px; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #333; padding: 4px 8px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ background: #171b21; }}
  tr.inv-row, td.inv-cell {{ background: #3a1414; }}
  td.lam-raw, td.lam-dial {{ font-weight: 600; }}
  table.summary td.lam-dial {{ background: #3a1414; }}
  .cells {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0; }}
  .ref-row {{ display: flex; align-items: flex-start; gap: 16px; flex-wrap: wrap; margin-bottom: 10px; }}
  .crop-note {{ font-size: 0.82rem; color: #99a; max-width: 420px; padding-top: 8px; }}
  .cell {{ width: 176px; background: #171b21; border: 1px solid #2a2f38; border-radius: 6px; padding: 8px; position: relative; }}
  .cell.inv {{ border-color: #c0392b; box-shadow: 0 0 0 2px #c0392b55; }}
  .cell.ref {{ width: 200px; }}
  .cell-title {{ font-size: 0.78rem; margin-bottom: 6px; display: flex; justify-content: space-between; }}
  .cell-title .dist {{ color: #889; font-weight: normal; }}
  .tile-full {{ width: 100%; display: block; border-radius: 3px; margin-bottom: 4px; }}
  .tile-crop {{ width: 100%; display: block; border-radius: 3px; outline: 1px dashed #445; }}
  .arrow {{ position: absolute; right: -22px; top: 50%; transform: translateY(-50%) rotate(90deg);
            font-size: 0.68rem; color: #ff6b5b; white-space: nowrap; }}
  footer {{ margin-top: 3rem; font-size: 0.78rem; color: #778; border-top: 1px solid #333; padding-top: 10px; }}
  @media (prefers-color-scheme: light) {{
    body {{ background: #fafafa; color: #111; }}
    .cell {{ background: #fff; border-color: #ddd; }}
    thead th {{ background: #f0f0f0; }}
    tr.inv-row {{ background: #ffe0dc; }}
    table.summary td.lam-dial {{ background: #ffe0dc; }}
    code {{ background: #eee; }}
  }}
</style>
</head>
<body>
<h1>D-peaks JXL floor ladders &mdash; the 4 failing 372-input peaks-block fits, cured by minus_f162</h1>
<p class="lede">
  These are the only 4 of JXL's 33 dial-grid reference ladders where the
  <code>lam1em3</code> (and <code>Dpeaks</code>/<code>lam2em3</code>) peaks-block
  lasso fits invert &mdash; the model's own raw output goes DOWN as the encoder
  is told to use less distortion (higher q). Shipped <code>ZensimProfile::D</code>
  and the <code>minus_f162</code> leave-one-out refit (drop feature f162, same
  &lambda;=1e-3 fit) are both monotone here; ssim2 (the mentor metric) is monotone
  on all 4 in truth. Full analysis lives in the zensim repo at
  <code>benchmarks/d_peaks_jxl_floor_2026-09-05.md</code>,
  <code>benchmarks/d_peaks_slot_ablation_2026-09-05.md</code>, and
  <code>benchmarks/d_peaks_jxl_ladders_2026-09-05.md</code> (this page's own record).
  Each ladder below shows the reference plus its 4 lowest-q JXL steps (q=0,8,16,24
  on the grid's normalized 0-100 quality scale &mdash; q=0 is the most aggressive
  setting, largest butteraugli distance) plus one mid-ladder step for context
  (nearest actual grid point to the ladder's own (min+max)/2 quality). Each tile
  pairs a Mitchell-downscaled full frame (&le;512px long side) with a native 1:1
  crop of the SAME detail region (origin stated per ladder) so the actual codec
  output bytes are visible, not a resample of them. Decoded/resized only through
  zenpng + zenresize (<code>zensim-bench/examples/ladder_tile_gen.rs</code>); no
  ImageMagick or third-party codec in this path.
</p>

<h2>Summary &mdash; the four inversions</h2>
<table class="summary">
  <thead><tr><th>ladder</th><th>failing step pair</th><th>ssim2 truth</th><th>D (shipped) dial</th>
             <th>lam1em3 dial (inverts)</th><th>minus_f162 dial (cured)</th></tr></thead>
  <tbody>
{chr(10).join(summary_tr)}
  </tbody>
</table>

{''.join(ladders_html)}

<footer>
  <p>Bakes read (frozen, read-only):</p>
  <table><thead><tr><th>label</th><th>path</th><th>sha256 (prefix)</th></tr></thead>
  <tbody>{bake_rows}</tbody></table>
  <p>Instrument: postC dial grid, 4,424 rows, 38 refs, sha256
     <code>506bdadfce7d2c4ea2ad37a6a2e7635f5eda6f945126033e4cf4365d3c695643</code>.
     ssim2 truth: <code>ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv</code>.
     Generated by <code>scripts/dpeaks_jxl_ladders_page.py</code>.</p>
</footer>
</body>
</html>
"""


if __name__ == "__main__":
    build()
