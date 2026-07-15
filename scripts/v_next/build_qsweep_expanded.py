#!/usr/bin/env python3
"""Expanded multi-codec dial-reach grid (2026-05-29).

Densifies the q-sweep where dial precision matters most, per request:
  - q0 added (dial floor)
  - step-1 across q90..q100 (near-lossless — where dials saturate)
  - JND zone densified (q70..q90 step 2 — the visually-lossless / perceptibility band)
  - JXL swept in BUTTERAUGLI DISTANCE (its native near-lossless axis), relabeled to a
    monotone q-equivalent so the monotonicity check sorts by quality.

Drives `zen-metrics sweep --metric zensim-gpu --zensim-features-regime with-iw
--feature-output` (372 features, encode+extract in one shot), merges the per-codec
feature parquets, builds the (features CSV + manifest TSV) qsweep_eval consumes, and
runs qsweep_eval on the candidate bakes.

JXL note: zen-metrics `encode_jxl_expert` uses the `distance` knob when present
(`--knob-grid '{"distance":[...]}'`), ignoring q. We sweep a distance ladder and
relabel each row q_equiv = round(100 - 7*distance) (clamped [0,100]) so lower
distance = higher quality = higher q on the monotonicity axis.
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np

# Binary path overridable via ZM_BIN — the f64 fractional-q-grid support
# (zenmetrics commit 759ab501) must be present, so during rollout point this
# at the workspace build that has it.
ZM = os.environ.get("ZM_BIN", "/home/lilith/work/zen/zenmetrics/target/release/zenmetrics")
SOURCES = "/tmp/qsweep_sources"
OUTDIR = "/mnt/v/output/zensim/qsweep_expanded_2026-05-29"
os.makedirs(OUTDIR, exist_ok=True)

# Expanded q-grid for q-parameterized codecs (jpeg/webp/avif). zenjpeg/webp/avif
# all take a FLOAT quality (with_generic_quality(f32) / with_quality(f32) /
# quality(f32)); the zen-metrics sweep --q-grid now threads f64 (was u32), so the
# near-lossless band can be sampled at fractional q to match JXL's distance ladder
# resolution (JXL resolves q-equiv 99.9 at d=0.025). Grid:
#   low/mid coarse + JND-zone (70..90 step 2) + near-lossless integer (90..96)
#   + fractional near-lossless (96..100, down to 0.1 q) + q0.
QGRID = sorted(set(
    [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]            # low/mid (existing coarse)
    + list(range(70, 91, 2))                          # JND zone: 70,72,...,90 (step 2)
    + list(range(90, 97, 1))                          # near-lossless integer: 90..96
    + [96.5, 97, 97.5, 98, 98.5, 99, 99.25, 99.5, 99.75, 99.9, 100]  # fractional near-lossless
    + [87, 89]                                        # keep prior near-JND points
))
# str() keeps ints as "90" and floats as "99.5"; Python dedupes 90==90.0 in the set.
QGRID_STR = ",".join(str(q) for q in QGRID)

# JXL butteraugli-distance ladder — variable density, finest near lossless:
#   - 0.0 → 0.3  step 0.025  (near-lossless, where the dial must resolve steps)
#   - 0.3 → 1.0  step 0.05   (high fidelity)
#   - 1.0 → 3.0  step 0.2    (mid fidelity)
#   - 3.5 → 10              (coarse mid)
#   - 13 → 25    step 2      (low-quality tail)
JXL_DISTANCES = sorted({
    round(d, 3)
    for d in (
        [i * 0.025 for i in range(0, 13)]          # 0.000 .. 0.300
        + [0.3 + i * 0.05 for i in range(1, 15)]   # 0.350 .. 1.000
        + [1.0 + i * 0.2 for i in range(1, 11)]    # 1.200 .. 3.000
        + [3.5, 4.0, 5.0, 6.5, 8.0, 10.0]
        + [13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0]
    )
})
# JXL q-equivalent (monotone axis): UNROUNDED float, k=4 so the full
# representable distance range [0, 25] maps to q [100, 0] — d=0→100,
# d=0.025→99.9, d=25→0. Unrounded preserves the 0.025-distance granularity
# (0.1 q apart) the old rounded k=7 map collapsed.
JXL_Q_K = 4.0

def jxl_q_equiv(d: float) -> float:
    return max(0.0, min(100.0, 100.0 - JXL_Q_K * float(d)))

Q_CODECS = [("zenjpeg", "jpeg"), ("zenwebp", "webp"), ("zenavif", "avif")]


def run_sweep(codec, q_grid, knob_grid, tag):
    feat = f"{OUTDIR}/{tag}_feat.parquet"
    tsv = f"{OUTDIR}/{tag}.tsv"
    cmd = [ZM, "sweep", "--codec", codec, "--sources", SOURCES,
           "--q-grid", q_grid, "--metric", "zensim-gpu",
           "--zensim-features-regime", "with-iw",
           "--feature-output", feat, "--output", tsv]
    if knob_grid:
        cmd += ["--knob-grid", knob_grid]
    print(f"  sweep {codec} ({tag}) q-grid={q_grid[:40]}{'...' if len(q_grid)>40 else ''} knob={knob_grid}", file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    print(f"    {tail}", file=sys.stderr)
    return feat if os.path.exists(feat) else None


def main():
    print(f"q-grid ({len(QGRID)} values): {QGRID_STR}", file=sys.stderr)
    merged_rows = []  # (image_id, codec, q, [f0..f371])

    # q-parameterized codecs
    for codec, fam in Q_CODECS:
        feat = run_sweep(codec, QGRID_STR, None, fam)
        if not feat:
            print(f"  WARN: {fam} produced no parquet", file=sys.stderr); continue
        t = pq.read_table(feat)
        df = t.to_pandas()
        for _, row in df.iterrows():
            img = os.path.basename(str(row["image_path"])).rsplit(".", 1)[0]
            q = float(row["q"])
            feats = [float(row[f"feat_{i}"]) for i in range(372)]
            if all(np.isfinite(feats)):
                # codec_param = native quality; param_kind = "q"
                merged_rows.append((img, fam, q, q, "q", feats))

    # JXL by distance (single dummy q=50, distance knob ladder)
    knob = json.dumps({"distance": JXL_DISTANCES})
    feat = run_sweep("zenjxl", "50", knob, "jxl")
    if feat:
        t = pq.read_table(feat)
        df = t.to_pandas()
        for _, row in df.iterrows():
            img = os.path.basename(str(row["image_path"])).rsplit(".", 1)[0]
            knob_json = str(row["knob_tuple_json"])
            try:
                d = json.loads(knob_json).get("distance", None)
            except Exception:
                d = None
            if d is None:
                continue
            q_equiv = jxl_q_equiv(float(d))
            feats = [float(row[f"feat_{i}"]) for i in range(372)]
            if all(np.isfinite(feats)):
                # codec_param = native butteraugli distance; param_kind = "distance".
                # q (monotone axis) = q_equiv so the sweep sorts by quality.
                merged_rows.append((img, "jxl", q_equiv, float(d), "distance", feats))

    print(f"\nmerged valid rows: {len(merged_rows)}", file=sys.stderr)
    from collections import Counter
    print("  per-codec:", dict(Counter(r[1] for r in merged_rows)), file=sys.stderr)

    # Consolidated dial-grid parquet (the canonical artifact bake_verdict reads):
    # image_id, codec, q (monotone axis), codec_param (native q/distance),
    # param_kind, f0..f371.
    import pyarrow as pa
    cols = {"image_id": [], "codec": [], "q": [], "codec_param": [], "param_kind": []}
    for i in range(372):
        cols[f"f{i}"] = []
    for img, codec, q, param, kind, feats in merged_rows:
        cols["image_id"].append(img); cols["codec"].append(codec)
        cols["q"].append(float(q)); cols["codec_param"].append(float(param))
        cols["param_kind"].append(kind)
        for i in range(372):
            cols[f"f{i}"].append(float(feats[i]))
    sc = {}
    for k, v in cols.items():
        if k in ("image_id", "codec", "param_kind"):
            sc[k] = pa.array(v)
        elif k in ("q", "codec_param"):
            sc[k] = pa.array(v, type=pa.float64())
        else:
            sc[k] = pa.array(v, type=pa.float32())
    grid_out = f"{OUTDIR}/dial_grid_372col.parquet"
    pq.write_table(pa.table(sc), grid_out, compression="zstd", compression_level=15)
    print(f"wrote {grid_out} ({len(merged_rows)} rows)", file=sys.stderr)
    print("Upload to R2 + point bake_verdict at it (--dial-grid / ZENSIM_DIAL_GRID):",
          file=sys.stderr)
    print("  aws s3 cp <grid> s3://zentrain/eval-grids/dial_grid_372col_<date>.parquet --endpoint-url <ep>",
          file=sys.stderr)


if __name__ == "__main__":
    main()
