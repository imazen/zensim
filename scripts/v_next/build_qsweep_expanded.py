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

ZM = "/home/lilith/work/zen/zenmetrics/target/release/zen-metrics"
SOURCES = "/tmp/qsweep_sources"
OUTDIR = "/mnt/v/output/zensim/qsweep_expanded_2026-05-29"
os.makedirs(OUTDIR, exist_ok=True)

# Expanded q-grid for q-parameterized codecs (jpeg/webp/avif):
# low (existing) + JND-zone densified (70..90 step ~2) + near-lossless step-1 (90..100) + q0.
QGRID = sorted(set(
    [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]            # low/mid (existing coarse)
    + list(range(70, 91, 2))                          # JND zone: 70,72,...,90 (step 2)
    + list(range(90, 101, 1))                         # near-lossless: 90..100 (step 1)
    + [87, 89]                                        # keep prior near-JND points
))
QGRID_STR = ",".join(str(q) for q in QGRID)

# JXL butteraugli-distance ladder (near-lossless dense → low quality coarse).
JXL_DISTANCES = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0]

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
                merged_rows.append((img, fam, q, feats))

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
            q_equiv = max(0.0, min(100.0, round(100.0 - 7.0 * float(d))))
            feats = [float(row[f"feat_{i}"]) for i in range(372)]
            if all(np.isfinite(feats)):
                merged_rows.append((img, "jxl", q_equiv, feats))

    print(f"\nmerged valid rows: {len(merged_rows)}", file=sys.stderr)
    from collections import Counter
    print("  per-codec:", dict(Counter(r[1] for r in merged_rows)), file=sys.stderr)

    # Emit features CSV (ref_basename, human_score=q, f0..f371) + manifest TSV
    # (ref_path, dist_path, image_id, codec, q) in matching row order for qsweep_eval.
    feat_csv = f"{OUTDIR}/expanded_features.csv"
    manifest = f"{OUTDIR}/expanded_manifest.tsv"
    with open(feat_csv, "w") as fc, open(manifest, "w") as mf:
        fc.write("ref_basename,human_score," + ",".join(f"f{i}" for i in range(372)) + "\n")
        mf.write("ref_path\tdist_path\timage_id\tcodec\tq\n")
        for img, codec, q, feats in merged_rows:
            fc.write(f"{img},{q}," + ",".join(f"{v:.6g}" for v in feats) + "\n")
            mf.write(f"-\t-\t{img}\t{codec}\t{q}\n")
    print(f"wrote {feat_csv} + {manifest}", file=sys.stderr)
    print(f"\nNext: ./target/release/qsweep_eval --features {feat_csv} --manifest {manifest} \\", file=sys.stderr)
    print(f"        --bake A_v47=zensim/weights/v47_strict_qat_native_2026-05-27.bin:clamp \\", file=sys.stderr)
    print(f"        --bake Cell5=zensim/weights/v02_372feat_cell5_2026-05-28.bin:clamp \\", file=sys.stderr)
    print(f"        --out {OUTDIR}/dialreach_expanded.md", file=sys.stderr)


if __name__ == "__main__":
    main()
