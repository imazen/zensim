#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V4 multi-codec PJND anchor builder (2026-05-19).

For every (source, codec) pair in the per-codec butter parquets, picks the
q whose butter_pnorm3 is closest to PJND (1.5), then emits one row per
(source, codec) at that q. All rows target the same score = 63.0 (CID22
paper Table 4 PJND anchor). This binds the score=63 ↔ PJND mapping ACROSS
codecs — V3 only had a single-codec anchor (zenjpeg), which is why V3
bakes diverged cross-codec at T=63.

Output schema mirrors the V3 anchor parquet:
  ref_basename, anchor_source, human_score, anchor_weight,
  pjnd_q, butter_pnorm3, codec, f0..f371

The trainer loads via `--anchor-parquet`; it ignores all columns except
`anchor_weight` (used as `human_score` slot) and `f0..f<n_features-1>`.
The trainer's `--anchor-target-score 63.0` overrides the human_score
column. Setting `anchor_weight = 1.0` per row gives equal weighting
across (source, codec) pairs.

Run with:
    python3 scripts/v_next/build_multi_codec_pjnd_anchors.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PJND_PIVOT = 1.5  # butter_pnorm3 at PJND threshold (per CORRECTION_FROM_PARENT)
TARGET_SCORE = 63.0  # CID22 paper Table 4 PJND ssim2 anchor

CODECS = ["zenjpeg", "zenwebp", "zenavif", "zenjxl"]
BUTTER_DIR = Path("/mnt/v/zen/picker-training/2026-05-19/butter")
OUT_PATH = Path(
    "/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/anchors_multi_codec_372col.parquet"
)


def build() -> None:
    rows = []
    feature_cols = [f"f{i}" for i in range(372)]

    for codec in CODECS:
        path = BUTTER_DIR / f"{codec}.parquet"
        if not path.exists():
            print(f"  skip {codec}: parquet missing at {path}")
            continue
        df = pq.read_table(path).to_pandas()
        print(f"{codec}: loaded {len(df)} rows ({df['ref_basename'].nunique()} sources)")

        # Per-source: argmin |butter_pnorm3 - PJND_PIVOT|
        for source, group in df.groupby("ref_basename"):
            distances = (group["butter_pnorm3"] - PJND_PIVOT).abs()
            idx = distances.idxmin()
            best = group.loc[idx]

            row = {
                "ref_basename": str(source),
                "anchor_source": f"{codec}_pjnd",
                "human_score": TARGET_SCORE,  # placeholder; trainer overrides
                "anchor_weight": 1.0,
                "pjnd_q": int(best["q"]),
                "butter_pnorm3": float(best["butter_pnorm3"]),
                "codec": codec,
            }
            for col in feature_cols:
                if col in best.index:
                    val = best[col]
                    row[col] = float(val) if val is not None else 0.0
                else:
                    row[col] = 0.0
            rows.append(row)

    print(f"\ntotal anchor rows: {len(rows)}")
    if not rows:
        raise SystemExit("no anchor rows built")

    # Build pyarrow table
    cols = {
        "ref_basename": pa.array([r["ref_basename"] for r in rows], type=pa.string()),
        "anchor_source": pa.array([r["anchor_source"] for r in rows], type=pa.string()),
        "human_score": pa.array([r["human_score"] for r in rows], type=pa.float64()),
        "anchor_weight": pa.array([r["anchor_weight"] for r in rows], type=pa.float64()),
        "pjnd_q": pa.array([r["pjnd_q"] for r in rows], type=pa.int64()),
        "butter_pnorm3": pa.array([r["butter_pnorm3"] for r in rows], type=pa.float64()),
        "codec": pa.array([r["codec"] for r in rows], type=pa.string()),
    }
    for col in feature_cols:
        cols[col] = pa.array([r[col] for r in rows], type=pa.float32())
    tbl = pa.table(cols)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, OUT_PATH, compression="zstd", compression_level=15)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KiB, {tbl.num_rows} rows × {tbl.num_columns} cols)")

    # Print summary stats
    print("\nper-codec stats:")
    print(f"  codec     n    median q    median butter_pnorm3")
    print(f"  -----     -    --------    --------------------")
    for codec in CODECS:
        sub = [r for r in rows if r["codec"] == codec]
        if not sub:
            continue
        qs = sorted(r["pjnd_q"] for r in sub)
        bs = sorted(r["butter_pnorm3"] for r in sub)
        med_q = qs[len(qs) // 2]
        med_b = bs[len(bs) // 2]
        print(f"  {codec:10s} {len(sub):4d}  {med_q:5d}       {med_b:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    if args.out != OUT_PATH:
        # Allow override
        globals()["OUT_PATH"] = args.out
    build()
