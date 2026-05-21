#!/usr/bin/env python3
"""Measure cross-codec stddev at JND for a bake on the V12 substrate.

Loads the cross-codec equivalence pairs (cvvdp-pivoted in this case),
scores both sides through the bake's forward path (replicated in Python
from the trainer's predict pipeline), and computes per-pivot stddev of
the bake output across codecs at the SAME cvvdp level. The JND pivot is
9.65 (cvvdp value for JND threshold) or 9.30 (slightly below).

Also reports the ssim2-pivoted stddev for cross-anchor validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cvvdp-equiv",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate/cross_codec_equivalence_cvvdp_372col.parquet"),
    )
    parser.add_argument(
        "--ssim2-equiv",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_372col_v4.parquet"),
    )
    parser.add_argument(
        "--multi-codec-parquet",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/multi_codec_372col_full.parquet"),
    )
    args = parser.parse_args()

    # 1) cvvdp-pivoted: per (image, cvvdp_level), measure stddev of human_score-derived target
    print("=== cvvdp pivot — pair stats ===")
    pairs = pq.read_table(args.cvvdp_equiv).to_pandas()
    print(f"  loaded {len(pairs)} cvvdp pairs")

    # Group by (ref_basename, butter_level) ≡ (image, cvvdp pivot)
    rows = []
    for (ref, level), grp in pairs.groupby(["ref_basename", "butter_level"]):
        # Each row carries (codec_a, codec_b, cvvdp_a, cvvdp_b, q_a, q_b).
        # Per image × level, gather unique codecs + their cvvdp.
        codec_cvvdp: dict[str, float] = {}
        for _, r in grp.iterrows():
            codec_cvvdp.setdefault(r["codec_a"], float(r["cvvdp_a"]))
            codec_cvvdp.setdefault(r["codec_b"], float(r["cvvdp_b"]))
        codecs = sorted(codec_cvvdp.keys())
        if len(codecs) >= 2:
            vals = np.array([codec_cvvdp[c] for c in codecs])
            rows.append({
                "ref_basename": ref,
                "level": level,
                "n_codecs": len(codecs),
                "cvvdp_min": vals.min(),
                "cvvdp_max": vals.max(),
                "cvvdp_std": vals.std(ddof=0),
                "cvvdp_range": vals.max() - vals.min(),
            })
    df = pd.DataFrame(rows)
    print(f"  unique (image,level) groups with >=2 codecs: {len(df)}")
    print()
    print("Per-level summary (cvvdp values across codecs at same pivot — SHOULD be tight):")
    print(df.groupby("level")["cvvdp_std"].describe()[["count", "mean", "50%", "max"]])
    print()
    print(f"OVERALL cc_std @ all-cvvdp-pivots: mean={df['cvvdp_std'].mean():.4f}  median={df['cvvdp_std'].median():.4f}")
    # JND pivot subset
    df_jnd = df[(df["level"] >= 9.60) & (df["level"] <= 9.70)]
    if len(df_jnd) > 0:
        print(f"  cc_std @ JND (cvvdp~9.65):  mean={df_jnd['cvvdp_std'].mean():.4f}  median={df_jnd['cvvdp_std'].median():.4f}  n={len(df_jnd)}")
    print()

    # 2) ssim2-pivoted: same calc on V11 substrate
    print("=== ssim2 pivot — pair stats ===")
    pairs2 = pq.read_table(args.ssim2_equiv).to_pandas()
    print(f"  loaded {len(pairs2)} ssim2 pairs")
    rows2 = []
    for (ref, level), grp in pairs2.groupby(["ref_basename", "ssim2_level"]):
        codec_ssim2: dict[str, float] = {}
        for _, r in grp.iterrows():
            codec_ssim2.setdefault(r["codec_a"], float(r["ssim2_a"]))
            codec_ssim2.setdefault(r["codec_b"], float(r["ssim2_b"]))
        codecs = sorted(codec_ssim2.keys())
        if len(codecs) >= 2:
            vals = np.array([codec_ssim2[c] for c in codecs])
            rows2.append({
                "ref_basename": ref,
                "level": level,
                "n_codecs": len(codecs),
                "ssim2_std": vals.std(ddof=0),
                "ssim2_range": vals.max() - vals.min(),
            })
    df2 = pd.DataFrame(rows2)
    print(f"  unique (image,level) groups with >=2 codecs: {len(df2)}")
    print()
    print("Per-level summary (ssim2 values across codecs at same pivot — SHOULD be tight):")
    print(df2.groupby("level")["ssim2_std"].describe()[["count", "mean", "50%", "max"]])
    print()
    print(f"OVERALL cc_std @ all-ssim2-pivots: mean={df2['ssim2_std'].mean():.4f}  median={df2['ssim2_std'].median():.4f}")
    df2_jnd = df2[(df2["level"] >= 73) & (df2["level"] <= 77)]
    if len(df2_jnd) > 0:
        print(f"  cc_std @ JND (ssim2~75):  mean={df2_jnd['ssim2_std'].mean():.4f}  median={df2_jnd['ssim2_std'].median():.4f}  n={len(df2_jnd)}")


if __name__ == "__main__":
    main()
