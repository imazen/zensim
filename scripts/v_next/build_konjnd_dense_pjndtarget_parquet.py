#!/usr/bin/env python3
"""
Build the densified KonJND-1k training parquet WITH PJND-threshold
broadcast targets — the alternative supervision to the ssim2-target
densification.

This variant:
- Takes the same 20,160 (source, distorted) rows from the ssim2-target
  dense parquet
- Overwrites `human_score` (and all mix_* columns) with the per-source
  PJND THRESHOLD from the legacy 1008-row val parquet
  (`/mnt/v/zen/zensim-training/2026-05-15-full-features/konjnd_features_372col_2026-05-15.parquet`)
- The 20 rows for any given source ALL share the same target value
  (the source's mean PJND threshold from `subjective_ratings.csv`)

Why: the legacy KonJND val set uses per-source PJND threshold as the
SROCC ground truth. Training with this same encoding (broadcast across
20 distortion variants) keeps the train/val target shape aligned.

Caveat: see `benchmarks/v24_konjnd_densified_methodology_2026-05-18.md`
for the load-bearing finding — this variant recovers KonJND-1k val SROCC
to within 0.003 of V_22, but tanks CID22 by -0.21 because the
broadcast-target encoding starves per-pair gradient signal.

Inputs:
- `/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/konjnd_dense_features_mix_targets_300col.parquet`
  (20,160 rows × 313 cols, ssim2-target schema)
- `/mnt/v/zen/zensim-training/2026-05-15-full-features/konjnd_features_372col_2026-05-15.parquet`
  (1,008 rows, PJND-threshold targets per source)

Output:
- `/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/konjnd_dense_pjndtarget_300col.parquet`
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SSIM2_DENSE = Path(
    "/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/"
    "konjnd_dense_features_mix_targets_300col.parquet"
)
VAL_PJND = Path(
    "/mnt/v/zen/zensim-training/2026-05-15-full-features/"
    "konjnd_features_372col_2026-05-15.parquet"
)
OUT = Path(
    "/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/"
    "konjnd_dense_pjndtarget_300col.parquet"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssim2-dense", type=Path, default=SSIM2_DENSE)
    ap.add_argument("--val-pjnd", type=Path, default=VAL_PJND)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.ssim2_dense.exists():
        print(f"ERROR: {args.ssim2_dense} not found", file=sys.stderr)
        return 1
    if not args.val_pjnd.exists():
        print(f"ERROR: {args.val_pjnd} not found", file=sys.stderr)
        return 1

    dense = pq.read_table(args.ssim2_dense).to_pandas()
    val = pq.read_table(args.val_pjnd).to_pandas()
    # The val file's ref_basename has '.png' suffix (e.g. SRC0001.png) but
    # the dense file's ref_basename is the stem (e.g. SRC0001). Map them.
    val["ref_stem"] = val["ref_basename"].str.replace(".png", "", regex=False)
    pjnd_by_ref = val.set_index("ref_stem")["human_score"]

    n_before = len(dense)
    dense["human_score"] = dense["ref_basename"].map(pjnd_by_ref)
    n_nan = dense["human_score"].isna().sum()
    if n_nan > 0:
        print(f"WARNING: {n_nan}/{n_before} rows had no PJND match; dropping", file=sys.stderr)
        dense = dense.dropna(subset=["human_score"]).reset_index(drop=True)

    # Overwrite all mix_* columns with the broadcast PJND threshold so the
    # trainer's --target-column mix_cv40_iw60 picks up the same value.
    mix_cols = [c for c in dense.columns if c.startswith("mix_")]
    for c in mix_cols:
        dense[c] = dense["human_score"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(dense, preserve_index=False)
    pq.write_table(table, args.out, compression="zstd")
    print(
        f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, "
        f"{len(dense)} rows × {len(dense.columns)} cols)",
        file=sys.stderr,
    )
    print(
        f"  human_score: range=[{dense['human_score'].min():.2f}, "
        f"{dense['human_score'].max():.2f}], median {dense['human_score'].median():.2f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
