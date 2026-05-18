#!/usr/bin/env python3
"""
Build the densified KonJND-1k training parquet for V_24 densification
experiment (2026-05-18).

Sources
-------
- `/mnt/v/zen/zensim-training/2026-05-15-full-features/konjnd_full_features_372col_2026-05-15.csv`
  contains 76,104 rows = 504 sources × 100 JPEG levels + 504 sources × 51
  BPG levels. Each row has `ref_basename` (e.g. `SRC0001`), `human_score`
  (= `gpu_ssimulacra2 / 100`, range ~-0.65..0.96), and 372 zensim
  features. The features were extracted via
  `zensim-bench/examples/extract_features_372col.rs` (load_konjnd_full).

Output
------
- `/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/konjnd_dense_features_mix_targets_372col.parquet`
  with N rows ~ 20k. Schema matches the legacy 300col konjnd parquet at
  `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet`:
      ref_basename, human_score, f0..f371, mix_cv25_iw75, ..., mix_cv75_iw25
  where every mix_* column is a copy of `human_score` (no real mix
  signal — konjnd's anchor IS the SSIM2-aligned per-pair score).

The trainer applies `--target-scale 100.0` at load time so the runtime
score range is 0..~96 and the legacy PJND-pair-weighting threshold=45
remains semantically meaningful.

Sampling strategy
-----------------
Goal: 20k rows. Two design constraints:

1. Trainer samples pairs uniformly within the group, so per-source
   distortion-level density matters more than total row count. We
   sample ~20 distortion levels per source uniformly across the
   quality ladder.

2. Both JPEG (504 sources × 100 levels) and BPG (504 sources × 51
   levels) halves should remain balanced. JPEG sources and BPG
   sources are disjoint, so total = 1008 sources × ~20 levels =
   20,160 rows.

Per-source sampling: rank that source's rows by `human_score`
(ssim2/100), then take every N/k-th rank to span the JND ladder
evenly. This preserves the full range (worst..best distortion)
and concentrates samples in the typical 5..95 ssim2 region where
PJND-weighting actually fires.
"""
import argparse
import csv
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC_CSV = Path(
    "/mnt/v/zen/zensim-training/2026-05-15-full-features/"
    "konjnd_full_features_372col_2026-05-15.csv"
)
OUT_PARQUET = Path(
    "/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/"
    "konjnd_dense_features_mix_targets_372col.parquet"
)
DEFAULT_LEVELS_PER_SRC = 20

MIX_COLS = [
    "mix_cv40_iw60", "mix_cv25_iw75", "mix_cv35_iw65", "mix_cv45_iw55",
    "mix_cv50_iw50", "mix_cv55_iw45", "mix_cv60_iw40", "mix_cv65_iw35",
    "mix_cv30_iw70", "mix_cv70_iw30", "mix_cv75_iw25",
]


def stratified_subsample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """For each ref_basename, take k rows spanning the human_score ladder.

    Sort by human_score (ascending = worst quality first). Pick k
    evenly-spaced ranks. If a source has fewer than k rows, keep all.
    """
    out = []
    for ref, g in df.groupby("ref_basename", sort=False):
        n = len(g)
        if n <= k:
            out.append(g)
            continue
        g_sorted = g.sort_values("human_score", kind="stable").reset_index(drop=True)
        # k evenly-spaced indices including both ends
        idx = np.round(np.linspace(0, n - 1, k)).astype(int)
        idx = np.unique(idx)
        out.append(g_sorted.iloc[idx])
    return pd.concat(out, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels-per-src", type=int, default=DEFAULT_LEVELS_PER_SRC,
                    help="approximate distortion levels kept per source (default %(default)s)")
    ap.add_argument("--in-csv", type=Path, default=SRC_CSV)
    ap.add_argument("--out", type=Path, default=OUT_PARQUET)
    args = ap.parse_args()

    if not args.in_csv.exists():
        print(f"ERROR: {args.in_csv} not found", file=sys.stderr)
        return 1

    print(f"reading {args.in_csv}", file=sys.stderr)
    df = pd.read_csv(args.in_csv)
    print(f"  loaded {len(df)} rows × {len(df.columns)} cols", file=sys.stderr)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    assert len(feat_cols) == 372, f"expected 372 features, got {len(feat_cols)}"
    assert "ref_basename" in df.columns and "human_score" in df.columns

    print(f"  unique sources: {df['ref_basename'].nunique()}", file=sys.stderr)
    print(f"  human_score range: [{df['human_score'].min():.4f}, {df['human_score'].max():.4f}]",
          file=sys.stderr)
    print(f"  human_score mean: {df['human_score'].mean():.4f} median: {df['human_score'].median():.4f}",
          file=sys.stderr)

    print(f"stratified subsample: {args.levels_per_src} levels per source", file=sys.stderr)
    sub = stratified_subsample(df, args.levels_per_src)
    print(f"  → {len(sub)} rows", file=sys.stderr)

    # Rescale `human_score` from ssim2/100 (range ~-0.65..0.96) to ssim2
    # 0..100 scale (range ~-65..96) to MATCH the legacy 1008-row konjnd
    # parquet's scale (where `human_score` was the PJND threshold in
    # 22..70 ssim2 units). The trainer applies `--target-scale 100.0` by
    # default, so the in-trainer scale ends up at 0..10000 either way —
    # but pre-scaling here keeps the unscaled parquet semantics consistent
    # with the legacy file (auditable median ~63 = the ssim2 PJND threshold
    # per CID22 paper Table 4).
    sub["human_score"] = sub["human_score"] * 100.0

    # Materialize mix_* columns as copies of human_score (konjnd's anchor
    # is the SSIM2-aligned per-pair score; no real mix signal exists).
    for col in MIX_COLS:
        sub[col] = sub["human_score"]

    # Column order: ref_basename, human_score, f0..f371, mix_*
    ordered = ["ref_basename", "human_score"] + feat_cols + MIX_COLS
    sub = sub[ordered]
    print(f"  final cols: {len(sub.columns)} (expected {2 + 372 + 11} = 385)",
          file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(sub, preserve_index=False)
    pq.write_table(table, args.out, compression="zstd")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, "
          f"{len(sub)} rows × {len(sub.columns)} cols)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
