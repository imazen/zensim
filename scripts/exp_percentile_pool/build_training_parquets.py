#!/usr/bin/env python3
"""EX-PERCENTILE-POOL: build P²-feature training parquets by joining
extracted 372-col P² CSVs with existing target columns from the
canonical V_24 training parquets.

Steps per corpus (kadid, tid, konjnd):
  1. Load existing /mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/<corpus>_mix_300col.parquet
     -- has ref_basename + targets + f0..f299 (L8 features).
  2. Load /mnt/v/zen/zensim-training/2026-05-18-percentile-pool/<corpus>_features_372col_p2.csv
     -- has ref_basename, human_score, f0..f371 (P² features).
  3. Take f0..f299 from P² CSV (basic + peak + masked, NO IW).
     Note: the 300-col canonical layout is basic+peak+masked too.
  4. Merge on ref_basename (positional alignment within ref_basename).
     The KADID/TID/konjnd CSVs preserve per-row identity via ref_basename
     ordering matching the loader.
  5. Write to /mnt/v/zen/zensim-training/2026-05-18-percentile-pool/<corpus>_mix_300col_p2.parquet.

For safesyn + cvvdp_iwssim_large, we skip — those need source paths
that aren't in the existing parquets. Training will use kadid+tid+konjnd
only (no safesyn) for the experiment.
"""

from pathlib import Path
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

CANONICAL_SRC = Path("/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer")
P2_DIR = Path("/mnt/v/zen/zensim-training/2026-05-18-percentile-pool")
OUT = P2_DIR

CORPORA = {
    "kadid": (CANONICAL_SRC / "kadid_mix_300col.parquet", P2_DIR / "kadid_features_372col_p2.csv"),
    "tid": (CANONICAL_SRC / "tid_mix_300col.parquet", P2_DIR / "tid_features_372col_p2.csv"),
    "konjnd": (CANONICAL_SRC / "konjnd_mix_300col.parquet", P2_DIR / "konjnd_features_372col_p2.csv"),
}


def build(name, src_parquet, p2_csv):
    print(f"[{name}] reading source: {src_parquet}")
    df_src = pq.read_table(src_parquet).to_pandas()
    print(f"[{name}]   src rows={len(df_src)} cols={len(df_src.columns)}")
    print(f"[{name}]   first cols: {df_src.columns[:10].tolist()}")
    print(f"[{name}]   last cols: {df_src.columns[-3:].tolist()}")

    print(f"[{name}] reading P² CSV: {p2_csv}")
    df_p2 = pd.read_csv(p2_csv, low_memory=False)
    print(f"[{name}]   p2 rows={len(df_p2)} cols={len(df_p2.columns)}")

    # Build refmap. The src parquet preserves ref_basename order from
    # the original extraction. The P² CSV is sorted alphabetically by
    # ref_basename (per the extractor). Just use ref_basename + position
    # within ref_basename as the join key.
    src_target_cols = [c for c in df_src.columns if not c.startswith("f")]
    print(f"[{name}]   target cols: {src_target_cols}")

    # Join by (ref_basename, position-within-ref). This requires both
    # frames to have rows in the SAME order per ref_basename. The
    # extractor in zensim-validate sorts within each ref by some order;
    # if order matches between L8 and P² CSVs, position-within-ref works.
    df_src["_rank_within_ref"] = df_src.groupby("ref_basename").cumcount()
    df_p2["_rank_within_ref"] = df_p2.groupby("ref_basename").cumcount()
    df_src["_join_key"] = df_src["ref_basename"].astype(str) + "::" + df_src["_rank_within_ref"].astype(str)
    df_p2["_join_key"] = df_p2["ref_basename"].astype(str) + "::" + df_p2["_rank_within_ref"].astype(str)

    # Extract P² features f0..f299
    p2_feat_cols = [f"f{i}" for i in range(300)]
    df_p2_feat = df_p2[["_join_key"] + p2_feat_cols]

    # Merge
    n_src = len(df_src)
    merged = df_src.drop(columns=p2_feat_cols, errors="ignore").merge(
        df_p2_feat, on="_join_key", how="inner"
    )
    print(f"[{name}]   merged: src={n_src} merged={len(merged)} loss={n_src-len(merged)}")
    merged = merged.drop(columns=["_rank_within_ref", "_join_key"])

    # Sanity: should have target cols + 300 features
    assert all(c in merged.columns for c in src_target_cols), "missing target cols"
    assert all(f"f{i}" in merged.columns for i in range(300)), "missing P² features"

    out_path = OUT / f"{name}_mix_300col_p2.parquet"
    pq.write_table(pa.Table.from_pandas(merged), out_path, compression="zstd")
    print(f"[{name}] wrote {out_path} ({len(merged)} rows)")


def main():
    for name, (src, p2) in CORPORA.items():
        if not src.exists():
            print(f"[{name}] WARNING: src parquet missing {src}; skipping")
            continue
        if not p2.exists():
            print(f"[{name}] WARNING: P² csv missing {p2}; skipping")
            continue
        build(name, src, p2)
    print()
    print("DONE — training parquets at:")
    for name in CORPORA:
        out = OUT / f"{name}_mix_300col_p2.parquet"
        if out.exists():
            print(f"  {out}")


if __name__ == "__main__":
    main()
