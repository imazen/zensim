#!/usr/bin/env python3
"""Join zenanalyze 108-feature source table into each per-codec parquet
and emit 80/20 train/val splits.

Outputs:
  {out_dir}/<codec>_with_features.parquet
  {out_dir}/<codec>_train.parquet
  {out_dir}/<codec>_val.parquet

Split policy: per-source 80/20 (deterministic seeded shuffle of distinct
ref_basenames, then all rows for the source go to one side). This keeps
every q for a given source on the same side — no per-row leakage.
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def split_basenames(basenames, train_frac, seed):
    rng = random.Random(seed)
    uniq = sorted(set(basenames))
    rng.shuffle(uniq)
    n_train = int(round(train_frac * len(uniq)))
    train = set(uniq[:n_train])
    val = set(uniq[n_train:])
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features",
        default="/mnt/v/zen/picker-training/2026-05-19/sources_zenanalyze_features.parquet",
    )
    ap.add_argument(
        "--codec-dir", default="/mnt/v/zen/picker-training/2026-05-19/"
    )
    ap.add_argument(
        "--codecs",
        default="zenjpeg,zenwebp,zenavif,zenjxl,zenpng",
    )
    ap.add_argument("--out-dir", default="/mnt/v/zen/picker-training/2026-05-19/splits/")
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=20260519)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_table = pq.read_table(args.features)
    n_feat_cols = len(feat_table.column_names) - 1  # minus ref_basename
    print(f"features: {feat_table.num_rows} sources × {n_feat_cols} feat cols")

    codecs = args.codecs.split(",")
    summary = []

    for codec in codecs:
        codec_path = Path(args.codec_dir) / f"{codec}.parquet"
        if not codec_path.exists():
            print(f"  SKIP {codec}: {codec_path} not found")
            continue

        t = pq.read_table(codec_path)
        print(f"\n{codec}: {t.num_rows} rows, cols={t.column_names}")

        # JOIN on ref_basename
        joined = t.join(feat_table, keys="ref_basename", join_type="inner")
        print(f"  joined: {joined.num_rows} rows, {len(joined.column_names)} cols")

        # Sanity: scores
        scores = joined.column("achieved_zensim_tuner").to_numpy(zero_copy_only=False)
        import numpy as np
        valid = np.isfinite(scores)
        print(
            f"  zensim_tuner: min={np.nanmin(scores):.2f} mean={np.nanmean(scores):.2f}"
            f" max={np.nanmax(scores):.2f} valid={valid.sum()}/{len(scores)}"
        )

        # Persist combined
        combined_out = out_dir / f"{codec}_with_features.parquet"
        pq.write_table(joined, combined_out, compression="zstd")
        print(f"  wrote {combined_out}")

        # 80/20 split per source basename
        basenames = joined.column("ref_basename").to_pylist()
        train_set, val_set = split_basenames(basenames, args.train_frac, args.seed)
        mask_train = pc.is_in(joined["ref_basename"], value_set=pa.array(sorted(train_set)))
        mask_val = pc.is_in(joined["ref_basename"], value_set=pa.array(sorted(val_set)))
        train_t = joined.filter(mask_train)
        val_t = joined.filter(mask_val)

        train_out = out_dir / f"{codec}_train.parquet"
        val_out = out_dir / f"{codec}_val.parquet"
        pq.write_table(train_t, train_out, compression="zstd")
        pq.write_table(val_t, val_out, compression="zstd")
        print(
            f"  split: train={train_t.num_rows} ({len(train_set)} sources)"
            f"  val={val_t.num_rows} ({len(val_set)} sources)"
        )
        summary.append({
            "codec": codec,
            "joined_rows": joined.num_rows,
            "train_rows": train_t.num_rows,
            "val_rows": val_t.num_rows,
            "train_sources": len(train_set),
            "val_sources": len(val_set),
            "score_min": float(np.nanmin(scores)),
            "score_mean": float(np.nanmean(scores)),
            "score_max": float(np.nanmax(scores)),
            "score_valid": int(valid.sum()),
            "score_total": int(len(scores)),
        })

    # write summary
    import json
    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps({
        "features_path": args.features,
        "n_feat_cols": n_feat_cols,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "codecs": summary,
    }, indent=2))
    print(f"\nsummary: {summary_path}")
    for s in summary:
        print(
            f"  {s['codec']:>8}: {s['joined_rows']:>5} rows"
            f"  ({s['train_sources']}/{s['val_sources']} src)"
            f"  zensim mean={s['score_mean']:.1f}"
        )


if __name__ == "__main__":
    sys.exit(main())
