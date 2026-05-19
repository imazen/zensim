#!/usr/bin/env python3
"""Build cross-codec JND-anchor parquet for PreviewV0_5TunerV2 (2026-05-19).

The product question: "user types score 70 — does it mean the same perceptual
quality across codecs?" Today's Tuner trains on safesyn ssim2-shaped target
which is internally consistent within a codec curve but doesn't ENFORCE that
PJND lands at the same score across codecs. Cross-codec eval
(`benchmarks/cross_codec_consistency_2026-05-19.md`) measured 6.68 butter
mean pairwise at T=63 — the PJND operating point.

This script builds the anchor dataset:

  1. **KonJND-1k PJND anchors** (real human data). The canonical
     val/konjnd.parquet has 1008 rows (504 JPEG + 504 BPG SRC0001..SRC1008)
     where `human_score` = PJND mean q at which raters cannot tell source
     vs distorted apart. Each row's features f0..f371 are extracted from
     the (SRC, distorted-at-q-PJND) pair. The natural target score for
     these pairs is **63** (CID22 paper Table 4 calibrates PJND ≈ 63 on
     ssim2 scale). KonJND covers JPEG + BPG natively.

  2. **Synthetic PJND anchors** for codecs without KonJND data. For each
     row in safesyn where ssim2_gpu ∈ [pjnd_low, pjnd_high] (default
     [61, 65], the ±1-score-unit band around 63), keep the row as a
     synthetic-PJND anchor — features computed from the pair where
     ssim2 ≈ 63 already.

The output parquet has columns:
  - ref_basename, anchor_source ('konjnd' | 'safesyn')
  - human_score = 63 (constant; this is the target the trainer regresses to)
  - anchor_weight (per-row weight; default 1.0 for safesyn, 1.5 for KonJND
    since human data is the gold standard for PJND)
  - f0..f371 features (same 372-feat layout as canonical train parquets)

Output: /mnt/v/zen/zensim-training/2026-05-19-jnd-anchors/anchors_372col.parquet

The trainer reads this via `--anchor-parquet <path> --anchor-loss-weight W`.
For each training step (or every N steps), the trainer samples a row from
this parquet, forwards through the per-sample-α head, and adds
W·(y - 63)² to the loss.
"""

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def build_konjnd_anchors(in_parquet, target_score):
    """Read val/konjnd.parquet, return DataFrame with anchor schema."""
    t = pq.read_table(in_parquet)
    df = t.to_pandas()
    print(f"  loaded {len(df)} KonJND rows from {in_parquet}", flush=True)
    # KonJND ordering: SRC0001..SRC0504 = JPEG, SRC0505..SRC1008 = BPG
    def codec_for(b):
        if not isinstance(b, str) or not b.startswith("SRC"):
            return "unknown"
        try:
            n = int(b[3:7])
        except Exception:
            return "unknown"
        return "jpeg" if n <= 504 else "bpg"
    out = df[["ref_basename"] + [f"f{i}" for i in range(372)]].copy()
    out["anchor_source"] = df.ref_basename.apply(lambda b: f"konjnd_{codec_for(b)}")
    out["human_score"] = float(target_score)
    # 1.5x weight: real human PJND is the gold standard
    out["anchor_weight"] = 1.5
    out["pjnd_q"] = df.human_score  # original PJND q
    out["ssim2_gpu"] = float("nan")
    return out


def build_safesyn_anchors(
    in_parquet, target_score, ssim2_low, ssim2_high, max_rows
):
    """Read safesyn.parquet; pull rows with ssim2_gpu in [low, high] as synthetic PJND anchors."""
    cols_to_load = ["ref_basename", "ssim2_gpu"] + [f"f{i}" for i in range(372)]
    t = pq.read_table(in_parquet, columns=cols_to_load)
    df = t.to_pandas()
    print(f"  loaded {len(df)} safesyn rows", flush=True)
    mask = (df.ssim2_gpu >= ssim2_low) & (df.ssim2_gpu <= ssim2_high)
    near = df[mask].copy()
    print(
        f"  near-PJND (ssim2 ∈ [{ssim2_low}, {ssim2_high}]): {len(near)} rows from {near.ref_basename.nunique()} sources",
        flush=True,
    )
    if max_rows and len(near) > max_rows:
        # Stratified sample by ref_basename to preserve source diversity
        near = (
            near.groupby("ref_basename", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, max_rows // near.ref_basename.nunique())),
                    random_state=20260519,
                )
            )
        ).reset_index(drop=True)
        print(f"  stratified-sampled to {len(near)} rows", flush=True)

    out = near[["ref_basename", "ssim2_gpu"] + [f"f{i}" for i in range(372)]].copy()
    out["anchor_source"] = "safesyn_synth"
    out["human_score"] = float(target_score)
    out["anchor_weight"] = 1.0
    out["pjnd_q"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--konjnd-parquet",
        default="/mnt/v/zen/zensim-training/canonical-2026-05-18/val/konjnd.parquet",
    )
    ap.add_argument(
        "--safesyn-parquet",
        default="/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet",
    )
    ap.add_argument(
        "--out",
        default="/mnt/v/zen/zensim-training/2026-05-19-jnd-anchors/anchors_372col.parquet",
    )
    ap.add_argument("--target-score", type=float, default=63.0)
    ap.add_argument("--ssim2-low", type=float, default=61.0)
    ap.add_argument("--ssim2-high", type=float, default=65.0)
    ap.add_argument(
        "--max-safesyn-rows",
        type=int,
        default=10000,
        help="cap synthetic-PJND anchors at this count (stratified-sampled)",
    )
    ap.add_argument("--skip-konjnd", action="store_true")
    ap.add_argument("--skip-safesyn", action="store_true")
    args = ap.parse_args()

    print(f"# build_jnd_anchors target_score={args.target_score}", flush=True)

    parts = []
    if not args.skip_konjnd:
        print(f"# loading KonJND anchors from {args.konjnd_parquet}", flush=True)
        konjnd = build_konjnd_anchors(args.konjnd_parquet, args.target_score)
        parts.append(konjnd)
        print(
            f"  KonJND anchor counts: {konjnd.anchor_source.value_counts().to_dict()}",
            flush=True,
        )

    if not args.skip_safesyn:
        print(f"# loading safesyn synth-PJND anchors from {args.safesyn_parquet}", flush=True)
        safesyn = build_safesyn_anchors(
            args.safesyn_parquet,
            args.target_score,
            args.ssim2_low,
            args.ssim2_high,
            args.max_safesyn_rows,
        )
        parts.append(safesyn)

    import pandas as pd

    combined = pd.concat(parts, ignore_index=True, sort=False)
    # Reorder: meta first, then features
    feat_cols = [f"f{i}" for i in range(372)]
    meta_cols = [
        "ref_basename",
        "anchor_source",
        "human_score",
        "anchor_weight",
        "pjnd_q",
        "ssim2_gpu",
    ]
    keep = meta_cols + feat_cols
    combined = combined[keep]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), out_path, compression="zstd")
    print(f"\n# wrote {len(combined)} rows × {len(combined.columns)} cols → {out_path}", flush=True)
    print(f"# anchor_source breakdown:")
    for src, n in combined.anchor_source.value_counts().items():
        w = combined.loc[combined.anchor_source == src, "anchor_weight"].iloc[0]
        print(f"  {src}: {n} rows (anchor_weight={w})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
