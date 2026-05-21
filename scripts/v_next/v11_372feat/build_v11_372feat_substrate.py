#!/usr/bin/env python3
"""V11-372feat substrate builder (task #192, 2026-05-20).

Companion to `extract_features_372col_omni` (zensim-bench/examples/).
The Rust extractor produces a parquet whose rows already carry both
the 372 features and the GPU metric scores (ssim2 / cvvdp / iwssim /
butter / dssim / zensim). This script does NOT join omni + features
separately — they're pre-joined. It:

1. Renames `f0..f371` → `feat_0..feat_371` (legacy convention).
2. Calls `build_v11_substrate_v2.main()` indirectly by replicating its
   anchor + cross-codec equivalence logic, but parameterised at
   n_features=372.

## Inputs

- `--input-parquet`: output of `extract_features_372col_omni`.
  Default: `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/multi_codec_372col_partial.parquet`

## Outputs

- `<out-dir>/anchors_ssim2_372col_v3.parquet` (V5 multi-band schema with
  human_score column + f0..f371 features)
- `<out-dir>/cross_codec_equivalence_ssim2_372col_v3.parquet` (pairs
  across codec_a/codec_b for same image at the same ssim2 pivot)

## Schema differences vs V11-V2 substrate

- Feature count: 372 (vs 300). f0..f299 are basic+peaks+masked, f300..f371
  are IW-pool block.
- Source data: the extractor uses zensim's own 372-feat extraction
  pipeline. The omni sidecar's `score_zensim_gpu` was computed by the
  GPU `zensim-gpu` crate (no IW block — 300 feat); our new
  CPU-extracted `score_*` values stay aligned with the omni since we
  carry them through, not recompute.
- Coverage: this session's run only extracts zenjpeg + zenwebp
  (62,600 rows out of 117,800). AVIF + JXL are skipped because
  image-0.25 lacks decoders for them. Documented gap.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the V11-V2 band definitions + helpers
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "v11_ssim2_v2"))
from build_v11_substrate_v2 import (  # noqa: E402
    ANCHOR_BANDS_V11_V2,
    build_anchor_rows,
    build_cross_codec_equivalence,
    write_anchor_parquet,
    write_equivalence_parquet,
)


def rename_f_to_feat(df: pd.DataFrame, n_features: int = 372) -> pd.DataFrame:
    """Rename f{i} → feat_{i} so the V11-V2 substrate helpers work."""
    rename_map = {f"f{i}": f"feat_{i}" for i in range(n_features)}
    return df.rename(columns=rename_map)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/multi_codec_372col_partial.parquet"),
        help="output of extract_features_372col_omni",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v11-substrate"),
    )
    parser.add_argument(
        "--n-features", type=int, default=372,
    )
    parser.add_argument(
        "--ssim2-tolerance", type=float, default=3.0,
    )
    parser.add_argument(
        "--ssim2-eq-tolerance", type=float, default=3.0,
    )
    parser.add_argument(
        "--skip-equivalence", action="store_true",
    )
    parser.add_argument(
        "--out-version",
        default="v3",
        help="version tag in output filenames: anchors_ssim2_372col_<TAG>.parquet etc."
             " v3=partial-coverage from 62.6k cells (2026-05-20),"
             " v4=full 4-codec-coverage from 117.8k cells after V11-DECODER-FIX.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading 372-feat extracted parquet ===")
    print(f"  {args.input_parquet}")
    df = pq.read_table(args.input_parquet).to_pandas()
    print(f"  rows: {len(df):,}")
    print(f"  codec counts: {df['codec'].value_counts().to_dict()}")
    if "ref_basename" not in df.columns:
        df["ref_basename"] = df["image_path"].apply(os.path.basename)
    df = rename_f_to_feat(df, args.n_features)

    # Phase 2: anchor rows
    print()
    df_anchors = build_anchor_rows(
        df, ANCHOR_BANDS_V11_V2, args.n_features, tolerance=args.ssim2_tolerance
    )
    write_anchor_parquet(
        df_anchors,
        args.out_dir / f"anchors_ssim2_372col_{args.out_version}.parquet",
        args.n_features,
    )

    # Phase 3: cross-codec equivalence pairs
    if not args.skip_equivalence:
        print()
        pivot_levels = [90.0, 75.0, 60.0, 45.0, 30.0, 18.0]
        df_pairs = build_cross_codec_equivalence(
            df, pivot_levels, args.ssim2_eq_tolerance, args.n_features
        )
        write_equivalence_parquet(
            df_pairs,
            args.out_dir / f"cross_codec_equivalence_ssim2_372col_{args.out_version}.parquet",
            args.n_features,
        )


if __name__ == "__main__":
    main()
