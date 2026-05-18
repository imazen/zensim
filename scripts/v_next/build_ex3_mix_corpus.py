#!/usr/bin/env python3
"""Build the V_24-mix-with-ssim2 training corpus by joining the
SSIMULACRA2 sidecars (LARGE corpus from EX-3 fleet) onto the
existing iwssim + cvvdp mix-target parquets and emitting new mix
targets.

## Inputs

1. **EX-3 fleet output** (ssim2-backfill-2026-05-18):
   - Per-chunk sidecars at `s3://zentrain/ssim2-backfill-2026-05-18/ssim2_imazen/`
     (~75,300 rows total, 754 chunks of 100 rows each)
   - Consolidated by `consolidate_ssim2.sh` to
     `/mnt/v/zen/zensim-training/2026-05-18-ssim2/ssim2_imazen_consolidated.parquet`

2. **Existing LARGE corpus**:
   - iwssim:  `/mnt/v/zen/zensim-training/2026-05-15-cvvdp-r2/iwssim_imazen_consolidated.parquet`
              (75,300 rows; `iwssim_imazen_v0_0_1` ∈ [0,1])
   - cvvdp:   `/mnt/v/zen/zensim-training/2026-05-15-cvvdp-r2/cvvdp_imazen_consolidated.parquet`
              (1,169,500 rows; `cvvdp_imazen_v0_0_1` ∈ [0,10] JOD)
   - features: `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_*_cvvdp.parquet`
               (300-feature input + identity tuple + per-corpus extras incl.
                `score_ssim2` and `score_zensim` as features)

3. **Existing 372-feature mix-target parquets** (for safesyn/KADID/TID):
   - `/mnt/v/zen/zensim-training/2026-05-17-cvvdp/{safesyn,kadid,tid}_features_mix_targets_372col.parquet`
   - Already have cvvdp + iwssim + mix_cv40_iw60. Need ssim2 added.

## Important — Anti-pattern #6 audit decision

The 300-feature merged corpus HAS `score_ssim2` and `score_zensim`
as per-pair feature columns (used as zenanalyze tier3 anchors).
**Using ssim2 as a target column on this corpus WITHOUT dropping
`score_ssim2` from features violates Anti-pattern #6** (don't use
the same metric as both target and feature).

The 372-feature safesyn/kadid/tid corpora do NOT have any ssim2
as feature (f0..f371 are pure zenanalyze image-property features),
so adding ssim2 as a target is safe there.

## Output

LARGE corpus output (where score_ssim2 is dropped from features):
  `/mnt/v/zen/zensim-training/2026-05-18-ssim2/large_3target_300feat_minus_ssim2.parquet`

  Schema: (basename, codec, q, knob_tuple_json, identity tuple)
        + 299 feature cols (300 minus score_ssim2)
        + iwssim_imazen_v0_0_1, iwssim_log_norm
        + cvvdp_imazen_v0_0_1, cvvdp_log_norm
        + ssim2_gpu, ssim2_log_norm
        + mix_cv33_iw33_sm33  (3-way blend, equal weights)

Safesyn/KADID/TID 4-target output (when local ssim2 scoring completes):
  `/mnt/v/zen/zensim-training/2026-05-18-ssim2/{safesyn,kadid,tid}_4target_372col.parquet`

  Schema: existing 372-feat + iwssim + cvvdp + ssim2
        + mix_cv33_iw33_sm33

## Status note (2026-05-18)

This script runs as soon as `ssim2_imazen_consolidated.parquet` lands.
The LARGE join is the primary EX-3 output. Safesyn ssim2 scoring on the
local workstation takes ~57hr (deferred); KADID+TID local scoring
takes ~30min and is in flight.
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DATA_DIR_FEAT_372 = "/mnt/v/zen/zensim-training/2026-05-17-cvvdp"
DATA_DIR_MERGED_300 = "/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged"
DATA_DIR_TARGETS = "/mnt/v/zen/zensim-training/2026-05-15-cvvdp-r2"
DATA_DIR_OUT = "/mnt/v/zen/zensim-training/2026-05-18-ssim2"

# Anchors from V_22-mix-LARGE methodology (commit 972201d)
# cvvdp lo=-2.1188 hi=13.8155, iwssim max_log=13.7202
# We re-anchor ssim2 from the actual sidecar distribution.
CVVDP_LO = -2.1188
CVVDP_HI = 13.8155


def log_norm_iwssim(x: np.ndarray) -> np.ndarray:
    """iwssim_log_norm: -log(1 - x + 1e-6), then min-max."""
    return -np.log(np.maximum(1.0 - x, 1e-6))


def log_norm_cvvdp(x: np.ndarray, lo: float = CVVDP_LO, hi: float = CVVDP_HI) -> np.ndarray:
    """cvvdp_log_norm: -log(10 - x + 1e-6), normalized to [0, 100]."""
    log_x = -np.log(np.maximum(10.0 - x, 1e-6))
    return np.clip((log_x - lo) / (hi - lo) * 100.0, 0.0, 100.0)


def log_norm_ssim2(x: np.ndarray) -> np.ndarray:
    """ssim2 native scale is roughly [-30, 100] (100=identical).

    Map to [0, 100] linearly: clip to [-30, 100] then rescale.
    """
    return np.clip((x + 30.0) / 130.0 * 100.0, 0.0, 100.0)


def build_large_3target(out_path: str, drop_ssim2_feature: bool = True) -> int:
    """Build LARGE 3-target corpus (cvvdp + iwssim + ssim2) on the 300-feat
    merged parquets. Drops score_ssim2 from features to avoid Anti-pattern #6.

    Returns row count of the inner join.
    """
    print("=== Building LARGE 3-target corpus ===")

    # Load all per-codec merged parquets
    merged_files = sorted(glob(os.path.join(DATA_DIR_MERGED_300, "unified_*_cvvdp.parquet")))
    print(f"  found {len(merged_files)} merged parquets")
    dfs = []
    for f in merged_files:
        df = pq.read_table(f).to_pandas()
        dfs.append(df)
    import pandas as pd
    merged = pd.concat(dfs, ignore_index=True, sort=False)
    merged["basename"] = merged["image_path"].apply(os.path.basename)
    print(f"  merged rows: {len(merged)}")

    # Load iwssim
    iw = pq.read_table(os.path.join(DATA_DIR_TARGETS, "iwssim_imazen_consolidated.parquet")).to_pandas()
    iw["basename"] = iw["image_path"].apply(os.path.basename)
    print(f"  iwssim rows: {len(iw)}")

    # Load ssim2
    ssim2_path = os.path.join(DATA_DIR_OUT, "ssim2_imazen_consolidated.parquet")
    if not os.path.isfile(ssim2_path):
        print(f"  MISSING: {ssim2_path} — run consolidate_ssim2.sh first")
        return 0
    sm = pq.read_table(ssim2_path).to_pandas()
    sm["basename"] = sm["image_path"].apply(os.path.basename)
    print(f"  ssim2 rows: {len(sm)}")

    # Inner-join all four on (basename, codec, q, knob_tuple_json)
    key = ["basename", "codec", "q", "knob_tuple_json"]
    j = merged.merge(
        iw[key + ["iwssim_imazen_v0_0_1"]],
        on=key,
        how="inner",
    )
    print(f"  merged ∩ iwssim: {len(j)}")
    j = j.merge(
        sm[key + ["ssim2_gpu"]],
        on=key,
        how="inner",
    )
    print(f"  merged ∩ iwssim ∩ ssim2: {len(j)}")

    # Add normalized targets
    j["iwssim_log_norm"] = log_norm_iwssim(j["iwssim_imazen_v0_0_1"].to_numpy()) * 100.0 / 13.7202
    j["iwssim_log_norm"] = np.clip(j["iwssim_log_norm"], 0.0, 100.0)
    j["cvvdp_log_norm"] = log_norm_cvvdp(j["cvvdp_imazen_v0_0_1"].to_numpy())
    j["ssim2_log_norm"] = log_norm_ssim2(j["ssim2_gpu"].to_numpy())

    # 3-way mix: equal weights (EX-3 § 8 recipe)
    j["mix_cv33_iw33_sm33"] = (
        j["cvvdp_log_norm"].to_numpy()
        + j["iwssim_log_norm"].to_numpy()
        + j["ssim2_log_norm"].to_numpy()
    ) / 3.0
    # Also emit the cv25_iw25_sm25_ext25 placeholder (just 3-way since
    # we don't have butteraugli backfilled yet — symmetric on 3 targets)
    j["mix_cv33_iw33_sm33_v2"] = j["mix_cv33_iw33_sm33"]

    # Drop score_ssim2 from features if requested (Anti-pattern #6 mitigation)
    if drop_ssim2_feature and "score_ssim2" in j.columns:
        j = j.drop(columns=["score_ssim2"])
        print("  dropped `score_ssim2` from features (Anti-pattern #6 mitigation)")

    # Drop the temporary basename helper
    if "basename" in j.columns:
        j = j.drop(columns=["basename"])

    out_table = pa.Table.from_pandas(j)
    pq.write_table(out_table, out_path, compression="zstd", compression_level=9)
    print(f"  written: {out_path}")
    print(f"  size: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")
    print(f"  rows: {len(j)}")
    return len(j)


def add_ssim2_to_372feat_corpus(corpus: str) -> int:
    """Add ssim2_gpu + mix_cv33_iw33_sm33 to an existing 372-feat
    mix_targets parquet. Requires a per-corpus local ssim2 score parquet.

    Returns 0 if the local parquet is missing.
    """
    print(f"=== Adding ssim2 to {corpus} 372-feat corpus ===")
    targets_path = os.path.join(
        DATA_DIR_FEAT_372, f"{corpus}_features_mix_targets_372col.parquet"
    )
    if not os.path.isfile(targets_path):
        print(f"  SKIP: {targets_path} missing")
        return 0
    local_ssim2_path = os.path.join(DATA_DIR_OUT, f"{corpus}_ssim2_local.parquet")
    if not os.path.isfile(local_ssim2_path):
        print(f"  SKIP: {local_ssim2_path} not yet computed")
        return 0

    targets = pq.read_table(targets_path).to_pandas()
    local = pq.read_table(local_ssim2_path).to_pandas()
    print(f"  targets {len(targets)}, local ssim2 {len(local)}")

    # The local ssim2 TSV uses image_path = basename of ref; the targets
    # parquet uses ref_basename. Join on ref_basename + codec + quality.
    local_dedup = local.drop_duplicates(
        subset=["image_path", "codec", "q", "knob_tuple_json"]
    )
    local_lite = local_dedup[["image_path", "codec", "q", "ssim2_gpu"]].rename(
        columns={"image_path": "ref_basename", "q": "quality"}
    )
    # The 372-feat parquets store ref_basename WITHOUT extension (e.g., "I01"),
    # while the local TSV uses the full filename ("I01.png"). Strip the
    # extension to align join keys.
    local_lite["ref_basename"] = local_lite["ref_basename"].apply(lambda s: os.path.splitext(s)[0])
    if "quality" not in targets.columns:
        if "q" in targets.columns:
            targets = targets.rename(columns={"q": "quality"})
    # Coerce types
    if "quality" in targets.columns:
        targets["quality"] = targets["quality"].astype(local_lite["quality"].dtype)

    # Per-ref aggregation: ssim2 may differ by codec/q so do a multi-key join
    # if both have those columns; else aggregate by ref+codec+quality
    join_keys = ["ref_basename", "codec", "quality"]
    available_keys = [k for k in join_keys if k in targets.columns and k in local_lite.columns]
    if not available_keys:
        print(f"  SKIP: no join keys overlap between targets and local")
        return 0
    print(f"  join keys: {available_keys}")
    # Aggregate (mean) the local scores by the available keys
    local_agg = local_lite.groupby(available_keys, as_index=False)["ssim2_gpu"].mean()
    merged = targets.merge(local_agg, on=available_keys, how="left")

    n_with = merged["ssim2_gpu"].notna().sum()
    print(f"  merged {len(merged)}; {n_with} with ssim2 ({n_with*100/len(merged):.1f}%)")

    merged["ssim2_log_norm"] = log_norm_ssim2(merged["ssim2_gpu"].fillna(0).to_numpy())
    # 3-way mix
    iw = merged.get("iwssim_log_norm", pa.array([0.0] * len(merged))).fillna(0).to_numpy() \
        if "iwssim_log_norm" in merged.columns else np.zeros(len(merged))
    cv = merged.get("cvvdp_log_norm", pa.array([0.0] * len(merged))).fillna(0).to_numpy() \
        if "cvvdp_log_norm" in merged.columns else np.zeros(len(merged))
    sm = merged["ssim2_log_norm"].fillna(0).to_numpy()
    merged["mix_cv33_iw33_sm33"] = (iw + cv + sm) / 3.0

    out_path = os.path.join(DATA_DIR_OUT, f"{corpus}_4target_372col.parquet")
    pq.write_table(pa.Table.from_pandas(merged), out_path, compression="zstd", compression_level=9)
    print(f"  written: {out_path}")
    print(f"  size: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")
    return n_with


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large-only", action="store_true")
    parser.add_argument("--no-drop-ssim2-feature", action="store_true")
    args = parser.parse_args()

    os.makedirs(DATA_DIR_OUT, exist_ok=True)

    # Step 1: LARGE 3-target corpus
    out_large = os.path.join(DATA_DIR_OUT, "large_3target_300feat_minus_ssim2.parquet")
    n_large = build_large_3target(
        out_large, drop_ssim2_feature=not args.no_drop_ssim2_feature
    )

    if args.large_only:
        print(f"\nDone: large = {n_large}")
        return

    # Step 2: 372-feat per-corpus joins (require local scoring)
    for corpus in ["kadid", "tid", "safesyn"]:
        add_ssim2_to_372feat_corpus(corpus)


if __name__ == "__main__":
    main()
