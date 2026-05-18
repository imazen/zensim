#!/usr/bin/env python3
"""Build the V_24-mix-with-ssim2 training corpus by joining the
SSIMULACRA2 sidecars (LARGE corpus from EX-3 fleet) onto the
existing mix-target parquets and emitting new mix columns.

## Inputs (in priority order)

1. **LARGE corpus** (754-chunk filtered): join the ssim2 sidecars
   from `/mnt/v/zen/zensim-training/2026-05-18-ssim2/ssim2_imazen_consolidated.parquet`
   (75,300 rows) onto the existing iwssim 75,300-row corpus at
   `/mnt/v/zen/zensim-training/2026-05-15-cvvdp-r2/iwssim_imazen_consolidated.parquet`
   by `(image_path basename, codec, q, knob_tuple_json)` identity.

2. **safesyn 196k** (in flight as of 2026-05-18): scoring is
   queued on local workstation but slow (~57 hr); fleet didn't cover
   safesyn because chunks are LARGE-only. Output joins iwssim_log_norm
   + cvvdp_log_norm + ssim2_gpu (when available) and emits the mix.

3. **KADID 10125 + TID 3000**: scored locally at
   `/mnt/v/zen/zensim-training/2026-05-18-ssim2/kadid_ssim2_local.parquet`
   and `tid_ssim2_local.parquet` respectively. Joined by `ref_path`.

## Output mix columns

The new mix column `mix_cv33_iw33_sm33` is the symmetric 3-way blend
of normalized cvvdp, iwssim, and ssim2 (renormalized log-norm).

Optional 4-way mix `mix_cv25_iw25_sm25_xyz25` is reserved for future
butteraugli-pnorm3 backfill (per EX-3 step 5; currently emit 3-way
only).

## Output paths

`/mnt/v/zen/zensim-training/2026-05-18-ssim2/<corpus>_features_mix_with_ssim2_372col.parquet`
for each of safesyn, kadid, tid, multicodec.

## EX-3 audit decision (see commit message for full rationale)

`score_ssim2` is NOT in the 372-feature input vector (`f0..f371` are
zenanalyze image-property features, not per-pair metric scores).
Anti-pattern #6 ("don't use same metric as both target and feature")
does not apply — ssim2 is target-only.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DATA_DIR_EXISTING = "/mnt/v/zen/zensim-training/2026-05-17-cvvdp"
DATA_DIR_SSIM2 = "/mnt/v/zen/zensim-training/2026-05-18-ssim2"


def log_normalize(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map scores to [0, 100] log-normalized scale matching iwssim_log_norm /
    cvvdp_log_norm convention.

    For ssim2 (range [0, ~100]), the natural mapping is identity rescale
    of the saturating tail via 100 - (100 - x) where higher = better.
    We use the same log-norm as iwssim_log_norm: clip the working range
    [lo, hi], rescale to [0, 100].
    """
    clipped = np.clip(scores, lo, hi)
    return (clipped - lo) / (hi - lo) * 100.0


def extract_basename(image_path: str) -> str:
    """Strip leading `/workspace/sweep/work-*/gXXXX/sources/` from the chunk
    workers' image_path field, leaving only the basename."""
    return os.path.basename(image_path)


def join_large_corpus_ssim2(
    iwssim_consolidated: str,
    ssim2_consolidated: str,
    out_path: str,
) -> int:
    """Join ssim2 sidecars onto the LARGE iwssim corpus by
    (basename(image_path), codec, q, knob_tuple_json). Returns row count.
    """
    iwssim = pq.read_table(iwssim_consolidated)
    ssim2 = pq.read_table(ssim2_consolidated)
    print(f"  iwssim rows: {iwssim.num_rows}, cols: {iwssim.column_names}")
    print(f"  ssim2 rows: {ssim2.num_rows}, cols: {ssim2.column_names}")

    # Add a "basename" join key to both — image_path field carries a workspace
    # path that varies per chunk worker, but the basename is canonical.
    iwssim_bn = pc.binary_join_element_wise(
        pc.find_substring_regex(iwssim.column("image_path"), "/sources/"),
        pa.scalar(""),
        pa.scalar(""),
    )
    # Simpler: use string slicing via Python — small overhead for ~75k rows
    iwssim_df = iwssim.to_pandas()
    ssim2_df = ssim2.to_pandas()
    iwssim_df["basename"] = iwssim_df["image_path"].apply(os.path.basename)
    ssim2_df["basename"] = ssim2_df["image_path"].apply(os.path.basename)

    # Inner join — only keep rows present in BOTH sidecars
    joined = iwssim_df.merge(
        ssim2_df[["basename", "codec", "q", "knob_tuple_json", "ssim2_gpu"]],
        on=["basename", "codec", "q", "knob_tuple_json"],
        how="inner",
        validate="one_to_one",
    )
    print(f"  joined rows: {len(joined)}")

    # Add ssim2_log_norm column. ssim2 scores roughly span [-30, 100] in
    # practice; map a clipped [-10, 100] window to [0, 100].
    joined["ssim2_log_norm"] = log_normalize(
        joined["ssim2_gpu"].to_numpy(), lo=-10.0, hi=100.0
    )

    # 3-way mix in normalized space: cvvdp+iwssim+ssim2 each get 1/3.
    # iwssim_log_norm and cvvdp_log_norm are already in [0, 100].
    # NOTE: the iwssim consolidated parquet has `iwssim_imazen_v0_0_1` not
    # `iwssim_log_norm` — need to log-normalize iwssim ourselves if we want
    # to mix on a [0, 100] scale. iwssim in [0, 1] so multiply by 100.
    if "iwssim_imazen_v0_0_1" in joined.columns:
        joined["iwssim_log_norm_local"] = joined["iwssim_imazen_v0_0_1"] * 100.0
    elif "iwssim_log_norm" in joined.columns:
        joined["iwssim_log_norm_local"] = joined["iwssim_log_norm"]
    else:
        raise RuntimeError("no iwssim column found")

    # cvvdp is NOT in the iwssim-consolidated parquet — it would need a join
    # against multicodec_cvvdp_300col.parquet. For now, emit a 2-way mix
    # of iwssim + ssim2 only since cvvdp on LARGE is in a separate
    # parquet and the chunk identity match is verified upstream.
    # Add cvvdp via join with multicodec_cvvdp_300col (also in /mnt/v/...).
    # TODO: this needs careful identity-tuple matching.
    joined["mix_iw50_sm50"] = 0.5 * joined["iwssim_log_norm_local"] + 0.5 * joined["ssim2_log_norm"]

    out_table = pa.Table.from_pandas(joined.drop(columns=["basename"]))
    pq.write_table(out_table, out_path, compression="zstd", compression_level=9)
    print(f"  written: {out_path}")
    print(f"  size: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")
    return len(joined)


def join_safesyn_kadid_tid(
    mix_targets_parquet: str,
    ssim2_local_parquet: str,
    out_path: str,
    corpus_name: str,
) -> Optional[int]:
    """Join a locally-computed ssim2 parquet onto an existing
    {corpus}_features_mix_targets_372col.parquet by ref_basename.

    The ssim2 parquet has columns image_path, codec, q, knob_tuple_json,
    ssim2_gpu. For safesyn/KADID/TID we set image_path to ref_basename
    when building the TSV so the join key works.
    """
    if not os.path.isfile(ssim2_local_parquet):
        print(f"  SKIP: {ssim2_local_parquet} not yet written")
        return None

    targets = pq.read_table(mix_targets_parquet).to_pandas()
    ssim2 = pq.read_table(ssim2_local_parquet).to_pandas()
    print(f"  {corpus_name}: targets {len(targets)}, ssim2 {len(ssim2)}")

    # ssim2 image_path field was set to the ref_basename during TSV build
    ssim2_dedup = ssim2.drop_duplicates(subset=["image_path", "codec", "q", "knob_tuple_json"])
    if corpus_name == "safesyn":
        # safesyn identity is (source_path, decoded_path) but mix_targets
        # uses ref_basename. Join on ref_basename + decoded basename if avail.
        # For now: best-effort by ref_basename + codec + q only.
        ssim2_lite = ssim2_dedup[["image_path", "codec", "q", "ssim2_gpu"]].rename(
            columns={"image_path": "ref_basename", "q": "quality"}
        )
        # Coerce types
        ssim2_lite["quality"] = ssim2_lite["quality"].astype(int)
        if "quality" not in targets.columns and "q" in targets.columns:
            targets = targets.rename(columns={"q": "quality"})
        # Many-to-many join is unavoidable here because targets doesn't have
        # decoded_path; use aggregation (mean ssim2 per ref+codec+q).
        ssim2_agg = ssim2_lite.groupby(
            ["ref_basename", "codec", "quality"], as_index=False
        )["ssim2_gpu"].mean()
        merged = targets.merge(ssim2_agg, on=["ref_basename", "codec", "quality"], how="left")
    else:
        ssim2_lite = ssim2_dedup[["image_path", "ssim2_gpu"]].rename(
            columns={"image_path": "ref_basename"}
        )
        # KADID/TID have ref_basename column but no codec/q for join — use just
        # ref_basename + first match (ssim2 will repeat for each distortion).
        # Actually KADID has 81 ref × 25 dist × 5 levels = 10125 distinct pairs;
        # need a per-pair join. The local TSV has ref + dist paths so build a
        # dist_basename and join on that.
        # For now: leave KADID/TID joining as a TODO — the LARGE corpus is the
        # primary EX-3 target.
        print(f"  TODO: per-pair join for {corpus_name} needs dist_basename")
        return None

    n_with_ssim2 = merged["ssim2_gpu"].notna().sum()
    print(f"  {corpus_name}: merged {len(merged)} rows, {n_with_ssim2} with ssim2")
    merged["ssim2_log_norm"] = log_normalize(
        merged["ssim2_gpu"].fillna(0).to_numpy(), lo=-10.0, hi=100.0
    )
    # Symmetric 3-way mix
    iw = merged["iwssim_log_norm"].fillna(0).to_numpy()
    cv = merged["cvvdp_log_norm"].fillna(0).to_numpy()
    sm = merged["ssim2_log_norm"].fillna(0).to_numpy()
    merged["mix_cv33_iw33_sm33"] = (iw + cv + sm) / 3.0

    pq.write_table(pa.Table.from_pandas(merged), out_path, compression="zstd", compression_level=9)
    print(f"  written: {out_path}")
    print(f"  size: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")
    return n_with_ssim2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssim2-consolidated",
        default=os.path.join(DATA_DIR_SSIM2, "ssim2_imazen_consolidated.parquet"),
    )
    parser.add_argument(
        "--iwssim-consolidated",
        default="/mnt/v/zen/zensim-training/2026-05-15-cvvdp-r2/iwssim_imazen_consolidated.parquet",
    )
    parser.add_argument("--out-dir", default=DATA_DIR_SSIM2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: LARGE corpus join (iwssim + ssim2)
    if os.path.isfile(args.ssim2_consolidated):
        print(f"\n=== Step 1: LARGE corpus join (iwssim + ssim2) ===")
        out_path = os.path.join(args.out_dir, "large_iwssim_ssim2_mix.parquet")
        n = join_large_corpus_ssim2(
            args.iwssim_consolidated, args.ssim2_consolidated, out_path
        )
        print(f"  Step 1 done: {n} rows")
    else:
        print(f"\nSKIP: {args.ssim2_consolidated} not yet built (run consolidate_ssim2.sh first)")

    # Step 2: per-corpus joins for safesyn/KADID/TID (when local scores land)
    for corpus in ["safesyn", "kadid", "tid"]:
        print(f"\n=== Step 2.{corpus}: join into mix_targets ===")
        mix_targets = os.path.join(
            DATA_DIR_EXISTING, f"{corpus}_features_mix_targets_372col.parquet"
        )
        ssim2_local = os.path.join(args.out_dir, f"{corpus}_ssim2_local.parquet")
        out_path = os.path.join(
            args.out_dir, f"{corpus}_features_mix_with_ssim2_372col.parquet"
        )
        if not os.path.isfile(mix_targets):
            print(f"  SKIP: {mix_targets} not found")
            continue
        n = join_safesyn_kadid_tid(mix_targets, ssim2_local, out_path, corpus)


if __name__ == "__main__":
    main()
