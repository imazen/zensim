#!/usr/bin/env python3
"""
Diagnostic + sanity-check tool for PJND-aware pair weighting on the konjnd group.

The trainer computes the per-pair weight inline from (score_a, score_b) — this script
materializes the same weight function for every cross-pair in the konjnd parquet so
the experiment log can include the distribution sanity check the user requested.

Weight definition (must match trainer implementation in mlp_train.rs):

  Given two konjnd rows with mcos scores (s_a, s_b) and hyperparams
  (threshold, sigma_mid, sigma_gap, gap_anchor):

    mid    = 0.5 * (s_a + s_b)
    gap    = |s_a - s_b|
    w_mid  = exp(-((mid - threshold) / sigma_mid)**2)
    w_gap  = exp(-((gap - gap_anchor) / sigma_gap)**2)
    w      = w_mid * w_gap

  Peaks when the pair MIDPOINT sits at the PJND threshold AND the pair GAP
  matches the typical JND-boundary gap.

For the trainer to preserve per-group total weight, the weights are normalized so
that the EXPECTED weight under uniform (ia, ib) sampling = 1.0. We compute that
normalization constant Z here, then divide w by Z when applying.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--konjnd-parquet", required=True,
                    help="path to konjnd parquet with human_score column")
    ap.add_argument("--threshold", type=float, default=45.0,
                    help="PJND midpoint threshold in mcos score units. Default 45 "
                         "(empirical midpoint of konjnd bimodal cluster centers ~31 and ~58)")
    ap.add_argument("--sigma-mid", type=float, default=8.0,
                    help="Gaussian sigma for pair-midpoint proximity to threshold")
    ap.add_argument("--gap-anchor", type=float, default=27.0,
                    help="Typical pair gap for JND-boundary pairs. Default 27 "
                         "(cluster center gap = 58 - 31 = 27 in this corpus)")
    ap.add_argument("--sigma-gap", type=float, default=10.0,
                    help="Gaussian sigma for pair-gap proximity to anchor")
    ap.add_argument("--output", required=True,
                    help="output parquet path: per-pair weights + diagnostics")
    ap.add_argument("--summary", required=False, default=None,
                    help="optional plain-text summary file")
    args = ap.parse_args()

    if not Path(args.konjnd_parquet).exists():
        print(f"konjnd parquet not found: {args.konjnd_parquet}", file=sys.stderr)
        return 1

    df = pq.read_table(args.konjnd_parquet).to_pandas()
    if "human_score" not in df.columns or "ref_basename" not in df.columns:
        print("konjnd parquet missing required columns", file=sys.stderr)
        return 1

    scores = df["human_score"].to_numpy(dtype=np.float64)
    refs = df["ref_basename"].to_numpy()
    n = len(scores)
    print(f"konjnd rows: {n}")
    print(f"score range: [{scores.min():.2f}, {scores.max():.2f}], "
          f"mean={scores.mean():.2f}")

    # Compute all-pairs weight matrix. With n=1008 that's ~1M floats — fine in RAM.
    sa = scores[:, None]
    sb = scores[None, :]
    mid = 0.5 * (sa + sb)
    gap = np.abs(sa - sb)
    w_mid = np.exp(-((mid - args.threshold) / args.sigma_mid) ** 2)
    w_gap = np.exp(-((gap - args.gap_anchor) / args.sigma_gap) ** 2)
    w_unnorm = w_mid * w_gap
    # Zero the diagonal (self-pairs are skipped by trainer)
    np.fill_diagonal(w_unnorm, 0.0)
    # Normalize so the mean off-diagonal weight = 1.0 (per-group total preserved
    # vs uniform sampling). This means probabilistic sampling that selects pair
    # (ia, ib) with prob w[ia, ib]/sum(w) — weighting-by-w yields the same
    # expected total contribution to the loss.
    mean_w = w_unnorm[~np.eye(n, dtype=bool)].mean()
    if mean_w <= 0:
        print("normalization failed: mean weight is zero", file=sys.stderr)
        return 1
    w_norm = w_unnorm / mean_w

    print(f"normalization Z (uniform-mean): {mean_w:.6f}")
    print(f"after normalization mean: {w_norm[~np.eye(n, dtype=bool)].mean():.6f} (should be ~1.0)")

    # Sanity: weight distribution
    print(f"weight percentiles (post-normalization):")
    flat = w_norm[~np.eye(n, dtype=bool)].flatten()
    for p in [1, 10, 25, 50, 75, 90, 99]:
        print(f"  p{p}: {np.percentile(flat, p):.4f}")
    print(f"  fraction with w > 0.5: {(flat > 0.5).mean():.4f}")
    print(f"  fraction with w > 1.0: {(flat > 1.0).mean():.4f}")
    print(f"  fraction with w > 2.0: {(flat > 2.0).mean():.4f}")
    print(f"  fraction with w < 0.1: {(flat < 0.1).mean():.4f}")

    # Histogram of weights
    hist, edges = np.histogram(flat, bins=np.linspace(0, max(2.0, flat.max()), 20))
    print("weight histogram:")
    for i, c in enumerate(hist):
        bar = "#" * int(c / max(1, hist.max()) * 40)
        print(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}]: {c:7d} {bar}")

    # Output parquet (only upper-triangle to save space)
    iu, ju = np.triu_indices(n, k=1)
    weights = w_norm[iu, ju]
    out_table = pa.table({
        "idx_a": pa.array(iu.astype(np.int32)),
        "idx_b": pa.array(ju.astype(np.int32)),
        "ref_a": pa.array(refs[iu]),
        "ref_b": pa.array(refs[ju]),
        "score_a": pa.array(scores[iu]),
        "score_b": pa.array(scores[ju]),
        "weight": pa.array(weights.astype(np.float32)),
    })
    pq.write_table(out_table, args.output, compression="zstd")
    print(f"wrote {args.output} ({len(iu)} upper-triangle pairs)")

    if args.summary:
        with open(args.summary, "w") as f:
            f.write(f"PJND pair-weight summary\n")
            f.write(f"=========================\n\n")
            f.write(f"konjnd_parquet: {args.konjnd_parquet}\n")
            f.write(f"konjnd rows: {n}\n")
            f.write(f"score range: [{scores.min():.2f}, {scores.max():.2f}], mean={scores.mean():.2f}\n\n")
            f.write(f"hyperparams:\n")
            f.write(f"  threshold     = {args.threshold}\n")
            f.write(f"  sigma_mid     = {args.sigma_mid}\n")
            f.write(f"  gap_anchor    = {args.gap_anchor}\n")
            f.write(f"  sigma_gap     = {args.sigma_gap}\n\n")
            f.write(f"normalization Z (uniform-mean): {mean_w:.6f}\n\n")
            f.write(f"weight distribution (post-normalization, off-diagonal):\n")
            for p in [1, 10, 25, 50, 75, 90, 99]:
                f.write(f"  p{p}: {np.percentile(flat, p):.4f}\n")
            f.write(f"  fraction with w > 0.5: {(flat > 0.5).mean():.4f}\n")
            f.write(f"  fraction with w > 1.0: {(flat > 1.0).mean():.4f}\n")
            f.write(f"  fraction with w > 2.0: {(flat > 2.0).mean():.4f}\n")
            f.write(f"  fraction with w < 0.1: {(flat < 0.1).mean():.4f}\n\n")
            f.write(f"histogram:\n")
            for i, c in enumerate(hist):
                bar = "#" * int(c / max(1, hist.max()) * 40)
                f.write(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}]: {c:7d} {bar}\n")
        print(f"wrote summary: {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
