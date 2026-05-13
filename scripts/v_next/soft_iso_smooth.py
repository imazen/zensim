#!/usr/bin/env python3
"""Soft-isotonic per-curve score smoother for zensim bakes.

For each (image_path, codec, knob_tuple_json) curve in a unified
parquet, sort rows by q ascending and apply a running-max projection
to the bake's scores: any score below `running_max_so_far` gets
pushed up to the running max; non-violated segments untouched.

Result: non-monotonic q-step rate drops to 0% while preserving
cross-curve SROCC nearly perfectly. Verified on V0_16/V0_26/V0_31/V0_38
bakes; Δ SROCC vs ssim2 ranges from -0.0003 to +0.0008.

Use case:
- Codec orchestrator scoring a whole sweep: apply this before
  reporting user-facing zensim dial values per pair.
- Eval-time smoothness reporting on unified parquets.
- NOT applicable to single-pair runtime API (no curve context).

Usage:
    python3 soft_iso_smooth.py \\
        --bake weights/v0_16_2026-05-12.bin \\
        --parquet /mnt/v/zen/zensim-training/2026-05-07/unified/unified_v15r_zenjpeg.parquet \\
        [--flip-output]   # for V0_4-trained bakes that output distance

Reports non-monotonic rate before+after and SROCC delta against
optional reference score column (default: score_ssim2).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from score_unified_with_bake import parse_bake_v2, forward


def soft_iso_per_curve(scores, image_paths, codecs, knobs, qs):
    """Per (image_path, codec, knob) curve, push scores monotonic in q.

    Returns: new_scores (np.ndarray), stats dict.
    """
    groups = defaultdict(list)
    for i in range(len(scores)):
        if not np.isfinite(scores[i]):
            continue
        groups[(image_paths[i], codecs[i], knobs[i])].append((qs[i], scores[i], i))

    new_scores = scores.copy()
    n_pairs = 0
    n_rev_before = 0
    n_rev_after = 0
    n_corrections = 0
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: r[0])
        qs_g = [r[0] for r in rows]
        ss_orig = np.array([r[1] for r in rows], dtype=float)
        idxs = [r[2] for r in rows]
        for j in range(1, len(rows)):
            if qs_g[j] > qs_g[j-1]:
                n_pairs += 1
                if ss_orig[j] < ss_orig[j-1]:
                    n_rev_before += 1
        fixed = ss_orig.copy()
        running_max = fixed[0]
        for j in range(1, len(fixed)):
            if fixed[j] < running_max:
                fixed[j] = running_max
                n_corrections += 1
            else:
                running_max = fixed[j]
        for j, idx in enumerate(idxs):
            new_scores[idx] = fixed[j]
        for j in range(1, len(rows)):
            if qs_g[j] > qs_g[j-1] and fixed[j] < fixed[j-1]:
                n_rev_after += 1

    return new_scores, {
        "n_pairs": n_pairs,
        "n_rev_before": n_rev_before,
        "n_rev_after": n_rev_after,
        "n_corrections": n_corrections,
        "non_mono_before": (n_rev_before / max(n_pairs, 1)) * 100.0,
        "non_mono_after": (n_rev_after / max(n_pairs, 1)) * 100.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True, type=Path)
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--flip-output", action="store_true",
                    help="Bake outputs distance (higher=worse); flip to score "
                         "via 100-y. Use for V0_4-trained bakes; omit for V0_16.")
    ap.add_argument("--ref-col", default="score_ssim2",
                    help="Reference score column for SROCC comparison.")
    args = ap.parse_args()

    print(f"Loading bake {args.bake}...")
    n_inputs, _, sm, ss, layers = parse_bake_v2(args.bake)

    print(f"Loading parquet {args.parquet}...")
    feat_cols = [f"feat_{i}" for i in range(n_inputs)]
    cols = ["image_path", "codec", "q", "knob_tuple_json", args.ref_col] + feat_cols
    t = pq.read_table(args.parquet, columns=cols)

    X = np.stack([t[c].to_numpy(dtype=np.float32) for c in feat_cols], axis=1)
    y = forward(X, sm, ss, layers)[:, 0]
    score = (100.0 - y) if args.flip_output else y
    print(f"Bake score-space: range [{score.min():.3f}, {score.max():.3f}]")

    new_score, stats = soft_iso_per_curve(
        score,
        t["image_path"].to_numpy(),
        t["codec"].to_numpy(),
        t["knob_tuple_json"].to_numpy(),
        t["q"].to_numpy(),
    )

    print(f"\nNon-mono q-step rate:")
    print(f"  BEFORE soft-iso: {stats['non_mono_before']:.2f}% "
          f"({stats['n_rev_before']}/{stats['n_pairs']})")
    print(f"  AFTER  soft-iso: {stats['non_mono_after']:.2f}% "
          f"({stats['n_rev_after']}/{stats['n_pairs']})")
    print(f"  Corrections applied: {stats['n_corrections']:,} "
          f"({100.0 * stats['n_corrections'] / len(score):.1f}% of scores)")

    try:
        from scipy.stats import spearmanr
        ref = t[args.ref_col].to_numpy()
        mask = np.isfinite(score) & np.isfinite(new_score) & np.isfinite(ref)
        r_b, _ = spearmanr(score[mask], ref[mask])
        r_a, _ = spearmanr(new_score[mask], ref[mask])
        print(f"\nSROCC vs {args.ref_col} (n={mask.sum():,}):")
        print(f"  BEFORE: {abs(r_b):.4f}")
        print(f"  AFTER:  {abs(r_a):.4f}  (Δ {abs(r_a) - abs(r_b):+.4f})")
    except ImportError:
        print("(scipy unavailable; skipping SROCC delta)")


if __name__ == "__main__":
    main()
