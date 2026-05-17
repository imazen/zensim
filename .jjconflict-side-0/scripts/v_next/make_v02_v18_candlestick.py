#!/usr/bin/env python3
"""Candlestick (box-and-whisker) mapping graph: V0_2 → V0_18.

Reads the per-pair TSV emitted by
`zensim-bench/examples/dataset_metric_baseline.rs` with `--v04-bake`
pointed at V0_18, then renders a candlestick chart where each X-axis
bin holds a V0_2 distance range and the box shows the p5/p25/p50/p75/p95
of corresponding V0_18 distances. The 10-band grid (B0..B9, width
0.10 on normalized [0,1] scale → score 0..100) is overlaid for
reference.

Output:
  benchmarks/v0_2_to_v0_18_candlestick_2026-05-14.png
  benchmarks/v0_2_to_v0_18_candlestick_2026-05-14.svg

The chart shows whether V0_18 is a smooth, monotonic function of
V0_2 (it should be roughly, since both metrics rank distortion) and
where the spread is largest — those bins are where V0_18 disagrees
most with V0_2, which is the heart of why V0_18 beats it on SROCC.

Usage:
  python3 scripts/v_next/make_v02_v18_candlestick.py \\
    --tsv /tmp/v02_v18_paired.tsv \\
    --out-png benchmarks/v0_2_to_v0_18_candlestick_2026-05-14.png \\
    --out-svg benchmarks/v0_2_to_v0_18_candlestick_2026-05-14.svg
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="per-pair TSV from dataset_metric_baseline")
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-svg", required=True)
    ap.add_argument(
        "--bin-count",
        type=int,
        default=20,
        help="Number of V0_2 bins (20 = step-5 on a 0..100 scale)",
    )
    ap.add_argument(
        "--map-distance-to-score",
        action="store_true",
        default=True,
        help="V0_2/V0_18 columns are distances (smaller = better). Convert to score (100 - d) for x/y axes.",
    )
    args = ap.parse_args()

    tsv = Path(args.tsv)
    if not tsv.exists() or tsv.stat().st_size == 0:
        print(f"ERROR: {tsv} is empty or missing", file=sys.stderr)
        return 1

    # Read paired columns. Skip rows where either column is NaN.
    v02_scores = []
    v18_scores = []
    datasets = []
    with tsv.open() as f:
        r = csv.reader(f)
        header = next(r)
        col_v02 = header.index("v02_distance")
        col_v18 = header.index("v04_distance")
        col_ds = header.index("dataset")
        for row in r:
            try:
                d2 = float(row[col_v02])
                d18 = float(row[col_v18])
            except (ValueError, IndexError):
                continue
            if math.isnan(d2) or math.isnan(d18):
                continue
            # V0_2 distance can range ~0..90 on synthetic; V0_18 is
            # MCOS-calibrated 0..100 score (score = raw output). Convert
            # both to 0..100 score by clamping to [0,100].
            if args.map_distance_to_score:
                s2 = max(0.0, min(100.0, 100.0 - d2))
                # V0_18 already outputs a score (skip_score_mapping=true);
                # but here d18 is the harness's v04_distance which IS the
                # raw bake output post-calibration. Treat as score 0..100.
                s18 = max(0.0, min(100.0, d18))
            else:
                s2 = d2
                s18 = d18
            v02_scores.append(s2)
            v18_scores.append(s18)
            datasets.append(row[col_ds])

    v02 = np.array(v02_scores)
    v18 = np.array(v18_scores)
    print(f"Loaded {len(v02)} valid paired rows across {len(set(datasets))} datasets")

    # Bin V0_2 scores into args.bin_count uniform bins on [0,100].
    bin_edges = np.linspace(0.0, 100.0, args.bin_count + 1)
    bin_indices = np.clip(np.digitize(v02, bin_edges, right=False) - 1, 0, args.bin_count - 1)

    # Per-bin: collect V0_18 scores, compute percentiles.
    bin_data = []
    for i in range(args.bin_count):
        mask = bin_indices == i
        n = int(mask.sum())
        if n < 4:
            bin_data.append(None)
            continue
        ys = v18[mask]
        p = np.percentile(ys, [5, 25, 50, 75, 95])
        bin_data.append((n, p[0], p[1], p[2], p[3], p[4]))

    # Render box-and-whisker (Tukey-style) chart.
    fig, ax = plt.subplots(figsize=(11, 7))
    box_positions = []
    box_stats = []
    for i, d in enumerate(bin_data):
        if d is None:
            continue
        n, p5, p25, p50, p75, p95 = d
        center = (bin_edges[i] + bin_edges[i + 1]) / 2.0
        box_positions.append(center)
        box_stats.append({
            "med": p50,
            "q1": p25,
            "q3": p75,
            "whislo": p5,
            "whishi": p95,
            "fliers": [],
            "label": f"{int(bin_edges[i])}..{int(bin_edges[i + 1])}\nn={n}",
        })

    if not box_stats:
        print("ERROR: no bins have n≥4. Cannot render chart.", file=sys.stderr)
        return 1

    bxp = ax.bxp(box_stats, positions=box_positions, widths=4.0, showfliers=False, patch_artist=True)
    for patch in bxp["boxes"]:
        patch.set_facecolor("#4a90e2")
        patch.set_alpha(0.6)
    for med in bxp["medians"]:
        med.set_color("#d63031")
        med.set_linewidth(2)

    # Overlay the identity line (perfect agreement V0_2 = V0_18).
    ax.plot([0, 100], [0, 100], color="#777", linestyle="--", linewidth=1, alpha=0.7, label="V0_18 = V0_2 (identity)")

    # 10-band grid lines (B0..B9, width-10).
    for x in range(0, 101, 10):
        ax.axvline(x, color="#ccc", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.axhline(x, color="#ccc", linestyle=":", linewidth=0.5, alpha=0.6)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("V0_2 score (100 − distance, clamped)")
    ax.set_ylabel("V0_18 score (MCOS-calibrated 0..100)")
    ax.set_title(
        "V0_2 → V0_18 mapping (candlestick by V0_2 bin)\n"
        f"n={len(v02)} paired rows across CID22/KADID/TID/AIC-3 ; whiskers = p5/p95, box = p25/p75, red = median\n"
        "10-band width-10 grid in dotted gray (CLAUDE.md 2026-05-14 release-gate)"
    )
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend(loc="lower right")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    fig.savefig(args.out_svg)
    print(f"Wrote {args.out_png} + {args.out_svg}")

    # Also dump the per-bin stats TSV for downstream consumers.
    tsv_out = Path(args.out_png).with_suffix(".tsv")
    with tsv_out.open("w") as f:
        f.write("bin_lo\tbin_hi\tn\tp5\tp25\tp50\tp75\tp95\n")
        for i, d in enumerate(bin_data):
            if d is None:
                continue
            n, p5, p25, p50, p75, p95 = d
            f.write(
                f"{bin_edges[i]:.1f}\t{bin_edges[i+1]:.1f}\t{n}\t{p5:.2f}\t{p25:.2f}\t{p50:.2f}\t{p75:.2f}\t{p95:.2f}\n"
            )
    print(f"Wrote per-bin stats TSV: {tsv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
