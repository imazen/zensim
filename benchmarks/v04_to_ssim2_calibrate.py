#!/usr/bin/env python3
"""Generate piecewise-21 anchors that map V0_4 distance → SSIMULACRA 2 score.

Reads per-pair CSV (with v04_distance + fast_ssim2_score columns) and outputs
21 (V0_4_distance, SSIM2_score) anchors at every 5th percentile of the
distance/score CDFs.

Difference from the original calibration: that one fit V0_4 distance → V0_2
score; this one fits → SSIM2 score directly. So a V0_4 score reported via this
mapping IS approximately the SSIMULACRA 2 score for the same pair.
"""

import sys
import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def fit_piecewise21(v04_dists, ssim2_scores):
    """Equipercentile match: best V0_4 distance ↔ best SSIM2 score.

    distance: higher = worse quality
    score:    higher = better quality
    Sort distances ASC, scores DESC, pair by index → monotone (distance ↑, score ↓).
    """
    v04_sorted = np.sort(v04_dists)               # ascending: best first (lowest dist)
    ssim2_sorted = np.sort(ssim2_scores)[::-1]    # descending: best first (highest score)
    n = len(v04_sorted)
    anchors = []
    for p in range(0, 101, 5):
        idx = int(round(p / 100.0 * (n - 1)))
        anchors.append((float(v04_sorted[idx]), float(ssim2_sorted[idx])))
    return anchors


def piecewise_eval(anchors, d):
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    if d <= xs[0]:
        return ys[0]
    if d >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, d)
    x0, y0 = anchors[i - 1]
    x1, y1 = anchors[i]
    t = (d - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def v02_score(distance):
    if distance <= 0:
        return 100.0
    s = 100.0 - 18.0 * (distance ** 0.7)
    return max(-100.0, min(100.0, s))


def main(csv_path, anchor_out, chart_out):
    df = pd.read_csv(csv_path)
    df_human = df[df["dataset"] != "KonJND-1k"].copy()
    print(f"loaded {len(df_human)} human-rated pairs")

    # Fit anchors V0_4 distance → SSIM2 score on all human-rated pairs.
    anchors = fit_piecewise21(
        df_human["v04_distance"].values,
        df_human["fast_ssim2_score"].values,
    )
    print(f"\n21 anchors (V0_4 distance, SSIM2 score):")
    for d, s in anchors:
        print(f"  ({d:>10.4f}, {s:>10.4f})")

    # Apply to all pairs.
    df_human["v04_ssim2_calibrated"] = df_human["v04_distance"].apply(
        lambda d: piecewise_eval(anchors, d)
    )
    df_human["v02_score"] = df_human["v02_distance"].apply(v02_score)

    # Sanity check: fit error
    errs = df_human["v04_ssim2_calibrated"] - df_human["fast_ssim2_score"]
    print(f"\nFit residual (V0_4-via-anchors − fast-ssim2):")
    print(f"  mean   = {errs.mean():+.3f}")
    print(f"  median = {errs.median():+.3f}")
    print(f"  σ      = {errs.std():.3f}")
    print(f"  RMSE   = {np.sqrt((errs**2).mean()):.3f}")

    # Build SSIM2 score buckets for the chart.
    df_in = df_human[
        (df_human["fast_ssim2_score"] >= 0) & (df_human["fast_ssim2_score"] <= 100)
    ].copy()
    df_in["ssim2_int"] = df_in["fast_ssim2_score"].round().astype(int).clip(0, 100)
    grp_ssim2 = df_in.groupby("ssim2_int")["fast_ssim2_score"]
    grp_v04 = df_in.groupby("ssim2_int")["v04_ssim2_calibrated"]
    bins = sorted(grp_ssim2.groups.keys())
    box_ssim2 = [grp_ssim2.get_group(b).values for b in bins]
    box_v04 = [grp_v04.get_group(b).values for b in bins]
    n_per = [len(g) for g in box_ssim2]

    THRESH = 30
    full = {b: 0 for b in range(0, 101)}
    for b, n in zip(bins, n_per):
        full[b] = n
    contiguous = []
    cur = None
    for b in range(0, 101):
        if full[b] >= THRESH:
            if cur is None:
                cur = b
        else:
            if cur is not None:
                contiguous.append((cur, b - 1))
                cur = None
    if cur is not None:
        contiguous.append((cur, 100))
    if contiguous:
        first_s, last_s = max(contiguous, key=lambda r: r[1] - r[0])
    else:
        first_s = last_s = None

    # Plot
    fig, (ax_main, ax_count) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    if first_s is not None:
        ax_main.axvspan(
            first_s - 0.5, last_s + 0.5, alpha=0.10, color="green",
            label=f"calibration range "
                  f"(SSIM2 {first_s}–{last_s}, ≥{THRESH} pairs/bin)",
        )

    pos_s = [b - 0.22 for b in bins]
    pos_v = [b + 0.22 for b in bins]
    bp_s = ax_main.boxplot(
        box_ssim2, positions=pos_s, widths=0.4,
        whis=(5, 95), showfliers=False, patch_artist=True, manage_ticks=False,
    )
    bp_v = ax_main.boxplot(
        box_v04, positions=pos_v, widths=0.4,
        whis=(5, 95), showfliers=False, patch_artist=True, manage_ticks=False,
    )
    for b in bp_s["boxes"]:
        b.set_facecolor("#ff8c42"); b.set_edgecolor("#933a00"); b.set_alpha(0.65)
    for m in bp_s["medians"]:
        m.set_color("#7a1300"); m.set_linewidth(1.4)
    for b in bp_v["boxes"]:
        b.set_facecolor("#3a86ff"); b.set_edgecolor("#1d3557"); b.set_alpha(0.65)
    for m in bp_v["medians"]:
        m.set_color("#0f1c40"); m.set_linewidth(1.4)

    ax_main.plot([0, 100], [0, 100], color="#888", linestyle="--",
                 linewidth=1.0, label="identity (metric = SSIM2)")
    legend_elems = [
        Patch(facecolor="#ff8c42", edgecolor="#933a00", alpha=0.65, label="SSIMULACRA 2"),
        Patch(facecolor="#3a86ff", edgecolor="#1d3557", alpha=0.65,
              label="V0_4-ssim2 (piecewise-21 calibrated to SSIM2)"),
        plt.Line2D([0], [0], color="#888", linestyle="--", label="identity (metric = SSIM2)"),
        Patch(facecolor="green", alpha=0.10,
              label=f"calibration range (≥{THRESH} pairs/bin)"),
    ]
    ax_main.legend(handles=legend_elems, loc="lower right")
    ax_main.set_ylabel("metric score")
    ax_main.set_title(
        "SSIMULACRA 2 vs V0_4-ssim2 calibrated to SSIM2 — per SSIM2 integer bucket\n"
        f"Box = IQR (25–75%), whiskers = 5–95%; n = {len(df_in)} pairs across "
        "KADID + TID + CID22-49"
    )
    ax_main.set_xlim(-1, 101)
    ax_main.set_ylim(-30, 110)
    ax_main.set_xticks(range(0, 101, 5))
    ax_main.grid(True, axis="y", alpha=0.3)

    ax_count.bar(bins, n_per, width=0.9, color="#457b9d", alpha=0.8)
    ax_count.axhline(THRESH, color="green", linestyle=":", linewidth=1.0,
                     label=f"density threshold ({THRESH})")
    ax_count.set_ylabel("# pairs / bin")
    ax_count.set_xlabel("SSIMULACRA 2 score (integer bucket)")
    ax_count.set_yscale("log")
    ax_count.grid(True, axis="y", alpha=0.3)
    ax_count.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(chart_out, dpi=120, bbox_inches="tight")
    print(f"\nwrote chart to {chart_out}")

    # Anchor table to a markdown file for direct paste-in.
    with open(anchor_out, "w") as f:
        f.write("# V0_4-ssim2 piecewise-21 anchors (V0_4 distance → SSIM2 score)\n\n")
        f.write(
            "Generated by benchmarks/v04_ssim2_to_ssim2_calibrate.py from the "
            f"per-pair CSV at\n`{csv_path}`. Fits V0_4 raw distance to SSIMULACRA 2 "
            "score (rather than V0_2 score) so that V0_4 reports a SSIM2-grade "
            "score value directly.\n\n"
        )
        f.write(f"n = {len(df_human)} human-rated pairs (KADID + TID + CID22-49).\n\n")
        f.write(f"Fit residual: mean={errs.mean():+.3f}, σ={errs.std():.3f}, "
                f"RMSE={np.sqrt((errs**2).mean()):.3f}.\n\n")
        f.write("```rust\nscore_mapping: ScoreMapping::PiecewiseLinear {\n    table: &[\n")
        for d, s in anchors:
            f.write(f"        ({d:>10.4f}, {s:>10.4f}),\n")
        f.write("    ],\n},\n```\n")
    print(f"wrote anchors to {anchor_out}")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/per_pair_v04_ssim2_holdout.csv"
    anchors = sys.argv[2] if len(sys.argv) > 2 else "/mnt/v/output/zensim/v04_to_ssim2_anchors.md"
    chart = sys.argv[3] if len(sys.argv) > 3 else "/mnt/v/output/zensim/whisker_v04_to_ssim2.png"
    main(csv, anchors, chart)
