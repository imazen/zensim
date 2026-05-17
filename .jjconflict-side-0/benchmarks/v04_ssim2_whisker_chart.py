#!/usr/bin/env python3
"""Whisker chart of SSIMULACRA 2 vs V0_4-ssim2 calibrated score across integer score buckets,
with the human-data-density region highlighted.

Reads per-pair CSV (dataset, human_score, v02_distance, v04_distance, fast_ssim2_score, butter_3norm)
emitted by dataset_metric_baseline --per-pair-output, applies the piecewise-21 calibration
from benchmarks/v04_ssim2_holdout_calibration_2026-05-01.md to map V0_4 raw distance → score,
and produces matplotlib boxplots.
"""

import sys
import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Piecewise-21 calibration anchors from benchmarks/v04_ssim2_holdout_calibration_2026-05-01.md
# Sorted by V0_4 distance ascending; score decreases monotonically.
ANCHORS = [
    (-59.8119, 100.0000),
    (-46.3979, 86.3832),
    (-39.8782, 82.4838),
    (-35.2807, 79.5798),
    (-30.7110, 76.5164),
    (-25.9624, 73.4357),
    (-20.0988, 69.9280),
    (-12.7256, 65.9895),
    ( -3.0700, 61.2748),
    (  4.6772, 55.6898),
    ( 10.5595, 48.6972),
    ( 17.3930, 40.9081),
    ( 24.9022, 32.4060),
    ( 33.9656, 22.3016),
    ( 45.3017, 11.4260),
    ( 57.9030, -0.1502),
    ( 72.9362,-12.2887),
    ( 92.7248,-28.9030),
    (124.7492,-50.6577),
    (176.7873,-81.7489),
    (623.7492,-100.0000),
]
ANCHOR_X = [a[0] for a in ANCHORS]
ANCHOR_Y = [a[1] for a in ANCHORS]


def piecewise_score(d: float) -> float:
    """Linear interpolation in the piecewise-21 anchor table."""
    if d <= ANCHOR_X[0]:
        return ANCHOR_Y[0]
    if d >= ANCHOR_X[-1]:
        return ANCHOR_Y[-1]
    i = bisect.bisect_right(ANCHOR_X, d)
    x0, y0 = ANCHORS[i - 1]
    x1, y1 = ANCHORS[i]
    t = (d - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def main(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} pairs from {csv_path}")
    print("dataset counts:", df["dataset"].value_counts().to_dict())

    # Apply piecewise-21 calibration.
    df["v04_calibrated_score"] = df["v04_distance"].apply(piecewise_score)
    # SSIM2 score from fast_ssim2 column is already 0-100.
    df["ssim2_score"] = df["fast_ssim2_score"]

    # Drop KonJND (no human MOS comparable to KADID/TID/CID22 — uses PJND threshold).
    df_human = df[df["dataset"] != "KonJND-1k"].copy()
    print(f"\npairs with human MOS: {len(df_human)}")

    # Bucket by integer SSIM2 score. Range observed: roughly [-150, 100].
    # We use bins centered at integers from -10 to 100 for the chart range,
    # but let's first see the actual range.
    print(
        f"\nSSIM2 score range: {df_human['ssim2_score'].min():.1f} to "
        f"{df_human['ssim2_score'].max():.1f}"
    )
    print(
        f"V0_4 calibrated score range: {df_human['v04_calibrated_score'].min():.1f} to "
        f"{df_human['v04_calibrated_score'].max():.1f}"
    )

    # Bin by integer SSIM2 score within [0, 100] only (the canonical 0-100 range).
    # Below 0 (poor quality) is sparse and goes to -150 for the worst KADID pairs.
    # We focus the whiskers on the 0-100 published range as the user asked.
    df_in = df_human[
        (df_human["ssim2_score"] >= 0) & (df_human["ssim2_score"] <= 100)
    ].copy()
    df_in["ssim2_int"] = df_in["ssim2_score"].round().astype(int).clip(0, 100)
    print(f"pairs in [0, 100] SSIM2 range: {len(df_in)}")

    # Group V0_4-ssim2 calibrated scores by integer SSIM2.
    grouped = df_in.groupby("ssim2_int")["v04_calibrated_score"]
    bins = sorted(grouped.groups.keys())
    box_data = [grouped.get_group(b).values for b in bins]
    n_per_bin = [len(g) for g in box_data]

    # The "calibration range" is the LARGEST contiguous run of bins
    # where every integer bucket has >= DENSITY_THRESHOLD pairs.
    DENSITY_THRESHOLD = 30
    full_bin_counts = {b: 0 for b in range(0, 101)}
    for b, n in zip(bins, n_per_bin):
        full_bin_counts[b] = n
    contiguous = []
    cur_start = None
    for b in range(0, 101):
        if full_bin_counts[b] >= DENSITY_THRESHOLD:
            if cur_start is None:
                cur_start = b
        else:
            if cur_start is not None:
                contiguous.append((cur_start, b - 1))
                cur_start = None
    if cur_start is not None:
        contiguous.append((cur_start, 100))
    if contiguous:
        # Pick the longest contiguous run.
        first_supported, last_supported = max(contiguous, key=lambda r: r[1] - r[0])
    else:
        first_supported = last_supported = None
    print(
        f"contiguous well-supported range (≥{DENSITY_THRESHOLD} pairs/bin): "
        f"[{first_supported}, {last_supported}] "
        f"(of {len(contiguous)} contiguous segments: {contiguous})"
    )

    # ---- Plot ----
    fig, (ax_main, ax_count) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    # Highlight calibration-supported range.
    if first_supported is not None and last_supported is not None:
        ax_main.axvspan(
            first_supported - 0.5, last_supported + 0.5,
            alpha=0.10, color="green",
            label=f"human-data calibration range "
                  f"(SSIM2 {first_supported}–{last_supported}, every integer bucket ≥{DENSITY_THRESHOLD} pairs)",
        )

    # Boxplots of V0_4-ssim2 calibrated score per integer SSIM2 bucket.
    bp = ax_main.boxplot(
        box_data,
        positions=bins,
        widths=0.7,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )
    for box in bp["boxes"]:
        box.set_facecolor("#3a86ff")
        box.set_edgecolor("#1d3557")
        box.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("#e63946")
        med.set_linewidth(1.6)

    # Identity line (V0_4-ssim2 calibrated == SSIM2).
    ax_main.plot(
        [0, 100], [0, 100],
        color="#888", linestyle="--", linewidth=1.0,
        label="identity (V0_4 = SSIM2)",
    )

    ax_main.set_ylabel("V0_4-ssim2 calibrated score")
    ax_main.set_title(
        "V0_4-ssim2 (piecewise-21 calibrated) per-bucket distribution vs SSIMULACRA 2 score\n"
        "Box = IQR (25–75%), whiskers = 5–95%, red = median; "
        "n = 17,417 pairs across KADID + TID + CID22-49"
    )
    ax_main.set_xlim(-1, 101)
    # Crop to the score range actually populated to give the chart breathing room.
    ax_main.set_ylim(-30, 110)
    ax_main.set_xticks(range(0, 101, 5))
    ax_main.grid(True, axis="y", alpha=0.3)
    ax_main.legend(loc="lower right")

    # Bin count chart underneath.
    ax_count.bar(bins, n_per_bin, width=0.9, color="#457b9d", alpha=0.8)
    ax_count.axhline(DENSITY_THRESHOLD, color="green", linestyle=":",
                     linewidth=1.0, label=f"density threshold ({DENSITY_THRESHOLD})")
    ax_count.set_ylabel("# pairs / bin")
    ax_count.set_xlabel("SSIMULACRA 2 score (integer bucket)")
    ax_count.set_yscale("log")
    ax_count.grid(True, axis="y", alpha=0.3)
    ax_count.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/per_pair_v04_ssim2_holdout.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "/mnt/v/output/zensim/whisker_v04_vs_ssim2.png"
    main(csv, out)
