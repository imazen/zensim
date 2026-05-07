#!/usr/bin/env python3
"""Side-by-side whisker chart of SSIMULACRA 2 and V0_4-ssim2 calibrated score
across V0_2 integer score buckets, so the user can see distribution overlap.

X axis: V0_2 score (integer 0..100), the independent reference.
Y axis: metric score.
Per X bucket: two box plots side-by-side — SSIM2 (orange) and V0_4-ssim2 (blue).
Identity line at Y = X = V0_2.
"""

import sys
import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Piecewise-21 calibration anchors from benchmarks/v04_ssim2_holdout_calibration_2026-05-01.md
ANCHORS = [
    (-59.8119, 100.0000), (-46.3979, 86.3832), (-39.8782, 82.4838),
    (-35.2807, 79.5798), (-30.7110, 76.5164), (-25.9624, 73.4357),
    (-20.0988, 69.9280), (-12.7256, 65.9895), (-3.0700, 61.2748),
    (4.6772, 55.6898),   (10.5595, 48.6972),  (17.3930, 40.9081),
    (24.9022, 32.4060),  (33.9656, 22.3016),  (45.3017, 11.4260),
    (57.9030, -0.1502),  (72.9362, -12.2887), (92.7248, -28.9030),
    (124.7492, -50.6577),(176.7873, -81.7489),(623.7492, -100.0000),
]
ANCHOR_X = [a[0] for a in ANCHORS]


def piecewise_score(d: float) -> float:
    if d <= ANCHOR_X[0]:
        return ANCHORS[0][1]
    if d >= ANCHOR_X[-1]:
        return ANCHORS[-1][1]
    i = bisect.bisect_right(ANCHOR_X, d)
    x0, y0 = ANCHORS[i - 1]
    x1, y1 = ANCHORS[i]
    t = (d - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def v02_score(distance: float) -> float:
    """V0_2 inherited score mapping: 100 - 18·d^0.7, clamped to [-100, 100]."""
    if distance <= 0:
        return 100.0
    s = 100.0 - 18.0 * (distance ** 0.7)
    return max(-100.0, min(100.0, s))


def main(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} pairs from {csv_path}")

    df["v02_score"] = df["v02_distance"].apply(v02_score)
    df["v04_calibrated_score"] = df["v04_distance"].apply(piecewise_score)
    df["ssim2_score"] = df["fast_ssim2_score"]

    # Drop KonJND-1k — its "human_score" is a PJND threshold, not MOS.
    df_human = df[df["dataset"] != "KonJND-1k"].copy()
    print(f"pairs with human MOS: {len(df_human)}")

    # X axis = V0_2 integer score in [0, 100].
    df_in = df_human[
        (df_human["v02_score"] >= 0) & (df_human["v02_score"] <= 100)
    ].copy()
    df_in["v02_int"] = df_in["v02_score"].round().astype(int).clip(0, 100)
    print(f"pairs with V0_2 score in [0, 100]: {len(df_in)}")

    # Group both metrics by V0_2 integer bucket.
    grp_ssim2 = df_in.groupby("v02_int")["ssim2_score"]
    grp_v04 = df_in.groupby("v02_int")["v04_calibrated_score"]
    bins = sorted(grp_ssim2.groups.keys())
    box_ssim2 = [grp_ssim2.get_group(b).values for b in bins]
    box_v04 = [grp_v04.get_group(b).values for b in bins]
    n_per_bin = [len(g) for g in box_ssim2]

    # Calibration support: contiguous buckets with ≥30 pairs.
    DENSITY_THRESHOLD = 30
    full_counts = {b: 0 for b in range(0, 101)}
    for b, n in zip(bins, n_per_bin):
        full_counts[b] = n
    contiguous = []
    cur_start = None
    for b in range(0, 101):
        if full_counts[b] >= DENSITY_THRESHOLD:
            if cur_start is None:
                cur_start = b
        else:
            if cur_start is not None:
                contiguous.append((cur_start, b - 1))
                cur_start = None
    if cur_start is not None:
        contiguous.append((cur_start, 100))
    if contiguous:
        first_supported, last_supported = max(contiguous, key=lambda r: r[1] - r[0])
    else:
        first_supported = last_supported = None
    print(
        f"contiguous calibration range (≥{DENSITY_THRESHOLD} pairs/bin): "
        f"[{first_supported}, {last_supported}] "
        f"(segments: {contiguous})"
    )

    # ---- Plot ----
    fig, (ax_main, ax_count) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    if first_supported is not None and last_supported is not None:
        ax_main.axvspan(
            first_supported - 0.5, last_supported + 0.5,
            alpha=0.10, color="green",
            label=f"calibration range "
                  f"(V0_2 {first_supported}–{last_supported}, ≥{DENSITY_THRESHOLD} pairs/bin)",
        )

    # Two side-by-side box plots per bucket: SSIM2 (left, orange) and V0_4-ssim2 (right, blue).
    # Use offset positions: SSIM2 at b-0.22, V0_4 at b+0.22.
    pos_ssim2 = [b - 0.22 for b in bins]
    pos_v04 = [b + 0.22 for b in bins]
    bp_ssim2 = ax_main.boxplot(
        box_ssim2, positions=pos_ssim2, widths=0.4,
        whis=(5, 95), showfliers=False, patch_artist=True, manage_ticks=False,
    )
    bp_v04 = ax_main.boxplot(
        box_v04, positions=pos_v04, widths=0.4,
        whis=(5, 95), showfliers=False, patch_artist=True, manage_ticks=False,
    )
    for box in bp_ssim2["boxes"]:
        box.set_facecolor("#ff8c42")
        box.set_edgecolor("#933a00")
        box.set_alpha(0.65)
    for med in bp_ssim2["medians"]:
        med.set_color("#7a1300")
        med.set_linewidth(1.4)
    for box in bp_v04["boxes"]:
        box.set_facecolor("#3a86ff")
        box.set_edgecolor("#1d3557")
        box.set_alpha(0.65)
    for med in bp_v04["medians"]:
        med.set_color("#0f1c40")
        med.set_linewidth(1.4)

    # Identity line: where V0_2 score == metric score.
    ax_main.plot(
        [0, 100], [0, 100], color="#888", linestyle="--", linewidth=1.0,
        label="identity (metric = V0_2)",
    )

    # Manual legend for the two box colors.
    legend_elems = [
        Patch(facecolor="#ff8c42", edgecolor="#933a00", alpha=0.65, label="SSIMULACRA 2"),
        Patch(facecolor="#3a86ff", edgecolor="#1d3557", alpha=0.65,
              label="V0_4-ssim2 (piecewise-21 calibrated)"),
        plt.Line2D([0], [0], color="#888", linestyle="--", label="identity (metric = V0_2)"),
        Patch(facecolor="green", alpha=0.10,
              label=f"calibration range (≥{DENSITY_THRESHOLD} pairs/bin)"),
    ]
    ax_main.legend(handles=legend_elems, loc="lower right")
    ax_main.set_ylabel("metric score")
    ax_main.set_title(
        "SSIMULACRA 2 vs V0_4-ssim2 calibrated score per V0_2 integer bucket\n"
        f"Box = IQR (25–75%), whiskers = 5–95%; n = {len(df_in)} pairs across "
        "KADID + TID + CID22-49"
    )
    ax_main.set_xlim(-1, 101)
    ax_main.set_ylim(-30, 110)
    ax_main.set_xticks(range(0, 101, 5))
    ax_main.grid(True, axis="y", alpha=0.3)

    # Bin count chart.
    ax_count.bar(bins, n_per_bin, width=0.9, color="#457b9d", alpha=0.8)
    ax_count.axhline(DENSITY_THRESHOLD, color="green", linestyle=":",
                     linewidth=1.0, label=f"density threshold ({DENSITY_THRESHOLD})")
    ax_count.set_ylabel("# pairs / bin")
    ax_count.set_xlabel("V0_2 score (integer bucket)")
    ax_count.set_yscale("log")
    ax_count.grid(True, axis="y", alpha=0.3)
    ax_count.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/per_pair_v04_ssim2_holdout.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "/mnt/v/output/zensim/whisker_overlap_v02axis.png"
    main(csv, out)
