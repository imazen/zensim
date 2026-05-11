#!/usr/bin/env python3
"""Generate CID22-paper-style plots from this cycle's bake measurements.

Output: /mnt/v/output/zensim/cycle_2026-05-11/*.png
Plots:
  1. pareto_scatter.png — Non-mono q-step rate vs CID22 SROCC (the main story)
  2. cid22_per_band.png — CID22 per-band SROCC, grouped bars across bakes
  3. tv_curve.png — CID22 vs TV at h=64 and h=128 (TV=10 h128 sweet spot)
  4. capacity_scaling.png — CID22 + val_mean vs hidden width at fixed TV=10
  5. dataset_aggregate.png — KADID / TID / CID22 aggregate bars for V0_5 vs TV=10 h128
"""
import re
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

OUT_DIR = pathlib.Path("/mnt/v/output/zensim/cycle_2026-05-11")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match CID22 paper aesthetic: sans-serif, clean grid, saturated colors
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# Color palette inspired by Cloudinary CID22 paper figures
COLORS = {
    "V0_5": "#1f77b4",        # blue
    "TV=0":  "#2ca02c",        # green
    "TV=5":  "#ff7f0e",        # orange
    "TV=10 h64": "#d62728",    # red
    "TV=10 h128": "#9467bd",   # purple
    "TV=10 h192": "#8c564b",   # brown
    "TV=20": "#e377c2",        # pink
    "TV=30": "#7f7f7f",        # gray
    "h128 WebP-mono": "#bcbd22", # olive
    "TV=5 h128": "#17becf",     # cyan
    "fast-ssim2": "#888888",    # gray reference
    "V0_2": "#cccccc",          # light gray
}

# All measured bakes from this cycle (CID22-sorted)
BAKES = [
    # (label, CID22, non-mono%, KADID, TID, hidden, TV, marker_color)
    ("h128 WebP-mono no-TV",   0.8941, 6.72, None,   None,   128, 0,    COLORS["h128 WebP-mono"]),
    ("TV=0 h64 KonJND",        0.8921, 5.46, 0.9395, 0.9490, 64,  0,    COLORS["TV=0"]),
    ("TV=10 h128 KonJND",      0.8900, 5.36, 0.9434, 0.9553, 128, 10,   COLORS["TV=10 h128"]),
    ("V0_5 shipped",           0.8893, 4.57, 0.8432, 0.8401, 64,  None, COLORS["V0_5"]),
    ("TV=5 h64 KonJND",        0.8880, 5.14, 0.9449, 0.9536, 64,  5,    COLORS["TV=5"]),
    ("TV=5 h128 KonJND",       0.8871, 5.44, 0.9434, 0.9540, 128, 5,    COLORS["TV=5 h128"]),
    ("TV=10 h192 KonJND",      0.8859, 5.80, 0.9424, 0.9531, 192, 10,   COLORS["TV=10 h192"]),
    ("TV=10 h64 KonJND",       0.8841, 5.09, 0.9380, 0.9437, 64,  10,   COLORS["TV=10 h64"]),
    ("TV=20 h64 KonJND",       0.8812, 5.29, 0.9318, 0.9409, 64,  20,   COLORS["TV=20"]),
    ("TV=30 h128 KonJND",      0.8803, 5.39, 0.9397, 0.9482, 128, 30,   COLORS["TV=30"]),
]

# Per-band CID22 SROCC: (B0_<50, B1_50-65, B2_65-90, B3_>=90)
# Source: per-band tables extracted in Ticks 95, 99
PER_BAND_CID22 = {
    "V0_5 shipped":      (0.4396, 0.4488, 0.7746, 0.0642),
    "TV=5 h64 KonJND":   (0.4195, 0.4532, 0.7736, 0.1641),
    "TV=10 h64 KonJND":  (0.4234, 0.4260, 0.7680, 0.1403),
    "TV=20 h64 KonJND":  (0.3792, 0.4310, 0.7645, 0.1133),
    "TV=10 h128 KonJND": (0.4301, 0.4363, 0.7808, 0.1780),
    "TV=30 h128 KonJND": (0.4081, 0.4317, 0.7604, 0.1638),
}
PER_BAND_KADID = {
    "V0_5 shipped":      (0.6636, 0.3355, 0.2135, 0.2420),
    "TV=5 h64 KonJND":   (0.8866, 0.4291, 0.2357, 0.2480),
    "TV=10 h128 KonJND": (0.8858, 0.4201, 0.2359, 0.2522),
}
PER_BAND_TID = {
    "V0_5 shipped":      (0.7350, 0.3588, 0.1856, 0.2683),
    "TV=5 h64 KonJND":   (0.8893, 0.6396, 0.3641, 0.1176),
    "TV=10 h128 KonJND": (0.8931, 0.6359, 0.3648, 0.2214),
}
BAND_LABELS = ["B0\n<50\n(low quality)", "B1\n50-65\n(medium)", "B2\n65-90\n(high)", "B3\n≥90\n(visually-lossless)"]

# -----------------------------------------------------------------------------
# Plot 1: Pareto scatter — non-mono q-step rate vs CID22 SROCC
# -----------------------------------------------------------------------------
def plot_pareto():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, c, nm, k, t, h, tv, col in BAKES:
        marker = "o" if h == 64 else ("s" if h == 128 else "^")
        if label == "V0_5 shipped":
            marker = "*"
            size = 320
        elif "WebP-mono" in label:
            marker = "D"
            size = 120
        else:
            size = 120
        ax.scatter(nm, c, c=col, marker=marker, s=size, edgecolors="black",
                   linewidths=0.7, label=label, zorder=3)
    # target zones (filled)
    ax.axvspan(0, 4.86, alpha=0.08, color="green", zorder=1)
    ax.axhspan(0.8934, 0.95, alpha=0.08, color="green", zorder=1)
    ax.axvline(4.86, color="green", linestyle="--", linewidth=1, alpha=0.7,
               label="smoothness target (4.86%)")
    ax.axhline(0.8934, color="green", linestyle=":", linewidth=1, alpha=0.7,
               label="CID22 target (0.8934)")
    # double-target zone
    ax.fill_between([0, 4.86], 0.8934, 0.895, color="green", alpha=0.18, zorder=2)
    ax.text(2.0, 0.8945, "dual-target\nzone (empty)", color="darkgreen", fontsize=9,
            ha="center", style="italic")
    ax.set_xlabel("Non-monotonic q-step rate (%, JPEG unified parquet, lower is better)")
    ax.set_ylabel("CID22 SROCC (n=4292, higher is better)")
    ax.set_title("Pareto frontier: V0_5 only meets smoothness; no bake meets both targets\nCycle of 2026-05-11, all measured bakes")
    ax.set_xlim(4.3, 7.0)
    ax.set_ylim(0.878, 0.898)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95)
    fig.tight_layout()
    out = OUT_DIR / "pareto_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    return out

# -----------------------------------------------------------------------------
# Plot 2: CID22 per-band grouped bar
# -----------------------------------------------------------------------------
def plot_per_band_cid22():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bakes_to_plot = ["V0_5 shipped", "TV=10 h128 KonJND", "TV=5 h64 KonJND"]
    x = np.arange(4)
    width = 0.27
    offsets = [-width, 0, width]
    for i, bake in enumerate(bakes_to_plot):
        col = COLORS["V0_5"] if "V0_5" in bake else (
              COLORS["TV=10 h128"] if "h128" in bake else COLORS["TV=5"])
        bars = ax.bar(x + offsets[i], PER_BAND_CID22[bake], width, label=bake,
                      color=col, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, PER_BAND_CID22[bake]):
            ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.set_ylabel("CID22 SROCC (per band)")
    ax.set_title("CID22 SROCC per perceptibility band (Table 5 cutoffs)\nV0_5 wins B0/B1; TV=10 h128 wins B2/B3 (product-critical bands)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 0.95)
    fig.tight_layout()
    out = OUT_DIR / "cid22_per_band.png"
    fig.savefig(out)
    plt.close(fig)
    return out

# -----------------------------------------------------------------------------
# Plot 3: TV-curve — CID22 vs TV at h=64 and h=128
# -----------------------------------------------------------------------------
def plot_tv_curve():
    fig, ax = plt.subplots(figsize=(7, 5))
    h64_data = [(b[6], b[1]) for b in BAKES
                if b[5] == 64 and b[6] is not None and "KonJND" in b[0]]
    h128_data = [(b[6], b[1]) for b in BAKES
                 if b[5] == 128 and b[6] is not None and "KonJND" in b[0]]
    h64_data.sort(); h128_data.sort()
    h64_tv, h64_cid22 = zip(*h64_data)
    h128_tv, h128_cid22 = zip(*h128_data)
    ax.plot(h64_tv, h64_cid22, "o-", color=COLORS["TV=10 h64"], label="h=64 KonJND-aligned",
            markersize=9, linewidth=2, markeredgecolor="black", markeredgewidth=0.7)
    ax.plot(h128_tv, h128_cid22, "s-", color=COLORS["TV=10 h128"], label="h=128 KonJND-aligned",
            markersize=9, linewidth=2, markeredgecolor="black", markeredgewidth=0.7)
    # annotate TV=10 h128 as peak
    h128_peak_idx = list(h128_cid22).index(max(h128_cid22))
    ax.annotate("h=128 peak\n(TV=10)", xy=(h128_tv[h128_peak_idx], h128_cid22[h128_peak_idx]),
                xytext=(15, 12), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    # V0_5 baseline
    ax.axhline(0.8893, color=COLORS["V0_5"], linestyle="--", linewidth=1.2,
               label="V0_5 shipped (no KonJND)")
    ax.axhline(0.8934, color="green", linestyle=":", linewidth=1.2,
               label="CID22 target (0.8934)")
    ax.set_xlabel("TV regularizer weight")
    ax.set_ylabel("CID22 SROCC (n=4292)")
    ax.set_title("TV regularizer trades CID22 for smoothness\nh=64 peaks at TV=0; h=128 peaks at TV=10")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(-2, 35)
    ax.set_ylim(0.878, 0.898)
    fig.tight_layout()
    out = OUT_DIR / "tv_curve.png"
    fig.savefig(out)
    plt.close(fig)
    return out

# -----------------------------------------------------------------------------
# Plot 4: Capacity scaling — CID22 vs hidden width at TV=10
# -----------------------------------------------------------------------------
def plot_capacity():
    fig, ax = plt.subplots(figsize=(7, 5))
    # h=64, h=128, h=192 measured CID22 at TV=10 + KonJND
    cap_data = [(64, 0.8841), (128, 0.8900), (192, 0.8859)]
    h, c = zip(*cap_data)
    ax.plot(h, c, "o-", color=COLORS["TV=10 h128"], markersize=12,
            linewidth=2.2, markeredgecolor="black", markeredgewidth=0.8,
            label="measured CID22 (TV=10, KonJND-aligned)")
    # Annotate the peak
    ax.annotate("peak", xy=(128, 0.8900), xytext=(20, 12),
                textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.annotate("regression", xy=(192, 0.8859), xytext=(15, -25),
                textcoords="offset points", fontsize=9, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
    ax.axhline(0.8893, color=COLORS["V0_5"], linestyle="--", linewidth=1.2,
               label="V0_5 shipped baseline")
    ax.axhline(0.8934, color="green", linestyle=":", linewidth=1.2,
               label="CID22 target (0.8934)")
    ax.set_xlabel("Hidden layer width (RankNet, single layer)")
    ax.set_ylabel("CID22 SROCC (n=4292)")
    ax.set_title("Capacity peaks at h=128 — NOT monotonic, NOT saturating\nh=64→h=128 +0.0059; h=128→h=192 -0.0041 (regression)")
    ax.set_xticks([64, 128, 192])
    ax.set_xlim(40, 220)
    ax.set_ylim(0.878, 0.898)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "capacity_scaling.png"
    fig.savefig(out)
    plt.close(fig)
    return out

# -----------------------------------------------------------------------------
# Plot 5: Aggregate KADID/TID/CID22 — V0_5 vs TV=10 h128
# -----------------------------------------------------------------------------
def plot_dataset_aggregate():
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = ["KADIK10k", "TID2013", "CID22"]
    v05 = [0.8432, 0.8401, 0.8893]
    tv10_h128 = [0.9434, 0.9553, 0.8900]
    fast_ssim2 = [0.8133, 0.8460, 0.8895]  # CID22 SSIMULACRA2-tuned reference
    x = np.arange(3)
    width = 0.25
    bars1 = ax.bar(x - width, v05, width, label="V0_5 shipped",
                   color=COLORS["V0_5"], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, tv10_h128, width, label="TV=10 h128 KonJND",
                   color=COLORS["TV=10 h128"], edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, fast_ssim2, width, label="fast-ssim2 (CID22-tuned ref)",
                   color=COLORS["fast-ssim2"], edgecolor="black", linewidth=0.5)
    for bars, vals in [(bars1, v05), (bars2, tv10_h128), (bars3, fast_ssim2)]:
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.4f}", xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Aggregate SROCC")
    ax.set_title("Aggregate SROCC vs human MOS — TV=10 h128 KonJND vs V0_5\nTV=10 h128 dominates KADID/TID; CID22 essentially tied (+0.0007)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out = OUT_DIR / "dataset_aggregate.png"
    fig.savefig(out)
    plt.close(fig)
    return out

# -----------------------------------------------------------------------------
def main():
    outs = []
    outs.append(plot_pareto())
    outs.append(plot_per_band_cid22())
    outs.append(plot_tv_curve())
    outs.append(plot_capacity())
    outs.append(plot_dataset_aggregate())
    print("Generated plots:")
    for p in outs:
        sz = p.stat().st_size / 1024
        print(f"  {p} ({sz:.0f} KB)")

if __name__ == "__main__":
    main()
