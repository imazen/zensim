#!/usr/bin/env python3
"""CID22 paper-style scatter: predicted score vs human MOS.

For each pair in CID22, plot (model prediction) vs (MOS) for V0_5 and TV=10 h128.
This is the canonical "metric vs human" scatter from the CID22 paper.

Output: /mnt/v/output/zensim/cycle_2026-05-11/cid22_scatter.png
"""
import csv
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

V05_CSV = pathlib.Path("/tmp/zensim_loop/v0_5_per_pair_cid22.csv")
TV10_CSV = pathlib.Path("/tmp/zensim_loop/tv10_h128_per_pair_cid22.csv")
OUT = pathlib.Path("/mnt/v/output/zensim/cycle_2026-05-11/cid22_scatter.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

def load_pairs(path):
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                rows.append({
                    "mos": float(r["human_score"]),
                    "v04": float(r["v04_distance"]),
                    "v02": float(r["v02_distance"]),
                    "fast_ssim2": float(r["fast_ssim2_score"]),
                })
            except (ValueError, KeyError):
                continue
    return rows

v05_rows = load_pairs(V05_CSV)
tv10_rows = load_pairs(TV10_CSV)
print(f"V0_5: {len(v05_rows)} pairs; TV=10 h128: {len(tv10_rows)} pairs")

# 1x3 panel: V0_5, TV=10 h128, fast-ssim2 reference
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

# CID22 human_score column is RMOS (normalized 0-1, ≈ MCOS/100)
# v04_distance is the raw MLP output (signed; lower = more similar)
# Plot raw distance/score vs MOS; expect negative correlation for distance metrics

mos_v05  = np.array([r["mos"] for r in v05_rows])
v05_d    = np.array([r["v04"] for r in v05_rows])
mos_tv10 = np.array([r["mos"] for r in tv10_rows])
tv10_d   = np.array([r["v04"] for r in tv10_rows])
fast_ssim2 = np.array([r["fast_ssim2"] for r in v05_rows])

def plot_scatter(ax, mos, pred, title, color, ylabel, srocc):
    # Convert MOS to 0-100 MCOS for axis consistency with CID22 paper
    mcos = mos * 100
    ax.scatter(mcos, pred, c=color, s=4, alpha=0.3, edgecolors="none")
    ax.set_xlim(0, 100)
    ax.set_xlabel("CID22 Human MCOS (0-100)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # SROCC box top-left
    ax.text(0.03, 0.97, f"|SROCC| = {srocc:.4f}", transform=ax.transAxes,
            fontsize=10, weight="bold", verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))

plot_scatter(axes[0], mos_v05, v05_d,
             "V0_5 shipped\n(raw MLP distance, signed)",
             "#1f77b4", "V0_5 distance (lower = better)", 0.8893)
plot_scatter(axes[1], mos_tv10, tv10_d,
             "TV=10 h128 KonJND-aligned\n(raw MLP distance, signed)",
             "#9467bd", "TV=10 h128 distance (lower = better)", 0.8900)
plot_scatter(axes[2], mos_v05, fast_ssim2,
             "fast-ssim2 (CID22-tuned reference)\n(native ssim2 score, higher = better)",
             "#7f7f7f", "fast-ssim2 score (higher = better)", 0.8895)

fig.suptitle("CID22 validation set (n=4292): metric vs human MCOS — V0_5 vs TV=10 h128 vs fast-ssim2\nV0_5 / TV=10 h128 output negative distance; fast-ssim2 outputs positive score (note Y-axis polarity)",
             fontsize=11, y=1.04)

fig.tight_layout()
fig.savefig(OUT)
plt.close(fig)
print(f"Wrote {OUT}")
print(f"  size {OUT.stat().st_size / 1024:.0f} KB")
