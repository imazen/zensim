#!/usr/bin/env python3
"""CID22 paper-style 4-panel per-band TV trend plot.

Each panel = one perceptibility band (B0/B1/B2/B3).
X axis = TV regularizer weight.
Lines: h=64 KonJND-aligned, h=128 KonJND-aligned, V0_5 baseline (horizontal).

Output: /mnt/v/output/zensim/cycle_2026-05-11/per_band_tv_trends.png
"""
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

# Per-band CID22 SROCC (V0_4 column from dataset_metric_baseline full evals)
# Format: TV → (B0, B1, B2, B3)
H64 = {
    5:  (0.4195, 0.4532, 0.7736, 0.1641),
    10: (0.4234, 0.4260, 0.7680, 0.1403),
    20: (0.3792, 0.4310, 0.7645, 0.1133),
}
H128 = {
    5:  (0.4287, 0.4419, 0.7751, 0.1578),
    10: (0.4301, 0.4363, 0.7808, 0.1780),
    30: (0.4081, 0.4317, 0.7604, 0.1638),
}
H192 = {
    10: (0.4413, 0.4487, 0.7686, 0.1809),
}
V0_5 = (0.4396, 0.4488, 0.7746, 0.0642)

BAND_LABELS = ["B0 below medium (<50)\nlow quality",
               "B1 medium [50,65)\nvisible artifacts",
               "B2 high [65,90)\nsubtle artifacts",
               "B3 visually-lossless (≥90)\nn=43 (CI wide)"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

COLORS_H = {64: "#d62728", 128: "#9467bd", 192: "#8c564b"}

for b_idx, ax in enumerate(axes):
    h64_tv = sorted(H64.keys())
    h64_vals = [H64[t][b_idx] for t in h64_tv]
    h128_tv = sorted(H128.keys())
    h128_vals = [H128[t][b_idx] for t in h128_tv]
    h192_tv = sorted(H192.keys())
    h192_vals = [H192[t][b_idx] for t in h192_tv]

    ax.plot(h64_tv, h64_vals, "o-", color=COLORS_H[64], label="h=64 KonJND",
            markersize=9, linewidth=2, markeredgecolor="black", markeredgewidth=0.6)
    ax.plot(h128_tv, h128_vals, "s-", color=COLORS_H[128], label="h=128 KonJND",
            markersize=9, linewidth=2, markeredgecolor="black", markeredgewidth=0.6)
    ax.scatter(h192_tv, h192_vals, color=COLORS_H[192], marker="^", s=120,
               edgecolor="black", linewidths=0.6, label="h=192 KonJND")

    ax.axhline(V0_5[b_idx], color="#1f77b4", linestyle="--", linewidth=1.5,
               label="V0_5 shipped (no KonJND)")

    ax.set_xlabel("TV regularizer weight")
    ax.set_ylabel(f"CID22 SROCC (n=4292)")
    ax.set_title(BAND_LABELS[b_idx])
    ax.set_xlim(0, 35)
    if b_idx == 0:
        ax.legend(loc="lower left", fontsize=8)

fig.suptitle("Per-band CID22 SROCC vs TV regularizer weight (KonJND-aligned recipe)\n"
             "h=128 mostly above V0_5 baseline at B2 + B3; below at B0 + B1",
             fontsize=12, y=1.02)
fig.tight_layout()

OUT = pathlib.Path("/mnt/v/output/zensim/cycle_2026-05-11/per_band_tv_trends.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print(f"Wrote {OUT}")
print(f"  size {OUT.stat().st_size / 1024:.0f} KB")
