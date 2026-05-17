#!/usr/bin/env python3
"""Pathology plots: per-bin SROCC at 5-step granularity, scatters, residuals.

For each of V0_5 shipped and TV=10 h128 KonJND-aligned, on CID22 + JPEG synth:
  1. Bin ground truth at 5-unit intervals across 0-100
  2. SROCC within each bin (requires n >= 5)
  3. Bar chart of per-bin SROCC
  4. Scatter (truth vs predicted-score) with bin coloring
  5. Residual plot (pred_score − truth vs truth)

OUT: /mnt/v/output/zensim/cycle_2026-05-11/pathology_*.png
"""
import csv
import pathlib
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr
import pyarrow.parquet as pq

OUT = pathlib.Path("/mnt/v/output/zensim/cycle_2026-05-11")

# CID22 paper-style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

V0_5_BIN  = pathlib.Path("/home/lilith/work/zen/zensim/zensim/weights/v0_4_2026-04-30.bin")
TV10_BIN  = pathlib.Path("/home/lilith/work/zen/zensim/benchmarks/rust_v05recipe_konjnd_tv10_h128_seed1_2026-05-11.bin")
JPEG_PQ   = pathlib.Path("/mnt/v/zen/zensim-training/2026-05-07/unified/unified_v15r_zenjpeg.parquet")
V0_5_CID  = pathlib.Path("/tmp/zensim_loop/v0_5_per_pair_cid22.csv")
TV10_CID  = pathlib.Path("/tmp/zensim_loop/tv10_h128_per_pair_cid22.csv")

# ---------- ZNPR v2 parser (from score_unified_with_bake.py) ----------
LAYER_ENTRY_SIZE = 48
def parse_bake_v2(path):
    data = path.read_bytes()
    assert data[0:4] == b"ZNPR"
    version = struct.unpack("<H", data[4:6])[0]
    assert version == 2
    n_inputs = struct.unpack("<I", data[8:12])[0]
    n_layers = struct.unpack("<I", data[16:20])[0]
    scaler_mean_off = struct.unpack("<I", data[32:36])[0]
    scaler_scale_off = struct.unpack("<I", data[40:44])[0]
    layer_table_off = struct.unpack("<I", data[48:52])[0]
    scaler_mean = np.frombuffer(data, dtype=np.float32, count=n_inputs, offset=scaler_mean_off).copy()
    scaler_scale = np.frombuffer(data, dtype=np.float32, count=n_inputs, offset=scaler_scale_off).copy()
    layers = []
    for i in range(n_layers):
        e_off = layer_table_off + i * LAYER_ENTRY_SIZE
        in_dim, out_dim = struct.unpack("<II", data[e_off:e_off+8])
        activation = data[e_off+8]
        w_off, w_len = struct.unpack("<II", data[e_off+12:e_off+20])
        b_off, b_len = struct.unpack("<II", data[e_off+28:e_off+36])
        # row-major (in_dim, out_dim)
        w = np.frombuffer(data, dtype=np.float32, count=in_dim*out_dim, offset=w_off).reshape(in_dim, out_dim).copy()
        b = np.frombuffer(data, dtype=np.float32, count=out_dim, offset=b_off).copy()
        layers.append((w, b, activation))
    return scaler_mean, scaler_scale, layers

def predict(features, scaler_mean, scaler_scale, layers):
    x = (features - scaler_mean[None, :]) / scaler_scale[None, :]
    for i, (w, b, _) in enumerate(layers):
        x = x @ w + b[None, :]
        if i < len(layers) - 1:
            x = np.where(x > 0, x, 0.01 * x)
    return x.squeeze(-1) if x.ndim > 1 else x

# ---------- Load CID22 per-pair ----------
def load_cid22(path):
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                rows.append((float(r["human_score"]) * 100, float(r["v04_distance"])))
            except (ValueError, KeyError):
                continue
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]

# ---------- Synth predictions ----------
print("Loading JPEG synth parquet...")
tbl = pq.read_table(JPEG_PQ, columns=["score_ssim2"] + [f"feat_{i}" for i in range(228)])
synth_truth = tbl.column("score_ssim2").to_numpy()
feat_cols = [tbl.column(f"feat_{i}").to_numpy() for i in range(228)]
synth_features = np.stack(feat_cols, axis=1).astype(np.float32)
print(f"  {len(synth_truth):,} synth pairs, features shape {synth_features.shape}")

# Subsample synth for speed (30k random)
rng = np.random.default_rng(42)
n_synth = min(30000, len(synth_truth))
idx = rng.choice(len(synth_truth), size=n_synth, replace=False)
synth_truth_sub = synth_truth[idx]
synth_features_sub = synth_features[idx]
print(f"  Subsampled to {n_synth:,} for plotting speed")

# Score with both bakes
print("Loading bakes...")
v05_params = parse_bake_v2(V0_5_BIN)
tv10_params = parse_bake_v2(TV10_BIN)

print("Scoring synth with V0_5...")
synth_v05 = predict(synth_features_sub, *v05_params)
print("Scoring synth with TV=10 h128...")
synth_tv10 = predict(synth_features_sub, *tv10_params)

# ---------- CID22 per-pair ----------
print("Loading CID22 per-pair...")
cid_truth, cid_v05 = load_cid22(V0_5_CID)
_, cid_tv10 = load_cid22(TV10_CID)

# Convert distance to score: higher = more similar. Use 100 - 18·|d|^0.7 as in zensim.
def dist_to_score(d):
    return 100 - 18 * (np.abs(d) ** 0.7) * np.sign(d)  # signed power for negative dists

def dist_to_score_unsigned(d):
    # Simpler: just negate distance for sign so higher = better.
    return -d

# For the scatter, plot raw distance vs truth — that's the visual story.
# For SROCC, the sign matters but only consistently within one model.

# Datasets we plot:
datasets = {
    "CID22 (n={})".format(len(cid_truth)): {
        "truth": cid_truth,
        "v05":   cid_v05,   # higher truth = better; v05 is signed distance
        "tv10":  cid_tv10,
    },
    "JPEG synth (n={})".format(n_synth): {
        "truth": synth_truth_sub,
        "v05":   synth_v05,
        "tv10":  synth_tv10,
    },
}

# ---------- 5-step per-bin SROCC ----------
def per_bin_srocc(truth, pred, n_bins=20, low=0, high=100):
    """20 bins of 5 units each; |SROCC| per bin."""
    edges = np.linspace(low, high, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sroccs = []
    counts = []
    for i in range(n_bins):
        mask = (truth >= edges[i]) & (truth < edges[i+1])
        n = mask.sum()
        counts.append(n)
        if n < 5:
            sroccs.append(np.nan)
        else:
            rho, _ = spearmanr(pred[mask], truth[mask])
            sroccs.append(abs(rho) if not np.isnan(rho) else np.nan)
    return centers, np.array(sroccs), np.array(counts)

# ---------- Plot 1: per-bin SROCC ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), sharey=True)

for col, (ds_name, d) in enumerate(datasets.items()):
    truth = d["truth"]
    for row, (mname, pred, color) in enumerate([
        ("V0_5 shipped", d["v05"], "#1f77b4"),
        ("TV=10 h128 KonJND", d["tv10"], "#9467bd"),
    ]):
        ax = axes[row, col]
        centers, sroccs, counts = per_bin_srocc(truth, pred)
        bars = ax.bar(centers, sroccs, width=4.5, color=color, edgecolor="black",
                      linewidth=0.4, alpha=0.85)
        for c, s, n in zip(centers, sroccs, counts):
            if not np.isnan(s):
                ax.text(c, s + 0.02, f"{n}", ha="center", fontsize=6.5, color="gray")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.axhline(0.5, color="red", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.axhline(0.8, color="green", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_title(f"{mname} — {ds_name}")
        if col == 0:
            ax.set_ylabel("|SROCC| (within-bin)")
        if row == 1:
            ax.set_xlabel("Ground truth bin (5-unit, 0-100 scale)")

fig.suptitle("Per-5-unit-bin SROCC — pathology view\n"
             "Red line=0.5 (random+), green=0.8 (acceptable); bin-n shown above bar",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "pathology_per_bin_srocc.png")
plt.close(fig)
print("Wrote pathology_per_bin_srocc.png")

# ---------- Plot 2: full scatters ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for col, (ds_name, d) in enumerate(datasets.items()):
    truth = d["truth"]
    for row, (mname, pred, color) in enumerate([
        ("V0_5 shipped", d["v05"], "#1f77b4"),
        ("TV=10 h128 KonJND", d["tv10"], "#9467bd"),
    ]):
        ax = axes[row, col]
        ax.scatter(truth, pred, c=color, s=3, alpha=0.2, edgecolors="none")
        # Overlay binned median
        centers, _, counts = per_bin_srocc(truth, pred)
        medians = []
        for i, c in enumerate(centers):
            mask = (truth >= c - 2.5) & (truth < c + 2.5)
            medians.append(np.median(pred[mask]) if mask.sum() >= 5 else np.nan)
        ax.plot(centers, medians, "o-", color="black", markersize=4, linewidth=1.2,
                label="bin median", alpha=0.85)
        ax.set_xlim(-3, 103)
        ax.set_xlabel(f"Ground truth ({ds_name.split(' ')[0]})")
        ax.set_ylabel(f"{mname} distance" if "shipped" in mname or "TV" in mname else "score")
        ax.set_title(f"{mname} — {ds_name}")
        ax.legend(loc="best", fontsize=8)

fig.suptitle("Full scatter: model prediction (distance) vs ground truth\n"
             "Bin-median line shows central tendency per 5-unit bin",
             fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig(OUT / "pathology_scatter.png")
plt.close(fig)
print("Wrote pathology_scatter.png")

# ---------- Plot 3: residuals (synth only, signed) ----------
# Convert prediction to score-space for residuals. For synth, ground truth is 0-100;
# our predictions are signed distance. Compute rank-based residual: percentile_rank(pred) - percentile_rank(truth).
def percentile_rank(x):
    order = np.argsort(x)
    rank = np.empty_like(order, dtype=float)
    rank[order] = np.arange(len(x))
    return 100.0 * rank / max(len(x) - 1, 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for col, (ds_name, d) in enumerate(datasets.items()):
    truth = d["truth"]
    ax = axes[col]
    for mname, pred, color in [
        ("V0_5 shipped", d["v05"], "#1f77b4"),
        ("TV=10 h128 KonJND", d["tv10"], "#9467bd"),
    ]:
        # Convert pred to percentile so residual is in same scale as truth
        # For distance-output predictions: HIGHER distance = LOWER quality, so flip.
        pred_pctile = 100 - percentile_rank(pred)
        residual = pred_pctile - truth
        # Plot binned mean residual
        centers, _, _ = per_bin_srocc(truth, pred)
        means = []
        stds = []
        for i, c in enumerate(centers):
            mask = (truth >= c - 2.5) & (truth < c + 2.5)
            if mask.sum() >= 5:
                means.append(residual[mask].mean())
                stds.append(residual[mask].std())
            else:
                means.append(np.nan)
                stds.append(np.nan)
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(centers, means, yerr=stds, fmt="o-", color=color, markersize=5,
                    linewidth=1.2, capsize=3, alpha=0.85, label=mname)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Ground truth bin ({ds_name.split(' ')[0]})")
    if col == 0:
        ax.set_ylabel("Residual: (100 − pred percentile) − truth\n(neg = predicts lower quality than reality)")
    ax.set_title(ds_name)
    ax.legend(loc="best", fontsize=9)

fig.suptitle("Per-bin residual: rank-based prediction vs ground truth\n"
             "Where residual ≠ 0, the model systematically over/under-predicts that band",
             fontsize=11, y=1.04)
fig.tight_layout()
fig.savefig(OUT / "pathology_residuals.png")
plt.close(fig)
print("Wrote pathology_residuals.png")

print("\nAll 3 pathology plots saved to", OUT)
