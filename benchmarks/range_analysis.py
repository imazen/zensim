#!/usr/bin/env python3
"""Per-quality-range performance breakdown for the 4 metrics.

For each metric, bucket pairs by ground-truth ssim2_score into bands
and compute SROCC, MAE, and dynamic-range smoothness within each band.
This shows where the metric breaks down — the 25-60 SSIM2 band is the
operationally-critical range for web compression.

Two views:
1. Synthetic ground-truth: predicted_distance vs gpu_ssim2 score, per
   ssim2 band. 218k pairs.
2. Human MOS: predicted_distance vs human_score per ssim2 band, on
   KADID/TID/CID22 holdouts. ~4500 pairs (1500/dataset).
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BANDS = [
    ("≤ 0",      lambda s: s < 0),
    ("0–25",     lambda s: 0 <= s < 25),
    ("25–40",    lambda s: 25 <= s < 40),
    ("40–60",    lambda s: 40 <= s < 60),
    ("60–75",    lambda s: 60 <= s < 75),
    ("75–90",    lambda s: 75 <= s < 90),
    ("≥ 90",     lambda s: s >= 90),
]
BAND_LABELS = [b[0] for b in BANDS]


def per_band(df, score_col):
    """Return per-band SROCC of predicted_distance vs ssim2_score (synth)
    or vs human_score (human MOS). Higher predicted_distance = lower
    quality, so we expect *negative* SROCC vs ssim2/human; report |SROCC|."""
    rows = []
    for label, pred in BANDS:
        mask = df["ssim2_score"].apply(pred) if score_col == "ssim2_score" else df["fast_ssim2_score"].apply(pred)
        sub = df[mask]
        n = len(sub)
        if n < 30:
            rows.append({"band": label, "n": n, "srocc": float("nan"),
                         "mae_vs_ssim2": float("nan"), "pred_p05": float("nan"),
                         "pred_p95": float("nan")})
            continue
        target = sub[score_col].values
        pred_d = sub["predicted_distance"].values if "predicted_distance" in sub.columns else sub["v04_distance"].values
        srocc = abs(stats.spearmanr(pred_d, target).correlation)
        # For synthetic: MAE between (-pred_d) scaled to ssim2 range.
        # Calibrate by linear fit per band so the MAE reflects in-band fit
        # rather than a global calibration offset.
        if score_col == "ssim2_score":
            # Fit pred_d = a*ssim2 + b in this band (per-band recalibration)
            slope, intercept, *_ = stats.linregress(target, pred_d)
            ssim2_hat = (pred_d - intercept) / slope if slope != 0 else target
            mae = float(np.mean(np.abs(ssim2_hat - target)))
        else:
            mae = float("nan")
        rows.append({"band": label, "n": n, "srocc": srocc,
                     "mae_vs_ssim2": mae,
                     "pred_p05": float(np.percentile(pred_d, 5)),
                     "pred_p95": float(np.percentile(pred_d, 95))})
    return rows


def show_synth_analysis():
    print("\n## Synthetic q-sweeps: predicted_distance vs gpu_ssim2 ground truth\n")
    print("|SROCC| within each SSIM2 band (lower = worse local ranking).")
    print("MAE = post-calibration absolute error in SSIM2 units within the band.")
    print("Higher MAE = the metric's slope isn't faithful inside that band.\n")
    metric_files = {
        "V0_4-smooth": "/tmp/synth_scored/v04smooth.csv",
        "V0_5":        "/tmp/synth_scored/v05.csv",
        "V0_4-smooth-konjnd-train": "/tmp/synth_scored/v04smoothk.csv",
        "V0_6 dct_hf": "/tmp/synth_scored/v06.csv",
    }
    summary = {}
    for name, p in metric_files.items():
        df = pd.read_csv(p)
        summary[name] = per_band(df, "ssim2_score")

    # SROCC table.
    print("### |SROCC| vs ssim2_score within each band\n")
    print("| metric \\ band | " + " | ".join(BAND_LABELS) + " |")
    print("|---|" + "--:|" * len(BAND_LABELS))
    for name in summary:
        cells = []
        for r in summary[name]:
            cells.append(f"{r['srocc']:.4f}" if not np.isnan(r["srocc"]) else "—")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n### Pair count per band (constant across metrics):\n")
    print("| band | " + " | ".join(BAND_LABELS) + " |")
    print("|---|" + "--:|" * len(BAND_LABELS))
    cells = [str(r["n"]) for r in summary["V0_5"]]
    print("| pairs | " + " | ".join(cells) + " |")
    pct = [f"{100*r['n']/sum(rr['n'] for rr in summary['V0_5']):.1f}%" for r in summary["V0_5"]]
    print("| % of training | " + " | ".join(pct) + " |")

    # MAE within band (after per-band calibration).
    print("\n### Per-band-calibrated MAE (in SSIM2 units; lower = more faithful slope)\n")
    print("| metric \\ band | " + " | ".join(BAND_LABELS) + " |")
    print("|---|" + "--:|" * len(BAND_LABELS))
    for name in summary:
        cells = []
        for r in summary[name]:
            cells.append(f"{r['mae_vs_ssim2']:.2f}" if not np.isnan(r["mae_vs_ssim2"]) else "—")
        print(f"| {name} | " + " | ".join(cells) + " |")


def show_human_analysis():
    print("\n## Human-MOS holdouts: predicted vs human, bucketed by SSIM2 ground truth\n")
    print("|SROCC| of predicted_distance vs human_score within each SSIM2 band.\n")
    metric_files = {
        "V0_4-smooth": "/home/lilith/work/zen/zensim--v04-mlp/benchmarks/v04smooth_perpair_2026-05-01.csv",
        "V0_5":        "/home/lilith/work/zen/zensim--v04-mlp/benchmarks/v05_perpair_2026-05-01.csv",
        "V0_4-smooth-konjnd-train": "/home/lilith/work/zen/zensim--v04-mlp/benchmarks/v04smooth_konjnd_train_perpair_2026-05-01.csv",
        "V0_6 dct_hf": "/home/lilith/work/zen/zensim--v04-mlp/benchmarks/v06_dct_hf_perpair_2026-05-01.csv",
    }
    by_metric = {}
    for name, p in metric_files.items():
        df = pd.read_csv(p)
        # rename predicted column for compat
        df = df.rename(columns={"v04_distance": "predicted_distance"})
        by_metric[name] = df

    # Compute per-band SROCC per (metric, dataset).
    print("\n### Per-band |SROCC| vs human_score (averaged across KADID/TID/CID22)\n")
    print("| metric \\ band | " + " | ".join(BAND_LABELS) + " |")
    print("|---|" + "--:|" * len(BAND_LABELS))
    for name, df in by_metric.items():
        cells = []
        for label, pred in BANDS:
            ssim2 = df["fast_ssim2_score"]
            mask = ssim2.apply(pred)
            sub = df[mask]
            if len(sub) < 30:
                cells.append("—")
                continue
            srocc = abs(stats.spearmanr(sub["predicted_distance"], sub["human_score"]).correlation)
            cells.append(f"{srocc:.3f}" if not np.isnan(srocc) else "—")
        print(f"| {name} | " + " | ".join(cells) + " |")
    # Reference SSIM2 in same bands.
    df0 = next(iter(by_metric.values()))
    print()
    print("References (same image pairs):")
    cells_ssim2 = []
    cells_butter = []
    for label, pred in BANDS:
        mask = df0["fast_ssim2_score"].apply(pred)
        sub = df0[mask]
        n = len(sub)
        if n < 30:
            cells_ssim2.append(f"— ({n})")
            cells_butter.append(f"— ({n})")
        else:
            r1 = abs(stats.spearmanr(sub["fast_ssim2_score"], sub["human_score"]).correlation)
            r2 = abs(stats.spearmanr(sub["butter_3norm"], sub["human_score"]).correlation)
            cells_ssim2.append(f"{r1:.3f}")
            cells_butter.append(f"{r2:.3f}")
    print("| ref SSIMULACRA 2 | " + " | ".join(cells_ssim2) + " |")
    print("| ref Butteraugli 3-norm | " + " | ".join(cells_butter) + " |")
    # Pair count per band.
    print()
    print("Holdout pair count per band:")
    cells_n = []
    for label, pred in BANDS:
        mask = df0["fast_ssim2_score"].apply(pred)
        cells_n.append(str(int(mask.sum())))
    print("| band pairs | " + " | ".join(cells_n) + " |")


if __name__ == "__main__":
    show_synth_analysis()
    show_human_analysis()
