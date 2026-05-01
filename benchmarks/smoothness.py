#!/usr/bin/env python3
"""Smoothness analysis of metric predictions on synthetic q-sweeps.

For each metric, group by (source_path, codec), sort by quality, and
compute on the predicted_distance vs quality curve:
- monotonicity_violation_rate: % of consecutive pairs where higher q gave
  WORSE metric (predicted_distance higher = worse, so monotonic means
  decreasing as q increases)
- normalized step jaggedness: stdev(consecutive |Δd|) / mean(consecutive |Δd|)
  → 0 means perfectly even steps, larger = more jagged
- max consecutive step: largest |Δd| between adjacent quality levels
- pearson(predicted, ssim2): does the predicted curve correlate with the
  ground-truth SSIM2 curve for the same source-codec sweep

Smoothness matters when the metric is a human-facing quality target: a
user picking q=80 expects a stable score that doesn't ZIGZAG with small
changes.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def analyze(csv_path: Path):
    df = pd.read_csv(csv_path)
    # SSIM2 in synthetic: higher = better. Predicted distance: higher = worse.
    # So a smooth metric should DECREASE as q (quality setting) increases for
    # the same (source, codec) — same direction as SSIM2 increases.
    out = {
        "violations": [],     # fraction of monotonicity violations per sweep
        "step_cv": [],        # CV of consecutive step sizes per sweep
        "max_step": [],       # max consecutive |Δ| per sweep
        "ssim2_pearson": [],  # |corr(predicted_distance, -ssim2_score)| per sweep — higher = closer to SSIM2 ground truth
        "n_sweeps": 0,
        "n_skipped": 0,
    }
    grouped = df.groupby(["source_path", "codec"])
    for _, g in grouped:
        if len(g) < 4:
            out["n_skipped"] += 1
            continue
        gs = g.sort_values("quality")
        d = gs["predicted_distance"].values
        ss = gs["ssim2_score"].values
        # Higher q should give lower distance. Count consecutive pairs where
        # d INCREASES as q increases — that's a monotonicity violation.
        deltas = np.diff(d)
        n_violations = int(np.sum(deltas > 0))
        violation_rate = n_violations / len(deltas)
        out["violations"].append(violation_rate)
        # Step size CV.
        abs_steps = np.abs(deltas)
        if abs_steps.mean() > 1e-9:
            out["step_cv"].append(abs_steps.std() / abs_steps.mean())
        # Max step.
        out["max_step"].append(float(abs_steps.max()))
        # Pearson with -SSIM2 (since high distance = low quality = low SSIM2).
        if len(d) >= 3:
            r = stats.pearsonr(d, -ss).statistic
            out["ssim2_pearson"].append(abs(r) if np.isfinite(r) else 0.0)
        out["n_sweeps"] += 1
    return out


def summary(metric, a):
    v = np.array(a["violations"])
    cv = np.array(a["step_cv"])
    ms = np.array(a["max_step"])
    sp = np.array(a["ssim2_pearson"])
    return {
        "metric": metric,
        "n_sweeps": a["n_sweeps"],
        "violation_rate_mean": float(v.mean()) if len(v) else float("nan"),
        "violation_rate_p95": float(np.percentile(v, 95)) if len(v) else float("nan"),
        "step_cv_mean": float(cv.mean()) if len(cv) else float("nan"),
        "step_cv_p95": float(np.percentile(cv, 95)) if len(cv) else float("nan"),
        "max_step_p50": float(np.median(ms)) if len(ms) else float("nan"),
        "max_step_p95": float(np.percentile(ms, 95)) if len(ms) else float("nan"),
        "ssim2_pearson_mean": float(sp.mean()) if len(sp) else float("nan"),
        "ssim2_pearson_p10": float(np.percentile(sp, 10)) if len(sp) else float("nan"),
    }


def main():
    metrics = {
        "V0_4-smooth": "/tmp/synth_scored/v04smooth.csv",
        "V0_5":        "/tmp/synth_scored/v05.csv",
        "V0_4-smooth-konjnd-train": "/tmp/synth_scored/v04smoothk.csv",
        "V0_6 dct_hf": "/tmp/synth_scored/v06.csv",
    }
    rows = []
    for name, p in metrics.items():
        a = analyze(Path(p))
        rows.append(summary(name, a))
    print("# Smoothness analysis on synthetic q-sweeps\n")
    print("Per (source_path, codec) group, sorted by quality. Metrics:\n")
    print("- **violation_rate_mean**: % consecutive (q, q+) pairs where the model")
    print("  predicted MORE distance for the HIGHER quality (lower = better).")
    print("  Monotonicity violations are operationally bad: a user raising q")
    print("  expects a *better* quality score, not worse.")
    print("- **step_cv_mean**: stdev(|Δdist|) / mean(|Δdist|) per sweep")
    print("  (lower = more consistent step sizes → smoother gradient).")
    print("- **max_step_p95**: 95th-percentile of the largest single-step jump")
    print("  per sweep (lower = no nasty cliffs at random q).")
    print("- **ssim2_pearson_mean**: per-sweep |corr(predicted, -SSIM2)| —")
    print("  how closely each sweep tracks the ground-truth SSIM2 sweep")
    print("  for the same (source, codec). Higher = closer to SSIM2's behavior.\n")
    print("| metric | n_sweeps | violation_rate (mean / p95) | step_cv (mean / p95) | max_step (p50 / p95) | ssim2_pearson (mean / p10) |")
    print("|---|--:|--:|--:|--:|--:|")
    for r in rows:
        print(
            f"| {r['metric']} | {r['n_sweeps']} | "
            f"{r['violation_rate_mean']:.3f} / {r['violation_rate_p95']:.3f} | "
            f"{r['step_cv_mean']:.3f} / {r['step_cv_p95']:.3f} | "
            f"{r['max_step_p50']:.2f} / {r['max_step_p95']:.2f} | "
            f"{r['ssim2_pearson_mean']:.4f} / {r['ssim2_pearson_p10']:.4f} |"
        )

    # Per-codec breakdown for the violation rate, to spot whether one codec
    # is responsible for the jaggedness.
    print("\n## Violation rate by codec (mean across sweeps)\n")
    by_codec = {}
    for name, p in metrics.items():
        df = pd.read_csv(p)
        per_codec = {}
        for codec, g in df.groupby("codec"):
            sub = g.groupby("source_path")
            rates = []
            for _, src_g in sub:
                if len(src_g) < 4: continue
                src_g = src_g.sort_values("quality")
                d = src_g["predicted_distance"].values
                deltas = np.diff(d)
                rates.append(np.sum(deltas > 0) / len(deltas))
            if rates:
                per_codec[codec] = float(np.mean(rates))
        by_codec[name] = per_codec
    codec_keys = sorted({c for pc in by_codec.values() for c in pc})
    print("| metric \\ codec | " + " | ".join(c.replace('-v0.5.4','').replace('-v0.3.1','') for c in codec_keys) + " |")
    print("|---" + "|--:" * len(codec_keys) + "|")
    for name, pc in by_codec.items():
        cells = [f"{pc.get(c, float('nan')):.3f}" if c in pc else "—" for c in codec_keys]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
