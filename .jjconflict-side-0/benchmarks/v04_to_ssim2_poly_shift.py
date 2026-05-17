#!/usr/bin/env python3
"""Test whether a polynomial shift on top of the V0_2-anchored V0_4 score
can match SSIM2 with lower RMSE than fitting V0_4 directly to SSIM2.

The path under test:
  d_v04  --(V0_2 piecewise-21)--> z (V0_2-grade score)  --p(z)--> ssim2_hat

vs. direct:
  d_v04  --(SSIM2 piecewise-21)--> ssim2_hat
"""

import sys
import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# V0_2-target piecewise-21 anchors (from benchmarks/v04_ssim2_holdout_calibration_2026-05-01.md)
V02_ANCHORS = [
    (-59.8119, 100.0000), (-46.3979, 86.3832), (-39.8782, 82.4838),
    (-35.2807, 79.5798), (-30.7110, 76.5164), (-25.9624, 73.4357),
    (-20.0988, 69.9280), (-12.7256, 65.9895), (-3.0700, 61.2748),
    (4.6772, 55.6898),   (10.5595, 48.6972),  (17.3930, 40.9081),
    (24.9022, 32.4060),  (33.9656, 22.3016),  (45.3017, 11.4260),
    (57.9030, -0.1502),  (72.9362, -12.2887), (92.7248, -28.9030),
    (124.7492, -50.6577),(176.7873, -81.7489),(623.7492, -100.0000),
]
V02_X = [a[0] for a in V02_ANCHORS]


def v02_score_via_anchors(d):
    if d <= V02_X[0]:
        return V02_ANCHORS[0][1]
    if d >= V02_X[-1]:
        return V02_ANCHORS[-1][1]
    i = bisect.bisect_right(V02_X, d)
    x0, y0 = V02_ANCHORS[i - 1]
    x1, y1 = V02_ANCHORS[i]
    t = (d - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def main(csv_path, out_path):
    df = pd.read_csv(csv_path)
    df_human = df[df["dataset"] != "KonJND-1k"].copy()
    print(f"loaded {len(df_human)} human-rated pairs")

    # Compute V0_4 score via V0_2 anchors (the prior calibration).
    df_human["v04_via_v02"] = df_human["v04_distance"].apply(v02_score_via_anchors)
    z = df_human["v04_via_v02"].values
    y = df_human["fast_ssim2_score"].values

    # Direct fit baseline: piecewise-21 V0_4-distance → SSIM2 score.
    # Reuse the SSIM2-target anchors from v04_to_ssim2_anchors_2026-05-01.md.
    SSIM2_ANCHORS = [
        (-59.8119, 100.0000), (-46.3979, 84.7138), (-39.8782, 79.7890),
        (-35.2807, 76.1965),  (-30.7110, 72.5210), (-25.9624, 68.9566),
        (-20.0988, 64.4709),  (-12.7256, 59.2763), (-3.0700, 53.2772),
        (4.6772, 45.9756),    (10.5595, 38.4177),  (17.3930, 29.4720),
        (24.9022, 18.5703),   (33.9656, 6.8987),   (45.3017, -5.1928),
        (57.9030, -16.6924),  (72.9362, -28.5106), (92.7248, -41.7359),
        (124.7492, -54.1474), (176.7873, -62.9324),(623.7492, -367.3303),
    ]
    SSIM2_X = [a[0] for a in SSIM2_ANCHORS]

    def ssim2_score_via_anchors(d):
        if d <= SSIM2_X[0]:
            return SSIM2_ANCHORS[0][1]
        if d >= SSIM2_X[-1]:
            return SSIM2_ANCHORS[-1][1]
        i = bisect.bisect_right(SSIM2_X, d)
        x0, y0 = SSIM2_ANCHORS[i - 1]
        x1, y1 = SSIM2_ANCHORS[i]
        t = (d - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    df_human["ssim2_direct"] = df_human["v04_distance"].apply(ssim2_score_via_anchors)
    direct_rmse = np.sqrt(((df_human["ssim2_direct"] - y) ** 2).mean())

    # Polynomial shift: fit p(z) → y for degrees 1..7, no monotonicity constraint.
    print(f"\n{'shape':<25} {'RMSE':>8} {'monotonic over [-100,100]':>30}")
    print("-" * 70)
    print(f"{'direct piecewise-21 → SSIM2':<25} {direct_rmse:>8.3f} {'(by construction)':>30}")

    # Polynomial fit baseline.
    best_rmse = float("inf")
    best_deg = None
    best_coeffs = None
    for deg in range(1, 8):
        coeffs = np.polyfit(z, y, deg=deg)
        y_hat = np.polyval(coeffs, z)
        rmse = np.sqrt(((y_hat - y) ** 2).mean())
        # Check monotonicity over [-100, 100].
        zs = np.linspace(-100, 100, 2001)
        ys_eval = np.polyval(coeffs, zs)
        diffs = np.diff(ys_eval)
        monotonic = (diffs >= -1e-9).all() or (diffs <= 1e-9).all()
        print(f"{'poly deg-' + str(deg) + ' p(z) → SSIM2':<25} {rmse:>8.3f} {str(monotonic):>30}")
        if rmse < best_rmse and monotonic:
            best_rmse = rmse
            best_deg = deg
            best_coeffs = coeffs

    if best_coeffs is None:
        # Fall back to lowest-RMSE non-monotonic
        for deg in range(1, 8):
            coeffs = np.polyfit(z, y, deg=deg)
            y_hat = np.polyval(coeffs, z)
            rmse = np.sqrt(((y_hat - y) ** 2).mean())
            if rmse < best_rmse:
                best_rmse = rmse
                best_deg = deg
                best_coeffs = coeffs

    print(f"\nBest polynomial shift: degree {best_deg}, RMSE = {best_rmse:.3f}")
    print("Coefficients (high to low):", best_coeffs.tolist())

    # Plot scatter + best fit overlay.
    df_human["best_poly"] = np.polyval(best_coeffs, z)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sample = df_human.sample(min(3000, len(df_human)), random_state=42)
    axes[0].scatter(sample["v04_via_v02"], sample["fast_ssim2_score"],
                    alpha=0.3, s=8, color="#3a86ff")
    zs = np.linspace(-100, 100, 401)
    axes[0].plot(zs, np.polyval(best_coeffs, zs), color="#e63946", linewidth=2,
                 label=f"poly deg-{best_deg} (RMSE {best_rmse:.2f})")
    axes[0].plot([-100, 100], [-100, 100], color="#888", linestyle="--",
                 linewidth=1.0, label="identity")
    axes[0].set_xlabel("V0_4 score via V0_2 anchors (z)")
    axes[0].set_ylabel("SSIMULACRA 2 score")
    axes[0].set_title(f"Polynomial shift fit\n(direct piecewise → SSIM2 RMSE = {direct_rmse:.2f})")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(-100, 105)
    axes[0].set_ylim(-100, 105)

    # Residual histogram.
    direct_resid = df_human["ssim2_direct"] - y
    poly_resid = df_human["best_poly"] - y
    bins = np.linspace(-50, 50, 81)
    axes[1].hist(direct_resid, bins=bins, alpha=0.5, color="#ff8c42",
                 label=f"direct piecewise (RMSE {direct_rmse:.2f})")
    axes[1].hist(poly_resid, bins=bins, alpha=0.5, color="#3a86ff",
                 label=f"poly shift (RMSE {best_rmse:.2f})")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("residual: estimate − fast-ssim2")
    axes[1].set_ylabel("count")
    axes[1].set_title("Per-pair residuals")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/per_pair_v04_ssim2_holdout.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "/mnt/v/output/zensim/poly_shift_test.png"
    main(csv, out)
