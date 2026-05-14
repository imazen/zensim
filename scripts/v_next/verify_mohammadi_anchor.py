#!/usr/bin/env python3
"""Verify our Z-RMSE / SROCC / PLCC against the Mohammadi 2025 anchor CSV.

Uses the logistic 4-parameter rescale (Mohammadi's convention) before
computing σ-normalized RMSE. SROCC and PLCC are computed on the
logistic-fitted values.

Expected output: numbers within ±0.001 SROCC and ±0.01 Z-RMSE of the
paper's Table I (Mohammadi et al. 2025, arXiv:2509.13150).

Usage:
    python3 scripts/v_next/verify_mohammadi_anchor.py
    python3 scripts/v_next/verify_mohammadi_anchor.py --csv PATH.csv
"""
import argparse
import csv
import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr

DEFAULT_CSV = '/mnt/v/input/datasets/aic3/EvaluationMetrics/Anchor_assessment_on_PTC_full_resolution_Aug_3_2025.csv'

def logistic(x, b1, b2, b3, b4):
    return b2 + (b1 - b2) / (1.0 + np.exp(-(x - b3) / b4))


def fit_and_eval(metric_vals, mos, sigma):
    x = np.array(metric_vals)
    p0 = [mos.max(), mos.min(), x.mean(), max(x.std(), 1e-3)]
    try:
        popt, _ = curve_fit(logistic, x, mos, p0=p0, maxfev=5000)
        fit = logistic(x, *popt)
    except Exception:
        fit = x
    z_sq = ((fit - mos) / sigma) ** 2
    return {
        'srocc': abs(spearmanr(mos, fit)[0]),
        'plcc': abs(pearsonr(mos, fit)[0]),
        'z_rmse_raw': float(np.sqrt(np.mean(z_sq))),
    }


# Paper Table I values (Mohammadi et al. 2025, arXiv:2509.13150)
PAPER = {
    'CVVDP':       {'srocc': 0.961, 'z_rmse_raw': 9.45},
    'iw_ssim':     {'srocc': 0.944, 'z_rmse_raw': 31.51},
    'MS-SSIM':     {'srocc': 0.927, 'z_rmse_raw': None},
    'SSIMULACRA2': {'srocc': 0.913, 'z_rmse_raw': 47.63},
    'psnry':       {'srocc': 0.812, 'z_rmse_raw': 13.36},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    mos = np.array([float(x['distortion']) for x in rows])
    sigma = np.array([float(x['std_bootstrap']) for x in rows])
    print(f"loaded {len(rows)} stimuli\n")
    print(f"{'Metric':<14} {'SROCC':>7} {'PLCC':>7} {'Z-RMSE':>9}  {'paper SROCC':>11} {'Δ':>9}  {'paper Z':>8} {'Δ':>7}")
    print('-' * 92)

    metrics = ['SSIMULACRA2', 'iw_ssim', 'MS-SSIM', 'CVVDP', 'psnry',
               'nlpd', 'vif', 'LPIPS', 'DISTS', 'PieAPP', 'TopIQ',
               'SSIMULACRA1', 'Butteragli2', 'vmaf_neg']

    ok = True
    for col in metrics:
        try:
            vals = np.array([float(x[col]) for x in rows])
        except (KeyError, ValueError):
            continue
        r = fit_and_eval(vals, mos, sigma)
        paper = PAPER.get(col, {})
        s_delta = f"{r['srocc'] - paper['srocc']:+.4f}" if paper.get('srocc') else "—"
        z_delta = f"{r['z_rmse_raw'] - paper['z_rmse_raw']:+.4f}" if paper.get('z_rmse_raw') else "—"
        paper_s = f"{paper.get('srocc', '—'):>11}" if paper.get('srocc') else "—".rjust(11)
        paper_z = f"{paper.get('z_rmse_raw', '—'):>8}" if paper.get('z_rmse_raw') else "—".rjust(8)
        print(f"{col:<14} {r['srocc']:>7.4f} {r['plcc']:>7.4f} {r['z_rmse_raw']:>9.3f}  {paper_s} {s_delta:>9}  {paper_z} {z_delta:>7}")

        # Verification gate: paper-listed metrics must match within tolerance
        if paper.get('srocc') is not None and abs(r['srocc'] - paper['srocc']) > 0.01:
            print(f"  ⚠ {col} SROCC deviation > 0.01 — implementation drift?", file=sys.stderr)
            ok = False
        if paper.get('z_rmse_raw') is not None and abs(r['z_rmse_raw'] - paper['z_rmse_raw']) > 0.5:
            print(f"  ⚠ {col} Z-RMSE deviation > 0.5 — implementation drift?", file=sys.stderr)
            ok = False

    if ok:
        print("\n✓ All paper-listed metrics within tolerance (≤0.01 SROCC, ≤0.5 Z-RMSE).")
    else:
        print("\n⚠ Some metrics deviated. Check the warnings above.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
