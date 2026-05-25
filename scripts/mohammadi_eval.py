#!/usr/bin/env python3
"""Mohammadi 2025 exact-methodology evaluation on AIC-3.

Reproduces Tables 2-3 from "Evaluation of Objective Image Quality
Metrics for High-Fidelity Image Compression" (IEEE Access 2026).

Usage:
  python3 scripts/mohammadi_eval.py <bake.bin>

Requires: the enriched AIC-3 parquet at
  canonical-2026-05-21/val/aic3_with_sigma.parquet
which carries distortion_jnd (μ) and sigma_bootstrap (σ) from the
Mohammadi paper's 1000× bootstrap on 300 PTC full-resolution stimuli.
"""
import sys, subprocess, json, math
import pyarrow.parquet as pq
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr, kendalltau, pearsonr

PARQUET = "/mnt/v/zen/zensim-training/canonical-2026-05-21/val/aic3_with_sigma.parquet"
FEATURES_PARQUET = "/mnt/v/zen/zensim-training/2026-05-15-full-features/aic3_features_372col_2026-05-15.parquet"

def logistic_4param(b, x):
    """4-parameter logistic (Mohammadi Eq. 1)."""
    b4 = max(abs(b[3]), 1e-8) * (1 if b[3] >= 0 else -1)
    arg = -(x - b[2]) / b4
    arg = np.clip(arg, -500, 500)
    return b[1] + (b[0] - b[1]) / (1 + np.exp(arg))

def fit_logistic(pred, target):
    """Fit 4-param logistic via nonlinear least-squares (Mohammadi §VI)."""
    def residuals(b):
        return logistic_4param(b, pred) - target
    b0 = [max(target), min(target), np.median(pred), np.std(pred)]
    result = least_squares(residuals, b0, method='lm', max_nfev=5000)
    return logistic_4param(result.x, pred)

def srocc(a, b):
    r, _ = spearmanr(a, b)
    return abs(r)

def plcc(pred, target):
    rescaled = fit_logistic(pred, target)
    r, _ = pearsonr(rescaled, target)
    return abs(r)

def krocc(a, b):
    r, _ = kendalltau(a, b)
    return abs(r)

def rmse(pred, target):
    rescaled = fit_logistic(pred, target)
    return np.sqrt(np.mean((rescaled - target) ** 2))

def outlier_ratio(pred, target):
    rescaled = fit_logistic(pred, target)
    residuals = np.abs(rescaled - target)
    sigma = max(np.std(residuals), 1e-9)
    return np.mean(residuals > 2 * sigma)

def pwrc(a, b):
    n = len(a)
    if n < 4:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    mid = (n - 1) / 2.0
    max_dev = max(mid, 1e-12)
    w = np.abs(ra - mid) / max_dev
    wsum = w.sum()
    if wsum < 1e-12:
        return 0.0
    mean_a = np.average(ra, weights=w)
    mean_b = np.average(rb, weights=w)
    num = np.sum(w * (ra - mean_a) * (rb - mean_b))
    da = np.sum(w * (ra - mean_a) ** 2)
    db = np.sum(w * (rb - mean_b) ** 2)
    den = np.sqrt(da * db)
    return abs(num / den) if den > 1e-12 else 0.0

def z_rmse_per_sample(pred, target, sigma):
    """Mohammadi Eq. 6: per-stimulus σ-normalized RMSE."""
    rescaled = fit_logistic(pred, target)
    valid = sigma > 0
    z = (rescaled[valid] - target[valid]) / sigma[valid]
    return np.sqrt(np.mean(z ** 2))

def eval_subset(pred, target, sigma, label):
    """Compute full Mohammadi panel on a subset."""
    n = len(pred)
    if n < 4:
        print(f"  {label}: n={n} — too few samples")
        return
    s = srocc(pred, target)
    p = plcc(pred, target)
    k = krocc(pred, target)
    r = rmse(pred, target)
    o = outlier_ratio(pred, target)
    pw = pwrc(pred, target)
    zr = z_rmse_per_sample(pred, target, sigma) if sigma is not None and len(sigma) == n else float('nan')
    print(f"  {label:6s} (n={n:3d}): SROCC={s:.4f}  PLCC={p:.4f}  KT={k:.4f}  RMSE={r:.4f}  OR={o:.4f}  PWRC={pw:.4f}  Z-RMSE={zr:.4f}")
    return {'label': label, 'n': n, 'srocc': s, 'plcc': p, 'krocc': k, 'rmse': r, 'or': o, 'pwrc': pw, 'z_rmse': zr}

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <bake.bin>")
        sys.exit(1)
    bake_path = sys.argv[1]

    # Load enriched AIC-3 parquet
    t = pq.read_table(PARQUET)
    sigmas = np.array([s if s is not None else np.nan for s in t.column('sigma_bootstrap').to_pylist()])
    dists = np.array([d if d is not None else np.nan for d in t.column('distortion_jnd').to_pylist()])

    # Only use the 300 rows with Mohammadi data
    valid = ~np.isnan(sigmas)
    print(f"AIC-3 PTC full-resolution: {valid.sum()} stimuli with bootstrap σ")

    # Load features from the features parquet
    ft = pq.read_table(FEATURES_PARQUET)
    print(f"Features parquet: {ft.num_rows} rows × {ft.num_columns} cols")

    # Extract features for the valid rows
    feature_cols = [c for c in ft.column_names if c.startswith('f')]
    n_features = len(feature_cols)
    print(f"Feature columns: {n_features}")

    # We need to match the 600 rows in the enriched parquet to the 600 rows
    # in the features parquet (same ordering assumed — both from canonical build)
    features = np.zeros((ft.num_rows, n_features))
    for i, col in enumerate(feature_cols):
        features[:, i] = ft.column(col).to_numpy()

    # Run bake predictor on all 600 rows, filter to valid 300
    # Use bake_verdict's predict_features_with_bake binary if available
    predict_bin = "./target/release/predict_features_with_bake"
    import os
    # Check multiple locations
    for path in ["./target/release/predict_features_with_bake",
                  "/home/lilith/work/zen/zensim/target/release/predict_features_with_bake"]:
        if os.path.exists(path):
            predict_bin = path
            break

    # Write features to a temp CSV for the predictor
    # Actually, let's use the Python zenpredict binding if available,
    # or just call bake_verdict and parse the per-row output.
    # Simplest: use bake_verdict's per-pair output mode.

    # For now, extract predictions from bake_verdict's summary
    # (it already computed SROCC 0.8082 on AIC-3 for the v2 bake)
    # But we need per-row predictions for proper eval.

    # Let's check if predict_features_with_bake exists
    if not os.path.exists(predict_bin):
        print(f"predict_features_with_bake not found at {predict_bin}")
        print("Building...")
        os.system("cd /home/lilith/work/zen/zensim--goal-push && cargo build --release -p zensim-validate --bin predict_features_with_bake 2>&1 | tail -3")
        predict_bin = "/home/lilith/work/zen/zensim--goal-push/target/release/predict_features_with_bake"

    if os.path.exists(predict_bin):
        import tempfile, struct
        # Write binary features file: u32 n_features, u32 n_rows, then f32[] row-major
        features_f32 = features.astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            tmppath = f.name
            f.write(struct.pack('<II', n_features, ft.num_rows))
            f.write(features_f32.tobytes())

        result = subprocess.run(
            [predict_bin, '--bake', bake_path, '--features-file', tmppath],
            capture_output=True, text=True, timeout=120
        )
        os.unlink(tmppath)

        if result.returncode != 0:
            print(f"Predictor failed: {result.stderr[:500]}")
            # Fall back to bake_verdict aggregate numbers
            print("\nFalling back to bake_verdict aggregate (no per-row predictions)")
            return

        # Parse predictions (one float per line)
        preds_all = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                preds_all.append(float(line))
            except ValueError:
                continue

        preds_all = np.array(preds_all)
        print(f"Predictions: {len(preds_all)} rows")
    else:
        print("No predictor available — cannot run per-row eval")
        return

    if len(preds_all) != ft.num_rows:
        print(f"Prediction count mismatch: {len(preds_all)} vs {ft.num_rows}")
        return

    # Filter to valid 300 rows
    preds = preds_all[valid]
    targets = dists[valid]
    sigs = sigmas[valid]

    # HF/MF split per Mohammadi paper
    hf_mask = targets <= 1.0
    mf_mask = targets > 1.0

    print(f"\n{'='*70}")
    print(f"Mohammadi 2025 evaluation — {bake_path}")
    print(f"{'='*70}")
    print(f"AIC-3 PTC full-resolution, {valid.sum()} stimuli")
    print(f"  HF (≤1 JND): {hf_mask.sum()}")
    print(f"  MF (>1 JND): {mf_mask.sum()}")
    print()

    eval_subset(preds, targets, sigs, "All")
    eval_subset(preds[hf_mask], targets[hf_mask], sigs[hf_mask], "HF")
    eval_subset(preds[mf_mask], targets[mf_mask], sigs[mf_mask], "MF")

    print(f"\nMohammadi Table 2 reference (SOTA):")
    print(f"  CVVDP:        SROCC=0.960  PLCC=0.958  KT=0.838  Z-RMSE=9.45")
    print(f"  IW-SSIM:      SROCC=0.944  PLCC=0.940  KT=0.802  Z-RMSE=10.48")
    print(f"  SSIMULACRA2:  SROCC=0.905  PLCC=0.906  KT=0.715  Z-RMSE=10.06")
    print(f"  BUTTERAUGLI:  SROCC=0.893  PLCC=0.881  KT=0.717  Z-RMSE=9.92")

if __name__ == '__main__':
    main()
