#!/usr/bin/env python3
"""Fit a PCHIP output calibration spline mapping bake predictions to
AIC-3 JND scale. The spline knots are baked as
`zentrain.output_calibration_spline` metadata so the runtime applies
the mapping automatically.

Usage:
  python3 scripts/fit_output_spline.py <bake.bin> <output_spline_payload.bin>

The output payload is the raw bytes for the metadata entry (u32 n_knots
+ n_knots × (f32 x, f32 y) little-endian).
"""
import sys, subprocess, struct, os
import numpy as np
import pyarrow.parquet as pq

PARQUET = "/mnt/v/zen/zensim-training/canonical-2026-05-21/val/aic3_with_sigma.parquet"
FEATURES_PARQUET = "/mnt/v/zen/zensim-training/2026-05-15-full-features/aic3_features_372col_2026-05-15.parquet"

def get_predictions(bake_path, predict_bin):
    ft = pq.read_table(FEATURES_PARQUET)
    feature_cols = [c for c in ft.column_names if c.startswith('f')]
    n_features = len(feature_cols)
    features = np.zeros((ft.num_rows, n_features), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        features[:, i] = ft.column(col).to_numpy().astype(np.float32)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        tmppath = f.name
        f.write(struct.pack('<II', n_features, ft.num_rows))
        f.write(features.tobytes())

    result = subprocess.run(
        [predict_bin, '--bake', bake_path, '--features-file', tmppath],
        capture_output=True, text=True, timeout=120
    )
    os.unlink(tmppath)
    preds = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            preds.append(float(line))
        except ValueError:
            continue
    return np.array(preds)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <bake.bin> <output_spline_payload.bin>")
        sys.exit(1)

    bake_path = sys.argv[1]
    output_path = sys.argv[2]

    predict_bin = None
    for path in ["./target/release/predict_features_with_bake",
                  "/home/lilith/work/zen/zensim/target/release/predict_features_with_bake",
                  "/home/lilith/work/zen/zensim/target/release/predict_features_with_bake"]:
        if os.path.exists(path):
            predict_bin = path
            break
    if not predict_bin:
        print("predict_features_with_bake not found")
        sys.exit(1)

    # Get predictions on AIC-3
    preds = get_predictions(bake_path, predict_bin)
    print(f"Predictions: {len(preds)} rows, range [{preds.min():.2f}, {preds.max():.2f}]")

    # Load AIC-3 ground truth
    t = pq.read_table(PARQUET)
    dists = np.array([d if d is not None else np.nan for d in t.column('distortion_jnd').to_pylist()])
    valid = ~np.isnan(dists)

    preds_valid = preds[valid]
    dists_valid = dists[valid]
    print(f"Valid pairs: {valid.sum()}, pred range [{preds_valid.min():.2f}, {preds_valid.max():.2f}], "
          f"JND range [{dists_valid.min():.3f}, {dists_valid.max():.3f}]")

    # Fit knots: bin predictions into 8-12 quantile bins, compute
    # median prediction + median JND per bin. These become the spline
    # knot points (x=prediction, y=JND).
    n_knots = 10
    quantiles = np.linspace(0, 100, n_knots + 2)[1:-1]  # skip 0 and 100
    pred_thresholds = np.percentile(preds_valid, quantiles)

    # Ensure strictly increasing x by adding small epsilon
    for i in range(1, len(pred_thresholds)):
        if pred_thresholds[i] <= pred_thresholds[i-1]:
            pred_thresholds[i] = pred_thresholds[i-1] + 0.001

    knot_xs = []
    knot_ys = []

    # Add an extrapolation knot below the range
    low_mask = preds_valid < pred_thresholds[0]
    if low_mask.sum() > 0:
        knot_xs.append(float(np.median(preds_valid[low_mask])))
        knot_ys.append(float(np.median(dists_valid[low_mask])))

    # Interior knots from quantile bins
    for i in range(len(pred_thresholds) - 1):
        mask = (preds_valid >= pred_thresholds[i]) & (preds_valid < pred_thresholds[i+1])
        if mask.sum() >= 2:
            knot_xs.append(float(np.median(preds_valid[mask])))
            knot_ys.append(float(np.median(dists_valid[mask])))

    # Add an extrapolation knot above the range
    high_mask = preds_valid >= pred_thresholds[-1]
    if high_mask.sum() > 0:
        knot_xs.append(float(np.median(preds_valid[high_mask])))
        knot_ys.append(float(np.median(dists_valid[high_mask])))

    # Ensure strictly increasing x
    clean_xs = [knot_xs[0]]
    clean_ys = [knot_ys[0]]
    for i in range(1, len(knot_xs)):
        if knot_xs[i] > clean_xs[-1] + 0.001:
            clean_xs.append(knot_xs[i])
            clean_ys.append(knot_ys[i])

    knot_xs = clean_xs
    knot_ys = clean_ys
    n = len(knot_xs)

    print(f"\nSpline knots ({n}):")
    print(f"  {'pred':>8s}  {'JND':>8s}")
    for x, y in zip(knot_xs, knot_ys):
        print(f"  {x:8.3f}  {y:8.4f}")

    # Write payload: u32 n_knots + n_knots × (f32 x, f32 y)
    payload = struct.pack('<I', n)
    for x, y in zip(knot_xs, knot_ys):
        payload += struct.pack('<ff', float(x), float(y))

    with open(output_path, 'wb') as f:
        f.write(payload)
    print(f"\nWrote spline payload ({len(payload)} bytes) to {output_path}")

    # Verify by computing calibrated predictions and Z-RMSE
    from scipy.interpolate import PchipInterpolator
    spline = PchipInterpolator(knot_xs, knot_ys, extrapolate=True)
    calibrated = spline(preds_valid)

    sigmas = np.array([s if s is not None else np.nan for s in t.column('sigma_bootstrap').to_pylist()])
    sigs_valid = sigmas[valid]
    z = (calibrated - dists_valid) / sigs_valid
    z_finite = z[np.isfinite(z)]
    z_rmse = np.sqrt(np.mean(z_finite**2))
    print(f"\nCalibrated Z-RMSE (per-sample σ): {z_rmse:.4f}")
    print(f"  (vs uncalibrated: run mohammadi_eval.py for comparison)")

if __name__ == '__main__':
    main()
