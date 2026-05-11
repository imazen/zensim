#!/usr/bin/env python3
"""Apply affine calibration `y' = α + β · y` directly to a ZNPR v2 bake.

Mathematically, composing the final Linear layer (y = W·h + b) with an
affine output transform (y' = α + β·y) is equivalent to:
    W' = β · W
    b' = β · b + α
This is zero-runtime-cost and rank-preserving (SROCC unchanged).

The companion script `affine_calibrate_bake.py` does the same operation
on a PyTorch model.pt run-dir. THIS script operates directly on a baked
ZNPR v2 binary, which is the format produced by the Rust trainer
`zensim_mlp_train`.

Usage:
    python3 affine_calibrate_znpr_v2.py \\
        --in-bake  path/to/bake.bin \\
        --out-bake path/to/calibrated.bin \\
        --alpha A --beta B

Self-verification: if --verify-features and --verify-truth are provided
(JPEG unified parquet path), the script will score N=1000 random rows
with both the original and calibrated bake, verify SROCC is identical
(rank-invariant), and report the calibrated output range.
"""
import argparse
import struct
from pathlib import Path

import numpy as np

LAYER_ENTRY_SIZE = 48
HEADER_SIZE = 128


def parse_bake_v2_offsets(data: bytes):
    """Return parsed header info + per-layer offsets so we can mutate in place."""
    assert data[0:4] == b"ZNPR"
    version = struct.unpack("<H", data[4:6])[0]
    assert version == 2, f"expected v2, got {version}"
    n_inputs = struct.unpack("<I", data[8:12])[0]
    n_outputs = struct.unpack("<I", data[12:16])[0]
    n_layers = struct.unpack("<I", data[16:20])[0]
    layer_table_off = struct.unpack("<I", data[48:52])[0]
    assert n_outputs == 1, f"calibration assumes scalar output, got {n_outputs}"

    layers = []
    for i in range(n_layers):
        e = layer_table_off + i * LAYER_ENTRY_SIZE
        in_dim, out_dim = struct.unpack("<II", data[e:e+8])
        activation = data[e+8]
        w_off, w_len = struct.unpack("<II", data[e+12:e+20])
        b_off, b_len = struct.unpack("<II", data[e+28:e+36])
        layers.append({
            "in_dim": in_dim, "out_dim": out_dim,
            "activation": activation,
            "w_off": w_off, "w_len": w_len,
            "b_off": b_off, "b_len": b_len,
        })
    return n_inputs, n_layers, layers


def calibrate_in_place(data: bytearray, layers: list, alpha: float, beta: float) -> None:
    """Modify final-layer weights and bias in-place.

    For final Linear: y = W·h + b. Composed with y' = α + β·y:
        W' = β · W
        b' = β · b + α
    Both are in-place float32 mutations.
    """
    final = layers[-1]
    assert final["out_dim"] == 1, "final layer must be scalar output"

    # Weights: in_dim × 1 float32. Multiply by β.
    n_w = final["in_dim"] * final["out_dim"]
    w_view = np.frombuffer(data, dtype=np.float32, count=n_w, offset=final["w_off"])
    w_view *= beta  # in-place mutation of the bytearray

    # Bias: 1 float32. b' = β·b + α.
    b_view = np.frombuffer(data, dtype=np.float32, count=final["out_dim"], offset=final["b_off"])
    b_view *= beta
    b_view += alpha


def predict(features: np.ndarray, data: bytes) -> np.ndarray:
    """Forward pass through a ZNPR v2 bake."""
    n_inputs = struct.unpack("<I", data[8:12])[0]
    scaler_mean_off = struct.unpack("<I", data[32:36])[0]
    scaler_scale_off = struct.unpack("<I", data[40:44])[0]
    sm = np.frombuffer(data, dtype=np.float32, count=n_inputs, offset=scaler_mean_off)
    ss = np.frombuffer(data, dtype=np.float32, count=n_inputs, offset=scaler_scale_off)
    _, n_layers, layers = parse_bake_v2_offsets(data)
    x = (features - sm[None, :]) / ss[None, :]
    for i, L in enumerate(layers):
        n_w = L["in_dim"] * L["out_dim"]
        w = np.frombuffer(data, dtype=np.float32, count=n_w, offset=L["w_off"]).reshape(L["in_dim"], L["out_dim"])
        b = np.frombuffer(data, dtype=np.float32, count=L["out_dim"], offset=L["b_off"])
        x = x @ w + b[None, :]
        if i < n_layers - 1:
            x = np.where(x > 0, x, 0.01 * x)
    return x.squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-bake", required=True, type=Path)
    ap.add_argument("--out-bake", required=True, type=Path)
    ap.add_argument("--alpha", type=float, required=True,
                    help="affine offset: y' = alpha + beta * y")
    ap.add_argument("--beta", type=float, required=True,
                    help="affine scale: y' = alpha + beta * y")
    ap.add_argument("--verify-parquet", type=Path,
                    help="optional unified parquet for SROCC self-check")
    args = ap.parse_args()

    print(f"Reading {args.in_bake} ({args.in_bake.stat().st_size} bytes)")
    orig_bytes = args.in_bake.read_bytes()
    n_inputs, n_layers, layers = parse_bake_v2_offsets(orig_bytes)
    print(f"  ZNPR v2: {n_inputs} inputs → {n_layers} layers")
    for i, L in enumerate(layers):
        act = {0: "Identity", 2: "LeakyReLU"}.get(L["activation"], f"?{L['activation']}")
        print(f"    layer {i}: {L['in_dim']} → {L['out_dim']}  act={act}")
    print(f"\nApplying calibration: y' = {args.alpha:.4f} + {args.beta:.4f} * y")
    print(f"  Modifying final layer: W' = β·W, b' = β·b + α")

    # Copy and mutate
    new_bytes = bytearray(orig_bytes)
    calibrate_in_place(new_bytes, layers, args.alpha, args.beta)

    args.out_bake.write_bytes(new_bytes)
    print(f"Wrote {args.out_bake} ({len(new_bytes)} bytes)")

    # Self-verify
    if args.verify_parquet:
        from scipy.stats import spearmanr
        import pyarrow.parquet as pq
        print(f"\nVerifying on {args.verify_parquet}...")
        cols = ["score_ssim2"] + [f"feat_{i}" for i in range(n_inputs)]
        tbl = pq.read_table(args.verify_parquet, columns=cols)
        truth = tbl.column("score_ssim2").to_numpy()
        feats = np.stack([tbl.column(f"feat_{i}").to_numpy() for i in range(n_inputs)], axis=1).astype(np.float32)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(truth), size=min(1000, len(truth)), replace=False)
        orig_pred = predict(feats[idx], orig_bytes)
        new_pred = predict(feats[idx], bytes(new_bytes))
        orig_srocc, _ = spearmanr(orig_pred, truth[idx])
        new_srocc, _ = spearmanr(new_pred, truth[idx])
        print(f"  Original SROCC : {orig_srocc:.6f}   range=[{orig_pred.min():.2f}, {orig_pred.max():.2f}]")
        print(f"  Calibrated SROCC: {new_srocc:.6f}   range=[{new_pred.min():.2f}, {new_pred.max():.2f}]")
        # |SROCC| should be identical (β > 0) or negated (β < 0); abs should match exactly
        assert abs(abs(orig_srocc) - abs(new_srocc)) < 1e-6, "rank invariance violated!"
        print("  ✓ |SROCC| identical (rank-invariant calibration verified)")
        # Predict on all and confirm calibrated range is reasonable
        new_pred_all = predict(feats, bytes(new_bytes))
        print(f"  Calibrated output on full parquet ({len(feats):,} rows):")
        print(f"    range=[{new_pred_all.min():.2f}, {new_pred_all.max():.2f}]")
        print(f"    mean={new_pred_all.mean():.2f}, std={new_pred_all.std():.2f}")
        print(f"    p5={np.percentile(new_pred_all, 5):.2f}, p50={np.median(new_pred_all):.2f}, p95={np.percentile(new_pred_all, 95):.2f}")


if __name__ == "__main__":
    main()
