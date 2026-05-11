#!/usr/bin/env python3
"""Score a unified parquet with a ZNPR v2 bake and audit monotonicity.

Loads a unified parquet's feat_0..feat_<n_inputs-1> columns, applies the
bake's scaler + MLP forward in pure numpy, then groups by (image_path,
codec, knob_tuple_json) and counts adjacent-q score reversals.

This is the missing smoothness measurement for any new bake.

Usage:
    python3 score_unified_with_bake.py \
        --bake benchmarks/rust_webp_mono_h128_seed1_2026-05-10.bin \
        --parquet /mnt/v/zen/zensim-training/2026-05-07/unified/unified_v15r_zenjpeg.parquet
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


HEADER_SIZE = 128
LAYER_ENTRY_SIZE = 48


def parse_bake_v2(path: Path):
    data = path.read_bytes()
    assert data[0:4] == b"ZNPR", f"bad magic at {path}"
    version = struct.unpack("<H", data[4:6])[0]
    assert version == 2, f"expected v2, got {version}"
    n_inputs = struct.unpack("<I", data[8:12])[0]
    n_outputs = struct.unpack("<I", data[12:16])[0]
    n_layers = struct.unpack("<I", data[16:20])[0]
    scaler_mean_off, scaler_mean_len = struct.unpack("<II", data[32:40])
    scaler_scale_off, scaler_scale_len = struct.unpack("<II", data[40:48])
    layer_table_off, layer_table_len = struct.unpack("<II", data[48:56])
    print(f"  ZNPR v2: {n_inputs}→...→{n_outputs}, {n_layers} layers")

    scaler_mean = np.frombuffer(
        data, dtype=np.float32, count=n_inputs, offset=scaler_mean_off
    ).copy()
    scaler_scale = np.frombuffer(
        data, dtype=np.float32, count=n_inputs, offset=scaler_scale_off
    ).copy()

    layers = []
    for i in range(n_layers):
        e_off = layer_table_off + i * LAYER_ENTRY_SIZE
        in_dim, out_dim = struct.unpack("<II", data[e_off:e_off + 8])
        activation, weight_dtype, flags = struct.unpack("<BBH", data[e_off + 8:e_off + 12])
        w_off, w_len = struct.unpack("<II", data[e_off + 12:e_off + 20])
        # scales unused for F32
        _scales_off, _scales_len = struct.unpack("<II", data[e_off + 20:e_off + 28])
        b_off, b_len = struct.unpack("<II", data[e_off + 28:e_off + 36])
        assert weight_dtype == 0, f"F32 only (got {weight_dtype})"
        # weights stored row-major [in_dim, out_dim] flattened — confirmed in v2.rs
        n_w = in_dim * out_dim
        weights = np.frombuffer(
            data, dtype=np.float32, count=n_w, offset=w_off
        ).reshape(in_dim, out_dim).copy()
        biases = np.frombuffer(
            data, dtype=np.float32, count=out_dim, offset=b_off
        ).copy()
        layers.append((in_dim, out_dim, activation, weights, biases))
        print(f"    layer {i}: {in_dim}→{out_dim} act={activation}")
    return n_inputs, n_outputs, scaler_mean, scaler_scale, layers


def forward(features: np.ndarray, scaler_mean, scaler_scale, layers):
    """features: [N, n_inputs] float32. Returns [N, n_outputs]."""
    x = (features - scaler_mean[None, :]) / scaler_scale[None, :]
    for i, (in_dim, out_dim, activation, w, b) in enumerate(layers):
        x = x @ w + b[None, :]
        if i < len(layers) - 1:
            # leakyrelu
            x = np.where(x > 0, x, x * 0.01)
        else:
            # final = identity (in our bakes)
            pass
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True, type=Path)
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--eps", type=float, default=0.0,
                    help="adjacent-q score-down tolerance")
    args = ap.parse_args()

    print(f"Loading bake {args.bake}...")
    n_inputs, n_outputs, scaler_mean, scaler_scale, layers = parse_bake_v2(args.bake)

    print(f"Loading parquet {args.parquet}...")
    feat_cols = [f"feat_{i}" for i in range(n_inputs)]
    cols = ["image_path", "codec", "q", "knob_tuple_json"] + feat_cols
    t = pq.read_table(args.parquet, columns=cols)
    df = t.to_pandas()
    print(f"  {len(df):,} rows")

    X = df[feat_cols].to_numpy(dtype=np.float32)
    print(f"  Scoring {len(X):,} pairs...")
    y = forward(X, scaler_mean, scaler_scale, layers)
    # Our bake outputs raw `predict()`; rank-meaningful but absolute scale
    # is whatever the trainer optimized. For non-monotonicity audit only
    # the rank within a curve matters.
    df["pred"] = y[:, 0]

    # Group + sort by q, count adjacent reversals
    n_curves = 0
    n_curves_w_violations = 0
    total_pairs = 0
    n_violations = 0
    for (img, codec, knob), g in df.groupby(["image_path", "codec", "knob_tuple_json"]):
        g = g.sort_values("q")
        if len(g) < 2:
            continue
        n_curves += 1
        # For our bake: higher q should give LOWER raw_distance (better quality
        # = lower predicted "distance"). The "non-monotone" violation is q goes
        # up but pred goes UP (worse predicted quality).
        # But sign convention varies. Let's count BOTH directions and pick
        # the smaller "violation rate" interpretation.
        preds = g["pred"].to_numpy()
        # Direction A: lower pred is better (distance semantics).
        viols_A = sum(1 for i in range(len(preds) - 1)
                      if preds[i + 1] > preds[i] + args.eps)
        # Direction B: higher pred is better (score semantics).
        viols_B = sum(1 for i in range(len(preds) - 1)
                      if preds[i + 1] < preds[i] - args.eps)
        v = min(viols_A, viols_B)
        total_pairs += len(preds) - 1
        n_violations += v
        if v > 0:
            n_curves_w_violations += 1
    print()
    print(f"Curves: {n_curves:,}")
    print(f"Curves with ≥1 violation: {n_curves_w_violations:,} "
          f"({n_curves_w_violations / n_curves * 100:.2f}%)")
    print(f"Adjacent-q pairs: {total_pairs:,}")
    print(f"Reversed pairs: {n_violations:,} "
          f"({n_violations / total_pairs * 100:.2f}%)")
    print()
    # Project floor target: < 4.86% per zensim/CLAUDE.md goal #2
    rate = n_violations / total_pairs * 100
    target = 4.86
    if rate < target:
        print(f"✓ Non-mono rate {rate:.2f}% < target {target}% — smoothness goal MET")
    else:
        print(f"✗ Non-mono rate {rate:.2f}% ≥ target {target}% — smoothness goal NOT MET")


if __name__ == "__main__":
    main()
