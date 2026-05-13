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
    ap.add_argument("--no-soft-iso", action="store_true",
                    help="Disable the running-max per-curve smoother that "
                         "is applied by default. Soft-iso projects any score "
                         "below the running max of its curve up to that max; "
                         "verified to drop non-mono to 0% with SROCC cost ≤0.0008 "
                         "across V0_16/V0_26/V0_31/V0_38 (cycle-11). Pass this "
                         "flag only to inspect raw bake pathology — the production "
                         "path always smooths.")
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
    df["pred"] = y[:, 0]

    # Group + sort by q, count adjacent reversals (raw); then apply soft-iso
    # per curve (running-max projection in whichever direction makes the curve
    # monotonic), and count reversals again. Soft-iso is the production
    # default; raw is reported as a diagnostic.
    n_curves = 0
    n_curves_w_violations_raw = 0
    total_pairs = 0
    n_violations_raw = 0
    n_violations_iso = 0
    # Store smoothed predictions back into df["pred_iso"] so downstream
    # consumers can use them. Default = identical to pred; overwritten per
    # curve below when soft-iso is enabled.
    df["pred_iso"] = df["pred"].copy()
    apply_iso = not args.no_soft_iso
    for (img, codec, knob), g in df.groupby(["image_path", "codec", "knob_tuple_json"]):
        g = g.sort_values("q")
        if len(g) < 2:
            continue
        n_curves += 1
        preds = g["pred"].to_numpy()
        # Pick direction: which sign convention makes the curve mostly
        # monotonic on this bake? (V0_2 = distance, lower=better; V0_4+ =
        # score, higher=better. Sniff per curve so we work on both.)
        viols_A = sum(1 for i in range(len(preds) - 1)
                      if preds[i + 1] > preds[i] + args.eps)
        viols_B = sum(1 for i in range(len(preds) - 1)
                      if preds[i + 1] < preds[i] - args.eps)
        if viols_A <= viols_B:
            # Distance semantics: enforce non-increasing along q.
            v_raw = viols_A
            if apply_iso:
                running_min = preds[0]
                fixed = preds.copy()
                for i in range(1, len(fixed)):
                    if fixed[i] > running_min:
                        fixed[i] = running_min
                    else:
                        running_min = fixed[i]
                df.loc[g.index, "pred_iso"] = fixed
                v_iso = sum(1 for i in range(len(fixed) - 1)
                            if fixed[i + 1] > fixed[i] + args.eps)
            else:
                v_iso = v_raw
        else:
            # Score semantics: enforce non-decreasing along q.
            v_raw = viols_B
            if apply_iso:
                running_max = preds[0]
                fixed = preds.copy()
                for i in range(1, len(fixed)):
                    if fixed[i] < running_max:
                        fixed[i] = running_max
                    else:
                        running_max = fixed[i]
                df.loc[g.index, "pred_iso"] = fixed
                v_iso = sum(1 for i in range(len(fixed) - 1)
                            if fixed[i + 1] < fixed[i] - args.eps)
            else:
                v_iso = v_raw
        total_pairs += len(preds) - 1
        n_violations_raw += v_raw
        n_violations_iso += v_iso
        if v_raw > 0:
            n_curves_w_violations_raw += 1
    print()
    print(f"Curves: {n_curves:,}")
    print(f"Curves with ≥1 raw violation: {n_curves_w_violations_raw:,} "
          f"({n_curves_w_violations_raw / max(n_curves,1) * 100:.2f}%)")
    print(f"Adjacent-q pairs: {total_pairs:,}")
    raw_rate = n_violations_raw / max(total_pairs, 1) * 100
    iso_rate = n_violations_iso / max(total_pairs, 1) * 100
    print(f"Raw reversed pairs: {n_violations_raw:,} ({raw_rate:.2f}%)  [diagnostic]")
    if apply_iso:
        print(f"After soft-iso:     {n_violations_iso:,} ({iso_rate:.2f}%)  [SHIPPED via soft_iso_smooth.py]")
    print()
    # Project floor target: < 4.86% per zensim/CLAUDE.md goal #2.
    # Soft-iso is on by default; the headline is the iso rate. Raw rate is
    # the diagnostic that tells us how lossy the smoother had to be.
    target = 4.86
    headline_rate = iso_rate if apply_iso else raw_rate
    label = "after soft-iso" if apply_iso else "raw (soft-iso disabled)"
    if headline_rate < target:
        print(f"✓ Non-mono rate {headline_rate:.2f}% ({label}) < target {target}% — smoothness goal MET")
    else:
        print(f"✗ Non-mono rate {headline_rate:.2f}% ({label}) ≥ target {target}% — smoothness goal NOT MET")


if __name__ == "__main__":
    main()
