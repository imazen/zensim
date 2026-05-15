#!/usr/bin/env python3
"""V_20b Phase 4 — bake the fine-tuned (encoder + head) into ZNPR v3.

Reads the npz from `finetune_head.py` and emits a ZNPR v3 bake
compatible with the existing zensim runtime (PreviewV0_3 slot).

## Architectural collapse

The fine-tuned model is:
  x → LeakyReLU(W0 x + b0)         (228 → hidden_dim)
    → W1 LeakyReLU(...) + b1        (hidden_dim → embedding_dim, IDENTITY)
    → W_head embedding + b_head     (embedding_dim → 1, IDENTITY)

Because the embedding output is Identity (no nonlinearity between
encoder fc1 and head), the W1·W_head + head_bias composition is a
single linear map. We mathematically collapse:
  W_final = W_head @ W1           (1 × hidden_dim)
  b_final = W_head @ b1 + b_head   (scalar)

producing an equivalent 2-layer (228 → hidden_dim → 1) bake with no
loss of accuracy. The wire format is identical to V_18's, so the
existing zensim PreviewV0_3 loader works unchanged.

## Usage

  python3 scripts/v_next/v0_20b/bake_v3.py \\
    --finetune /tmp/v0_20b_full.npz \\
    --out benchmarks/v0_20b_seed1_2026-05-15.bin
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    blob = np.load(args.finetune)
    n_inputs = int(blob["n_inputs"][0])
    hidden_dim = int(blob["hidden_dim"][0])
    embedding_dim = int(blob["embedding_dim"][0])
    scaler_mean = blob["scaler_mean"]
    scaler_scale = blob["scaler_scale"]
    w0 = blob["encoder_w0"]  # [hidden_dim, n_inputs]
    b0 = blob["encoder_b0"]  # [hidden_dim]
    w1 = blob["encoder_w1"]  # [embedding_dim, hidden_dim]
    b1 = blob["encoder_b1"]  # [embedding_dim]
    head_w = blob["head_w"]  # [1, embedding_dim]
    head_b = blob["head_b"]  # [1]

    print(
        f"loaded V_20b: {n_inputs} -> {hidden_dim} (LeakyReLU) -> "
        f"{embedding_dim} (Identity) -> 1 (Identity)"
    )

    # Collapse W_head @ W1, head_b + W_head @ b1
    # Shapes: head_w [1, embedding_dim], w1 [embedding_dim, hidden_dim]
    # → W_final [1, hidden_dim] = head_w @ w1
    w_final = head_w @ w1  # [1, hidden_dim]
    b_final = head_w @ b1 + head_b  # [1]
    print(
        f"collapsed to 2-layer: ({n_inputs} -> {hidden_dim} LeakyReLU) -> "
        f"({hidden_dim} -> 1 Identity)"
    )
    print(f"  w_final shape: {w_final.shape}, b_final: {b_final}")

    # Emit a TSV/json the Rust bake binary can consume, OR write the
    # ZNPR v3 wire format directly. Easiest path: write a small Rust
    # binary that ingests the npz and uses zenpredict_bake::bake().
    #
    # For now: dump the collapsed weights to a side file and instruct
    # the user to run the existing concat_three_way binary or a new
    # bake helper. The simplest immediate path is to drop into
    # `zensim-validate/src/bin/bake_v0_20b.rs` (TODO).

    # For now, save the collapsed weights as a follow-up baker input.
    collapsed_path = args.out.with_suffix(".collapsed.npz")
    np.savez(
        collapsed_path,
        n_inputs=np.array([n_inputs], dtype=np.int32),
        n_hidden=np.array([hidden_dim], dtype=np.int32),
        n_outputs=np.array([1], dtype=np.int32),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        # Layer 0: (n_inputs -> hidden_dim, LeakyReLU)
        # Note: numpy convention is (out, in); the Rust trainer expects
        # the same. ZNPR v3 weights are also stored as [out, in] per
        # the `bake_two_layer_znpr_v2` convention (row-major,
        # input-major: W[i * out_dim + o]).
        w0=w0.astype(np.float32),
        b0=b0.astype(np.float32),
        # Layer 1: (hidden_dim -> 1, Identity) — collapsed from head @ encoder_fc1
        w1=w_final.astype(np.float32),
        b1=b_final.astype(np.float32),
    )
    print(f"saved collapsed weights to {collapsed_path}")
    print(
        "\nTo finalize the ZNPR v3 bake, build a small Rust helper that"
        f" reads {collapsed_path} and calls zensim-validate's"
        " bake_two_layer_znpr_v2. (Phase 4b — pending)."
    )
    print(
        f"Or, run the affine-calibrate pipeline:"
        f"\n  python3 scripts/v_next/affine_calibrate_znpr_v2.py "
        f"--in-bake {collapsed_path} --out-bake {args.out} ..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
