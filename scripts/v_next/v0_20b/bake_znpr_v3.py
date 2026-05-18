#!/usr/bin/env python3
"""V_20b Phase 4 — bake V_20b weights into ZNPR v3 via the JSON pipeline.

zensim CLAUDE.md mandate: bake-side serialization goes through the
zenpredict-bake JSON pipeline (`zenpredict-bake <in.json> <out.bin>`).
No ad-hoc Python emitters for the byte format.

The encoder + head from `finetune_head.py` is mathematically a 3-layer
MLP (228 → hidden_dim → embedding_dim → 1) but the embedding output
is Identity, so it collapses to a 2-layer net (228 → hidden_dim → 1)
via `W_final = W_head @ W_encoder_1`. The result is V_18-shape — same
ZNPR v3 wire format used by every shipped bake.

## Usage

  python3 scripts/v_next/v0_20b/bake_znpr_v3.py \\
    --finetune /tmp/v0_20b_full_unfrozen.npz \\
    --out benchmarks/v0_20b_seed1_2026-05-15.bin

Requires `zenpredict-bake` binary built and on PATH (or set
`--zenpredict-bake /path/to/zenpredict-bake`). Default lookup:
- `target/release/zenpredict-bake` under `~/work/zen/zenanalyze/`
- `target/release/zenpredict-bake` under the current dir
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def find_baker(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"--zenpredict-bake {explicit} not found")
    candidates = [
        Path.home() / "work/zen/zenanalyze/target/release/zenpredict-bake",
        Path.cwd() / "target/release/zenpredict-bake",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"zenpredict-bake binary not found in {candidates}. "
        "Build it: cd ~/work/zen/zenanalyze && cargo build --release -p zenpredict-bake"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--zenpredict-bake",
        type=Path,
        default=None,
        help="Path to zenpredict-bake binary. Default auto-detects.",
    )
    ap.add_argument(
        "--zerobias-tau",
        type=float,
        default=0.0,
        help="Pre-quantization per-layer zerobias threshold "
        "(zenpredict-bake 0.1.1+). Default 0.0 (disabled). Calibrated "
        "value: 0.005 → 87.5%% i8 zero density at -0.0001 SROCC on "
        "V0_18. Pair with --compress.",
    )
    ap.add_argument(
        "--compress",
        action="store_true",
        help="LZ4-block-compress the post-header payload at bake time.",
    )
    ap.add_argument(
        "--optimize",
        action="store_true",
        help="Run bake_optimized (permutation + compressed-flag search "
        "+ bounded swap hillclimb). ~1-2 s budget.",
    )
    args = ap.parse_args()

    baker = find_baker(args.zenpredict_bake)

    blob = np.load(args.finetune)
    n_inputs = int(blob["n_inputs"][0])
    hidden_dim = int(blob["hidden_dim"][0])
    scaler_mean = blob["scaler_mean"].astype(np.float32)
    scaler_scale = blob["scaler_scale"].astype(np.float32)
    enc_w0 = blob["encoder_w0"]  # PyTorch shape: [hidden_dim, n_inputs]
    enc_b0 = blob["encoder_b0"]  # [hidden_dim]
    enc_w1 = blob["encoder_w1"]  # [embedding_dim, hidden_dim]
    enc_b1 = blob["encoder_b1"]  # [embedding_dim]
    head_w = blob["head_w"]  # [1, embedding_dim]
    head_b = blob["head_b"]  # [1]

    # Collapse: final layer = head @ encoder_fc1, accounting for biases.
    w_final = head_w @ enc_w1  # [1, hidden_dim]
    b_final = (head_w @ enc_b1 + head_b).astype(np.float32)  # [1]

    # ZNPR v3 weight storage: row-major INPUT-MAJOR.
    # `W[i * out_dim + o]` is contribution of input `i` to output `o`.
    # PyTorch nn.Linear weight shape: [out, in].
    # → emit as `pytorch_W.T.reshape(-1)` for the bake JSON.
    w0_flat = enc_w0.T.astype(np.float32).reshape(-1).tolist()
    b0_flat = enc_b0.astype(np.float32).tolist()
    w1_flat = w_final.T.astype(np.float32).reshape(-1).tolist()
    b1_flat = b_final.astype(np.float32).tolist()

    req = {
        "schema_hash": 0,
        "flags": 0,
        "scaler_mean": scaler_mean.tolist(),
        "scaler_scale": scaler_scale.tolist(),
        "layers": [
            {
                "in_dim": n_inputs,
                "out_dim": hidden_dim,
                "activation": "leakyrelu",
                "dtype": "f32",
                "weights": w0_flat,
                "biases": b0_flat,
            },
            {
                "in_dim": hidden_dim,
                "out_dim": 1,
                "activation": "identity",
                "dtype": "f32",
                "weights": w1_flat,
                "biases": b1_flat,
            },
        ],
        # No feature_transforms metadata — V_20b is plain 228-feature input.
        "metadata": [],
    }
    # Bake-time compression knobs (zenpredict-bake 0.1.1+). Each key
    # is emitted only when non-default; pre-0.1.1 baker binaries
    # ignore unknown keys silently.
    if args.zerobias_tau > 0.0:
        req["zerobias_tau"] = float(args.zerobias_tau)
    if args.compress:
        req["compressed"] = True
    if args.optimize:
        req["optimize"] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(req, f)
        json_path = Path(f.name)

    try:
        result = subprocess.run(
            [str(baker), str(json_path), str(args.out)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(
                f"zenpredict-bake failed (exit {result.returncode})", file=sys.stderr
            )
            return result.returncode
    finally:
        json_path.unlink(missing_ok=True)
    knobs = []
    if args.zerobias_tau > 0.0:
        knobs.append(f"zerobias_tau={args.zerobias_tau:.6g}")
    if args.compress:
        knobs.append("compressed")
    if args.optimize:
        knobs.append("optimize")
    knob_str = (" knobs=[" + ", ".join(knobs) + "]") if knobs else ""
    print(
        f"baked {args.out} via {baker.name}: {n_inputs} → {hidden_dim} "
        f"(LeakyReLU) → 1 (Identity){knob_str}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
