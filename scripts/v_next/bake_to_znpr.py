#!/usr/bin/env python3
"""Convert a trained PyTorch MLP into a ZNPR v2 binary for zensim's V0_4 path.

Pipeline:
    1. Load `model.pt` (PyTorch state_dict from train_v_next_mlp.py).
    2. Load `scaler.npz` (per-feature mean + std from train_v_next_mlp.py).
    3. Build the BakeRequestJson shape that zenpredict-bake consumes.
    4. Spawn zenpredict-bake to produce the `.bin`.

The resulting `.bin` is what zensim's V0_4 path's `include_bytes!`
loads via `zenpredict::Model::from_bytes`.

Usage:
    python3 scripts/v_next/bake_to_znpr.py \\
        --run-dir /mnt/v/zen/zensim-training/2026-05-07/runs/<timestamp>_<tag> \\
        --out zensim/weights/v0_4_<date>.bin

The `--bake-bin` defaults to `~/work/zen/zenanalyze/target/release/zenpredict-bake`;
override if your zenanalyze checkout is elsewhere.
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
import torch


DEFAULT_BAKE_BIN = Path.home() / "work/zen/zenanalyze/target/release/zenpredict-bake"


def state_dict_to_layers(sd: dict[str, torch.Tensor]) -> tuple[list[dict], int, int]:
    """Walk a Sequential(Linear, LeakyReLU, ..., Linear) state_dict and emit
    JSON-shaped layers + (n_inputs, n_outputs).

    Mirrors the architecture in `MLP` from train_v_next_mlp.py:
        Linear -> LeakyReLU -> Linear -> LeakyReLU -> ... -> Linear (final)
    """
    keys = sorted(sd.keys(),
                   key=lambda k: int(k.split(".")[1]))  # net.0.weight → 0
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    cur_w = None
    for k in keys:
        if k.endswith(".weight"):
            cur_w = sd[k]
        elif k.endswith(".bias"):
            assert cur_w is not None, f"bias before weight at {k}"
            pairs.append((cur_w, sd[k]))
            cur_w = None
    if cur_w is not None:
        raise SystemExit("dangling weight tensor without matching bias")

    last = len(pairs) - 1
    layers = []
    for i, (W, b) in enumerate(pairs):
        out_dim, in_dim = W.shape
        # Trainer uses LeakyReLU between layers and identity on the final.
        activation = "identity" if i == last else "leaky_relu"
        layers.append({
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "activation": activation,
            "dtype": "f32",
            # PyTorch Linear weight is (out, in); ZNPR expects row-major
            # (in, out) flattened per-output. Transpose first.
            "weights": W.t().contiguous().flatten().tolist(),
            "biases": b.flatten().tolist(),
        })
    return layers, layers[0]["in_dim"], layers[-1]["out_dim"]


def build_bake_request(run_dir: Path) -> dict:
    sd_path = run_dir / "model.pt"
    scaler_path = run_dir / "scaler.npz"
    meta_path = run_dir / "meta.json"
    if not sd_path.exists():
        raise SystemExit(f"missing {sd_path}")
    if not scaler_path.exists():
        raise SystemExit(f"missing {scaler_path}")

    sd = torch.load(sd_path, map_location="cpu", weights_only=True)
    scaler = np.load(scaler_path)
    layers_json, n_in, n_out = state_dict_to_layers(sd)

    if scaler["mean"].shape[0] != n_in:
        raise SystemExit(
            f"scaler.mean has {scaler['mean'].shape[0]} entries, "
            f"first layer expects {n_in} inputs"
        )

    # Read training metadata so we can stamp it into the bake's metadata
    # blob — this lets `zenpredict::Model::metadata()` surface the train
    # provenance at runtime.
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    metadata = []
    for k, v in (meta.get("config") or {}).items():
        metadata.append({"key": f"train.{k}", "type": "text",
                         "text": str(v)})
    for k, v in (meta.get("metrics") or {}).items():
        metadata.append({"key": f"metric.{k}", "type": "text",
                         "text": str(v)})
    metadata.append({"key": "zensim.profile", "type": "text",
                     "text": "zensim-preview-v0.4"})

    return {
        # 0 = no schema enforcement; runtime accepts any 228-input MLP.
        "schema_hash": 0,
        "flags": 0,
        "scaler_mean": scaler["mean"].astype(np.float32).tolist(),
        "scaler_scale": scaler["std"].astype(np.float32).tolist(),
        "layers": layers_json,
        "feature_bounds": [],
        "metadata": metadata,
        # ZNPR v3 fields (optional; empty arrays = v2-style passthrough).
        "output_specs": [],
        "discrete_sets": [],
        "sparse_overrides": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="Path to runs/<ts>_<tag>/ written by train_v_next_mlp.py")
    ap.add_argument("--out", required=True,
                    help="Output .bin path")
    ap.add_argument("--bake-bin", default=str(DEFAULT_BAKE_BIN),
                    help=f"Path to zenpredict-bake (default: {DEFAULT_BAKE_BIN})")
    ap.add_argument("--keep-json", action="store_true",
                    help="Persist the intermediate BakeRequestJson next to --out")
    args = ap.parse_args()

    bake_bin = Path(args.bake_bin)
    if not bake_bin.is_file():
        raise SystemExit(f"zenpredict-bake not found at {bake_bin}; "
                         f"run `cargo build --release -p zenpredict-bake` "
                         f"in zenanalyze first")

    run_dir = Path(args.run_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building BakeRequestJson from {run_dir} ...")
    req = build_bake_request(run_dir)
    n_in = req["layers"][0]["in_dim"]
    n_out = req["layers"][-1]["out_dim"]
    n_layers = len(req["layers"])
    n_meta = len(req["metadata"])
    print(f"  {n_in} inputs → {n_layers} layers → {n_out} outputs, "
          f"{n_meta} metadata entries")

    if args.keep_json:
        json_out = out_path.with_suffix(".bake.json")
    else:
        json_fd, json_out = tempfile.mkstemp(suffix=".bake.json", text=True)
        os.close(json_fd)
        json_out = Path(json_out)
    json_out.write_text(json.dumps(req))
    print(f"Wrote BakeRequestJson to {json_out} ({json_out.stat().st_size:,} bytes)")

    cmd = [str(bake_bin), str(json_out), str(out_path)]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(f"zenpredict-bake exit {res.returncode}")
    if res.stdout:
        print(res.stdout)

    if not args.keep_json:
        try:
            json_out.unlink()
        except OSError:
            pass

    print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
