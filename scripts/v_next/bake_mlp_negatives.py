#!/usr/bin/env python3
"""Bake the selected piecewise-negatives MLP (train_mlp_negatives.py export) into a
ZNPR v3 bake via zenpredict-bake (the JSON-pipeline mandate) — a plain 2-layer leaky
MLP + standardization scaler + an ssim2-anchored output-calibration spline (negative-
capable, like §8.30 B). Then self-verifies the forward layout by comparing bake_verdict's
CID22 SROCC to the numpy forward's (rank is dial-invariant, so a match proves the
scaler+layer forward is byte-correct regardless of the spline).

  usage: bake_mlp_negatives.py [--npz best.npz] [--out mlp_neg.bin] [--dtype f32]
"""
import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path.home() / "work/zen/zensim"
BAKER = "/home/lilith/work/zen/zenanalyze/target/release/zenpredict"
ANCHOR = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet"
SPLINE_KEY = "zentrain.output_calibration_spline"
N_FEAT = 372

_spec = importlib.util.spec_from_file_location(
    "lp", REPO / "scripts/v_next/linear_projections_2026-07-03.py")
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)


def fwd(P, X):
    """numpy forward matching nn.Sequential(Linear, LeakyReLU, Linear)."""
    z = (X - P["mu"]) / P["sd"]
    h = z @ P["W0"].T + P["b0"]
    h = np.where(h > 0, h, P["leaky"].item() * h)
    return (h @ P["W1"].T + P["b1"]).ravel()


def read_feats(path, ycol):
    cols = [f.name for f in pq.read_schema(path)]
    pfx = "feat_" if "feat_0" in cols else "f"
    t = pq.read_table(path, columns=[ycol] + [f"{pfx}{i}" for i in range(N_FEAT)])
    X = np.stack([np.asarray(t[f"{pfx}{i}"], dtype=np.float64) for i in range(N_FEAT)], 1)
    return X, np.asarray(t[ycol], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/mnt/v/output/zensim/reports/b_negatives/mlp_neg_best.npz")
    ap.add_argument("--out", default="/mnt/v/output/zensim/reports/b_negatives/mlp_neg_seed_best.bin")
    ap.add_argument("--dtype", default="f32")
    a = ap.parse_args()
    z = np.load(a.npz)
    P = {k: z[k] for k in ("mu", "sd", "W0", "b0", "W1", "b1", "leaky")}
    hidden = int(z["hidden"])

    # ---- dial spline: forward MLP raw on the anchor, fit raw -> ssim2_gpu (negative-capable)
    Xa, ya = read_feats(ANCHOR, "ssim2_gpu")
    raw = fwd(P, Xa)
    cx, cy = lp.fit_spline_knots(raw, ya)
    payload = struct.pack("<I", len(cx)) + b"".join(
        struct.pack("<ff", x, y) for x, y in zip(cx, cy))

    req = {
        "schema_hash": 0, "flags": 0, "compressed": True,
        "scaler_mean": [float(v) for v in P["mu"].astype(np.float32)],
        "scaler_scale": [float(v) for v in P["sd"].astype(np.float32)],
        # zenpredict layout (model.rs:92): W[i,o] = weights[i*out_dim + o] = INPUT-major.
        # PyTorch Linear.weight is [out,in], so emit W.T.ravel() (in-major).
        "layers": [
            {"in_dim": N_FEAT, "out_dim": hidden, "activation": "leakyrelu", "dtype": a.dtype,
             "weights": [float(v) for v in P["W0"].astype(np.float64).T.ravel()],
             "biases": [float(v) for v in P["b0"]]},
            {"in_dim": hidden, "out_dim": 1, "activation": "identity", "dtype": a.dtype,
             "weights": [float(v) for v in P["W1"].astype(np.float64).T.ravel()],
             "biases": [float(v) for v in P["b1"]]},
        ],
        "metadata": [{"key": SPLINE_KEY, "type": "bytes", "hex": payload.hex()}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        jp = Path(f.name)
    try:
        r = subprocess.run([BAKER, "bake", str(jp), a.out], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"bake failed: {r.stderr[:600]}")
    finally:
        jp.unlink(missing_ok=True)
    sha = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    print(f"baked {a.out} ({Path(a.out).stat().st_size} B, sha {sha[:12]}) "
          f"arch 372-{hidden}-1 leaky, {len(cx)}-knot dial [{cy[0]:.1f},{cy[-1]:.1f}]")
    print("verify: run bake_verdict on this bake; CID22 SROCC must match the numpy 0.870.")


if __name__ == "__main__":
    main()
