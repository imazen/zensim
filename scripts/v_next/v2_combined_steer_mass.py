#!/usr/bin/env python3
"""Coherence-risk proxy for the combined (720) append-only model: how much of the
model's OUTPUT sensitivity sits on the NON-spatializable v1 block (f156..371 =
v1 peak/masked/iw), which the diffmap fold cannot represent per-pixel.

This is the cheap early read on the M3 risk that combining old+new re-introduces
(shipped-B's 38% non-basic mass caps its M3 at 0.66). NOT the full M3 (that needs
the diffmap fold extended to read the v2 block); it's the sensitivity-mass proxy,
computed with the existing forward tool — no Rust surgery.

Method: sample N rows; perturb each feature k by 1σ_k; one batched forward of all
720 perturbations; importance_k = mean|Δscore|. Report Σimp over the blocks:
  spatializable  = f0..155 (v1 basic) ∪ f372..719 (v2, all bounded per-pixel maps)
  non-spatial.   = f156..371 (v1 peak/masked/iw)  ← the coherence risk

Usage: python3 scripts/v_next/v2_combined_steer_mass.py <bake> <ext_parquet> [ncap]
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
PREDICT = REPO / "target/release/predict_features_with_bake"
N = 200  # sample rows
EPS_SIGMA = 1.0


def forward(bake, mat):  # mat: (rows, feats) float32
    rows, feats = mat.shape
    with tempfile.NamedTemporaryFile(suffix=".blob", delete=False) as tf:
        tf.write(struct.pack("<II", feats, rows))
        tf.write(mat.astype("<f4").tobytes())
        blob = tf.name
    out = subprocess.run(
        [str(PREDICT), "--bake", str(bake), "--bake-post", "raw", "--features-file", blob],
        capture_output=True, text=True, check=True)
    Path(blob).unlink()
    return np.array([float(x) for x in out.stdout.split()])


def main():
    bake, pqf = Path(sys.argv[1]), Path(sys.argv[2])
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else None
    t = pq.read_table(pqf)
    fc = sorted([c for c in t.column_names if c.startswith("f") and c[1:].isdigit()],
                key=lambda c: int(c[1:]))
    if cap:
        fc = fc[:cap]
    F = len(fc)
    X = np.column_stack([t.column(c).to_numpy() for c in fc]).astype("f4")
    rng = np.random.default_rng(13)
    idx = rng.choice(X.shape[0], size=min(N, X.shape[0]), replace=False)
    S = X[idx]  # (n, F)
    sigma = S.std(axis=0)  # per-feature spread
    base = forward(bake, S)  # (n,)
    # batched perturbation: block k = S with column k += sigma_k; stack (F*n, F)
    big = np.repeat(S[None, :, :], F, axis=0)  # (F, n, F)
    for k in range(F):
        big[k, :, k] += sigma[k]
    big = big.reshape(F * S.shape[0], F)
    pert = forward(bake, big).reshape(F, S.shape[0])
    imp = np.abs(pert - base[None, :]).mean(axis=1)  # (F,) importance per feature
    total = imp.sum()
    nonspat = imp[156:372].sum() if F > 156 else 0.0
    v1basic = imp[0:156].sum()
    v2 = imp[372:F].sum() if F > 372 else 0.0
    print(f"features={F}  sample={S.shape[0]}")
    print(f"  v1-basic (f0..155)      importance mass: {v1basic/total:6.1%}")
    print(f"  v1 NON-spat (f156..371) importance mass: {nonspat/total:6.1%}  <-- coherence risk")
    if F > 372:
        print(f"  v2 block (f372..{F-1})    importance mass: {v2/total:6.1%}")
    spat = v1basic + v2
    print(f"  => spatializable total: {spat/total:6.1%} | non-spatializable: {nonspat/total:6.1%}")


if __name__ == "__main__":
    main()
