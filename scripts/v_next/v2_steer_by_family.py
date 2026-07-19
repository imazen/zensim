#!/usr/bin/env python3
"""Where does the combined (720) model STEER from? Break output-sensitivity mass
down by v2 feature family, across held-out corpora. Extends v2_combined_steer_mass.py.

v2 block starts at f372; v2 feature (scale s, ch c, local l) = 372 + s*87 + c*29 + l.
Families by local index (per FEATURE_V2 idx):
  basic 0-8, soft-peak 9-11, masked 12-15, iw 16-19, pjnd 20-21,
  gms 22, transducer-bank 23-24, blockiness 25, ringing 26, banding 27, edge-width 28.
Plus the v1 blocks: v1-basic f0-155, v1-NONspat (peak/masked/iw) f156-371.

Usage: python3 scripts/v_next/v2_steer_by_family.py <bake> <corpus1,corpus2,...>
"""
import struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
PREDICT = REPO / "target/release/predict_features_with_bake"
AB = Path("/mnt/v/output/zensim/v2-ab-2026-07-19")
N = 200
W = 29
FAM = {  # local-index ranges within a v2 channel block
    "v2:basic": range(0, 9), "v2:soft-peak": range(9, 12), "v2:masked": range(12, 16),
    "v2:iw": range(16, 20), "v2:pjnd": range(20, 22), "v2:gms": range(22, 23),
    "v2:transducer-bank": range(23, 25), "v2:blockiness": range(25, 26),
    "v2:ringing": range(26, 27), "v2:banding": range(27, 28), "v2:edge-width": range(28, 29),
}


def forward(bake, mat):
    rows, feats = mat.shape
    with tempfile.NamedTemporaryFile(suffix=".blob", delete=False) as tf:
        tf.write(struct.pack("<II", feats, rows)); tf.write(mat.astype("<f4").tobytes()); blob = tf.name
    out = subprocess.run([str(PREDICT), "--bake", str(bake), "--bake-post", "raw",
                          "--features-file", blob], capture_output=True, text=True, check=True)
    Path(blob).unlink()
    return np.array([float(x) for x in out.stdout.split()])


def v2_cols(family_locals):
    return [372 + s * 87 + c * W + l for s in range(4) for c in range(3) for l in family_locals]


def importance(bake, pqf):
    t = pq.read_table(pqf)
    fc = sorted([c for c in t.column_names if c.startswith("f") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    X = np.column_stack([t.column(c).to_numpy() for c in fc]).astype("f4")
    S = X[np.random.default_rng(13).choice(X.shape[0], min(N, X.shape[0]), replace=False)]
    sig = S.std(axis=0); base = forward(bake, S)
    big = np.repeat(S[None], len(fc), axis=0)
    for k in range(len(fc)):
        big[k, :, k] += sig[k]
    imp = np.abs(forward(bake, big.reshape(len(fc) * S.shape[0], len(fc))).reshape(len(fc), S.shape[0]) - base[None]).mean(axis=1)
    return imp


bake = AB / sys.argv[1]
corpora = sys.argv[2].split(",")
blocks = {"v1-basic (f0-155)": list(range(0, 156)), "v1-NONspat (f156-371)": list(range(156, 372))}
for fam, locs in FAM.items():
    blocks[fam] = v2_cols(locs)
print(f"{'family':24s}" + "".join(f"{c:>10}" for c in corpora))
agg = {b: [] for b in blocks}
for c in corpora:
    imp = importance(bake, AB / f"ext_{c}.parquet")
    tot = imp.sum()
    for b, cols in blocks.items():
        agg[b].append(imp[[i for i in cols if i < len(imp)]].sum() / tot)
for b in blocks:
    print(f"{b:24s}" + "".join(f"{v:9.1%} " for v in agg[b]))
