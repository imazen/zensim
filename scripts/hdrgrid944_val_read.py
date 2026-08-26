#!/usr/bin/env python3
"""hdrgrid944 in-domain val read — mirror of `hdrp1_val_read.py` at the leg's
native 944 width: forward a bake over the hdrgrid944 val split, report
SROCC/PLCC/ZRMSE vs the cvvdp-mix target (selection/sanity only; never a UPIQ
substitute). Owners only: forward = predict_features_with_bake, stats =
scripts/lib/zen_stats.panel.

  usage: hdrgrid944_val_read.py <bake.bin> [--parquet P]
"""
import argparse, os, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.lib import zen_stats  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("bake")
ap.add_argument("--parquet",
                default="/mnt/v/output/zensim/hdrgrid944-leg/hdrgrid944_v3mix_valdigits_2026-08-26.parquet")
a = ap.parse_args()

t = pq.read_table(a.parquet)
fcols = [f"f{i}" for i in range(944)]
assert all(c in t.schema.names for c in fcols), "parquet lacks f0..f371"
X = np.column_stack([np.asarray(t[c].to_pylist(), np.float32) for c in fcols])
y = np.asarray(t["human_score"].to_pylist(), float)

tool = os.environ.get(
    "ZL_PREDICT", str(REPO / "target/release/predict_features_with_bake"))
with tempfile.NamedTemporaryFile(suffix=".wire", delete=False) as f:
    f.write(struct.pack("<II", X.shape[1], X.shape[0]))
    f.write(X.astype("<f4").tobytes())
    wire = f.name
try:
    r = subprocess.run([tool, "--bake", a.bake, "--features-file", wire],
                       capture_output=True, text=True, check=True)
finally:
    os.unlink(wire)
pred = np.array([float(v) for v in r.stdout.split()])
assert len(pred) == len(y), (len(pred), len(y))
p = zen_stats.panel(list(pred), list(y))
print(f"{os.path.basename(a.bake)} vs {os.path.basename(a.parquet)} (n={len(y)}): "
      f"SROCC={p['srocc']:.4f} PLCC={p['plcc']:.4f} ZRMSE={p.get('z_rmse', float('nan')):.4f}")
