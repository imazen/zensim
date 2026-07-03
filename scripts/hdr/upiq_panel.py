#!/usr/bin/env python3
"""UPIQ-HDR panel for a zensim bake: forward the 372-feature UPIQ extraction
through the bake (dial-grid pred-dump path) and correlate vs JOD.

  usage: upiq_panel.py <bake.bin> [--features F.parquet] [--jod J.csv]

Prints |SROCC| + |PLCC| (JOD is negative-going; sign conventions differ per
metric, so absolute values — matching scripts/hdr/upiq_corr.py's convention
and the registry reference bars: cvvdp 0.758, iwssim-HDR 0.808,
ssim2-HDR 0.704, Profile A 0.694).
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("bake")
ap.add_argument("--features", default="/mnt/v/output/zensim-multicodec-probe/upiq_features_372.parquet")
ap.add_argument("--jod", default="/mnt/v/output/zenmetrics/upiq-pu/upiq_cid_jod.csv")
a = ap.parse_args()

t = pq.read_table(a.features)
n = t.num_rows
fcols = sorted((c for c in t.schema.names if c.startswith("feat_")), key=lambda c: int(c.split("_")[1]))
assert len(fcols) == 372 and n > 0, f"bad features table: {n} rows, {len(fcols)} fcols"

jod = [float(l.split(",")[1]) for l in open(a.jod).read().splitlines() if l.strip()]
assert len(jod) == n, f"jod {len(jod)} != features {n}"

# Shape as a dial grid: one 'codec ladder' per row so the pred dump's
# (group-ordinal, local-index) re-keying is trivially invertible by q.
data = {
    "ref_basename": pa.array([f"upiq{i}" for i in range(n)]),
    "human_score": pa.array([0.5] * n),
    "image_path": pa.array([f"upiq{i}" for i in range(n)]),
    "image_id": pa.array(["upiq"] * n),
    "codec": pa.array(["upiq"] * n),
    "q": pa.array([float(i) for i in range(n)]),
    "knob_tuple_json": pa.array(["{}"] * n),
}
for i, c in enumerate(fcols):
    data[f"f{i}"] = t[c]
grid = pa.table(data)
with tempfile.TemporaryDirectory() as td:
    gp = os.path.join(td, "upiq_grid.parquet")
    pq.write_table(grid, gp)
    pred = os.path.join(td, "pred.tsv")
    env = dict(os.environ, ZENSIM_DIAL_GRID=gp, ZENSIM_DIAL_PRED_OUT=pred)
    subprocess.run(
        [os.path.expanduser("~/work/zen/zensim/target/release/bake_verdict"),
         "--bake", a.bake, "--corpora", "aic3", "--output", os.devnull],
        env=env, capture_output=True, check=True)
    rows = [l.split("\t") for l in open(pred).read().splitlines()[1:]]
    hdr_cols = open(pred).readline().rstrip("\n").split("\t")
    qi, pi = hdr_cols.index("q"), hdr_cols.index("pred")
    preds = np.full(n, np.nan)
    for r in rows:
        preds[int(float(r[qi]))] = float(r[pi])
assert not np.isnan(preds).any(), "pred dump incomplete"

from scipy.stats import spearmanr, pearsonr
sr = abs(spearmanr(preds, jod).statistic)
pl = abs(pearsonr(preds, jod).statistic)
print(f"{os.path.basename(a.bake)}: UPIQ-HDR |SROCC|={sr:.4f} |PLCC|={pl:.4f} (n={n})")
