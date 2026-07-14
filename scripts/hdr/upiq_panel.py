#!/usr/bin/env python3
"""UPIQ-HDR panel for a zensim bake: forward the 372-feature UPIQ extraction
through the bake (dial-grid pred-dump path) and correlate vs JOD.

  usage: upiq_panel.py <bake.bin> [--features F.parquet] [--jod J.csv]
                       [--strata] [--compare OTHER.bin] [--boot 10000]

Prints |SROCC| + |PLCC| (JOD is negative-going; sign conventions differ per
metric, so absolute values — matching scripts/hdr/upiq_corr.py's convention
and the registry reference bars: cvvdp 0.758, iwssim-HDR 0.808,
ssim2-HDR 0.704, Profile A 0.694).

--strata additionally reports per-study |SROCC| (the honest read per
bhdr_improvement_split_lineage §8.1 — pooled UPIQ is scale-misalignment-
confounded across narwaria/korshunov). Strata come from the JOD csv pair-id
prefix (`n-…` narwaria, `k-…` korshunov), not hardcoded row ranges.

--compare OTHER.bin scores a second bake on the identical grid and reports
the per-stratum paired bootstrap on Δ|SROCC| (resampling pairs within the
stratum, `--boot` resamples): p = fraction of resamples where OTHER ≥ bake.
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("bake")
ap.add_argument("--features", default="/mnt/v/output/zensim-multicodec-probe/upiq_features_372.parquet",
                help="REGIME MATTERS (bhdr_improvement §8.13): this default is the v1 PU21 u8-shell extraction. PU-linear bakes (BHdr production path) need upiq_features_372_pulinear.parquet — mis-feeding understates SROCC by ~0.05 and reshuffles strata.")
ap.add_argument("--jod", default="/mnt/v/output/zenmetrics/upiq-pu/upiq_cid_jod.csv")
ap.add_argument("--strata", action="store_true")
ap.add_argument("--compare", default=None, help="second bake for paired per-stratum bootstrap")
ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()

t = pq.read_table(a.features)
n = t.num_rows
fcols = sorted((c for c in t.schema.names if c.startswith("feat_")), key=lambda c: int(c.split("_")[1]))
assert len(fcols) == 372 and n > 0, f"bad features table: {n} rows, {len(fcols)} fcols"

lines = [l for l in open(a.jod).read().splitlines() if l.strip()]
jod = np.array([float(l.split(",")[1]) for l in lines])
pair_ids = [l.split(",")[0] for l in lines]
assert len(jod) == n, f"jod {len(jod)} != features {n}"
STUDY = {"n": "narwaria", "k": "korshunov"}
strata = np.array([STUDY[p.split("-")[0]] for p in pair_ids])

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


def score_bake(bake_path: str, grid_parquet: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        pred = os.path.join(td, "pred.tsv")
        env = dict(os.environ, ZENSIM_DIAL_GRID=grid_parquet, ZENSIM_DIAL_PRED_OUT=pred)
        subprocess.run(
            [os.path.expanduser("~/work/zen/zensim/target/release/bake_verdict"),
             "--bake", bake_path, "--corpora", "aic3", "--output", os.devnull],
            env=env, capture_output=True, check=True)
        rows = [l.split("\t") for l in open(pred).read().splitlines()[1:]]
        hdr_cols = open(pred).readline().rstrip("\n").split("\t")
        qi, pi = hdr_cols.index("q"), hdr_cols.index("pred")
        preds = np.full(n, np.nan)
        for r in rows:
            preds[int(float(r[qi]))] = float(r[pi])
    assert not np.isnan(preds).any(), "pred dump incomplete"
    return preds


from scipy.stats import spearmanr, pearsonr


def report(name: str, preds: np.ndarray):
    sr = abs(spearmanr(preds, jod).statistic)
    pl = abs(pearsonr(preds, jod).statistic)
    print(f"{name}: UPIQ-HDR |SROCC|={sr:.4f} |PLCC|={pl:.4f} (n={n})")
    out = {}
    if a.strata or a.compare:
        for s in ("narwaria", "korshunov"):
            m = strata == s
            ss = abs(spearmanr(preds[m], jod[m]).statistic)
            out[s] = ss
            print(f"  {s:10s} |SROCC|={ss:.4f} (n={int(m.sum())})")
    return out


with tempfile.TemporaryDirectory() as td:
    gp = os.path.join(td, "upiq_grid.parquet")
    pq.write_table(grid, gp, compression="zstd")
    preds_a = score_bake(a.bake, gp)
    report(os.path.basename(a.bake), preds_a)
    if a.compare:
        preds_b = score_bake(a.compare, gp)
        report(os.path.basename(a.compare), preds_b)
        rng = np.random.default_rng(20260714)
        print(f"paired bootstrap Δ|SROCC| (A − B), {a.boot} resamples, within stratum:")
        for s in ("narwaria", "korshunov"):
            m = np.where(strata == s)[0]
            da, db, dj = preds_a[m], preds_b[m], jod[m]
            point = abs(spearmanr(da, dj).statistic) - abs(spearmanr(db, dj).statistic)
            wins = 0
            for _ in range(a.boot):
                idx = rng.integers(0, len(m), len(m))
                d = (abs(spearmanr(da[idx], dj[idx]).statistic)
                     - abs(spearmanr(db[idx], dj[idx]).statistic))
                if d <= 0:
                    wins += 1
            # one-sided p for "A > B": fraction of resamples where A does NOT beat B
            print(f"  {s:10s} Δ={point:+.4f}  p(A≤B)={wins / a.boot:.4f}")
