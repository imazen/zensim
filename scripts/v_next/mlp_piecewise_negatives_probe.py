#!/usr/bin/env python3
"""Prototype: can ONE MLP represent B's piecewise projection —
hold CID22 (positive human-MOS, B's job) AND rank the deep-negative tail
(where B's winsor-linear head saturates to SROCC 0.047)?

Context (bhdr_improvement_split_lineage §8.31–§8.32, user 2026-07-14: "maybe
piecewise joint should lie above zero? should the joint blend? can an mlp
represent a piecewise projection?"):
  - §8.31 measured that shipped-B (winsor-linear) CANNOT rank negatives (0.047)
    and a monotone dial cannot add rank. A re-FIT can, but a single LINEAR head
    trades −0.061 CID22 for it (positive & negative want different projections).
  - The join between the two regimes lies ABOVE zero (B-score ~50 ≈ ssim2 0;
    §8.32 crossover table) and the two heads DISAGREE at the crossover
    (local SROCC 0.24–0.46) so a hand-built soft blend SCRAMBLES rank.
  - A ReLU MLP is piecewise-linear → it represents the gated two-projection
    function with a LEARNED, CONTINUOUS join (no hand threshold, no scrambling
    blend). This probe proves it: a 372-64-1 MLP trained on B's positive corpora
    (safesyn + cid22_train, ssim2 target) + KADIS-700k negatives (ssim2 target,
    down-weighted) holds CID22 ~0.865 AND lifts deep<-64 to ~0.76, robustly
    across seeds (no collapse), for −0.011 CID22 — 5x cheaper than the linear
    hard-switch and with better negatives.

  cid22_val is the SACRED holdout — human MOS, NEVER trained (loaded only to eval).
  Common target = ssim2 (the metric all corpora share; B is itself ssim2-shaped).

  usage: mlp_piecewise_negatives_probe.py [--kadis-weight 0.3] [--seeds 1,7,13,17,23]
"""
import argparse

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from scipy.stats import spearmanr

TRAIN = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
KADIS = "/mnt/v/output/zensim/reports/b_negatives/kadis_sample_negrich.parquet"
CID22_VAL = ("/mnt/v/zen/zensim-training/2026-05-15-full-features/"
             "cid22_features_372col_2026-05-15.parquet")
N_FEAT = 372


def load(path, ycol, npmax=None, seed=0):
    cols = [f.name for f in pq.read_schema(path)]
    pfx = "feat_" if "feat_0" in cols else "f"
    t = pq.read_table(path, columns=[ycol] + [f"{pfx}{i}" for i in range(N_FEAT)])
    X = np.stack([np.asarray(t[f"{pfx}{i}"], dtype=np.float32) for i in range(N_FEAT)], 1)
    y = np.asarray(t[ycol], dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    if npmax and len(y) > npmax:
        i = np.random.RandomState(seed).permutation(len(y))[:npmax]
        X, y = X[i], y[i]
    return X, y


def train_eval(seed, kw, Xct, yct, Xcv, ycv):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xsf, ysf = load(f"{TRAIN}/safesyn.parquet", "ssim2_gpu", 90000, seed)
    Xkd, ykd = load(KADIS, "score_ssim2_gpu", 90000, seed)

    def sp(X, y):
        m = np.random.rand(len(y)) < 0.75
        return X[m], y[m], X[~m], y[~m]

    a, b, c = sp(Xsf, ysf), sp(Xct, yct), sp(Xkd, ykd)
    Xtr = np.concatenate([a[0], b[0], c[0]])
    ytr = np.concatenate([a[1], b[1], c[1]])
    wtr = np.concatenate([np.ones(len(a[1])), np.ones(len(b[1])), np.full(len(c[1]), kw)])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    sc = lambda X: (X - mu) / sd
    sclip = np.clip(ytr, -150, 100)          # tame ssim2's −1834 tail for the LOSS only
    yn = (sclip - sclip.mean()) / sclip.std()  # (rank eval is clip-invariant)
    Xd = torch.tensor(sc(Xtr), dtype=torch.float32, device=dev)
    yd = torch.tensor(yn, dtype=torch.float32, device=dev).unsqueeze(1)
    wd = torch.tensor(wtr, dtype=torch.float32, device=dev).unsqueeze(1)
    net = nn.Sequential(nn.Linear(N_FEAT, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(400):
        opt.zero_grad()
        loss = (nn.functional.smooth_l1_loss(net(Xd), yd, reduction="none") * wd).mean()
        loss.backward()
        opt.step()
    pr = lambda X: net(torch.tensor(sc(X), dtype=torch.float32, device=dev)
                       ).detach().cpu().numpy().ravel()
    dk = c[3] < -64
    return (spearmanr(pr(Xcv), ycv).correlation,        # CID22 human-MOS (holdout)
            spearmanr(pr(c[2])[dk], c[3][dk]).correlation,  # KADIS deep<-64
            spearmanr(pr(a[2]), a[3]).correlation)          # safesyn-val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kadis-weight", type=float, default=0.3)
    ap.add_argument("--seeds", default="1,7,13,17,23")
    a = ap.parse_args()
    Xct, yct = load(f"{TRAIN}/cid22_train.parquet", "ssim2_gpu")     # ssim2-anchored, NOT MOS
    Xcv, ycv = load(CID22_VAL, "human_score")                       # SACRED MOS holdout
    print(f"seed   CID22    deep<-64   safesyn   (KADIS wt={a.kadis_weight}, 372-64-1)")
    R = []
    for s in [int(x) for x in a.seeds.split(",")]:
        cid, deep, ss = train_eval(s, a.kadis_weight, Xct, yct, Xcv, ycv)
        R.append((cid, deep))
        print(f"  {s:<4d} {cid:+.4f}  {deep:+.3f}   {ss:+.3f}")
    R = np.array(R)
    print(f"\n  CID22 {R[:,0].mean():.4f} ± {R[:,0].std():.4f}   "
          f"deep {R[:,1].mean():.3f} ± {R[:,1].std():.3f}   (collapse if any CID22<0.75)")
    print("  baselines: shipped-B CID22 0.876 / deep 0.047 | A 0.866 / 0.233 | ssim2 0.8895")


if __name__ == "__main__":
    main()
