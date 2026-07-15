#!/usr/bin/env python3
"""Test the column-audit's central actionable conclusion (§8.37 A / feedback_cvvdp_scalar_target):
raw cvvdp_score is a RANK-fine but MSE-saturated signal, so use it as a PAIRWISE-RANK auxiliary on
the near-lossless/HF band ALONGSIDE ssim2-MSE — never as an MSE target. Does that lift the HF/PJND
holdouts (KonJND, AIC-3) WITHOUT dropping CID22?

Shared 372-64-1 leaky MLP, de-poisoned (winsor p0.1/p99.9), safesyn+cid22_train.
  loss = smooth_l1(pred, z(ssim2)) + λ · pairwise_margin_rank(pred, cvvdp_score)
Sweep λ ∈ {0 (baseline), 0.1, 0.3, 0.6}. Eval CID22 (must hold), KonJND |SROCC|, AIC-3 |SROCC|
(structurally-signed → |·|). seed-avg. This is a SEED=1-style cheap probe (Step 3) before any bake.

  usage: cvvdp_hf_probe.py [--seeds 1,7] [--lambdas 0,0.1,0.3,0.6]
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

REPO = Path.home() / "work/zen/zensim"
_s = importlib.util.spec_from_file_location("T", REPO / "scripts/v_next/train_mlp_diverse.py")
T = importlib.util.module_from_spec(_s); _s.loader.exec_module(T)
N = 372
VAL = "/mnt/v/zen/zensim-training/canonical-2026-05-21/val"
KONJND = f"{VAL}/konjnd.parquet"
AIC3 = f"{VAL}/aic3.parquet"


def pairwise_rank_loss(pred, tgt, k, gen):
    """margin rank on k random pairs: if tgt_i>tgt_j want pred_i>pred_j (sign-margin=0.1)."""
    n = pred.shape[0]
    i = torch.randint(0, n, (k,), generator=gen, device=pred.device)
    j = torch.randint(0, n, (k,), generator=gen, device=pred.device)
    s = torch.sign(tgt[i] - tgt[j]).squeeze(-1)          # +1 if i better
    d = (pred[i] - pred[j]).squeeze(-1)
    return torch.clamp(0.1 - s * d, min=0.0).mean()


def run(lam, seeds, dev):
    Xcv, ycv = T.load(T.CID22_VAL, "human_score")
    Xkj, ykj = T.load(KONJND, "human_score")
    Xa3, ya3 = T.load(AIC3, "human_score")
    cids, kjs, a3s = [], [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        g = torch.Generator(device=dev); g.manual_seed(seed)
        Xs, ys = T.load(f"{T.TRAIN}/safesyn.parquet", "ssim2_gpu", 1.0, 90000, seed)
        Xc, yc = T.load(f"{T.TRAIN}/cid22_train.parquet", "ssim2_gpu", 1.0, None, seed)
        _, vs = T.load(f"{T.TRAIN}/safesyn.parquet", "cvvdp_score", 1.0, 90000, seed)
        _, vc = T.load(f"{T.TRAIN}/cid22_train.parquet", "cvvdp_score", 1.0, None, seed)
        X = np.concatenate([Xs, Xc]); y = np.concatenate([ys, yc]); v = np.concatenate([vs, vc])
        ok = np.isfinite(y) & np.isfinite(v) & np.isfinite(X).all(1); X, y, v = X[ok], y[ok], v[ok]
        rs = np.random.RandomState(1000 + seed); m = rs.rand(len(y)) < 0.8
        lo = np.percentile(X[m], 0.1, 0); hi = np.percentile(X[m], 99.9, 0)
        clip = lambda A: np.clip(A, lo, hi)
        Xtr = clip(X[m]); mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
        sc = lambda A: (clip(A) - mu) / sd
        yc2 = np.clip(y[m], np.percentile(y[m], 0.5), np.percentile(y[m], 99.5))
        tm, ts = yc2.mean(), yc2.std() or 1.0
        Xd = torch.tensor(sc(Xtr), dtype=torch.float32, device=dev)
        yd = torch.tensor((np.clip(y[m], tm - 5 * ts, tm + 5 * ts) - tm) / ts,
                          dtype=torch.float32, device=dev).unsqueeze(1)
        vd = torch.tensor(v[m], dtype=torch.float32, device=dev).unsqueeze(1)
        net = nn.Sequential(nn.Linear(N, 64), nn.LeakyReLU(0.01), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        for _ in range(400):
            opt.zero_grad()
            p = net(Xd)
            loss = nn.functional.smooth_l1_loss(p, yd)
            if lam > 0:
                loss = loss + lam * pairwise_rank_loss(p, vd, 4096, g)
            loss.backward(); opt.step()
        pr = lambda A: net(torch.tensor(sc(A), dtype=torch.float32, device=dev)).detach().cpu().numpy().ravel()
        cids.append(spearmanr(pr(Xcv), ycv).correlation)
        kjs.append(abs(spearmanr(pr(Xkj), ykj).correlation))
        a3s.append(abs(spearmanr(pr(Xa3), ya3).correlation))
    return np.mean(cids), np.mean(kjs), np.mean(a3s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,7")
    ap.add_argument("--lambdas", default="0,0.1,0.3,0.6")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in a.seeds.split(",")]
    lams = [float(x) for x in a.lambdas.split(",")]
    print(f"cvvdp-rank aux (de-poisoned, {len(seeds)} seeds). want: KonJND/AIC-3 ↑ WITHOUT CID22 ↓")
    print(f"{'λ(cvvdp-rank)':>14}  {'CID22':>8}  {'KonJND|S|':>9}  {'AIC-3|S|':>9}")
    base = None
    for lam in lams:
        c, k, a3 = run(lam, seeds, dev)
        if base is None:
            base = (c, k, a3)
        dc, dk, da = c - base[0], k - base[1], a3 - base[2]
        print(f"{lam:14.2f}  {c:8.4f}  {k:9.4f}  {a3:9.4f}   "
              f"(Δ CID22 {dc:+.4f}  KonJND {dk:+.4f}  AIC-3 {da:+.4f})")
    print("\nverdict: if a λ>0 row lifts KonJND+AIC-3 with |Δ CID22|<0.005, raw-cvvdp-rank is a real")
    print("HF lever -> promote to a full bake + bake_verdict (per-band Z-RMSE). Else falsify here.")


if __name__ == "__main__":
    main()
