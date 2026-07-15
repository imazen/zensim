#!/usr/bin/env python3
"""Diagnose WHY cvvdp trains worse (user: 'we should diagnose why data is bad, not just that it
failed a gate'). V41 (2026-05-27) found cvvdp-target -> CID22 0.66 vs ssim2's 0.88 and called
cvvdp a dead end. But the paper (Mohammadi 2025) ranks cvvdp the BEST metric, and on safesyn
cvvdp is learnable (feat->cvvdp 0.987) and agrees with ssim2 (0.984). So the 0.66 smells like a
CONFOUND (saturated raw target and/or the IW poison), not a real cvvdp limit.

This runs the SAME de-poisoned pipeline (winsor + 372-64-1 leaky MLP, safesyn+cid22_train, CID22
a pure holdout) toward three targets and reports CID22 + safesyn-val SROCC for each:
  - ssim2_gpu       (the shipped baseline target)
  - cvvdp_log_norm  ([0,100], un-saturated)
  - cvvdp_score     (raw [1.68,10], median 9.80 — SATURATED; V41's likely target)
If cvvdp_log_norm recovers CID22 toward ssim2's level, V41 was confounded by the raw saturation.

  usage: cvvdp_target_probe.py [--seeds 1,7,13,17]
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


def run(target, seeds, dev):
    Xcv, ycv = T.load(T.CID22_VAL, "human_score")  # pure holdout
    cids, svals = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        Xs, ys = T.load(f"{T.TRAIN}/safesyn.parquet", target, 1.0, 90000, seed)
        Xc, yc = T.load(f"{T.TRAIN}/cid22_train.parquet", target, 1.0, None, seed)
        X = np.concatenate([Xs, Xc]); y = np.concatenate([ys, yc])
        ok = np.isfinite(y) & np.isfinite(X).all(1); X, y = X[ok], y[ok]
        rs = np.random.RandomState(1000 + seed); m = rs.rand(len(y)) < 0.8
        # winsor de-poison (same as the shipped fix)
        lo = np.percentile(X[m], 0.1, 0); hi = np.percentile(X[m], 99.9, 0)
        clip = lambda A: np.clip(A, lo, hi)
        Xtr = clip(X[m]); mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
        sc = lambda A: (clip(A) - mu) / sd
        yc_clip = np.clip(y[m], np.percentile(y[m], 0.5), np.percentile(y[m], 99.5))
        tm, ts = yc_clip.mean(), yc_clip.std() or 1.0
        Xd = torch.tensor(sc(Xtr), dtype=torch.float32, device=dev)
        yd = torch.tensor((np.clip(y[m], tm - 5 * ts, tm + 5 * ts) - tm) / ts,
                          dtype=torch.float32, device=dev).unsqueeze(1)
        net = nn.Sequential(nn.Linear(N, 64), nn.LeakyReLU(0.01), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        for _ in range(400):
            opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(net(Xd), yd)
            loss.backward(); opt.step()
        pr = lambda A: net(torch.tensor(sc(A), dtype=torch.float32, device=dev)
                           ).detach().cpu().numpy().ravel()
        cids.append(spearmanr(pr(Xcv), ycv).correlation)
        svals.append(spearmanr(pr(X[~m]), y[~m]).correlation)
    return np.mean(cids), np.std(cids), np.mean(svals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,7,13,17")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"target             CID22(holdout)     train-val   (de-poisoned pipeline, {len(seeds)} seeds)")
    for tgt in ["ssim2_gpu", "cvvdp_log_norm", "cvvdp_score"]:
        cm, cs, sv = run(tgt, seeds, dev)
        print(f"  {tgt:16s}  {cm:.4f} ± {cs:.4f}    {sv:.4f}")
    print("\nV41 claimed cvvdp -> CID22 0.66. If cvvdp_log_norm here is ~0.85+, V41 was")
    print("confounded (raw-saturated target / old poison), NOT a real cvvdp-target limit.")


if __name__ == "__main__":
    main()
