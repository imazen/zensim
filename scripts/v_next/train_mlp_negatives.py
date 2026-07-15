#!/usr/bin/env python3
"""Productionize the §8.32 piecewise-negatives MLP into a ship candidate, judged on
B's own crucible. Trains a plain 2-layer leaky MLP (bakeable via zenpredict-bake,
forwarded by bake_verdict's standard Predictor path — no per-sample-alpha/tanh heads),
multi-seed with a collapse gate, and TRAINING-SIDE checkpoint selection (CID22 stays a
pure holdout — never used to pick a seed).

Corpora = B's SDR positive set (safesyn + cid22_train + kadid + tid, ssim2 target) +
KADIS-700k negatives (ssim2 target, down-weighted). Common target = ssim2 (what B's kon
head used: per-corpus ssim2-derived anchor). Exports the SELECTED seed's weights + scaler
to npz for baking.

  usage: train_mlp_negatives.py [--hidden 64] [--kadis-weight 0.3]
                                [--seeds 1,3,7,13,17,23,31,41] [--out best.npz]
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
# positive corpora (canonical, ssim2_gpu target) + their per-run row cap.
# kadid/tid DELIBERATELY EXCLUDED: their canonical `ssim2_gpu` column is the known
# ref-vs-ref misjoin data bug (CLAUDE.md) — training on it ranks BACKWARDS (val SROCC
# −0.07..−0.13) and drags CID22 + the negatives. safesyn+cid22_train is B's clean
# CID22-relevant positive set; KADIS supplies the negatives.
POS = [("safesyn", f"{TRAIN}/safesyn.parquet", "ssim2_gpu", 90000),
       ("cid22_train", f"{TRAIN}/cid22_train.parquet", "ssim2_gpu", None)]


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


def build(seed):
    """Return train/val splits for every corpus (25% held out per corpus)."""
    rs = np.random.RandomState(1000 + seed)
    tr_X, tr_y, tr_w, val = [], [], [], {}
    for name, path, ycol, cap in POS:
        X, y = load(path, ycol, cap, seed)
        m = rs.rand(len(y)) < 0.75
        tr_X.append(X[m]); tr_y.append(y[m]); tr_w.append(np.ones(m.sum()))
        val[name] = (X[~m], y[~m])
    Xk, yk = load(KADIS, "score_ssim2_gpu", 90000, seed)
    mk = rs.rand(len(yk)) < 0.75
    val["kadis"] = (Xk[~mk], yk[~mk])
    return tr_X, tr_y, tr_w, (Xk[mk], yk[mk]), val


def train_one(seed, hidden, kw, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_X, tr_y, tr_w, (Xk, yk), val = build(seed)
    Xtr = np.concatenate(tr_X + [Xk]); ytr = np.concatenate(tr_y + [yk])
    wtr = np.concatenate(tr_w + [np.full(len(yk), kw)])
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    sc = lambda X: (X - mu) / sd
    sclip = np.clip(ytr, -150, 100); tm, ts = sclip.mean(), sclip.std()
    yn = (sclip - tm) / ts
    Xd = torch.tensor(sc(Xtr), dtype=torch.float32, device=dev)
    yd = torch.tensor(yn, dtype=torch.float32, device=dev).unsqueeze(1)
    wd = torch.tensor(wtr, dtype=torch.float32, device=dev).unsqueeze(1)
    net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.LeakyReLU(0.01),
                        nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(400):
        opt.zero_grad()
        loss = (nn.functional.smooth_l1_loss(net(Xd), yd, reduction="none") * wd).mean()
        loss.backward(); opt.step()
    pr = lambda X: net(torch.tensor(sc(X), dtype=torch.float32, device=dev)
                       ).detach().cpu().numpy().ravel()
    metrics = {}
    for name, (Xv, yv) in val.items():
        if name == "kadis":
            dk = yv < -64
            metrics["kadis_deep"] = spearmanr(pr(Xv)[dk], yv[dk]).correlation
            metrics["kadis_all"] = spearmanr(pr(Xv), yv).correlation
        else:
            metrics[name] = spearmanr(pr(Xv), yv).correlation
    # export payload for the selected seed
    W0 = net[0].weight.detach().cpu().numpy(); b0 = net[0].bias.detach().cpu().numpy()
    W1 = net[2].weight.detach().cpu().numpy(); b1 = net[2].bias.detach().cpu().numpy()
    return metrics, dict(mu=mu, sd=sd, W0=W0, b0=b0, W1=W1, b1=b1, hidden=hidden,
                         leaky=0.01, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--kadis-weight", type=float, default=0.3)
    ap.add_argument("--seeds", default="1,3,7,13,17,23,31,41")
    ap.add_argument("--out", default="/mnt/v/output/zensim/reports/b_negatives/mlp_neg_best.npz")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xcv, ycv = load(CID22_VAL, "human_score")  # SACRED holdout — report only, never select
    pos_names = [n for n, *_ in POS]
    print("seed  " + " ".join(f"{n[:8]:>8s}" for n in pos_names)
          + " kadis_deep kadis_all | CID22(holdout)  sel(geomean)")
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        m, payload = train_one(s, a.hidden, a.kadis_weight, dev)
        # collapse gate: any positive-corpus val SROCC < 0.85 = collapse -> reject
        collapse = min(m[n] for n in pos_names) < 0.85
        # training-side selection score (NO CID22): geomean(min positive, kadis_deep)
        sel = float(np.sqrt(max(1e-6, min(m[n] for n in pos_names)) * max(1e-6, m["kadis_deep"])))
        cid = cid22_score(payload, Xcv, ycv, dev)  # holdout — report only
        tag = "COLLAPSE" if collapse else f"{sel:.3f}"
        print(f" {s:<4d} " + " ".join(f"{m[n]:+8.3f}" for n in pos_names)
              + f"  {m['kadis_deep']:+.4f}  {m['kadis_all']:+.3f}  |  {cid:+.4f}      {tag}")
        if not collapse:
            rows.append((sel, cid, m, payload))
    if not rows:
        print("ALL SEEDS COLLAPSED — aborting"); return
    rows.sort(key=lambda r: -r[0])
    sel, cid, m, payload = rows[0]
    np.savez(a.out, **{k: v for k, v in payload.items()})
    print(f"\nSELECTED seed {payload['seed']} (training-side geomean {sel:.3f}): "
          f"CID22 {cid:.4f}, kadis_deep {m['kadis_deep']:.3f}, safesyn {m['safesyn']:.3f}")
    print(f"exported -> {a.out}")
    print("baselines: shipped-B CID22 0.876 / deep 0.047 | A 0.866 / 0.233 | ssim2 0.8895")


def cid22_score(payload, Xcv, ycv, dev):
    mu, sd = payload["mu"], payload["sd"]
    X = (Xcv - mu) / sd
    h = X @ payload["W0"].T + payload["b0"]
    h = np.where(h > 0, h, 0.01 * h)
    o = h @ payload["W1"].T + payload["b1"]
    return spearmanr(o.ravel(), ycv).correlation


if __name__ == "__main__":
    main()
