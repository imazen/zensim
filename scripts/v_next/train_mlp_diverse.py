#!/usr/bin/env python3
"""Task #17 — imazen-26 DIVERSE retrain of the §8.33 piecewise-negatives MLP.

The user's concern: the §8.33 MLP (and B) train on a photographic-dominant set
(safesyn tiles + cid22_train photos + KADIS Pixabay/analytic). imazen-26 fills the
content gaps — screen/UI, line-art/vector/charts, documents/scans/bilevel, AI-gen
graphics, artwork — AND supplies REAL modern-codec distortions (the RD the dial must
rank). See docs/DATASET_HISTORY.md §5 for the full rationale + guards.

Corpus set (all on the ssim2 0..100 scale, standardized for the loss):
  POS  : safesyn (cap) + cid22_train         — B's clean CID22-relevant positive set
  DIV  : bigcodec_hqdedup_traindigits        — 2.32M imazen-26 real-codec cells,
         target human_score=ssim2/100 -> ×100. HONORS the HQ-saturation confound
         (DATASET_HISTORY.md §0.1/§5): ssim2 saturates >0.85 (cvvdp-agree ~0.48) and
         densifying HQ made prior bakes WORSE -> rows with target>85 are down-weighted.
  NEG  : KADIS-700k neg-rich                  — ssim2 negative tail (solves B's floor)

Guards honored:
  - MLP not linear (bigcodec poisons a LINEAR CID22 0.65-0.76; MLPs absorb it — §2).
  - Target = ssim2, NOT score_zensim (that would distill profile A — §5).
  - CID22-49 is a PURE holdout: loaded only to REPORT, NEVER to select a seed (§0.3).
  - Training-side selection = geomean(min train-positive-val incl DIV, kadis_deep).
  - Collapse gate: any train-positive-val SROCC < 0.85 -> reject the seed.
  - Multi-seed; exports the SELECTED seed's weights+scaler npz for bake_mlp_negatives.py.

  usage: train_mlp_diverse.py [--hidden 64] [--kadis-weight 0.3] [--div-weight 1.0]
                              [--div-cap 120000] [--hq-band 85] [--hq-weight 0.3]
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
BIGCODEC = "/mnt/v/output/zensim-multicodec-probe/bigcodec_hqdedup_traindigits_2026-07-02.parquet"
CID22_VAL = ("/mnt/v/zen/zensim-training/2026-05-15-full-features/"
             "cid22_features_372col_2026-05-15.parquet")
N_FEAT = 372
# positive corpora: (name, path, target-col, target-scale, row-cap). ssim2_gpu is on
# the 0..100 scale already (scale=1). kadid/tid DELIBERATELY EXCLUDED — their canonical
# ssim2_gpu is the ref-vs-ref misjoin bug (ranks backwards; DATASET_HISTORY.md §3.1).
POS = [("safesyn", f"{TRAIN}/safesyn.parquet", "ssim2_gpu", 1.0, 90000),
       ("cid22_train", f"{TRAIN}/cid22_train.parquet", "ssim2_gpu", 1.0, None)]


_CACHE = {}  # (path, ycol, scale) -> full finite (X, y); read once, subsample per seed


def load(path, ycol, scale=1.0, npmax=None, seed=0):
    key = (path, ycol, scale)
    if key not in _CACHE:
        cols = [f.name for f in pq.read_schema(path)]
        pfx = "feat_" if "feat_0" in cols else "f"
        t = pq.read_table(path, columns=[ycol] + [f"{pfx}{i}" for i in range(N_FEAT)])
        X = np.stack([np.asarray(t[f"{pfx}{i}"], dtype=np.float32) for i in range(N_FEAT)], 1)
        y = np.asarray(t[ycol], dtype=np.float64) * scale
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        _CACHE[key] = (X[ok], y[ok])
    X, y = _CACHE[key]
    if npmax and len(y) > npmax:
        i = np.random.RandomState(seed).permutation(len(y))[:npmax]
        X, y = X[i], y[i]
    return X, y


def build(seed, div_cap, hq_band, hq_weight, div_weight, kadis_weight):
    """Per-corpus 75/25 train/val split; returns train arrays (X, y, w) + val dict."""
    rs = np.random.RandomState(1000 + seed)
    tr_X, tr_y, tr_w, val = [], [], [], {}
    for name, path, ycol, scale, cap in POS:
        X, y = load(path, ycol, scale, cap, seed)
        m = rs.rand(len(y)) < 0.75
        tr_X.append(X[m]); tr_y.append(y[m]); tr_w.append(np.ones(m.sum()))
        val[name] = (X[~m], y[~m])
    # DIVERSE (imazen-26): human_score=ssim2/100 -> ×100; HQ (>hq_band) down-weighted.
    Xd, yd = load(BIGCODEC, "human_score", 100.0, div_cap, seed)
    md = rs.rand(len(yd)) < 0.75
    wd = np.where(yd[md] > hq_band, hq_weight, 1.0) * div_weight
    tr_X.append(Xd[md]); tr_y.append(yd[md]); tr_w.append(wd)
    val["bigcodec"] = (Xd[~md], yd[~md])
    # NEGATIVES (KADIS): down-weighted flat.
    Xk, yk = load(KADIS, "score_ssim2_gpu", 1.0, 90000, seed)
    mk = rs.rand(len(yk)) < 0.75
    tr_X.append(Xk[mk]); tr_y.append(yk[mk]); tr_w.append(np.full(mk.sum(), kadis_weight))
    val["kadis"] = (Xk[~mk], yk[~mk])
    return (np.concatenate(tr_X), np.concatenate(tr_y), np.concatenate(tr_w)), val


def train_one(seed, a, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    (Xtr, ytr, wtr), val = build(seed, a.div_cap, a.hq_band, a.hq_weight,
                                 a.div_weight, a.kadis_weight)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    sc = lambda X: (X - mu) / sd
    sclip = np.clip(ytr, -150, 100); tm, ts = sclip.mean(), sclip.std()
    yn = (sclip - tm) / ts
    Xd = torch.tensor(sc(Xtr), dtype=torch.float32, device=dev)
    yd = torch.tensor(yn, dtype=torch.float32, device=dev).unsqueeze(1)
    wd = torch.tensor(wtr, dtype=torch.float32, device=dev).unsqueeze(1)
    net = nn.Sequential(nn.Linear(N_FEAT, a.hidden), nn.LeakyReLU(0.01),
                        nn.Linear(a.hidden, 1)).to(dev)
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
    W0 = net[0].weight.detach().cpu().numpy(); b0 = net[0].bias.detach().cpu().numpy()
    W1 = net[2].weight.detach().cpu().numpy(); b1 = net[2].bias.detach().cpu().numpy()
    return metrics, dict(mu=mu, sd=sd, W0=W0, b0=b0, W1=W1, b1=b1, hidden=a.hidden,
                         leaky=0.01, seed=seed)


def cid22_score(payload, Xcv, ycv):
    X = (Xcv - payload["mu"]) / payload["sd"]
    h = X @ payload["W0"].T + payload["b0"]
    h = np.where(h > 0, h, 0.01 * h)
    o = h @ payload["W1"].T + payload["b1"]
    return spearmanr(o.ravel(), ycv).correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--kadis-weight", type=float, default=0.3)
    ap.add_argument("--div-weight", type=float, default=1.0)
    ap.add_argument("--div-cap", type=int, default=120000)
    ap.add_argument("--hq-band", type=float, default=85.0)
    ap.add_argument("--hq-weight", type=float, default=0.3)
    ap.add_argument("--seeds", default="1,3,7,13,17,23,31,41")
    ap.add_argument("--out", default="/mnt/v/output/zensim/reports/b_negatives/mlp_diverse_best.npz")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xcv, ycv = load(CID22_VAL, "human_score")  # SACRED holdout — report only, never select
    tv_names = [n for n, *_ in POS] + ["bigcodec"]  # train-positive-val corpora
    print(f"cfg: hidden={a.hidden} div_cap={a.div_cap} div_w={a.div_weight} "
          f"hq(>{a.hq_band:.0f})_w={a.hq_weight} kadis_w={a.kadis_weight}")
    print("seed  " + " ".join(f"{n[:8]:>9s}" for n in tv_names)
          + "  kadis_deep kadis_all | CID22(holdout)  sel")
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        m, payload = train_one(s, a, dev)
        collapse = min(m[n] for n in tv_names) < 0.85
        sel = float(np.exp(np.mean(np.log([max(1e-6, min(m[n] for n in tv_names)),
                                           max(1e-6, m["kadis_deep"])]))))
        cid = cid22_score(payload, Xcv, ycv)
        tag = "COLLAPSE" if collapse else f"{sel:.3f}"
        print(f" {s:<4d} " + " ".join(f"{m[n]:+9.3f}" for n in tv_names)
              + f"  {m['kadis_deep']:+.4f}  {m['kadis_all']:+.3f}  |  {cid:+.4f}      {tag}")
        if not collapse:
            rows.append((sel, cid, m, payload))
    if not rows:
        print("ALL SEEDS COLLAPSED — aborting"); return
    rows.sort(key=lambda r: -r[0])
    sel, cid, m, payload = rows[0]
    np.savez(a.out, **payload)
    print(f"\nSELECTED seed {payload['seed']} (training-side sel {sel:.3f}): "
          f"CID22 {cid:.4f}, bigcodec {m['bigcodec']:.3f}, "
          f"kadis_deep {m['kadis_deep']:.3f}, safesyn {m['safesyn']:.3f}")
    print(f"exported -> {a.out}")
    print("baselines: shipped-B CID22 0.876 / deep 0.047 | §8.33 MLP 0.870 / 0.784 | ssim2 0.8895")


if __name__ == "__main__":
    main()
