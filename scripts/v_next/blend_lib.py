#!/usr/bin/env python3
"""Shared core for optimal-input-blend search + bandwise dashboard (user 2026-07-15:
"keep iterating to get the optimal input blend and best output").

Now that canonical kadid/tid ssim2_gpu is FIXED (§3.18 promoted 2026-07-15), the diverse
trainer's hard exclusion of kadid/tid (ref-vs-ref misjoin) is obsolete — they're usable
positives again. This module makes the WHOLE blend configurable (which corpora, weights,
targets, winsor, hq-downweight) and scores any resulting net on the FULL held-out Mohammadi
panel per corpus + 10-band, so a search can rank by real held-out performance (not train-val).

Exports:
  TRAIN_CORPORA / VAL_CORPORA registries
  load_train(name) / load_val(name)              cached (path, ycol) -> finite (X, y)
  train_blend(spec, hp, seed) -> payload         weighted smooth_l1 372->H->1 leaky MLP, winsor
  forward(payload, X) -> pred                     numpy inference (matches the baked runtime)
  panel(pred, human, per_band=True) -> dict       SROCC/PLCC/KROCC/ZRMSE + B0..B9
  composite(panel_by_corpus) -> (score, reject)   goal-aware ranking scalar
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import kendalltau, pearsonr, spearmanr

N_FEAT = 372
TRAIN = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
FULLFEAT = "/mnt/v/zen/zensim-training/2026-05-15-full-features"
KADIS = "/mnt/v/output/zensim/reports/b_negatives/kadis_sample_negrich.parquet"
BIGCODEC = "/mnt/v/output/zensim-multicodec-probe/bigcodec_hqdedup_traindigits_2026-07-02.parquet"

# name -> (path, target-col, target-scale, role). role: pos|div|neg. kadid/tid now FIXED (§3.18).
TRAIN_CORPORA = {
    "safesyn":     (f"{TRAIN}/safesyn.parquet",      "ssim2_gpu",       1.0, "pos"),
    "cid22_train": (f"{TRAIN}/cid22_train.parquet",  "ssim2_gpu",       1.0, "pos"),
    "kadid":       (f"{TRAIN}/kadid.parquet",        "ssim2_gpu",       1.0, "pos"),   # FIXED 2026-07-15
    "tid":         (f"{TRAIN}/tid.parquet",          "ssim2_gpu",       1.0, "pos"),   # FIXED 2026-07-15
    "bigcodec":    (BIGCODEC,                         "human_score",   100.0, "div"),
    "kadis":       (KADIS,                            "score_ssim2_gpu", 1.0, "neg"),
}

# held-out eval panel — the SAME parquets bake_verdict uses (so numbers match). `sign`:
# +1 rank-positive (higher pred = higher MOS), -1 structural-negative (report |SROCC|).
VAL_CORPORA = {
    "cid22":    (f"{FULLFEAT}/cid22_features_372col_2026-05-15.parquet",  "human_score", +1, True),
    "kadid":    (f"{FULLFEAT}/kadid_features_372col_2026-05-15.parquet",  "human_score", +1, True),
    "tid":      (f"{FULLFEAT}/tid_features_372col_2026-05-15.parquet",    "human_score", +1, True),
    "konjnd":   (f"{FULLFEAT}/konjnd_features_372col_2026-05-15.parquet", "human_score", -1, False),
    "aic3":     (f"{FULLFEAT}/aic3_features_372col_2026-05-15.parquet",   "human_score", -1, False),
    "aic4":     (f"{FULLFEAT}/aic4_features_372col_2026-05-20.parquet",   "human_score", -1, False),
    "nonphoto": (f"{FULLFEAT}/nonphoto_features_372col_2026-07-15.parquet", "human_score", +1, False),
}
# corpora where train==val image overlap inflates SROCC (memorization, not skill) -> guards only
INTEGRITY_GUARDS = {"kadid", "tid"}
# true held-outs used for selection
HOLDOUTS = ["cid22", "aic3", "aic4", "konjnd", "nonphoto"]

_CACHE: dict = {}


def _load(path, ycol, scale=1.0):
    key = (path, ycol, scale)
    if key not in _CACHE:
        names = [f.name for f in pq.read_schema(path)]
        pfx = "feat_" if "feat_0" in names else "f"
        t = pq.read_table(path, columns=[ycol] + [f"{pfx}{i}" for i in range(N_FEAT)])
        X = np.stack([np.asarray(t[f"{pfx}{i}"], dtype=np.float32) for i in range(N_FEAT)], 1)
        y = np.asarray(t[ycol], dtype=np.float64) * scale
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        _CACHE[key] = (X[ok], y[ok])
    return _CACHE[key]


def load_train(name):
    path, ycol, scale, _role = TRAIN_CORPORA[name]
    return _load(path, ycol, scale)


def load_val(name):
    path, ycol, _sign, _pb = VAL_CORPORA[name]
    return _load(path, ycol)


def train_blend(spec, hp, seed):
    """spec: {name: weight}. hp: hidden, epochs, winsor_pct, div_cap, hq_band, hq_weight,
    safesyn_cap. Returns payload dict (numpy weights + winsor + scaler)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rs = np.random.RandomState(1000 + seed)
    tr_X, tr_y, tr_w = [], [], []
    for name, w in spec.items():
        if w <= 0:
            continue
        path, ycol, scale, role = TRAIN_CORPORA[name]
        X, y = _load(path, ycol, scale)
        cap = hp.get("safesyn_cap", 90000) if name == "safesyn" else (
              hp.get("div_cap", 120000) if role == "div" else (
              90000 if name == "kadis" else None))
        if cap and len(y) > cap:
            idx = np.random.RandomState(seed).permutation(len(y))[:cap]
            X, y = X[idx], y[idx]
        m = rs.rand(len(y)) < 0.75
        Xt, yt = X[m], y[m]
        wt = np.full(len(yt), float(w))
        if role == "div":  # HQ-saturation down-weight (DATASET_HISTORY §0.1/§5)
            wt = wt * np.where(yt > hp.get("hq_band", 85.0), hp.get("hq_weight", 0.3), 1.0)
        tr_X.append(Xt); tr_y.append(yt); tr_w.append(wt)
    Xtr = np.concatenate(tr_X); ytr = np.concatenate(tr_y); wtr = np.concatenate(tr_w)

    wp = hp.get("winsor_pct", 0.1)
    if wp > 0:
        lo = np.percentile(Xtr, wp, axis=0); hi = np.percentile(Xtr, 100 - wp, axis=0)
    else:
        lo = np.full(N_FEAT, -np.inf); hi = np.full(N_FEAT, np.inf)
    Xtr = np.clip(Xtr, lo, hi)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    sc = lambda A: (np.clip(A, lo, hi) - mu) / sd
    sclip = np.clip(ytr, -150, 100); tm, ts = sclip.mean(), sclip.std() or 1.0
    yn = (sclip - tm) / ts
    H = hp.get("hidden", 64)
    layers = hp.get("layers", 1)
    Xd = torch.tensor(sc(Xtr), dtype=torch.float32, device=dev)
    yd = torch.tensor(yn, dtype=torch.float32, device=dev).unsqueeze(1)
    wd = torch.tensor(wtr, dtype=torch.float32, device=dev).unsqueeze(1)
    if layers == 2:
        net = nn.Sequential(nn.Linear(N_FEAT, H), nn.LeakyReLU(0.01),
                            nn.Linear(H, H), nn.LeakyReLU(0.01), nn.Linear(H, 1)).to(dev)
    else:
        net = nn.Sequential(nn.Linear(N_FEAT, H), nn.LeakyReLU(0.01), nn.Linear(H, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(hp.get("epochs", 400)):
        opt.zero_grad()
        loss = (nn.functional.smooth_l1_loss(net(Xd), yd, reduction="none") * wd).mean()
        loss.backward(); opt.step()
    out = dict(mu=mu, sd=sd, lo=lo, hi=hi, hidden=H, leaky=0.01, seed=seed, layers=layers,
               tm=tm, ts=ts, spec=spec, hp=hp)
    out["W0"] = net[0].weight.detach().cpu().numpy(); out["b0"] = net[0].bias.detach().cpu().numpy()
    if layers == 2:
        out["W1"] = net[2].weight.detach().cpu().numpy(); out["b1"] = net[2].bias.detach().cpu().numpy()
        out["W2"] = net[4].weight.detach().cpu().numpy(); out["b2"] = net[4].bias.detach().cpu().numpy()
    else:
        out["W1"] = net[2].weight.detach().cpu().numpy(); out["b1"] = net[2].bias.detach().cpu().numpy()
    return out


def save_payload(path, payload):
    """Save npz (arrays only) + a <path>.spec.json sidecar recording what this bake trained on,
    so a dashboard derives train/cheat provenance from the ACTUAL bake — it can't desync. The
    sidecar is the single source of truth for 'what did this model learn from'."""
    import json
    np.savez(path, **{k: v for k, v in payload.items() if k not in ("spec", "hp")})
    spec = payload.get("spec", {})
    hp = payload.get("hp", {})
    layers = int(payload.get("layers", hp.get("layers", 1)))
    H = int(payload.get("hidden", hp.get("hidden", 64)))
    meta = {
        "train_corpora": sorted([k for k, w in spec.items() if w > 0]),
        "weights": {k: float(w) for k, w in spec.items()},
        "target": "ssim2_gpu / ssim2-derived (NOT human MOS, NOT score_zensim)",
        "arch": f"372-{H}-{H}-1" if layers == 2 else f"372-{H}-1",
        "hp": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in hp.items()},
        "seed": int(payload.get("seed", -1)),
    }
    Path(str(path) + ".spec.json").write_text(json.dumps(meta, indent=2))


def forward(p, X):
    lo, hi = p.get("lo"), p.get("hi")
    Xc = np.clip(X, lo, hi) if lo is not None else X
    z = (Xc - p["mu"]) / p["sd"]
    lk = float(p.get("leaky", 0.01))
    h = z @ p["W0"].T + p["b0"]; h = np.where(h > 0, h, lk * h)
    if int(p.get("layers", 1)) == 2 and "W2" in p:
        h = h @ p["W1"].T + p["b1"]; h = np.where(h > 0, h, lk * h)
        return (h @ p["W2"].T + p["b2"]).ravel()
    return (h @ p["W1"].T + p["b1"]).ravel()


BANDS = [(i * 10, i * 10 + 10) for i in range(10)]  # B0..B9 on a 0..100 MOS scale


def _srocc(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return spearmanr(a, b).correlation


def panel(pred, human, sign=+1, per_band=True, band_scale=None):
    """Full stat panel. band_scale: (lo,hi) of human to map to 0..100 for banding; None=auto."""
    pred = np.asarray(pred, float); human = np.asarray(human, float)
    ok = np.isfinite(pred) & np.isfinite(human)
    pred, human = pred[ok], human[ok]
    out = {"n": int(len(human))}
    sr = _srocc(pred, human)
    out["srocc"] = sr
    out["srocc_abs"] = abs(sr) if np.isfinite(sr) else np.nan
    out["plcc"] = pearsonr(pred, human)[0] if len(pred) > 2 and np.std(pred) > 0 else np.nan
    out["krocc"] = kendalltau(pred, human).correlation if len(pred) > 2 else np.nan
    # Z-RMSE: affine-rescale pred to human, residual / corpus-sigma (per-sample sigma unavailable here)
    if len(pred) > 2 and np.std(pred) > 0:
        A = np.polyfit(pred, human, 1); fit = np.polyval(A, pred)
        out["zrmse"] = float(np.sqrt(np.mean(((fit - human) / (np.std(human) or 1)) ** 2)))
    else:
        out["zrmse"] = np.nan
    if per_band:
        h = human
        blo, bhi = (band_scale if band_scale else (np.nanmin(h), np.nanmax(h)))
        h100 = (h - blo) / ((bhi - blo) or 1) * 100.0
        bands = {}
        for i, (a, b) in enumerate(BANDS):
            sel = (h100 >= a) & (h100 < b) if i < 9 else (h100 >= a) & (h100 <= b + 1e-9)
            bands[f"B{i}"] = {"n": int(sel.sum()),
                              "srocc": _srocc(pred[sel], human[sel]) if sel.sum() >= 3 else np.nan}
        out["bands"] = bands
    return out


def score_all(payload):
    """Score a payload on every VAL corpus -> {corpus: panel}."""
    res = {}
    for name, (_p, _y, sign, want_band) in VAL_CORPORA.items():
        X, h = load_val(name)
        res[name] = panel(forward(payload, X), h, sign=sign, per_band=want_band)
        res[name]["sign"] = sign
    return res


def composite(res):
    """Goal-aware scalar for ranking + a reject flag (collapse). Rewards CID22 (primary) +
    the two standing weaknesses (non-photo content-blindness, KonJND/G5), guards the rest."""
    def sabs(n):
        v = res[n]["srocc_abs"] if VAL_CORPORA[n][2] < 0 else res[n]["srocc"]
        return v if np.isfinite(v) else 0.0
    cid = sabs("cid22"); npho = sabs("nonphoto"); kon = sabs("konjnd")
    a3 = sabs("aic3"); a4 = sabs("aic4"); kad = sabs("kadid"); ti = sabs("tid")
    reject = (cid < 0.84) or (npho < 0.80) or (kad < 0.45) or (ti < 0.45)
    score = cid + 0.30 * npho + 0.20 * kon + 0.10 * a3 + 0.05 * a4
    return score, reject


if __name__ == "__main__":
    # reusable single-spec trainer: blend_lib.py --spec safesyn:1,cid22_train:1 --out x.npz
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="corpus:weight,corpus:weight,...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hidden", type=int, default=64)
    a = ap.parse_args()
    spec = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in a.spec.split(",")}
    p = train_blend(spec, {"hidden": a.hidden, "epochs": 400, "winsor_pct": 0.1}, a.seed)
    save_payload(a.out, p)
    res = score_all(p)
    print(f"saved {a.out}  CID22 {res['cid22']['srocc']:.4f}  nonphoto {res['nonphoto']['srocc']:.4f}  "
          f"konjnd {abs(res['konjnd']['srocc']):.4f}  aic3 {abs(res['aic3']['srocc']):.4f}")
