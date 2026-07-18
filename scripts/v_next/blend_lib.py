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

from pathlib import Path

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
    # HF/near-threshold: KonJND-1k densified (20,160 near-JND compression pairs). human_score =
    # active_mix_raw [-64,95] (per-pair). The one near-threshold human-anchored corpus the blend
    # omits — targets the G5 weakness. NOTE: overlaps val/konjnd -> makes KonJND a CHEAT corpus.
    "konjnd_dense": (f"{TRAIN}/konjnd-dense.parquet", "human_score",    1.0, "pos"),
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
    # FR-corpus expansion 2026-07-18 (held-out; human_score quality-oriented [0,1], sign +1).
    "csiq":     (f"{FULLFEAT}/csiq_features_372col_2026-07-18.parquet",   "human_score", +1, True),
    "live":     (f"{FULLFEAT}/live_features_372col_2026-07-18.parquet",   "human_score", +1, True),
    "pipal":    (f"{FULLFEAT}/pipal_features_372col_2026-07-18.parquet",  "human_score", +1, False),
}
# corpora where train==val image overlap inflates SROCC (memorization, not skill) -> guards only
INTEGRITY_GUARDS = {"kadid", "tid"}
# true held-outs used for selection (csiq/live/pipal are FR holdouts our bakes never train on)
HOLDOUTS = ["cid22", "aic3", "aic4", "konjnd", "nonphoto", "csiq", "live", "pipal"]

_CACHE: dict = {}


def _load(path, ycol, scale=1.0, batch_rows=50_000):
    """Load (X[N,372] float32, y[N] float64) from a feature parquet, cached.

    Reads in ROW-GROUP BATCHES rather than one `read_table`. bigcodec is 5.3 GB on disk; a
    single read materializes the full arrow table AND a stacked numpy copy (~2x peak) in one
    giant allocation, which fails with ENOMEM once the page cache is warm (measured
    2026-07-15: `[errno 12] Cannot allocate memory` at a 40G cap with only 9.5 GiB RSS —
    the box had 45 GiB in buff/cache and 4.7 GiB free, so the big contiguous alloc could not
    be served). Batching keeps peak to one batch + the accumulating result and is strictly
    better for every corpus.
    """
    key = (path, ycol, scale)
    if key not in _CACHE:
        names = [f.name for f in pq.read_schema(path)]
        pfx = "feat_" if "feat_0" in names else "f"
        cols = [ycol] + [f"{pfx}{i}" for i in range(N_FEAT)]
        fcols = [f"{pfx}{i}" for i in range(N_FEAT)]
        Xs, ys = [], []
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=batch_rows, columns=cols):
            Xb = np.stack([np.asarray(b[c], dtype=np.float32) for c in fcols], 1)
            yb = np.asarray(b[ycol], dtype=np.float64) * scale
            ok = np.isfinite(yb) & np.isfinite(Xb).all(1)   # filter per batch -> smaller keep
            Xs.append(Xb[ok]); ys.append(yb[ok])
        _CACHE[key] = (np.concatenate(Xs), np.concatenate(ys))
    return _CACHE[key]


# --- HF (near-lossless) corpus: the ONLY post-jxl-fix data below distance 0.03 -------------
# 200 refs x 6 distances {0.005..0.03}. Features live in features.parquet (feat_0..371,
# with-iw regime); the ssim2 target lives in the sibling pareto.tsv -> joined on the cell key.
# MEASURED (pointer doc jxl_nearlossless_corpus_2026-07-06): pooled SROCC vs -distance is only
# +0.204 because cross-image scale swamps it, but PER-REF it is +0.916 — the ladder moves ssim2
# ~0.92 pts within an image vs ~6 pts between images. So this corpus is usable ONLY as a
# WITHIN-REF RANK signal; an absolute/MSE target here fits between-image noise.
HF_DIR = "/mnt/v/output/zensim-jxl-nearlossless/refit"
_HF_CACHE = {}


def load_hf(split="train", eval_every=4):
    """(X, y_ssim2, ref_idx) for the near-lossless corpus. Ref-level split (leak-free): refs are
    sorted, every `eval_every`-th goes to 'val'. split in {'train','val','all'}."""
    import csv as _csv
    import json as _json
    if "all" not in _HF_CACHE:
        names = [f.name for f in pq.read_schema(f"{HF_DIR}/features.parquet")]
        pfx = "feat_" if "feat_0" in names else "f"
        t = pq.read_table(f"{HF_DIR}/features.parquet",
                          columns=["image_path", "codec", "q", "knob_tuple_json"]
                                  + [f"{pfx}{i}" for i in range(N_FEAT)])
        X = np.stack([np.asarray(t[f"{pfx}{i}"], dtype=np.float32) for i in range(N_FEAT)], 1)
        # NOTE: q is float in the parquet ("90.0") but a string in the TSV ("90") -> normalize to
        # float on BOTH sides or the join silently yields zero rows.
        key = list(zip(t["image_path"].to_pylist(), t["codec"].to_pylist(),
                       [float(v) for v in t["q"].to_pylist()], t["knob_tuple_json"].to_pylist()))
        # join the ssim2 target from pareto.tsv on the same cell key
        tgt = {}
        for r in _csv.DictReader(open(f"{HF_DIR}/pareto.tsv"), delimiter="\t"):
            tgt[(r["image_path"], r["codec"], float(r["q"]), r["knob_tuple_json"])] = float(r["score_ssim2"])
        y = np.array([tgt.get(k, np.nan) for k in key], float)
        refs = np.array([k[0] for k in key])
        uref = sorted(set(refs.tolist()))
        rid = np.array([uref.index(r) for r in refs])
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        _HF_CACHE["all"] = (X[ok], y[ok], rid[ok], len(uref))
    X, y, rid, nref = _HF_CACHE["all"]
    if split == "all":
        return X, y, rid
    is_val = (rid % eval_every) == 0
    m = is_val if split == "val" else ~is_val
    return X[m], y[m], rid[m]


def hf_pairs(rid):
    """All within-ref index pairs (i, j) — the only comparisons this corpus supports."""
    pi, pj = [], []
    for r in np.unique(rid):
        idx = np.where(rid == r)[0]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pi.append(idx[a]); pj.append(idx[b])
    return np.array(pi), np.array(pj)


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

    # HF (near-lossless) group — RANK-ONLY, within-ref pairwise (RankNet). Deliberately NOT an
    # MSE term: its ssim2 ladder moves only ~0.92 pts within an image vs ~6 pts of cross-image
    # spread, so an absolute target would fit between-image noise (see load_hf docstring). Only
    # the training refs are used; the held-out refs are the honest HF readout.
    hfw = float(hp.get("hf_rank_weight", 0.0))
    hf = None
    if hfw > 0:
        Xh, yh, rh = load_hf("train", eval_every=int(hp.get("hf_eval_every", 4)))
        pi, pj = hf_pairs(rh)
        Xhd = torch.tensor(sc(Xh), dtype=torch.float32, device=dev)
        # target: 1 if cell i is the better-quality (higher ssim2) member of the pair
        tgt = torch.tensor((yh[pi] > yh[pj]).astype(np.float32), device=dev)
        hf = (Xhd, torch.tensor(pi, device=dev), torch.tensor(pj, device=dev), tgt)

    for _ in range(hp.get("epochs", 400)):
        opt.zero_grad()
        loss = (nn.functional.smooth_l1_loss(net(Xd), yd, reduction="none") * wd).mean()
        if hf is not None:
            Xhd, pi_t, pj_t, tgt = hf
            sh = net(Xhd).squeeze(1)
            # RankNet: P(i beats j) = sigmoid(s_i - s_j); scale-free, so it cannot drag the
            # absolute dial of the MSE groups around.
            loss = loss + hfw * nn.functional.binary_cross_entropy_with_logits(
                sh[pi_t] - sh[pj_t], tgt)
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
    # Range-restriction-robust EXTREME-QUALITY rank skill (bottom/top 30% by human).
    # Width-10 bands crush within-band target variance to ~1.5% of full range at the
    # extremes, so within-band SROCC there is pure noise (routinely negative at small n)
    # even for a near-perfect model — a reporting artifact, not a model defect. A 30%
    # tail keeps a usable target spread, so its SROCC is the honest low-/high-q rank number.
    if len(human) >= 10 and np.std(pred) > 0:
        order = np.argsort(human)
        k = max(3, int(round(0.30 * len(human))))
        lo_idx, hi_idx = order[:k], order[-k:]
        out["srocc_lowtail"] = _srocc(pred[lo_idx], human[lo_idx]); out["n_lowtail"] = int(k)
        out["srocc_hightail"] = _srocc(pred[hi_idx], human[hi_idx]); out["n_hightail"] = int(k)
    else:
        out["srocc_lowtail"] = out["srocc_hightail"] = np.nan
        out["n_lowtail"] = out["n_hightail"] = 0
    if per_band:
        h = human
        blo, bhi = (band_scale if band_scale else (np.nanmin(h), np.nanmax(h)))
        h100 = (h - blo) / ((bhi - blo) or 1) * 100.0
        full_std = np.std(h) or 1.0
        bands = {}
        for i, (a, b) in enumerate(BANDS):
            sel = (h100 >= a) & (h100 < b) if i < 9 else (h100 >= a) & (h100 <= b + 1e-9)
            n = int(sel.sum())
            hs = float(np.std(h[sel])) if n > 1 else 0.0
            restrict = hs / full_std  # within-band target spread as fraction of full (small => range-restricted)
            ps, hsub = pred[sel], human[sel]
            bsr = _srocc(ps, hsub) if n >= 3 else np.nan
            bpl = pearsonr(ps, hsub)[0] if n >= 3 and np.std(ps) > 0 and np.std(hsub) > 0 else np.nan
            # MAE after affine-rescale of pred onto human (calibration error, meaningful within a narrow band)
            if n >= 3 and np.std(ps) > 0:
                A = np.polyfit(ps, hsub, 1); bmae = float(np.mean(np.abs(np.polyval(A, ps) - hsub)))
            else:
                bmae = np.nan
            # Within-band SROCC is untrustworthy when n is tiny (CI > ±0.3, sign is noise —
            # CLAUDE.md rule) — that is B0's negative. Separately, `restrict` records how
            # badly the width-10 band compresses target spread (small => low SROCC ceiling
            # regardless of model); the dashboard annotates it. The honest extreme-quality
            # rank is srocc_lowtail / srocc_hightail above, which are not range-restricted.
            lowconf = (n < 30)
            bands[f"B{i}"] = {"n": n, "srocc": bsr, "plcc": bpl, "mae": bmae,
                              "restrict": restrict, "lowconf": bool(lowconf)}
        out["bands"] = bands
    return out


def hf_eval(payload, split="val", eval_every=4):
    """HONEST near-lossless readout: mean/median PER-REF SROCC of the bake vs the ssim2 ladder on
    HELD-OUT refs (never trained on when hf_rank_weight>0). Per-ref is mandatory here — pooled
    conflates cross-image scale with the within-ladder question (pooled ssim2 is only +0.204 vs
    +0.916 per-ref). Reference points measured 2026-07-15 on all 200 refs:
    ssim2 per-ref +0.916 (median 0.943), zensim per-ref +0.966 (median 1.000), 0% negative."""
    X, y, rid = load_hf(split, eval_every=eval_every)
    pred = forward(payload, X)
    per = []
    for r in np.unique(rid):
        m = rid == r
        if m.sum() >= 3 and np.std(pred[m]) > 0 and np.std(y[m]) > 0:
            c = _srocc(pred[m], y[m])
            if np.isfinite(c):
                per.append(c)
    per = np.array(per) if per else np.array([np.nan])
    return {"n_refs": int(len(per)), "per_ref_mean": float(np.nanmean(per)),
            "per_ref_median": float(np.nanmedian(per)),
            "frac_perfect": float(np.mean(per > 0.99)), "frac_negative": float(np.mean(per < 0))}


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
