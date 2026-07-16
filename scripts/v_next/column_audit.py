#!/usr/bin/env python3
"""Per-column poison audit (user: 'dig into every column and how to make it helpful rather than
poisoning'). For EVERY candidate target/feature-target column in the canonical corpora, diagnose
whether it HELPS, POISONS, or is NEUTRAL as a training signal, and say HOW to fix a poisoning one.

Two parts:
  A. Cheap distribution/coverage/agreement report (pandas), per column per corpus:
       - coverage % (non-null, finite)
       - distribution (min / median / max) and a TAIL-EXPANSION flag (log-expanded targets like
         cvvdp_log_norm over-weight the near-lossless sliver under MSE — the §8.36 confound)
       - rank-agreement with ssim2_gpu (SROCC) where both present (is it even a sane quality axis?)
  B. Train-toward-it probe (de-poisoned 372-64-1 leaky MLP, winsor-clipped, safesyn+cid22_train,
     CID22 a PURE holdout) for each learnable scalar target -> CID22 holdout SROCC + train-val.
     A column whose CID22-as-target SROCC ~ ssim2's 0.88 is a usable target; one that craters
     (<0.75) is poisoning AS A REGRESSION TARGET even if it's rank-fine (shape confound).

Verdict logic (per column):
  - HELPFUL  : coverage>50%, ssim2-agree>0.9 (or IS ssim2), CID22-as-target>0.80
  - POISON   : CID22-as-target<0.75 despite rank-agree>0.9  -> TARGET-SHAPE poison (fix: raw/rank
               loss, never MSE on the log-expanded form); OR ssim2-agree<0.5 -> WRONG-AXIS poison
  - NEUTRAL  : usable but no edge over ssim2 (|Δ|<0.02)
  - SPARSE   : coverage<50% on the training corpora (can't train on it broadly)

  usage: column_audit.py [--seeds 1,7] [--out benchmarks/column_audit_2026-07-15.md]
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

REPO = Path.home() / "work/zen/zensim"
_s = importlib.util.spec_from_file_location("T", REPO / "scripts/v_next/train_mlp_diverse.py")
T = importlib.util.module_from_spec(_s); _s.loader.exec_module(T)
N = 372

# Corpora to scan for coverage/shape (name -> path). Training-side + a couple val.
CORPORA = {
    "safesyn": f"{T.TRAIN}/safesyn.parquet",
    "cid22_train": f"{T.TRAIN}/cid22_train.parquet",
    "kadid": f"{T.TRAIN}/kadid.parquet",
    "tid": f"{T.TRAIN}/tid.parquet",
    "konjnd-dense": f"{T.TRAIN}/konjnd-dense.parquet",
    "bigcodec": T.BIGCODEC,
    "kadis": T.KADIS,
}
# Every non-feature target/label column we've ever considered training on.
TARGET_COLS = [
    "human_score", "ssim2_gpu", "ssim2_log_norm", "cvvdp_score", "cvvdp_log_norm",
    "iwssim", "iwssim_log_norm", "pjnd_target", "active_mix_norm", "active_mix_raw",
    "score_ssim2_gpu", "score_zensim",  # KADIS naming
    "mix_cv50_iw50", "mix_cv25_iw75", "mix_cv75_iw25", "mix_cv33_iw33_sm33",
]
# Scalar targets to actually TRAIN toward (present + varying on safesyn+cid22_train).
PROBE_TARGETS = ["ssim2_gpu", "ssim2_log_norm", "cvvdp_score", "cvvdp_log_norm",
                 "iwssim", "iwssim_log_norm", "mix_cv50_iw50"]

# ---- Part C: FEATURE-side "bad data" audit over EVERY loaded parquet (2026-07-16,
# user: 'sweep all input parquets for bad data that might be dragging things down').
FEATURE_COLS = [f"f{i}" for i in range(N)]
CAN21 = "/mnt/v/zen/zensim-training/canonical-2026-05-21"
CAN0715 = "/mnt/v/zen/zensim-training/canonical-2026-07-15"
MC = "/mnt/v/output/zensim-multicodec-probe"
DI = "/mnt/v/output/zensim/depth-iter"
# name -> (path, role). Every parquet a depth/psa recipe actually loads, train AND val.
ALL_PARQUETS = {
    "safesyn":            (f"{CAN21}/train/safesyn.parquet", "train"),
    "cid22_train":        (f"{CAN21}/train/cid22_train.parquet", "train"),
    "kadid":              (f"{CAN21}/train/kadid.parquet", "train"),
    "tid":                (f"{CAN21}/train/tid.parquet", "train"),
    "konjnd-dense-norm":  (f"{CAN21}/train/konjnd-dense-norm.parquet", "train"),
    "multiband_anchor":   (f"{CAN21}/train/multiband_anchor_dial100.parquet", "anchor"),
    "hf_nearlossless":    (f"{CAN0715}/train/hf_nearlossless_train.parquet", "train"),
    "bigcodec_train":     (f"{DI}/bigcodec_train_120k_stride.parquet", "train"),
    "kadis_train":        (f"{DI}/kadis_train_60k_stride.parquet", "train"),
    "bigcodec_val":       (f"{MC}/bigcodec_hqdedup_valdigits_2026-07-02.parquet", "val"),
    "val/cid22":          (f"{CAN21}/val/cid22.parquet", "holdout"),
    "val/kadid":          (f"{CAN21}/val/kadid.parquet", "holdout"),
    "val/tid":            (f"{CAN21}/val/tid.parquet", "holdout"),
    "val/konjnd":         (f"{CAN21}/val/konjnd.parquet", "holdout"),
    "val/aic3":           (f"{CAN21}/val/aic3.parquet", "holdout"),
    "val/aic4":           (f"{CAN21}/val/aic4.parquet", "holdout"),
    "konfig_triplet_stim": (f"{MC}/konfig_features_ladders_2026-07-02.csv", "triplet"),
}


def feature_health(path):
    """Scan f0..f371 for bad data: NaN/Inf rows, IW-explosion outliers, constants, scale drift."""
    if path.endswith(".csv"):
        import pandas as pd
        head = pd.read_csv(path, nrows=1)
        fcols = [c for c in FEATURE_COLS if c in head.columns]
        X = pd.read_csv(path, usecols=fcols).to_numpy(dtype=np.float64)
    else:
        sch = set(pq.read_schema(path).names)
        fcols = [c for c in FEATURE_COLS if c in sch]
        if not fcols:
            return dict(err=f"no feature cols ({len(sch)} cols total)")
        X = pq.read_table(path, columns=fcols).to_pandas().to_numpy(dtype=np.float64)
    n, nf = X.shape
    finite = np.isfinite(X)
    nan_rows = int((~finite.all(1)).sum())
    nan_feats = np.where(~finite.all(0))[0]
    Xf = np.where(finite, X, np.nan)
    absmax = np.nanmax(np.abs(Xf), axis=0)
    stds = np.nanstd(Xf, axis=0)
    const_feats = np.where((stds == 0) | ~np.isfinite(stds))[0]
    exploded = np.where(absmax > 1e4)[0]  # IW-explosion: unbounded HF-moment features
    wf = int(np.nanargmax(np.where(np.isfinite(absmax), absmax, -1)))
    return dict(n=n, nf=nf, nan_rows=nan_rows, nan_feats=nan_feats.tolist()[:8],
                n_nan_feats=len(nan_feats), n_const=len(const_feats),
                const_feats=const_feats.tolist()[:8], worst_feat=wf,
                worst_absmax=float(absmax[wf]) if np.isfinite(absmax[wf]) else float("nan"),
                n_exploded=len(exploded), exploded=exploded.tolist()[:12])


def leakage_check(path):
    """Is human_score byte-identical to a feature or a metric column (target-leak, the kadid/tid bug)?"""
    if path.endswith(".csv"):
        return []
    sch = list(pq.read_schema(path).names)
    if "human_score" not in sch:
        return None
    cand = [c for c in sch if c in ("ssim2_gpu", "iwssim", "cvvdp_score",
            "ssim2_log_norm", "iwssim_log_norm") or c.startswith("f")]
    t = pq.read_table(path, columns=["human_score"] + cand).to_pandas()
    hs = t["human_score"].to_numpy(np.float64)
    leaks = []
    for c in cand:
        v = t[c].to_numpy(np.float64)
        ok = np.isfinite(hs) & np.isfinite(v)
        if ok.sum() > 30 and np.allclose(hs[ok], v[ok], rtol=1e-6, atol=1e-6):
            leaks.append(c)
    return leaks


def col_stats(path, col):
    """coverage, (min,median,max), tail-expansion ratio, ssim2-agreement — cheap, projected read."""
    sch = set(pq.read_schema(path).names)
    if col not in sch:
        return None
    want = [col] + (["ssim2_gpu"] if "ssim2_gpu" in sch and col != "ssim2_gpu" else [])
    t = pq.read_table(path, columns=want).to_pandas()
    v = t[col].to_numpy(dtype=np.float64)
    fin = np.isfinite(v)
    cov = fin.mean()
    if fin.sum() < 10 or np.nanstd(v[fin]) == 0:
        return dict(cov=cov, lo=np.nan, med=np.nan, hi=np.nan, tail=np.nan, agree=np.nan, n=int(fin.sum()))
    vf = v[fin]
    lo, med, hi = np.percentile(vf, [0.5, 50, 99.5]).tolist()
    # tail-expansion: how much does the top 5% stretch vs a uniform-rank reference?
    # ratio of (p99.5-p95) span to (p50-p5) span in RANK-normalized value; >3 => log-expanded top.
    p5, p50, p95, p995 = np.percentile(vf, [5, 50, 95, 99.5])
    tail = ((p995 - p95) / max(p50 - p5, 1e-9))
    agree = np.nan
    if "ssim2_gpu" in t.columns:
        s = t["ssim2_gpu"].to_numpy(dtype=np.float64)
        ok = fin & np.isfinite(s)
        if ok.sum() > 30 and np.std(s[ok]) > 0:
            agree = abs(spearmanr(vf[ok[fin]] if False else v[ok], s[ok]).correlation)
    return dict(cov=cov, lo=lo, med=med, hi=hi, tail=tail, agree=agree, n=int(fin.sum()))


def probe_cid22(target, seeds, dev):
    """de-poisoned MLP trained toward `target` -> CID22 holdout SROCC (reuses the §8.36 pipeline)."""
    import torch, torch.nn as nn
    Xcv, ycv = T.load(T.CID22_VAL, "human_score")
    cids, svals = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        try:
            Xs, ys = T.load(f"{T.TRAIN}/safesyn.parquet", target, 1.0, 90000, seed)
            Xc, yc = T.load(f"{T.TRAIN}/cid22_train.parquet", target, 1.0, None, seed)
        except Exception as e:
            return None, None, f"absent/unloadable: {e}"
        X = np.concatenate([Xs, Xc]); y = np.concatenate([ys, yc])
        ok = np.isfinite(y) & np.isfinite(X).all(1); X, y = X[ok], y[ok]
        if len(y) < 1000 or np.std(y) == 0:
            return None, None, "constant/empty target"
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
        net = nn.Sequential(nn.Linear(N, 64), nn.LeakyReLU(0.01), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        for _ in range(400):
            opt.zero_grad(); nn.functional.smooth_l1_loss(net(Xd), yd).backward(); opt.step()
        pr = lambda A: net(torch.tensor(sc(A), dtype=torch.float32, device=dev)).detach().cpu().numpy().ravel()
        cids.append(spearmanr(pr(Xcv), ycv).correlation)
        svals.append(spearmanr(pr(X[~m]), y[~m]).correlation)
    return float(np.mean(cids)), float(np.mean(svals)), None


def verdict(agree, cid22, ssim2_cid):
    if cid22 is None:
        return "SPARSE/UNLOADABLE"
    if not np.isnan(agree) and agree < 0.5:
        return "POISON (wrong-axis: rank-disagrees ssim2)"
    if cid22 < 0.75:
        return "POISON (target-shape: rank-fine but MSE-craters)"
    if abs(cid22 - ssim2_cid) < 0.02:
        return "NEUTRAL (usable, no edge vs ssim2)"
    return "HELPFUL" if cid22 >= ssim2_cid - 0.02 else "WEAK (usable, below ssim2)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,7")
    ap.add_argument("--mode", default="features", choices=["features", "full"],
                    help="features=fast feature-health+leakage sweep of every parquet; "
                         "full=also the slow per-target CID22 train-probes (A+B).")
    ap.add_argument("--out", default=str(REPO / "benchmarks/column_audit_2026-07-15.md"))
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]

    # ---- Part C: feature-side bad-data sweep over EVERY loaded parquet (always) ----
    print("=== Part C: feature-health + leakage sweep (all loaded parquets) ===")
    LC = ["# Feature-health + leakage sweep — 2026-07-16",
          "",
          "User: *'sweep all input parquets for bad data that might be dragging things down'.* "
          "Scans f0..f371 in every parquet a depth/psa recipe loads. `nan_rows`=rows with any "
          "non-finite feature; `exploded`=features with |max|>1e4 (unbounded IW HF-moment — the "
          "5.8M-on-graphics bug); `const`=zero-variance features; `leak`=human_score byte-identical "
          "to a feature/metric column (the kadid/tid iwssim==human_score bug).",
          "",
          "| parquet | role | n | nan_rows | n_nan_feat | n_const | worst_feat |max| | exploded | leak |",
          "|---|---|--:|--:|--:|--:|---|--:|---|---|"]
    for name, (path, role) in ALL_PARQUETS.items():
        try:
            fh = feature_health(path)
        except Exception as e:
            LC.append(f"| {name} | {role} | — | — | — | — | ERR | — | {str(e)[:40]} | — |")
            print(f"  {name:20s} ERR {str(e)[:60]}")
            continue
        if "err" in fh:
            LC.append(f"| {name} | {role} | — | — | — | — | {fh['err']} | — | — | — |")
            print(f"  {name:20s} {fh['err']}")
            continue
        try:
            lk = leakage_check(path)
        except Exception:
            lk = None
        leakstr = "—" if lk is None else ("**" + ",".join(lk) + "**" if lk else "clean")
        expl = "—" if fh["n_exploded"] == 0 else f"**{fh['n_exploded']}: {fh['exploded']}**"
        LC.append(f"| {name} | {role} | {fh['n']} | {fh['nan_rows']} | {fh['n_nan_feats']} | "
                  f"{fh['n_const']} | f{fh['worst_feat']} | {fh['worst_absmax']:.3g} | {expl} | {leakstr} |")
        print(f"  {name:20s} n={fh['n']:>7} nan_rows={fh['nan_rows']:>6} const={fh['n_const']:>3} "
              f"exploded={fh['n_exploded']:>3} worst=f{fh['worst_feat']}={fh['worst_absmax']:.3g} leak={leakstr}")
    Path(str(REPO / "benchmarks/feature_health_sweep_2026-07-16.md")).write_text("\n".join(LC) + "\n")
    print(f"wrote {REPO / 'benchmarks/feature_health_sweep_2026-07-16.md'}")
    if a.mode == "features":
        return

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    L = ["# Per-column poison audit — 2026-07-15",
         "",
         "User: *'dig into every column and how to make it helpful rather than poisoning'.* "
         "De-poisoned 372-64-1 leaky MLP (winsor p0.1/p99.9), safesyn+cid22_train, CID22 a PURE "
         f"holdout, {len(seeds)} seeds. Verdicts: HELPFUL / NEUTRAL / WEAK / POISON / SPARSE.",
         ""]

    # ---- Part A: coverage / shape / ssim2-agreement per column per corpus
    L += ["## A. Coverage + distribution + ssim2-agreement (per training corpus)", "",
          "`tail` = (p99.5−p95)/(p50−p5); ≫3 ⇒ log-expanded top (MSE over-weights near-lossless "
          "sliver — the cvvdp_log_norm confound). `agree` = |SROCC vs ssim2_gpu|.", ""]
    for col in TARGET_COLS:
        rows = []
        for cname, cpath in CORPORA.items():
            try:
                st = col_stats(cpath, col)
            except Exception as e:
                st = None
            if st is None:
                continue
            rows.append((cname, st))
        if not rows:
            continue
        L.append(f"### `{col}`")
        L.append("")
        L.append("| corpus | cov% | n | min | median | max | tail | agree(ssim2) |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
        for cname, st in rows:
            fmt = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (
                f"{x:.3g}" if abs(x) < 1e4 else f"{x:.2e}")
            L.append(f"| {cname} | {st['cov']*100:.1f} | {st['n']} | {fmt(st['lo'])} | "
                     f"{fmt(st['med'])} | {fmt(st['hi'])} | {fmt(st['tail'])} | {fmt(st['agree'])} |")
        L.append("")

    # ---- Part B: train-toward-target -> CID22 holdout
    L += ["## B. Train-toward-target → CID22 holdout SROCC (is it a usable TARGET?)", "",
          "| target | CID22(holdout) | train-val | verdict |", "|---|--:|--:|---|"]
    print("probing targets -> CID22 holdout ...")
    # ssim2 baseline first
    base_cid, base_sv, base_err = probe_cid22("ssim2_gpu", seeds, dev)
    ssim2_cid = base_cid if base_cid is not None else 0.88
    results = {"ssim2_gpu": (base_cid, base_sv, base_err)}
    for tgt in PROBE_TARGETS:
        if tgt == "ssim2_gpu":
            continue
        results[tgt] = probe_cid22(tgt, seeds, dev)
        c, s, e = results[tgt]
        print(f"  {tgt:18s} CID22={c if c is None else round(c,4)}  err={e}")
    for tgt in PROBE_TARGETS:
        c, s, e = results[tgt]
        # need agree for verdict — pull safesyn agree
        try:
            ag = col_stats(f"{T.TRAIN}/safesyn.parquet", tgt)
            agree = ag["agree"] if ag else np.nan
        except Exception:
            agree = np.nan
        vd = verdict(agree if not np.isnan(agree) else 1.0, c, ssim2_cid) if e is None else f"SPARSE ({e})"
        cc = "—" if c is None else f"{c:.4f}"
        ss = "—" if s is None else f"{s:.4f}"
        L.append(f"| `{tgt}` | {cc} | {ss} | {vd} |")
    L += ["", f"_ssim2_gpu baseline CID22 = {ssim2_cid:.4f}. A target that rank-agrees with ssim2 "
          "(agree≈1) yet craters CID22-as-target is a TARGET-SHAPE poison — fix with raw/rank loss, "
          "never MSE on the log-expanded form (§8.36)._"]

    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
