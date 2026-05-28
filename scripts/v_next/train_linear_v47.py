#!/usr/bin/env python3
"""v47-LINEAR: a 0-hidden-layer "weights table" zensim variant for robustness
(image-engine error / corruption / near-lossless), as requested 2026-05-27.

Design (first-pass MVP):
  - Pure linear: y = w · standardize(features) + b   (372 weights + 1 bias)
  - Masked-monotone-by-construction: w_i ≥ 0 on the 300 sign-safe distortion
    features (per benchmarks/feature_sign_mask_2026-05-26.tsv); 72 "free"
    features unconstrained — same mask v47-strict uses.
  - Trained on the SAME 5 canonical groups + targets as v47 (safesyn,
    cid22_train, kadid, tid, konjnd_dense), per-group min-max-normalized to
    [0,1] (SROCC is rank-invariant; this only matters for the LS fit scale).
  - Group weights from the v47 recipe (safesyn 1.0, cid22_train 1.5,
    kadid 0.5, tid 0.5, konjnd_dense 1.2).
  - Fit via scipy.optimize.lsq_linear with per-weight bounds.
  - FIRST PASS: RAW features (no Yeo-Johnson yet — that's an MVP shortcut).
    If linear-on-raw is in the ballpark, follow-up adds Yeo-Johnson +
    bake emission.

Evaluation:
  Load the 6 held-out validation parquets (CID22, KADID, TID, KonJND, AIC-3,
  AIC-4), predict, report SROCC + Z-RMSE (after 4-param logistic rescale, the
  Mohammadi 2025 panel convention) + side-by-side vs the shipped v47 numbers.
"""
from __future__ import annotations
import csv, os, sys, time
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import lsq_linear
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
VAL_DIR   = "/mnt/v/zen/zensim-training/canonical-2026-05-18/val"
MASK_TSV  = os.path.join(REPO, "benchmarks/feature_sign_mask_2026-05-26.tsv")

GROUPS = [
    # (name, parquet basename, train_w, target_column)
    # konjnd_dense INTENTIONALLY excluded from the linear variant: its pjnd_target
    # is on a different scale from MOS (corr ≈ −0.03 with human_score on this
    # corpus) and including it forces the single linear weight vector to
    # compromise → drags CID22/TID rank by −0.20. v47's nonlinear head bridges
    # MOS vs PJND; a linear model can't. KonJND + AIC-4 are accepted KNOWN
    # LIMITS for the linear variant — it's a codec/corruption-robustness tool,
    # not a PJND tool.
    ("safesyn",      "safesyn",          1.0, "human_score"),
    ("cid22_train",  "cid22_train_norm", 1.5, "human_score"),
    ("kadid",        "kadid",            0.5, "human_score"),
    ("tid",          "tid",              0.5, "human_score"),
]
HOLDOUTS = ["cid22", "kadid", "tid", "konjnd", "aic3", "aic4"]
# v47-strict-QAT held-out numbers, for side-by-side
V47_REF = {
    "cid22":  (0.8657, 0.512),
    "kadid":  (0.7933, 0.613),
    "tid":    (0.7927, 0.577),
    "konjnd": (0.4185, 0.932),
    "aic3":   (0.7680, 0.620),
    "aic4":   (0.8854, 0.481),
}

def load_group(basename: str, target_col: str = "human_score") -> tuple[np.ndarray, np.ndarray]:
    p = os.path.join(TRAIN_DIR, f"{basename}.parquet")
    t = pq.read_table(p, columns=[f"f{i}" for i in range(372)] + [target_col])
    X = np.column_stack([np.asarray(t[f"f{i}"], dtype=np.float64) for i in range(372)])
    y = np.asarray(t[target_col], dtype=np.float64)
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[keep], y[keep]

def load_mask() -> np.ndarray:
    """Returns a 372-bool array: True = pin_geq0 (w ≥ 0), False = free."""
    pin = np.zeros(372, dtype=bool)
    with open(MASK_TSV) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            idx = int(row["feat_idx"])
            if idx < 372:
                pin[idx] = (row["sign_mask"] == "pin_geq0")
    return pin

def minmax01(y: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(y, 0.001), np.quantile(y, 0.999)
    return np.clip((y - lo) / max(hi - lo, 1e-9), 0.0, 1.0)

def z_rmse_via_logistic(pred: np.ndarray, target: np.ndarray) -> float:
    """4-param logistic rescale → σ-normalized RMSE. σ = corpus-wide stdev of
    target (per-sample σ unavailable here)."""
    # 4-param logistic: y = b1 + (b2-b1) / (1 + exp(-(x-b3)/|b4|))
    def f(x, b1, b2, b3, b4): return b1 + (b2-b1)/(1.0+np.exp(-(x-b3)/(abs(b4)+1e-9)))
    p0 = [float(target.min()), float(target.max()), float(np.median(pred)), float(np.std(pred) or 1.0)]
    try:
        popt, _ = curve_fit(f, pred, target, p0=p0, maxfev=20000)
        resc = f(pred, *popt)
    except Exception:
        resc = pred
    sigma = float(np.std(target)) or 1.0
    return float(np.sqrt(np.mean(((resc - target)/sigma)**2)))


def main():
    t0 = time.time()
    print("== v47-LINEAR (0-hidden, masked-monotone, raw-features MVP) ==")
    pin = load_mask()
    n_pin = int(pin.sum()); n_free = 372 - n_pin
    print(f"sign mask: {n_pin} pin_geq0, {n_free} free")

    # 1) Load + concatenate training groups, with sample weights and per-group target normalization.
    Xs, ys, ws = [], [], []
    for name, base, w, tcol in GROUPS:
        X, y = load_group(base, tcol)
        y01 = minmax01(y)
        Xs.append(X); ys.append(y01); ws.append(np.full(len(y), w))
        print(f"  {name:14s} {len(y):>7d} rows  target={tcol:14s} [{y.min():+.3f},{y.max():+.3f}] → [0,1]  train_w={w}")
    X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    print(f"total: {len(y)} rows")

    # 2) Standardize features (z-score). Save mean+std for held-out.
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd

    # 3) Bounded LS with sample weights: prepend bias column; bounds: 0..inf on pin,
    #    -inf..inf on free + bias.
    A = np.hstack([Xs, np.ones((len(y), 1))])  # 372 features + 1 bias
    # sample weighting: pre-multiply rows by sqrt(w)
    sw = np.sqrt(w)
    A_w = A * sw[:, None]; y_w = y * sw

    lo = np.full(373, -np.inf); hi = np.full(373, np.inf)
    lo[:372] = np.where(pin, 0.0, -np.inf)
    print("fitting bounded LS...")
    res = lsq_linear(A_w, y_w, bounds=(lo, hi), method="bvls", verbose=0, max_iter=2000)
    coef = res.x[:372]; bias = res.x[372]
    n_nz = int(np.sum(np.abs(coef) > 1e-6))
    print(f"  cost={res.cost:.4f} n_iter={res.nit} active={n_nz}/372 (≈{n_nz/372*100:.0f}%)")

    # 4) Evaluate on the 6 held-out corpora.
    print()
    print(f"{'corpus':10s} {'n':>5s}  {'SROCC':>7s} {'Z-RMSE':>7s}  | v47:  {'SROCC':>7s} {'Z-RMSE':>7s}  | Δ-SROCC")
    print("-" * 88)
    for v in HOLDOUTS:
        p = os.path.join(VAL_DIR, f"{v}.parquet")
        t = pq.read_table(p, columns=[f"f{i}" for i in range(372)] + ["human_score"])
        Xv = np.column_stack([np.asarray(t[f"f{i}"], dtype=np.float64) for i in range(372)])
        yv = np.asarray(t["human_score"], dtype=np.float64)
        keep = np.isfinite(yv) & np.all(np.isfinite(Xv), axis=1)
        Xv, yv = Xv[keep], yv[keep]
        Xv_s = (Xv - mu) / sd
        pred = Xv_s @ coef + bias
        srocc = float(spearmanr(pred, yv).statistic)
        zrmse = z_rmse_via_logistic(pred, yv)
        ref_s, ref_z = V47_REF[v]
        d = srocc - ref_s
        print(f"{v:10s} {len(yv):5d}  {srocc:7.4f} {zrmse:7.3f}  | v47:  {ref_s:7.4f} {ref_z:7.3f}  | {d:+.4f}")
    # Persist the trained model (coef + bias + standardization stats) for the
    # next-step bake emission. JSON is portable + small.
    import json
    out = {
        "fit_seconds": time.time() - t0,
        "n_rows": int(len(y)),
        "n_features": 372,
        "n_active": int(n_nz),
        "scaler_mean": mu.tolist(),
        "scaler_scale": sd.tolist(),
        "weights": coef.tolist(),
        "bias": float(bias),
        "sign_mask_pin_geq0": pin.tolist(),
        "groups": [{"name": g[0], "rows": int(len(yi)), "train_w": g[2], "target": g[3]}
                   for g, yi in zip(GROUPS, ys)],
    }
    out_dir = "/mnt/v/output/zensim/linear_v47_mvp_2026-05-27"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump(out, f)
    print(f"saved → {out_dir}/weights.json  (t={time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
