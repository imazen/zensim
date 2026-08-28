#!/usr/bin/env python3
"""G-OUT variant study (user request: try OR / Z-RMSE / p99 / p1 forms).

Variants, per (candidate x axis):
  V1  or       — the panel's own outlier ratio (READ from rank blocks; ITU-T
                 P.1401 semantics; owner stat, no new math)
  V2  z_rmse   — the panel's uncertainty-weighted RMSE (READ from rank blocks)
  V3a p99_absz — 99th pct of |chart-z| (OLS+MAD residual; n-stable tail)
  V3b p1_z/p99_z — signed tails: p1 = worst under-prediction tail,
                 p99 = worst over-prediction tail (asymmetry finds the
                 corruption-scored-high class specifically)
  V4  max_absz + n_extreme — the first-pass form (n-sensitive; kept for
                 comparison) + BOUNDED = predictions within declared range.
Sense-making test: a variant "makes sense" if it (a) separates the models
from the known-bad peers, (b) is stable in n, (c) flags the named outlier
classes (t2 unboundedness, corruption-high-scores), (d) is owner-stat where
possible."""
import json, sys
import numpy as np

def tails(pred, mos):
    x = np.asarray(mos, float); y = np.asarray(pred, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    n = len(x)
    if n < 100: return None
    A = np.vstack([x, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    mad = np.median(np.abs(res - np.median(res))) * 1.4826 or 1e-9
    z = res / mad
    return {"n": n, "p99_absz": float(np.percentile(np.abs(z), 99)),
            "p1_z": float(np.percentile(z, 1)), "p99_z": float(np.percentile(z, 99)),
            "max_absz": float(np.max(np.abs(z))),
            "pred_min": float(y.min()), "pred_max": float(y.max())}

def main():
    rows = []
    for path in sys.argv[1:]:
        o = json.load(open(path))
        name = o.get("name", path)
        rank = o.get("rank") or {}
        pp = o.get("per_pair") or {}
        for ax in sorted(set(rank) | set(pp)):
            rb = rank.get(ax) or {}
            blk = pp.get(ax) or {}
            tcol = next((k for k in ("mos", "jnd", "pjnd", "target") if k in blk), None)
            t = tails(blk.get("pred", []), blk.get(tcol, [])) if tcol else None
            rows.append({"bake": name, "axis": ax, "or": rb.get("or"), "z_rmse": rb.get("z_rmse"), **(t or {})})
    axes = ["cid22", "imazen26", "nonphoto", "hfnlproxy", "kadid", "live"]
    hdr = f"{'bake':<26}{'axis':<11}{'OR':>7}{'ZRMSE':>7}{'p99|z|':>8}{'p1_z':>8}{'p99_z':>8}{'max|z|':>8}{'pred_rng':>18}"
    print(hdr)
    for ax in axes:
        for r in rows:
            if r["axis"] != ax or r.get("n") is None: continue
            fmt = lambda v, w: (f"{v:>{w}.3f}" if isinstance(v, (int, float)) and v is not None else " " * (w - 1) + "—")
            rng = f"[{r['pred_min']:>6.0f},{r['pred_max']:>5.0f}]"
            print(f"{r['bake'][:25]:<26}{ax:<11}{fmt(r.get('or'),7)}{fmt(r.get('z_rmse'),7)}{fmt(r['p99_absz'],8)}{fmt(r['p1_z'],8)}{fmt(r['p99_z'],8)}{fmt(r['max_absz'],8)}{rng:>18}")
        print()

if __name__ == "__main__":
    main()
