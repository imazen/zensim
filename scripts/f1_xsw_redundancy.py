#!/usr/bin/env python3
"""F1 kill-test for the append3 XSW family (A6 design §8, F1).

Methodology = the v1-IW death study
(`benchmarks/iw_pool_underuse_investigation_2026-05-25.md`): standardized
columns, OLS, R² of each NEW feature explained by the predictor block.
Kill criterion (pre-registered in `docs/CROSS_SCALE_A6_DESIGN_2026-07-28.md`
and the coordinator brief): median R² >= 0.99 across the 9 live XSW lanes
==> the parent-scale weight field is in the same-scale span (the v1-IW
signature, p50=0.998) and the family is stillborn.

Two predictor sets per lane's scale s (Y channel):
  P21 = v1-basic-Y(13) + v2-Y masked/iw pools (8)   — the design-F1 set
  P42 = v1-basic-Y(13) + ALL 29 v2-Y locals          — harder, conservative
Permutation control: same fit on row-shuffled targets (10 seeds) to show
the small-n OLS inflation floor for each predictor-set size.

Usage: f1_xsw_redundancy.py <foldapp3_964.csv>
CSV = v2_ab_extract output: ref_basename,human_score,f0..f963.
"""

import sys

import numpy as np

V1_CH_STRIDE, V1_SC_STRIDE = 13, 39  # v1 basic: scale*39 + ch*13
V2_BASE, V2_LOCALS = 372, 29  # v2: 372 + scale*87 + ch*29 + local
APP3_BASE, APP3_PER_SCALE = 944, 5
MASKED_IW = list(range(12, 20))  # masked_ssim..iw_mse
XSW_LOCALS = {0: "XSW_SSIM", 1: "XSW_MSE", 2: "XSW_MSE_FLAT"}


def v1_basic_y(scale):
    b = scale * V1_SC_STRIDE + 1 * V1_CH_STRIDE
    return list(range(b, b + 13))


def v2_y(scale, locals_):
    b = V2_BASE + scale * 3 * V2_LOCALS + 1 * V2_LOCALS
    return [b + l for l in locals_]


def r2(x, y):
    """OLS R² of y on standardized x (+intercept), dropping dead columns."""
    keep = x.std(axis=0) > 1e-12
    x = x[:, keep]
    if x.shape[1] == 0 or y.std() < 1e-12:
        return float("nan")
    xs = (x - x.mean(axis=0)) / x.std(axis=0)
    xs = np.hstack([xs, np.ones((len(y), 1))])
    beta, *_ = np.linalg.lstsq(xs, y, rcond=None)
    resid = y - xs @ beta
    return 1.0 - resid.var() / y.var()


def main():
    data = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=1)
    feats = data[:, 2:]  # f0..f963
    n = len(feats)
    assert feats.shape[1] == 964, f"expected 964 columns, got {feats.shape[1]}"
    rng = np.random.default_rng(42)

    rows, all_r2 = [], {"P21": [], "P42": []}
    perm_floor = {"P21": [], "P42": []}
    for scale in range(3):
        p21 = v1_basic_y(scale) + v2_y(scale, MASKED_IW)
        p42 = v1_basic_y(scale) + v2_y(scale, range(V2_LOCALS))
        for local, name in XSW_LOCALS.items():
            tcol = APP3_BASE + scale * APP3_PER_SCALE + local
            y = feats[:, tcol]
            for tag, cols in (("P21", p21), ("P42", p42)):
                x = feats[:, cols]
                v = r2(x, y.copy())
                all_r2[tag].append(v)
                perms = [
                    r2(x, rng.permutation(y)) for _ in range(10)
                ]
                perm_floor[tag].append(np.mean(perms))
            rows.append(
                (scale, name, all_r2["P21"][-1], all_r2["P42"][-1],
                 perm_floor["P21"][-1], perm_floor["P42"][-1])
            )

    print(f"n = {n} pairs (aic3 ab TSV), 9 live XSW lanes")
    print("scale lane          R2(P21)  R2(P42)  permfloor(P21) permfloor(P42)")
    for s, name, a, b, pa, pb in rows:
        print(f"s{s}    {name:13s} {a:.5f}  {b:.5f}  {pa:.4f}         {pb:.4f}")
    for tag in ("P21", "P42"):
        v = np.array(all_r2[tag])
        print(
            f"\n[{tag}] median {np.median(v):.5f}  p25 {np.percentile(v, 25):.5f}"
            f"  p75 {np.percentile(v, 75):.5f}  min {v.min():.5f}  max {v.max():.5f}"
        )
        for thr in (0.99, 0.95, 0.90):
            print(f"[{tag}] lanes with R2 >= {thr}: {(v >= thr).sum()} / {len(v)}")
    med = np.median(all_r2["P21"])
    verdict = "STILLBORN (kill)" if med >= 0.99 else "ALIVE (proceed)"
    print(f"\nF1 VERDICT (P21 median {med:.5f} vs 0.99): {verdict}")


if __name__ == "__main__":
    main()
