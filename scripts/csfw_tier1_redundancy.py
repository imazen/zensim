#!/usr/bin/env python3
"""Non-absorption kill-test for the chunk-3 CSFW tier-1 family (f944..f955).

Methodology = `scripts/f1_xsw_redundancy.py` (the v1-IW death study /
append3 F1 instrument): standardized columns, OLS, R² of each NEW lane
explained by the predictor block. Pre-registered criterion (coordinator
brief, 2026-07-28): median R² >= 0.99 across the 12 weighted lanes over
the P-design set ==> the weighted GLOBAL_* twins are affine-recoverable
from their unweighted twins + same-scale basics and the family is
stillborn; do not merge.

Predictor sets per lane's scale s (Y channel):
  P24 = v1-basic-Y(13) + v2-Y masked/iw pools (8)
        + the unweighted GLOBAL trio (Y, same scale) (3)   — design set
  P59 = v1-basic-Y(13) + ALL 29 v2-Y locals
        + ALL 17 Y append locals                            — conservative
Permutation control: same fit on row-shuffled targets (10 seeds).

R-CLASS CAVEAT (append3 escapee addendum): the GLOBAL_* family is a
rare-fire family on aic3 codec pairs — per-lane std is printed so a
near-constant lane's R² can be read as the noise ratio it is.

Usage: csfw_tier1_redundancy.py <foldcsfw_956.csv>
CSV = v2_ab_extract output (ZENSIM_AB_MODE=foldcsfw):
ref_basename,human_score,f0..f955.
"""

import sys

import numpy as np

V1_CH_STRIDE, V1_SC_STRIDE = 13, 39  # v1 basic: scale*39 + ch*13
V2_BASE, V2_LOCALS = 372, 29  # v2: 372 + scale*87 + ch*29 + local
APP_BASE, APP_LOCALS = 720, 17  # append: 720 + scale*51 + ch*17 + local
CSFW_BASE, CSFW_PER_SCALE = 944, 3
MASKED_IW = list(range(12, 20))  # masked_ssim..iw_mse
GLOBAL_TRIO = [13, 14, 15]  # GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
CSFW_LOCALS = {0: "W_GLOBAL_DMEAN", 1: "W_GLOBAL_CGAIN", 2: "W_GLOBAL_CLOSS"}


def v1_basic_y(scale):
    b = scale * V1_SC_STRIDE + 1 * V1_CH_STRIDE
    return list(range(b, b + 13))


def v2_y(scale, locals_):
    b = V2_BASE + scale * 3 * V2_LOCALS + 1 * V2_LOCALS
    return [b + l for l in locals_]


def app_y(scale, locals_):
    b = APP_BASE + scale * 3 * APP_LOCALS + 1 * APP_LOCALS
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
    feats = data[:, 2:]  # f0..f955
    n = len(feats)
    assert feats.shape[1] == 956, f"expected 956 columns, got {feats.shape[1]}"
    rng = np.random.default_rng(42)

    rows, all_r2 = [], {"P24": [], "P59": []}
    perm_floor = {"P24": [], "P59": []}
    for scale in range(4):
        p24 = v1_basic_y(scale) + v2_y(scale, MASKED_IW) + app_y(scale, GLOBAL_TRIO)
        p59 = (
            v1_basic_y(scale)
            + v2_y(scale, range(V2_LOCALS))
            + app_y(scale, range(APP_LOCALS))
        )
        for local, name in CSFW_LOCALS.items():
            tcol = CSFW_BASE + scale * CSFW_PER_SCALE + local
            y = feats[:, tcol]
            for tag, cols in (("P24", p24), ("P59", p59)):
                x = feats[:, cols]
                v = r2(x, y.copy())
                all_r2[tag].append(v)
                perms = [r2(x, rng.permutation(y)) for _ in range(10)]
                perm_floor[tag].append(np.mean(perms))
            rows.append(
                (
                    scale,
                    name,
                    all_r2["P24"][-1],
                    all_r2["P59"][-1],
                    perm_floor["P24"][-1],
                    perm_floor["P59"][-1],
                    y.std(),
                )
            )

    print(f"n = {n} pairs, 12 CSFW tier-1 lanes")
    print(
        "scale lane            R2(P24)  R2(P59)  permfloor(P24) permfloor(P59)  lane_std"
    )
    for s, name, a, b, pa, pb, sd in rows:
        print(
            f"s{s}    {name:15s} {a:.5f}  {b:.5f}  {pa:.4f}         {pb:.4f}"
            f"          {sd:.3e}"
        )
    for tag in ("P24", "P59"):
        v = np.array(all_r2[tag])
        v = v[~np.isnan(v)]
        print(
            f"\n[{tag}] median {np.median(v):.5f}  p25 {np.percentile(v, 25):.5f}"
            f"  p75 {np.percentile(v, 75):.5f}  min {v.min():.5f}  max {v.max():.5f}"
            f"  (n_live {len(v)})"
        )
        for thr in (0.99, 0.95, 0.90):
            print(f"[{tag}] lanes with R2 >= {thr}: {(v >= thr).sum()} / {len(v)}")
    med = np.median(np.array(all_r2["P24"])[~np.isnan(all_r2["P24"])])
    verdict = "STILLBORN (kill)" if med >= 0.99 else "ALIVE (proceed)"
    print(f"\nFALSIFIER-A VERDICT (P24 median {med:.5f} vs 0.99): {verdict}")


if __name__ == "__main__":
    main()
