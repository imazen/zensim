#!/usr/bin/env python3
"""§8.26/§8.28 over-smoothing blind-spot probe on the real HDR JXL codec ladder.

Hypothesis (from the KADIS analytic-denoise finding, §8.26): BHdr under-penalizes
high-frequency loss, i.e. it might reward over-quantized / over-smoothed codec
output. Test it on REAL JXL HDR encodes (no new human labels) using cvvdp as a
detail-aware reference and the ssim2-derived target as the smoothing-TOLERANT
baseline. If BHdr behaved like ssim2, it would side with ssim2 on the cells where
ssim2 forgives smoothing but cvvdp penalizes it.

Inputs: rescore the hdr_zenjxl_v3 {train,val}digits parquets with
`rescore_parquet --profile bhdr --score-col score_bhdr --feat-prefix f` first
(they already carry score_cvvdp for a subset + human_score=ssim2 + zensim_score=A).

Result (2026-07-14, both splits): BHdr tracks cvvdp at 0.96–0.97 — ABOVE ssim2
(0.93–0.94) — so the blind spot does NOT severely manifest on real codec output.
It inherits ~1/3 of ssim2's residual leniency (cvvdp-mix target bought back most
detail-awareness) and trails full-multiscale A by ~0.10 only in the aggressive-
compression band = a linear-head capacity limit, not a target failure.
"""
import sys
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr, rankdata

ROOT = "/mnt/v/output/zensim/reports/oversmooth_probe"


def load(split):
    t = pq.read_table(f"{ROOT}/hdr_jxl_{split}_bhdr.parquet",
                      columns=["ref_basename", "score_cvvdp", "zensim_score",
                               "score_bhdr", "human_score"]).to_pydict()
    cv = np.array(t["score_cvvdp"], float)   # detail-aware HDR reference
    bh = np.array(t["score_bhdr"], float)    # BHdr (cvvdp-mix target)
    s2 = np.array(t["human_score"], float)   # ssim2-derived (smoothing-tolerant)
    a = np.array(t["zensim_score"], float)   # old A (full multi-scale)
    m = np.isfinite(cv) & np.isfinite(bh) & np.isfinite(s2) & np.isfinite(a)
    return cv[m], bh[m], s2[m], a[m]


def sr(x, y):
    return spearmanr(x, y).correlation


def report(split):
    cv, bh, s2, a = load(split)
    print(f"\n=== {split}: {len(cv)} JXL HDR cells with cvvdp ===")
    print(f"  overall vs cvvdp:  BHdr {sr(bh, cv):+.4f}  ssim2 {sr(s2, cv):+.4f}  A {sr(a, cv):+.4f}")
    q = np.quantile(cv, [0, .25, .5, .75, 1.0])
    lab = ["Q1 most-compressed", "Q2", "Q3", "Q4 near-lossless"]
    for i in range(4):
        b = (cv >= q[i]) & (cv <= q[i + 1]) if i == 3 else (cv >= q[i]) & (cv < q[i + 1])
        print(f"  {lab[i]:20s} n={b.sum():4d}  BHdr {sr(bh[b], cv[b]):+.3f}  "
              f"ssim2 {sr(s2[b], cv[b]):+.3f}  A {sr(a[b], cv[b]):+.3f}")
    # disagreement test: where ssim2 forgives smoothing vs cvvdp, does BHdr follow?
    n = len(cv)
    lenient = (rankdata(s2) - rankdata(cv)) / n
    bhdr_side = (rankdata(bh) - rankdata(cv)) / n
    sel = lenient >= np.quantile(lenient, 0.85)
    print(f"  ssim2-forgiven top15% (n={sel.sum()}): ssim2 leniency {lenient[sel].mean():+.3f}, "
          f"BHdr {bhdr_side[sel].mean():+.3f} ({(bhdr_side[sel] > 0).mean() * 100:.0f}% same dir); "
          f"corr(all) {np.corrcoef(lenient, bhdr_side)[0, 1]:+.3f}")


if __name__ == "__main__":
    for split in (sys.argv[1:] or ["val", "train"]):
        report(split)
