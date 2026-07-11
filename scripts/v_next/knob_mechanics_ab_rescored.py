#!/usr/bin/env python3
"""Knob MECHANICS on real per-image q-ladders (ab_rescored, picker TEST corpus).
See benchmarks/b_knob_validation_real_encoders_2026-07-11.md.

A knob converges under binary search iff, within a fixed encoder mode, sweeping q
moves the metric monotonically with no flat dead-zones. Per (ref,box,cell) ladder,
sort by q and measure, for M in {B=pred_b, A=pred_a, ssim2}:
  - |Spearman(M,q)|           rank monotonicity of metric vs quality param
  - inversion rate            adjacent-q reversals / steps (the binary-search killer)
  - tie rate @0.5 dial pts    adjacent-q |dM|<0.5 -> search can't localize
  - span                      reachable dial range on the ladder
Reference-free: pure mechanics of the metric-vs-encoder-param relationship on real
bitstreams. Reads the ab_rescored parquets directly; no new encodes.

Usage: python3 knob_mechanics_ab_rescored.py [AB_DIR]
  AB_DIR default /mnt/v/output/zensim/ab_rescored_2026-07-05
"""
import json
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

AB_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/ab_rescored_2026-07-05"
EPS_TIE = 0.5   # dial points below which two encodes are indistinguishable to the search


def srocc(a, q):
    if len(a) < 3 or np.ptp(a) == 0:
        return np.nan
    def rk(x):
        o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
    ra, rq = rk(a), rk(q); n = len(a)
    return 1 - 6 * np.sum((ra - rq) ** 2) / (n * (n * n - 1))


rows = []
for codec in ["zenjpeg_lossy", "zenavif_lossy", "zenjxl_lossy", "zenwebp_lossy"]:
    base = f"{AB_DIR}/{codec}"
    tb = pq.read_table(f"{base}.b.parquet",
                       columns=["ref_filename", "box", "q", "pred_b", "score_ssim2", "knob_tuple_json"])
    ta = pq.read_table(f"{base}.a.parquet", columns=["pred_a", "score_ssim2"])
    assert np.allclose(tb.column("score_ssim2").to_numpy(), ta.column("score_ssim2").to_numpy(), atol=1e-6), codec
    df = tb.to_pandas()
    df["pred_a"] = ta.column("pred_a").to_numpy()
    df["cell"] = [json.loads(k)["cell"] for k in df["knob_tuple_json"]]
    stats = {m: {"rho": [], "inv": [], "tie": [], "span": [], "n_lad": 0} for m in ["B", "A", "ssim2"]}
    for _, g in df.groupby(["ref_filename", "box", "cell"], sort=False):
        if len(g) < 3:
            continue
        g = g.sort_values("q")
        q = g["q"].to_numpy()
        for m, col in [("B", "pred_b"), ("A", "pred_a"), ("ssim2", "score_ssim2")]:
            v = g[col].to_numpy()
            rho = srocc(v, q)
            sgn = 1.0 if (np.isnan(rho) or rho >= 0) else -1.0  # orient quality-increasing
            dv = np.diff(sgn * v)
            stats[m]["rho"].append(abs(rho) if not np.isnan(rho) else np.nan)
            stats[m]["inv"].append(np.mean(dv < -1e-6))
            stats[m]["tie"].append(np.mean(np.abs(dv) < EPS_TIE))
            stats[m]["span"].append(np.ptp(v))
            stats[m]["n_lad"] += 1
    for m in ["B", "A", "ssim2"]:
        s = stats[m]
        rows.append({
            "codec": codec.replace("zen", "").replace("_lossy", ""), "metric": m, "ladders": s["n_lad"],
            "|rho|": np.nanmean(s["rho"]), "strict_mono%": 100 * np.mean(np.array(s["inv"]) == 0),
            "inv_rate": np.nanmean(s["inv"]), "tie@0.5": np.nanmean(s["tie"]), "span": np.nanmean(s["span"]),
        })

R = pd.DataFrame(rows)
pd.set_option("display.width", 160, "display.float_format", lambda x: f"{x:.4f}")
print(R.to_string(index=False))
print("\n=== per-metric mean across codecs (weighted by ladders) ===")
for m in ["B", "A", "ssim2"]:
    sub = R[R.metric == m]; w = sub["ladders"]
    print(f"  {m:6} |rho|={np.average(sub['|rho|'], weights=w):.4f}  "
          f"strict_mono={np.average(sub['strict_mono%'], weights=w):.2f}%  "
          f"inv={np.average(sub['inv_rate'], weights=w):.4f}  "
          f"tie@0.5={np.average(sub['tie@0.5'], weights=w):.4f}  "
          f"span={np.average(sub['span'], weights=w):.2f}")
