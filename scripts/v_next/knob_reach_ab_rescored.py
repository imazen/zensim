#!/usr/bin/env python3
"""Normalized span + REACHABILITY / unreachable-region analysis on real q-ladders.
See benchmarks/b_knob_validation_real_encoders_2026-07-11.md.

Runs on the CURRENT-B/A re-forwarded compact parquets (reforward step): stored
ab_rescored pred_b is the pre-inclusive-winsor B. Two notions of "unreachable":
  * metric-limited  — the dial compresses a single image's quality range so targets
    inside the true range are still unreachable (a metric defect). Measured by
    per-ladder NORMALIZED span (span / metric's own p1..p99): if B's normalized span
    < ssim2's on the same ladders, B is a stiffer knob (more per-image unreachable).
  * encoder-limited — even at max/min q the encode can't reach a high/low target
    (real, unavoidable, all metrics share it). Measured by the global reachability
    curve reach_frac(t) = fraction of ladders whose [min_q,max_q] spans target t.

Usage: python3 knob_reach_ab_rescored.py [DIR]
  DIR default /mnt/v/output/zensim-multicodec-probe/knob_reforward
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim-multicodec-probe/knob_reforward"
CODECS = ["zenjpeg_lossy", "zenavif_lossy", "zenjxl_lossy", "zenwebp_lossy"]
METRICS = ["b", "a", "ssim2"]
LABEL = {"b": "B", "a": "A", "ssim2": "ssim2"}
TGRID = np.arange(0, 100.001, 2.5)
KEY_T = [30, 50, 70, 85, 90, 95]

span_rows, reach_rows, edge_rows = [], [], []
reach_curves = {}
for codec in CODECS:
    df = pq.read_table(f"{DIR}/{codec}.parquet").to_pandas()
    p1 = {m: np.percentile(df[m], 1) for m in METRICS}
    p99 = {m: np.percentile(df[m], 99) for m in METRICS}
    # per-ladder min/max/span per metric
    lad = {m: {"mn": [], "mx": [], "span": [], "spanfrac": []} for m in METRICS}
    for _, g in df.groupby(["ref", "box", "cell"], sort=False):
        if len(g) < 3:
            continue
        for m in METRICS:
            v = g[m].to_numpy(); mn, mx = v.min(), v.max()
            lad[m]["mn"].append(mn); lad[m]["mx"].append(mx)
            lad[m]["span"].append(mx - mn)
            lad[m]["spanfrac"].append((mx - mn) / max(p99[m] - p1[m], 1e-9))
    nlad = len(lad["b"]["mn"])
    for m in METRICS:
        mn = np.array(lad[m]["mn"]); mx = np.array(lad[m]["mx"])
        span_rows.append({"codec": codec.replace("zen", "").replace("_lossy", ""), "metric": LABEL[m],
                          "ladders": nlad, "span_native": np.mean(lad[m]["span"]),
                          "span_frac": np.mean(lad[m]["spanfrac"]),
                          "usable_p1..p99": f"{p1[m]:.1f}..{p99[m]:.1f}"})
        # reachability curve: fraction of ladders spanning target t
        rf = np.array([np.mean((mn <= t) & (t <= mx)) for t in TGRID])
        reach_curves[(codec, m)] = rf
        band = TGRID[rf >= 0.5]
        reach_rows.append({"codec": codec.replace("zen", "").replace("_lossy", ""), "metric": LABEL[m],
                           "reach>=50%_band": f"{band.min():.0f}..{band.max():.0f}" if len(band) else "none",
                           "band_width": (band.max() - band.min()) if len(band) else 0,
                           **{f"reach@{t}": np.mean((mn <= t) & (t <= mx)) for t in KEY_T}})
        # edges: typical reachable ceiling/floor
        edge_rows.append({"codec": codec.replace("zen", "").replace("_lossy", ""), "metric": LABEL[m],
                          "floor_p10": np.percentile(mn, 10), "floor_med": np.median(mn),
                          "ceil_med": np.median(mx), "ceil_p90": np.percentile(mx, 90),
                          "max_any": mx.max()})

pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.3f}")
S = pd.DataFrame(span_rows); R = pd.DataFrame(reach_rows); E = pd.DataFrame(edge_rows)
print("=== PART 1: span (native vs normalized to each metric's own p1..p99) ===")
print(S.to_string(index=False))
print("\n  weighted-mean normalized span (apples-to-apples cross-metric):")
for m in ["B", "A", "ssim2"]:
    sub = S[S.metric == m]
    print(f"    {m:6} span_frac={np.average(sub['span_frac'], weights=sub['ladders']):.4f}  "
          f"span_native={np.average(sub['span_native'], weights=sub['ladders']):.2f}")
print("\n=== PART 2: reachability — fraction of ladders whose [min_q,max_q] spans target t ===")
print(R.to_string(index=False))
print("\n=== PART 3: reachable ceiling/floor (per-ladder max/min dial percentiles) ===")
print(E.to_string(index=False))
