#!/usr/bin/env python3
"""Per-codec unreachable zones for B, from current-B re-forwarded ladders.
Ladder = one (image, encoder-mode) q-sweep. reach(t) = fraction of ladders whose
[min_q B, max_q B] spans dial target t. Zone boundaries by reach thresholds:
  reliable  (>=90% of ladders can hit t), usually (>=50%), effectively-unreachable (<10%).
Split into bottom (floor: can't make quality bad enough) and top (ceiling: can't make
it good enough). Also A's ceiling for contrast (the B-vs-A ceiling gap)."""
import numpy as np, pyarrow.parquet as pq

DIR = "/mnt/v/output/zensim-multicodec-probe/knob_reforward"
CODECS = ["zenjpeg_lossy", "zenavif_lossy", "zenjxl_lossy", "zenwebp_lossy"]
T = np.arange(0, 100.01, 1.0)

def zones(col, df):
    mn, mx = [], []
    for _, g in df.groupby(["ref", "box", "cell"], sort=False):
        if len(g) < 3: continue
        v = g[col].to_numpy(); mn.append(v.min()); mx.append(v.max())
    mn, mx = np.array(mn), np.array(mx)
    reach = np.array([np.mean((mn <= t) & (t <= mx)) for t in T])
    def lo(thr): idx = np.where(reach >= thr)[0]; return T[idx[0]] if len(idx) else np.nan
    def hi(thr): idx = np.where(reach >= thr)[0]; return T[idx[-1]] if len(idx) else np.nan
    return dict(reach=reach, floor_med=np.median(mn), ceil_med=np.median(mx), ceil_max=mx.max(),
                rel=(lo(0.9), hi(0.9)), usu=(lo(0.5), hi(0.5)), core10=(lo(0.1), hi(0.1)))

hdr = f"{'codec':6} | {'reliable(>=90%)':>15} | {'usually(>=50%)':>15} | {'ceil med/top10/max':>19} | top-unreach(B<10%)"
print(hdr)
rowsB = {}
for codec in CODECS:
    df = pq.read_table(f"{DIR}/{codec}.parquet").to_pandas()
    zB = zones("b", df); zA = zones("a", df); rowsB[codec] = (zB, zA)
    c = codec.replace("zen","").replace("_lossy","")
    rel, usu, core = zB["rel"], zB["usu"], zB["core10"]
    relS = f"{rel[0]:.0f}..{rel[1]:.0f}"; usuS = f"{usu[0]:.0f}..{usu[1]:.0f}"
    ceilS = f"{zB['ceil_med']:.0f} / {core[1]:.0f} / {zB['ceil_max']:.0f}"
    print(f"{c:6} | {relS:>15} | {usuS:>15} | {ceilS:>19} | > {core[1]:.0f}")

print("\n=== per codec: reach(t) at key targets, B vs A (top-end ceiling contrast) ===")
keys = [50, 70, 80, 85, 90, 92, 95]
print(f"{'codec':6} {'metric':6} " + " ".join(f'{("@"+str(t)):>6}' for t in keys))
for codec in CODECS:
    zB, zA = rowsB[codec]; c = codec.replace("zen","").replace("_lossy","")
    for lab, z in [("B", zB), ("A", zA)]:
        vals = [z["reach"][int(t)] for t in keys]
        print(f"{c:6} {lab:6} " + " ".join(f'{v:6.2f}' for v in vals))

print("\n=== bottom-unreachable (floor: fraction of ladders that CANNOT go below t) ===")
bt = [10, 20, 30, 40]
print(f"{'codec':6} " + " ".join(f'{("<"+str(t)):>6}' for t in bt))
for codec in CODECS:
    c = codec.replace("zen","").replace("_lossy","")
    df = pq.read_table(f"{DIR}/{codec}.parquet").to_pandas()
    mn = np.array([g["b"].to_numpy().min() for _, g in df.groupby(["ref","box","cell"], sort=False) if len(g) >= 3])
    print(f"{c:6} " + " ".join(f'{np.mean(mn > t):6.2f}' for t in bt))
