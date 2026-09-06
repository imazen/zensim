#!/usr/bin/env python3
"""REV2b sibling audit: per-slot max/quantiles over every R6 rev1 (ssim2-arm) table.

Reads only STORED tables. No extraction. Emits a TSV keyed on slot id with the
registry name attached, so 'which slots are unbounded on real pixels' is a sort.
"""
import csv, json, os, sys
import numpy as np

R6 = "/mnt/v/output/zensim/rev2-2026-09-05/r6/tables/ssim2"
OUT = "/mnt/v/output/zensim/rev2-2026-09-05/r6b"
os.makedirs(OUT, exist_ok=True)

# --- layout, mirroring zensim::feature_defs::def_at at n_scales = 4 ---------
BASIC = ["ssim_mean","ssim_4th","ssim_2nd","edge_art_mean","edge_art_4th","edge_art_2nd",
         "edge_det_mean","edge_det_4th","edge_det_2nd","mse","var_loss","tex_loss","contrast_inc"]
PEAKS = ["ssim_max","edge_art_max","edge_det_max","ssim_l8","edge_art_l8","edge_det_l8"]
MASKED = ["ssim_mean","ssim_4th","ssim_2nd","edge_art_4th","edge_det_4th","mse"]
IW = MASKED
CH = ["X","Y","B"]

def layout(n_scales=4):
    names, fam, loc = [], [], []
    base = 0
    for f, sigs in (("basic",BASIC),("peaks",PEAKS),("masked",MASKED),("iw",IW)):
        per = len(sigs)
        for off in range(n_scales*3*per):
            cell = off//per
            names.append(f"{f}_{sigs[off%per]}_s{cell//3}_{CH[cell%3]}")
            fam.append(f); loc.append(off%per)
        base += n_scales*3*per
    return names, fam, loc

NAMES, FAM, LOC = layout()
assert len(NAMES) == 372, len(NAMES)

LEGS = ["cid22","kadid","tid","konjnd","aic3","csiq","live"]

def scan_csv(path, ncol=372):
    """Streamed max/|.|-quantile accumulation over the f0.. columns."""
    mx = np.full(ncol, -np.inf); mn = np.full(ncol, np.inf)
    n = 0
    chunks = []
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        i0 = hdr.index("f0")
        buf = []
        for row in r:
            buf.append(row[i0:i0+ncol])
            if len(buf) >= 20000:
                a = np.asarray(buf, dtype=np.float64); buf = []
                mx = np.maximum(mx, a.max(0)); mn = np.minimum(mn, a.min(0))
                n += a.shape[0]; chunks.append(a)
        if buf:
            a = np.asarray(buf, dtype=np.float64)
            mx = np.maximum(mx, a.max(0)); mn = np.minimum(mn, a.min(0))
            n += a.shape[0]; chunks.append(a)
    full = np.concatenate(chunks, 0)
    return full, mx, mn, n

per_leg = {}
allmax = np.full(372, -np.inf); allmin = np.full(372, np.inf); ntot = 0
p999 = {}
for leg in LEGS:
    p = f"{R6}/{leg}.csv"
    full, mx, mn, n = scan_csv(p)
    per_leg[leg] = dict(max=mx.tolist(), n=n)
    allmax = np.maximum(allmax, mx); allmin = np.minimum(allmin, mn); ntot += n
    p999[leg] = np.quantile(np.abs(full), 0.999, axis=0)
    print(f"{leg}: n={n} max_over_all_slots={mx.max():.6g} at f{int(mx.argmax())} ({NAMES[int(mx.argmax())]})", flush=True)

# safesyn from parquet (1.5 GB CSV; the parquet is the same rows)
import pyarrow.parquet as pq
pf = pq.ParquetFile(f"{R6}/safesyn.parquet")
cols = [f"f{i}" for i in range(372)]
smx = np.full(372, -np.inf); smn = np.full(372, np.inf); sn = 0
q_acc = []
for b in pf.iter_batches(batch_size=50000, columns=cols):
    a = np.column_stack([b.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float64)
    smx = np.maximum(smx, a.max(0)); smn = np.minimum(smn, a.min(0)); sn += a.shape[0]
    q_acc.append(np.abs(a[::7]))          # 1-in-7 stratified subsample for the quantile
per_leg["safesyn"] = dict(max=smx.tolist(), n=sn)
allmax = np.maximum(allmax, smx); allmin = np.minimum(allmin, smn); ntot += sn
p999["safesyn"] = np.quantile(np.concatenate(q_acc,0), 0.999, axis=0)
print(f"safesyn: n={sn} max_over_all_slots={smx.max():.6g} at f{int(smx.argmax())} ({NAMES[int(smx.argmax())]})", flush=True)

photo_p999 = np.max(np.stack([p999[k] for k in p999]), axis=0)

with open(f"{OUT}/slot_audit_rev1.tsv","w") as fh:
    fh.write("slot\tname\tfamily\tblock_local\tmax_all_legs\tmin_all_legs\tp999_worst_leg\tmax_over_p999\n")
    for i in range(372):
        r = allmax[i]/photo_p999[i] if photo_p999[i] > 0 else float("nan")
        fh.write(f"f{i}\t{NAMES[i]}\t{FAM[i]}\t{LOC[i]}\t{allmax[i]:.6g}\t{allmin[i]:.6g}\t{photo_p999[i]:.6g}\t{r:.6g}\n")

json.dump({"n_rows_total": int(ntot), "legs": {k: v["n"] for k,v in per_leg.items()},
           "per_leg_max": {k: v["max"] for k,v in per_leg.items()},
           "names": NAMES}, open(f"{OUT}/slot_audit_rev1.json","w"))
print(f"\nTOTAL rows {ntot}")
order = np.argsort(-allmax)
print("\nTop 20 slots by max over every leg:")
for i in order[:20]:
    print(f"  f{i:<4d} {NAMES[i]:<34s} max={allmax[i]:12.4f}  p99.9={photo_p999[i]:.4g}  x{allmax[i]/max(photo_p999[i],1e-12):.4g}")
