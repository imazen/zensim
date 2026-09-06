#!/usr/bin/env python3
"""The HF-ratio family's three members, per leg: are the other two bounded?
Plus the exactly-recoverable ratio r = var_dst/var_src and the arm previews
that are pure functions of r."""
import csv, json, os
import numpy as np, pyarrow.parquet as pq

R6="/mnt/v/output/zensim/rev2-2026-09-05/r6/tables/ssim2"; OUT="/mnt/v/output/zensim/rev2-2026-09-05/r6b"
CH=["X","Y","B"]
VL=[c*13+10 for c in range(12)]; TL=[c*13+11 for c in range(12)]; CI=[c*13+12 for c in range(12)]
def nm(i): c=i//13; return f"s{c//3}_{CH[c%3]}"
LEGS=["cid22","kadid","tid","konjnd","aic3","csiq","live"]

def cols_csv(path, idx):
    out=[]
    with open(path,newline="") as fh:
        r=csv.reader(fh); hdr=next(fh_r:=r); i0=hdr.index("f0")
        for row in r: out.append([row[i0+j] for j in idx])
    return np.asarray(out,dtype=np.float64)

def cols_pq(path, idx):
    pf=pq.ParquetFile(path); cs=[f"f{i}" for i in idx]; acc=[]
    for b in pf.iter_batches(batch_size=50000, columns=cs):
        acc.append(np.column_stack([b.column(c).to_numpy(zero_copy_only=False) for c in cs]).astype(np.float64))
    return np.concatenate(acc,0)

idx = VL+TL+CI
rows={}
for leg in LEGS: rows[leg]=cols_csv(f"{R6}/{leg}.csv", idx)
rows["safesyn"]=cols_pq(f"{R6}/safesyn.parquet", idx)

rep={}
print(f"{'leg':9s} {'n':>7s} {'max var_loss':>12s} {'max tex_loss':>12s} {'max contrast_inc':>17s} {'rows ci>1':>10s} {'rows ci>100':>12s} {'p99.9 ci':>10s}")
for leg,a in rows.items():
    vl=a[:,0:12]; tl=a[:,12:24]; ci=a[:,24:36]
    n=a.shape[0]
    r1=int((ci>1).any(1).sum()); r100=int((ci>100).any(1).sum())
    p999=float(np.quantile(ci,0.999))
    rep[leg]=dict(n=n, max_var_loss=float(vl.max()), max_tex_loss=float(tl.max()),
                  max_contrast_inc=float(ci.max()), rows_ci_gt1=r1, rows_ci_gt100=r100,
                  p999_ci=p999, p99_ci=float(np.quantile(ci,0.99)),
                  frac_ci_nonzero=float((ci>0).mean()))
    print(f"{leg:9s} {n:7d} {vl.max():12.6f} {tl.max():12.6f} {ci.max():17.4f} {r1:10d} {r100:12d} {p999:10.4g}")

# the ratio r = 1 + contrast_inc - var_loss  (one of the two is always 0)
allci=np.concatenate([a[:,24:36] for a in rows.values()],0)
allvl=np.concatenate([a[:,0:12] for a in rows.values()],0)
r = 1.0 + allci - allvl
both = int(((allci>0)&(allvl>0)).sum())
print(f"\ncells where BOTH contrast_inc>0 and var_loss>0 (must be 0 by construction): {both}")
print(f"recovered ratio r: min {r.min():.6g}  max {r.max():.6g}  median {np.median(r):.6g}")
for q in (0.5,0.9,0.99,0.999,0.9999,1.0):
    print(f"  r q{q}: {np.quantile(r,q):.6g}")
cells=allci.size
print(f"\ntotal contrast_inc cells: {cells}")
for t in (1,2,10,100,1000):
    print(f"  cells > {t}: {int((allci>t).sum())} ({100*(allci>t).mean():.4f} %)")

# CID22-only photographic reference (the gold holdout, no pathological leg)
cid=rows["cid22"][:,24:36]
print(f"\nCID22 contrast_inc: max {cid.max():.6g}  p99.9 {np.quantile(cid,0.999):.6g}  p99 {np.quantile(cid,0.99):.6g}  p95 {np.quantile(cid,0.95):.6g}")
rep["_pooled"]=dict(cells=int(cells), both_positive=both,
    r_max=float(r.max()), r_p999=float(np.quantile(r,0.999)),
    cid22_p999=float(np.quantile(cid,0.999)), cid22_p99=float(np.quantile(cid,0.99)),
    cid22_max=float(cid.max()),
    cells_gt={str(t): int((allci>t).sum()) for t in (1,2,10,100,1000)})
json.dump(rep, open(f"{OUT}/hf_family_audit.json","w"), indent=1)
np.save(f"{OUT}/contrast_inc_all.npy", allci.astype(np.float32))
np.save(f"{OUT}/var_loss_all.npy", allvl.astype(np.float32))
