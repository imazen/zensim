#!/usr/bin/env python3
"""Column-subset parquets for the v2 per-family ablation
(benchmarks/v2_trainability_ab_2026-07-19.md, KILL-prescription follow-up).

v2 flat layout: f[scale*3*29 + ch*29 + local], scales 0..3, ch 0..2 (X/Y/B),
local per zensim::feature_v2::idx. Subsetting drops a local-idx set from every
(scale, ch) block and renames the survivors to contiguous f0..fN so the trainer's
f-column autodetect works unchanged.
"""
import pyarrow.parquet as pq
from pathlib import Path

AB = Path("/mnt/v/output/zensim/v2-ab-2026-07-19")
W = 29
VARIANTS = {
    "base22": set(range(22, 29)),   # drop all phase-2 candidates
    "noBB": {25, 27},               # drop blockiness + banding
    "noPJND": {20, 21, 23, 24},     # drop transducer core+bank+fragility
}
CORPORA = ["safesyn", "kadid", "tid", "cid22val", "csiq", "live"]

for vname, drop in VARIANTS.items():
    keep_locals = [l for l in range(W) if l not in drop]
    keep_old = [
        s * 3 * W + ch * W + l for s in range(4) for ch in range(3) for l in keep_locals
    ]
    for corpus in CORPORA:
        t = pq.read_table(AB / f"v2_{corpus}.parquet")
        meta = [c for c in t.column_names if not (c.startswith("f") and c[1:].isdigit())]
        sub = t.select(meta + [f"f{i}" for i in keep_old])
        sub = sub.rename_columns(meta + [f"f{k}" for k in range(len(keep_old))])
        pq.write_table(sub, AB / f"v2{vname}_{corpus}.parquet", compression="zstd")
    print(vname, "->", len(keep_old), "features")
