#!/usr/bin/env python3
"""Channel-aware column subsets for the transducer-recalibration screen
(benchmarks/v2_trainability_ab_2026-07-19.md ablation follow-up).

v2 flat layout: f[scale*3*29 + ch*29 + local], scale 0..3, ch 0..2 (X/Y/B).
Transducer locals: 20 core, 21 fragility, 23 low_k, 24 high_k.

Variants test WHICH part of the transducer family costs CID22 (the ablation
showed the whole family costs ~0.10 CID22 but carries CSIQ 0.31->0.18):
  luma    — keep transducers on ch1 (Y) only; drop for ch 0,2 (chroma masking noise?)
  corelow — keep core(20)+low_k(23); drop fragility(21)+high_k(24) (over-masking?)
  nohighk — drop high_k(24) only (the most aggressive member?)
"""
import sys
import pyarrow.parquet as pq
from pathlib import Path

AB = Path("/mnt/v/output/zensim/v2-ab-2026-07-19")
W = 29
# (local, ch) pairs to DROP.
VARIANTS = {
    "luma": [(l, ch) for l in (20, 21, 23, 24) for ch in (0, 2)],
    "corelow": [(l, ch) for l in (21, 24) for ch in (0, 1, 2)],
    "nohighk": [(24, ch) for ch in (0, 1, 2)],
}
# override the corpus list on argv (decision stage adds safesyn_full + aic3);
# override the variant list with a second arg (e.g. "luma" only)
CORPORA = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "safesyn", "kadid", "tid", "cid22val", "csiq", "live"
]
WANT = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)

for vname in WANT:
    drop_set = set(VARIANTS[vname])
    keep_old = [
        s * 3 * W + ch * W + l
        for s in range(4)
        for ch in range(3)
        for l in range(W)
        if (l, ch) not in drop_set
    ]
    for corpus in CORPORA:
        t = pq.read_table(AB / f"v2_{corpus}.parquet")
        meta = [c for c in t.column_names if not (c.startswith("f") and c[1:].isdigit())]
        sub = t.select(meta + [f"f{i}" for i in keep_old])
        sub = sub.rename_columns(meta + [f"f{k}" for k in range(len(keep_old))])
        pq.write_table(sub, AB / f"v2{vname}_{corpus}.parquet", compression="zstd")
    print(vname, "->", len(keep_old), "features x", len(CORPORA), "corpora")
