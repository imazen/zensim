#!/usr/bin/env python3
"""Feature masking for the append-only extended (720) parquets.

Per the feature-numbering directive (2026-07-19): deprecated features are MASKED
(their column zeroed) — never dropped/renumbered. Masking keeps the vector width
constant (720) across variants, so the MLP's input layer is identical and only
the information content changes — a properly controlled ablation. A constant-0
column standardizes to 0 (trainer guards std with .max(1e-12)) and is inert.

ext layout: f0..f371 = FROZEN v1 block; f372..f719 = v2 block, where the v2 flat
index is `scale*3*29 + ch*29 + local` (29 features/ch, ch 0=X/1=Y/2=B), so
ext_col = 372 + scale*87 + ch*29 + local.

Variants (mask sets of (local, ch) in the v2 block):
  luma    — transducers (20,21,23,24) on chroma ch 0,2 → Y-only masking (screen winner)
  nobad   — luma + blockiness(25) + banding(27) on all ch (ablation demotion candidates)

Usage: python3 scripts/v_next/v2_ext_mask.py <corpus1,corpus2,...> <variant>
"""
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

AB = Path("/mnt/v/output/zensim/v2-ab-2026-07-19")
V2_BASE = 372
W = 29
VARIANTS = {
    "luma": [(l, c) for l in (20, 21, 23, 24) for c in (0, 2)],
    "nobad": [(l, c) for l in (20, 21, 23, 24) for c in (0, 2)]
    + [(l, c) for l in (25, 27) for c in (0, 1, 2)],
}


def mask_cols(drops):
    return {
        V2_BASE + s * 3 * W + c * W + l
        for (l, c) in drops
        for s in range(4)
    }


corpora = sys.argv[1].split(",")
variant = sys.argv[2]
masked = mask_cols(VARIANTS[variant])
zero = pa.array([0.0])  # broadcast via compute below

for corpus in corpora:
    t = pq.read_table(AB / f"ext_{corpus}.parquet")
    n = t.num_rows
    cols = {}
    for name in t.column_names:
        if name.startswith("f") and name[1:].isdigit() and int(name[1:]) in masked:
            cols[name] = pa.array([0.0] * n, type=pa.float64())
        else:
            cols[name] = t.column(name)
    pq.write_table(pa.table(cols), AB / f"ext{variant}_{corpus}.parquet", compression="zstd")
print(f"{variant}: masked {len(masked)} cols x {len(corpora)} corpora (width unchanged 720)")
