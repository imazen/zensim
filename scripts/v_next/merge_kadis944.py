#!/usr/bin/env python3
"""Merge the kadis944 rescore chunk(s) into the canonical kadis-944 trio
(PLAN_SOTA944 P1): kadis700k_944 + kadis_negrich_944 + kadis_944_ssim2_50k.

Construction guarantees like-for-like with the 924 files:
  - kadis700k_944: the 15 non-feature columns are carried VERBATIM from
    kadis700k_924.parquet in its exact row order; f0..f943 (f32) come from
    the fresh foldapp2 extraction, joined by distorted_url. Every 924 row
    must be present in the extraction (missing rows = hard abort, honest
    failure — no partial silent merge).
  - kadis_negrich_944: filter score_zensim_gpu < 0 (the documented negrich
    rule) — row set identical to kadis_negrich_924 by construction since
    scores are carried verbatim.
  - kadis_944_ssim2_50k: the 924 view has no distorted_url; each of its
    50k rows is matched to the full table by the md5 of its f0..f923 f32
    byte string (unique in practice; collisions/misses reported + abort).
    Non-feature cols carried verbatim from the 924 view, f0..f943 from the
    matched full-944 row.

Gate each output after with scripts/canonical_corpus/gate_backfill944.py.
Env: KADIS944_OUT (chunk dir), KADIS944_DEST (default
/mnt/v/zen/zensim-training/kadis-944-<date>).
"""
import glob
import hashlib
import os
import shutil
import sys
from datetime import date

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

D924 = "/mnt/v/zen/zensim-training/kadis-924-2026-07-27"


def read_table_staged(path, stage_dir):
    """pq.read_table via a local-disk staged copy. The threaded dataset
    scanner fails with ENOMEM on multi-GB files on the /mnt/v DrvFS mount
    (measured 2026-08-01); a byte copy to ext4 + normal read is reliable."""
    os.makedirs(stage_dir, exist_ok=True)
    local = os.path.join(stage_dir, os.path.basename(path))
    if not os.path.exists(local) or os.path.getsize(local) != os.path.getsize(path):
        shutil.copyfile(path, local)
    return pq.read_table(local)
OUTDIR = os.path.expanduser(os.environ.get("KADIS944_OUT", "~/tmp/backfill944/kadis944"))
DEST = os.environ.get("KADIS944_DEST", f"/mnt/v/zen/zensim-training/kadis-944-{date.today()}")
FEATCOLS = [f"f{i}" for i in range(944)]
F924 = [f"f{i}" for i in range(924)]

os.makedirs(DEST, exist_ok=True)

chunks = sorted(glob.glob(f"{OUTDIR}/out/kadis944_c*.parquet"))
if not chunks:
    sys.exit(f"ABORT: no chunk parquets under {OUTDIR}/out/")
print(f"merging {len(chunks)} chunk(s)", flush=True)
t = pa.concat_tables([pq.read_table(c) for c in chunks])
urls = t.column("distorted_url").to_pylist()
seen, keep = set(), []
for i, u in enumerate(urls):
    if u not in seen:
        seen.add(u)
        keep.append(i)
if len(keep) != t.num_rows:
    t = t.take(keep)
print(f"extraction rows (deduped): {t.num_rows}", flush=True)

# --- kadis700k_944: verbatim 924 metadata + fresh features in 924 row order ---
old = read_table_staged(f"{D924}/kadis700k_924.parquet", f"{OUTDIR}/stage924")
old_nonf = [n for n in old.column_names if not (n.startswith("f") and n[1:].isdigit())]
old_urls = old.column("distorted_url").to_pylist()
pos = {u: i for i, u in enumerate(t.column("distorted_url").to_pylist())}
order, missing = [], 0
for u in old_urls:
    p = pos.get(u)
    if p is None:
        missing += 1
    else:
        order.append(p)
if missing:
    sys.exit(f"ABORT: {missing}/{len(old_urls)} 924 rows missing from the 944 extraction")
feat = {c: t.column(c).take(pa.array(order, pa.int64())) for c in FEATCOLS}
full = pa.table({**{c: old.column(c) for c in old_nonf}, **feat})
full_p = f"{DEST}/kadis700k_944.parquet"
pq.write_table(full, full_p, compression="zstd", row_group_size=64000)
print(f"kadis700k_944: {full.num_rows} rows x {full.num_columns} cols -> {full_p}", flush=True)

# --- negrich: the documented filter (rows == negrich_924 by construction) ---
import pyarrow.compute as pc

neg = full.filter(pc.less(full.column("score_zensim_gpu"), 0))
neg_p = f"{DEST}/kadis_negrich_944.parquet"
pq.write_table(neg, neg_p, compression="zstd", row_group_size=64000)
n_old_neg = pq.ParquetFile(f"{D924}/kadis_negrich_924.parquet").metadata.num_rows
print(f"kadis_negrich_944: {neg.num_rows} rows (924-era: {n_old_neg}) -> {neg_p}", flush=True)
if neg.num_rows != n_old_neg:
    sys.exit("ABORT: negrich row count differs from 924")

# --- ssim2_50k view: match rows by f0..f923 f32 byte-hash ---
v924 = read_table_staged(f"{D924}/kadis_924_ssim2_50k.parquet", f"{OUTDIR}/stage924")
v_nonf = [n for n in v924.column_names if not (n.startswith("f") and n[1:].isdigit())]

def row_hashes(tbl, cols):
    mats = np.stack(
        [np.asarray(tbl.column(c).combine_chunks().to_numpy(zero_copy_only=False), np.float32)
         for c in cols],
        axis=1,
    )
    return [hashlib.md5(mats[i].tobytes()).digest() for i in range(mats.shape[0])]

print("hashing full table f0..f923 ...", flush=True)
fh = row_hashes(full, F924)
fmap = {}
dups = 0
for i, h in enumerate(fh):
    if h in fmap:
        dups += 1
    else:
        fmap[h] = i
print(f"full-table hash map: {len(fmap)} unique, {dups} duplicate vectors", flush=True)
vh = row_hashes(v924, F924)
vidx, vmiss = [], 0
for h in vh:
    p = fmap.get(h)
    if p is None:
        vmiss += 1
    else:
        vidx.append(p)
if vmiss:
    sys.exit(f"ABORT: {vmiss}/{len(vh)} ssim2_50k rows unmatched by feature-hash")
take = pa.array(vidx, pa.int64())
view = pa.table(
    {**{c: v924.column(c) for c in v_nonf},
     **{c: full.column(c).take(take) for c in FEATCOLS}}
)
view_p = f"{DEST}/kadis_944_ssim2_50k.parquet"
pq.write_table(view, view_p, compression="zstd", row_group_size=64000)
print(f"kadis_944_ssim2_50k: {view.num_rows} rows -> {view_p}", flush=True)
print("MERGE DONE — run gate_backfill944.py on all three vs the 924 files.")
