#!/usr/bin/env python3
"""Merge the 24 rescore chunks -> kadis700k_720.parquet, derive negrich-720
(severe subset), consolidate row groups (kill the per-batch bloat)."""
import glob, os, pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc

D = os.path.expanduser("~/kadis924")
chunks = sorted(glob.glob(f"{D}/out/kadis924_c*.parquet"))
valid, dead = [], []
for c in chunks:
    try:
        pq.ParquetFile(c).metadata.num_rows
        valid.append(c)
    except Exception:
        dead.append(os.path.basename(c))
t = pa.concat_tables([pq.read_table(c) for c in valid])
# dedupe on distorted_url (resume overlaps / rare double-writes)
seen, keep = set(), []
urls = t.column("distorted_url").to_pylist()
for i, u in enumerate(urls):
    if u not in seen:
        seen.add(u)
        keep.append(i)
if len(keep) != t.num_rows:
    t = t.take(keep)

# add ref_id (leak-free split key for train_corruption_head.py --severe-720)
t = t.append_column("ref_id", pa.array([f"kadis/{s}" for s in t.column("source_id").to_pylist()]))

full = f"{D}/kadis700k_720.parquet"
pq.write_table(t, full, compression="zstd", row_group_size=64000)
neg = t.filter(pc.less(t.column("score_zensim_gpu"), 0))
negf = f"{D}/kadis_negrich_720.parquet"
pq.write_table(neg, negf, compression="zstd", row_group_size=64000)

nfeat = sum(1 for n in t.column_names if n[0] == "f" and n[1:].isdigit())
usrc = len(set(t.column("source_id").to_pylist()))
print(f"MERGED: {t.num_rows} rows, {nfeat} feat, {t.num_columns} cols, {usrc} uniq source_ids")
print(f"  valid chunks {len(valid)}/24, dead {dead}")
print(f"  full  -> {full} ({os.path.getsize(full)//1024//1024} MB)")
print(f"  negrich-720 (score_zensim_gpu<0): {neg.num_rows} rows -> {negf} ({os.path.getsize(negf)//1024//1024} MB)")
