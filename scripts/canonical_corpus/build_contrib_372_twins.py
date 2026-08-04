#!/usr/bin/env python3
"""Width-discriminator 372-twin builders (sota944 campaign, bake_contrib
appendix §C.6 / results §6). Two datasets:

1. tbig_372_200k.parquet — KEYED twin of tbig_944_200k: select from the Tower ext720
bigcodec train views exactly the encoded_filename cells of tbig_944_200k and
emit them in tbig_944_200k's row order. Stride replication was G-T1-rejected
(the 720 views' row order differs from the 944 views'); the keyed join is
stronger — exact cell identity, exact order, per-row score equality asserted.
"""
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/mnt/tower/output/zensim-ext720-canonical-2026-07-22/bigcodec"
VIEWS = ["zenjpeg_lossy", "zenwebp_lossy", "zenjxl_lossy", "zenavif_lossy"]
OUT = "/mnt/v/zen/zensim-training/tbig_372_200k.parquet"
REF944 = "/mnt/v/zen/zensim-training/tbig_944_200k.parquet"

ref = pq.read_table(REF944, columns=["ref_basename", "human_score",
                                     "encoded_filename"])
keys = ref.column("encoded_filename").to_pylist()
keyset = set(keys)
print(f"ref keys: {len(keys)} ({len(keyset)} distinct)", flush=True)

feat_cols = [f"f{i}" for i in range(372)]
collected = {}  # encoded_filename -> (ref_basename, human_score, feat row idx in part tables)
parts = []
for view in VIEWS:
    path = f"{ROOT}/{view}/train_720.parquet"
    t0 = time.time()
    f = pq.ParquetFile(path)
    names = set(f.schema_arrow.names)
    refcol = "ref_filename" if "ref_filename" in names else "ref_basename"
    cols = [refcol, "score_ssim2", "encoded_filename"] + feat_cols
    for rg in range(f.metadata.num_row_groups):
        tb = f.read_row_group(rg, columns=cols)
        enc = tb.column("encoded_filename").to_pylist()
        local = [i for i, e in enumerate(enc) if e in keyset]
        if local:
            parts.append(tb.take(pa.array(local, type=pa.int64())))
        if rg % 5 == 0:
            print(f"  [{view}] rg {rg}/{f.metadata.num_row_groups} "
                  f"kept-so-far={sum(len(p) for p in parts)} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"[{view}] done {time.time()-t0:.0f}s", flush=True)

allp = pa.concat_tables(parts)
refcol = "ref_filename" if "ref_filename" in allp.column_names else "ref_basename"
print(f"collected {len(allp)} rows", flush=True)

# Reindex to the 944 slice's row order by key.
enc = allp.column("encoded_filename").to_pylist()
pos = {}
for i, e in enumerate(enc):
    pos.setdefault(e, i)  # keep-first on dupes (dedupe keep-first upstream)
missing = [k for k in keys if k not in pos]
print(f"missing keys: {len(missing)}", flush=True)
assert not missing, missing[:5]
take = pa.array([pos[k] for k in keys], type=pa.int64())
sel = allp.take(take)

ss = np.clip(sel.column("score_ssim2").to_numpy(zero_copy_only=False) / 100.0,
             0, 1)
arrays = [sel.column(refcol), pa.array(ss, type=pa.float64()),
          sel.column("encoded_filename")]
names_out = ["ref_basename", "human_score", "encoded_filename"]
for c in feat_cols:
    arrays.append(sel.column(c))
    names_out.append(c)
out = pa.table(dict(zip(names_out, arrays)))

# G-T1: (ref_basename, human_score) row-for-row vs the 944 slice.
a = list(zip(out.column("ref_basename").to_pylist(),
             [round(x, 9) for x in out.column("human_score").to_pylist()]))
b = list(zip(ref.column("ref_basename").to_pylist(),
             [round(x, 9) for x in ref.column("human_score").to_pylist()]))
n_mism = sum(1 for x, y in zip(a, b) if x != y)
print(f"G-T1 sequence equal: {a == b} (mismatches {n_mism})", flush=True)
assert a == b
pq.write_table(out, OUT, compression="zstd")
print(f"wrote {OUT}: {len(out)} rows", flush=True)


# ---------------------------------------------------------------------------
# 2. kadis_372_ssim2_50k_twin.parquet — 372-era features for the exact 50k
#    cells of kadis_944_ssim2_50k, joined from the KADIS GPU canonical by
#    (source_id, round(score_ssim2_gpu, 6)); human_score copied VERBATIM from
#    the 944 file so training targets are twin-exact. Run earlier inline
#    (2026-08-04, 0 join misses / 50,000; 981 duplicate keys among the 700k
#    canonical rows — bounded same-source near-tie feature-swap risk only).
# ---------------------------------------------------------------------------
def build_kadis_twin():
    k9 = pq.read_table(
        "/mnt/v/zen/zensim-training/kadis-944-2026-08-01/kadis_944_ssim2_50k.parquet",
        columns=["source_id", "score_ssim2_gpu", "human_score", "ref_basename"])
    featc = [f"feat_{i}" for i in range(372)]
    t = pq.read_table(
        "/mnt/v/datasets/kadis700k/canonical/kadis700k_canonical_gpu_2026-07-01.parquet",
        columns=["source_id", "score_ssim2_gpu"] + featc)
    sid = t.column("source_id").to_numpy()
    ss = t.column("score_ssim2_gpu").to_numpy()
    key = {}
    for i in range(len(sid)):
        key[(int(sid[i]), round(float(ss[i]), 6))] = i
    qsid = k9.column("source_id").to_numpy()
    qss = k9.column("score_ssim2_gpu").to_numpy()
    idx = [key[(int(qsid[i]), round(float(qss[i]), 6))] for i in range(len(qsid))]
    feats = t.select(featc).take(pa.array(idx, type=pa.int64()))
    arrays = [k9.column("ref_basename"), k9.column("human_score")]
    names_out = ["ref_basename", "human_score"]
    for i, c in enumerate(featc):
        arrays.append(feats.column(c))
        names_out.append(f"f{i}")
    outk = pa.table(dict(zip(names_out, arrays)))
    pq.write_table(outk,
                   "/mnt/v/zen/zensim-training/kadis_372_ssim2_50k_twin.parquet",
                   compression="zstd")
    print(f"wrote kadis twin: {len(outk)} rows", flush=True)


if __name__ == "__main__" and "--kadis" in __import__("sys").argv:
    build_kadis_twin()
