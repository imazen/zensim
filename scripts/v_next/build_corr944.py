#!/usr/bin/env python3
"""corruption_grid_944col builder (PLAN_SOTA944 P1): foldapp2-extract the
persisted corruption_gate PNGs (all pixels survive — PURE re-extraction, no
re-encode) in the exact `entry` order of corruption_grid_924col_2026-07-27,
and write entry + f0..f943 (f64, zstd). Gate after with
scripts/canonical_corpus/gate_backfill944.py (f0..f923 bitwise vs the 924
grid).

Env: ZM944_BIN (extractor binary), CORR944_WORK (scratch for pairs/csv,
default ~/tmp/backfill944/corr944).
"""
import csv
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

OLD = "/mnt/v/output/zensim/v2-eval-924-2026-07-27/corruption_grid_924col_2026-07-27.parquet"
OUT = os.environ.get(
    "CORR944_OUT",
    "/mnt/v/output/zensim/v2-eval-944-2026-08-01/corruption_grid_944col_2026-08-01.parquet",
)
PNG_DIR = "/mnt/v/output/zensim/corruption_gate"
WORK = os.path.expanduser(os.environ.get("CORR944_WORK", "~/tmp/backfill944/corr944"))
EXTRACT = os.environ.get("ZM944_BIN") or sys.exit("ABORT: ZM944_BIN env required")

os.makedirs(WORK, exist_ok=True)
entries = pq.read_table(OLD, columns=["entry"]).column("entry").to_pylist()
ref = os.path.join(PNG_DIR, "gb82_dog__reference.png")
assert os.path.exists(ref), f"reference missing: {ref}"

pairs = os.path.join(WORK, "corr944_pairs.tsv")
with open(pairs, "w") as f:
    f.write("ref_path\tdist_path\thuman_score\n")
    for i, e in enumerate(entries):
        d = os.path.join(PNG_DIR, f"{e}.png")
        if not os.path.exists(d):
            sys.exit(f"ABORT: missing PNG for entry {e}")
        f.write(f"{ref}\t{d}\t{i}\n")
print(f"{len(entries)} pairs -> {pairs}", flush=True)

csvp = os.path.join(WORK, "corr944_feat.csv")
env = dict(os.environ, ZENSIM_AB_MODE="foldapp2")
r = subprocess.run([EXTRACT, pairs, csvp], env=env)
if r.returncode != 0:
    sys.exit(f"ABORT: extract rc={r.returncode}")

feats = [None] * len(entries)
with open(csvp, newline="") as f:
    rd = csv.reader(f)
    hdr = next(rd)
    fi = [hdr.index(f"f{i}") for i in range(944)]
    hs = hdr.index("human_score")
    for row in rd:
        k = int(float(row[hs]))
        feats[k] = [float(row[j]) for j in fi]
missing = sum(1 for x in feats if x is None)
if missing:
    sys.exit(f"ABORT: {missing} entries missing from extraction")

cols = {"entry": pa.array(entries, pa.utf8())}
for i in range(944):
    cols[f"f{i}"] = pa.array([r_[i] for r_ in feats], pa.float64())
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pq.write_table(pa.table(cols), OUT, compression="zstd")
print(f"G-CORR: {len(entries)}/{len(entries)} matched -> {OUT}")
