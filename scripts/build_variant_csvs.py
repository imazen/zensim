#!/usr/bin/env python3
"""Build V0_7 e1-fill subsampling ablation variant CSVs.

Each variant CSV = base 218k CSV + sampled fraction of e1 fill rows.

Output: /mnt/v/output/zensim/synthetic-v2/v07_e1_<pct>pct.csv
"""
import csv
import random
import sys
from pathlib import Path

BASE_CSV = Path("/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv")
TRAINING_CSV = Path("/mnt/v/output/zensim/training.csv")
OUT_DIR = Path("/mnt/v/output/zensim/synthetic-v2")
SEED = 7

# Read all e1 rows from training.csv
print(f"Reading e1 rows from {TRAINING_CSV}", flush=True)
e1_rows = []
header = None
with open(TRAINING_CSV) as f:
    reader = csv.reader(f)
    header = next(reader)
    codec_idx = header.index("codec")
    for row in reader:
        if "zenjpeg-420-e1" in row[codec_idx]:
            e1_rows.append(row)
print(f"  {len(e1_rows)} e1 rows", flush=True)

# Read base header to confirm match
with open(BASE_CSV) as f:
    base_header = next(csv.reader(f))
assert base_header == header, f"header mismatch: base={base_header} train={header}"

# Read base content (without rewriting header)
print(f"Reading base CSV {BASE_CSV}", flush=True)
with open(BASE_CSV, "rb") as f:
    base_bytes = f.read()
n_base_rows = base_bytes.count(b"\n") - 1  # -1 for header
print(f"  {n_base_rows} base rows", flush=True)

variants = [
    ("0pct", 0),
    ("5pct", int(round(len(e1_rows) * 0.05))),
    ("10pct", int(round(len(e1_rows) * 0.10))),
    ("20pct", int(round(len(e1_rows) * 0.20))),
    ("50pct", int(round(len(e1_rows) * 0.50))),
    ("100pct", len(e1_rows)),
]

for name, n_sample in variants:
    rng = random.Random(SEED)
    out_path = OUT_DIR / f"v07_e1_{name}.csv"
    print(f"\nVariant {name}: base + {n_sample} e1 rows -> {out_path}", flush=True)

    # Sample
    if n_sample == 0:
        sample = []
    elif n_sample == len(e1_rows):
        sample = e1_rows
    else:
        sample = rng.sample(e1_rows, n_sample)

    # Write: base bytes (with header) + appended sampled rows
    with open(out_path, "wb") as out:
        out.write(base_bytes)
        if not base_bytes.endswith(b"\n"):
            out.write(b"\n")
        w = csv.writer(out.__class__.__bases__[0]) if False else None
        # csv.writer doesn't take binary; use text wrapper for appended
    # Append rows in text mode
    with open(out_path, "a", newline="") as out:
        w = csv.writer(out)
        for row in sample:
            w.writerow(row)

    n_total = sum(1 for _ in open(out_path)) - 1
    print(f"  total rows: {n_total} (expected {n_base_rows + n_sample})", flush=True)
    assert n_total == n_base_rows + n_sample, f"row count mismatch for {name}"

print("\nAll variants written.", flush=True)
