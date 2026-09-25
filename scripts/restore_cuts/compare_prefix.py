#!/usr/bin/env python3
"""Bit-compare the first K feature columns of two extractor CSVs, joined by row_id.

Usage: compare_prefix.py <a.csv> <b.csv> [K=1502]
Prints `PREFIX_IDENTITY rows=N cols=K cells=N*K mismatches=M`; exit 1 on any mismatch or a
row_id set difference. Values are parsed as float64 and compared by bit pattern.
"""
import csv
import struct
import sys


def load(path, k):
    out = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        rid = hdr.index("row_id")
        pos = [hdr.index(f"f{i}") for i in range(k)]
        for row in r:
            out[int(row[rid])] = [struct.pack("<d", float(row[j])) for j in pos]
    return out


a_path, b_path = sys.argv[1:3]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 1502
a, b = load(a_path, k), load(b_path, k)
if a.keys() != b.keys():
    print(f"PREFIX_IDENTITY row_id sets differ: {len(a)} vs {len(b)}")
    sys.exit(1)
bad = sum(x != y for rid in a for x, y in zip(a[rid], b[rid]))
print(f"PREFIX_IDENTITY rows={len(a)} cols={k} cells={len(a) * k} mismatches={bad}")
sys.exit(1 if bad else 0)
