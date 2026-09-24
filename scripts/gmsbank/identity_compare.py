#!/usr/bin/env python3
"""Bitwise f0..f1321 comparison for corrected-base and C8 toggle gates."""
import csv
import json
import struct
import sys
from pathlib import Path

WIDTH = 1322


def load(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    assert rows, f'empty CSV: {path}'
    for row in rows:
        assert all(f'f{i}' in row for i in range(WIDTH)), f'missing prefix: {path}'
    return rows


def bits(x):
    return struct.pack('<d', float(x))


def main():
    a_path, b_path = sys.argv[1:3]
    a, b = load(a_path), load(b_path)
    assert len(a) == len(b), f'row count {len(a)} != {len(b)}'
    diffs = []
    for ri, (left, right) in enumerate(zip(a, b)):
        assert left['ref_basename'] == right['ref_basename'], f'ref row {ri}'
        for i in range(WIDTH):
            name = f'f{i}'
            if bits(left[name]) != bits(right[name]):
                diffs.append((ri, i, left[name], right[name]))
    report = dict(left=a_path, right=b_path, rows=len(a), cells=len(a) * WIDTH,
                  differing=len(diffs), first_differences=diffs[:8])
    print(json.dumps(report, sort_keys=True))
    assert not diffs, f'{len(diffs)} prefix cells moved'


if __name__ == '__main__':
    main()
