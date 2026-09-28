#!/usr/bin/env python3
"""Require all 180 C8 slots to be nonzero on the 144 TRAIN-pair gate."""
import csv
import json
import sys
from pathlib import Path

BASE, END = 1322, 1502


def main():
    paths = [Path(s) for s in sys.argv[1:]]
    assert len(paths) == 3, 'CID22-64, SafeSyn-64, KADID-16 CSVs required'
    counts = [0] * (END - BASE)
    nrows = 0
    for path in paths:
        with path.open() as f:
            rows = csv.DictReader(f)
            n = 0
            for row in rows:
                for i in range(BASE, END):
                    counts[i - BASE] += float(row[f'f{i}']) != 0.0
                n += 1
        nrows += n
        print(f'{path.name}: {n} rows')
    assert nrows == 144, f'expected 144 rows, got {nrows}'
    dead = [BASE + i for i, n in enumerate(counts) if not n]
    report = {'rows': nrows, 'slots': len(counts), 'dead_slots': dead,
              'minimum_nonzero_pairs': min(counts), 'maximum_nonzero_pairs': max(counts)}
    print(json.dumps(report, sort_keys=True))
    assert not dead, f'dead C8 slots: {dead}'


if __name__ == '__main__':
    main()
