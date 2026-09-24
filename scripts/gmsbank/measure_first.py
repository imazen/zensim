#!/usr/bin/env python3
"""Prepare and audit the frozen 2,000 SafeSyn TRAIN measure-first row sample."""
import csv
import json
import struct
import sys
from pathlib import Path

import pyarrow.parquet as pq

BANK = Path('/var/tmp/rev4-featbank/bank/safesyn')
ROOT = Path('/var/tmp/gmsbank/measure_first')
N = 2000


def first_rows(file, cols):
    p = pq.ParquetFile(file)
    out = []
    for rg in range(p.num_row_groups):
        t = p.read_row_group(rg, columns=cols)
        out.extend(t.to_pylist()[:N - len(out)])
        if len(out) >= N:
            break
    if len(out) != N:
        raise ValueError(f'{file}: {len(out)} rows, expected {N}')
    return out


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    keys = first_rows(BANK / 'keys.parquet', ['pair_key', 'ref_path', 'dist_path'])
    assert len({r['pair_key'] for r in keys}) == N
    with (ROOT / 'pairs.tsv').open('w') as f:
        f.write('ref_path\tdist_path\trow_index\n')
        for i, r in enumerate(keys):
            f.write(f"{r['ref_path']}\t{r['dist_path']}\t{i}\n")
    (ROOT / 'pair_keys.txt').write_text(''.join(r['pair_key'] + '\n' for r in keys))
    print(f'prepared={N} unique_keys={N}')


def bits32(x):
    return struct.pack('<f', float(x))


def check(csv_path):
    feature_path = next(BANK.glob('features__*.parquet'))
    names = pq.ParquetFile(feature_path).schema_arrow.names
    features = [n for n in names if n.startswith('f') and n[1:].isdigit() and int(n[1:]) < 986]
    assert features and max(map(lambda n: int(n[1:]), features)) < 986
    missing = [i for i in range(986) if f'f{i}' not in features]
    bank = first_rows(feature_path, ['pair_key'] + features)
    keys = (ROOT / 'pair_keys.txt').read_text().splitlines()
    assert [r['pair_key'] for r in bank] == keys
    with csv_path.open() as f:
        extracted = list(csv.DictReader(f))
    assert len(extracted) == N
    extracted.sort(key=lambda r: int(float(r['row_index'])))
    assert [int(float(r['row_index'])) for r in extracted] == list(range(N))
    differing = 0
    for i, (old, new) in enumerate(zip(bank, extracted)):
        for name in features:
            if bits32(old[name]) != bits32(new[name]):
                differing += 1
                if differing <= 4:
                    print(f'difference row={i} feature={name} old={old[name]} new={new[name]}')
    summary = dict(rows=N, stored_columns=len(features), cells=N * len(features),
                   differing=differing, stored_max_id=max(int(n[1:]) for n in features),
                   missing_bank_columns=missing, missing_bank_column_count=len(missing))
    (ROOT / 'parity.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))
    assert differing == 0


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'check':
        check(Path(sys.argv[2]))
    else:
        raise SystemExit('prepare | check <features.csv>')
