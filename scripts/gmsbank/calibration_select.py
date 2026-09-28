#!/usr/bin/env python3
"""Select pixel-only CID22/SafeSyn calibration pairs by frozen strata/order."""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

BANK = Path('/var/tmp/rev4-featbank/bank')
OUT = Path('/var/tmp/gmsbank/calibration')
SETS = ('cid22_train', 'safesyn')


def rows(set_name):
    cols = ('pair_key', 'ref_group', 'ref_pixels_sha256', 'dist_pixels_sha256',
            'ref_path', 'dist_path', 'width', 'height')
    table = pq.read_table(BANK / set_name / 'keys.parquet', columns=list(cols))
    return [dict(zip(cols, row)) for row in zip(*(table[c].to_pylist() for c in cols))]


def size_class(w, h):
    m = max(w, h)
    return 'tiny' if m < 256 else 'small' if m < 512 else 'medium' if m < 1024 else 'large'


def group_key(group):
    return hashlib.sha256(group.encode()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1]
    data = {s: rows(s) for s in SETS}
    if mode == 'refs':
        groups = {}
        for vals in data.values():
            for r in vals:
                groups[r['ref_group']] = r['ref_path']
        with (OUT / 'refs.tsv').open('w') as f:
            for group, path in sorted(groups.items()):
                f.write(f'{group}\t{path}\n')
        print(f'refs {len(groups)}')
        return
    if mode != 'select':
        raise SystemExit('mode: refs|select')
    classes = {}
    with (OUT / 'classes.tsv').open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            classes[r['ref_group']] = r['class']
    selected = []
    counts = Counter()
    ref_counts = Counter()
    for set_name, vals in data.items():
        strata = defaultdict(lambda: defaultdict(list))
        for r in vals:
            stratum = (classes[r['ref_group']], size_class(r['width'], r['height']))
            strata[stratum][r['ref_group']].append(r)
        for stratum, groups in sorted(strata.items()):
            for group in sorted(groups, key=group_key)[:32]:
                ref_counts[(set_name, *stratum)] += 1
                for r in sorted(groups[group], key=lambda x: x['pair_key'])[:2]:
                    selected.append(dict(set=set_name, content_class=stratum[0],
                                         size_class=stratum[1], **r))
                    counts[(set_name, *stratum)] += 1
    with (OUT / 'pairs.tsv').open('w') as f:
        for r in selected:
            f.write('\t'.join(str(r[c]) for c in ('pair_key', 'ref_pixels_sha256',
                'dist_pixels_sha256', 'ref_path', 'dist_path')) + '\n')
    (OUT / 'selection.json').write_text(json.dumps({
        'schema': 'gmsbank-calibration-selection-v1',
        'counts': {'/'.join(k): v for k, v in sorted(counts.items())},
        'reference_counts': {'/'.join(k): v for k, v in sorted(ref_counts.items())},
        'pairs': [{k: r[k] for k in ('set', 'content_class', 'size_class', 'pair_key')}
                  for r in selected]
    }, indent=2) + '\n')
    print(f'pairs {len(selected)}')
    for k, n in sorted(counts.items()):
        print('/'.join(k), n)


if __name__ == '__main__':
    main()
