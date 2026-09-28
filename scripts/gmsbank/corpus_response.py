#!/usr/bin/env python3
"""Check four TRAIN reference blur/noise polarity and report zenjpeg ladder.

The extractor input pairs.tsv and metadata.tsv have identical order. No
human scores or model fits enter this diagnostic.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path('/var/tmp/gmsbank/corpus_probe')
BASE = 1322
CELLS = 12
SIGNALS = 15


def main():
    with (ROOT / 'metadata.tsv').open() as f:
        metadata = list(csv.DictReader(f, delimiter='\t'))
    with Path(sys.argv[1]).open() as f:
        features = list(csv.DictReader(f))
    if len(metadata) != 84 or len(features) != len(metadata):
        raise ValueError(f'expected 84 ordered pairs: {len(metadata)} metadata, {len(features)} features')
    features.sort(key=lambda r: int(float(r['pair_index'])))
    if [int(float(r['pair_index'])) for r in features] != list(range(84)):
        raise ValueError('missing or duplicated fixture pair_index')
    rows = []
    algebra_cells = 0
    for meta, feat in zip(metadata, features):
        totals = []
        for k in range(5):
            loss = sum(float(feat[f'f{BASE + cell * SIGNALS + 3 * k}']) for cell in range(CELLS))
            gain = sum(float(feat[f'f{BASE + cell * SIGNALS + 3 * k + 1}']) for cell in range(CELLS))
            totals.append((loss, gain))
        for k in range(4):
            if sum(totals[k]) + 1e-12 < sum(totals[k + 1]):
                raise AssertionError(f'c-bank algebra {meta["id"]} {meta["distortion"]} q{meta["quality"]} k{k}')
            algebra_cells += 1
        rows.append(dict(meta, totals=totals))
    polarity = []
    for row in rows:
        if row['distortion'] not in ('blur', 'noise'):
            continue
        loss = sum(x[0] for x in row['totals'])
        gain = sum(x[1] for x in row['totals'])
        passed = loss > gain if row['distortion'] == 'blur' else gain > loss
        polarity.append(dict(id=row['id'], kind=row['distortion'], loss=loss,
                             gain=gain, passed=passed))
    ladder = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['distortion'] == 'zenjpeg420':
            for k, (loss, gain) in enumerate(row['totals']):
                ladder[int(row['quality'])][k].append((loss + gain) / CELLS)
    ladder_mean = {str(q): [sum(ladder[q][k]) / 4 for k in range(5)]
                   for q in sorted(ladder, reverse=True)}
    report = dict(schema='gmsbank-corpus-response-v1', references=4,
                  pairs=len(rows), algebra_cells=algebra_cells,
                  polarity=polarity, polarity_passes=sum(r['passed'] for r in polarity),
                  ladder_mean_by_k=ladder_mean,
                  ladder_note='mean(loss+gain) over 4 refs and 12 channel-scale cells; no quality monotonicity assertion')
    (ROOT / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))
    assert report['polarity_passes'] == 8, 'blur/noise polarity failed on a TRAIN reference'


if __name__ == '__main__':
    main()
