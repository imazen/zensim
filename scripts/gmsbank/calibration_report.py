#!/usr/bin/env python3
"""Report preregistered label-free GMSBANK units conversion."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/var/tmp/gmsbank/calibration')


def main():
    selection = json.loads((ROOT / 'selection.json').read_text())
    metadata = {r['pair_key']: r for r in selection['pairs']}
    rows = []
    with (ROOT / 'ratios.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            key = row['pair_key']
            if key not in metadata:
                raise ValueError(f'unselected key: {key}')
            rows.append(dict(metadata[key], ratio=float(row['ratio']),
                             eligible_sites=int(row['eligible_sites'])))
    if len(rows) != len(metadata) or len({r['pair_key'] for r in rows}) != len(rows):
        raise ValueError('missing or duplicated calibration pair')
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['set'], r['content_class'], r['size_class'])].append(r['ratio'])

    def summary(values):
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('invalid ratios')
        p = np.quantile(values, [0.25, 0.5, 0.75], method='linear')
        return dict(n=int(values.size), p25=float(p[0]), p50=float(p[1]), p75=float(p[2]))

    pooled = summary([r['ratio'] for r in rows])
    c_mid = 0.0026 / pooled['p50'] ** 2
    constants = [c_mid * (4.0 ** (k - 2)) for k in range(5)]
    all_strata = [(s, c, z) for s in ('cid22_train', 'safesyn')
                  for c in ('photo', 'screen', 'line_art', 'mixed')
                  for z in ('tiny', 'small', 'medium', 'large')]
    strata = {'/'.join(k): (summary(grouped[k]) if grouped[k] else {'n': 0, 'p25': None, 'p50': None, 'p75': None})
              for k in all_strata}
    report = dict(schema='gmsbank-calibration-v1', selection=selection['schema'],
                  ratio_floor=1e-6, sample_unit='pair_median_over_ref_and_dist_sites',
                  stratum=strata,
                  pooled=pooled, c_mid=c_mid, constants=constants,
                  eligible_sites=sum(r['eligible_sites'] for r in rows),
                  pairs=len(rows))
    (ROOT / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(pairs=report['pairs'], pooled=pooled,
                          c_mid=c_mid, constants=constants,
                          eligible_sites=report['eligible_sites']), sort_keys=True))
    for k, v in report['stratum'].items():
        print(k, json.dumps(v, sort_keys=True))


if __name__ == '__main__':
    main()
