#!/usr/bin/env python3
"""Report preregistered label-free GMSBANK units conversion."""
import csv
import json
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/var/tmp/gmsbank/calibration')


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('invalid ratios')
    p = np.quantile(values, [0.25, 0.5, 0.75], method='linear')
    return dict(n=int(values.size), p25=float(p[0]), p50=float(p[1]), p75=float(p[2]))


def chroma(root, stdout_only=False):
    selection=json.loads((root/'selection.json').read_text())
    metadata={r['pair_key']:r for r in selection['rows']}
    kinds=['x_value','x_gradient','b_value','b_gradient']
    with (root/'chroma_ratios.tsv').open() as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    assert len(rows)==len({(r['pair_key'],r['ratio_kind']) for r in rows})==len(metadata)*4
    assert {(r['pair_key'],r['ratio_kind']) for r in rows}=={(k,t) for k in metadata for t in kinds}
    all_strata=[(s,c,z) for s in ['cid22_train','safesyn']
                for c in ['photo','screen','line_art','mixed']
                for z in ['tiny','small','medium','large']]
    result={}
    for kind in kinds:
        selected=[r for r in rows if r['ratio_kind']==kind]
        grouped=defaultdict(list)
        valid=[]
        for row in selected:
            sites=int(row['eligible_sites'])
            assert (row['median']=='NA') == (sites==0)
            if sites==0:continue
            value=float(row['median']);valid.append(value)
            m=metadata[row['pair_key']]
            grouped[tuple(m[k] for k in ['set','content_class','size_class'])].append(value)
        pooled=summary(valid)
        middle=(550.0 if kind.endswith('value') else 140.0)/pooled['p50']**2
        result[kind]=dict(pooled=pooled,c_mid=middle,
            constants=[middle*4.0**(k-2) for k in range(5)],
            empty_pairs=sum(int(r['eligible_sites'])==0 for r in selected),
            eligible_sites=sum(int(r['eligible_sites']) for r in selected),
            strata={'/'.join(k):summary(grouped[k]) if grouped[k] else {'n':0} for k in all_strata})
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(schema='gmsbank-chroma-calibration-v1',pairs=len(metadata),
                selection_sha256=sha(root/'selection.json'),ratios_sha256=sha(root/'chroma_ratios.tsv'),
                sample_unit='pair_geometry_median_over_reference_and_distorted_sites',
                source_classes_inherited=True,human_labels_read=False,ratios=result)
    if not stdout_only:
        with (root/'chroma_report.json').open('x') as f:f.write(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({**report,'ratios':{k:{a:b for a,b in v.items() if a!='strata'} for k,v in result.items()}},sort_keys=True))


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
    if len(sys.argv) in (3, 4) and sys.argv[1]=='--chroma-dir':
        stdout_only = len(sys.argv) == 4 and sys.argv[3] == '--stdout'
        assert len(sys.argv) == 3 or stdout_only
        chroma(Path(sys.argv[2]), stdout_only)
    else:
        main()
