#!/usr/bin/env python3
"""transplant lane: emit the nested-size subset tables (944-wide) for the
permuted-column sensitivity curve.

Sizes (rows across the 4 train legs):
  s53k   — cohort v1 only (= v1's tables; verification, not re-fit)
  s61k   — v1 + v2reused
  s83k   — v1 + v2reused + ~half the v2fresh rows, rendition-blocked
           (a rendition's cells stay together — they are correlated)
  s105k  — everything (the full master features/*.parquet)

Outputs: subsets/<size>/<leg>.parquet (f0..f943 + meta).
"""
import csv, json, os, collections
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

V2 = '/mnt/v/output/zensim/joint-core-v2'
OUT = '/mnt/v/output/zensim/dvifm-transplant-2026-09-20'
BASE = [f'f{i}' for i in range(944)]
META = ['ref_basename', 'human_score', 'leg', 'group', 'codec', 'q',
        'band', 'kernel', 'cohort']
LEGS = ['fresh_safesyn', 'fresh_imazen26', 'cid22', 'human']

leg_rows = json.load(open(f'{OUT}/subsets/leg_rows.json'))
# rendition per master row (provenance order == pairs_core order)
rend = {}
rows = list(csv.DictReader(open(f'{V2}/pairs/pairs_core.tsv'),
                           delimiter='\t'))
prov = list(csv.DictReader(open(f'{V2}/pairs/pairs_provenance.tsv'),
                           delimiter='\t'))
assert len(rows) == len(prov)
# map master-row -> (leg, leg_pos)
pos = {leg: {i: j for j, i in enumerate(
    [i for i, r in enumerate(rows) if r['leg'] == leg])} for leg in LEGS}
for i, r in enumerate(rows):
    if r['cohort'] == 'v2fresh':
        rend.setdefault(prov[i]['rendition'], []).append((r['leg'], pos[r['leg']][i]))

rng = np.random.default_rng(4477)
rend_names = sorted(rend)
rng.shuffle(rend_names)
half_fresh = collections.defaultdict(set)
TARGET = 21700  # ~half of the 44.3k v2fresh rows
acc = 0
for name in rend_names:
    if acc >= TARGET:
        break
    for leg, p in rend[name]:
        half_fresh[leg].add(p)
        acc += 1
print('half-fresh rows:', acc)

SIZES = {
    's53k': lambda leg: leg_rows[leg]['v1'],
    's61k': lambda leg: leg_rows[leg]['v1'] + leg_rows[leg]['v2reused'],
    's83k': lambda leg: (leg_rows[leg]['v1'] + leg_rows[leg]['v2reused'] +
                         sorted(half_fresh[leg])),
}
for size, sel in SIZES.items():
    os.makedirs(f'{OUT}/subsets/{size}', exist_ok=True)
    tot = 0
    for leg in LEGS:
        t = pq.read_table(f'{OUT}/features/{leg}.parquet').to_pandas()
        sub = t.iloc[sorted(sel(leg))]
        sub[BASE + META].to_parquet(
            f'{OUT}/subsets/{size}/{leg}.parquet', index=False,
            compression='zstd')
        tot += len(sub)
    print(size, tot)

# s105k = 944-wide projection of the master tables (max-features 944
# runs must not see the appended f944..f1039 columns).
os.makedirs(f'{OUT}/subsets/s105k', exist_ok=True)
for leg in LEGS:
    t = pq.read_table(f'{OUT}/features/{leg}.parquet').to_pandas()
    t[BASE + META].to_parquet(
        f'{OUT}/subsets/s105k/{leg}.parquet', index=False,
        compression='zstd')
    print('s105k', leg, len(t))

# perm30 variants — f64..f93 row-permuted WITHIN each size's table (the
# v1 protocol, applied per size), seed recorded for the record.
rng = np.random.default_rng(6619)
PERM30 = list(range(64, 94))
for size in ('s61k', 's83k', 's105k'):
    os.makedirs(f'{OUT}/subsets/{size}/perm30', exist_ok=True)
    for leg in LEGS:
        src = (f'{OUT}/subsets/{size}/{leg}.parquet' if size != 's105k'
               else f'{OUT}/features/{leg}.parquet')
        t = pq.read_table(src).to_pandas()[BASE + META].copy()
        for k in PERM30:
            t[f'f{k}'] = rng.permutation(t[f'f{k}'].to_numpy())
        t.to_parquet(f'{OUT}/subsets/{size}/perm30/{leg}.parquet', compression='zstd',
                     index=False)
    print('perm30', size)
