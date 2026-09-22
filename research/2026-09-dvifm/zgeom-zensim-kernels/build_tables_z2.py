#!/usr/bin/env python3
"""zgeom Z2: extraction CSVs -> per-arm fit tables.

Per kernel arm (box2|bin121|bin1331):
  z2/{arm}/features/{leg}.parquet           228-col train tables,
                                          pairs_core leg order, keyed
                                          by row_id (not path join)
  z2/{arm}/dev/{leg}_development.parquet    frozen dev id cols + the
                                          arm's 228 features, TSV order

The stored master is NOT the baseline: its v1/v2reused rows carry the
v1-era feature function, so every arm's box2/baseline table is a fresh
rev-3 extraction over the same rows — one feature function per arm,
one era per lane.
"""
import csv, os, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = '/mnt/v/output/zensim/zgeom-2026-09-21'
EX = f'{OUT}/extract'
V2 = '/mnt/v/output/zensim/joint-core-v2'
DEDUP = '/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables'
VDIR = '/mnt/v/output/zensim/dvifm-verdict-2026-09-20/pairs'
TRAIN_LEGS = ('fresh_safesyn', 'fresh_imazen26', 'cid22', 'human')
META = ['leg', 'group', 'codec', 'q', 'band', 'kernel', 'cohort',
        'ref_path', 'dist_path']
FC = [f'f{i}' for i in range(228)]
ARMS = sys.argv[1:] or ['box2', 'bin121', 'bin1331']

core = list(csv.DictReader(open(f'{V2}/pairs/pairs_core.tsv'), delimiter='\t'))
train_idx = [i for i, r in enumerate(core) if r['leg'] in TRAIN_LEGS]
leg_sel = {leg: [i for i in train_idx if core[i]['leg'] == leg]
           for leg in TRAIN_LEGS}

for arm in ARMS:
    # --- train tables ---
    df = pd.read_csv(f'{EX}/z2_{arm}_train.csv').set_index('row_id')
    assert sorted(df.index.tolist()) == train_idx, \
        f'{arm}: train csv rows != train-leg pairs_core rows'
    for leg in TRAIN_LEGS:
        sel = leg_sel[leg]
        sub = pd.DataFrame({c: df.loc[sel, c].to_numpy() for c in FC})
        sub.insert(0, 'human_score',
                   [float(core[i]['human_score']) for i in sel])
        sub.insert(0, 'ref_basename', [core[i]['ref_basename'] for i in sel])
        for k in META:
            sub[k] = [core[i][k] for i in sel]
        os.makedirs(f'{OUT}/z2/{arm}/features', exist_ok=True)
        sub.to_parquet(f'{OUT}/z2/{arm}/features/{leg}.parquet',
                       index=False, compression='zstd')
        print(f'z2/{arm}/features/{leg}.parquet', len(sub))
    # --- dev tables ---
    for dev_leg in ('safesyn', 'cid22', 'human', 'codec'):
        frozen = pq.read_table(
            f'{DEDUP}/{dev_leg}_development.parquet').to_pandas()
        tsv = list(csv.DictReader(
            open(f'{VDIR}/{dev_leg}_dev.tsv'), delimiter='\t'))
        assert len(tsv) == len(frozen), (dev_leg, len(tsv), len(frozen))
        rid2pos = {int(r['row_id']): j for j, r in enumerate(tsv)}
        fmap = pd.read_csv(f'{EX}/z2_{arm}_{dev_leg}_dev.csv')
        fmap = fmap.set_index('row_id')
        feats = np.zeros((len(frozen), 228))
        for rid, j in rid2pos.items():
            feats[j] = fmap.loc[rid, FC].to_numpy(np.float64)
        id_cols = [c for c in frozen.columns
                   if not (c.startswith('f') and c[1:].isdigit())]
        out = frozen[id_cols[:2]].copy()
        for k, c in enumerate(FC):
            out[c] = feats[:, k]
        for c in id_cols[2:]:
            out[c] = frozen[c]
        os.makedirs(f'{OUT}/z2/{arm}/dev', exist_ok=True)
        out.to_parquet(
            f'{OUT}/z2/{arm}/dev/{dev_leg}_development.parquet',
            compression='zstd', index=False)
        print(f'z2/{arm}/dev/{dev_leg}_development.parquet', len(out))
print('done')
