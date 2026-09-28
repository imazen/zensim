#!/usr/bin/env python3
"""zgeom Z1: extraction CSVs -> per-arm fit tables + permuted controls.

Arms (228 cols each — the whole basic+peaks surface is the pooled
statistic; the replacement repools every slot over the 5x5 lattice):
  z1gate    box2:b5gate  (block-peak x two-state gate — the replacement)
  z1max     box2:b5max   (ungated block-peak — decomposition)
  z1gateperm/z1maxperm  same tables, all 228 feature columns
                        row-permuted within each leg (the control —
                        every slot is a block statistic, so every slot
                        is permuted; baseline box2:glob stays in z2/)

Baseline for Z1 fits = z2/box2 tables (same rows, same row order).
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
ARMS = {'z1gate': 'b5gate', 'z1max': 'b5max'}

core = list(csv.DictReader(open(f'{V2}/pairs/pairs_core.tsv'), delimiter='\t'))
train_idx = [i for i, r in enumerate(core) if r['leg'] in TRAIN_LEGS]
leg_sel = {leg: [i for i in train_idx if core[i]['leg'] == leg]
           for leg in TRAIN_LEGS}

for arm, token in ARMS.items():
    df = pd.read_csv(f'{EX}/z1_{token}_train.csv').set_index('row_id')
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
        os.makedirs(f'{OUT}/z1/{arm}/features', exist_ok=True)
        sub.to_parquet(f'{OUT}/z1/{arm}/features/{leg}.parquet',
                       index=False, compression='zstd')
        print(f'z1/{arm}/features/{leg}.parquet', len(sub))
    for dev_leg in ('safesyn', 'cid22', 'human', 'codec'):
        frozen = pq.read_table(
            f'{DEDUP}/{dev_leg}_development.parquet').to_pandas()
        tsv = list(csv.DictReader(
            open(f'{VDIR}/{dev_leg}_dev.tsv'), delimiter='\t'))
        assert len(tsv) == len(frozen), (dev_leg, len(tsv), len(frozen))
        rid2pos = {int(r['row_id']): j for j, r in enumerate(tsv)}
        fmap = pd.read_csv(
            f'{EX}/z1_{token}_{dev_leg}_dev.csv').set_index('row_id')
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
        os.makedirs(f'{OUT}/z1/{arm}/dev', exist_ok=True)
        out.to_parquet(
            f'{OUT}/z1/{arm}/dev/{dev_leg}_development.parquet',
            compression='zstd', index=False)
        print(f'z1/{arm}/dev/{dev_leg}_development.parquet', len(out))

# --- permuted controls: every feature column row-permuted within leg ---
rng = np.random.default_rng(6619)
for arm in ARMS:
    perm = f'{arm}perm'
    for leg in TRAIN_LEGS:
        t = pq.read_table(
            f'{OUT}/z1/{arm}/features/{leg}.parquet').to_pandas()
        for c in FC:
            t[c] = rng.permutation(t[c].to_numpy())
        os.makedirs(f'{OUT}/z1/{perm}/features', exist_ok=True)
        t.to_parquet(f'{OUT}/z1/{perm}/features/{leg}.parquet',
                     index=False, compression='zstd')
        print(f'z1/{perm}/features/{leg}.parquet', len(t))
    for dev_leg in ('safesyn', 'cid22', 'human', 'codec'):
        t = pq.read_table(
            f'{OUT}/z1/{arm}/dev/{dev_leg}_development.parquet').to_pandas()
        for c in FC:
            t[c] = rng.permutation(t[c].to_numpy())
        os.makedirs(f'{OUT}/z1/{perm}/dev', exist_ok=True)
        t.to_parquet(
            f'{OUT}/z1/{perm}/dev/{dev_leg}_development.parquet',
            compression='zstd', index=False)
        print(f'z1/{perm}/dev/{dev_leg}_development.parquet', len(t))
print('done')
