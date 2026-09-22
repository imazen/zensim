#!/usr/bin/env python3
"""joint-core-v1: assemble per-leg 944-feature parquets.

Inputs:
  features/fresh_944.csv     -- extractor output (unordered rows)
  pairs/pairs_core.tsv       -- leg/provenance for every pair
  plan/reused_rows.json      -- positional row indices into reused tables
Output:
  features/<leg>.parquet     -- ref_basename,human_score,f0..f943 + provenance
"""
import csv, json, os, sys, collections
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = '/mnt/v/output/zensim/joint-core-v1'
DEDUP = '/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables'
KONFIG = '/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/konfig_originsplit_train_944.parquet'
HDR_PURE = '/mnt/v/output/zensim/hdr944-leg-pure-2026-08-28/hdr_v3mix944_train_pure.parquet'
FCOLS = [f'f{i}' for i in range(944)]
META = ['leg', 'group', 'codec', 'q', 'band', 'kernel', 'ref_path', 'dist_path']


def band_of(le):
    if le >= 384:
        return 'mid'
    if le >= 192:
        return 'small'
    return 'tiny'


def fresh_frame():
    """join unordered extractor CSV -> pairs_core provenance on
    (ref_basename, human_score). 4 (ref,score) collisions are same-leg;
    disambiguated by exact feature re-extraction in disambiguate()."""
    pmap = collections.defaultdict(list)
    with open(f'{ROOT}/pairs/pairs_core.tsv') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['leg'].startswith('fresh'):
                pmap[(r['ref_basename'], round(float(r['human_score']), 6))].append(r)
    rows, ambiguous = [], []
    with open(f'{ROOT}/features/fresh_944.csv') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            key = (row[0], round(float(row[1]), 6))
            cands = pmap.get(key)
            if not cands:
                print('NO PAIR for', key)
                sys.exit(1)
            meta = cands[0] if len(cands) == 1 else None
            if meta is None:
                ambiguous.append((key, cands, row))
                continue
            rows.append((row, meta))
    print(f'fresh: {len(rows)} direct, {len(ambiguous)} ambiguous')
    return rows, ambiguous


def disambiguate(ambiguous, rows):
    """re-extract each candidate pair individually; match feature vectors."""
    import subprocess, tempfile
    for key, cands, csvrow in ambiguous:
        want = np.array([float(x) for x in csvrow[2:]])
        matched = None
        for c in cands:
            with tempfile.NamedTemporaryFile('w', suffix='.tsv', delete=False,
                                             dir='/home/lilith/tmp/devin') as t:
                t.write('ref_path\tdist_path\thuman_score\tleg\n')
                t.write(f"{c['ref_path']}\t{c['dist_path']}\t{c['human_score']}\t{c['leg']}\n")
                tpath = t.name
            out = tpath + '.csv'
            subprocess.run(
                ['/home/lilith/work/zen/zensim/zensim-bench/target/release/examples/extract_features_372col',
                 '--corpus', 'pairs-tsv', '--path', tpath, '--out', out, '--full-944'],
                env={**os.environ, 'ZENSIM_FORMULA_REV': '3', 'RAYON_NUM_THREADS': '4'},
                check=True, capture_output=True)
            with open(out) as f:
                rr = csv.reader(f)
                next(rr)
                got = np.array([float(x) for x in next(rr)[2:]])
            os.unlink(tpath)
            os.unlink(out)
            if np.abs(got - want).max() < 1e-6:
                matched = c
                break
        if matched is None:
            print('disambiguation FAILED for', key)
            sys.exit(1)
        rows.append((csvrow, matched))


def leg_parquet(name, df, extra):
    df = df.copy()
    for k, v in extra.items():
        df[k] = v
    path = f'{ROOT}/features/{name}.parquet'
    df.to_parquet(path, index=False)
    print(name, len(df), '->', path)
    return df


def main():
    os.makedirs(f'{ROOT}/features', exist_ok=True)
    rows, ambiguous = fresh_frame()
    if ambiguous:
        disambiguate(ambiguous, rows)
    fres = pd.DataFrame(
        [[r[0], float(r[1])] + [float(x) for x in r[2:]] +
         [m['leg'], m['group'], m['codec'], m['q'], m['band'], m['kernel'],
          m['ref_path'], m['dist_path']] for r, m in rows],
        columns=['ref_basename', 'human_score'] + FCOLS + META)
    for leg in ('fresh_imazen26', 'fresh_safesyn'):
        leg_parquet(leg, fres[fres.leg == leg], {})

    sel = json.load(open(f'{ROOT}/plan/reused_rows.json'))
    for name, table in (('cid22', f'{DEDUP}/cid22_fit.parquet'),
                        ('human', f'{DEDUP}/human_fit.parquet')):
        df = pq.read_table(table).to_pandas().iloc[sel[name]]
        meta = {}
        with open(f'{ROOT}/pairs/pairs_core.tsv') as f:
            pm = [r for r in csv.DictReader(f, delimiter='\t') if r['leg'] == name]
        # pairs_core order == sel order (selector emitted in sel order)
        assert len(pm) == len(df)
        for i, k in enumerate(META):
            meta[k] = [p[k] for p in pm]
        keep = df[['ref_basename', 'human_score'] + FCOLS].reset_index(drop=True)
        for k, v in meta.items():
            keep[k] = v
        leg_parquet(name, keep, {})

    df = pq.read_table(KONFIG).to_pandas()
    keep = df[['ref_basename', 'human_score'] + FCOLS].copy()
    keep['human_score'] *= 100.0
    with open(f'{ROOT}/pairs/pairs_core.tsv') as f:
        pm = [r for r in csv.DictReader(f, delimiter='\t') if r['leg'] == 'konfig']
    assert len(pm) == len(keep)
    for k in META:
        keep[k] = [p[k] for p in pm]
    leg_parquet('konfig', keep, {})

    # hdr: sel indices are positional into the <=1024 subset `el`
    df = pq.read_table(HDR_PURE).to_pandas()
    le = df['ref_basename'].str.extract(r'\.scale(\d+)x(\d+)\.').astype(int).max(axis=1)
    el = df[le.map(band_of) != 'big']
    hdf = el.iloc[sel['hdr']][['ref_basename', 'human_score'] + FCOLS].copy()
    hdf['human_score'] *= 100.0
    with open(f'{ROOT}/pairs/pairs_core.tsv') as f:
        pm = [r for r in csv.DictReader(f, delimiter='\t') if r['leg'] == 'hdr']
    assert len(pm) == len(hdf)
    hdf = hdf.reset_index(drop=True)
    for k in META:
        hdf[k] = [p[k] for p in pm]
    hdf['group'] = 'photo'
    leg_parquet('hdr_pq_regime', hdf, {})

    print('done')


if __name__ == '__main__':
    main()
