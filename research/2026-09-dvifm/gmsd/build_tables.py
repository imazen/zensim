#!/usr/bin/env python3
"""gmsd lane: extraction CSVs -> per-arm fit tables + size-matched permuted
controls, in the block5 lane's exact table shape (same rows, same row order,
same metadata columns, same dev-leg frozen identity columns).

Extraction (`tools/gmsd/extract.sh`) emits per row:
  f0..f227     basic+peaks, production pooling (box2:glob) — the base `G`
  f228..f239   GMS map std per (scale, channel), cell = scale*3 + ch
  f240..f251   GMS map mean per cell (emitted, not an arm)
  f252..f347   map-deviation: std of the 8 materialized maps per cell
               (sd, art, det, mse, hf_sq_src, hf_sq_dst, hf_abs_src,
               hf_abs_dst), cell-major, map-minor

Arms (one change each; `p` = the added segment row-permuted within each
leg, column by column — same values, same width, no row correspondence):
  A_gms   = G + 12 GMS-std columns                 (240)
  A_gmsp  = its size-matched permuted control      (240)
  A_dev   = G + 96 map-std columns                 (324)
  A_devp  = its size-matched permuted control      (324)
  R_<x>   = replacement at matched column count — built only for an
            addition that beats its permuted control (see --replace)

Gates before any table is written:
  * the base f0..f227 of every row must be BIT-IDENTICAL to the zgeom
    lane's z2_box2 CSV (the `G` tables' source) — a fresh extraction that
    moved the base surface would make every contrast meaningless;
  * the 12 GMS / 96 dev columns must be finite.

Also writes reports/redundancy.json: how much of each map's std is already
determined by zensim's existing mean + L2 columns (std² = n/(n-1)·(L2² −
mean²) exactly for sd/art/det, whose L1 and L2 pools are emitted).
"""
import csv, json, os, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

L = '/var/tmp/gmsd-lane'
EX = f'{L}/extract'
TAB = f'{L}/tables'
ZG = '/mnt/v/output/zensim/zgeom-2026-09-21'
V2 = '/mnt/v/output/zensim/joint-core-v2'
DEDUP = '/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables'
VDIR = '/mnt/v/output/zensim/dvifm-verdict-2026-09-20/pairs'
TRAIN_LEGS = ('fresh_safesyn', 'fresh_imazen26', 'cid22', 'human')
DEV_LEGS = ('safesyn', 'cid22', 'human', 'codec')
META = ['leg', 'group', 'codec', 'q', 'band', 'kernel', 'cohort',
        'ref_path', 'dist_path']
BASE = [f'f{i}' for i in range(228)]
GMS = [f'f{i}' for i in range(228, 240)]
DEV = [f'f{i}' for i in range(252, 348)]
MAPS = ['sd', 'art', 'det', 'mse', 'hf_sq_src', 'hf_sq_dst', 'hf_abs_src', 'hf_abs_dst']


def load(tag):
    df = pd.read_csv(f'{EX}/gm_{tag}.csv').set_index('row_id')
    base = pd.read_csv(f'{ZG}/extract/z2_box2_{tag}.csv', usecols=['row_id'] + BASE).set_index('row_id')
    df = df.loc[base.index]
    a = df[BASE].to_numpy(np.float64)
    b = base[BASE].to_numpy(np.float64)
    same = np.array_equal(a.view(np.uint64), b.view(np.uint64))
    if not same:
        bad = np.argwhere(a.view(np.uint64) != b.view(np.uint64))
        raise SystemExit(f'{tag}: base f0..227 NOT bit-identical to z2_box2 '
                         f'({len(bad)} cells differ, first {bad[:5].tolist()})')
    ext = df[GMS + DEV].to_numpy(np.float64)
    assert np.isfinite(ext).all(), f'{tag}: non-finite gms/dev column'
    print(f'{tag}: {len(df)} rows, base bit-identical to z2_box2')
    return df


_core = None


def core():
    global _core
    if _core is None:
        _core = list(csv.DictReader(open(f'{V2}/pairs/pairs_core.tsv'), delimiter='\t'))
    return _core


def assemble(df, idx, add_cols, perm, seed):
    base = df.loc[idx, BASE].to_numpy(np.float64)
    add = df.loc[idx, add_cols].to_numpy(np.float64)
    if perm:
        rng = np.random.default_rng(seed)
        for j in range(add.shape[1]):
            add[:, j] = add[rng.permutation(len(idx)), j]
    return np.concatenate([base, add], axis=1)


def slots_hash8(n):
    """`zensim::feature_set_id::slots_hash8(0..n)` (FNV-1a over the
    comma-joined decimal slot list, 64->32 fold) — checked against the
    registered 228 (3fb78648) and 240 (c5c9da1b) values."""
    h, prime, mask = 0xcbf29ce484222325, 0x100000001b3, (1 << 64) - 1
    for i in range(n):
        if i:
            h = ((h ^ ord(',')) * prime) & mask
        for b in str(i).encode():
            h = ((h ^ b) * prime) & mask
    return '%08x' % ((h >> 32) ^ (h & 0xffffffff))


def write_manifest(root, name, width, add_desc, perm):
    man = {
        'feature_set_id': f'basic+peaks@w{width}/zgeom_box2_glob_arm_{name.lower()}#{slots_hash8(width)}',
        'era': f'zgeom_box2_glob_gmsdev_mapdev_arm_{name.lower()}',
        'formula_revision': 3,
        'producer': 'tools/gmsd/build_tables.py over extract_features_372col '
                    '--zgeom box2:glob --zgeom-gmsdev --zgeom-mapdev (gmsd lane 2026-09-22)',
        'lane': 'gmsd-2026-09-22',
        'component_sets': ['basic+peaks@w228/zgeom_box2_glob#3fb78648 (f0..227, '
                           'bit-identical to z2_box2)', add_desc],
        'permuted_segment': perm,
        'note': 'Same rows/row order as the zgeom/block5 arm tables. The perm '
                'control row-permutes only the added segment, per column, per leg.',
    }
    for sub in ('features', 'dev'):
        json.dump(man, open(f'{root}/{sub}/_MANIFEST.json', 'w'), indent=1)


def build(name, add_cols, add_desc, perm, dfs, seed):
    root = f'{TAB}/{name}'
    os.makedirs(f'{root}/features', exist_ok=True)
    os.makedirs(f'{root}/dev', exist_ok=True)
    c = core()
    for k, leg in enumerate(TRAIN_LEGS):
        sel = [i for i, r in enumerate(c) if r['leg'] == leg]
        idx = sel  # z2_train.tsv row_id == pairs_core.tsv index (verified)
        mat = assemble(dfs['train'], idx, add_cols, perm, seed + k)
        out = pd.DataFrame(mat, columns=[f'f{i}' for i in range(mat.shape[1])])
        out.insert(0, 'human_score', [float(c[i]['human_score']) for i in sel])
        out.insert(0, 'ref_basename', [c[i]['ref_basename'] for i in sel])
        for m in META:
            out[m] = [c[i][m] for i in sel]
        out.to_parquet(f'{root}/features/{leg}.parquet', index=False, compression='zstd')
    for k, dleg in enumerate(DEV_LEGS):
        frozen = pq.read_table(f'{DEDUP}/{dleg}_development.parquet').to_pandas()
        tsv = list(csv.DictReader(open(f'{VDIR}/{dleg}_dev.tsv'), delimiter='\t'))
        assert len(tsv) == len(frozen)
        idx = [int(r['row_id']) for r in tsv]  # TSV order == frozen order
        mat = assemble(dfs[f'{dleg}_dev'], idx, add_cols, perm, seed + 100 + k)
        id_cols = [col for col in frozen.columns if not (col.startswith('f') and col[1:].isdigit())]
        out = frozen[id_cols[:2]].copy()
        for j in range(mat.shape[1]):
            out[f'f{j}'] = mat[:, j]
        for col in id_cols[2:]:
            out[col] = frozen[col]
        out.to_parquet(f'{root}/dev/{dleg}_development.parquet', compression='zstd', index=False)
    write_manifest(root, name, 228 + len(add_cols), add_desc, perm)
    print('built', name, 228 + len(add_cols), 'cols')


def redundancy(df):
    """For sd/art/det: how well do the existing L1 (pos 0/3/6) and L2
    (pos 2/5/8) columns determine the emitted map std? Exact identity:
    std^2 = n/(n-1) (L2^2 - L1^2) — n is unknown per row here, so report the
    Spearman of emitted std vs sqrt(max(L2^2-L1^2,0)) and the max relative
    deviation (n/(n-1) ~ 1 for n >= ~64^2)."""
    from scipy.stats import spearmanr
    out = {}
    for m_i, (m, l1, l2) in enumerate([('sd', 0, 2), ('art', 3, 5), ('det', 6, 8)]):
        rhos, rel = [], []
        for s in range(4):
            for ch in range(3):
                cell = s * 3 + ch
                base = s * 39 + ch * 13
                a = df[f'f{base + l1}'].to_numpy(np.float64)
                b = df[f'f{base + l2}'].to_numpy(np.float64)
                implied = np.sqrt(np.maximum(b * b - a * a, 0.0))
                emitted = df[f'f{252 + cell * 8 + m_i}'].to_numpy(np.float64)
                ok = emitted > 0
                rhos.append(float(spearmanr(implied, emitted).correlation))
                rel.append(float(np.max(np.abs(implied[ok] - emitted[ok]) / emitted[ok])) if ok.any() else 0.0)
        out[m] = dict(spearman_min=min(rhos), spearman_median=float(np.median(rhos)),
                      max_rel_dev_per_cell=rel)
    return out


WEAK = '/mnt/v/output/zensim/block5-2026-09-22/specs/weak_slots.json'


def build_replace(name, add_cols, add_desc, dfs):
    """R arm at MATCHED column count (228): the k weakest glob slots by the
    block5 lane's registered TRAIN ranking (|Spearman(f_j, human_score)|,
    pooled TRAIN; weak_slots.json `order`) are overwritten, position for
    position, by the k added columns. No new selection is made here."""
    order = json.load(open(WEAK))['order']
    k = len(add_cols)
    drop = sorted(order[:k])
    root = f'{TAB}/{name}'
    os.makedirs(f'{root}/features', exist_ok=True)
    os.makedirs(f'{root}/dev', exist_ok=True)
    c = core()

    def mat_for(df, idx):
        m = df.loc[idx, BASE].to_numpy(np.float64).copy()
        m[:, drop] = df.loc[idx, add_cols].to_numpy(np.float64)
        return m
    for leg in TRAIN_LEGS:
        sel = [i for i, r in enumerate(c) if r['leg'] == leg]
        m = mat_for(dfs['train'], sel)
        out = pd.DataFrame(m, columns=[f'f{i}' for i in range(228)])
        out.insert(0, 'human_score', [float(c[i]['human_score']) for i in sel])
        out.insert(0, 'ref_basename', [c[i]['ref_basename'] for i in sel])
        for mm in META:
            out[mm] = [c[i][mm] for i in sel]
        out.to_parquet(f'{root}/features/{leg}.parquet', index=False, compression='zstd')
    for dleg in DEV_LEGS:
        frozen = pq.read_table(f'{DEDUP}/{dleg}_development.parquet').to_pandas()
        tsv = list(csv.DictReader(open(f'{VDIR}/{dleg}_dev.tsv'), delimiter='\t'))
        m = mat_for(dfs[f'{dleg}_dev'], [int(r['row_id']) for r in tsv])
        id_cols = [col for col in frozen.columns if not (col.startswith('f') and col[1:].isdigit())]
        out = frozen[id_cols[:2]].copy()
        for j in range(228):
            out[f'f{j}'] = m[:, j]
        for col in id_cols[2:]:
            out[col] = frozen[col]
        out.to_parquet(f'{root}/dev/{dleg}_development.parquet', compression='zstd', index=False)
    write_manifest(root, name, 228, add_desc + f' written over glob slots {drop} '
                   '(block5 weak_slots.json order, TRAIN |rho|)', False)
    print('built', name, 'replacing', drop)


def main():
    if '--replace' in sys.argv:
        which = sys.argv[sys.argv.index('--replace') + 1]
        dfs = {'train': load('train')}
        for leg in DEV_LEGS:
            dfs[f'{leg}_dev'] = load(f'{leg}_dev')
        cols, desc = {'gms': (GMS, 'GMS-std f228..239 (12)'),
                      'dev': (DEV, 'map-dev f252..347 (96)')}[which]
        build_replace(f'R_{which}', cols, desc, dfs)
        return
    dfs = {'train': load('train')}
    for leg in DEV_LEGS:
        dfs[f'{leg}_dev'] = load(f'{leg}_dev')
    os.makedirs(f'{L}/reports', exist_ok=True)
    red = redundancy(dfs['train'])
    json.dump(red, open(f'{L}/reports/redundancy.json', 'w'), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ('spearman_min', 'spearman_median')} for k, v in red.items()}))
    arms = {
        'A_gms': (GMS, 'GMS-std f228..239 (12)'),
        'A_dev': (DEV, 'map-dev f252..347 (96)'),
    }
    for i, (name, (cols, desc)) in enumerate(arms.items()):
        build(name, cols, desc, False, dfs, 1000 * (i + 1))
        build(name + 'p', cols, desc, True, dfs, 1000 * (i + 1))
    print('done')


if __name__ == '__main__':
    main()
