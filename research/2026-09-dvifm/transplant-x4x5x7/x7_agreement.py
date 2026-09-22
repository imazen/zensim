#!/usr/bin/env python3
"""transplant lane X7 — the §5 within-image agreement gate.

On TRAIN codec pairs, per codec family, does each new term raise
within-image agreement with ssim2 ∧ butteraugli on JPEG/WebP (blocking
dominant) without hurting AVIF/JXL? Terms:

  f968..f970  t1  raw-L0 5x5 block peak |D|          (X,Y,B)
  f971..f973  t2  across-boundary vs within-block    (X,Y,B)
  f974..f976  t3f fixed-phase 8-lattice boundary     (X,Y,B)
  f977..f979  t3m 8-phase-max boundary               (X,Y,B)

Baseline: f397 — the existing oriented-blockiness slot (v2 idx 25,
fixed-phase). Controls: the same terms read from the permuted tables.

Agreement per ref: Spearman(term, -ssim2) and Spearman(term, +butteraugli)
over the ref's cells; mean per codec family. Higher = better ranking.
"""
import csv, json, os, collections
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as sst

V2 = '/mnt/v/output/zensim/joint-core-v2'
OUT = '/mnt/v/output/zensim/dvifm-transplant-2026-09-20'
LEGS = ['fresh_imazen26', 'fresh_safesyn']   # codec legs with ssim2+bau
FAM = {'zenjpeg': 'jpeg', 'mozjpeg': 'jpeg', 'jpeg': 'jpeg',
       'webp': 'webp', 'zenwebp': 'webp',
       'avif': 'avif', 'zenavif': 'avif', 'cld_avif': 'avif',
       'jxl': 'jxl', 'zenjxl': 'jxl',
       'png': 'png', 'zenpng': 'png',
       'aom': 'av1', 'cld_heic': 'heic'}
TERMS = {'f397_base': ['f397'],
         't1': [f'f{i}' for i in range(968, 971)],
         't2': [f'f{i}' for i in range(971, 974)],
         't3fix': [f'f{i}' for i in range(974, 977)],
         't3max': [f'f{i}' for i in range(977, 980)]}


def fam_of(codec):
    c = codec.split('-')[0].lower()
    for k, v in FAM.items():
        if k in codec.lower():
            return v
    return FAM.get(c, 'other')


def ref_agreement(df, tcol):
    """mean Spearman(term, -ssim2) / Spearman(term, +bau) within refs."""
    a_s, a_b = [], []
    for _, g in df.groupby('ref_basename'):
        if len(g) < 4:
            continue
        x = g[tcol].to_numpy(float)
        if np.ptp(x) == 0:
            continue
        a_s.append(sst.spearmanr(x, -g['ssim2'].to_numpy(float)).statistic)
        a_b.append(sst.spearmanr(x, g['butteraugli'].to_numpy(float)).statistic)
    a_s = np.array([v for v in a_s if np.isfinite(v)])
    a_b = np.array([v for v in a_b if np.isfinite(v)])
    return (float(np.nanmean(a_s)) if len(a_s) else None,
            float(np.nanmean(a_b)) if len(a_b) else None, len(a_s))


def main():
    prov = {(r['ref_path'], r['dist_path']): r for r in csv.DictReader(
            open(f'{V2}/pairs/pairs_provenance.tsv'), delimiter='\t')}
    def metrics(t):
        s2, ba = [], []
        for rp, dp in zip(t['ref_path'], t['dist_path']):
            p = prov.get((rp, dp), {})
            s2.append(float(p['ssim2']) if p.get('ssim2') else np.nan)
            ba.append(float(p['butteraugli']) if p.get('butteraugli') else np.nan)
        return s2, ba
    frames = []
    for leg in LEGS:
        t = pq.read_table(f'{OUT}/features/{leg}.parquet').to_pandas()
        t = t[t['codec'] != '']
        t['ssim2'], t['butteraugli'] = metrics(t)
        t['fam'] = t['codec'].map(fam_of)
        frames.append(t)
    df = pd.concat(frames).dropna(subset=['ssim2', 'butteraugli'])
    print('rows with both metrics:', len(df),
          dict(df['fam'].value_counts()))

    out = {}
    for fam, gdf in df.groupby('fam'):
        out[fam] = {'n_rows': len(gdf), 'n_refs': gdf['ref_basename'].nunique()}
        for tname, cols in TERMS.items():
            if len(cols) == 1:
                g = gdf.copy(); g['_t'] = g[cols[0]]
            else:
                g = gdf.copy(); g['_t'] = g[cols].mean(axis=1)
            s_ag, b_ag, n = ref_agreement(g, '_t')
            out[fam][tname] = {'ssim2_rho': s_ag, 'bau_rho': b_ag,
                               'n_refs': n}
    # permuted control: t3max read from the perm table
    perm = {}
    for leg in LEGS:
        t = pq.read_table(f'{OUT}/perm/{leg}.parquet').to_pandas()
        t['ssim2'], t['butteraugli'] = metrics(t)
        t['fam'] = t['codec'].map(fam_of)
        perm[leg] = t
    pdf = pd.concat(perm.values()).dropna(subset=['ssim2', 'butteraugli'])
    for fam, gdf in pdf.groupby('fam'):
        g = gdf.copy()
        g['_t'] = g[[f'f{i}' for i in range(977, 980)]].mean(axis=1)
        s_ag, b_ag, n = ref_agreement(g, '_t')
        out.setdefault(fam, {})['t3max_PERM'] = {
            'ssim2_rho': s_ag, 'bau_rho': b_ag, 'n_refs': n}
    json.dump(out, open(f'{OUT}/reports/x7_agreement.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
