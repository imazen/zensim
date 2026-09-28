#!/usr/bin/env python3
"""joint-core-v2 assembly: merge generated fresh-cell rows into the
v1-union growth table, verify paths/coverage, emit the extraction input
and the full provenance table.

Inputs (all under /mnt/v/output/zensim/joint-core-v2/):
  pairs/pairs_v2.tsv          — v1 rows (cohort=v1) + v2 reused-leg growth
                              (cohort=v2reused), written by select_core_v2.py
  pairs/rows/<rendition>.tsv  — one line per generated cell, FRESH_COLS
  plan/renditions_v2.tsv      — the growth-rendition plan (for `missing`)

Outputs:
  pairs/pairs_core.tsv        — extraction input (all v2 pairs, cohort col)
  pairs/pairs_provenance.tsv  — full per-pair provenance (shas, kernel, leg)
  plan/coverage_report.json   — band/class/origin/codec/q coverage numbers
"""
import csv, json, os, sys, collections
import numpy as np

ROOT = '/mnt/v/output/zensim/joint-core-v2'

FRESH_COLS = ['rendition', 'src_id', 'group', 'src_class', 'rung', 'band',
              'kernel', 'ref_path', 'ref_sha256', 'ref_w', 'ref_h',
              'dist_path', 'dist_sha256', 'codec', 'q', 'ssim2', 'butteraugli',
              'dist_bytes']


def load_fresh():
    plan = {r['rendition']: r for r in csv.DictReader(
            open(f'{ROOT}/plan/renditions_v2.tsv'), delimiter='\t')}
    rows = []
    missing = []
    rows_dir = f'{ROOT}/pairs/rows'
    have = {f[:-4] for f in os.listdir(rows_dir)} if os.path.isdir(rows_dir) else set()
    for name, p in plan.items():
        if name not in have:
            missing.append(name)
            continue
        with open(f'{rows_dir}/{name}.tsv') as f:
            for line in f:
                c = line.rstrip('\n').split('\t')
                if len(c) != len(FRESH_COLS):
                    continue
                r = dict(zip(FRESH_COLS, c))
                r['leg'] = 'fresh_safesyn' if name.startswith('ss_') else 'fresh_imazen26'
                r['ref_path'] = f"{ROOT}/{r['ref_path']}"
                r['dist_path'] = f"{ROOT}/{r['dist_path']}"
                r['human_score'] = r['ssim2']  # codec-leg convention: raw ssim2
                r['origin'] = r['src_id']
                r['ref_basename'] = name + '.png'
                r['cohort'] = 'v2fresh'
                rows.append(r)
    return rows, missing


def load_merged():
    rows = []
    with open(f'{ROOT}/pairs/pairs_v2.tsv') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            r['origin'] = r['ref_basename']
            for k in ('ref_sha256', 'dist_sha256', 'src_class', 'rung',
                      'ref_w', 'ref_h', 'ssim2', 'butteraugli', 'dist_bytes',
                      'rendition'):
                r.setdefault(k, '')
            r.setdefault('cohort', 'v1')
            rows.append(r)
    return rows


def main():
    fresh, missing = load_fresh()
    merged = load_merged()
    if missing:
        print(f'ABORT: {len(missing)} planned renditions have no rows:', missing[:10])
        sys.exit(1)
    all_rows = merged + fresh

    bad = [r for r in all_rows
           if not (os.path.exists(r['ref_path']) and os.path.exists(r['dist_path']))]
    if bad:
        print(f'ABORT: {len(bad)} pairs with missing files, e.g.', bad[0]['ref_path'])
        sys.exit(1)

    n = len(all_rows)
    cov = {'total_pairs': n}
    cov['by_leg'] = dict(collections.Counter(r['leg'] for r in all_rows))
    cov['by_cohort'] = dict(collections.Counter(r['cohort'] for r in all_rows))
    cov['by_band'] = dict(collections.Counter(r['band'] for r in all_rows))
    cov['band_share'] = {k: v / n for k, v in cov['by_band'].items()}
    cov['by_codec'] = dict(collections.Counter(r['codec'] for r in all_rows))
    cov['by_kernel'] = dict(collections.Counter(r['kernel'] for r in all_rows))
    cov['by_group'] = dict(collections.Counter(r['group'] for r in all_rows))
    cov['photo_share'] = cov['by_group'].get('photo', 0) / n
    # The >=55%-at-384..1024 rule is the 'mid' band share (v1's own
    # measure: mid = the 384..1024 rung band).
    cov['mid_share_gate_ok'] = cov['band_share'].get('mid', 0) >= 0.55
    cov['photo_share_gate_ok'] = cov['photo_share'] >= 0.75
    dom = {}
    for band in ('mid', 'small', 'tiny'):
        oc = collections.Counter(r['origin'] for r in all_rows if r['band'] == band)
        tot = sum(oc.values())
        dom[band] = {'top_origin': oc.most_common(1)[0] if oc else None,
                     'top_share': oc.most_common(1)[0][1] / tot if oc else 0,
                     'n_origins': len(oc)}
    cov['band_dominance'] = dom
    qs = sorted(int(r['q']) for r in all_rows if str(r['q']).isdigit())
    cov['q_deciles'] = [int(np.percentile(qs, p)) for p in range(0, 101, 10)]

    with open(f'{ROOT}/pairs/pairs_core.tsv', 'w') as f:
        f.write('ref_path\tdist_path\thuman_score\tleg\tcodec\tq\tband\tkernel\tref_basename\tgroup\tcohort\n')
        for r in all_rows:
            f.write('\t'.join(str(r[k]) for k in
                              ['ref_path', 'dist_path', 'human_score', 'leg',
                               'codec', 'q', 'band', 'kernel', 'ref_basename',
                               'group', 'cohort']) + '\n')
    with open(f'{ROOT}/pairs/pairs_provenance.tsv', 'w') as f:
        cols = ['leg', 'cohort', 'rendition', 'origin', 'group', 'src_class',
                'rung', 'band', 'kernel', 'ref_path', 'ref_sha256', 'ref_w',
                'ref_h', 'dist_path', 'dist_sha256', 'codec', 'q', 'ssim2',
                'butteraugli', 'dist_bytes', 'human_score']
        f.write('\t'.join(cols) + '\n')
        for r in all_rows:
            f.write('\t'.join(str(r.get(k, '')) for k in cols) + '\n')
    with open(f'{ROOT}/plan/coverage_report.json', 'w') as f:
        json.dump(cov, f, indent=1)
    print(json.dumps(cov, indent=1))


if __name__ == '__main__':
    main()
