#!/usr/bin/env python3
"""transplant lane: assemble the 1040-wide master feature tables.

Column layout (the transplant-era table):
  f0..f943     canonical era (frozen dedup fit tables / v1 parquets /
               pass-A extraction — provenance per cohort below)
  f944..f979   transplant-36 (X4 pooled 24 + X7 12) — pass B / pass A
  f980..f1009  DVIFM ycbcr_cb-30 (dense pass C)
  f1010..f1039 DVIFM ycbcr_cr-30 (dense pass D)

Canonical f0..f943 per cohort:
  v1        — v1 features/<leg>.parquet rows, joined on (ref_path,dist_path)
  v2reused  — frozen {cid22,human}_fit.parquet / hdr pure table at the
              reused_rows_v2.json indices (parallel order to
              reused_pairs_v2.tsv, asserted)
  v2fresh   — the pass-A extraction CSV itself

Dev legs: frozen {safesyn,cid22,human,codec}_development.parquet (identity
+ f0..f943 untouched) + appended cols extracted on the verdict pair TSVs
(verified positional match, row_id joins).

Also emits: subset row-index lists, permuted-column variants, era audit.
"""
import csv, json, os, sys, collections
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

V2 = '/mnt/v/output/zensim/joint-core-v2'
V1 = '/mnt/v/output/zensim/joint-core-v1'
OUT = '/mnt/v/output/zensim/dvifm-transplant-2026-09-20'
DEDUP = '/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables'
HDR_PURE = '/mnt/v/output/zensim/hdr944-leg-pure-2026-08-28/hdr_v3mix944_train_pure.parquet'
VERDICT = '/mnt/v/output/zensim/dvifm-verdict-2026-09-20'
EX = f'{OUT}/extract'

BASE = [f'f{i}' for i in range(944)]
APP = [f'f{i}' for i in range(944, 1040)]          # 96 appended cols
TRAIN_LEGS = {'fresh_safesyn', 'fresh_imazen26', 'cid22', 'human'}
META = ['leg', 'group', 'codec', 'q', 'band', 'kernel', 'cohort',
        'ref_path', 'dist_path']
report = {'checks': {}, 'counts': {}}


def read_csv_feats(path):
    """extractor CSV -> {row_id: np.array(f0..fN)} — uses header names."""
    df = pd.read_csv(path)
    fcols = [c for c in df.columns if c.startswith('f') and
             c[1:].isdigit()]
    fcols.sort(key=lambda c: int(c[1:]))
    out = {}
    for rid, arr in zip(df['row_id'].astype(int),
                        df[fcols].to_numpy(dtype=np.float64)):
        out[rid] = arr
    return out, len(fcols)


def load_pairs():
    rows = list(csv.DictReader(open(f'{V2}/pairs/pairs_core.tsv'),
                               delimiter='\t'))
    print('pairs_core:', len(rows),
          dict(collections.Counter(r['cohort'] for r in rows)))
    return rows


def canonical_frames():
    """cohort v1: v1 per-leg parquets keyed on (ref_path,dist_path)."""
    v1 = {}
    for leg in ('fresh_safesyn', 'fresh_imazen26', 'cid22', 'human',
                'konfig'):
        t = pq.read_table(f'{V1}/features/{leg}.parquet').to_pandas()
        for rp, dp, vals in zip(t['ref_path'], t['dist_path'],
                                t[BASE].to_numpy(dtype=np.float64)):
            v1[(rp, dp)] = vals
    t = pq.read_table(f'{V1}/features/hdr_pq/hdr_pq_regime.parquet'
                      ).to_pandas()
    for rp, dp, vals in zip(t['ref_path'], t['dist_path'],
                            t[BASE].to_numpy(dtype=np.float64)):
        v1[(rp, dp)] = vals
    print('v1 canonical keys:', len(v1))
    return v1


def reused_fit_rows():
    """v2reused: (leg -> {fit_table_row_idx: f0..f943}) and the
    pairs->idx positional map asserted against reused_pairs_v2.tsv."""
    rr = json.load(open(f'{V2}/plan/reused_rows_v2.json'))
    src = {'cid22': f'{DEDUP}/cid22_fit.parquet',
           'human': f'{DEDUP}/human_fit.parquet',
           'hdr': HDR_PURE}
    tables = {leg: pq.read_table(p).to_pandas() for leg, p in src.items()}
    # reused_pairs_v2.tsv order is parallel to out_rows[leg] (both iterate
    # the same sorted selection) -> position j in leg block = rows[leg][j]
    per_leg_pos = collections.defaultdict(int)
    pos2idx = {}   # (leg, position-within-leg) -> fit row index
    for r in csv.DictReader(open(f'{V2}/plan/reused_pairs_v2.tsv'),
                            delimiter='\t'):
        j = per_leg_pos[r['leg']]
        pos2idx[(r['leg'], j)] = rr[r['leg']][j]
        per_leg_pos[r['leg']] += 1
    for leg in rr:
        assert per_leg_pos[leg] == len(rr[leg]), (leg, per_leg_pos[leg])
    return tables, pos2idx


def build_master(rows):
    v1 = canonical_frames()
    tables, pos2idx = reused_fit_rows()
    a, na = read_csv_feats(f'{EX}/v2fresh_986t.csv'); assert na == 1022, na
    b, nb = read_csv_feats(f'{EX}/rest_t.csv');      assert nb == 37, nb
    cb, nc = read_csv_feats(f'{EX}/all_cb.csv');     assert nc == 30, nc
    cr, nd = read_csv_feats(f'{EX}/all_cr.csv');     assert nd == 30, nd
    # v1 feature-table gap: 4 zenavif-s6 cells are in pairs_core but absent
    # from v1's features/*.parquet (v1 extraction gap, dist files exist and
    # were re-extracted into missing4_986.csv, keyed by pairs_core index).
    m4 = {}
    if os.path.exists(f'{EX}/missing4_986.csv'):
        m4, nm4 = read_csv_feats(f'{EX}/missing4_986.csv'); assert nm4 == 986, nm4
    print('csv maps:', len(a), len(b), len(cb), len(cr), 'missing4:', len(m4))

    recs, miss = [], collections.Counter()
    recovered = 0
    hdr_zeroed = 0
    leg_pos = collections.defaultdict(int)  # position within v2reused leg
    for i, r in enumerate(rows):
        key = (r['ref_path'], r['dist_path'])
        co = r['cohort']
        if co == 'v1':
            base = v1.get(key)
            if base is None:
                # the 4-cell v1 feature-table gap -> supplemental re-extract
                row4 = m4.get(i)
                if row4 is None:
                    miss['v1'] += 1; continue
                base = row4[:944]
                recovered += 1
        elif co == 'v2reused':
            # position of this row inside its reused leg = rank among
            # v2reused rows of the same leg (pairs_core repeats the
            # reused_pairs_v2 order verbatim)
            idx = pos2idx.get((r['leg'], leg_pos[r['leg']]))
            leg_pos[r['leg']] += 1
            if idx is None:
                miss['v2reused-map'] += 1; continue
            src = tables[r['leg']].iloc[idx]
            # integrity: basename stem + score (hdr table is 0..1 scale)
            stem = os.path.splitext(r['ref_basename'])[0]
            s_tab = float(src['human_score'])
            s_pair = float(r['human_score'])
            if r['leg'] == 'hdr':
                s_tab *= 100.0
            ok_stem = (src['ref_basename'] == stem or
                       src['ref_basename'].endswith(':' + stem) or
                       src['ref_basename'] == r['ref_basename'] or
                       r['ref_basename'].endswith(':' + src['ref_basename']))
            if not (ok_stem and abs(s_tab - s_pair) < 1e-3):
                miss['v2reused-path'] += 1; continue
            base = src[BASE].to_numpy(dtype=np.float64)
        else:  # v2fresh
            row = a.get(i)
            if row is None:
                miss['v2fresh-csv'] += 1; continue
            base = row[:944]
        if r['leg'] == 'hdr':
            # PQ-regime rows cannot take the SDR research path; hdr is not
            # a train leg, so appended cols are inert zeros (recorded).
            hdr_zeroed += 1
            recs.append((i, np.concatenate([base, np.zeros(96)])))
            continue
        if co == 'v2fresh':
            t = a[i][986:1022]
        else:
            brow = b.get(i)
            if brow is None:
                miss['transplant'] += 1; continue
            t = brow[1:37]
        if len(t) != 36:
            miss['transplant'] += 1; continue
        ccb = cb.get(i)
        ccr = cr.get(i)
        if ccb is None or ccr is None:
            miss['chroma'] += 1; continue
        recs.append((i, np.concatenate([base, t, ccb, ccr])))
    if miss:
        print('MISSING:', dict(miss)); sys.exit(1)
    report['checks']['hdr_zeroed_appended'] = hdr_zeroed
    report['checks']['v1_gap_recovered'] = recovered
    recs.sort(key=lambda x: x[0])
    assert [i for i, _ in recs] == list(range(len(rows)))
    feats = np.stack([v for _, v in recs])
    print('master feats:', feats.shape)
    return feats


def emit_train(rows, feats):
    os.makedirs(f'{OUT}/features', exist_ok=True)
    df = pd.DataFrame(feats, columns=[f'f{i}' for i in range(1040)])
    df.insert(0, 'human_score', [float(r['human_score']) for r in rows])
    df.insert(0, 'ref_basename', [r['ref_basename'] for r in rows])
    for k in META:
        df[k] = [r.get(k, '') for r in rows]
    counts = {}
    for leg in TRAIN_LEGS:
        sub = df[df['leg'] == leg]
        sub.to_parquet(f'{OUT}/features/{leg}.parquet', index=False, compression='zstd')
        counts[leg] = len(sub)
        print(f'features/{leg}.parquet', len(sub))
    report['counts']['train'] = counts
    return df


def emit_dev():
    """frozen dev + appended cols from the dev extraction CSVs."""
    os.makedirs(f'{OUT}/dev', exist_ok=True)
    legmap = {'safesyn': 'safesyn_dev', 'cid22': 'cid22_dev',
              'human': 'human_dev', 'codec': 'codec_dev'}
    counts = {}
    for dev_leg, tsv_leg in legmap.items():
        frozen = pq.read_table(
            f'{DEDUP}/{dev_leg}_development.parquet').to_pandas()
        # TSV order == frozen order (verified): row_id -> position
        tsv = list(csv.DictReader(open(f'{VERDICT}/pairs/{tsv_leg}.tsv'),
                                  delimiter='\t'))
        assert len(tsv) == len(frozen), (dev_leg, len(tsv), len(frozen))
        rid2pos = {int(r['row_id']): j for j, r in enumerate(tsv)}
        app = np.zeros((len(frozen), 96))
        for name, sl in (('t', (0, 36)), ('cb', (36, 66)), ('cr', (66, 96))):
            fmap, nw = read_csv_feats(f'{EX}/dev_{dev_leg}_{name}.csv')
            want = 37 if name == 't' else 30
            assert nw == want, (dev_leg, name, nw)
            for rid, arr in fmap.items():
                j = rid2pos[rid]
                app[j, sl[0]:sl[1]] = (arr[1:37] if name == 't' else arr)
        # f-run must stay CONTIGUOUS for the loader — appended cols go
        # between f943 and the trailing identity cols.
        id_cols = [c for c in frozen.columns
                   if not (c.startswith('f') and c[1:].isdigit())]
        fcols = [f'f{i}' for i in range(944)]
        out = frozen[id_cols[:2] + fcols].copy()
        for k in range(96):
            out[f'f{944 + k}'] = app[:, k]
        for c in id_cols[2:]:
            out[c] = frozen[c]
        out.to_parquet(f'{OUT}/dev/{dev_leg}_development.parquet', compression='zstd',
                       index=False)
        counts[dev_leg] = len(out)
        print(f'dev/{dev_leg}_development.parquet', len(out))
    report['counts']['dev'] = counts


def era_audit(rows, feats):
    """pass-A verify CSV vs canonical f0..f943 -> era compatibility."""
    path = f'{EX}/verify_986.csv'
    if not os.path.exists(path):
        report['checks']['era_audit'] = 'skipped (no verify csv)'
        return
    fmap, nw = read_csv_feats(path)
    assert nw == 986, nw   # plain full-986 verify pass (no appended cols)
    diffs = []
    for rid, arr in fmap.items():
        diffs.append(np.abs(arr[:944] - feats[rid, :944]).max())
    diffs = np.array(diffs)
    report['checks']['era_audit'] = {
        'n': len(diffs), 'max_abs_diff': float(diffs.max()),
        'p99': float(np.percentile(diffs, 99)),
        'mean': float(diffs.mean())}
    print('era audit:', report['checks']['era_audit'])


def emit_variants(rows, feats):
    """permuted controls + nested subsets.

    perm-append: f944..f1039 row-permuted per leg (one set, all X arms).
    perm30:      f64..f93 row-permuted per leg, f0..f943 only (step-0).
    subsets:     row-index lists {v1_53k, reused_61k, fresh83k, full}."""
    rng = np.random.default_rng(6619)
    os.makedirs(f'{OUT}/perm', exist_ok=True)
    meta_cols = ['ref_basename', 'human_score'] + META
    allcols = [f'f{i}' for i in range(1040)]
    for leg in TRAIN_LEGS:
        sel = [i for i, r in enumerate(rows) if r['leg'] == leg]
        base = pd.DataFrame(
            {c: feats[sel, k] for k, c in enumerate(allcols)})
        for k, c in enumerate(meta_cols):
            if c == 'human_score':
                base[c] = [float(rows[i][c]) for i in sel]
            else:
                base[c] = [rows[i][c] for i in sel]
        pa = base.copy()
        for k in range(944, 1040):
            pa[f'f{k}'] = rng.permutation(pa[f'f{k}'].to_numpy())
        pa.to_parquet(f'{OUT}/perm/{leg}.parquet', index=False, compression='zstd')
        print('perm', leg, len(pa))
    # dev perm-append
    for dev_leg in ('safesyn', 'cid22', 'human', 'codec'):
        t = pq.read_table(
            f'{OUT}/dev/{dev_leg}_development.parquet').to_pandas()
        for k in range(944, 1040):
            t[f'f{k}'] = rng.permutation(t[f'f{k}'].to_numpy())
        t.to_parquet(f'{OUT}/perm/dev_{dev_leg}.parquet', index=False, compression='zstd')
    # dev perm30 — reuse v1's (identical frozen rows, same protocol seed)
    report['checks']['dev_perm30'] = 'reused v1 fits/perm30/dev tables'
    # subset row-index lists (positions in the master per-leg tables)
    os.makedirs(f'{OUT}/subsets', exist_ok=True)
    idx = {}
    for leg in TRAIN_LEGS:
        sel = [i for i, r in enumerate(rows) if r['leg'] == leg]
        pos = {i: j for j, i in enumerate(sel)}   # master row -> leg pos
        leg_rows = {'v1': [], 'v2reused': [], 'v2fresh': []}
        for i in sel:
            leg_rows[rows[i]['cohort']].append(pos[i])
        idx[leg] = leg_rows
    json.dump(idx, open(f'{OUT}/subsets/leg_rows.json', 'w'), indent=0)
    print('subset lists written')


# zensim_mlp_train's table-admission gate resolves each root's producer id
# from `_MANIFEST.json`. The 944-wide canonical projections declare the
# registered w944/ceiling_rev3 set (qualified). The 1040-wide master/dev/perm
# tables carry the same canonical base plus unregistered appended research
# cols (f944..f1039), so they declare the canonical id AND run their arms
# under --historical-replay (the coverage issue for f944+ is recorded, not
# guessed away).
CANON_ID = ('basic+peaks+masked+iw+v2+append+append2@w944/'
            'ceiling_rev3#b782e349')
APPENDED = ('f944-967 X4 pooling-24, f968-979 X7 edge-12, '
            'f980-1009 dvifm-Cb30, f1010-1039 dvifm-Cr30 '
            '(transplant lane 2026-09-20; unregistered)')


def _manifest(path, extra):
    os.makedirs(path, exist_ok=True)
    m = {'feature_set_id': CANON_ID, 'era': 'ceiling_rev3',
         'formula_revision': 3}
    m.update(extra)
    json.dump(m, open(f'{path}/_MANIFEST.json', 'w'), indent=1)


def emit_manifests():
    for s in ('s53k', 's61k', 's83k', 's105k'):
        _manifest(f'{OUT}/subsets/{s}', {
            'note': 'transplant-lane nested subset; canonical 944 projection'})
        _manifest(f'{OUT}/subsets/{s}/perm30', {
            'note': 'f64-93 row-permuted control within this size; '
                    'canonical layout, seed 6619'})
    for d, note in (
            ('features', 'v2 master 1040-wide; canonical f0-943 + appended'),
            ('dev', 'v2 dev 1040-wide; canonical f0-943 + appended'),
            ('perm', 'v2 1040-wide with f944-1039 row-permuted per leg, '
                     'seed 6619 (appended control)')):
        _manifest(f'{OUT}/{d}', {'note': note, 'appended': APPENDED})


def main():
    rows = load_pairs()
    feats = build_master(rows)
    emit_train(rows, feats)
    emit_dev()
    era_audit(rows, feats)
    emit_variants(rows, feats)
    emit_manifests()
    report['layout'] = {'f0..f943': 'canonical era',
                        'f944..f979': 'transplant-36 (X4 24 + X7 12)',
                        'f980..f1009': 'dvifm ycbcr_cb-30',
                        'f1010..f1039': 'dvifm ycbcr_cr-30'}
    os.makedirs(f'{OUT}/manifests', exist_ok=True)
    json.dump(report, open(f'{OUT}/manifests/tables_report.json', 'w'),
              indent=1)
    print(json.dumps(report['counts'], indent=1))


if __name__ == '__main__':
    main()
