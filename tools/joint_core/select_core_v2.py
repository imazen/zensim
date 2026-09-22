#!/usr/bin/env python3
"""joint-core-v2 growth selector — extends joint-core-v1 unchanged.

v1 rows are carried verbatim (same parquets/paths; v1 root referenced read-only).
Growth cohort, same rules (PLAN_JOINT_CORE_SET_2026-09-19):
  * reused legs: the complement of v1's kmeans selection inside each reused
    table (native kernel, no resample): cid22 +4,663, human +1,878,
    hdr +1,790 (<=1024 subset)
  * fresh_imazen26: NEW (origin, rung) renditions on unused rung slots
    + EXTRA cells (the 12-q complement) on a slice of v1 photo renditions,
    re-rendered into the v2 root (identical deterministic Mitchell output)
  * fresh_safesyn: next CLIC fit-side sources x 4 mid rungs (same regen rule)
  * non-photo floors: screen/doc/lineart kept >=5% of the grown core

Outputs under /mnt/v/output/zensim/joint-core-v2/:
  plan/renditions_v2.tsv  — NEW + re-render renditions for core_variant_gen
  plan/cells_v2.tsv       — cells for both cohorts above
  plan/reused_rows_v2.json, plan/reused_pairs_v2.tsv
  pairs/pairs_v2.tsv      — v1 verbatim + growth (extraction/provenance input)
"""
import csv, json, os, re, sys, collections
import numpy as np
import pyarrow.parquet as pq

V1 = '/mnt/v/output/zensim/joint-core-v1'
ROOT = '/mnt/v/output/zensim/joint-core-v2'
LOCAL = '/mnt/v/output/imazen-26-png-v3'
TRAIN_TSV = '/home/lilith/work/zen/imazen-26/manifests/train.tsv'
DEDUP = '/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables'
REC14 = '/var/tmp/zensim-validation-2026-09-14/baseline-recovery'
HDR_PURE = '/mnt/v/output/zensim/hdr944-leg-pure-2026-08-28/hdr_v3mix944_train_pure.parquet'
HDR_TSV = '/mnt/v/output/zensim/hdr944-leg/hdr944_features_all.tsv'
SS_SRC = '/mnt/v/input/zensim/sources'

PHOTO = ['1000-', '1200-', '1400-', '1600-', '2000-', '2400-', '3000-', '3300-']
SCREEN = ['8000-', '8100-']
DOC = ['5000-', '5200-', '5300-', '6000-', '6800-']
LINEART = ['6600-', '7000-', '2200-']
AI = ['9000-', '9094-', '9226-']
RUNGS = {'mid': [384, 512, 640, 768, 896, 1024],
         'small': [192, 224, 256],
         'tiny': [64, 96, 128]}
QGRID = [3, 5, 8, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 85, 90, 95, 98]
CELLS_PER_REND = 32
CODECS = ['zenjpeg-420-e2', 'zenwebp-m4', 'zenavif-s6', 'zenjxl-e7']

# growth quotas (pairs) for THIS step (v2a ~ +53k -> ~106k)
NEW_REND = {  # group -> band -> new renditions (pairs = rends*32)
    'photo':   {'mid': 250, 'small': 120, 'tiny': 80},
    'screen':  {'mid': 30, 'small': 25, 'tiny': 20},
    'doc':     {'mid': 30, 'small': 25, 'tiny': 20},
    'lineart': {'mid': 30, 'small': 25, 'tiny': 20},
    'ai':      {'mid': 6, 'small': 3, 'tiny': 1},
}
EXTRA_CELL_RENDS = 300      # v1 photo renditions that get the +32-cell complement
SAFESYN_NEW_SRC = 100       # next CLIC sources, 4 mid rungs each -> +12,800 pairs
SAFESYN_RUNGS = [512, 640, 768, 1024]


def cls_group(cc):
    for g, pref in [('photo', PHOTO), ('screen', SCREEN), ('doc', DOC),
                    ('lineart', LINEART), ('ai', AI)]:
        if any(cc.startswith(p) for p in pref):
            return g
    return 'other'


def load_train():
    rows = []
    with open(TRAIN_TSV) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            u = r['png_v3_sdr_url']
            if not u:
                continue
            lp = f"{LOCAL}/{r['content_class']}/{u.rsplit('/', 1)[1]}"
            if not os.path.exists(lp) or os.path.getsize(lp) < 100:
                continue
            rows.append(dict(id=r['id'], cc=r['content_class'], group=cls_group(r['content_class']),
                             w=int(r['width']), h=int(r['height']), path=lp,
                             src_sha=r['sha256']))
    return rows


def eval_side_ids():
    ids = set()
    for leg in ['codec_development', 'safesyn_development',
                'cid22_development', 'human_development']:
        t = pq.read_table(f'{DEDUP}/{leg}.parquet', columns=['ref_basename']).to_pandas()
        for b in t['ref_basename'].unique():
            m = re.match(r'o_(\d+)\.png', str(b))
            if m:
                ids.add(m.group(1))
    return ids


def plan_fresh_growth(rows):
    """NEW (src,rung) renditions on rung slots v1 did not use, plus the
    +32-cell complement on a slice of v1 photo renditions."""
    v1r = list(csv.DictReader(open(f'{V1}/plan/renditions_fresh.tsv'), delimiter='\t'))
    v1c = list(csv.DictReader(open(f'{V1}/plan/cells_fresh.tsv'), delimiter='\t'))
    used_rung = collections.defaultdict(set)          # (src_id, band) -> rungs used
    for r in v1r:
        if r['rendition'].startswith('ss_'):
            continue
        used_rung[(r['src_id'], r['band'])].add(int(r['rung']))
    used_q = collections.defaultdict(set)
    for c in v1c:
        used_q[c['rendition']].add(int(c['q']))

    rends, cells = [], []
    rid = 1173  # continue v1's rid sequence (drives the q rotation offset)
    for group, bands in NEW_REND.items():
        origs = sorted((o for o in rows if o['group'] == group), key=lambda o: o['id'])
        for band, n_rend in bands.items():
            rungs = RUNGS[band]
            made = 0
            oi = 0
            guard = 0
            while made < n_rend and origs:
                o = origs[oi % len(origs)]
                oi += 1
                guard += 1
                if guard > 100000:
                    print(f'ABORT: rung slots exhausted for {group}/{band} at {made}')
                    sys.exit(1)
                src_le = max(o['w'], o['h'])
                taken = used_rung[(o['id'], band)]
                rung = next((r for r in rungs if r <= src_le and r not in taken), None)
                if rung is None:
                    continue
                taken.add(rung)
                rname = f"{o['id']}.s{rung}"
                rends.append(dict(rendition=rname, src_id=o['id'], group=group,
                                  src_class=o['cc'], rung=rung, band=band,
                                  src_path=o['path'], src_sha256=o['src_sha'],
                                  src_w=o['w'], src_h=o['h'],
                                  kernel='zenresize-mitchell-sharpen0-srgb8'))
                off = (rid * 8) % len(QGRID)
                qs = sorted({QGRID[(off + j * 2) % len(QGRID)] for j in range(8)})
                for c in CODECS:
                    for q in qs:
                        cells.append(dict(rendition=rname, codec=c, q=q))
                rid += 1
                made += 1

    # extra cells on v1 photo renditions (mid first): the 12-q complement,
    # first 8 strided -> +32 cells matching the 4x8 cell shape.
    v1_photo = [r for r in v1r if r['group'] == 'photo' and not r['rendition'].startswith('ss_')]
    v1_photo.sort(key=lambda r: (r['band'] != 'mid', r['rendition']))
    for r in v1_photo[:EXTRA_CELL_RENDS]:
        comp = sorted(set(QGRID) - used_q[r['rendition']])
        qs = comp[:8]
        for c in CODECS:
            for q in qs:
                cells.append(dict(rendition=r['rendition'], codec=c, q=q))
        rends.append(dict(r))  # re-rendered into v2 root; sha256 must match v1's
    return rends, cells


def plan_safesyn_growth():
    dev = pq.read_table(f'{DEDUP}/safesyn_development.parquet', columns=['ref_basename']).to_pandas()
    dev_bases = {re.sub(r'_(\d+x\d+|\d+sq)$', '', b) for b in dev['ref_basename']}
    fit = pq.read_table(f'{DEDUP}/safesyn_fit.parquet', columns=['ref_basename']).to_pandas()
    fit_bases = sorted({re.sub(r'_(\d+x\d+|\d+sq)$', '', b) for b in fit['ref_basename']})
    cand = [b for b in fit_bases if b not in dev_bases
            and os.path.exists(f'{SS_SRC}/{b}.jpg')]
    srcs = cand[48:48 + SAFESYN_NEW_SRC]  # v1 used cand[:48]
    rends, cells = [], []
    rid = 10_000
    for i, b in enumerate(srcs):
        my = SAFESYN_RUNGS[i % len(SAFESYN_RUNGS):] + SAFESYN_RUNGS[: i % len(SAFESYN_RUNGS)]
        for r in my[:len(SAFESYN_RUNGS)]:
            rname = f'ss_{b}.s{r}'
            rends.append(dict(rendition=rname, src_id=b, group='photo',
                              src_class='clic-safesyn', rung=r, band='mid',
                              src_path=f'{SS_SRC}/{b}.jpg', src_sha256='',
                              src_w=0, src_h=0,
                              kernel='zenresize-mitchell-sharpen0-srgb8'))
            off = (rid * 8) % len(QGRID)
            qs = sorted({QGRID[(off + j * 2) % len(QGRID)] for j in range(8)})
            for c in CODECS:
                for q in qs:
                    cells.append(dict(rendition=rname, codec=c, q=q))
            rid += 1
    return rends, cells


def pairs_tsv_map(path):
    m = {}
    with open(path) as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            m[int(row[3])] = (row[0], row[1])
    return m


def ref_order_csv(path):
    out = collections.defaultdict(list)
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        i_ref = hdr.index('ref_basename'); i_row = hdr.index('row_id')
        for row in r:
            out[row[i_ref]].append(int(row[i_row]))
    return out


def codec_of(dist_path):
    m = re.search(r'/([^/]+)/[^/]+$', dist_path)
    return m.group(1) if m else ''


def q_of(dist_path):
    m = re.search(r'/q(\d+)\.', dist_path)
    return int(m.group(1)) if m else ''


def band_of(le):
    return 'tiny' if le <= 128 else 'small' if le <= 256 else 'mid' if le <= 1024 else 'big'


def plan_reused_growth(v1sel):
    """Complement of v1's selection inside each reused table, joined to
    image pairs with the same positional machinery as v1."""
    pairs = []
    out_rows = {}
    # --- cid22 complement ---
    df = pq.read_table(f'{DEDUP}/cid22_fit.parquet').to_pandas()
    sel = sorted(set(range(len(df))) - set(v1sel['cid22']))
    out_rows['cid22'] = sel
    cmap = ref_order_csv(f'{REC14}/cid22-train944.csv')
    pmap = pairs_tsv_map(f'{REC14}/cid22-train-pairs.tsv')
    # positional assignment must skip the rows v1 consumed: per ref, the
    # table order is the csv order, so per-ref ordinal index -> row_id.
    ref_ord = collections.defaultdict(int)   # table ordinal per ref
    per_ref_pos = {}
    for i in range(len(df)):
        rb = df['ref_basename'].iat[i] + '.png'
        per_ref_pos[i] = ref_ord[rb]
        ref_ord[rb] += 1
    for i in sel:
        rb = df['ref_basename'].iat[i] + '.png'
        rid_ = cmap[rb][per_ref_pos[i]] if per_ref_pos[i] < len(cmap[rb]) else None
        rp, dp = pmap.get(rid_, ('', ''))
        pairs.append(dict(leg='cid22', ref_path=rp, dist_path=dp,
                          human_score=df['human_score'].iat[i], kernel='native,no-resample',
                          band='mid', codec=codec_of(dp), q=q_of(dp), ref_basename=rb))
    # --- human complement (score-sequence join, same as v1) ---
    df = pq.read_table(f'{DEDUP}/human_fit.parquet').to_pandas()
    sel = sorted(set(range(len(df))) - set(v1sel['human']))
    out_rows['human'] = sel
    hpairs = collections.defaultdict(list)
    for pf in ['/mnt/v/dataset/kadid10k/kadid_pairs_ab.tsv',
               '/mnt/v/dataset/tid2013/tid_pairs_ab.tsv']:
        with open(pf) as f:
            for r in csv.DictReader(f, delimiter='\t'):
                fam = 'kadid' if 'kadid' in r['ref_path'] else 'tid'
                rb = os.path.basename(r['ref_path'])[:-4]
                key = (fam, rb.upper() if fam == 'tid' else rb)
                hs = float(r['human_score'])
                qv = (1.0 - hs) * 100.0 if fam == 'kadid' else hs * 100.0
                hpairs[key].append((r['ref_path'], r['dist_path'], qv))
    # join by score within each (fam, ref) group, skipping scores v1 used:
    # per group, collect all candidate pairs then match each selected row to
    # the unused candidate with the closest score (exact sequence, unique).
    used = collections.defaultdict(set)
    hmiss = 0
    for i in sel:
        row = df.iloc[i]
        fam = row['source_family'].split(':')[1]
        rb = row['source_family'].split(':')[2]
        key = (fam, rb)
        target = float(row['human_score'])
        best = None
        for j, cand in enumerate(hpairs[key]):
            if j in used[key]:
                continue
            if best is None or abs(cand[2] - target) < abs(hpairs[key][best][2] - target):
                best = j
        if best is None or abs(hpairs[key][best][2] - target) > 1e-4:
            hmiss += 1
            rp = dp = ''
        else:
            used[key].add(best)
            rp, dp, _ = hpairs[key][best]
        pairs.append(dict(leg='human', ref_path=rp, dist_path=dp,
                          human_score=target, kernel='native,no-resample',
                          band='mid', codec=codec_of(dp), q=q_of(dp),
                          ref_basename=row['source_family']))
    if hmiss:
        print(f'human growth join misses: {hmiss}')
    # --- hdr complement inside the <=1024 subset ---
    df = pq.read_table(HDR_PURE).to_pandas()
    le = df['ref_basename'].str.extract(r'\.scale(\d+)x(\d+)\.').astype(int).max(axis=1)
    df['band'] = le.map(band_of)
    el = df[df['band'] != 'big']
    sel_pos = sorted(set(range(len(el))) - set(v1sel['hdr']))
    out_rows['hdr'] = sel_pos
    sel_idx = [el.index[i] for i in sel_pos]
    htsv = collections.defaultdict(list)
    with open(HDR_TSV) as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            m = re.match(r'(.+\.hdr)_([0-9a-f]+)_(\w+)_q(\d+)_', row[0])
            if m:
                htsv[m.group(1) + '.png'].append(
                    (np.array([float(x) for x in row[2:]], dtype=np.float64),
                     row[0], int(m.group(4))))
    hdr_ref_root = '/mnt/v/output/imazen-26-hdr-grid-2026-06-14'
    hdr_enc = ['/mnt/v/output/zenmetrics/datagen-2026-06-23-hdr/enc/zenjxl',
               '/mnt/v/output/zenmetrics/datagen-2026-07-03-hdr-hq/enc/zenjxl']
    hmiss = 0
    elpos = {ix: p for p, ix in enumerate(el.index)}
    for ix in sel_idx:
        i = elpos[ix]
        rb = df['ref_basename'].iat[i]
        cand = htsv.get(rb, [])
        fv = df[[f'f{j}' for j in range(944)]].iloc[i].values.astype(np.float64)
        best = min(cand, key=lambda c: float(((c[0] - fv) ** 2).sum())) if cand else None
        if best is None or float(((best[0] - fv) ** 2).sum()) > 1e-6:
            hmiss += 1
            dp = ''; qv = ''
        else:
            qv = best[2]
            dp = ''
            for d in hdr_enc:
                p = f'{d}/{best[1]}'
                if os.path.exists(p):
                    dp = p
                    break
            if not dp:
                hmiss += 1
        pairs.append(dict(leg='hdr',
                          ref_path=f'{hdr_ref_root}/{rb}' if os.path.exists(f'{hdr_ref_root}/{rb}') else '',
                          dist_path=dp,
                          human_score=float(df['human_score'].iat[i]) * 100.0,
                          kernel='zenresize-mitchell-sharpen10-linearPQ-16bit',
                          band=df['band'].iat[i], codec='zenjxl-hdr',
                          q=qv, ref_basename=rb))
    if hmiss:
        print(f'hdr growth join misses: {hmiss}')
    return out_rows, pairs


def main():
    os.makedirs(f'{ROOT}/plan', exist_ok=True)
    os.makedirs(f'{ROOT}/pairs', exist_ok=True)
    rows = load_train()
    dev_ids = eval_side_ids()
    rows = [o for o in rows if o['id'] not in dev_ids]
    print('eligible train origins:', len(rows),
          dict(collections.Counter(o['group'] for o in rows)))

    rends, cells = plan_fresh_growth(rows)
    sr, sc = plan_safesyn_growth()
    rends += sr
    cells += sc
    print('growth renditions:', len(rends), 'cells:', len(cells),
          dict(collections.Counter(r['group'] for r in rends)))

    with open(f'{ROOT}/plan/renditions_v2.tsv', 'w') as f:
        f.write('rendition\tsrc_id\tgroup\tsrc_class\trung\tband\tsrc_path\tsrc_sha256\tsrc_w\tsrc_h\tkernel\n')
        for r in rends:
            f.write('\t'.join(str(r[k]) for k in
                              ['rendition', 'src_id', 'group', 'src_class', 'rung', 'band',
                               'src_path', 'src_sha256', 'src_w', 'src_h', 'kernel']) + '\n')
    with open(f'{ROOT}/plan/cells_v2.tsv', 'w') as f:
        f.write('rendition\tcodec\tq\n')
        for c in cells:
            f.write(f"{c['rendition']}\t{c['codec']}\t{c['q']}\n")

    v1sel = json.load(open(f'{V1}/plan/reused_rows.json'))
    rows_sel, reused = plan_reused_growth(v1sel)
    with open(f'{ROOT}/plan/reused_rows_v2.json', 'w') as f:
        json.dump({k: [int(i) for i in v] for k, v in rows_sel.items()}, f)
    with open(f'{ROOT}/plan/reused_pairs_v2.tsv', 'w') as f:
        f.write('leg\tref_path\tdist_path\thuman_score\tkernel\tband\tcodec\tq\tref_basename\n')
        for p in reused:
            f.write('\t'.join(str(p[k]) for k in
                              ['leg', 'ref_path', 'dist_path', 'human_score', 'kernel',
                               'band', 'codec', 'q', 'ref_basename']) + '\n')
    print('reused growth pairs:', len(reused),
          dict(collections.Counter(p['leg'] for p in reused)))

    # pairs_v2.tsv: v1 verbatim (cohort=v1) + growth rows (cohort=v2new).
    # Fresh growth rows get ref/dist paths AFTER core_variant_gen writes
    # pairs/rows/<rendition>.tsv — assemble_core_v2 merges them.
    with open(f'{ROOT}/pairs/pairs_v2.tsv', 'w') as f:
        f.write('ref_path\tdist_path\thuman_score\tleg\tcodec\tq\tband\tkernel\tref_basename\tgroup\tcohort\n')
        with open(f'{V1}/pairs/pairs_core.tsv') as g:
            for r in csv.DictReader(g, delimiter='\t'):
                f.write('\t'.join([r['ref_path'], r['dist_path'], r['human_score'], r['leg'],
                                   r['codec'], r['q'], r['band'], r['kernel'],
                                   r['ref_basename'], r['group'], 'v1']) + '\n')
        for p in reused:
            f.write('\t'.join([p['ref_path'], p['dist_path'], str(p['human_score']), p['leg'],
                               p['codec'], str(p['q']), p['band'], p['kernel'],
                               p['ref_basename'], 'photo', 'v2reused']) + '\n')
    print('pairs_v2 written (fresh growth rows appended by assemble_core_v2)')


if __name__ == '__main__':
    main()
