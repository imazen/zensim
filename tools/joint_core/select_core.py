#!/usr/bin/env python3
"""joint-core-v1 selector: fresh-leg origin/rendition/cell plan + reused-leg
k-means subsample, per docs/PLAN_JOINT_CORE_SET_2026-09-19 (corrected v3).

Hard rules baked in (supervisor correction 2026-09-20):
  * every resample is plain zenresize Mitchell, sharpen=0 (fresh legs) or a
    recorded known kernel (hdr leg: Mitchell+sharpen10 linear-light PQ)
  * scale bands: >=55% pairs at 384-1024, ~25% at 192-256, ~20% at 64-128,
    hard cap 1024, no origin dominates a band
  * >=75% camera photography by pairs; >=5% each screen/document/line-art;
    AI <=5%; TRAIN ids only (canonical last-digit split)
  * reused legs: native/no-resample only (safesyn PIL-Lanczos variants are
    excluded -> replaced by safesyn-regen from CLIC jpg originals; the
    leaders' codec_fit refs are clean-picker Lanczos -> excluded entirely)
"""
import csv, json, os, re, sys, collections, hashlib
import numpy as np
import pyarrow.parquet as pq

ROOT = '/mnt/v/output/zensim/joint-core-v1'
LOCAL = '/mnt/v/output/imazen-26-png-v3'
TRAIN_TSV = '/home/lilith/work/zen/imazen-26/manifests/train.tsv'
FEAT_PARQ = '/mnt/v/output/imazen-26-features/imazen26_features_2026-06-23.parquet'
SS_SRC = '/mnt/v/input/zensim/sources'
DEDUP = '/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables'
REC14 = '/var/tmp/zensim-validation-2026-09-14/baseline-recovery'
HDR_PURE = '/mnt/v/output/zensim/hdr944-leg-pure-2026-08-28/hdr_v3mix944_train_pure.parquet'
HDR_TSV = '/mnt/v/output/zensim/hdr944-leg/hdr944_features_all.tsv'
KONFIG = '/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/konfig_originsplit_train_944.parquet'
KONFIG_PAIRS = '/mnt/v/output/zensim/konfig944/build/konfig_pairs.tsv'

# class groups -> canonical train content_class prefixes
PHOTO = ['1000-', '1200-', '1400-', '1600-', '2000-', '2400-', '3000-', '3300-']
SCREEN = ['8000-', '8100-']
DOC = ['5000-', '5200-', '5300-', '6000-', '6800-']
LINEART = ['6600-', '7000-', '2200-']
AI = ['9000-', '9094-', '9226-']

# band -> rung ladder (long edge, px). mid carries the bulk.
RUNGS = {'mid': [384, 512, 640, 768, 896, 1024],
         'small': [192, 224, 256],
         'tiny': [64, 96, 128]}
QGRID = [3, 5, 8, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 85, 90, 95, 98]
CELLS_PER_REND = 32  # 4 codecs x 8 q
CODECS = ['zenjpeg-420-e2', 'zenwebp-m4', 'zenavif-s6', 'zenjxl-e7']

# pair quotas (post-correction design v3, T ~= 52.9k)
FRESH_QUOTA = {  # (band -> pairs) per class group
    'photo':   {'mid': 7900, 'small': 8200, 'tiny': 7100},
    'screen':  {'mid': 1000, 'small': 1100, 'tiny': 900},
    'doc':     {'mid': 1000, 'small': 1100, 'tiny': 900},
    'lineart': {'mid': 900,  'small': 1100, 'tiny': 1000},
    'ai':      {'mid': 500,  'small': 200,  'tiny': 100},
}
SAFESYN_REGEN_PAIRS = 4600  # all mid-band, CLIC camera sources
REUSED = {'cid22': 7500, 'human': 5100, 'hdr': 2500}


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
                             src_sha=r['sha256'], manifest_dims=(int(r['width']), int(r['height']))))
    return rows


def eval_side_ids():
    """imazen-26 origin ids present in any frozen *_development table —
    reference-level disjointness forbids them in the core. The clean-picker
    codec legs name refs `o_<id>.png.scaleWxH.png` where <id> IS the
    imazen-26 origin id (verified: corpus file o_1046.png ==
    train.tsv id 1046)."""
    ids = set()
    for leg in ['codec_development', 'safesyn_development',
                'cid22_development', 'human_development']:
        t = pq.read_table(f'{DEDUP}/{leg}.parquet', columns=['ref_basename']).to_pandas()
        for b in t['ref_basename'].unique():
            m = re.match(r'o_(\d+)\.png', str(b))
            if m:
                ids.add(m.group(1))
    return ids


def band_of(le):
    return 'tiny' if le <= 128 else 'small' if le <= 256 else 'mid' if le <= 1024 else 'big'


def origin_features(ids):
    """native-size feature row per origin id -> vector over descriptor cols."""
    t = pq.read_table(FEAT_PARQ).to_pandas()
    t = t[(t['split'] == 'train') & (t['size_class'] == 'native') & (t['crop_label'] == 'full')]
    feat_cols = [c for c in t.columns if '@' in c]
    t['id'] = t['image_path'].str.extract(r'/(\d+)_[^/]+\.png$')[0]
    t = t[t['id'].isin(ids)]
    return t.set_index('id')[feat_cols].astype(np.float32)


def strata(origins, feat_df, k):
    """kmeans strata assignment; ids missing features go to a 'cold' stratum."""
    ids = [o['id'] for o in origins]
    have = [i for i in ids if i in feat_df.index]
    X = feat_df.loc[have].values
    X = np.nan_to_num(np.log1p(np.clip(X, 0, None)))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=min(k, max(2, len(have))), n_init=4, random_state=17).fit(X)
    lab = dict(zip(have, km.labels_))
    return {i: lab.get(i, -1) for i in ids}


def plan_fresh(rows, feat_df):
    """Assign (origin, rung) renditions + (codec, q) cells per class group."""
    rends = []
    cells = []
    rng = np.random.RandomState(20260920)
    rid = 0
    for group, quotas in FRESH_QUOTA.items():
        origs = [o for o in rows if o['group'] == group]
        origs.sort(key=lambda o: o['id'])
        st = strata(origs, feat_df, k=16)
        for o in origs:
            o['stratum'] = st[o['id']]
        # renditions needed per band
        for band, pairs in quotas.items():
            n_rend = int(round(pairs / CELLS_PER_REND))
            # eligible origins: rung must fit source (no upscale)
            elig = [o for o in origs if min(max(RUNGS[band]), max(RUNGS[band])) <= max(o['w'], o['h'])
                    or band != 'tiny']  # tiny always fits
            elig = [o for o in origs if max(RUNGS[band]) <= max(o['w'], o['h']) or band == 'tiny']
            # stratum-round-robin ordering so no cluster dominates the band
            by_stratum = collections.defaultdict(list)
            for o in elig:
                by_stratum[o['stratum']].append(o)
            order = []
            strata_ids = sorted(by_stratum)
            i = 0
            while len(order) < len(elig):
                s = strata_ids[i % len(strata_ids)]
                if by_stratum[s]:
                    order.append(by_stratum[s].pop(0))
                i += 1
                if all(not v for v in by_stratum.values()):
                    break
            rungs = RUNGS[band]
            used_rungs = collections.defaultdict(set)  # src_id -> rungs taken in this band
            rot = 0
            for j in range(n_rend):
                o = order[j % len(order)]
                src_le = max(o['w'], o['h'])
                # first rung (rotated for coverage) that fits the source and is
                # not already assigned to this origin in this band
                rung = None
                for k in range(len(rungs)):
                    cand = rungs[(rot + k) % len(rungs)]
                    if cand <= src_le and cand not in used_rungs[o['id']]:
                        rung = cand
                        break
                if rung is None:
                    # origin exhausted in this band (or too small) — skip slot
                    continue
                used_rungs[o['id']].add(rung)
                rot += 1
                rname = f"{o['id']}.s{rung}"
                rends.append(dict(rendition=rname, src_id=o['id'], group=group,
                                  src_class=o['cc'], rung=rung, band=band,
                                  src_path=o['path'], src_sha256=o['src_sha'],
                                  src_w=o['w'], src_h=o['h'],
                                  kernel='zenresize-mitchell-sharpen0-srgb8'))
                # deterministic strided q rotation over the dense-below-60 grid
                off = (rid * 8) % len(QGRID)
                qs = sorted({QGRID[(off + j2 * 2) % len(QGRID)] for j2 in range(8)})
                for c in CODECS:
                    for q in qs:
                        cells.append(dict(rendition=rname, codec=c, q=q))
                rid += 1
    return rends, cells


def plan_safesyn_regen(feat_ok=True):
    """CLIC photo originals -> Mitchell mid-band rungs. fit-side bases only
    (frozen safesyn_development holds 156 disjoint bases)."""
    dev = pq.read_table(f'{DEDUP}/safesyn_development.parquet', columns=['ref_basename']).to_pandas()
    dev_bases = {re.sub(r'_(\d+x\d+|\d+sq)$', '', b) for b in dev['ref_basename']}
    fit = pq.read_table(f'{DEDUP}/safesyn_fit.parquet', columns=['ref_basename']).to_pandas()
    fit_bases = sorted({re.sub(r'_(\d+x\d+|\d+sq)$', '', b) for b in fit['ref_basename']})
    cand = [b for b in fit_bases if b not in dev_bases
            and os.path.exists(f'{SS_SRC}/{b}.jpg')]
    n_rend = SAFESYN_REGEN_PAIRS // CELLS_PER_REND  # ~144
    rungs = [512, 640, 768, 1024]
    per_src = 3
    srcs = cand[: int(np.ceil(n_rend / per_src))]
    rends, cells = [], []
    rid = 0
    for i, b in enumerate(srcs):
        my = rungs[i % len(rungs):] + rungs[: i % len(rungs)]
        for r in my[:per_src]:
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


def kmeans_pick(df_feats, n, seed=17, k=500):
    """centroid-nearest pick preserving singleton clusters."""
    from sklearn.cluster import KMeans
    X = df_feats.astype(np.float32).values
    X = np.nan_to_num(X)
    k = min(k, len(X))
    km = KMeans(n_clusters=k, n_init=2, random_state=seed).fit(X)
    lab = km.labels_
    pick = []
    per = int(np.ceil(n / k))
    for c in range(k):
        idx = np.where(lab == c)[0]
        if len(idx) == 0:
            continue
        d = ((X[idx] - km.cluster_centers_[c]) ** 2).sum(1)
        order = idx[np.argsort(d, kind='stable')]
        pick.extend(order[:max(1, per)])
    # dedup + trim/pad to n deterministically
    seen = list(dict.fromkeys(pick))
    if len(seen) > n:
        seen = seen[:n]
    elif len(seen) < n:
        rest = np.setdiff1d(np.arange(len(X)), np.array(seen))
        c = km.predict(X[rest]) if len(rest) else []
        seen.extend(rest[np.argsort(-((X[rest] ** 2).sum(1)))][: n - len(seen)])
    return sorted(seen)


def pairs_tsv_map(path):
    """row_id -> (ref_path, dist_path)"""
    m = {}
    with open(path) as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            m[int(row[3])] = (row[0], row[1])
    return m


def ref_order_csv(path):
    """ref_basename -> [row_id in csv order]"""
    out = collections.defaultdict(list)
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        i_ref = hdr.index('ref_basename'); i_row = hdr.index('row_id')
        for row in r:
            out[row[i_ref]].append(int(row[i_row]))
    return out


def plan_reused():
    """kmeans subsample of native-kernel legs + positional join to dist paths."""
    out_rows = {}   # leg -> parquet row indices kept
    pairs = []
    # --- cid22 (native refs, dataset dists) ---
    df = pq.read_table(f'{DEDUP}/cid22_fit.parquet').to_pandas()
    fcols = [f'f{i}' for i in range(228)]
    sel = kmeans_pick(df[fcols], REUSED['cid22'])
    out_rows['cid22'] = sel
    cmap = ref_order_csv(f'{REC14}/cid22-train944.csv')
    pmap = pairs_tsv_map(f'{REC14}/cid22-train-pairs.tsv')
    cid_pos = collections.defaultdict(int)
    for i in sel:
        rb = df['ref_basename'].iat[i] + '.png'
        pos = cid_pos[rb]; cid_pos[rb] += 1
        rid_ = cmap[rb][pos] if pos < len(cmap[rb]) else None
        rp, dp = pmap.get(rid_, ('', ''))
        pairs.append(dict(leg='cid22', ref_path=rp, dist_path=dp,
                          human_score=df['human_score'].iat[i], kernel='native,no-resample',
                          band='mid', codec=codec_of(dp), q=q_of(dp), ref_basename=rb))
    # --- human kadid+tid (native) ---
    # tables/human_fit row order == pairs-tsv order per (family, ref) with a few
    # pre-dedup drops; kadid: table == (1 - dmos)*100, tid: table == mos*100.
    # Align by score sequence within each ref group.
    df = pq.read_table(f'{DEDUP}/human_fit.parquet').to_pandas()
    sel = kmeans_pick(df[fcols], REUSED['human'])
    out_rows['human'] = sel
    hpairs = collections.defaultdict(list)
    for pf in ['/mnt/v/dataset/kadid10k/kadid_pairs_ab.tsv',
               '/mnt/v/dataset/tid2013/tid_pairs_ab.tsv']:
        with open(pf) as f:
            for r in csv.DictReader(f, delimiter='\t'):
                fam = 'kadid' if 'kadid' in r['ref_path'] else 'tid'
                rb = os.path.basename(r['ref_path'])[:-4]
                key = (fam, rb.upper() if fam == 'tid' else rb)
                # kadid pairs tsv stores DMOS (higher=worse); tid stores MOS
                # (higher=better). table human_score = quality on 0-100.
                hs = float(r['human_score'])
                qv = (1.0 - hs) * 100.0 if fam == 'kadid' else hs * 100.0
                hpairs[key].append((r['ref_path'], r['dist_path'], qv))
    hptr = collections.defaultdict(int)
    hmiss = 0
    for i in sel:
        row = df.iloc[i]
        fam = row['source_family'].split(':')[1]
        rb = row['source_family'].split(':')[2]
        key = (fam, rb)
        lst = hpairs[key]
        target = float(row['human_score'])
        j = hptr[key]
        while j < len(lst) and abs(lst[j][2] - target) > 1e-4:
            j += 1
        if j >= len(lst):
            hmiss += 1
            rp = dp = ''
        else:
            hptr[key] = j + 1
            rp, dp, _ = lst[j]
        pairs.append(dict(leg='human', ref_path=rp, dist_path=dp,
                          human_score=target, kernel='native,no-resample',
                          band='mid', codec=codec_of(dp), q=q_of(dp),
                          ref_basename=row['source_family']))
    if hmiss:
        print(f'human join misses: {hmiss}')
    # --- konfig (all 327, native) ---
    # konfig_944.parquet row order == konfig_pairs.tsv order per ref (verified:
    # q_jnd sequences identical for all 6 train refs).
    df = pq.read_table(KONFIG).to_pandas()
    out_rows['konfig'] = list(range(len(df)))
    kpaths = collections.defaultdict(list)
    with open(f'{KONFIG_PAIRS}') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            kpaths[r['source'] + '_' + r['part']].append((r['ref_path'], r['dist_path']))
    kptr = collections.defaultdict(int)
    for i in range(len(df)):
        rb = df['ref_basename'].iat[i]
        rp, dp = kpaths[rb][kptr[rb]]
        kptr[rb] += 1
        pairs.append(dict(leg='konfig', ref_path=rp, dist_path=dp,
                          human_score=float(df['human_score'].iat[i]) * 100.0,
                          kernel='native,no-resample', band='mid', codec='jnd-levels',
                          q='', ref_basename=rb))
    # --- hdr <=1024 subset (mitchell+sharpen10 linearPQ recorded kernel) ---
    df = pq.read_table(HDR_PURE).to_pandas()
    le = df['ref_basename'].str.extract(r'\.scale(\d+)x(\d+)\.').astype(int).max(axis=1)
    df['band'] = le.map(band_of)
    el = df[df['band'] != 'big']
    # take all tiny+small, stratified mid fill to quota
    ts = el[el['band'].isin(['tiny', 'small'])]
    mid = el[el['band'] == 'mid']
    need = REUSED['hdr'] - len(ts)
    mid_sel = kmeans_pick(mid[[f'f{i}' for i in range(228)]], max(0, need)) if need else []
    sel_idx = list(ts.index) + [mid.index[i] for i in mid_sel]
    out_rows['hdr'] = sorted(el.index.get_indexer_for(sel_idx))
    # join dist via hdr944_features_all.tsv: per ref, nearest tsv row in
    # feature space identifies the exact dist_basename (extraction source).
    htsv = collections.defaultdict(list)  # ref_basename -> [(fvec, dist_base, q)]
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
    for i in sel_idx:
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
        print(f'hdr join misses: {hmiss}')
    return out_rows, pairs


def codec_of(dist_path):
    m = re.search(r'/([^/]+)/[^/]+$', dist_path)
    return m.group(1) if m else ''


def q_of(dist_path):
    m = re.search(r'/q(\d+)\.', dist_path)
    return int(m.group(1)) if m else ''


def main():
    os.makedirs(f'{ROOT}/plan', exist_ok=True)
    rows = load_train()
    dev_ids = eval_side_ids()
    rows = [o for o in rows if o['id'] not in dev_ids]
    print('eligible train origins:', len(rows),
          dict(collections.Counter(o['group'] for o in rows)),
          f'(excluded {len(dev_ids)} frozen-dev origin ids)')
    ids = [o['id'] for o in rows]
    fdf = origin_features(set(ids))
    print('origins with native feature rows:', len(fdf))

    rends, cells = plan_fresh(rows, fdf)
    sr, sc = plan_safesyn_regen()
    all_rends = rends + sr
    all_cells = cells + sc
    print('renditions:', len(all_rends), 'cells:', len(all_cells))
    bc = collections.Counter(r['band'] for r in all_rends)
    print('rendition bands:', dict(bc), 'pair-est', {k: v * CELLS_PER_REND for k, v in bc.items()})

    with open(f'{ROOT}/plan/renditions_fresh.tsv', 'w') as f:
        f.write('rendition\tsrc_id\tgroup\tsrc_class\trung\tband\tsrc_path\tsrc_sha256\tsrc_w\tsrc_h\tkernel\n')
        for r in all_rends:
            f.write('\t'.join(str(r[k]) for k in
                              ['rendition', 'src_id', 'group', 'src_class', 'rung', 'band',
                               'src_path', 'src_sha256', 'src_w', 'src_h', 'kernel']) + '\n')
    with open(f'{ROOT}/plan/cells_fresh.tsv', 'w') as f:
        f.write('rendition\tcodec\tq\n')
        for c in all_cells:
            f.write(f"{c['rendition']}\t{c['codec']}\t{c['q']}\n")

    rows_sel, pairs = plan_reused()
    with open(f'{ROOT}/plan/reused_rows.json', 'w') as f:
        json.dump({k: [int(i) for i in v] for k, v in rows_sel.items()}, f)
    with open(f'{ROOT}/plan/reused_pairs.tsv', 'w') as f:
        f.write('leg\tref_path\tdist_path\thuman_score\tkernel\tband\tcodec\tq\tref_basename\n')
        for p in pairs:
            f.write('\t'.join(str(p[k]) for k in
                              ['leg', 'ref_path', 'dist_path', 'human_score', 'kernel',
                               'band', 'codec', 'q', 'ref_basename']) + '\n')
    print('reused pairs:', len(pairs), dict(collections.Counter(p['leg'] for p in pairs)))


if __name__ == '__main__':
    main()
