#!/usr/bin/env python3
"""gmsd lane, Part 2: GMSD next to fast-ssim2, butteraugli, zensim B, D and
the frozen Rev3 rich ensemble on the ADMITTED EVAL populations only.

Populations (the September 14 Rev3 qualification's admitted EVAL views —
`~/work/zensim-validation-2026-09-14/rev3-qualification/ADMISSION.json`):

* KADID SELECT — 25 references, 3,125 pairs (DATA_SPLITS §8).
* KonFiG origin-validation — SRC01/03/31/45, 436 pairs (§8; SSIMULACRA2 was
  tuned on KonFiG, so it is not independent ssim2-superiority evidence).

NOT read, by rule: KonJND (the lane preamble lists "KonJND val" among the
holdouts; the JPEG SELECT view is the only KonJND eval surface), CID22 (B half
sealed; the 49-ref human set is TEST), AIC-3/4, AIC2026, KonFiG test, KADID
terminal refs, any secret holdout.

Arms and where their per-pair numbers come from:

* gmsd, butteraugli (max + pnorm3): scored here by `zenmetrics score-pairs`
  (CPU) on the admitted pixels — `--scores` points at the two parquets.
* fast-ssim2: the matched CPU SSIMULACRA2 column already in the admitted
  TSVs (`ssim2`, the `peer_ssim2_mt914` row's own input).
* zensim B / D: `MT914_matched_{B,D}.full.json` per_pair preds (frozen bakes,
  scored on these exact admitted rows by `bake_verdict`).
* Rev3 rich: `R915_basic228_h128_ens5.full.json` per_pair preds.
Row alignment of every stored vector is verified by exact equality of its
stored target vector with the TSV's `human_score` column, in order.

Distances (gmsd, butteraugli) are negated so every arm is quality-oriented;
`srocc_signed` is then positive for a correct metric. Every statistic comes
from `zen_stats` -> the Rust `panel` binary (zenstats); nothing is computed
here.

Aggregations, per corpus:
* global  — all pairs in one bucket (the primary number).
* per-distortion — mean of within-family SROCC (KADID's 25 types; KonFiG's 7
  families). Secondary.
* per-source — mean of within-reference SROCC. Secondary; within-ladder
  numbers are trivially high for every metric.

Usage: peer_eval.py --scores-dir DIR --out-json F --out-md F
"""
import argparse, csv, hashlib, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'lib'))
from zen_stats import panel_batch  # noqa: E402

Q = '/home/lilith/work/zensim-validation-2026-09-14/rev3-qualification'
R915 = ('/mnt/data/dataset/output/zensim/reports/recovery-completion-2026-09-15/'
        'assessment/R915_basic228_h128_ens5.full.json')
TSV = {'kadid': f'{Q}/ssim2-kadid.tsv', 'konfig': f'{Q}/ssim2-konfig.tsv'}
TSV_SHA = {  # from rev3-qualification/PEER.json
    'kadid': '14c8742e6350cd170f91394c9285475f9121fd209bbcf90ec46b519d5093e973',
    'konfig': 'c8a0239ccd825c0dfd4fcd821dcd21242edb22b3239d2c17c521da5e8818e583',
}
STORED = {
    'zensim_B': f'{Q}/verdicts/MT914_matched_B.full.json',
    'zensim_D': f'{Q}/verdicts/MT914_matched_D.full.json',
    'rev3_rich_basic228_ens5': R915,
}


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def family(corpus, dist):
    if corpus == 'kadid':
        return 'kadid_type_' + os.path.basename(dist).split('_')[1]
    return dist.split('/')[-2]


def load_scores(scores_dir, metric_prefix):
    """score-pairs parquet -> {row: {col: value}} keyed by knob_tuple_json."""
    import pyarrow.parquet as pq
    out = {}
    for corpus in TSV:
        t = pq.read_table(os.path.join(scores_dir, f'{metric_prefix}-{corpus}.parquet')).to_pylist()
        m = {}
        for r in t:
            k = json.loads(r['knob_tuple_json'])['row']
            m[k] = {c: v for c, v in r.items() if c not in ('image_path', 'codec', 'q', 'knob_tuple_json')}
        out[corpus] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores-dir', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', required=True)
    a = ap.parse_args()

    reads = []
    rows = {}
    for c, p in TSV.items():
        h = sha(p)
        assert h == TSV_SHA[c], f'{c}: admitted TSV hash changed'
        rows[c] = list(csv.DictReader(open(p), delimiter='\t'))
        reads.append(dict(corpus=c, path=p, sha256=h, rows=len(rows[c]), role='eval',
                          what='admitted EVAL view: pixels + human_score'))

    preds = {c: {} for c in TSV}
    notes = {}
    for c in TSV:
        preds[c]['fast_ssim2'] = [float(r['ssim2']) for r in rows[c]]
    gm = load_scores(a.scores_dir, 'gmsd')
    bu = load_scores(a.scores_dir, 'butteraugli')
    for c in TSV:
        n = len(rows[c])
        gcol = next(k for k in gm[c][0] if k.startswith('gmsd'))
        preds[c]['gmsd'] = [-gm[c][i][gcol] for i in range(n)]
        notes['gmsd'] = f'column {gcol}, negated (distance)'
        preds[c]['butteraugli_pnorm3'] = [-bu[c][i]['butteraugli_pnorm3'] for i in range(n)]
        preds[c]['butteraugli_max'] = [-bu[c][i]['butteraugli_max'] for i in range(n)]
    notes['butteraugli'] = 'CPU butteraugli via zenmetrics score-pairs, negated (distance)'
    for arm, path in STORED.items():
        d = json.load(open(path))
        reads.append(dict(arm=arm, path=path, sha256=sha(path),
                          what='stored frozen per-pair predictions (no pixels re-read)',
                          bake_sha256=d.get('bake_sha256')))
        for c in TSV:
            pp = d['per_pair'][c]
            tgt = [float(r['human_score']) for r in rows[c]]
            assert len(pp['pred']) == len(tgt)
            assert all(abs(x - y) < 1e-9 for x, y in zip(pp['mos'], tgt)), f'{arm}/{c}: row order'
            preds[c][arm] = [float(v) for v in pp['pred']]

    arms = ['gmsd', 'fast_ssim2', 'butteraugli_pnorm3', 'butteraugli_max',
            'zensim_B', 'zensim_D', 'rev3_rich_basic228_ens5']
    jobs = []
    keys = []
    for c in TSV:
        tgt = [float(r['human_score']) for r in rows[c]]
        fam = [family(c, r['dist_path']) for r in rows[c]]
        src = [r['origin'] for r in rows[c]]
        for arm in arms:
            p = preds[c][arm]
            jobs.append((f'{c}|{arm}|global', p, tgt)); keys.append((c, arm, 'global', None))
            for gname, gkey in (('family', fam), ('source', src)):
                idx = defaultdict(list)
                for i, g in enumerate(gkey):
                    idx[g].append(i)
                for g, ii in sorted(idx.items()):
                    jobs.append((f'{c}|{arm}|{gname}|{g}', [p[i] for i in ii], [tgt[i] for i in ii]))
                    keys.append((c, arm, gname, g))
    res = panel_batch(jobs)
    table = defaultdict(lambda: defaultdict(dict))
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (c, arm, kind, g), st in zip(keys, res):
        if kind == 'global':
            table[c][arm] = {k: st[k] for k in ('srocc', 'srocc_signed', 'krocc', 'plcc', 'n')}
        else:
            groups[c][arm][kind].append((g, st['srocc_signed'], st['n']))
    for c in table:
        for arm in table[c]:
            for kind in ('family', 'source'):
                v = [s for _, s, _ in groups[c][arm][kind]]
                table[c][arm][f'{kind}_mean_srocc'] = sum(v) / len(v)
                table[c][arm][f'{kind}_min_srocc'] = min(v)
                table[c][arm][f'{kind}_n_groups'] = len(v)
    out = dict(schema='gmsd-lane-peer-eval-v1', date='2026-09-22',
               populations={c: dict(n=len(rows[c]),
                                    resolution='native full resolution (KADID 512x384; KonFiG as distributed), no crop')
                            for c in rows},
               reads=reads, notes=notes, arms=arms,
               table={c: dict(v) for c, v in table.items()},
               per_group={c: {arm: {k: v for k, v in d.items()} for arm, d in g.items()}
                          for c, g in groups.items()})
    json.dump(out, open(a.out_json, 'w'), indent=1)
    lines = []
    for c in table:
        lines.append(f'### {c} (n={len(rows[c])}, full resolution)\n')
        lines.append('| arm | global SROCC | global KROCC | PLCC | per-distortion mean SROCC (min) | per-source mean SROCC (min) |')
        lines.append('|---|---|---|---|---|---|')
        for arm in arms:
            t = table[c][arm]
            lines.append(f"| {arm} | {t['srocc_signed']:.4f} | {t['krocc']:.4f} | {t['plcc']:.4f} | "
                         f"{t['family_mean_srocc']:.4f} ({t['family_min_srocc']:.3f}) | "
                         f"{t['source_mean_srocc']:.4f} ({t['source_min_srocc']:.3f}) |")
        lines.append('')
    open(a.out_md, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
