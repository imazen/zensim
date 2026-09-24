#!/usr/bin/env python3
"""gmsd lane: the `peer_gmsd` board row through the EXISTING owner —
`scripts/v_next/build_peer_fullevals.py::build_admitted` (the admitted,
hash-bound `zensim-peer-eval-v1` path that produced `peer_ssim2_mt914`).

Writes per-corpus TSVs (`ref_path dist_path origin human_score gmsd
neg_gmsd`) in the admitted row order, a v1 manifest binding their hashes to
the Rev3 qualification ADMISSION.json, and runs build_admitted into
`--out-dir`. EVAL-role corpora only (KADID SELECT, KonFiG origin-val); the
row carries no TEST corpus, so no public-test exposure record is involved.
GMSD is a distance: the row's metric column is `neg_gmsd` (quality-oriented,
as the legacy peer rows negate butteraugli), recorded in the note.
"""
import argparse, csv, hashlib, json, os, sys
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'v_next'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'lib'))
from build_peer_fullevals import build_admitted  # noqa: E402

Q = '/home/lilith/work/zensim-validation-2026-09-14/rev3-qualification'
AXIS = {'kadid': 'mos', 'konfig': 'mos'}


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores-dir', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--scorer-bin', required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    corpora = {}
    for c in AXIS:
        rows = list(csv.DictReader(open(f'{Q}/ssim2-{c}.tsv'), delimiter='\t'))
        t = pq.read_table(f'{a.scores_dir}/gmsd-{c}.parquet').to_pylist()
        col = next(k for k in t[0] if k.startswith('gmsd'))
        by = {json.loads(r['knob_tuple_json'])['row']: r[col] for r in t}
        p = f'{a.work}/gmsd-{c}.tsv'
        with open(p, 'w') as f:
            f.write('ref_path\tdist_path\torigin\thuman_score\tgmsd\tneg_gmsd\n')
            for i, r in enumerate(rows):
                g = by[i]
                f.write(f"{r['ref_path']}\t{r['dist_path']}\t{r['origin']}\t{r['human_score']}\t{g!r}\t{-g!r}\n")
        corpora[c] = dict(path=p, sha256=sha(p), role='eval', rows=len(rows),
                          metric_column='neg_gmsd', target_column='human_score', axis=AXIS[c])
    spec = {
        'schema': 'zensim-peer-eval-v1', 'role': 'eval', 'name': 'peer_gmsd',
        'note': (f'GMSD (Xue et al. 2014) via the zenmetrics gmsd crate (libgmsd port; '
                 f'column {col}), CPU, on the admitted Rev3-qualification EVAL pixels. '
                 'Distance negated to quality orientation (neg_gmsd). Parameter-free: '
                 'nothing was fitted or selected on these rows.'),
        'admission': {
            'path': f'{Q}/ADMISSION.json', 'sha256': sha(f'{Q}/ADMISSION.json'),
            'scorer_binary': a.scorer_bin, 'scorer_sha256': sha(a.scorer_bin),
        },
        'corpora': corpora,
    }
    mp = f'{a.work}/PEER_GMSD.json'
    json.dump(spec, open(mp, 'w'), indent=1)
    build_admitted(mp, a.out_dir)


if __name__ == '__main__':
    main()
