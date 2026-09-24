#!/usr/bin/env python3
"""gmsd lane: parameter-free GMSD standalone on the four joint-core-v2
DEVELOPMENT legs (TRAIN-role internal development tables), next to the
trained 228-feature zensim MLP (`G`, five seeds) on the same rows.

GMSD scores: `zenmetrics score-pairs --metric gmsd` over the dev TSVs, in
TSV order (== the frozen dev-table order the trainer reads). Targets: the
frozen dev tables' `human_score` (checked equal to the TSV's). GMSD is a
distance, so it is negated before ranking.

Two SROCCs per leg, both from the `panel` owner:
* `subsampled` — the trainer's exact statistic
  (`zensim_validate::panel::compute_light_panel_subsampled`: stride
  decimation to <= 4096 rows), directly comparable with G's printed
  per-leg `srocc` at its best epoch;
* `full` — every row.
"""
import csv, json, os, sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'lib'))
from zen_stats import panel_batch  # noqa: E402

L = '/var/tmp/gmsd-lane'
DEDUP = '/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables'
VDIR = '/mnt/v/output/zensim/dvifm-verdict-2026-09-20/pairs'
LEGS = ('safesyn', 'cid22', 'human', 'codec')
CAP = 4096  # LIGHT_PANEL_PWRC_SUBSAMPLE_CAP


def stride_sub(v, n):
    stride = -(-n // CAP) if n > CAP else 1
    return v[::stride]


def main():
    fr = json.load(open(f'{L}/reports/fit_results.json'))
    jobs, meta = [], []
    for leg in LEGS:
        tsv = list(csv.DictReader(open(f'{VDIR}/{leg}_dev.tsv'), delimiter='\t'))
        frozen = pq.read_table(f'{DEDUP}/{leg}_development.parquet', columns=['human_score']).to_pandas()
        tgt = frozen['human_score'].to_numpy(np.float64)
        # The frozen dev tables carry the TSV targets x100; ranks are what
        # matter, so require an exact affine identity, not equal values.
        t_tsv = np.array([float(r['human_score']) for r in tsv])
        assert np.allclose(100.0 * t_tsv, tgt, rtol=0, atol=1e-6), leg
        t = pq.read_table(f'{L}/eval/scores/gmsd-dev-{leg}.parquet').to_pylist()
        col = next(k for k in t[0] if k.startswith('gmsd'))
        by = {json.loads(r['knob_tuple_json'])['row']: r[col] for r in t}
        assert len(by) == len(tsv), leg
        pred = -np.array([by[i] for i in range(len(tsv))])
        n = len(tsv)
        jobs.append((f'{leg}|full', pred.tolist(), tgt.tolist())); meta.append((leg, 'full', n))
        jobs.append((f'{leg}|sub', stride_sub(pred, n).tolist(), stride_sub(tgt, n).tolist()))
        meta.append((leg, 'subsampled', len(stride_sub(pred, n))))
    res = panel_batch(jobs)
    out = {'gmsd': {}, 'G': {}}
    for (leg, kind, n), st in zip(meta, res):
        out['gmsd'].setdefault(leg, {})[kind] = dict(srocc=st['srocc_signed'], krocc=st['krocc'], n=n)
    for leg in LEGS:
        vals = [fr['dev_legs'][f'G_s{s}'][leg] for s in fr['seeds'] if f'G_s{s}' in fr['dev_legs']]
        out['G'][leg] = dict(mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)),
                             n_seeds=len(vals))
    json.dump(out, open(f'{L}/reports/dev_standalone.json', 'w'), indent=1)
    print('| dev leg | n | GMSD SROCC (trainer subsample) | GMSD SROCC (all rows) | zensim G 228-MLP SROCC, 5 seeds mean [min, max] |')
    print('|---|---|---|---|---|')
    for leg in LEGS:
        g, z = out['gmsd'][leg], out['G'][leg]
        print(f"| {leg} | {g['full']['n']} | {g['subsampled']['srocc']:.4f} | {g['full']['srocc']:.4f} | "
              f"{z['mean']:.4f} [{z['min']:.4f}, {z['max']:.4f}] |")


if __name__ == '__main__':
    main()
