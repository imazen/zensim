#!/usr/bin/env python3
"""transplant lane: parse fit logs -> results.json + the markdown record.

Metric: `MLP train: best validation mean SROCC` (geomean3 over the four
dev legs) — the v1 gate metric, unchanged.

Analyses:
  step0: per size, base mean, perm30 mean, cost = base - perm30 (positive =
         scrambling those columns hurt), and the seed spread of base (the
         noise floor). Plan gate: the destruction cost must resolve ABOVE the
         noise floor — i.e. a known 30-column perturbation must be detectable
         against seed noise. v1 failed because +0.0021 sat under the ~0.0033
         floor; growing the core shrinks the floor until the cost clears it.
         verdict: clear / resolving / below_floor / buried, plus raw margin.
  X4/X5/X7: per-seed paired diff (arm - base@s105k on the same seed),
         sign count, mean diff; and the same against the arm's permuted
         control (a gain that does not beat its control is not a gain).
"""
import json, os, re, glob, collections
import numpy as np

OUT = '/mnt/v/output/zensim/dvifm-transplant-2026-09-20'
SEEDS = [17101, 17103, 17107, 17111, 17113]
RE = re.compile(r'best validation mean SROCC = ([0-9.]+)')


def load_scores():
    scores = {}
    for f in glob.glob(f'{OUT}/fits/logs/*.train.log'):
        rid = os.path.basename(f)[:-10]
        txt = open(f).read()
        m = RE.findall(txt)
        scores[rid] = float(m[-1]) if m else None
    return scores


def arm_scores(scores, arm):
    return {s: scores.get(f'{arm}_s{s}') for s in SEEDS}


def paired(scores, a, b):
    """per-seed diffs a-b where both exist."""
    A, B = arm_scores(scores, a), arm_scores(scores, b)
    d = {s: (A[s] - B[s]) for s in SEEDS
         if A[s] is not None and B[s] is not None}
    return d


def stats(d):
    if not d:
        return None
    v = list(d.values())
    return {'n': len(v), 'mean': float(np.mean(v)),
            'min': float(np.min(v)), 'max': float(np.max(v)),
            'pos': sum(1 for x in v if x > 0),
            'neg': sum(1 for x in v if x < 0),
            'zero': sum(1 for x in v if x == 0),
            'per_seed': {str(s): d[s] for s in sorted(d)}}


def main():
    scores = load_scores()
    res = {'seeds': SEEDS, 'scores': scores, 'step0': {}, 'x4': {}, 'x5': {},
           'x7': {}}
    def gate(base_mean, spread, cost):
        """Plan criterion (v1-doc semantics): the cost of destroying the 30
        real inputs (real-perm) must be DETECTABLE above seed noise — i.e.
        reliably positive across the paired seeds. Detection uses the paired
        design's own noise (per-seed diff std + sign consistency), which is
        the correct null for a same-seed comparison; the marginal base seed
        *range* is also reported (v1 convention) but is a noisy 5-seed floor.
        verdict: 'clear' if every seed is positive and paired_t>=3,
        'resolving' if all-positive and paired_t>=2, 'below_floor' when the
        cost is not consistently positive (v1: 4/5, one negative), 'buried'
        when the mean cost is <=0."""
        if base_mean is None or cost is None or not cost.get('per_seed'):
            return None
        diffs = list(cost['per_seed'].values())
        n = len(diffs)
        cm = cost['mean']
        sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        t = cm / (sd / np.sqrt(n)) if sd > 0 else (np.inf if cm > 0 else 0.0)
        ratio = cm / spread if spread else None
        verdict = ('buried' if cm <= 0 else
                   'clear' if cost['neg'] == 0 and t >= 3 else
                   'resolving' if cost['neg'] == 0 and t >= 2 else
                   'below_floor')
        return {'floor_seed_spread': spread, 'cost_mean': cm,
                'cost_over_floor': ratio,
                'paired_diff_std': sd, 'paired_t': t,
                'pos_seeds': cost['pos'], 'neg_seeds': cost['neg'],
                'verdict': verdict,
                'pass': verdict in ('clear', 'resolving')}

    # v1's own 53k measurements (the same rows, cited for the curve)
    v1 = {'base': {17101: .9696, 17103: .9715, 17107: .9711, 17111: .9729,
                   17113: .9718},
          'perm30': {17101: .9701, 17103: .9705, 17107: .9687, 17111: .9700,
                     17113: .9670}}
    vd = {s: v1['base'][s] - v1['perm30'][s] for s in v1['base']}
    v1_cost = stats(vd)
    v1_bmean = float(np.mean(list(v1['base'].values())))
    v1_bspread = float(np.ptp(list(v1['base'].values())))
    res['step0']['s53k_v1'] = {
        'base_mean': v1_bmean,
        'perm_mean': float(np.mean(list(v1['perm30'].values()))),
        'cost': v1_cost,
        'cost_mean': float(np.mean(list(vd.values()))),
        'base_spread': v1_bspread,
        'gate': gate(v1_bmean, v1_bspread, v1_cost),
        'note': 'v1 numbers cited verbatim — identical rows/dev/recipe'}

    for size in ('s61k', 's83k', 's105k'):
        b, p = arm_scores(scores, f'base@{size}'), arm_scores(scores, f'perm30@{size}')
        d = paired(scores, f'base@{size}', f'perm30@{size}')
        bv = [x for x in b.values() if x is not None]
        bmean = float(np.mean(bv)) if bv else None
        bspread = float(np.ptp(bv)) if len(bv) > 1 else None
        cost = stats(d)
        res['step0'][size] = {
            'base': b, 'perm30': p,
            'base_mean': bmean,
            'base_spread': bspread,
            'cost': cost,
            'gate': gate(bmean, bspread, cost),
        }
    # X arms vs base@s105k; each vs its permuted control
    for exp, arms in (
            ('x4', ('x4', 'x4perm')),
            ('x5', ('x5', 'x5perm')),
            ('x7', ('x7t1', 'x7t1p', 'x7t2', 'x7t2p',
                    'x7t3fix', 'x7t3max', 'x7t3maxp'))):
        for arm in arms:
            res[exp][arm] = {
                'scores': arm_scores(scores, arm),
                'vs_base': stats(paired(scores, arm, 'base@s105k'))}
    # permuted-control contrasts
    res['x4']['x4_vs_x4perm'] = stats(paired(scores, 'x4', 'x4perm'))
    res['x5']['x5_vs_x5perm'] = stats(paired(scores, 'x5', 'x5perm'))
    for t in ('x7t1', 'x7t2', 'x7t3max'):
        res['x7'][f'{t}_vs_{t}p'] = stats(paired(scores, t, f'{t}p'))
    res['x7']['x7t3max_vs_x7t3fix'] = stats(
        paired(scores, 'x7t3max', 'x7t3fix'))
    os.makedirs(f'{OUT}/reports', exist_ok=True)
    json.dump(res, open(f'{OUT}/reports/fit_results.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'scores'},
                     indent=1)[:6000])


if __name__ == '__main__':
    main()
