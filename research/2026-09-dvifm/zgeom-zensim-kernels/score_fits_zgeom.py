#!/usr/bin/env python3
"""zgeom lane: parse fit logs -> reports/fit_results.json.

Metric: `MLP train: best validation mean SROCC` (geomean3 over the four
dev legs) — the v1 gate metric, unchanged.

Z2 contrasts (per-seed paired diffs):
  z2bin121  - z2box2   kernel gain vs production box2
  z2bin1331 - z2box2
Z1 contrasts:
  z1gate - z2box2      replacement pooling vs canonical pooling
  z1gate - z1gateperm  mechanism vs its permuted control
  z1max  - z2box2      ungated block-peak decomposition
  z1max  - z1maxperm
"""
import json, os, re, glob
import numpy as np

OUT = '/mnt/v/output/zensim/zgeom-2026-09-21'
SEEDS = [17101, 17103, 17107, 17111, 17113]
RE = re.compile(r'best validation mean SROCC = ([0-9.]+)')


LEG_RE = {
    leg: re.compile(rf'{leg}_development: srocc=([0-9.]+)')
    for leg in ('safesyn', 'cid22', 'human', 'codec')}
BEST_EPOCH_RE = re.compile(r'epoch (\d+) .*?best=([0-9.]+)')


def load_scores():
    """geomean best + per-leg dev srocc AT the best-epoch line."""
    scores, legs = {}, {}
    for f in glob.glob(f'{OUT}/fits/logs*/*.log') + \
            glob.glob(f'{OUT}/fits/*.log'):
        rid = os.path.basename(f)
        rid = rid[:-10] if rid.endswith('.train.log') else rid[:-4]
        txt = open(f).read()
        m = RE.findall(txt)
        if not m:
            continue
        best = float(m[-1])
        scores[rid] = best
        # the epoch line that first prints best=<best> carries the
        # per-leg values at that epoch (best only improves on the line
        # where it is set)
        for line in txt.splitlines():
            bm = BEST_EPOCH_RE.search(line)
            if bm and float(bm.group(2)) == best:
                legs[rid] = {leg: float(LEG_RE[leg].search(line).group(1))
                             for leg in LEG_RE
                             if LEG_RE[leg].search(line)}
                break
    return scores, legs


def arm_scores(scores, arm):
    return {s: scores.get(f'{arm}_s{s}') for s in SEEDS}


def paired(scores, a, b):
    A, B = arm_scores(scores, a), arm_scores(scores, b)
    return {s: (A[s] - B[s]) for s in SEEDS
            if A[s] is not None and B[s] is not None}


def stats(d):
    if not d:
        return None
    v = list(d.values())
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    mean = float(np.mean(v))
    t = mean / (sd / np.sqrt(len(v))) if sd > 0 else \
        (np.inf if mean > 0 else 0.0)
    return {'n': len(v), 'mean': mean,
            'min': float(np.min(v)), 'max': float(np.max(v)),
            'pos': sum(1 for x in v if x > 0),
            'neg': sum(1 for x in v if x < 0),
            'paired_t': float(t),
            'per_seed': {str(s): d[s] for s in sorted(d)}}


def main():
    scores, legs = load_scores()
    res = {'seeds': SEEDS, 'scores': scores, 'dev_legs': legs,
           'z2': {}, 'z1': {}}
    for arm in ('z2box2', 'z2bin121', 'z2bin1331'):
        res['z2'][arm] = {'scores': arm_scores(scores, arm),
                          'dev_legs': {s: legs.get(f'{arm}_s{s}')
                                       for s in SEEDS}}
        if arm != 'z2box2':
            res['z2'][f'{arm}_vs_box2'] = stats(
                paired(scores, arm, 'z2box2'))
    res['z2']['bin1331_vs_bin121'] = stats(
        paired(scores, 'z2bin1331', 'z2bin121'))
    for arm in ('z1gate', 'z1max', 'z1gateperm', 'z1maxperm'):
        res['z1'][arm] = {'scores': arm_scores(scores, arm)}
    res['z1']['z1gate_vs_box2'] = stats(paired(scores, 'z1gate', 'z2box2'))
    res['z1']['z1gate_vs_gateperm'] = stats(
        paired(scores, 'z1gate', 'z1gateperm'))
    res['z1']['z1max_vs_box2'] = stats(paired(scores, 'z1max', 'z2box2'))
    res['z1']['z1max_vs_maxperm'] = stats(
        paired(scores, 'z1max', 'z1maxperm'))
    res['z1']['z1gate_vs_z1max'] = stats(paired(scores, 'z1gate', 'z1max'))
    os.makedirs(f'{OUT}/reports', exist_ok=True)
    json.dump(res, open(f'{OUT}/reports/fit_results.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'scores'},
                     indent=1))


if __name__ == '__main__':
    main()
