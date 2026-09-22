#!/usr/bin/env python3
"""zgeom Z2: label-free stability gates from the stability CSV.

Metric (geometry-lane stage1_analyze.py semantics, ported to zensim's
228-col surface):

  dln[pair,kernel,tx] = max over the 228 features of
      |ln(f_tx) - ln(f_base)|  with f clamped to EPS=1e-12
      (their `worst feature column per row`; here N=228 not 3).
  spread[kernel,tx]   = median over pairs of dln.
  excess[kernel,tx]   = median of the PAIRED differences
      dln[kernel] - dln[box2] over the same (pair,tx) rows — the
      common-mode per-pair sensitivity cancels.
  noise               = |median(diffs[0::2]) - median(diffs[1::2])|
      — the geometry lane's half-sample estimator verbatim.
  FAIL iff excess > max(3*noise, 0.002)  (their gate, 0.002 floor).

Also reports median-feature |dln| (median over the 228 cols instead of
max) as a robustness secondary — the max over 228 is a true extreme
statistic and can be dominated by a few near-zero slots; both share the
same paired-excess gate shape.

Report per kernel: shift1, worst of p0..p7, crop — the three gates —
plus per-phase detail. Usage: analyze_stability.py <stability.csv>
"""
import csv, sys, json, math
from collections import defaultdict
import numpy as np

EPS = 1e-12
NOISE_K = 3.0
FLOOR = 0.002
TXS = ['base', 'shift1', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7',
       'crop']


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float('nan')
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    path = sys.argv[1]
    # pair -> kernel -> tx -> np.array(228)
    data = defaultdict(lambda: defaultdict(dict))
    kernels = set()
    with open(path) as f:
        r = csv.DictReader(f)
        fcols = [c for c in r.fieldnames if c.startswith('f')]
        for row in r:
            p = int(row['pair']); k = row['kernel']; t = row['transform']
            kernels.add(k)
            data[p][k][t] = np.array([float(row[c]) for c in fcols])
    kernels = sorted(kernels)
    base_k = 'box2'

    # per (pair, kernel, tx != base): worst-feature AND median-feature |dln|
    worst, medf = {}, {}
    for p, kd in data.items():
        for k in kernels:
            b = np.log(np.clip(kd[k]['base'], EPS, None))
            for t in TXS[1:]:
                d = np.abs(np.log(np.clip(kd[k][t], EPS, None)) - b)
                worst[(p, k, t)] = float(d.max())
                medf[(p, k, t)] = float(np.median(d))

    pairs = sorted(data)
    n = len(pairs)

    def gate_table(src):
        out = {}
        for k in kernels:
            kd = {}
            for t in TXS[1:]:
                spread = med([src[(p, k, t)] for p in pairs])
                if k == base_k:
                    ex, noise = 0.0, 0.0
                else:
                    diffs = [src[(p, k, t)] - src[(p, base_k, t)]
                             for p in pairs]
                    ex = med(diffs)
                    e, o = diffs[0::2], diffs[1::2]
                    noise = abs(med(e) - med(o))
                kd[t] = {'spread': spread, 'excess': ex, 'noise': noise,
                         'fail': bool(k != base_k and
                                      ex > max(NOISE_K * noise, FLOOR))}
            out[k] = kd
        return out

    gw, gm = gate_table(worst), gate_table(medf)
    out = {'n_pairs': n, 'eps': EPS, 'floor': FLOOR, 'noise_k': NOISE_K,
           'kernels_worst': gw, 'kernels_medianfeat': gm}

    for label, g in (('WORST-FEATURE', gw), ('MEDIAN-FEATURE', gm)):
        print(f'\n== {label} ==')
        print(f'{"kernel":<10} {"shift1":>22} {"worstP":>22} {"crop":>22}'
              '  fails')
        for k in kernels:
            kd = g[k]
            worstp = max((kd[f'p{i}'] for i in range(8)),
                         key=lambda d: d['excess'])
            row, fails = [], []
            for name, d in (('shift1', kd['shift1']),
                            ('worstP', worstp), ('crop', kd['crop'])):
                tag = '' if k == base_k else \
                    (' FAIL' if d['fail'] else '')
                row.append(f"{d['spread']:.4f} ({d['excess']:+.4f}{tag})")
                if d['fail']:
                    fails.append(name)
            print(f'{k:<10} {row[0]:>22} {row[1]:>22} {row[2]:>22}  '
                  f'{",".join(fails) or "-"}')
    out_path = path.replace('.csv', '_gates.json')
    json.dump(out, open(out_path, 'w'), indent=1)
    print('\nwrote', out_path)


if __name__ == '__main__':
    main()
