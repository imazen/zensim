#!/usr/bin/env python3
"""Recompute Rev4 ST marginal costs from the pinned zenbench raw medians."""

import json
from pathlib import Path

RAW = Path('/var/tmp/featbank-impl/evidence/st1.zenbench')
ARMS = {
    'fold944_full': 'fold944_full',
    'fold986_dvifm': 'fold986_dvifm',
    'gridblk': 'fold986_gridblk',
    'ringbasis': 'fold986_ringbasis',
    'tailhist': 'fold986_tailhist',
    'arttype': 'fold986_arttype',
    'rev4_all': 'fold1322_rev4',
}


def fit(xs, ys):
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    xx = sum((x - xbar) ** 2 for x in xs)
    beta = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / xx
    alpha = ybar - beta * xbar
    ss_res = sum((y - alpha - beta * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    return alpha, beta, 1.0 - ss_res / ss_tot


def main():
    groups = json.loads(RAW.read_text())['comparisons']
    values = {}
    for group in groups:
        size = int(group['group_name'].rsplit('_', 1)[1])
        arms = {b['name']: b['summary']['median'] / 1e6
                for b in group['benchmarks']}
        values[size] = {name: arms[raw] for name, raw in ARMS.items()}
    for size in sorted(values):
        print('MEDIAN', size, *(f'{name}={values[size][name]:.5f}' for name in ARMS))
    xs = [size * size for size in sorted(values)]
    base = values[1024]['fold944_full']
    for name in ('gridblk', 'ringbasis', 'tailhist', 'arttype', 'rev4_all'):
        ys = [values[size][name] - values[size]['fold986_dvifm']
              for size in sorted(values)]
        alpha, beta, r2 = fit(xs, ys)
        fit_pct = (alpha + beta * 1024 * 1024) / base * 100
        raw_pct = (values[1024][name] - values[1024]['fold986_dvifm']) / base * 100
        print(f'FIT {name} alpha_ms={alpha:.6f} beta_ns_px={beta*1e6:.6f} '
              f'r2={r2:.6f} fit_pct={fit_pct:.6f} raw_pct={raw_pct:.6f}')


if __name__ == '__main__':
    main()
