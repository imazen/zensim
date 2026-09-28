#!/usr/bin/env python3
"""Marginal cost of each restored cut from interleaved zenbench raw files.

The layout chain is nested, so arm k carries families 1..k; family k's marginal
cost is (arm k median) - (arm k-1 median), the first arm's baseline being the
C8-on arm `fold1502_gmsbank`. alpha + beta*pixels is fitted per family over the
sizes present; percentages are of the C8-on median at the same size. Descriptive:
the run is reported, never used as a gate.
Usage: cost_fit.py <raw.zenbench> [<raw.zenbench> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

CHAIN = ['fold1502_gmsbank', 'fold1562_mapdev', 'fold1790_z1max',
         'fold1820_gmsnative', 'fold1825_dvifmgate']


def analyze(path):
    d = json.loads(Path(path).read_text())
    groups = {}
    for g in d['comparisons']:
        name = g['group_name']
        if not name.startswith('extract_paths_'):
            continue
        n = int(name.rsplit('_', 1)[1])
        groups[n] = {b['name']: b for b in g['benchmarks']}
        groups[n]['_rounds'] = g['completed_rounds']
    sizes = sorted(groups)
    px = np.array([n * n for n in sizes], dtype=np.float64)
    med = {a: np.array([groups[n][a]['summary']['median'] for n in sizes], dtype=np.float64)
           for a in CHAIN}
    out = dict(path=str(path), sizes=sizes, rounds={n: groups[n]['_rounds'] for n in sizes},
               unreliable=d.get('unreliable'), gate_waits=d.get('gate_waits'), families={})
    for prev, cur in zip(CHAIN, CHAIN[1:]):
        delta = med[cur] - med[prev]
        fit = np.linalg.lstsq(np.stack([np.ones_like(px), px], axis=1), delta, rcond=None)[0]
        pred = fit[0] + fit[1] * px
        ss = np.sum((delta - np.mean(delta)) ** 2)
        out['families'][cur] = dict(
            alpha_ns=float(fit[0]), beta_ns_per_px=float(fit[1]),
            r2=float(1 - np.sum((delta - pred) ** 2) / ss) if ss else None,
            median_delta_ns={n: float(x) for n, x in zip(sizes, delta)},
            pct_of_c8_on={n: float(100 * x / med[CHAIN[0]][i]) for i, (n, x) in enumerate(zip(sizes, delta))})
    total = med[CHAIN[-1]] - med[CHAIN[0]]
    out['all_on_pct_of_c8_on'] = {n: float(100 * x / med[CHAIN[0]][i])
                                  for i, (n, x) in enumerate(zip(sizes, total))}
    out['c8_on_median_ns'] = {n: float(x) for n, x in zip(sizes, med[CHAIN[0]])}
    return out


if __name__ == '__main__':
    for name in sys.argv[1:]:
        print(json.dumps(analyze(name), sort_keys=True))
