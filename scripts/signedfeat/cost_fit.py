#!/usr/bin/env python3
"""Marginal cost of the SIGNEDFEAT families from interleaved zenbench raw files (see
scripts/restore_cuts/cost_fit.py, whose analysis this reuses with a different chain).

Chain: fold1825_dvifmgate (all four restored cuts on = the side pass is already running) ->
fold1837_texgain -> fold1853_satsign. `all_restore_over_c8` is fold1825_dvifmgate vs fold1502_gmsbank (the
C8-on baseline), i.e. what the side pass itself costs; a standalone sidecar run pays the side pass once.
alpha + beta*pixels per marginal family over the sizes present. Descriptive, never a gate.
Usage: cost_fit.py <raw.zenbench> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np

CHAIN = ['fold1825_dvifmgate', 'fold1837_texgain', 'fold1853_satsign']
C8 = 'fold1502_gmsbank'


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
           for a in [C8] + CHAIN}
    out = dict(path=str(path), sizes=sizes, rounds={n: groups[n]['_rounds'] for n in sizes},
               unreliable=d.get('unreliable'), gate_waits=d.get('gate_waits'), families={},
               median_ms={a: {n: float(x) / 1e6 for n, x in zip(sizes, med[a])} for a in med})
    pairs = [(C8, CHAIN[0], 'all_restore_over_c8')] + [(p, c, c) for p, c in zip(CHAIN, CHAIN[1:])]
    for prev, cur, label in pairs:
        delta = med[cur] - med[prev]
        fit = np.linalg.lstsq(np.stack([np.ones_like(px), px], axis=1), delta, rcond=None)[0]
        pred = fit[0] + fit[1] * px
        ss = np.sum((delta - np.mean(delta)) ** 2)
        out['families'][label] = dict(
            alpha_ns=float(fit[0]), beta_ns_per_px=float(fit[1]),
            r2=float(1 - np.sum((delta - pred) ** 2) / ss) if ss else None,
            median_delta_ms={n: float(x) / 1e6 for n, x in zip(sizes, delta)},
            pct_of_c8_on={n: float(100 * x / med[C8][i]) for i, (n, x) in enumerate(zip(sizes, delta))})
    return out


if __name__ == '__main__':
    for name in sys.argv[1:]:
        print(json.dumps(analyze(name), sort_keys=True))
