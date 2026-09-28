#!/usr/bin/env python3
"""Fit C8 marginal alpha+beta*pixels from interleaved zenbench raw files."""
import json
import sys
from pathlib import Path

import numpy as np

SIZES = (256, 1024, 2048, 4096)
OFF = 'fold1322_rev4'
ON = 'fold1502_gmsbank'


def analyze(path):
    d = json.loads(Path(path).read_text())
    pairs = {}
    for n in SIZES:
        group = next(g for g in d['comparisons'] if g['group_name'] == f'extract_paths_{n}')
        bench = {b['name']: b for b in group['benchmarks']}
        pairs[n] = dict(off_ns=bench[OFF]['summary']['median'],
                        on_ns=bench[ON]['summary']['median'],
                        rounds=group['completed_rounds'])
    px = np.array([n * n for n in SIZES], dtype=np.float64)
    off = np.array([pairs[n]['off_ns'] for n in SIZES], dtype=np.float64)
    delta = np.array([pairs[n]['on_ns'] - pairs[n]['off_ns'] for n in SIZES], dtype=np.float64)
    fit = np.linalg.lstsq(np.stack([np.ones_like(px), px], axis=1), delta, rcond=None)[0]
    predicted = fit[0] + fit[1] * px
    r2 = 1 - np.sum((delta - predicted) ** 2) / np.sum((delta - np.mean(delta)) ** 2)
    at1024 = fit[0] + fit[1] * 1024 ** 2
    result = dict(path=str(path), pairs=pairs, alpha_ns=float(fit[0]),
                  beta_ns_per_px=float(fit[1]), r2=float(r2),
                  fit_pct_at1024=float(100 * at1024 / pairs[1024]['off_ns']),
                  raw_median_pct_at1024=float(100 * delta[1] / off[1]),
                  unreliable=d.get('unreliable'), gate_waits=d.get('gate_waits'))
    return result


if __name__ == '__main__':
    for name in sys.argv[1:]:
        print(json.dumps(analyze(name), sort_keys=True))
