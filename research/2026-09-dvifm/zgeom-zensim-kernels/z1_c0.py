#!/usr/bin/env python3
"""zgeom Z1: derive the per-(channel, scale) c0 knees from a pilot
`zgeom-hist-v1` pass and emit the `zgeom-c0-v1` table the extractor's
`--zgeom-c0` flag consumes.

Policy (same convention as the transplant lane): c0 = the p10 quantile
of each cell's TRAIN-corpus ln(min(C̃_s, C̃_d)) distribution, converted
back to residual units — the gate counts a block when the smoother of
its two sides sits below the knee. p25/p50 recorded alongside.

Usage: z1_c0.py <hist.json> <c0-out.json>
"""
import json, math, sys

LN_LO, LN_HI, BINS = -16.0, 0.0, 128


def quantile(bins, n, q):
    if n == 0:
        return LN_HI
    want = q * n
    acc = 0
    for i, c in enumerate(bins):
        acc += c
        if acc >= want:
            lo = LN_LO + (LN_HI - LN_LO) * i / BINS
            hi = LN_LO + (LN_HI - LN_LO) * (i + 1) / BINS
            frac = 0.5 if c == 0 else (want - (acc - c)) / c
            return lo + (hi - lo) * min(1.0, max(0.0, frac))
    return LN_HI


def main():
    hist = json.load(open(sys.argv[1]))
    assert hist['schema'] == 'zgeom-hist-v1', hist['schema']
    cellmap = {(c['ch'], c['level']): c for c in hist['cells']}
    c0 = [[0.0] * 4 for _ in range(3)]
    report = {}
    for ch in range(3):
        for lv in range(4):
            c = cellmap[(ch, lv)]
            bins, n = c['bins'], c['n_blocks']
            q = {p: math.exp(quantile(bins, n, p))
                 for p in (0.10, 0.25, 0.50)}
            c0[ch][lv] = q[0.10]
            report[f'ch{ch}_s{lv}'] = {'n': n, 'p10': q[0.10],
                                      'p25': q[0.25], 'p50': q[0.50]}
    spec = {'schema': 'zgeom-c0-v1',
            'role': 'Z1 b5gate armed spec — c0 = p10 of TRAIN ln min-C̃ '
                    'per (channel, scale), zgeom-hist-v1 pass',
            'source_hist': hist.get('source', ''),
            'c0': c0}
    with open(sys.argv[2], 'w') as f:
        json.dump(spec, f, indent=1)
        f.write('\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
