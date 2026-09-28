#!/usr/bin/env python3
"""transplant lane: derive the per-(channel, level) c0 knees from the
pilot pass's `ln min C̃` histograms, emit the final transplant spec.

Policy (declared in dvifm_transplant.rs): c0 = the p10 quantile of each
cell's TRAIN-corpus ln(min(C̃_s, C̃_d)) distribution — the gate counts a
block when the smoother of its two sides sits below the knee. p25/p50 are
recorded alongside so the choice is auditable.

Usage: pilot_c0.py <hist.json> <spec-out.json>
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
            # linear position inside the bin
            lo = LN_LO + (LN_HI - LN_LO) * i / BINS
            hi = LN_LO + (LN_HI - LN_LO) * (i + 1) / BINS
            frac = 0.5 if c == 0 else (want - (acc - c)) / c
            return lo + (hi - lo) * min(1.0, max(0.0, frac))
    return LN_HI

def main():
    hist = json.load(open(sys.argv[1]))
    assert hist['schema'] == 'transplant-hist-v1', hist['schema']
    cellmap = {(c['ch'], c['level']): c for c in hist['cells']}
    cells, report = [], {}
    for ch in range(3):
        for lv in range(4):
            c = cellmap[(ch, lv)]
            bins, n = c['bins'], c['n_blocks']
            q = {p: math.exp(quantile(bins, n, p)) for p in (0.10, 0.25, 0.50)}
            cells.append({'gate': True, 'c0': q[0.10]})
            report[f'ch{ch}_l{lv}'] = {'n': n, 'p10': q[0.10],
                                      'p25': q[0.25], 'p50': q[0.50]}
    spec = {'schema': 'transplant-spec-v1',
            'role': 'X4/X7 armed spec — c0 = p10 of pilot ln min-C per cell',
            'cells': cells, 'g': 1.0, 'p': 1.0, 'edge': True, 'hist': True}
    with open(sys.argv[2], 'w') as f:
        json.dump(spec, f, indent=1)
        f.write('\n')
    print(json.dumps(report, indent=1))

if __name__ == '__main__':
    main()
