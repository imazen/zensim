#!/usr/bin/env python3
"""geometry lane Stage-1 analysis — stability gates + 1024² cost screen.

Reads:
  stability/stability.csv   (pair,cell,plane,transform,level,f1_off,f1_gate,f1_param,f2_sum)
  cost/cost_t1.json         (zenbench to_llm block + config json line)
Writes:
  stage1.json               (per-cell per-tx paired excess + gate + cost)
  stage1.md                 (human table)

Instability: per (cell, transform) the median over (pair, plane, level)
of |ln(f_tx) − ln(f_base)|, worst of the three feature columns per row.

Gate (per the lane brief): "not worse than the baseline on stability by
more than the baseline's own round-to-round spread". Operationalised as
a PAIRED difference: for each (pair, plane, level) row both cells see
the same transform, so excess_i = dln_cell_i − dln_base_i cancels the
heavy-tailed per-pair sensitivity. excess = median of the paired
differences; noise = |median(diffs_even) − median(diffs_odd)|.
The xyb cell shares no (pair, plane, level) rows with the baseline —
its excess is the mean of the two half-median differences and its noise
|diffA − diffB| (the halves still hold the same images, so common-mode
pair sensitivity cancels at aggregate level). A transform FAILS when
    excess > max(3×noise, 0.002)
(the 0.002 absolute floor is a materiality threshold — ~10 % of the
typical baseline instability — so a real-but-tiny effect is reported
but does not kill an arm). Codec phases p0..p7 gate per phase.
"""
import csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/v/output/zensim/geometry-2026-09-21')
BASE = 'bin121.local.n5'
FEATS = ('f1_off', 'f1_param', 'f2_sum')
EPS = 1e-12
NOISE_K = 3.0
FLOOR = 0.002  # materiality floor in ln units (~10 % of typical instability)

def med(xs):
    if not xs:
        return float('nan')
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

def load_dln():
    rows = list(csv.DictReader(open(OUT / 'stability' / 'stability.csv')))
    base_by = defaultdict(dict)
    for r in rows:
        if r['transform'] == 'base':
            base_by[(r['cell'], int(r['pair']), r['plane'], int(r['level']))] = {
                k: float(r[k]) for k in FEATS}
    dln = defaultdict(dict)  # (cell, tx) -> {(pair, plane, level): |dln|}
    half = defaultdict(list)  # (cell, tx, half) -> [|dln|] (xyb fallback)
    for r in rows:
        if r['transform'] == 'base':
            continue
        b = base_by.get((r['cell'], int(r['pair']), r['plane'], int(r['level'])))
        if not b:
            continue
        worst = max(abs(math.log(max(float(r[f]), EPS)) - math.log(max(b[f], EPS)))
                    for f in FEATS)
        dln[(r['cell'], r['transform'])][(int(r['pair']), r['plane'], int(r['level']))] = worst
        half[(r['cell'], r['transform'], int(r['pair']) % 2)].append(worst)
    return dln, half

def tx_median(dln, cell, tx):
    return med(list(dln.get((cell, tx), {}).values()))

NS_UNIT = {'s': 1e9, 'ms': 1e6, 'µs': 1e3, 'us': 1e3, 'ns': 1.0}

def _ns(v):
    m = re.match(r'(-?[0-9.]+)\s*(ns|µs|us|ms|s)$', v.strip())
    return float(m.group(1)) * NS_UNIT[m.group(2)] if m else float('nan')

def load_cost():
    txt = (OUT / 'cost' / 'cost_t1.json').read_text()
    med_ns = {}
    # to_llm lines: group=.. benchmark=name | vs_base_* | min=.. median=..
    for line in txt.splitlines():
        mb = re.search(r'\bbenchmark=(\S+)', line)
        mm = re.search(r'\bmedian=(-?[0-9.]+(?:ns|µs|us|ms|s))\b', line)
        if mb and mm:
            med_ns[mb.group(1)] = _ns(mm.group(1))
    return med_ns

def main():
    dln, half = load_dln()
    cells = sorted({k[0] for k in dln})
    txs = ['shift1'] + ['p' + str(i) for i in range(8)] + ['crop']

    cost = load_cost()
    base_cost = cost.get(BASE, float('nan'))

    table = {}
    for c in cells:
        per_tx = {}
        fails = []
        for tx in txs:
            m = tx_median(dln, c, tx)
            if c == BASE:
                exc, n = 0.0, 0.0
            else:
                bc = dln.get((BASE, tx), {})
                cc = dln.get((c, tx), {})
                common = sorted(set(bc) & set(cc))
                if common:
                    diffs = [cc[k] - bc[k] for k in common]
                    exc = med(diffs)
                    e, o = diffs[0::2], diffs[1::2]
                    n = abs(med(e) - med(o))
                else:
                    # xyb family — no shared rows; aggregate halves still
                    # hold the same images so common-mode cancels.
                    dA = med(half[(c, tx, 0)]) - med(half[(BASE, tx, 0)])
                    dB = med(half[(c, tx, 1)]) - med(half[(BASE, tx, 1)])
                    exc, n = (dA + dB) / 2.0, abs(dA - dB)
            fail = exc > max(NOISE_K * n, FLOOR)
            per_tx[tx] = {'median': m, 'excess': exc, 'noise': n, 'fail': fail}
            if fail:
                fails.append(tx)
        table[c] = {'per_tx': per_tx, 'fail_txs': fails,
                    'stab_gate': not fails,
                    'cost_ns_1024_t1': cost.get(c),
                    'cost_vs_base': (cost.get(c) / base_cost)
                                    if c in cost and base_cost == base_cost else None}

    res = {'baseline': BASE, 'noise_k': NOISE_K, 'floor': FLOOR,
           'baseline_tx_median': {t: tx_median(dln, BASE, t) for t in txs},
           'baseline_cost_ns': base_cost, 'cells': table,
           'survivors': [c for c in cells if table[c]['stab_gate']]}
    (OUT / 'stage1.json').write_text(json.dumps(res, indent=1) + '\n')

    lines = [f"# Stage-1 screen — baseline `{BASE}`", '',
             'per-tx median |Δln| (worst feature); paired half-difference '
             f'excess vs baseline; fail = excess > max({NOISE_K:.0f}×noise, {FLOOR})', '',
             '| cell | shift1 | worstP | crop | fails | ns @1024² | ×base |',
             '|---|---|---|---|---|---|---|']
    for c in cells:
        v = table[c]
        wp = max((v['per_tx']['p' + str(i)] for i in range(8)),
                 key=lambda d: d['excess'])
        s1, cr = v['per_tx']['shift1'], v['per_tx']['crop']
        lines.append(
            f"| `{c}` | {s1['median']:.4f} ({s1['excess']:+.4f}) | "
            f"{wp['median']:.4f} ({wp['excess']:+.4f}) | "
            f"{cr['median']:.4f} ({cr['excess']:+.4f}) | "
            f"{','.join(v['fail_txs']) or '—'} | "
            f"{v['cost_ns_1024_t1'] if v['cost_ns_1024_t1'] is not None else float('nan'):.4g} | "
            f"{v['cost_vs_base'] if v['cost_vs_base'] is not None else float('nan'):.2f} |")
    lines += ['', f"survivors: {', '.join(res['survivors'])}"]
    (OUT / 'stage1.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
