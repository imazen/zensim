#!/usr/bin/env python3
"""geometry lane Stage-2 analysis — dev-leg SROCC/KROCC, paired per-seed
differences against the baseline cell.

Reads fits/geometry_results.json (written incrementally by
fit_geometry.py) and emits stage2.json + stage2.md.

Per (cell, leg, seed): srocc/krocc of -E against y. The paired diff at a
seed is `m_cell - m_baseline` on the same leg+seed; the reported delta is
the median over seeds, with min/max as the paired range.
"""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/v/output/zensim/geometry-2026-09-21')
BASE = 'bin121.local.n5'
LEGS = ('codec_dev', 'human_dev', 'kadid135', 'konfig_val', 'safesyn_sub')

def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

def main():
    res = json.loads((OUT / 'fits' / 'geometry_results.json').read_text())
    cells = res['cells']
    seeds = [str(s) for s in res['seeds']]
    out = {'baseline': BASE, 'seeds': seeds, 'legs': {}, 'per_leg_pooled': {}}
    for leg in LEGS:
        tbl = {}
        base_by_seed = {}
        for s in seeds:
            b = cells.get(BASE, {}).get(s, {}).get('legs', {}).get(leg)
            if b:
                base_by_seed[s] = b
        for cell, per_seed in cells.items():
            row = {}
            for s in seeds:
                m = per_seed.get(s, {}).get('legs', {}).get(leg)
                if not m:
                    continue
                row[s] = {
                    'srocc': m['srocc'], 'krocc': m['krocc'],
                    'plcc': m['plcc'], 'mse': m['mse'],
                    'd_srocc': (m['srocc'] - base_by_seed[s]['srocc'])
                               if s in base_by_seed else None,
                    'd_krocc': (m['krocc'] - base_by_seed[s]['krocc'])
                               if s in base_by_seed else None,
                }
            ds = [r['d_srocc'] for r in row.values() if r['d_srocc'] is not None]
            dk = [r['d_krocc'] for r in row.values() if r['d_krocc'] is not None]
            tbl[cell] = {
                'per_seed': row,
                'srocc_med': med([r['srocc'] for r in row.values()]) if row else None,
                'd_srocc_med': med(ds) if ds else None,
                'd_srocc_min': min(ds) if ds else None,
                'd_srocc_max': max(ds) if ds else None,
                'd_krocc_med': med(dk) if dk else None,
            }
        out['legs'][leg] = tbl
    # pooled view: median d_srocc across legs per cell
    pooled = {}
    for cell in cells:
        ds = []
        for leg in LEGS:
            v = out['legs'][leg].get(cell, {}).get('d_srocc_med')
            if v is not None:
                ds.append(v)
        pooled[cell] = med(ds) if ds else None
    out['per_leg_pooled'] = pooled
    (OUT / 'stage2.json').write_text(json.dumps(out, indent=1) + '\n')

    lines = [f'# Stage-2 gate fits — baseline `{BASE}`, '
             f'seeds {",".join(seeds)}', '']
    for leg in LEGS:
        lines += [f'## {leg}', '',
                  '| cell | SROCC(med) | ΔSROCC med | min | max | ΔKROCC med |',
                  '|---|---|---|---|---|---|']
        tbl = out['legs'][leg]
        for cell in sorted(tbl, key=lambda c: -(tbl[c]['d_srocc_med'] or -9)):
            v = tbl[cell]
            if v['srocc_med'] is None:
                continue
            fm = lambda x: f'{x:+.4f}' if x is not None else '—'
            lines.append(
                f"| `{cell}` | {v['srocc_med']:.4f} | {fm(v['d_srocc_med'])} | "
                f"{fm(v['d_srocc_min'])} | {fm(v['d_srocc_max'])} | "
                f"{fm(v['d_krocc_med'])} |")
        lines.append('')
    lines += ['## pooled (median ΔSROCC across legs)', '',
              '| cell | ΔSROCC pooled |', '|---|---|']
    for cell in sorted(pooled, key=lambda c: -(pooled[c] or -9)):
        lines.append(f"| `{cell}` | {pooled[cell]:+.4f} |")
    (OUT / 'stage2.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
