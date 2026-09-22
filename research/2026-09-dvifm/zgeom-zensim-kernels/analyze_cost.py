#!/usr/bin/env python3
"""zgeom Z2: cost model from the zgeom_cost CSV.

Per (kernel, threads): least-squares fit  ns = alpha + beta*pixels
over the per-iteration timings at 256^2/1024^2/2048^2; report median
per size, alpha, beta, and the kernel ratio vs box2 at each size.
Usage: analyze_cost.py <cost.csv>
"""
import csv, sys
import numpy as np


def main():
    rows = {}
    for r in csv.DictReader(open(sys.argv[1])):
        key = (r['kernel'], int(r['threads']), int(r['size_px']))
        rows.setdefault(key, []).append(int(r['ns']))
    kernels = sorted({k for (k, _, _) in rows})
    threads = sorted({t for (_, t, _) in rows})
    print(f'{"kernel":<9} {"t":>2} {"256^2":>12} {"1024^2":>12} {"2048^2":>12}'
          f' {"alpha_ns":>12} {"beta_ns/px":>10} {"ratio@1024":>10}')
    fits = {}
    for k in kernels:
        for t in threads:
            sizes = sorted({s for (kk, tt, s) in rows if kk == k and tt == t})
            if not sizes:
                continue
            px = np.array([s * s for s in sizes], float)
            med = np.array([np.median(rows[(k, t, s)]) for s in sizes])
            A = np.vstack([np.ones_like(px), px]).T
            alpha, beta = np.linalg.lstsq(A, med, rcond=None)[0]
            fits[(k, t)] = (alpha, beta, dict(zip(sizes, med)))
    for k in kernels:
        for t in threads:
            if (k, t) not in fits:
                continue
            alpha, beta, d = fits[(k, t)]
            sizes = sorted(d)
            med = np.array([d[s] for s in sizes])
            base = fits.get(('box2', t), (None, None, {}))
            r1024 = (d.get(1024, np.nan) /
                     base[2].get(1024, np.nan)) if base[2] else np.nan
            print(f'{k:<9} {t:>2} '
                  + ' '.join(f'{med[i]:>12.3e}' for i in range(len(sizes)))
                  + f' {alpha:>12.3e} {beta:>10.4f} {r1024:>10.3f}')
    import json
    out = {f'{k}@t{t}': {'alpha': a, 'beta': b,
                        'median_ns': {str(s): m for s, m in d.items()}}
           for (k, t), (a, b, d) in fits.items()}
    op = sys.argv[1].replace('.csv', '_fit.json')
    json.dump(out, open(op, 'w'), indent=1)
    print('wrote', op)


if __name__ == '__main__':
    main()
