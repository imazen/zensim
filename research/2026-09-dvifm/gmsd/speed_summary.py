#!/usr/bin/env python3
"""gmsd lane: summarise the warm speed runs (zenbench raw JSON from
tools/gmsd/speed_run.sh) and the cold runs (tools/gmsd/cold_bench.py).

Warm, per (process, thread count, size, arm): zenbench's median and MAD over
32 single-call rounds, and the fraction of rounds whose pre-round gate was
clean. Cold: median/p10/p90 of process wall time over 30 interleaved rounds.

Size model, per arm (zenmetrics' metric-loop notes: a single OLS over
64²..4096² tilts negative because the 16 MP point dominates):
  beta  = OLS slope of median ms vs megapixels over 1024², 2048², 4096²
  alpha = OLS intercept over 64² and 256² (ms)
Both are reported; neither is extrapolated to a size that was not measured.

Usage: speed_summary.py --speed DIR --cold DIR --out-json F --out-md F
"""
import argparse, json, os
import numpy as np


def fit(xs, ys):
    A = np.vstack([np.ones(len(xs)), xs]).T
    a, b = np.linalg.lstsq(A, np.array(ys), rcond=None)[0]
    return float(a), float(b)


def warm(speed_dir):
    out = {}
    for tag in ('1t-rev1', '8t-rev1', '1t-rev3', '8t-rev3'):
        p = f'{speed_dir}/{tag}.json'
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        cells, gates = {}, {}
        for c in d['comparisons']:
            n = int(c['group_name'].rsplit('_', 1)[1])
            gates[n] = sum(s['gate_clean'] for s in c['samples']) / len(c['samples'])
            for b in c['benchmarks']:
                cells.setdefault(b['name'], {})[n] = dict(
                    median_ms=b['summary']['median'] / 1e6, mad_ms=b['summary']['mad'] / 1e6,
                    n=b['summary']['n'])
        models = {}
        for arm, by in cells.items():
            big = [n for n in (1024, 2048, 4096) if n in by]
            small = [n for n in (64, 256) if n in by]
            m = {}
            if len(big) >= 2:
                _, beta = fit([n * n / 1e6 for n in big], [by[n]['median_ms'] for n in big])
                m['beta_ms_per_mp'] = beta
            if len(small) >= 2:
                alpha, _ = fit([n * n / 1e6 for n in small], [by[n]['median_ms'] for n in small])
                m['alpha_ms'] = alpha
            models[arm] = m
        out[tag] = dict(cells=cells, gate_clean_fraction=gates, models=models)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', required=True)
    ap.add_argument('--cold', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', required=True)
    a = ap.parse_args()
    w = warm(a.speed)
    cold = {}
    for t in (1, 8):
        p = f'{a.cold}/cold_{t}t.json'
        if os.path.exists(p):
            cold[f'{t}t'] = json.load(open(p))
    for t, c in cold.items():
        c.pop('raw_ms', None)
        c['models'] = {}
        for arm in c['median_ms'][str(c['sizes'][0])]:
            by = {int(n): v[arm] for n, v in c['median_ms'].items()}
            big = [n for n in (1024, 2048, 4096) if n in by]
            small = [n for n in (64, 256) if n in by]
            _, beta = fit([n * n / 1e6 for n in big], [by[n] for n in big])
            alpha, _ = fit([n * n / 1e6 for n in small], [by[n] for n in small])
            c['models'][arm] = dict(alpha_ms=alpha, beta_ms_per_mp=beta)
    json.dump(dict(warm=w, cold=cold), open(a.out_json, 'w'), indent=1)

    L = []
    for tag, r in w.items():
        sizes = sorted(next(iter(r['cells'].values())))
        L.append(f'#### warm {tag} (median ms of 32 single-call rounds; gate-clean fraction per size: '
                 + ', '.join(f"{n}²={r['gate_clean_fraction'][n]:.2f}" for n in sizes) + ')\n')
        L.append('| arm | ' + ' | '.join(f'{n}²' for n in sizes) + ' | α ms | β ms/MP |')
        L.append('|---' * (len(sizes) + 3) + '|')
        for arm, by in r['cells'].items():
            m = r['models'][arm]
            L.append(f'| {arm} | ' + ' | '.join(f"{by[n]['median_ms']:.3f}" for n in sizes)
                     + f" | {m.get('alpha_ms', float('nan')):.3f} | {m.get('beta_ms_per_mp', float('nan')):.2f} |")
        L.append('')
    for t, c in cold.items():
        sizes = c['sizes']
        L.append(f"#### cold {t} (process start + zen-codec PNG decode + colour conversion + score; median wall ms of {c['rounds']} interleaved launches; `floor` = `zenmetrics --version`)\n")
        L.append('| arm | ' + ' | '.join(f'{n}²' for n in sizes) + ' | α ms | β ms/MP |')
        L.append('|---' * (len(sizes) + 3) + '|')
        for arm in c['median_ms'][str(sizes[0])]:
            m = c['models'][arm]
            L.append(f'| {arm} | ' + ' | '.join(f"{c['median_ms'][str(n)][arm]:.2f}" for n in sizes)
                     + f" | {m['alpha_ms']:.2f} | {m['beta_ms_per_mp']:.2f} |")
        L.append('')
    open(a.out_md, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
