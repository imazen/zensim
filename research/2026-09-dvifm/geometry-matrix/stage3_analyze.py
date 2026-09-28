#!/usr/bin/env python3
"""geometry lane Stage-3 — full cost protocol analysis.

Reads cost3/cost_t1.json + cost3/cost_t8.json (zenbench to_llm blocks +
config trailers), fits `time = alpha + beta * pixels` per (cell, threads)
on the 5-size medians, emits stage3.json + stage3.md.
"""
import json, re, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/v/output/zensim/geometry-2026-09-21')
SIZES = (64, 256, 1024, 2048, 4096)
NS_UNIT = {'s': 1e9, 'ms': 1e6, 'µs': 1e3, 'us': 1e3, 'ns': 1.0}

def _ns(v):
    m = re.match(r'(-?[0-9.]+)\s*(ns|µs|us|ms|s)$', v.strip())
    return float(m.group(1)) * NS_UNIT[m.group(2)] if m else float('nan')

def load(path):
    """to_llm -> {group: {bench: {median_ns, n, cv}}}."""
    txt = Path(path).read_text()
    out = {}
    for line in txt.splitlines():
        mg = re.search(r'\bgroup=(\S+)', line)
        mb = re.search(r'\bbenchmark=(\S+)', line)
        mm = re.search(r'\bmedian=(-?[0-9.]+(?:ns|µs|us|ms|s))\b', line)
        if not (mb and mm):
            continue
        g = mg.group(1) if mg else '?'
        # threaded benches tag `_batchN` onto the cell name — strip it so
        # t1 and t8 land under the same cell key.
        name = re.sub(r'_batch\d+$', '', mb.group(1))
        n = re.search(r'\bn=(\d+)', line)
        cv = re.search(r'\bcv=([0-9.]+)%', line)
        out.setdefault(g, {})[name] = {
            'median_ns': _ns(mm.group(1)),
            'n': int(n.group(1)) if n else None,
            'cv': float(cv.group(1)) if cv else None,
        }
    return out

def fit_ab(px, ns):
    """Least squares time = a + b*px. Returns (a, b, r2)."""
    n = len(px)
    sx, sy = sum(px), sum(ns)
    sxx = sum(x * x for x in px)
    sxy = sum(x * y for x, y in zip(px, ns))
    d = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / d
    a = (sy - b * sx) / n
    ybar = sy / n
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(px, ns))
    ss_tot = sum((y - ybar) ** 2 for y in ns)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2

def main():
    t1 = load(OUT / 'cost3' / 'cost_t1.json')
    t8 = load(OUT / 'cost3' / 'cost_t8.json')
    res = {'sizes': list(SIZES), 'cells': {}}
    lines = ['# Stage-3 full cost — `time = α + β·px` on 5-size medians', '']
    for tag, data in (('t1', t1), ('t8', t8)):
        lines += [f'## {tag}', '',
                  '| cell | size | median | cv |',
                  '|---|---|---|---|']
        for n in SIZES:
            g = f'geometry_{n}x{n}_t{tag[1:]}'
            for cell, v in sorted(data.get(g, {}).items()):
                cell_key = f'{cell}'
                ent = res['cells'].setdefault(cell_key, {}).setdefault(tag, {})
                ent[str(n)] = v
                lines.append(
                    f"| `{cell}` | {n}² | {v['median_ns'] / 1e6:.3f} ms | "
                    f"{v['cv'] if v['cv'] is not None else float('nan'):.1f}% |")
        lines.append('')
    lines += ['## α + β·px fits', '',
              '| cell | threads | α (ms) | β (ns/px) | R² |',
              '|---|---|---|---|---|']
    for cell, per_t in res['cells'].items():
        for tag in ('t1', 't8'):
            d = per_t.get(tag, {})
            px = [n * n for n in SIZES if str(n) in d]
            ns = [d[str(n)]['median_ns'] for n in SIZES if str(n) in d]
            if len(px) < 3:
                continue
            a, b, r2 = fit_ab(px, ns)
            per_t[tag]['fit'] = {'alpha_ms': a / 1e6, 'beta_ns_px': b, 'r2': r2}
            lines.append(
                f"| `{cell}` | {tag} | {a / 1e6:.4f} | {b:.4f} | {r2:.4f} |")
    (OUT / 'stage3.json').write_text(json.dumps(res, indent=1) + '\n')
    (OUT / 'stage3.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
