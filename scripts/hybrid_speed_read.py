#!/usr/bin/env python3
"""hybrid_speed_read.py — the amended-W4 estimator over `hybrid_speed.sh run` output.

Reads the per-start zenbench tables and reduces them the way the registered
protocol says: `min` over process starts, per (build, threads, size, arm) —
`min` because the spread across ASLR starts is layout lottery, not signal
(`benchmarks/blur_radius_locality_branches_2026-08-31.md` §2.2). The spread is
printed beside every number so the reader can see how much lottery there was.

Computes no statistic beyond min / max / median of the per-start means that
zenbench itself produced. Emits TSV.
"""
import re, sys, glob, os, statistics, json, argparse

ARM_RE = re.compile(r'^\s*[├╰]─\s+(\S+)\s+([0-9.]+)\s*±\s*([0-9.]+)ms')
GRP_RE = re.compile(r'^\s*ssim2_bar_(\d+)\s')

def parse(path):
    """-> {(size, arm): mean_ms}"""
    out, size = {}, None
    for line in open(path, errors='replace'):
        g = GRP_RE.match(line)
        if g:
            size = int(g.group(1)); continue
        m = ARM_RE.match(line)
        if m and size is not None:
            out[(size, m.group(1))] = float(m.group(2))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/mnt/v/output/zensim/hybrid-2026-09-01/speed')
    ap.add_argument('--json')
    a = ap.parse_args()
    cells = {}                      # (build, t, size, arm) -> [means]
    # zenbench renders its table on STDERR; the .txt holds only the header line.
    for f in sorted(glob.glob(os.path.join(a.dir, 's2_*_start*.txt.err'))):
        b = os.path.basename(f)[:-4]
        m = re.match(r's2_(\w+)_(\d+)t_start(\d+)\.txt$', b)
        if not m:
            continue
        build, t = m.group(1), int(m.group(2))
        for (size, arm), ms in parse(f).items():
            cells.setdefault((build, t, size, arm), []).append(ms)
    if not cells:
        print('no parsable starts found', file=sys.stderr); return 2
    print('build\tthreads\tsize\tarm\tn_starts\tmin_ms\tmedian_ms\tmax_ms\tspread_pct')
    rec = {}
    for k in sorted(cells, key=lambda k: (k[0], k[1], k[2], k[3])):
        v = sorted(cells[k])
        spread = 100.0 * (v[-1] - v[0]) / v[0] if v[0] else float('nan')
        print(f'{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{len(v)}\t{v[0]:.3f}\t'
              f'{statistics.median(v):.3f}\t{v[-1]:.3f}\t{spread:.1f}')
        rec['|'.join(map(str, k))] = dict(n=len(v), min=v[0],
                                          median=statistics.median(v), max=v[-1])
    if a.json:
        json.dump(rec, open(a.json, 'w'), indent=1)
    return 0

if __name__ == '__main__':
    sys.exit(main())
