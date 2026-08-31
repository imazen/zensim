#!/usr/bin/env python3
"""Radius cost table from radius_cost.tsv, sound estimator = min over starts of
min-of-ITERS-in-process (era-2 protocol, benchmarks/era2_perf_break §22.5)."""
import csv, collections, sys, math, os
path = sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'radius_cost.tsv')
rows=[r for r in csv.DictReader(open(path),delimiter='\t') if r['min_ms'] not in ('NA','')]
g=collections.defaultdict(list)
for r in rows:
    g[(int(r['radius']),int(r['size']),int(r['threads']))].append((float(r['min_ms']),int(r['rss_kb'])))
radii=sorted({k[0] for k in g}, reverse=True)
sizes=sorted({k[1] for k in g})
threads=sorted({k[2] for k in g})
def est(R,S,T):
    v=g.get((R,S,T))
    return (min(x[0] for x in v), min(x[1] for x in v), len(v)) if v else (float('nan'),0,0)

# halo model
def wide(R): return 128 + 4*R          # STRIP_ROWS + 2*HALO_P, HALO_P = 2R
print("### Halo model (STRIP_ROWS=128, HALO_P = 2*BLUR_RADIUS)\n")
print(f"{'R':>2} {'HALO_P':>6} {'wide window':>12} {'row redundancy':>15} {'vs R=5':>8}")
for R in radii:
    print(f"{R:>2} {2*R:>6} {wide(R):>12} {wide(R)/128:>15.4f} {wide(R)/wide(5):>8.4f}")

print("\n### Cost: min-of-min ms (n starts), and Δ vs radius 5\n")
hdr = f"{'size':>5} {'T':>3} " + " ".join(f"{'R='+str(R):>18}" for R in radii)
print(hdr)
for S in sizes:
    for T in threads:
        b,_,_ = est(5,S,T)
        cells=[]
        for R in radii:
            m,_,n = est(R,S,T)
            d = 100*(m-b)/b if b==b else float('nan')
            cells.append(f"{m:8.2f} ({d:+5.2f}%)")
        print(f"{S:>5} {T:>3} " + " ".join(f"{c:>18}" for c in cells))

print("\n### Working set: min peak Rss (MB) from smaps_rollup\n")
print(f"{'size':>5} {'T':>3} " + " ".join(f"{'R='+str(R):>18}" for R in radii))
for S in sizes:
    for T in threads:
        _,br,_ = est(5,S,T)
        cells=[]
        for R in radii:
            _,rs,_ = est(R,S,T)
            d = 100*(rs-br)/br if br else float('nan')
            cells.append(f"{rs/1024:8.1f} ({d:+5.2f}%)")
        print(f"{S:>5} {T:>3} " + " ".join(f"{c:>18}" for c in cells))

print("\n### Estimator honesty: spread of min_ms across process starts (ASLR)\n")
print(f"{'R':>2} {'size':>5} {'T':>3} {'n':>3} {'min':>9} {'p50':>9} {'max':>9} {'spread%':>8}")
for k in sorted(g):
    v=sorted(x[0] for x in g[k])
    print(f"{k[0]:>2} {k[1]:>5} {k[2]:>3} {len(v):>3} {v[0]:9.2f} {v[len(v)//2]:9.2f} {v[-1]:9.2f} {100*(v[-1]-v[0])/v[0]:8.2f}")
