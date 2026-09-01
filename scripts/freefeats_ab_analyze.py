#!/usr/bin/env python3
"""Analyze scripts/freefeats_ab.sh's ab_raw.tsv per the era-2 ASLR protocol
(benchmarks/era2_perf_break_2026-08-31.md §22.5): for each (size, threads,
arm), the estimator is min-of-N-inner-walks (already `min_ms`, taken inside
the process) THEN min-over->=15 process starts (taken here). Reports the two
isolated ratios the ab.sh header documents:

  15f / 15c  = the raw_moments marginal cost alone (bit-identical layout,
               arm-name length, output-vector width -- only the four
               accumulators differ)
  15c / 156  = the wider-layout cost alone (944-wide output vector vs the
               real 156-only production walk, accumulators off both sides)
  15f / 156  = the combined cost a caller who flips the toggle actually pays

A bootstrap (resample the 15 starts with replacement, recompute the min,
1000 draws) gives a CI on each ratio so "free" is a stated interval, not a
single point estimate that could be one lucky ASLR layout.

Usage: python3 scripts/freefeats_ab_analyze.py [ab_raw.tsv]
"""
import csv
import random
import statistics
import sys
from collections import defaultdict

random.seed(20260901)
B_ITERS = 1000

path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/freefeats-2026-09-01/ab_raw.tsv"

cells = defaultdict(list)  # (size, threads, arm) -> [min_ms, ...] over starts
with open(path) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        key = (int(row["size"]), int(row["threads"]), row["arm"])
        cells[key].append(float(row["min_ms"]))

sizes = sorted({k[0] for k in cells})
threads = sorted({k[1] for k in cells})

def boot_ratio(a_vals, b_vals, iters=B_ITERS):
    """Bootstrap CI for min(a)/min(b), resampling starts with replacement."""
    na, nb = len(a_vals), len(b_vals)
    draws = []
    for _ in range(iters):
        ra = min(random.choices(a_vals, k=na))
        rb = min(random.choices(b_vals, k=nb))
        draws.append(ra / rb)
    draws.sort()
    lo = draws[int(0.025 * iters)]
    hi = draws[int(0.975 * iters) - 1]
    return statistics.median(draws), lo, hi

print(f"n_starts per cell: {len(next(iter(cells.values())))}")
print(f"{'size':>5} {'thr':>3} {'156(ms)':>9} {'15c(ms)':>9} {'15f(ms)':>9}  "
      f"{'15f/15c med [95%CI]':>28}  {'15c/156 med [95%CI]':>28}  {'15f/156 med [95%CI]':>28}")

rows_out = []
for size in sizes:
    for th in threads:
        m156 = cells[(size, th, "156")]
        m15c = cells[(size, th, "15c")]
        m15f = cells[(size, th, "15f")]
        min156, min15c, min15f = min(m156), min(m15c), min(m15f)
        r_fc = boot_ratio(m15f, m15c)
        r_c1 = boot_ratio(m15c, m156)
        r_f1 = boot_ratio(m15f, m156)
        print(f"{size:>5} {th:>3} {min156:>9.3f} {min15c:>9.3f} {min15f:>9.3f}  "
              f"{r_fc[0]:.4f} [{r_fc[1]:.4f},{r_fc[2]:.4f}]      "
              f"{r_c1[0]:.4f} [{r_c1[1]:.4f},{r_c1[2]:.4f}]      "
              f"{r_f1[0]:.4f} [{r_f1[1]:.4f},{r_f1[2]:.4f}]")
        rows_out.append((size, th, min156, min15c, min15f,
                          r_fc[0], r_fc[1], r_fc[2],
                          r_c1[0], r_c1[1], r_c1[2],
                          r_f1[0], r_f1[1], r_f1[2]))

out_tsv = sys.argv[2] if len(sys.argv) > 2 else None
if out_tsv:
    with open(out_tsv, "w") as f:
        f.write("size\tthreads\tmin156_ms\tmin15c_ms\tmin15f_ms\t"
                 "r_15f_15c_med\tr_15f_15c_lo\tr_15f_15c_hi\t"
                 "r_15c_156_med\tr_15c_156_lo\tr_15c_156_hi\t"
                 "r_15f_156_med\tr_15f_156_lo\tr_15f_156_hi\n")
        for r in rows_out:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"\nwrote {out_tsv}")
