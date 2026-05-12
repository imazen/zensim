#!/usr/bin/env python3
"""Combine per-pair predictions from multiple seed bakes into an ensemble.

Each per-pair CSV has rows: dataset, human_score, v02_distance, v04_distance, fast_ssim2_score, butter_3norm.

Ensemble strategy: mean of v04_distance across N seeds.
Output: ensemble SROCC vs human_score (CID22 only).

Usage:
  python3 ensemble_seeds.py /tmp/zensim_loop/v0_16_per_pair.csv \\
                            /tmp/zensim_loop/v0_18_per_pair.csv \\
                            /tmp/zensim_loop/v0_19_per_pair.csv \\
                            /tmp/zensim_loop/v0_20_per_pair.csv
"""
import csv, sys
from collections import defaultdict


def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    def ranks(v):
        sv = sorted(enumerate(v), key=lambda t: t[1])
        r = [0.0] * len(v)
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1][1] == sv[i][1]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[sv[k][0]] = avg
            i = j + 1
        return r
    rx = ranks(xs)
    ry = ranks(ys)
    mean = (n + 1) / 2.0
    num = sum((rx[i] - mean) * (ry[i] - mean) for i in range(n))
    dx = sum((rx[i] - mean) ** 2 for i in range(n))
    dy = sum((ry[i] - mean) ** 2 for i in range(n))
    den = (dx * dy) ** 0.5
    return num / den if den > 1e-12 else 0.0


def band_of(mcos):
    if mcos < 50: return 0
    if mcos < 65: return 1
    if mcos < 90: return 2
    return 3


def main():
    args = sys.argv[1:]
    # Optional first arg: --dataset CID22|AIC-3 CTC
    dataset = "CID22"
    if args and args[0].startswith("--dataset"):
        if "=" in args[0]:
            dataset = args[0].split("=", 1)[1]
        else:
            dataset = args[1]; args = args[1:]
        args = args[1:]
    files = args
    if not files:
        print("Usage: ensemble_seeds.py [--dataset NAME] file1 file2 [...]", file=sys.stderr); sys.exit(1)
    print(f"Dataset filter: {dataset}", file=sys.stderr)

    # Read each file into list of (human, v04, ssim2, butter)
    runs = []
    for f in files:
        rows = []
        with open(f) as fp:
            r = csv.DictReader(fp)
            for row in r:
                if row.get("dataset") != dataset:
                    continue
                try:
                    h = float(row["human_score"]) * 100.0
                    v04 = float(row["v04_distance"])
                    s2 = float(row["fast_ssim2_score"])
                    bu = float(row["butter_3norm"])
                except (KeyError, ValueError):
                    continue
                rows.append((h, v04, s2, bu))
        runs.append(rows)
        print(f"  {f}: {len(rows)} pairs", file=sys.stderr)

    # Validate row alignment (all files same length)
    n = len(runs[0])
    for r in runs[1:]:
        if len(r) != n:
            print(f"ERROR: row counts differ ({n} vs {len(r)})", file=sys.stderr)
            sys.exit(1)

    # Compute ensemble: mean v04 across runs (per-row)
    humans = [runs[0][i][0] for i in range(n)]
    ssim2s = [runs[0][i][2] for i in range(n)]
    butters = [runs[0][i][3] for i in range(n)]
    v04_ensemble = [sum(runs[k][i][1] for k in range(len(runs))) / len(runs) for i in range(n)]

    # Per-seed SROCC for comparison
    print("\n=== Per-seed CID22 SROCC ===", file=sys.stderr)
    for k, f in enumerate(files):
        v04s = [runs[k][i][1] for i in range(n)]
        s = -spearman(v04s, humans)  # negate distance for positive correlation
        print(f"  {f}: SROCC = {s:.4f}", file=sys.stderr)

    # Reference metrics
    s_ssim2 = spearman(ssim2s, humans)
    s_butter = -spearman(butters, humans)  # butter is distance
    print(f"  fast-ssim2:  SROCC = {s_ssim2:.4f}", file=sys.stderr)
    print(f"  butter:      SROCC = {s_butter:.4f}", file=sys.stderr)

    # Ensemble SROCC
    s_ens = -spearman(v04_ensemble, humans)
    print(f"\n=== Ensemble (mean of {len(runs)} seeds) ===", file=sys.stderr)
    print(f"  Ensemble SROCC: {s_ens:.4f}", file=sys.stderr)
    print(f"  vs ssim2: {s_ens - s_ssim2:+.4f}", file=sys.stderr)

    # Per-band ensemble
    print(f"\n=== Ensemble per-band CID22 ===", file=sys.stderr)
    for band in range(4):
        idx = [i for i in range(n) if band_of(humans[i]) == band]
        if len(idx) < 3:
            continue
        h_b = [humans[i] for i in idx]
        e_b = [v04_ensemble[i] for i in idx]
        s2_b = [ssim2s[i] for i in idx]
        ens = -spearman(e_b, h_b)
        s2 = spearman(s2_b, h_b)
        print(f"  B{band} (n={len(idx)}): ens={ens:.4f}, ssim2={s2:.4f}, Δ={ens-s2:+.4f}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
