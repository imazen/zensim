#!/usr/bin/env python3
"""Apply the LIBERAL butter-concordance filter to the original safesyn CSV.

Reads training_safe_synthetic.csv and the audit TSV from
butter_concordance_audit.py. Drops adjacent-q pairs where ssim2 and
butter disagree on within-curve ranking.

The "liberal" strategy keeps as much data as possible:
- For each curve, sort by q and walk adjacent pairs.
- If ssim2 ranking (higher q → higher ssim2) and butter ranking
  (higher q → lower butter) disagree at pair (q_i, q_{i+1}), drop
  the LOWER-quality side (q_i) — keep the higher-quality reference.
- This removes one row per discordant adjacent step (≈ 13,039 rows).

Output: new CSV preserving original column structure, suitable as a
drop-in replacement for the trainer / feature extractor.

Usage:
  python3 apply_butter_filter.py \\
    --in /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \\
    --out /tmp/zensim_loop/training_safe_synthetic_butter_filtered.csv
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ssim2-col", default="gpu_ssimulacra2")
    ap.add_argument("--butter-col", default="gpu_butteraugli")
    args = ap.parse_args()

    # Pass 1: group by (source, codec) and identify discordant adjacent pairs.
    grouped: dict[tuple[str, str], list[tuple[int, int, float, float]]] = defaultdict(list)
    field_names: list[str] = []
    rows: list[dict] = []
    with open(args.in_csv) as f:
        r = csv.DictReader(f)
        field_names = r.fieldnames
        for i, row in enumerate(r):
            rows.append(row)
            try:
                ssim2 = float(row[args.ssim2_col])
                butter = float(row[args.butter_col])
                q = int(row["quality"])
                src = row["source_path"]
                codec = row["codec"]
                grouped[(src, codec)].append((i, q, ssim2, butter))
            except (KeyError, ValueError):
                continue

    print(f"Read {len(rows)} rows from {args.in_csv}", file=sys.stderr)

    # Pass 2: walk adjacent-q pairs; mark LOW-quality side as drop when discordant
    drop_idx: set[int] = set()
    for (src, codec), items in grouped.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda t: t[1])  # by q ascending
        for a, b in zip(items[:-1], items[1:]):
            i_a, q_a, s_a, bu_a = a
            i_b, q_b, s_b, bu_b = b
            ssim_up = s_b > s_a       # higher q → higher ssim2 expected
            butter_down = bu_b < bu_a  # higher q → lower butter expected
            if ssim_up != butter_down:
                # discordant — drop the LOW-q row (i_a)
                drop_idx.add(i_a)

    n_drop = len(drop_idx)
    print(f"Dropping {n_drop} rows ({n_drop/len(rows)*100:.2f}%) as butter-discordant",
          file=sys.stderr)

    # Pass 3: emit retained rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=field_names)
        w.writeheader()
        kept = 0
        for i, row in enumerate(rows):
            if i in drop_idx:
                continue
            w.writerow(row)
            kept += 1
    print(f"Wrote {kept} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
