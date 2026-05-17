#!/usr/bin/env python3
"""Build a band-balanced safesyn training CSV by oversampling B1.

The base `safe_synth_clean_features.csv` has:
  - B0 (<50): 27.6%
  - B1 [50, 65): 16.2%
  - B2 [65, 90): 43.6%
  - B3 (>=90): 12.6%

CID22 validation distribution is:
  - B0: 7.5%, B1: 23.5%, B2: 67.9%, B3: 1.0%

**B1 is undersampled in training (16.2%) vs its CID22 representation
(23.5%)**. This script duplicates B1 rows to lift the train share to
~22–23%, matching the CID22 distribution.

The duplicates are appended to the original rows so existing TV pairs
(which reference row indices 0..N-1 of the original CSV) remain valid
without regeneration. The duplicated rows simply don't participate in
TV — they only contribute extra RankNet gradient signal in the B1
range.

Usage:
  python3 band_balance_safesyn.py \\
    --in /tmp/zensim_loop/safe_synth_clean_features.csv \\
    --out /tmp/zensim_loop/safe_synth_b1_oversample.csv \\
    --b1-multiplier 1.5  # produces ~37.9k B1 rows (was 25.3k) → 21.6%
"""
import argparse
import csv
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--b1-multiplier", type=float, default=1.5,
                    help="How many extra copies of each B1 row to add. "
                         "1.5 = +50% (lifts 16.2%% → 21.6%%). "
                         "2.0 = +100% (lifts to 26.0%%, slightly over CID22).")
    ap.add_argument("--score-col", default="human_score",
                    help="Column to read ssim2 score from. Defaults to "
                         "human_score (in [0, 1], multiplied by 100 here).")
    args = ap.parse_args()

    if args.b1_multiplier <= 0:
        print("ERROR: --b1-multiplier must be > 0", file=sys.stderr)
        sys.exit(1)

    n_total = 0
    band_counts = [0, 0, 0, 0]  # B0, B1, B2, B3
    b1_rows: list[dict] = []  # rows to duplicate
    field_names = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.in_csv, newline="") as f_in, open(args.out, "w", newline="") as f_out:
        r = csv.DictReader(f_in)
        field_names = r.fieldnames
        if args.score_col not in field_names:
            print(f"ERROR: --score-col '{args.score_col}' not in CSV "
                  f"(have: {list(field_names)[:8]}...)", file=sys.stderr)
            sys.exit(1)

        w = csv.DictWriter(f_out, fieldnames=field_names)
        w.writeheader()

        for row in r:
            n_total += 1
            try:
                s = float(row[args.score_col])
                if args.score_col == "human_score":
                    s *= 100.0  # normalize [0,1] → [0,100]
            except (ValueError, KeyError):
                w.writerow(row)
                continue

            # Always write the original row first
            w.writerow(row)

            if s < 50.0:
                band_counts[0] += 1
            elif s < 65.0:
                band_counts[1] += 1
                b1_rows.append(row)
            elif s < 90.0:
                band_counts[2] += 1
            else:
                band_counts[3] += 1

        # Append duplicated B1 rows
        n_b1_extra = int(round(len(b1_rows) * args.b1_multiplier))
        if n_b1_extra > 0:
            # Cycle through B1 rows; if multiplier > 1.0 some get more copies
            for i in range(n_b1_extra):
                w.writerow(b1_rows[i % len(b1_rows)])

    n_b1 = band_counts[1]
    n_b1_extra = int(round(n_b1 * args.b1_multiplier))
    new_total = n_total + n_b1_extra
    new_b1 = n_b1 + n_b1_extra
    print(f"Read {n_total} rows from {args.in_csv}", file=sys.stderr)
    print(f"  band counts: B0={band_counts[0]} B1={band_counts[1]} "
          f"B2={band_counts[2]} B3={band_counts[3]}", file=sys.stderr)
    print(f"  B1 share before: {n_b1/n_total*100:.1f}%", file=sys.stderr)
    print(f"Added {n_b1_extra} duplicated B1 rows (multiplier {args.b1_multiplier})",
          file=sys.stderr)
    print(f"  B1 share after: {new_b1/new_total*100:.1f}%", file=sys.stderr)
    print(f"Wrote {new_total} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
