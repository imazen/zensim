#!/usr/bin/env python3
"""Regenerate TV pairs file for the zensim Rust trainer.

The TV regularizer needs `(lo_idx, hi_idx)` pairs where both indices
reference rows in the concatenated trainer feature space (group 0 then
group 1, etc.). Pairs are within-(source, codec) adjacent-quality
tuples — higher quality should produce LOWER predicted distance.

The original `safesyn_konjnd_tv_pairs.tsv` was indexed against the
ORIGINAL safe-synthetic CSV (218,089 rows). After perceptual-hash
filtering drops 61,669 rows, ALL its indices are wrong. This script
regenerates the safesyn-portion of TV pairs from the cleaned CSV.

For KonJND pairs (which live at offset safesyn_len in the concatenated
space) we leave them out of this regen — they need a separate pass.
The shipped V0_5 had ~271k pairs of which the majority come from
safesyn; the KonJND addition is small.

Usage:
    python3 regen_tv_pairs.py \\
        --training-csv /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_perceptual_clean.csv \\
        --safesyn-offset 0 \\
        --out /tmp/zensim_loop/safesyn_clean_tv_pairs.tsv
"""
import argparse
import csv
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--safesyn-offset", type=int, default=0,
                    help="Offset to add to row indices (for concatenated multi-group trainer)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Pass 1: gather (source, codec) -> list of (row_idx, quality)
    grouped = defaultdict(list)
    with open(args.training_csv) as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            src = row["source_path"]
            codec = row["codec"]
            q = int(row["quality"])
            grouped[(src, codec)].append((i, q))

    print(f"Read {sum(len(v) for v in grouped.values())} rows in {len(grouped)} (source,codec) groups",
          file=sys.stderr)

    # Pass 2: per group sort by q ascending, emit (lo, hi) for adjacent qualities.
    # Convention: LOW q = LOW quality = HIGHER predicted distance (lo_idx);
    #             HIGH q = HIGH quality = LOWER predicted distance (hi_idx).
    # So lo_idx in the TSV is the LOW-quality (lower-q-value) row,
    # hi_idx is the HIGH-quality (higher-q-value) row.
    n_pairs = 0
    skipped_singletons = 0
    with open(args.out, "w") as out:
        out.write("lo_trainer_idx\thi_trainer_idx\n")
        for key, rows in grouped.items():
            if len(rows) < 2:
                skipped_singletons += 1
                continue
            rows.sort(key=lambda t: t[1])
            for (lo_i, _), (hi_i, _) in zip(rows[:-1], rows[1:]):
                out.write(f"{lo_i + args.safesyn_offset}\t{hi_i + args.safesyn_offset}\n")
                n_pairs += 1

    print(f"Wrote {n_pairs} TV pairs to {args.out} (skipped {skipped_singletons} singleton groups)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
