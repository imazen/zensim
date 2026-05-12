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
    ap.add_argument("--emit-bands", action="store_true",
                    help="Emit `band_id` column in the TSV (0..3 for B0..B3 "
                         "per CID22 Table 5). Required for per-band-weighted "
                         "TV in zensim_mlp_train.")
    args = ap.parse_args()

    # Pass 1: gather (source, codec) -> list of (row_idx, quality, ssim2_score)
    grouped = defaultdict(list)
    with open(args.training_csv) as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            src = row["source_path"]
            codec = row["codec"]
            q = int(row["quality"])
            # ssim2 score column may be `gpu_ssimulacra2` (safe-synth) or
            # `score_ssim2` (unified parquet). Both are MCOS-aligned.
            ssim2 = None
            for key in ("gpu_ssimulacra2", "score_ssim2", "ssim2"):
                if key in row and row[key]:
                    try:
                        ssim2 = float(row[key])
                        break
                    except ValueError:
                        pass
            grouped[(src, codec)].append((i, q, ssim2))

    print(f"Read {sum(len(v) for v in grouped.values())} rows in {len(grouped)} (source,codec) groups",
          file=sys.stderr)

    def band_of(score: float | None) -> int:
        """Map ssim2 score to band id per CID22 paper Table 5.

        Returns 0/1/2/3 for B0/B1/B2/B3, or 2 (B2) as a safe default
        when score is missing / non-MCOS-aligned (e.g. KonJND F-JND
        rows). Callers that need stricter behavior should filter rows
        without a score upstream.
        """
        if score is None or not (-200.0 < score < 200.0):
            return 2  # B2 = "high" — the most populous band, safe default
        if score < 50.0:
            return 0
        if score < 65.0:
            return 1
        if score < 90.0:
            return 2
        return 3

    # Pass 2: per group sort by q ascending, emit (lo, hi, band_id) for adjacent.
    # Convention: LOW q = LOW quality = HIGHER predicted distance (lo_idx);
    #             HIGH q = HIGH quality = LOWER predicted distance (hi_idx).
    # The pair's band_id reflects WHERE in quality space the pair lives —
    # use the HI member's score (the higher-quality end of the adjacent step).
    n_pairs = 0
    band_counts = [0, 0, 0, 0]
    skipped_singletons = 0
    with open(args.out, "w") as out:
        if args.emit_bands:
            out.write("lo_trainer_idx\thi_trainer_idx\tband_id\n")
        else:
            out.write("lo_trainer_idx\thi_trainer_idx\n")
        for key, rows in grouped.items():
            if len(rows) < 2:
                skipped_singletons += 1
                continue
            rows.sort(key=lambda t: t[1])
            for (lo_i, _, _), (hi_i, _, hi_score) in zip(rows[:-1], rows[1:]):
                if args.emit_bands:
                    band = band_of(hi_score)
                    out.write(f"{lo_i + args.safesyn_offset}\t{hi_i + args.safesyn_offset}\t{band}\n")
                    band_counts[band] += 1
                else:
                    out.write(f"{lo_i + args.safesyn_offset}\t{hi_i + args.safesyn_offset}\n")
                n_pairs += 1

    print(f"Wrote {n_pairs} TV pairs to {args.out} (skipped {skipped_singletons} singleton groups)",
          file=sys.stderr)
    if args.emit_bands:
        total = sum(band_counts) or 1
        print(f"  band distribution: "
              f"B0={band_counts[0]} ({band_counts[0]/total*100:.1f}%), "
              f"B1={band_counts[1]} ({band_counts[1]/total*100:.1f}%), "
              f"B2={band_counts[2]} ({band_counts[2]/total*100:.1f}%), "
              f"B3={band_counts[3]} ({band_counts[3]/total*100:.1f}%)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
