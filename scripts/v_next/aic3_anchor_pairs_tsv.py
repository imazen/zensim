#!/usr/bin/env python3
"""Build pairs TSV from the Mohammadi 2025 AIC-3 Anchor CSV.

The Anchor CSV identifies stimuli by `source_image` (e.g. `ref_2`) and
`image_filename` (e.g. `AVIF_00002_853x945_1.png`). The decoded
images live under `/mnt/v/input/datasets/aic3/data/decoded/<source_dir>/`
and the original references under
`/mnt/v/input/datasets/aic3/data/original/<source_dir>.png`.

This script joins the CSV with the actual file paths and emits a
TSV compatible with `dataset_metric_baseline --pairs-tsv AIC-3:<path>`.

Output columns:
  ref, dist, codec, version, human_score

`human_score` is the AIC-3 `distortion` column in JND units (1.0 =
first noticeable). Per the loader's column-aliasing in
`zensim-bench/examples/dataset_metric_baseline.rs`, the column name
`human_score` is preserved as the score column.

## Usage

  python3 scripts/v_next/aic3_anchor_pairs_tsv.py \\
    --anchor-csv /mnt/v/input/datasets/aic3/EvaluationMetrics/Anchor_assessment_on_PTC_full_resolution_Aug_3_2025.csv \\
    --data-root  /mnt/v/input/datasets/aic3/data \\
    --out        /tmp/aic3_anchor_pairs.tsv

## Notes

- 250 of 300 Anchor stimuli resolve to existing file pairs on the
  current EvaluationMetrics download (some codec naming variants
  don't match; the 250 covered include AVIF / HEIC / JXL / VVC /
  JPEG XS).
- Filename schema: `<CODEC>_<source_dir>_<dist_level>.png` —
  source_dir is `<5-digit-id>_<WxH>`.
"""
import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-csv", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path,
                    help="root containing original/ and decoded/")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    orig = args.data_root / "original"
    dec = args.data_root / "decoded"
    if not orig.is_dir() or not dec.is_dir():
        print(f"ERROR: expected {orig} and {dec} to exist", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(open(args.anchor_csv)))
    written = 0
    skipped_ref = 0
    skipped_dist = 0
    with open(args.out, "w") as f:
        f.write("ref\tdist\tcodec\tversion\thuman_score\n")
        for r in rows:
            img = r["image_filename"]
            parts = img.replace(".png", "").split("_")
            if len(parts) < 4:
                continue
            codec = parts[0]
            source_dir = f"{parts[1]}_{parts[2]}"
            dist_level = parts[-1]
            ref_path = orig / f"{source_dir}.png"
            dist_path = dec / source_dir / img
            if not ref_path.exists():
                skipped_ref += 1
                continue
            if not dist_path.exists():
                skipped_dist += 1
                continue
            f.write(f"{ref_path}\t{dist_path}\t{codec}\t{dist_level}\t{r['distortion']}\n")
            written += 1

    print(f"wrote {written} of {len(rows)} (skipped {skipped_ref} ref + {skipped_dist} dist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
