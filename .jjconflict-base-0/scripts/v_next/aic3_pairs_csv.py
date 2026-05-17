#!/usr/bin/env python3
"""Build a paired CSV from AIC-3 CTC dataset for held-out validation.

AIC-3 CTC (EPFL) layout:
  /mnt/v/dataset/aic3_ctc_epfl/
    original/<img.name>.png       — reference images
    decoded/<img.name>/<codec>_<img.name>_<q>.png  — distorted variants
    decoded/info.csv              — score.jnd, codec, img.number, img.name, quality, quality.selected, method

Output columns:
  ref_path,dist_path,codec,quality_idx,quality_selected,score_jnd,method

score.jnd is the human-judged JND (just-noticeable-difference) measure;
more negative = more degraded. Per CID22 paper terminology, MOS-style
human scores ↔ negative-JND so we keep the raw sign and document it.

Usage:
  python3 aic3_pairs_csv.py \\
    --aic3-root /mnt/v/dataset/aic3_ctc_epfl \\
    --out /tmp/zensim_loop/aic3_ctc_pairs.csv
"""
import argparse
import csv
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aic3-root", required=True, type=Path,
                    help="Root of aic3_ctc_epfl, contains original/ and decoded/")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output paired CSV path")
    args = ap.parse_args()

    info_csv = args.aic3_root / "decoded" / "info.csv"
    if not info_csv.exists():
        print(f"ERROR: {info_csv} not found", file=sys.stderr)
        sys.exit(1)

    original_dir = args.aic3_root / "original"
    decoded_dir = args.aic3_root / "decoded"

    n_in = 0
    n_out = 0
    n_missing_ref = 0
    n_missing_dist = 0
    codec_counts = {}
    score_dist_buckets = [0] * 8  # -3..-2, -2..-1.5, -1.5..-1, -1..-0.75, -0.75..-0.5, -0.5..-0.25, -0.25..0, >=0
    score_bucket_labels = ["<-2", "[-2,-1.5)", "[-1.5,-1)", "[-1,-0.75)", "[-0.75,-0.5)", "[-0.5,-0.25)", "[-0.25,0)", ">=0"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(info_csv) as f_in, open(args.out, "w", newline="") as f_out:
        r = csv.DictReader(f_in)
        w = csv.writer(f_out)
        w.writerow(["ref_path", "dist_path", "codec", "quality_idx", "quality_selected", "score_jnd", "method"])

        for row in r:
            n_in += 1
            codec = row["codec"].strip()
            img_name = row["img.name"].strip()
            q_idx = row["quality"].strip()
            q_sel = row["quality.selected"].strip()
            score_jnd_str = row["score.jnd"].strip()
            method = row["method"].strip()

            # Skip blank/sentinel rows
            if not codec or not img_name or not q_idx:
                continue

            try:
                score_jnd = float(score_jnd_str)
            except ValueError:
                continue

            ref_path = original_dir / f"{img_name}.png"
            dist_path = decoded_dir / img_name / f"{codec}_{img_name}_{q_idx}.png"

            if not ref_path.exists():
                n_missing_ref += 1
                continue
            if not dist_path.exists():
                n_missing_dist += 1
                continue

            w.writerow([
                str(ref_path),
                str(dist_path),
                codec,
                q_idx,
                q_sel,
                f"{score_jnd:.4f}",
                method,
            ])
            n_out += 1
            codec_counts[codec] = codec_counts.get(codec, 0) + 1

            # Distribution check
            if score_jnd < -2:
                score_dist_buckets[0] += 1
            elif score_jnd < -1.5:
                score_dist_buckets[1] += 1
            elif score_jnd < -1:
                score_dist_buckets[2] += 1
            elif score_jnd < -0.75:
                score_dist_buckets[3] += 1
            elif score_jnd < -0.5:
                score_dist_buckets[4] += 1
            elif score_jnd < -0.25:
                score_dist_buckets[5] += 1
            elif score_jnd < 0:
                score_dist_buckets[6] += 1
            else:
                score_dist_buckets[7] += 1

    print(f"Read {n_in} rows from {info_csv}", file=sys.stderr)
    print(f"  emitted {n_out} paired rows to {args.out}", file=sys.stderr)
    print(f"  missing ref: {n_missing_ref}, missing dist: {n_missing_dist}", file=sys.stderr)
    print(f"  codec breakdown:", file=sys.stderr)
    for k, v in sorted(codec_counts.items()):
        print(f"    {k}: {v}", file=sys.stderr)
    print(f"  score.jnd distribution:", file=sys.stderr)
    for lab, cnt in zip(score_bucket_labels, score_dist_buckets):
        print(f"    {lab}: {cnt}", file=sys.stderr)


if __name__ == "__main__":
    main()
