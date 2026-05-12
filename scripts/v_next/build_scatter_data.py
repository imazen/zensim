#!/usr/bin/env python3
"""Build scatter plot data from per-pair CSV.

Output: site/data/scatter/<label>.json with paired arrays for
plotting V_X (negated distance) vs ssim2 / butter / human_score.

Each scatter point is (x, y, color_by_band) where:
- x: ssim2 score (or butter, etc.)
- y: -V_X distance (so higher = better quality, matches ssim2 direction)
- color_by_band: MCOS band index (0-3 per CID22 Table 5)
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def band_of(mcos):
    if mcos < 50: return 0
    if mcos < 65: return 1
    if mcos < 90: return 2
    return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--per-pair", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dataset", default="CID22")
    args = ap.parse_args()

    points = []
    with open(args.per_pair) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("dataset") != args.dataset:
                continue
            try:
                h = float(row["human_score"]) * 100.0
                v04 = -float(row["v04_distance"])  # negate distance → quality
                ssim2 = float(row["fast_ssim2_score"])
                butter = -float(row["butter_3norm"])  # negate distance → quality
            except (ValueError, KeyError):
                continue
            points.append({"h": round(h, 2), "v": round(v04, 3),
                           "s": round(ssim2, 3), "b": round(butter, 3),
                           "band": band_of(h)})

    out = {
        "label": args.label,
        "dataset": args.dataset,
        "n": len(points),
        "axes": {
            "human": "MCOS (0..100)",
            "v04": "-(V_X distance) — higher = better",
            "ssim2": "fast-ssim2 score",
            "butter": "-(butter 3-norm) — higher = better",
        },
        "points": points,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fout:
        json.dump(out, fout, separators=(",", ":"))
    print(f"Wrote {len(points)} scatter points to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
