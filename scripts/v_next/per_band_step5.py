#!/usr/bin/env python3
"""Compute step-5 per-band SROCC + MAE from per-pair eval CSV.

dataset_metric_baseline.rs emits a per-pair CSV with columns:
  dataset, human_score, v02_distance, v04_distance, fast_ssim2_score, butter_3norm

For CID22, human_score × 100 = MCOS-aligned ssim2 score (per Table 5
of the CID22 paper, MCOS and SSIMULACRA 2 share the same 0..100 scale
1:1). We bin pairs by `floor(human_score × 100 / 5) × 5` to produce
20 step-5 bins (0..5, 5..10, ..., 95..100). For each bin we compute
SROCC of distance/score vs MCOS within the bin.

Usage:
  python3 per_band_step5.py \\
    --label V0_15 \\
    --per-pair /tmp/zensim_loop/v0_15_per_pair.csv \\
    --out /home/lilith/work/zen/zensim/site/data/step5_bands/v0_15.json

DEPRECATED STAT MATH: the per-band SROCC here is superseded by the canonical
Rust `panel` (zensim-validate/src/bin/panel.rs), which natively supports a
`band` column for per-band breakdown. For NEW work:
    from scripts.lib.zen_stats import panel
    stats = panel(predicted, target, band=band_labels)  # stats["bands"][...]
verified to scipy <= 1e-9 by scripts/verify_panel_parity.py.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--per-pair", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--step", type=int, default=5, help="Bin width in MCOS points (default 5)")
    ap.add_argument("--dataset", default="CID22", help="Filter for this dataset (default CID22)")
    args = ap.parse_args()

    # Group pairs by bin → (human, v02, v04, ssim2, butter) tuples
    bins: dict[int, list] = defaultdict(list)
    n_in = 0
    n_kept = 0
    with open(args.per_pair) as f:
        r = csv.DictReader(f)
        for row in r:
            n_in += 1
            if row["dataset"] != args.dataset:
                continue
            try:
                h = float(row["human_score"]) * 100.0  # MCOS [0, 100]
                v02 = float(row["v02_distance"])
                v04 = float(row["v04_distance"])
                ssim2 = float(row["fast_ssim2_score"])
                butter = float(row["butter_3norm"])
            except (ValueError, KeyError):
                continue
            n_kept += 1
            bin_lo = int(h // args.step) * args.step
            bins[bin_lo].append((h, v02, v04, ssim2, butter))

    print(f"Read {n_in} rows, kept {n_kept} for {args.dataset}", file=sys.stderr)

    # Compute SROCC + n per bin
    band_rows = []
    for bin_lo in sorted(bins.keys()):
        pairs = bins[bin_lo]
        if len(pairs) < 3:
            continue
        humans = [p[0] for p in pairs]
        v02 = [p[1] for p in pairs]
        v04 = [p[2] for p in pairs]
        ssim2 = [p[3] for p in pairs]
        butter = [p[4] for p in pairs]
        # SROCC of metric vs human; negate distance since lower distance = higher quality
        # Distance metrics get negative SROCC unless we flip sign; report magnitude
        # consistent with the bake direction (V0_4 emits distance for V0_2 path).
        # For positive correlation, flip sign of distance metrics.
        srocc_v02 = -spearman(v02, humans) if humans else None
        srocc_v04 = -spearman(v04, humans) if humans else None
        srocc_ssim2 = spearman(ssim2, humans) if humans else None
        srocc_butter = -spearman(butter, humans) if humans else None  # butter is distance
        band_rows.append({
            "bin_lo": bin_lo,
            "bin_hi": bin_lo + args.step,
            "n": len(pairs),
            "srocc_v02": round(srocc_v02, 4) if srocc_v02 is not None else None,
            "srocc_v04": round(srocc_v04, 4) if srocc_v04 is not None else None,
            "srocc_ssim2": round(srocc_ssim2, 4) if srocc_ssim2 is not None else None,
            "srocc_butter": round(srocc_butter, 4) if srocc_butter is not None else None,
        })

    out_data = {
        "label": args.label,
        "dataset": args.dataset,
        "step": args.step,
        "total_pairs": n_kept,
        "bands": band_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Wrote {len(band_rows)} bins to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
