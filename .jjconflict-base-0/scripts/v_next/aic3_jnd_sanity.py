#!/usr/bin/env python3
"""AIC-3 JND calibration sanity check (V0_20d task).

Reads a per-pair CSV produced by `dataset_metric_baseline` when fed
the Mohammadi 2025 AIC-3 Anchor pairs TSV (via `--pairs-tsv
AIC-3:/tmp/aic3_anchor_pairs.tsv --v04-bake <ship-bake>`). Computes
median V_X output per AIC-3 JND band (sub-JND / AT-1-JND / 1.5-2 JND
/ visible) and asserts the AT-1-JND median falls within a target
window.

The default target window is **±5 of the V_18 ship's measured AT-JND
output (73.92)**, i.e. [68.92, 78.92]. If the ship is recalibrated
to a different anchor (e.g. V0_20d Option B midpoint at 70), update
`--target-at-jnd` accordingly.

This is a STABILITY CHECK, not an absolute calibration target:
- KonJND-1k says V_X should output ~63 at PJND mean.
- AIC-3 says V_X should output ~76.70 (matches SSIMULACRA2) at 1-JND.
- V_18 lands at 73.92 — close to AIC-3, ~11 above KonJND.
- Until V0_20d Option B ships, the sanity check tracks V_18's
  current AT-JND landing, not either of the two pure anchors.

## Usage

  python3 scripts/v_next/aic3_jnd_sanity.py \\
    --per-pair-csv benchmarks/per_pair_v0_18_aic3_anchor_2026-05-14.csv

Exit codes: 0 = pass, 1 = fail (median out of target window),
2 = invalid input.

## Regenerating the per-pair CSV

  cargo run --release -p zensim-bench --example dataset_metric_baseline -- \\
    --pairs-tsv 'AIC-3:/tmp/aic3_anchor_pairs.tsv' \\
    --v04-bake zensim/weights/v0_18_2026-05-13.bin \\
    --max-pairs 500 \\
    --per-pair-output benchmarks/per_pair_v0_18_aic3_anchor_<date>.csv

The TSV `/tmp/aic3_anchor_pairs.tsv` is built once per dataset
refresh from `scripts/v_next/aic3_anchor_pairs_tsv.py` (see that
script for the codec→path mapping). Run takes ~70s on the 250
files that resolve from the 300-row Anchor CSV.
"""
import argparse
import csv
import sys
from typing import List, Tuple


# AIC-3 JND bands. (lower, upper, label).
BANDS: List[Tuple[float, float, str]] = [
    (0.0, 0.5, "sub-JND"),
    (0.5, 0.9, "near-JND"),
    (0.9, 1.1, "AT-1-JND"),
    (1.1, 1.5, "1-1.5 JND"),
    (1.5, 2.0, "1.5-2 JND"),
    (2.0, 4.0, "visible"),
]


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = (len(s) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair-csv", required=True,
                    help="per-pair CSV from dataset_metric_baseline run on AIC-3 Anchor")
    ap.add_argument("--target-at-jnd", type=float, default=73.92,
                    help="expected V_X median at AIC-3 1-JND band (default: V_18 ship's measured)")
    ap.add_argument("--tolerance", type=float, default=5.0,
                    help="±tolerance around --target-at-jnd (default 5)")
    ap.add_argument("--min-n-at-jnd", type=int, default=10,
                    help="minimum number of samples in AT-1-JND band to assert (default 10)")
    args = ap.parse_args()

    try:
        rows = list(csv.DictReader(open(args.per_pair_csv)))
    except FileNotFoundError:
        print(f"ERROR: {args.per_pair_csv} not found", file=sys.stderr)
        return 2

    if not rows:
        print(f"ERROR: {args.per_pair_csv} is empty", file=sys.stderr)
        return 2

    if "human_score" not in rows[0] or "v04_distance" not in rows[0]:
        print(f"ERROR: {args.per_pair_csv} missing required columns "
              f"(have: {list(rows[0].keys())})", file=sys.stderr)
        return 2

    jnd = [float(r["human_score"]) for r in rows]
    vx = [float(r["v04_distance"]) for r in rows]

    print(f"# AIC-3 JND calibration sanity")
    print(f"")
    print(f"Per-pair CSV: {args.per_pair_csv}")
    print(f"n = {len(rows)}, JND range [{min(jnd):.3f}, {max(jnd):.3f}]")
    print(f"")
    print(f"| JND band   | JND range  | n  | V_X med | V_X p25 | V_X p75 |")
    print(f"|---|---|---:|---:|---:|---:|")

    median_at_jnd = None
    n_at_jnd = 0
    for lo, hi, label in BANDS:
        band = [vx[i] for i in range(len(rows)) if lo <= jnd[i] <= hi]
        if not band:
            continue
        med = percentile(band, 0.50)
        p25 = percentile(band, 0.25)
        p75 = percentile(band, 0.75)
        print(f"| {label:10} | [{lo:.1f}, {hi:.1f}] | {len(band):>2} | {med:.2f} | {p25:.2f} | {p75:.2f} |")
        if label == "AT-1-JND":
            median_at_jnd = med
            n_at_jnd = len(band)

    print(f"")

    if median_at_jnd is None or n_at_jnd < args.min_n_at_jnd:
        print(f"FAIL: only {n_at_jnd} samples in AT-1-JND band, need >= {args.min_n_at_jnd}")
        return 1

    lo = args.target_at_jnd - args.tolerance
    hi = args.target_at_jnd + args.tolerance
    if lo <= median_at_jnd <= hi:
        print(f"PASS: AT-1-JND median {median_at_jnd:.2f} ∈ [{lo:.2f}, {hi:.2f}] "
              f"(target {args.target_at_jnd} ± {args.tolerance})")
        return 0
    else:
        print(f"FAIL: AT-1-JND median {median_at_jnd:.2f} OUT OF [{lo:.2f}, {hi:.2f}] "
              f"(target {args.target_at_jnd} ± {args.tolerance})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
