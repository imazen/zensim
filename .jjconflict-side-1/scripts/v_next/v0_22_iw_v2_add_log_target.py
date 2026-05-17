#!/usr/bin/env python3
"""Add `iwssim_log_norm` target column to V_22-IW training CSVs.

T1.5 step 1: V_22-IW seed=1 collapsed on CID22 because the IW-SSIM
target distribution itself saturates near 1.0 (p95=0.99982,
p99=0.99999). The bake learned to predict ~99.9 for everything > 0.999
truth, producing rank ties that collapsed SROCC on CID22 mid-quality
bands.

This script pre-transforms the target column via
  iwssim_log_norm = (-log(1 - iwssim + 1e-6)) / max_log * 100

mapping the saturated tail [0.99, 1.0] to a wide range. Verified on
safesyn (n=196,086): the transform spreads p25/p50/p75/p99 from
0.967/0.988/0.997/0.99998 to 24.9/32.1/41.8/77.8.

Applies the same transform to the KADID/TID mock-iwssim CSVs (= each
val group's human_score under the name "iwssim") so the trainer's
global `--target-column iwssim_log_norm` flag reads consistent shapes
across groups. The val groups' rank order is preserved (the transform
is monotone), so val SROCC reporting is unchanged.

KonJND is SKIPPED for V_22-IW v2 — its human_score is in raw PJND
units (22-70 range, not [0, 1]), so the `1 - x` term goes negative
and the log explodes. The val mix loses one corpus but the remaining
three (safesyn train + KADID + TID val) are sufficient for V_22-IW v2
seed=1 cheap signal.

Output CSVs land at /mnt/v/zen/zensim-training/2026-05-16/v2/:
  safesyn_features_iwssim_log_372col.csv
  kadid_features_iwssim_log_372col.csv
  tid_features_iwssim_log_372col.csv
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


def transform_iwssim_to_log_norm(input_csv: Path, output_csv: Path, max_log: float) -> int:
    """Add iwssim_log_norm column = (-log(1 - iwssim + 1e-6)) / max_log * 100.

    Preserves all other columns. Returns row count written.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with input_csv.open() as fin, output_csv.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None or "iwssim" not in reader.fieldnames:
            raise SystemExit(f"{input_csv}: missing iwssim column")
        out_fields = list(reader.fieldnames) + ["iwssim_log_norm"]
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        for row in reader:
            iw = float(row["iwssim"])
            # Clamp to avoid -log(0) at iw = 1.0 exactly
            iw_clamped = min(iw, 0.9999999)
            log_val = -math.log(1.0 - iw_clamped + 1e-6)
            row["iwssim_log_norm"] = f"{log_val / max_log * 100.0:.6f}"
            writer.writerow(row)
            n += 1
    return n


def compute_max_log(input_csv: Path, col: str = "iwssim") -> float:
    """Find max(-log(1 - col + 1e-6)) across the CSV."""
    max_log = 0.0
    with input_csv.open() as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            iw = float(row[col])
            iw_clamped = min(iw, 0.9999999)
            log_val = -math.log(1.0 - iw_clamped + 1e-6)
            if log_val > max_log:
                max_log = log_val
    return max_log


def main() -> None:
    safesyn = Path("/mnt/v/zen/zensim-training/2026-05-16/safesyn_features_iwssim_372col.csv")
    kadid = Path(
        "/mnt/v/zen/zensim-training/2026-05-16/kadid_features_372col_2026-05-15_iwssim_mock.csv"
    )
    tid = Path(
        "/mnt/v/zen/zensim-training/2026-05-16/tid_features_372col_2026-05-15_iwssim_mock.csv"
    )

    out_dir = Path("/mnt/v/zen/zensim-training/2026-05-16/v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Anchor max_log to safesyn's distribution — that's the training
    # corpus, and the val corpora's mock iwssim values land in lower
    # log-space anyway (KADID/TID human_score has wider variance, so
    # log(1-x) is more moderate).
    print(f"Computing max_log from safesyn ...")
    max_log = compute_max_log(safesyn)
    print(f"  max_log = {max_log:.4f}  (safesyn-anchored)")
    print()

    for src, out_name in [
        (safesyn, "safesyn_features_iwssim_log_372col.csv"),
        (kadid, "kadid_features_iwssim_log_372col.csv"),
        (tid, "tid_features_iwssim_log_372col.csv"),
    ]:
        out = out_dir / out_name
        n = transform_iwssim_to_log_norm(src, out, max_log)
        print(f"wrote {out}  ({n:,} rows)")

    print()
    print("Distribution check on safesyn (post-transform):")
    vals = []
    with (out_dir / "safesyn_features_iwssim_log_372col.csv").open() as f:
        for row in csv.DictReader(f):
            vals.append(float(row["iwssim_log_norm"]))
    vals.sort()
    n = len(vals)
    print(
        f"  iwssim_log_norm: min={vals[0]:.2f}  p25={vals[n//4]:.2f}  p50={vals[n//2]:.2f}  "
        f"p75={vals[3*n//4]:.2f}  p99={vals[int(n*0.99)]:.2f}  max={vals[-1]:.2f}"
    )
    print()
    print("Next:")
    print("  zensim_mlp_train \\")
    print(
        f"    --group safesyn:{out_dir}/safesyn_features_iwssim_log_372col.csv:1.0:0.0 \\"
    )
    print(
        f"    --group kadid:{out_dir}/kadid_features_iwssim_log_372col.csv:0.3:1.0 \\"
    )
    print(
        f"    --group tid:{out_dir}/tid_features_iwssim_log_372col.csv:0.3:1.0 \\"
    )
    print("    --target-column iwssim_log_norm \\")
    print("    --target-scale 1.0 \\")
    print("    [...rest of V_22-IW seed=1 recipe...]")


if __name__ == "__main__":
    main()
