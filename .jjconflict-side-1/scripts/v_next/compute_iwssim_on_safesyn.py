#!/usr/bin/env python3
"""Compute Wang & Li 2011 IW-SSIM on the safesyn corpus + write a parquet
sidecar.

Per the user directive 2026-05-15 ("we didn't calculate iwssim to make
it trainable"): adds an `iwssim` target column to the safesyn corpus
so V_X training can switch its regression target from ssim2 → IW-SSIM
(or co-train against both).

## Input

Safesyn corpus TSV at `/mnt/v/output/zensim/synthetic-v2/
training_safe_synthetic.csv` — schema includes `source_path` and
`decoded_path` columns pointing at PNG files on `/mnt/v/input/zensim/`.

## Output

Parquet sidecar at `/mnt/v/output/zensim/synthetic-v2/
iwssim_targets_safesyn_<YYYY-MM-DD>.parquet` with one row per
input pair: `source_path, decoded_path, iwssim`. A downstream
feature-CSV pipeline reads this and merges by (source_path,
decoded_path) into the human_score column.

## Implementation

Uses `pyiqa` (already installed at 0.1.14.1) which ships IW-SSIM
implemented per the Wang & Li 2011 paper (`pyiqa/archs/iw_ssim_arch.py`).
The PyTorch implementation runs on GPU; expect ~50 ms per pair at
512×384 → ~115 min total on a 4090.

## Usage

    # Small smoke test (first 100 rows):
    python3 scripts/v_next/compute_iwssim_on_safesyn.py --max-rows 100

    # Full run:
    python3 scripts/v_next/compute_iwssim_on_safesyn.py

    # Custom input/output:
    python3 scripts/v_next/compute_iwssim_on_safesyn.py \\
        --in /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \\
        --out /mnt/v/output/zensim/synthetic-v2/iwssim_targets_safesyn_2026-05-15.parquet
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
import piq
import torchvision.transforms.functional as TF
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="input_csv",
        type=Path,
        default=Path("/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            f"/mnt/v/output/zensim/synthetic-v2/iwssim_targets_safesyn_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.parquet"
        ),
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Process first N rows only (0 = all). For smoke tests.",
    )
    ap.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Torch device. Default auto-detects CUDA.",
    )
    ap.add_argument(
        "--batch-log-every",
        type=int,
        default=500,
        help="Log progress every N pairs.",
    )
    args = ap.parse_args()

    if not args.input_csv.is_file():
        print(f"ERROR: {args.input_csv} not found", file=sys.stderr)
        return 2

    print(f"input:  {args.input_csv}")
    print(f"output: {args.out}")
    print(f"device: {args.device}")
    print()

    # piq.information_weighted_ssim computes Wang & Li 2011 IW-SSIM
    # on torch tensors. Inputs are NCHW in [0, data_range].
    device = torch.device(args.device)

    def score_pair(src_path: str, dst_path: str) -> float:
        src_pil = Image.open(src_path).convert("RGB")
        dst_pil = Image.open(dst_path).convert("RGB")
        if src_pil.size != dst_pil.size:
            return float("nan")
        # PIL → CHW uint8 → NCHW float [0, 1]
        src_t = TF.to_tensor(src_pil).unsqueeze(0).to(device)
        dst_t = TF.to_tensor(dst_pil).unsqueeze(0).to(device)
        # piq.information_weighted_ssim expects [0, data_range] and
        # returns a scalar tensor.
        with torch.no_grad():
            return piq.information_weighted_ssim(
                src_t, dst_t, data_range=1.0, reduction="mean"
            ).item()

    # Iterate the CSV, compute IW-SSIM per pair.
    sources: list[str] = []
    decoded: list[str] = []
    scores: list[float] = []

    with args.input_csv.open() as f:
        reader = csv.DictReader(f)
        if "source_path" not in reader.fieldnames or "decoded_path" not in reader.fieldnames:
            print(
                f"ERROR: input CSV must have source_path + decoded_path columns; got {reader.fieldnames}",
                file=sys.stderr,
            )
            return 2

        t0 = time.perf_counter()
        n_ok = 0
        n_err = 0
        for row_idx, row in enumerate(reader):
            if args.max_rows and row_idx >= args.max_rows:
                break
            src = row["source_path"]
            dst = row["decoded_path"]
            try:
                score = score_pair(src, dst)
            except Exception as e:
                # Common errors: dimension mismatch, missing file.
                # Don't crash — log and continue. Skipped pairs leave
                # NaN in the sidecar; downstream merge filters NaNs.
                if n_err < 20:
                    print(f"  err on row {row_idx} ({Path(src).name} → {Path(dst).name}): {e}", file=sys.stderr)
                elif n_err == 20:
                    print("  (further errors suppressed)", file=sys.stderr)
                n_err += 1
                score = float("nan")

            sources.append(src)
            decoded.append(dst)
            scores.append(float(score))
            n_ok += 1

            if (n_ok % args.batch_log_every) == 0:
                elapsed = time.perf_counter() - t0
                rate = n_ok / elapsed
                eta_s = ((args.max_rows or 138_000) - n_ok) / max(rate, 1e-6)
                print(
                    f"  {n_ok:7d} pairs ({rate:.1f}/s, errs {n_err}, ETA {eta_s/60:.1f} min)",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {len(sources)} pairs in {elapsed:.1f}s ({len(sources)/elapsed:.1f}/s, {n_err} errors)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "source_path": pa.array(sources, type=pa.string()),
            "decoded_path": pa.array(decoded, type=pa.string()),
            "iwssim": pa.array(scores, type=pa.float64()),
        },
        metadata={
            "generator": b"scripts/v_next/compute_iwssim_on_safesyn.py",
            "input_csv": str(args.input_csv).encode(),
            "device": args.device.encode(),
            "piq_version": piq.__version__.encode() if hasattr(piq, "__version__") else b"unknown",
            "wall_seconds": f"{elapsed:.1f}".encode(),
            "n_pairs": str(len(sources)).encode(),
            "n_errors": str(n_err).encode(),
        },
    )
    pq.write_table(table, args.out, compression="zstd", compression_level=3)
    print(f"wrote {args.out} ({args.out.stat().st_size / (1024 * 1024):.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
