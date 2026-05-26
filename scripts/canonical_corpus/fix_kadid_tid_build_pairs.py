#!/usr/bin/env python3
"""Build (ref_path, dist_path) pairs TSV for KADID-10k and TID2013, in the
EXACT row order of the canonical train parquets, so recomputed IW-SSIM +
SSIMULACRA2 scores can be re-joined positionally.

Background — DATA INTEGRITY FIX (2026-05-25):
  The canonical train parquets carry corrupt iwssim (= human_score copy)
  and ssim2_gpu (joined on ref_basename only → constant per ref) columns.
  This script recovers the per-row distorted-image filename so the metrics
  can be recomputed on the correct (ref, dist) pairing.

Row→image recovery (verified):
  KADID: parquet row order == dmos.csv row order, full match on ref order.
         row i → dmos.csv row i: dist_img + ref_img.
         human_score == (dmos - 1) / 4.
  TID:   parquet row order == mos_with_names.txt row order, full ref match.
         row i → names[i] (distorted), ref = I{NN}.png.
         human_score == mos / 9. PNG variants live in *_png/ dirs.

Outputs a TSV with columns: ref_path, dist_path, ref_basename, human_score
(extra columns are passed through unchanged by zen-metrics batch).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

KADID_DIR = Path("/mnt/v/dataset/kadid10k")
KADID_IMG = KADID_DIR / "images"
TID_DIR = Path("/mnt/v/dataset/tid2013")
TID_REF_PNG = TID_DIR / "reference_images_png"
TID_DIST_PNG = TID_DIR / "distorted_images_png"

CANON21 = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21/train")
CANON18 = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18/train")


def build_kadid(parquet_path: Path, out_tsv: Path, max_rows: int = 0) -> int:
    tbl = pq.read_table(str(parquet_path), columns=["ref_basename", "human_score"])
    prefs = tbl.column("ref_basename").to_pylist()
    ph = np.asarray(tbl.column("human_score").to_numpy(zero_copy_only=False), dtype=float)

    rows = list(csv.DictReader(open(KADID_DIR / "dmos.csv")))
    if len(rows) != len(prefs):
        raise RuntimeError(f"KADID row count mismatch: parquet {len(prefs)} vs dmos.csv {len(rows)}")

    # Verify alignment: ref order + human_score == (dmos-1)/4
    dref = [r["ref_img"].split(".")[0] for r in rows]
    if dref != prefs:
        raise RuntimeError("KADID ref_basename order does NOT match dmos.csv — mapping unrecoverable")
    dmos = np.array([float(r["dmos"]) for r in rows])
    hs_expected = (dmos - 1.0) / 4.0
    max_err = float(np.max(np.abs(hs_expected - ph)))
    if max_err > 1e-4:
        raise RuntimeError(f"KADID human_score != (dmos-1)/4; max_err={max_err}")
    print(f"  KADID alignment OK: ref order matches, human_score==(dmos-1)/4 (max_err={max_err:.2e})")

    n = len(rows) if max_rows == 0 else min(max_rows, len(rows))
    with open(out_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "ref_basename", "human_score"])
        for i in range(n):
            ref_path = KADID_IMG / rows[i]["ref_img"]
            dist_path = KADID_IMG / rows[i]["dist_img"]
            if not ref_path.is_file() or not dist_path.is_file():
                raise RuntimeError(f"KADID missing image: {ref_path} or {dist_path}")
            w.writerow([str(ref_path), str(dist_path), prefs[i], f"{ph[i]:.10g}"])
    print(f"  WROTE {out_tsv} ({n} pairs)")
    return n


def build_tid(parquet_path: Path, out_tsv: Path, max_rows: int = 0) -> int:
    tbl = pq.read_table(str(parquet_path), columns=["ref_basename", "human_score"])
    prefs = tbl.column("ref_basename").to_pylist()
    ph = np.asarray(tbl.column("human_score").to_numpy(zero_copy_only=False), dtype=float)

    lines = [l.strip().split() for l in open(TID_DIR / "mos_with_names.txt") if l.strip()]
    if len(lines) != len(prefs):
        raise RuntimeError(f"TID row count mismatch: parquet {len(prefs)} vs mos.txt {len(lines)}")
    mos = np.array([float(a) for a, b in lines])
    names = [b for a, b in lines]
    refs_from_name = [n.split("_")[0].upper() for n in names]
    if refs_from_name != prefs:
        raise RuntimeError("TID ref_basename order does NOT match mos_with_names.txt — mapping unrecoverable")
    hs_expected = mos / 9.0
    max_err = float(np.max(np.abs(hs_expected - ph)))
    if max_err > 1e-4:
        raise RuntimeError(f"TID human_score != mos/9; max_err={max_err}")
    print(f"  TID alignment OK: ref order matches, human_score==mos/9 (max_err={max_err:.2e})")

    n = len(lines) if max_rows == 0 else min(max_rows, len(lines))
    with open(out_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "ref_basename", "human_score"])
        for i in range(n):
            ref_basename = prefs[i]  # I01..I25
            ref_path = TID_REF_PNG / f"{ref_basename}.png"
            # distorted PNG: name stem from bmp filename, .png extension
            dist_stem = Path(names[i]).stem
            dist_path = TID_DIST_PNG / f"{dist_stem}.png"
            if not ref_path.is_file():
                raise RuntimeError(f"TID missing ref: {ref_path}")
            if not dist_path.is_file():
                raise RuntimeError(f"TID missing dist: {dist_path}")
            w.writerow([str(ref_path), str(dist_path), ref_basename, f"{ph[i]:.10g}"])
    print(f"  WROTE {out_tsv} ({n} pairs)")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["kadid", "tid", "both"], default="both")
    ap.add_argument("--canonical", choices=["2026-05-21", "2026-05-18"], default="2026-05-21")
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/v/output/zensim/data-fix-2026-05-25"))
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows; >0 for PoC")
    args = ap.parse_args()

    canon = CANON21 if args.canonical == "2026-05-21" else CANON18
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_poc{args.max_rows}" if args.max_rows else ""

    if args.corpus in ("kadid", "both"):
        print("--- KADID")
        build_kadid(canon / "kadid.parquet", args.out_dir / f"kadid_pairs{suffix}.tsv", args.max_rows)
    if args.corpus in ("tid", "both"):
        print("--- TID")
        build_tid(canon / "tid.parquet", args.out_dir / f"tid_pairs{suffix}.tsv", args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
