#!/usr/bin/env python3
"""V11 task #7 (2026-05-24): backfill CVVDP + IW-SSIM scores into
the canonical cid22_train.parquet.

The original v11_extract_cid22_train.py emitted cvvdp/iwssim/mix_* as
all-null because zen-metrics had not yet scored those metrics on the
17,611 cid22-train pairs. This script joins the new TSV outputs from
zen-metrics into the existing parquet and rebuilds the file in-place
under a date suffix.

Inputs (must exist; produced by `zen-metrics batch` runs):
  - canonical-2026-05-21/_workspace/cid22_train_pairs.tsv  (composite join key)
  - canonical-2026-05-21/_workspace/cid22_train_ssim2.tsv  (already populated)
  - canonical-2026-05-21/_workspace/cid22_train_cvvdp.tsv  (run with --metric cvvdp)
  - canonical-2026-05-21/_workspace/cid22_train_iwssim.tsv (run with --metric iwssim-gpu)

Outputs:
  - canonical-2026-05-21/train/cid22_train.parquet  (in-place update — backed up to .bak first)
  - _MANIFEST.json updated with new sha256 + canonical_metadata note

Join key: (ref_basename, codec, q) — the composite from cid22_train_pairs.tsv.
The parquet's `ref_basename` column is actually a composite `"<basename>|<codec>|<q>"`
per extract_features_372col --corpus cid22_train; we split on '|' to join.

Mix-target columns rebuilt (consistent with canonical-2026-05-18 anchors):
  - cvvdp_log_norm   = clamp((cvvdp + 2.1188) / 15.93, 0, 1) · 100
    (safesyn-anchored: lo=-2.1188 hi=13.8155, range=15.93)
  - iwssim_log_norm  = ((1 - iwssim) → log_norm via safesyn max_log=13.7202)
  - mix_cv40_iw60   = 0.4·cvvdp_log_norm + 0.6·iwssim_log_norm
  - mix_target      = mix_cv40_iw60 (canonical default)

For the mix grid (cv25..cv75), only mix_cv40_iw60 is populated by
this script. The other mix columns remain null — extending them would
require recomputing the full 11-knob grid, which is downstream of any
ship decision.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CANONICAL = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21")
TRAIN_PARQUET = CANONICAL / "train" / "cid22_train.parquet"
MANIFEST_PATH = CANONICAL / "_MANIFEST.json"
WORKSPACE = CANONICAL / "_workspace"
PAIRS_TSV = WORKSPACE / "cid22_train_pairs.tsv"
CVVDP_TSV = WORKSPACE / "cid22_train_cvvdp.tsv"
IWSSIM_TSV = WORKSPACE / "cid22_train_iwssim.tsv"

# Same constants used by v11_extract_cid22_train.py (safesyn-anchored).
CVVDP_LO = -2.1188
CVVDP_HI = 13.8155
CVVDP_RANGE = CVVDP_HI - CVVDP_LO  # = 15.9343
IWSSIM_MAX_LOG = 13.7202


def cvvdp_log_norm(cvvdp_arr: np.ndarray) -> np.ndarray:
    """Match canonical-2026-05-18 cvvdp_log_norm column.
    log_norm = clamp((cvvdp - lo) / (hi - lo), 0, 1) * 100.
    cvvdp ∈ [0, 10] JOD scale → log_norm ∈ [~13, ~76] (lo/hi pulled
    from safesyn's empirical bounds)."""
    return np.clip((cvvdp_arr - CVVDP_LO) / CVVDP_RANGE, 0.0, 1.0) * 100.0


def iwssim_log_norm(iwssim_arr: np.ndarray) -> np.ndarray:
    """Match canonical-2026-05-18 iwssim_log_norm column.
    log_norm = log(1 / (1 - iwssim + 1e-6)) / max_log * 100, clamped.
    iwssim ∈ [0, 1] (1 = identical) → log_norm ∈ [0, ~100]."""
    # Guard for iwssim very close to 1.0 (would blow log).
    safe = np.clip(iwssim_arr, 0.0, 1.0 - 1e-6)
    raw = np.log(1.0 / (1.0 - safe + 1e-6))
    return np.clip(raw / IWSSIM_MAX_LOG * 100.0, 0.0, 100.0)


def load_tsv_join_keyed(path: Path, score_col: str) -> dict:
    """Load a zen-metrics batch output TSV. Returns dict mapping
    (basename, codec, q) → score."""
    out = {}
    n = 0
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            basename = row["ref_basename"]
            codec = row["codec"]
            q = row["q"]
            try:
                score = float(row[score_col])
            except (KeyError, ValueError) as e:
                raise RuntimeError(
                    f"{path}: row missing/malformed {score_col!r}: {row} ({e})"
                )
            if not np.isfinite(score):
                continue  # silently drop NaN/Inf — these become nulls in the merge
            out[(basename, codec, q)] = score
            n += 1
    print(f"  loaded {n:,} rows from {path.name} (col={score_col})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="don't write the parquet, just report join stats")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the .bak copy (dangerous; only for re-runs)")
    args = ap.parse_args()

    for p in (TRAIN_PARQUET, PAIRS_TSV, CVVDP_TSV, IWSSIM_TSV):
        if not p.exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 2

    print(f"[1/5] loading existing parquet {TRAIN_PARQUET.name} …")
    tbl_in = pq.read_table(TRAIN_PARQUET)
    n_rows = tbl_in.num_rows
    n_cols = tbl_in.num_columns
    print(f"      {n_rows:,} rows × {n_cols} cols")

    composite_arr = tbl_in.column("ref_basename").to_pylist()
    # ref_basename has composite "<basename>|<codec>|<q>" — split.
    split_keys = []
    bad = 0
    for s in composite_arr:
        parts = s.split("|")
        if len(parts) != 3:
            bad += 1
            split_keys.append(None)
        else:
            split_keys.append((parts[0], parts[1], parts[2]))
    if bad:
        print(f"  WARNING: {bad} composite keys with split errors")

    print(f"[2/5] loading cvvdp + iwssim score TSVs …")
    cvvdp_map = load_tsv_join_keyed(CVVDP_TSV, "cvvdp_imazen_v0_0_1")
    iwssim_map = load_tsv_join_keyed(IWSSIM_TSV, "iwssim_imazen_v0_0_1")

    cvvdp_vals = np.full(n_rows, np.nan, dtype=np.float64)
    iwssim_vals = np.full(n_rows, np.nan, dtype=np.float64)
    miss_cvvdp = 0
    miss_iwssim = 0
    for i, key in enumerate(split_keys):
        if key is None:
            miss_cvvdp += 1
            miss_iwssim += 1
            continue
        if key in cvvdp_map:
            cvvdp_vals[i] = cvvdp_map[key]
        else:
            miss_cvvdp += 1
        if key in iwssim_map:
            iwssim_vals[i] = iwssim_map[key]
        else:
            miss_iwssim += 1

    coverage_cvvdp = 100.0 * (1.0 - miss_cvvdp / n_rows)
    coverage_iwssim = 100.0 * (1.0 - miss_iwssim / n_rows)
    print(f"      cvvdp coverage:  {coverage_cvvdp:.2f}%  ({n_rows-miss_cvvdp:,}/{n_rows:,})")
    print(f"      iwssim coverage: {coverage_iwssim:.2f}%  ({n_rows-miss_iwssim:,}/{n_rows:,})")

    # Derived columns.
    cvvdp_log = np.where(np.isfinite(cvvdp_vals),
                         cvvdp_log_norm(cvvdp_vals),
                         np.nan)
    iwssim_log = np.where(np.isfinite(iwssim_vals),
                          iwssim_log_norm(iwssim_vals),
                          np.nan)
    mix_cv40_iw60 = np.where(
        np.isfinite(cvvdp_log) & np.isfinite(iwssim_log),
        0.4 * cvvdp_log + 0.6 * iwssim_log,
        np.nan,
    )

    print(f"[3/5] building updated parquet …")
    new_columns = {
        "cvvdp_score": cvvdp_vals,
        "cvvdp_log_norm": cvvdp_log,
        "iwssim": iwssim_vals,
        "iwssim_log_norm": iwssim_log,
        "mix_cv40_iw60": mix_cv40_iw60,
        # mix_target alias — used by --target-column "mix_target" callers.
        "mix_target": mix_cv40_iw60,
    }
    # Build new table by replacing the targeted columns. PyArrow doesn't
    # have set_column-by-name; rebuild via fields() iteration.
    new_arrays = []
    new_field_names = []
    for field in tbl_in.schema:
        col = tbl_in.column(field.name)
        if field.name in new_columns:
            new_arr = pa.array(new_columns[field.name], type=pa.float64())
            new_arrays.append(new_arr)
        else:
            new_arrays.append(col)
        new_field_names.append(field.name)

    tbl_out = pa.Table.from_arrays(new_arrays, names=new_field_names)

    # Sanity-check ranges.
    for col_name in ["cvvdp_score", "iwssim", "mix_cv40_iw60"]:
        arr = np.array(tbl_out.column(col_name).to_pylist(), dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            print(f"      {col_name}: n_finite={finite.size:,} "
                  f"min={finite.min():.3f} p50={np.median(finite):.3f} "
                  f"max={finite.max():.3f}")

    if args.dry_run:
        print(f"[4/5] DRY-RUN — no parquet written")
        return 0

    # Backup before overwrite.
    if not args.no_backup:
        bak = TRAIN_PARQUET.with_suffix(".bak.parquet")
        shutil.copy2(TRAIN_PARQUET, bak)
        print(f"      backup written to {bak}")
    print(f"[4/5] writing new parquet to {TRAIN_PARQUET} …")
    pq.write_table(tbl_out, TRAIN_PARQUET, compression="zstd",
                   compression_level=15)
    new_size = TRAIN_PARQUET.stat().st_size

    print(f"[5/5] updating _MANIFEST.json …")
    new_sha = hashlib.sha256(TRAIN_PARQUET.read_bytes()).hexdigest()
    manifest = json.loads(MANIFEST_PATH.read_text())
    updated = False
    for entry in manifest["entries"]:
        if entry["path"] == "train/cid22_train.parquet":
            entry["rows"] = tbl_out.num_rows
            entry["columns"] = tbl_out.num_columns
            entry["byte_size"] = new_size
            entry["sha256"] = new_sha
            entry.setdefault("canonical_metadata", {})
            entry["canonical_metadata"]["schema_version"] = "canonical-2026-05-24.v2"
            entry["canonical_metadata"]["schema_note"] = (
                "task #7 (2026-05-24) backfill: cvvdp / cvvdp_log_norm / iwssim / "
                "iwssim_log_norm / mix_cv40_iw60 / mix_target populated via zen-metrics "
                "batch (cvvdp + iwssim-gpu) on the 17,611 pairs. Other mix_* columns "
                "remain null by design. CLAUDE.md 'CID22 is VALIDATION-ONLY' rule "
                "unchanged: NEVER use human MOS as a training target on this corpus."
            )
            updated = True
            break
    if not updated:
        print(f"WARNING: train/cid22_train.parquet entry not found in manifest", file=sys.stderr)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"      sha256={new_sha[:16]}…  size={new_size//1024:,} KB")
    print(f"done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
