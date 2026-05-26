#!/usr/bin/env python3
"""Build iPhone-14 CVVDP dial-target training parquets for zensim-b-phone.

Joins per-row iPhone-14 CVVDP JOD scores (computed by `zen-metrics batch
--display-model iphone_14_pro`, row-aligned to the canonical KADID/TID
train parquets via fix_kadid_tid_build_pairs.py) onto the 372-feature
canonical parquets, replacing `human_score` with a 0-100 dial value
derived from the iPhone-14 CVVDP via the V12 piecewise-linear band map.

The V12 CVVDP->dial band table (build_v12_cvvdp_substrate.py) is the
canonical, perceptually-anchored CVVDP-JOD -> 0..100 score mapping; the
same family the continuous anchor parquet uses. Monotone, so SROCC vs raw
iPhone-14 CVVDP is preserved (the tracking metric is rank-honest either
way).

Outputs (per corpus) to <out-dir>:
  <corpus>_iphone14_cvvdptgt.parquet  — feature parquet w/ human_score
                                         = dial(cvvdp_iphone14),
                                         + cvvdp_iphone14 raw column for
                                         the SROCC tracking eval.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# V12 (cvvdp_target_JOD, target_score) band table — canonical
# CVVDP->dial mapping (scripts/v_next/v12_cvvdp/build_v12_cvvdp_substrate.py).
V12_BANDS = [
    (10.00, 100.0),
    (9.95, 95.0),
    (9.85, 90.0),
    (9.65, 80.0),   # JND
    (9.30, 65.0),
    (8.50, 50.0),   # JOD
    (7.50, 35.0),
    (6.50, 20.0),
    (5.00, 10.0),
    (3.00, 0.0),
]
# Ascending x for np.interp.
_BX = [b[0] for b in V12_BANDS][::-1]
_BY = [b[1] for b in V12_BANDS][::-1]


def cvvdp_to_dial(cvvdp: np.ndarray) -> np.ndarray:
    """Monotone piecewise-linear CVVDP-JOD -> 0..100 dial (V12 band map).
    Clamps below 3.0 JOD -> 0 and above 10.0 -> 100 (np.interp clamps to
    endpoints by default)."""
    return np.interp(cvvdp, _BX, _BY)


def load_scores_tsv(path: Path) -> np.ndarray:
    """Read the cvvdp_iphone14 score column (row order == parquet order)."""
    scores = []
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        col = next(c for c in r.fieldnames if c.startswith("cvvdp"))
        for row in r:
            try:
                scores.append(float(row[col]))
            except (ValueError, TypeError):
                scores.append(np.nan)
    return np.asarray(scores, dtype=float)


def build_one(corpus: str, parquet: Path, scores_tsv: Path, out: Path) -> int:
    tbl = pq.read_table(str(parquet))
    n = tbl.num_rows
    scores = load_scores_tsv(scores_tsv)
    if len(scores) != n:
        raise RuntimeError(
            f"{corpus}: row count mismatch — parquet {n} vs scores {len(scores)}"
        )
    finite = np.isfinite(scores)
    if finite.sum() < n:
        print(f"  {corpus}: {n - finite.sum()} non-finite CVVDP rows dropped")

    dial = cvvdp_to_dial(scores)  # 0..100 dial target

    # Replace human_score with the dial; add raw iphone14 cvvdp column.
    names = tbl.column_names
    cols = {nm: tbl.column(nm) for nm in names}
    cols["human_score"] = pa.array(dial / 100.0)  # trainer x100 -> 0..100
    cols["cvvdp_iphone14"] = pa.array(scores)
    cols["cvvdp_iphone14_dial"] = pa.array(dial)
    new_names = list(names)
    if "cvvdp_iphone14" not in new_names:
        new_names.append("cvvdp_iphone14")
    if "cvvdp_iphone14_dial" not in new_names:
        new_names.append("cvvdp_iphone14_dial")
    out_tbl = pa.table({nm: cols[nm] for nm in new_names})

    # Keep only finite-CVVDP rows for training cleanliness.
    if finite.sum() < n:
        out_tbl = out_tbl.filter(pa.array(finite))

    pq.write_table(out_tbl, str(out), compression="zstd", compression_level=15)
    print(
        f"  WROTE {out} ({out_tbl.num_rows} rows; "
        f"dial p5={np.nanpercentile(dial[finite],5):.1f} "
        f"p50={np.nanpercentile(dial[finite],50):.1f} "
        f"p95={np.nanpercentile(dial[finite],95):.1f})"
    )
    return out_tbl.num_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-dir",
        default="/mnt/v/zen/zensim-training/canonical-2026-05-21/train",
    )
    ap.add_argument(
        "--scores-dir",
        default="/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25",
    )
    ap.add_argument(
        "--out-dir",
        default="/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25",
    )
    args = ap.parse_args()

    train = Path(args.train_dir)
    scores = Path(args.scores_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for corpus, pq_name in [("kadid", "kadid.parquet"), ("tid", "tid.parquet")]:
        print(f"--- {corpus}")
        n = build_one(
            corpus,
            train / pq_name,
            scores / f"{corpus}_cvvdp_iphone14.tsv",
            out / f"{corpus}_iphone14_cvvdptgt.parquet",
        )
        total += n
    print(f"TOTAL {total} rows across corpora")


if __name__ == "__main__":
    main()
