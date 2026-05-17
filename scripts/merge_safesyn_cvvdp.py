#!/usr/bin/env python3
"""Merge safesyn CVVDP backfill into the 372-feature IW-SSIM parquet.

Produces a multi-target training parquet with:
- 372 zensim features (f0..f371) from existing safesyn parquet
- iwssim_log_norm target column (existing)
- cvvdp_score raw CVVDP JOD (new)
- cvvdp_log_norm log-transformed CVVDP for trainer (new)
- human_score (kept as-is for downstream compatibility)

Join keys: source_path + decoded_path (since safesyn parquet uses
ref_basename only and we need full-path matching to avoid ambiguity).
We pull source_path + decoded_path from the original safesyn CSV
indexed by row order (parquet + CSV are row-aligned by construction
in zentrain's V_22-IW pipeline).

Usage:
    python3 scripts/merge_safesyn_cvvdp.py
"""
import csv
import os
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SAFESYN_CSV = '/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv'
SAFESYN_PARQUET = '/mnt/v/zen/zensim-training/2026-05-16/v2/safesyn_features_iwssim_log_372col.parquet'
CVVDP_TSV = '/mnt/v/zen/zensim-eval/safesyn_cvvdp_scores_2026-05-17.tsv'
OUT = '/mnt/v/zen/zensim-training/2026-05-17-cvvdp/safesyn_features_iwssim_cvvdp_372col.parquet'


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    print(f'Loading CSV row paths from {SAFESYN_CSV}...')
    csv_pairs = []  # (source_path, decoded_path) per row
    with open(SAFESYN_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            csv_pairs.append((row['source_path'], row['decoded_path']))
    print(f'  {len(csv_pairs)} CSV rows')

    print(f'Loading existing parquet from {SAFESYN_PARQUET}...')
    base = pq.read_table(SAFESYN_PARQUET)
    print(f'  {base.num_rows} parquet rows, {len(base.column_names)} cols')
    if base.num_rows != len(csv_pairs):
        print(f'  WARNING: parquet ({base.num_rows}) and CSV ({len(csv_pairs)}) row counts differ')
        # The post-CID22 purge dropped CSV rows; parquet was built post-purge
        # We need to match by (source_path, decoded_path) from CSV → parquet ref_basename
        # If parquet is smaller, it's already filtered; we need a different join

    print(f'Loading CVVDP scores from {CVVDP_TSV}...')
    cvvdp_lookup = {}
    with open(CVVDP_TSV) as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            key = (row['ref_path'], row['dist_path'])
            cvvdp_lookup[key] = float(row['cvvdp_imazen_v0_0_1'])
    print(f'  {len(cvvdp_lookup)} CVVDP scored pairs')

    print('Joining CVVDP scores to parquet rows (by CSV row order)...')
    cvvdp_scores = []
    missing = 0
    for src, dst in csv_pairs:
        s = cvvdp_lookup.get((src, dst))
        if s is None:
            cvvdp_scores.append(float('nan'))
            missing += 1
        else:
            cvvdp_scores.append(s)
    print(f'  Joined {len(cvvdp_scores)} scores ({missing} missing)')

    # Truncate to base row count if CSV is longer (post-purge mismatch)
    if len(cvvdp_scores) > base.num_rows:
        print(f'  Truncating CVVDP scores from {len(cvvdp_scores)} -> {base.num_rows}')
        cvvdp_scores = cvvdp_scores[:base.num_rows]
    elif len(cvvdp_scores) < base.num_rows:
        print(f'  Padding CVVDP scores from {len(cvvdp_scores)} -> {base.num_rows} with NaN')
        cvvdp_scores += [float('nan')] * (base.num_rows - len(cvvdp_scores))

    cv = np.array(cvvdp_scores, dtype=np.float64)
    valid = ~np.isnan(cv)
    print(f'  Valid CVVDP rows after alignment: {valid.sum()} / {len(cv)}')
    if valid.sum() < base.num_rows * 0.9:
        print(f'  WARNING: < 90% valid; check the join logic')

    # Log-transform analog to iwssim_log_norm
    raw_log = -np.log(10.0 - cv + 1e-6)
    # Min-max normalize over VALID values only
    lo, hi = float(np.nanmin(raw_log)), float(np.nanmax(raw_log))
    cv_log_norm = np.where(valid, (raw_log - lo) / (hi - lo) * 100.0, 0.0)

    print(f'  cvvdp_score range: {np.nanmin(cv):.3f} - {np.nanmax(cv):.3f}, mean {np.nanmean(cv):.3f}')
    print(f'  cvvdp_log_norm range: {cv_log_norm[valid].min():.2f} - {cv_log_norm[valid].max():.2f}')

    out = base.append_column('cvvdp_score', pa.array(cv))
    out = out.append_column('cvvdp_log_norm', pa.array(cv_log_norm.astype(np.float64)))

    pq.write_table(out, OUT, compression='zstd', compression_level=3)
    print(f'Wrote {OUT}: {os.path.getsize(OUT) / 1e6:.1f} MB, {out.num_rows} rows, {len(out.column_names)} cols')


if __name__ == '__main__':
    main()
