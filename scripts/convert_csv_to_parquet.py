#!/usr/bin/env python3
"""Convert one or more zensim feature CSVs to Parquet (zstd, level 3).

Reads with pyarrow.csv (fast, schema-inferring) and writes columnar parquet.
Designed for the V_22-IW training corpora at
/mnt/v/zen/zensim-training/2026-05-16/v2/*.csv.

Usage:
    python3 scripts/convert_csv_to_parquet.py <csv_path> [<csv_path> ...]

Each input <name>.csv produces a sibling <name>.parquet (same directory).
Existing .parquet files are NOT overwritten unless --force is passed.
"""
import argparse
import sys
from pathlib import Path

import pyarrow.csv as pv
import pyarrow.parquet as pq


def convert(csv_path: Path, force: bool = False) -> tuple[Path, int, int]:
    parquet_path = csv_path.with_suffix('.parquet')
    if parquet_path.exists() and not force:
        print(f'  SKIP {csv_path.name} (parquet exists; use --force to overwrite)', file=sys.stderr)
        return parquet_path, 0, 0
    print(f'  READ {csv_path}...', file=sys.stderr)
    table = pv.read_csv(str(csv_path))
    print(f'  WRITE {parquet_path}: {table.num_rows} rows x {table.num_columns} cols', file=sys.stderr)
    pq.write_table(
        table,
        str(parquet_path),
        compression='zstd',
        compression_level=3,
        use_dictionary=False,
    )
    return parquet_path, table.num_rows, table.num_columns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('csv', nargs='+', type=Path)
    ap.add_argument('--force', action='store_true', help='Overwrite existing .parquet outputs')
    args = ap.parse_args()
    for csv_path in args.csv:
        if not csv_path.is_file():
            print(f'NOT FOUND: {csv_path}', file=sys.stderr)
            return 1
        out, n_rows, n_cols = convert(csv_path, force=args.force)
        if n_rows > 0:
            csv_bytes = csv_path.stat().st_size
            par_bytes = out.stat().st_size
            print(f'  OK   {n_rows} rows x {n_cols} cols, '
                  f'{csv_bytes / 1e6:.1f} MB CSV -> {par_bytes / 1e6:.1f} MB parquet '
                  f'({csv_bytes / par_bytes:.1f}x compression)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
