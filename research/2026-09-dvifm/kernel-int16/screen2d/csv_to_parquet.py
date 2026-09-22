#!/usr/bin/env python3
"""Convert an extract_features_372col CSV (ref_basename,human_score,f0..fN)
to the parquet layout parquet_loader::load_parquet expects: a human_score
target column plus a contiguous f0..f{n-1} float feature run.

Usage: csv_to_parquet.py <in.csv> <out.parquet>
"""
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    df = pd.read_csv(src)
    fcols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    fcols.sort(key=lambda c: int(c[1:]))
    ids = [int(c[1:]) for c in fcols]
    if ids != list(range(len(ids))):
        raise SystemExit(
            f"non-contiguous feature columns: 0..{len(ids)-1} expected, got gaps"
        )
    if "human_score" not in df.columns:
        raise SystemExit("missing human_score column")
    # parquet_loader wants the contiguous run f0..f{n-1}; keep only meta +
    # the feature run so the admission check cannot see stray f-cols.
    keep = ["ref_basename", "human_score"] + fcols
    df = df[keep]
    for c in fcols + ["human_score"]:
        df[c] = df[c].astype("float64")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, dst, compression="zstd")
    print(f"wrote {dst}: {table.num_rows} rows x {table.num_columns} cols "
          f"(f0..f{len(ids)-1})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
