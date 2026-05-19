#!/usr/bin/env python3
"""EX-PERCENTILE-POOL: convert P² validation CSVs to parquets for
bake_verdict consumption.

bake_verdict expects parquets with `ref_basename, human_score, f0..f371`
columns. We have the CSVs; this just writes them as parquet.
"""
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

P2_DIR = Path("/mnt/v/zen/zensim-training/2026-05-18-percentile-pool")

CORPORA = ["cid22", "kadid", "tid", "konjnd", "aic3"]


def main():
    for name in CORPORA:
        csv_p = P2_DIR / f"{name}_features_372col_p2.csv"
        out = P2_DIR / f"{name}_features_372col_p2.parquet"
        if not csv_p.exists():
            print(f"[{name}] missing {csv_p}; skipping")
            continue
        df = pd.read_csv(csv_p, low_memory=False)
        print(f"[{name}] read {len(df)} rows, {len(df.columns)} cols")
        pq.write_table(pa.Table.from_pandas(df), out, compression="zstd")
        print(f"[{name}] wrote {out}")


if __name__ == "__main__":
    main()
