#!/usr/bin/env python3
"""Build chunks.jsonl in omni-backfill format for the multi-codec sweep.

Output schema (one chunk = N rows of input parquet, indexed by row_range):
    {
      "chunk_id": "...",
      "input_parquet": "multi_codec_input.parquet",
      "input_parquet_r2": "s3://...",
      "row_range": [s, e],
      "source_dir_r2": "s3://...",
      "image_basenames": [...],
      "row_count": ...,
      "run_id": "..."
    }
"""
import argparse
import json
import os
import sys
from pathlib import Path
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-parquet', type=Path, required=True)
    p.add_argument('--input-parquet-r2', type=str, required=True,
                   help='s3:// URI where the input parquet will be uploaded')
    p.add_argument('--source-r2-prefix', type=str, required=True,
                   help='s3:// prefix containing source images')
    p.add_argument('--output-r2-prefix', type=str, required=True,
                   help='s3:// prefix where sidecars + encoded outputs land')
    p.add_argument('--run-id', type=str, required=True)
    p.add_argument('--chunk-size', type=int, default=200)
    p.add_argument('--out', type=Path, default=Path('/tmp/multi_codec_chunks.jsonl'))
    args = p.parse_args()

    t = pq.read_table(str(args.input_parquet), columns=['image_path'])
    n_rows = t.num_rows
    parquet_filename = args.input_parquet.name

    with args.out.open('w') as f:
        for i, start in enumerate(range(0, n_rows, args.chunk_size)):
            end = min(start + args.chunk_size, n_rows)
            slc = t.slice(start, end - start)
            bns = sorted(set(os.path.basename(ip) for ip in slc['image_path'].to_pylist()))
            chunk_id = f"multi-codec-{i:04d}"
            spec = {
                "chunk_id": chunk_id,
                "input_parquet": parquet_filename,
                "input_parquet_r2": args.input_parquet_r2,
                "row_range": [start, end],
                "source_dir_r2": args.source_r2_prefix,
                "image_basenames": bns,
                "row_count": end - start,
                "run_id": args.run_id,
                "out_sidecar_omni": f"{args.output_r2_prefix}/omni/{chunk_id}.parquet",
            }
            f.write(json.dumps(spec))
            f.write("\n")

    n_chunks = (n_rows + args.chunk_size - 1) // args.chunk_size
    print(f"wrote {n_chunks} chunks ({n_rows} rows) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
