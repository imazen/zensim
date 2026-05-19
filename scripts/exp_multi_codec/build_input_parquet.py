#!/usr/bin/env python3
"""Build the input parquet that omni_backfill_chunk_worker.sh expects.

Each row is one (image_path, codec, q, knob_tuple_json) cell that the worker
will encode + score. The 200 sources × multi-codec grid produces ~22,400 rows.
Upload to R2 + reference from chunks.jsonl.

Schema matches what the omni worker queries from .row_range:
    image_path : string
    codec      : string
    q          : uint32
    knob_tuple_json : string
"""
import json
import os
import sys
from pathlib import Path
from itertools import product
import pyarrow as pa
import pyarrow.parquet as pq

SOURCES_FILE = Path('/tmp/iwssim_200_sources.txt')
SOURCE_R2_PREFIX = 's3://zentrain/multi-codec-2026-05-18/sources'  # we'll upload sources there
OUT_PATH = Path('/tmp/multi_codec_input.parquet')

Q_GRID = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

KNOB_GRIDS = {
    "zenwebp": {"method": [4, 6]},
    "zenavif": {"speed": [3, 5, 7], "complex_prediction_modes": [False, True]},
    "zenjxl": {"effort": [5, 7], "distance": [0.1, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0], "butteraugli_iters": [0, 1]},
}


def expand_knob(knob_dict):
    """Cartesian product of knob lists → list of knob_tuple_json strings."""
    if not knob_dict:
        return ["{}"]
    keys = list(knob_dict.keys())
    vals = [knob_dict[k] for k in keys]
    out = []
    for combo in product(*vals):
        d = {k: v for k, v in zip(keys, combo)}
        # Use compact JSON like existing sidecars
        out.append(json.dumps(d, separators=(",", ":"), sort_keys=True))
    return out


def main():
    sources = []
    with SOURCES_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                sources.append(line)
    sources.sort()
    print(f"{len(sources)} sources", file=sys.stderr)

    rows_image_path = []
    rows_codec = []
    rows_q = []
    rows_knob = []

    for codec, knob_dict in KNOB_GRIDS.items():
        knob_jsons = expand_knob(knob_dict)
        # For jxl, q is dummy — but we still emit one q value (5) to avoid
        # multiplying jxl rows by 10. The worker can be configured to skip q
        # for jxl, OR we accept duplicate scoring on the same effective cell.
        if codec == "zenjxl":
            q_values = [5]  # single dummy q; distance dominates
        else:
            q_values = Q_GRID
        for src in sources:
            # image_path uses R2 prefix-style path (matches existing rows).
            ip = f"{SOURCE_R2_PREFIX}/{src}"
            for q in q_values:
                for k in knob_jsons:
                    rows_image_path.append(ip)
                    rows_codec.append(codec)
                    rows_q.append(q)
                    rows_knob.append(k)

    n = len(rows_image_path)
    print(f"total rows: {n}", file=sys.stderr)

    tbl = pa.table({
        "image_path": pa.array(rows_image_path, type=pa.string()),
        "codec": pa.array(rows_codec, type=pa.string()),
        "q": pa.array(rows_q, type=pa.uint32()),
        "knob_tuple_json": pa.array(rows_knob, type=pa.string()),
    })
    pq.write_table(tbl, str(OUT_PATH), compression='zstd', compression_level=15)
    sz = os.path.getsize(OUT_PATH)
    print(f"wrote {OUT_PATH} ({n} rows, {sz/1e6:.2f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
