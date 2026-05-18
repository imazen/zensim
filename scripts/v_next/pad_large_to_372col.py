#!/usr/bin/env python3
"""Pad cvvdp_iwssim_large_300col_v2.parquet to 372 cols by appending f300..f371 = 0.0.

This is the structural-padding shim that lets the V_22-mix-LARGE-372
trainer load all 5 groups at the same width when the LARGE corpus's
distorted images are unavailable for IW re-extraction.

Rationale: the LARGE corpus (73,300 rows) was scored on vast.ai
workers whose distortion artifacts were ephemeral. Re-encoding 73,300
pairs locally to extract f300..f371 (IW-pool features) is several
hours of work and requires reproducing the exact knob_tuple_json for
each pair across 6 codecs. Padding with zeros lets us run the
ablation NOW — the 4 anchor groups (safesyn / kadid / tid / konjnd)
ALREADY carry real f300..f371 values, so the trainer learns the IW
columns' contribution from them while the LARGE rows act as
"IW-signal-absent" baseline samples.

Output: /mnt/v/zen/zensim-training/2026-05-18-372feat/cvvdp_iwssim_large_372col_padded.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path("/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet")
DST_DIR = Path("/mnt/v/zen/zensim-training/2026-05-18-372feat")
DST = DST_DIR / "cvvdp_iwssim_large_372col_padded.parquet"


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"reading {SRC} ({SRC.stat().st_size / 1e6:.1f} MB)")
    t0 = time.time()
    table = pq.read_table(str(SRC))
    print(f"  rows={table.num_rows} cols={table.num_columns} elapsed={time.time()-t0:.1f}s")

    # Determine current feature columns + their dtype.
    feat_cols = [n for n in table.column_names if n.startswith("f")]
    feat_idx = [int(n[1:]) for n in feat_cols if n[1:].isdigit()]
    max_existing = max(feat_idx)
    print(f"  existing f-cols: {len(feat_idx)} (max={max_existing})")
    assert max_existing == 299, f"expected f0..f299, got max f{max_existing}"

    # Match dtype of the existing f-cols (Float32 or Float64).
    sample_col = table.column(table.column_names.index("f0"))
    feat_dtype = sample_col.type
    print(f"  f0 dtype: {feat_dtype}")

    # Append f300..f371 as all-zero columns of the same dtype.
    n_rows = table.num_rows
    new_table = table
    if feat_dtype == pa.float32():
        zeros = pa.array(np.zeros(n_rows, dtype=np.float32))
    elif feat_dtype == pa.float64():
        zeros = pa.array(np.zeros(n_rows, dtype=np.float64))
    else:
        raise SystemExit(f"unexpected dtype {feat_dtype}")

    t1 = time.time()
    for i in range(300, 372):
        new_table = new_table.append_column(f"f{i}", zeros)
    print(f"  padded to {new_table.num_columns} cols in {time.time()-t1:.1f}s")

    print(f"writing {DST}")
    t2 = time.time()
    pq.write_table(new_table, str(DST), compression="zstd", compression_level=3)
    print(f"  wrote {DST.stat().st_size / 1e6:.1f} MB in {time.time()-t2:.1f}s")
    print("done.")


if __name__ == "__main__":
    main()
