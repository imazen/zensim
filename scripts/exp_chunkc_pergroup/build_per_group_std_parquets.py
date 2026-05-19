#!/usr/bin/env python3
"""Build per-group-standardized training + validation parquets for EXP-CHUNKC-PERGROUP.

Per-group standardizer: for each training group (safesyn, kadid, tid, konjnd,
cvvdp_iwssim_large), compute (mu, sigma) over the 19 EX-4 Chunk C features
(f324..f342), then z-score within the group. Zero-fill corpora (safesyn,
cvvdp_iwssim_large) keep zeros since their data is all-zero.

For each validation corpus (cid22, kadid, tid, konjnd, aic3), compute its own
(mu, sigma) over the SAME 19 features and standardize. Per-corpus inference-time
standardization is the simplest answer for cross-corpus distribution shift —
each corpus stands on its own.

Output:
  /mnt/v/zen/zensim-training/2026-05-18-chunkc-pergroup/{group}_per_group_std.parquet
  /mnt/v/zen/zensim-training/2026-05-18-chunkc-pergroup/val/{corpus}_per_group_std.parquet
  /mnt/v/zen/zensim-training/2026-05-18-chunkc-pergroup/standardizers.json (metadata)

Schema: identical to source parquets except f324..f342 are replaced with z-scored
values. All other features (f0..f323) and target columns are unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC_DIR = Path("/mnt/v/zen/zensim-training/2026-05-18-extfeat")
OUT_DIR = Path("/mnt/v/zen/zensim-training/2026-05-18-chunkc-pergroup")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "val").mkdir(parents=True, exist_ok=True)

CHUNKC_FEATURES = [f"f{i}" for i in range(324, 343)]  # f324..f342

TRAIN_GROUPS = {
    "safesyn": "safesyn_extfeat_343.parquet",
    "kadid": "kadid_extfeat_343.parquet",
    "tid": "tid_extfeat_343.parquet",
    "konjnd": "konjnd_extfeat_343.parquet",
    "cvvdp_iwssim_large": "cvvdp_iwssim_large_extfeat_343.parquet",
}

VAL_CORPORA = {
    "cid22": "cid22_extfeat_343.parquet",
    "kadid": "kadid_val_extfeat_343.parquet",
    "tid": "tid_val_extfeat_343.parquet",
    "konjnd": "konjnd_val_extfeat_343.parquet",
    "aic3": "aic3_extfeat_343.parquet",
}


def compute_standardizer(table: pa.Table) -> dict[str, tuple[float, float]]:
    """Compute (mu, sigma) for each f324..f342 column.

    If a column is all zero, returns mu=0, sigma=1.0 (so z=0 stays 0; we don't
    floor sigma at 1e-8 here because that would explode the small numeric
    noise; we keep zero-fill as zero).
    """
    out = {}
    for col in CHUNKC_FEATURES:
        if col not in table.column_names:
            raise RuntimeError(f"missing column {col}")
        arr = table.column(col).to_numpy().astype(np.float64)
        nz = np.count_nonzero(arr)
        if nz == 0:
            out[col] = (0.0, 1.0)  # zero-fill stays zero, sigma=1 to keep z=0 as 0
            continue
        mu = float(arr.mean())
        sigma = float(arr.std())
        if sigma < 1e-8:
            sigma = 1.0  # degenerate (constant non-zero): keep z=0
        out[col] = (mu, sigma)
    return out


def apply_standardizer(table: pa.Table, std: dict[str, tuple[float, float]]) -> pa.Table:
    """Z-score the f324..f342 columns of `table` using `std`.

    Returns a new pyarrow Table with the 19 columns replaced.
    """
    cols = {}
    for name in table.column_names:
        if name in std:
            mu, sigma = std[name]
            arr = table.column(name).to_numpy().astype(np.float64)
            z = (arr - mu) / sigma
            cols[name] = pa.array(z.astype(np.float64))
        else:
            cols[name] = table.column(name)
    return pa.Table.from_pydict(cols)


def main():
    metadata = {"date": "2026-05-18", "experiment": "exp-chunkc-pergroup", "groups": {}}

    # === Training groups ===
    for group_name, fname in TRAIN_GROUPS.items():
        src = SRC_DIR / fname
        if not src.exists():
            raise RuntimeError(f"missing source parquet: {src}")
        print(f"[train] {group_name}: loading {src.name}")
        table = pq.read_table(src)
        std = compute_standardizer(table)
        non_zero_count = sum(1 for c, (_, s) in std.items() if s != 1.0 or c not in CHUNKC_FEATURES)
        print(f"  {group_name}: rows={table.num_rows}, std with non-zero data: {sum(1 for c,(m,s) in std.items() if not (m==0 and s==1.0))}/{len(CHUNKC_FEATURES)} features")

        new_table = apply_standardizer(table, std)
        out = OUT_DIR / f"{group_name}_per_group_std.parquet"
        pq.write_table(new_table, out, compression="zstd", compression_level=15)
        print(f"  wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)")

        metadata["groups"][group_name] = {
            "source": str(src),
            "rows": table.num_rows,
            "standardizer": {c: {"mu": float(m), "sigma": float(s)} for c, (m, s) in std.items()},
            "output": str(out),
        }

    # === Validation corpora ===
    metadata["val_corpora"] = {}
    for corpus_name, fname in VAL_CORPORA.items():
        src = SRC_DIR / fname
        if not src.exists():
            raise RuntimeError(f"missing val parquet: {src}")
        print(f"[val] {corpus_name}: loading {src.name}")
        table = pq.read_table(src)
        std = compute_standardizer(table)
        new_table = apply_standardizer(table, std)
        out = OUT_DIR / "val" / f"{corpus_name}_per_group_std.parquet"
        pq.write_table(new_table, out, compression="zstd", compression_level=15)
        print(f"  wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
        metadata["val_corpora"][corpus_name] = {
            "source": str(src),
            "rows": table.num_rows,
            "standardizer": {c: {"mu": float(m), "sigma": float(s)} for c, (m, s) in std.items()},
            "output": str(out),
        }

    # Save metadata
    meta_out = OUT_DIR / "standardizers.json"
    with meta_out.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nstandardizer metadata: {meta_out}")
    print(f"output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
