#!/usr/bin/env python3
"""build_anchor944_dial.py — materialize the registered SOTA-944 dial anchor
as ONE parquet (packaging pass; benchmarks/sota944_campaign_2026-08-03.md,
"REGISTERED APPENDIX — the packaging pass").

The §3d anchor (FIXED for every arm-A/B cell of the campaign) exists only as
fit-chain flags (`--anchor-parquet ... --anchor-stride ...` on
`bake_dial_refit fit-lasso`); `add-spline` and `pack` take a single anchor
parquet. This builder materializes the IDENTICAL row set once, so every
packaging step consumes the same bytes:

    ext_safesyn_full.parquet   stride 139
    ext_cid22_train201.parquet stride  44
    ext_kadid.parquet          stride  25
    ext_tid.parquet            stride   7
    target_score = max(human_score * 100, -100)   (upper unclipped)

Stride rule matches the fit-chain loader exactly (rows 0, s, 2s, ...).
The 372-era `multiband_anchor_dial100.parquet` (the shipped-B/v47 anchor,
`pack`'s default) is a REGIME VIOLATION at 944 (campaign amendment 2) and
must not be used for 944 bakes.

CID22-49 human MOS is NOT involved: cid22_train201 is the training-legal
ssim2-anchored leg; the cid22val 49-ref holdout is untouched.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

EXT_ROOT = Path("/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01")
LEGS = [  # (leg file, stride) — frozen in campaign §3d; do not edit.
    ("ext_safesyn_full.parquet", 139),
    ("ext_cid22_train201.parquet", 44),
    ("ext_kadid.parquet", 25),
    ("ext_tid.parquet", 7),
]
N_FEAT = 944
CLIP_MIN = -100.0
OUT = EXT_ROOT / "anchor944_dial.parquet"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    feat_cols = [f"f{i}" for i in range(N_FEAT)]
    parts, prov = [], []
    for name, stride in LEGS:
        p = EXT_ROOT / name
        t = pq.read_table(p, columns=["human_score"] + feat_cols)
        take = np.arange(0, t.num_rows, stride, dtype=np.int64)
        sub = t.take(pa.array(take))
        y = pc.multiply(pc.cast(sub["human_score"], pa.float64()), 100.0)
        y = pc.max_element_wise(y, CLIP_MIN)
        cols = {"anchor_leg": pa.array([name.removeprefix("ext_").removesuffix(".parquet")] * sub.num_rows),
                "target_score": y}
        for c in feat_cols:
            cols[c] = sub[c]
        parts.append(pa.table(cols))
        prov.append({"leg": str(p), "sha256": sha256_file(p), "rows_total": t.num_rows,
                     "stride": stride, "rows_taken": sub.num_rows})
        print(f"  {name}: {sub.num_rows:,} of {t.num_rows:,} rows (stride {stride})", flush=True)
    full = pa.concat_tables(parts)
    ts = np.asarray(full["target_score"], dtype=np.float64)
    assert full.num_rows >= 50, f"anchor too small: {full.num_rows}"
    pq.write_table(full, OUT, compression="zstd", compression_level=7)
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent.parent),
         "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    (OUT.parent / (OUT.name + "._MANIFEST.json")).write_text(json.dumps({
        "group": "anchor944_dial — the registered SOTA-944 §3d dial anchor, materialized",
        "sha256": sha256_file(OUT), "rows": full.num_rows,
        "target_rule": "target_score = max(human_score * 100, -100); upper unclipped",
        "target_stats": {"min": float(ts.min()), "max": float(ts.max()),
                         "mean": float(ts.mean())},
        "build_commit": commit, "legs": prov,
        "note": "multiband_anchor_dial100 (372-era) is a regime violation at 944 "
                "(campaign amendment 2); this is the registered same-role anchor.",
    }, indent=1))
    print(f"wrote {OUT}: {full.num_rows:,} rows, target [{ts.min():.1f}, {ts.max():.1f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
