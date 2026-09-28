#!/usr/bin/env python3
"""Phase-2d Part D — deterministic imazen26 (wlin7 store) subset selector.

Source table: `uris_tbig.parquet` (208,169 rows; every origin_id already
satisfies the even/odd TRAIN-side rule — all origin last digits are in
{0,2,4,6,8}, so no further origin filter is needed). The block-cache cap
(~6 GB/plane, ~0.4 MB/row at 384x288) bounds the subset at ~12k rows.

Rule (recorded, deterministic, no RNG): sort by (origin_id,
encoded_filename); take rows whose global rank % STRIDE == 0. STRIDE=17
yields ~12.2k rows (~5 GB/plane of block cache).

Output: `imazen26_sub_uris.parquet` — the fetcher's input schema
(passes through all uris_tbig columns verbatim).
"""
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path("/mnt/v/output/zensim/dvifm-screen2d-2026-09-19")
SRC = "/mnt/v/zen/zensim-training/wlin7-pools944-2026-08-30/uris_tbig.parquet"
STRIDE = 17


def main():
    t = pq.read_table(SRC)
    df = t.to_pandas()
    df["_oid"] = df["origin_id"].astype(str)
    df = df.sort_values(["_oid", "encoded_filename"], kind="stable").reset_index(drop=True)
    sel = df[df.index % STRIDE == 0].drop(columns=["_oid"])
    out = OUT / "pairs" / "imazen26_sub_uris.parquet"
    pq.write_table(pa.Table.from_pandas(sel, preserve_index=False), out)
    man = {
        "source": SRC,
        "source_rows": len(df),
        "rule": f"sort by (origin_id, encoded_filename); rank % {STRIDE} == 0",
        "rows": len(sel),
        "origins": int(sel["origin_id"].astype(str).nunique()),
        "out": str(out),
        "out_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "target_col": "score_ssim2",
        "note": "imazen26 = wlin7 store (bigcodec 4-codec sweep over "
                "clean-picker-corpus-2026-06-26 refs); oracle = our fast-ssim2. "
                "TRAIN-side only (all origin last digits in {0,2,4,6,8}).",
    }
    (OUT / "pairs" / "imazen26_sub_uris.manifest.json").write_text(
        json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: man[k] for k in ("rows", "origins", "out")}, indent=1))
    print("codecs:", sel["codec"].value_counts().to_dict())
    print("fetch_mode:", sel["fetch_mode"].value_counts().to_dict())


if __name__ == "__main__":
    main()
