#!/usr/bin/env python3
"""Resolve a KEYED bigcodec table (`encoded_filename` + `codec`) to fetchable
byte locations — the R1b step that turns "which encode is this row?" into
"where are its bytes?".

Input   a key table from `build_eval_slices_944.py --emit-keys` (or any table
        carrying `encoded_filename` + `codec` + `ref_filename`).
Side    the canonical-picker pair tables
        `<picker-root>/<codec>_lossy/pairs.<split>.parquet`, which carry
        `ref_path` (an individually addressable object), `dist_member`
        (== `encoded_filename`) and `dist_tar`.
Output  `<name>_uris.parquet` — one row per input row, in input order, with
        `ref_local` / `ref_uri` / `dist_uri` / `dist_tar` / `dist_member` /
        `fetch_mode`.

Two fetch modes, decided per dataset by what actually exists on the store
(MEASURED 2026-08-30, recorded in the manifest, not assumed):
  `object`   the per-file encode exists under
             `s3://<bucket>/canonical/2026-06-27/<ds>/encodes/<member>` — a
             plain GET. True for zenjpeg / zenwebp / zenpng / zenjxl-lossless.
  `tarrange` the `_regroup` pass never ran for this dataset, so the bytes live
             only inside `<dist_tar>`; fetched by byte range using the
             prebuilt index `s3://<bucket>/jobs/bf-<tag>-t<N>/variant_index.tsv`
             (`member \\t offset \\t size \\t name`, built by zenmetrics
             `scripts/jobsys/index_tar_byterange.py`). True for zenavif /
             zenjxl-lossy.

The join runs through `join_safety.safe_metric_join` on `encoded_filename` —
a per-pair distortion identifier, never a ref-only key — so a partial or
duplicating join is refused rather than silently broadcast. 100 % resolution
is REQUIRED; any unresolved row aborts.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from join_safety import safe_key_join_arrow  # noqa: E402

# dataset dir  ->  (fetch_mode, tar-index run tag). Measured on the store.
DATASETS = {
    "zenjpeg_lossy": ("object", "bf-zjpeg"),
    "zenwebp_lossy": ("object", "bf-zwebp"),
    "zenpng_lossless": ("object", "bf-zpng"),
    "zenjxl_lossless": ("object", "bf-zjxlm"),
    "zenavif_lossy": ("tarrange", "bf-zavif"),
    # NB the tag names are the OPPOSITE of the obvious reading, verified by
    # reading the indexes: bf-zjxlm-t* holds `zenjxl_q0` (modular = LOSSLESS,
    # 10 boxes) and bf-zjxll-t* holds the 24-box lossy VarDCT run.
    "zenjxl_lossy": ("tarrange", "bf-zjxll"),
}
ENCODES_PREFIX = "canonical/2026-06-27"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True, help="keyed table (parquet)")
    ap.add_argument("--picker-root",
                    default="/mnt/v/output/canonical-picker-2026-06-27")
    ap.add_argument("--split", default="validate")
    ap.add_argument("--ref-local-root",
                    default="/mnt/v/output/clean-picker-corpus-2026-06-26")
    ap.add_argument("--bucket", default="zentrain")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    k = pq.read_table(a.keys)
    need = {"encoded_filename", "codec", "ref_filename"}
    missing = need - set(k.column_names)
    if missing:
        sys.exit(f"ABORT: key table missing {sorted(missing)}")

    sides, seen = [], set()
    for ds, (mode, tag) in DATASETS.items():
        p = Path(a.picker_root) / ds / f"pairs.{a.split}.parquet"
        if not p.is_file():
            continue
        t = pq.read_table(p, columns=["ref_path", "dist_member", "dist_tar"])
        mem = t["dist_member"].to_pylist()
        keep = [i for i, m in enumerate(mem) if m not in seen and not seen.add(m)]
        t = t.take(pa.array(keep))
        n = t.num_rows
        sides.append(pa.table({
            "encoded_filename": t["dist_member"],
            "ref_path": t["ref_path"],
            "dist_tar": t["dist_tar"],
            "dataset": pa.array([ds] * n),
            "fetch_mode": pa.array([mode] * n),
            "index_tag": pa.array([tag] * n),
        }))
    if not sides:
        sys.exit(f"ABORT: no pairs.{a.split}.parquet under {a.picker_root}")
    side = pa.concat_tables(sides)

    out = safe_key_join_arrow(
        k, side, ["encoded_filename"],
        ["ref_path", "dist_tar", "dataset", "fetch_mode", "index_tag"])

    ref_paths = out["ref_path"].to_pylist()
    nun = sum(1 for v in ref_paths if v is None)
    if nun:
        ex = [m for m, v in zip(out["encoded_filename"].to_pylist(), ref_paths)
              if v is None][:3]
        sys.exit(f"ABORT: {nun} of {out.num_rows} rows did not resolve to a pair row "
                 f"(e.g. {ex}) — R1b requires 100% resolution, never a partial cut")
    out = out.to_pydict()

    def box_of(tar: str) -> int:
        m = re.search(r"box-(\d+)\.tar$", tar)
        return int(m.group(1)) if m else -1

    n_rows = len(out["ref_path"])
    out["ref_local"] = [str(Path(a.ref_local_root) / Path(r).name)
                        for r in out["ref_path"]]
    out["dist_uri"] = [
        (f"s3://{a.bucket}/{ENCODES_PREFIX}/{ds}/encodes/{m}" if fm == "object" else "")
        for ds, m, fm in zip(out["dataset"], out["encoded_filename"], out["fetch_mode"])
    ]
    out["tar_index_uri"] = [
        (f"s3://{a.bucket}/jobs/{tag}-t{box_of(tar)}/variant_index.tsv"
         if fm == "tarrange" else "")
        for tag, tar, fm in zip(out["index_tag"], out["dist_tar"], out["fetch_mode"])
    ]
    out["dist_member"] = out["encoded_filename"]

    n_missing_ref = sum(1 for p in out["ref_local"] if not os.path.exists(p))
    if n_missing_ref:
        sys.exit(f"ABORT: {n_missing_ref} reference PNGs absent under "
                 f"{a.ref_local_root} — refusing a partial pair list")

    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({c: pa.array(v) for c, v in out.items()})
    pq.write_table(tbl, op, compression="zstd", compression_level=7)

    from collections import Counter
    by_mode = Counter(zip(out["dataset"], out["fetch_mode"]))
    print(f"{Path(a.keys).name}: {n_rows} rows -> {op}")
    for (ds, fm), n in sorted(by_mode.items()):
        print(f"   {ds:20s} {fm:9s} {n}")
    man = {
        "input_keys": str(a.keys), "input_keys_sha256": sha256_file(Path(a.keys)),
        "output": str(op), "output_sha256": sha256_file(op),
        "rows": int(n_rows), "split": a.split,
        "picker_root": a.picker_root, "ref_local_root": a.ref_local_root,
        "resolution": "100% (gated)",
        "by_dataset": {f"{d}:{m}": int(n) for (d, m), n in by_mode.items()},
        "join": "join_safety.safe_key_join_arrow on encoded_filename (per-pair id)",
    }
    Path(str(op) + "._MANIFEST.json").write_text(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
