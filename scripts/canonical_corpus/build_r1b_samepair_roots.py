#!/usr/bin/env python3
"""Build the SAME-PAIR 372 / 944 root pair for the R1b family axes.

A 372-input bake (shipped B) and a 944-class candidate can only be compared on
the family axes (nonphoto / imazen26 / hfnlproxy) if they are read on the SAME
pairs. R1b extracts those pairs at both regimes from one pairs TSV, but v1's
feature vector LENGTH depends on image size — a rendition too small for the 4th
scale emits 3 scales x 93 = 279 features instead of 372 (MEASURED: 453 of 6,953
imazen26 slice rows, and similarly for the other two slices). Those rows have no
372 vector at all, so they cannot enter a 372 read.

This script takes the intersection: rows whose v1 CSV carries the full 372, and
writes BOTH roots restricted to exactly those rows, in the same order. Every
short row is COUNTED and recorded in the manifest — never silently dropped.

  build_r1b_samepair_roots.py --v1-csv-dir D --pools-root P --out-372 A --out-944 B
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SLICES = ["imazen26", "nonphoto", "hfnlproxy"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-csv-dir", required=True)
    ap.add_argument("--pools-root", required=True)
    ap.add_argument("--out-372", required=True)
    ap.add_argument("--out-944", required=True)
    ap.add_argument("--n-feat-372", type=int, default=372)
    a = ap.parse_args()
    o372, o944 = Path(a.out_372), Path(a.out_944)
    o372.mkdir(parents=True, exist_ok=True)
    o944.mkdir(parents=True, exist_ok=True)
    man = {"description": "R1b same-pair 372/944 roots: the D1 validate slices "
                          "restricted to rows that HAVE a full v1-372 vector, so a "
                          "372-input bake and a 944-class candidate read identical "
                          "pairs on the family axes.",
           "pools_root": a.pools_root, "v1_csv_dir": a.v1_csv_dir, "slices": {}}
    want = 2 + a.n_feat_372
    for s in SLICES:
        src = Path(a.v1_csv_dir) / f"ext_{s}.csv"
        keep_idx, rows, short = [], [], 0
        with open(src, newline="") as f:
            r = csv.reader(f)
            names = next(r)
            if len(names) != want:
                sys.exit(f"ABORT {src.name}: header {len(names)} cols, want {want}")
            for i, row in enumerate(r):
                if len(row) != want:
                    short += 1
                    continue
                keep_idx.append(i)
                rows.append(row)
        n = len(rows)
        arrays = [pa.array([x[0] for x in rows], type=pa.utf8())] + [
            pa.array([float(x[j]) for x in rows], type=pa.float64())
            for j in range(1, want)
        ]
        p372 = o372 / f"ext_{s}.parquet"
        pq.write_table(pa.table(arrays, names=names), p372,
                       compression="zstd", compression_level=7)

        pools = pq.read_table(Path(a.pools_root) / f"ext_{s}.parquet")
        sub = pools.take(pa.array(keep_idx))
        p944 = o944 / f"ext_{s}.parquet"
        pq.write_table(sub, p944, compression="zstd", compression_level=7)

        got = sub["ref_basename"].to_pylist()
        exp = [x[0] for x in rows]
        if got != exp:
            sys.exit(f"ABORT {s}: 944 subset ref_basename != 372 rows "
                     f"(first mismatch "
                     f"{next((k for k,(x,y) in enumerate(zip(got,exp)) if x!=y),'len')})")
        print(f"{s}: kept {n}, dropped {short} short-v1 rows "
              f"({100*short/(n+short):.1f}%); ref_basename identical across roots")
        man["slices"][s] = {
            "kept_rows": n, "dropped_short_v1_rows": short,
            "parquet_372": str(p372), "sha256_372": sha256_file(p372),
            "parquet_944": str(p944), "sha256_944": sha256_file(p944),
            "row_identity": "ref_basename equal row-for-row across the two roots (gated)",
        }
    (o372 / "_MANIFEST_samepair.json").write_text(json.dumps(man, indent=1))
    (o944 / "_MANIFEST_samepair.json").write_text(json.dumps(man, indent=1))
    # the 944 side must still declare its regime so the wrong-regime guard
    # recognises the block as live
    (o944 / "_MANIFEST.json").write_text(json.dumps(
        {"description": man["description"], "regime": "folded720append2pools",
         "regime_purity": "never column-mix with any other regime",
         "entries": []}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
