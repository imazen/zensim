#!/usr/bin/env python3
"""Join the fleet's merged 720-feature sidecar (fleet_blob_fetch_720.py
fetch-all output, keyed `encode_sha` == canonical `encoded_filename`) to the
canonical-picker-2026-06-27 per-codec train/validate/test datasets, emitting
720-feature training views with ssim2/zensim targets.

EXACT key join (encoded_filename) — no fingerprint matching, no fabrication:
canonical rows without a fleet feature row are counted and listed, never
synthesized. One streaming pass over the fleet parquet (row-group at a
time); the canonical side is held as a key -> (dataset, split, targets)
map (~5.7M entries).

Outputs, per codec-dataset and split:
  <out>/<dataset>/<split>_720.parquet
    origin_id, ref_filename, encoded_filename, codec, q, knob_tuple_json,
    score_ssim2, score_zensim, f0..f719
Plus <out>/_JOIN_REPORT.json with per-(dataset, split) match rates and the
unmatched-key counts (E1 gate: skip rate <= 0.1%).

Usage:
  python3 tbig_join_720.py --fleet ~/tmp/tbig_720_full.parquet \
      --canonical-root /mnt/v/output/canonical-picker-2026-06-27 \
      --out /mnt/v/zen/zensim-training/ext720-canonical-2026-07-22/bigcodec
"""

import argparse
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "validate", "test")
ID_COLS = [
    "origin_id",
    "ref_filename",
    "encoded_filename",
    "codec",
    "q",
    "knob_tuple_json",
    "score_ssim2",
    "score_zensim",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", required=True)
    ap.add_argument("--canonical-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", help="comma list; default = all subdirs with train.parquet")
    args = ap.parse_args()

    root = args.canonical_root
    datasets = (
        args.datasets.split(",")
        if args.datasets
        else sorted(
            d
            for d in os.listdir(root)
            if os.path.isfile(os.path.join(root, d, "train.parquet"))
        )
    )
    print(f"{len(datasets)} canonical datasets: {datasets}", file=sys.stderr)

    # key -> (dataset_idx, split_idx, id-col tuple)
    lookup = {}
    canon_counts = {}
    dup_keys = 0
    for di, dset in enumerate(datasets):
        for si, split in enumerate(SPLITS):
            p = os.path.join(root, dset, f"{split}.parquet")
            t = pq.read_table(p, columns=ID_COLS)
            canon_counts[(dset, split)] = t.num_rows
            cols = [t.column(c).to_pylist() for c in ID_COLS]
            for row in zip(*cols):
                key = row[2]  # encoded_filename
                if key in lookup:
                    dup_keys += 1
                    continue
                lookup[key] = (di, si, row)
            print(
                f"  loaded {dset}/{split}: {t.num_rows} rows (lookup now {len(lookup)})",
                file=sys.stderr,
                flush=True,
            )
    print(f"lookup: {len(lookup)} keys, {dup_keys} cross-dataset dup keys skipped", file=sys.stderr)

    feat_names = [f"f{i}" for i in range(720)]
    out_schema = pa.schema(
        [
            ("origin_id", pa.string()),
            ("ref_filename", pa.string()),
            ("encoded_filename", pa.string()),
            ("codec", pa.string()),
            ("q", pa.string()),
            ("knob_tuple_json", pa.string()),
            ("score_ssim2", pa.float64()),
            ("score_zensim", pa.float64()),
        ]
        + [(f, pa.float64()) for f in feat_names]
    )

    writers = {}
    matched = {}
    fleet_rows = 0
    fleet_unmatched = 0
    pf = pq.ParquetFile(args.fleet)
    fleet_cols = ["encode_sha"] + feat_names
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=fleet_cols)
        keys = t.column("encode_sha").to_pylist()
        feats = [t.column(f) for f in feat_names]
        # bucket row indices per (dataset, split)
        buckets = {}
        for i, k in enumerate(keys):
            hit = lookup.get(k)
            if hit is None:
                fleet_unmatched += 1
                continue
            buckets.setdefault((hit[0], hit[1]), []).append((i, hit[2]))
        fleet_rows += len(keys)
        for (di, si), rows in buckets.items():
            dset, split = datasets[di], SPLITS[si]
            idx = [r[0] for r in rows]
            ids = [r[1] for r in rows]
            arrs = [pa.array([x[j] for x in ids], type=out_schema.field(j).type) for j in range(6)]
            arrs.append(pa.array([_f(x[6]) for x in ids], type=pa.float64()))
            arrs.append(pa.array([_f(x[7]) for x in ids], type=pa.float64()))
            idx_arr = pa.array(idx, type=pa.int32())
            for f in feats:
                arrs.append(f.take(idx_arr))
            tbl = pa.table(dict(zip(out_schema.names, arrs)), schema=out_schema)
            wkey = (dset, split)
            if wkey not in writers:
                os.makedirs(os.path.join(args.out, dset), exist_ok=True)
                writers[wkey] = pq.ParquetWriter(
                    os.path.join(args.out, dset, f"{split}_720.parquet"),
                    out_schema,
                    compression="zstd",
                )
            writers[wkey].write_table(tbl)
            matched[wkey] = matched.get(wkey, 0) + len(rows)
        done_total = sum(matched.values())
        print(
            f"  rg {rg + 1}/{pf.num_row_groups}: fleet={fleet_rows} matched={done_total} "
            f"unmatched={fleet_unmatched}",
            file=sys.stderr,
            flush=True,
        )
    for w in writers.values():
        w.close()

    report = {
        "fleet_rows": fleet_rows,
        "fleet_rows_unmatched_by_canonical": fleet_unmatched,
        "lookup_keys": len(lookup),
        "per_split": {
            f"{d}/{s}": {
                "canonical_rows": canon_counts[(d, s)],
                "matched": matched.get((d, s), 0),
                "match_rate": matched.get((d, s), 0) / max(canon_counts[(d, s)], 1),
            }
            for d in datasets
            for s in SPLITS
        },
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "_JOIN_REPORT.json"), "w") as f:
        json.dump(report, f, indent=2)
    worst = min(report["per_split"].values(), key=lambda r: r["match_rate"])
    print(
        f"DONE: fleet {fleet_rows} rows, {sum(matched.values())} matched into "
        f"{len(writers)} split files; worst split match_rate={worst['match_rate']:.4f}",
        file=sys.stderr,
    )
    return 0


def _f(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(main())
