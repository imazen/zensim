#!/usr/bin/env python3
"""Build the 21 bigcodec 944-feature split views FROM the canonical 924 views +
the bf944 fleet table — emitting rows in the 924 VIEW'S ORDER with every
non-feature column byte-carried from the 924 view, and f0..f943 taken from the
tier-matched fleet re-extraction (tbig_944_full.parquet, keyed encode_sha ==
encoded_filename).

Why this shape (vs re-running the tbig_join_924.py canonical-root join): the
G-BF1/G-BF2 gate (`gate_backfill944.py`) compares NEW-vs-OLD parquet pairs
POSITIONALLY, row-for-row. The fleet table's row order is fetch-order (never
the view order), so the 944 views must be emitted in the 924 views' order.
Deriving the ID/target columns verbatim from the frozen 924 views also makes
G-BF2 an exact byte-carry by construction — and G-BF1 stays a TRUE gate: the
fleet's independently re-extracted f0..f923 must equal the 924 view's stored
features bitwise (tier-matched extraction; zenmetrics
scripts/jobsys/declare_bf944_tiered.py).

Coverage is hard-gated: every 924-view row MUST be matched by exactly one
fleet row (match_rate 1.0000); unmatched rows are counted and FAIL the build —
never synthesized. Duplicate fleet keys keep-first (same rule as the 924
assembler), counted.

Memory: one dataset (3 splits) at a time — per-dataset peak ≈ rows x 944 x 8B.

Usage:
  python3 tbig_join_944.py --fleet /mnt/v/output/zensim/tbig-944-2026-08-02/tbig_944_full.parquet \
      --views-root /mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/bigcodec \
      --out /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec
"""

import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "validate", "test")
N_FEAT = 944
KEY = "encoded_filename"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", required=True)
    ap.add_argument("--views-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", help="comma list; default = all subdirs with train_924.parquet")
    args = ap.parse_args()

    root = args.views_root
    datasets = (
        args.datasets.split(",")
        if args.datasets
        else sorted(
            d for d in os.listdir(root)
            if os.path.isfile(os.path.join(root, d, "train_924.parquet"))
        )
    )
    print(f"{len(datasets)} datasets: {datasets}", file=sys.stderr)

    feat_names = [f"f{i}" for i in range(N_FEAT)]
    fleet_cols = ["encode_sha"] + feat_names
    # --fleet may be a single parquet, a directory of part files, or a base
    # path whose `<stem>.part-NNN.parquet` siblings exist (the assembler's
    # crash-safe part-rolling output).
    fp = args.fleet
    if os.path.isdir(fp):
        fleet_files = sorted(
            os.path.join(fp, f) for f in os.listdir(fp)
            if f.endswith(".parquet") and ".bak" not in f
        )
    elif os.path.isfile(fp):
        fleet_files = [fp]
    else:
        stem = os.path.splitext(os.path.basename(fp))[0]
        d = os.path.dirname(fp)
        fleet_files = sorted(
            os.path.join(d, f) for f in os.listdir(d)
            if f.startswith(stem + ".part-") and f.endswith(".parquet")
        )
    if not fleet_files:
        print(f"FATAL: no fleet parquet(s) at {fp}", file=sys.stderr)
        return 1
    print(f"fleet: {len(fleet_files)} file(s)", file=sys.stderr)
    report = {"per_split": {}, "fleet": fleet_files, "views_root": root}
    any_fail = False

    for dset in datasets:
        # 1) load the 924 views' key columns (order-defining) per split
        split_keys = {}
        for split in SPLITS:
            p = os.path.join(root, dset, f"{split}_924.parquet")
            t = pq.read_table(p, columns=[KEY])
            split_keys[split] = t.column(KEY).to_pylist()
        # key -> (split, row_idx); duplicate view keys are a structural error
        lookup = {}
        dup_view_keys = 0
        for split, keys in split_keys.items():
            for i, k in enumerate(keys):
                if k in lookup:
                    dup_view_keys += 1
                    continue
                lookup[k] = (split, i)
        n_rows = {s: len(k) for s, k in split_keys.items()}
        print(f"{dset}: rows {n_rows} dup_view_keys={dup_view_keys}",
              file=sys.stderr, flush=True)

        # 2) one streaming pass over the fleet parquet filling per-split arrays
        feats = {s: np.full((n_rows[s], N_FEAT), np.nan, np.float64) for s in SPLITS}
        filled = {s: np.zeros(n_rows[s], bool) for s in SPLITS}
        dup_fleet = 0
        for fpath in fleet_files:
            pf = pq.ParquetFile(fpath)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["encode_sha"])
                keys = t.column("encode_sha").to_pylist()
                hits = [(i, lookup[k]) for i, k in enumerate(keys) if k in lookup]
                if not hits:
                    continue
                t = pf.read_row_group(rg, columns=fleet_cols)
                keys = t.column("encode_sha").to_pylist()
                cols = [t.column(f).to_numpy(zero_copy_only=False) for f in feat_names]
                for i, (split, ridx) in hits:
                    if filled[split][ridx]:
                        dup_fleet += 1
                        continue
                    for j in range(N_FEAT):
                        feats[split][ridx, j] = cols[j][i]
                    filled[split][ridx] = True
                del t, cols
        # 3) coverage gate + write, per split, in view order with carried columns
        os.makedirs(os.path.join(args.out, dset), exist_ok=True)
        for split in SPLITS:
            missing = int((~filled[split]).sum())
            mr = 1.0 - missing / max(n_rows[split], 1)
            report["per_split"][f"{dset}/{split}"] = {
                "rows": n_rows[split],
                "matched": int(filled[split].sum()),
                "match_rate": mr,
                "dup_fleet_keys": dup_fleet,
            }
            if missing:
                any_fail = True
                print(f"FAIL {dset}/{split}: {missing} view rows unmatched by fleet",
                      file=sys.stderr)
                continue
            old_path = os.path.join(root, dset, f"{split}_924.parquet")
            old = pq.read_table(old_path)
            nonfeat = [n for n in old.column_names
                       if not (n.startswith("f") and n[1:].isdigit())]
            arrays = [old.column(n) for n in nonfeat]
            names = list(nonfeat)
            for j, fn in enumerate(feat_names):
                arrays.append(pa.array(feats[split][:, j], type=pa.float64()))
                names.append(fn)
            out_p = os.path.join(args.out, dset, f"{split}_944.parquet")
            pq.write_table(pa.table(dict(zip(names, arrays))), out_p,
                           compression="zstd")
            print(f"  wrote {out_p} ({n_rows[split]} rows)", file=sys.stderr, flush=True)
            del old, arrays
        del feats, filled

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "_JOIN_REPORT.json"), "w") as f:
        json.dump(report, f, indent=2)
    worst = min(report["per_split"].values(), key=lambda r: r["match_rate"])
    print(f"DONE worst match_rate={worst['match_rate']:.4f} "
          f"({'FAIL' if any_fail else 'PASS'})", file=sys.stderr)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
