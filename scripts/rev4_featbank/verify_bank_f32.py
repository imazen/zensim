#!/usr/bin/env python3
"""Independent re-verification of the bank's f32 claim.

For every set, rebuilds each SOURCE stimulus's (pair_key, f64 feature row)
from the original inputs — source parquets + audits for converted sets,
pairs + audit.jsonl + feats.csv for extracted sets — then checks that the
bank sidecar row at that pair_key stores exactly np.float32(source f64).

Collapsed stimuli check the same shared key row (their features were
verified bit-identical at emit). Counts every stimulus x populated-id cell.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from featbank_sets import SETS, POPULATED_IDS, audit_row_id  # noqa: E402
from convert_cache import (  # noqa: E402
    BANK, SIDECAR_NAME, pair_key, load_audit_keyed, load_audit_positional,
    read_pairs, read_feats_csv,
)


def source_rows(name, cfg):
    """Return (pair_keys per stimulus, (n_stimuli, 944) f64 rows)."""
    if cfg["path"] == "parquet":
        tabs = [pq.read_table(p) for p in cfg["parquets"]]
        src = pa.concat_tables(tabs) if len(tabs) > 1 else tabs[0]
        if "select_row_ids" in cfg:
            keep = set(cfg["select_row_ids"])
            src = src.filter(
                pa.array([r in keep for r in src.column("row_id").to_pylist()])
            )
        src = src.take(pa.array(np.argsort(src.column("row_id").to_numpy())))
        audit = load_audit_keyed(cfg["audit"])
        pks = []
        for r in cfg["row_meta"](src):
            a = audit[r["row_id"]]
            pks.append(pair_key(
                a["reference_pixels_sha256"], a["distorted_pixels_sha256"]))
        feats = np.stack(
            [src.column(f"f{i}").to_numpy().astype(np.float64)
             for i in range(944)], axis=1)
        return pks, feats

    pairs = read_pairs(cfg["pairs"])
    audits = {audit_row_id(a): a for a in load_audit_positional(cfg["audit"])}
    feats_by = read_feats_csv(cfg["feats"])
    sel = cfg.get("select")
    pks, frs = [], []
    for i, pr in enumerate(pairs):
        rid = int(pr["row_id"])
        a = audits[rid]
        if sel is not None and not sel(i, pr, a):
            continue
        pks.append(pair_key(
            a["reference_pixels_sha256"], a["distorted_pixels_sha256"]))
        frs.append(feats_by[rid])
    return pks, np.asarray(frs, dtype=np.float64)


def main():
    grand_stim = grand_cells = grand_bad = 0
    for name, cfg in SETS.items():
        pks, feats = source_rows(name, cfg)
        keys = pq.read_table(BANK / name / "keys.parquet") \
            .column("pair_key").to_pylist()
        kidx = {k: j for j, k in enumerate(keys)}
        order = np.array([kidx[p] for p in pks])
        side = pq.read_table(BANK / name / SIDECAR_NAME)
        bad = 0
        for fid in POPULATED_IDS:
            got = side.column(f"f{fid}").to_numpy()[order]
            want = feats[:, fid].astype(np.float32)
            bad += int((got.view(np.uint32) != want.view(np.uint32)).sum())
        cells = len(order) * len(POPULATED_IDS)
        print(f"{name}: stimuli={len(order)} cells={cells} bad={bad}")
        grand_stim += len(order)
        grand_cells += cells
        grand_bad += bad
    print(f"TOTAL stimuli={grand_stim} cells={grand_cells} bad={grand_bad}")


if __name__ == "__main__":
    main()
