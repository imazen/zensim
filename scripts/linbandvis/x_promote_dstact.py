#!/usr/bin/env python3
"""x_promote_dstact.py — APPENDIX X (Thread 2): promote the BANDVIS-ON
extraction CSVs to parquet AND run GATE X-G1 (lanes-only at corpus scale).

This is EXPERIMENT data, not canonical: the ON rows are a DIFFERENT feature
definition for the four BANDVIS_GAIN slots {f924, f929, f934, f939}
(`V2NewFeatureToggles::append2_dst_activity`, the P1.5 shipped GAIN-only
combine). REGIME PURITY: never column-mix these rows into any canonical
table; this root exists only for appendix X's paired arms.

Per leg:
  1. CSV -> parquet with the canonical schema rules (ref_basename utf8,
     human_score f64, f0..f943 f64, ZSTD-7, NaN/null hard abort).
     ext_kadid additionally gets `human_score := 1 - human_score` — the
     exact wave-10 orientation fix (fix_ext_kadid_orientation.py), because
     its pairs TSV predates the fix.
  2. X-G1 vs the stored canonical parquet (row-for-row):
       - every column EXCEPT the 4 GAIN slots: bitwise identical (f64 view)
       - each GAIN slot: differs on > 0 rows (the toggle is live)
     Any other column moving => STOP (exit 1) — either the toggle leaks or
     the environment drifts; both invalidate the paired arms.

Usage:
  x_promote_dstact.py --src <csv dir> --dest <parquet dir> \
      --canon /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01 \
      --commit <full extractor sha> [--legs a,b,c]
"""

import argparse
import csv as csvmod
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GAIN_SLOTS = (924, 929, 934, 939)
N_FEAT = 944

LEGS = [
    "ext_sdr25",
    "ext_aic4",
    "ext_konjnd_jpeg_val",
    "ext_aic3",
    "ext_live",
    "ext_csiq",
    "ext_tid",
    "ext_cid22val",
    "ext_kadid",
    "ext_cid22_train201",
    "konjnd_bpg_train_944",
    "konjnd_bpg_val_944",
    "ext_safesyn_full",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_to_table(csv_path: Path, fix_kadid: bool) -> pa.Table:
    with open(csv_path, newline="") as f:
        rdr = csvmod.reader(f)
        header = next(rdr)
        assert header[0] == "ref_basename" and header[1] == "human_score", header[:2]
        feat_cols = header[2:]
        assert len(feat_cols) == N_FEAT, f"{csv_path}: {len(feat_cols)} features"
        names, scores, feats = [], [], []
        for row in rdr:
            names.append(row[0])
            scores.append(float(row[1]))
            feats.append([float(x) for x in row[2:]])
    y = np.asarray(scores, dtype=np.float64)
    if fix_kadid:
        y = 1.0 - y  # the exact wave-10 orientation fix
    fmat = np.asarray(feats, dtype=np.float64)
    if not np.isfinite(fmat).all() or not np.isfinite(y).all():
        sys.exit(f"ABORT {csv_path}: non-finite values")
    cols = [pa.array(names, type=pa.string()), pa.array(y, type=pa.float64())]
    fields = [("ref_basename", pa.string()), ("human_score", pa.float64())]
    for i, c in enumerate(feat_cols):
        assert c == f"f{i}", c
        cols.append(pa.array(fmat[:, i], type=pa.float64()))
        fields.append((f"f{i}", pa.float64()))
    return pa.Table.from_arrays(cols, schema=pa.schema(fields))


def gate_leg(new: pa.Table, canon_path: Path, leg: str) -> dict:
    old = pq.read_table(canon_path)
    if new.num_rows != old.num_rows:
        sys.exit(f"X-G1 FAIL {leg}: rows {new.num_rows} vs {old.num_rows}")
    gain_names = {f"f{i}" for i in GAIN_SLOTS}
    n_gain_diff = {}
    for name in old.column_names:
        if name not in new.column_names:
            sys.exit(f"X-G1 FAIL {leg}: canonical column {name} missing in new")
        a = old.column(name).to_pandas().to_numpy()
        b = new.column(name).to_pandas().to_numpy()
        if a.dtype.kind == "f":
            a64 = a.astype(np.float64, copy=False).view(np.uint64)
            b64 = b.astype(np.float64, copy=False).view(np.uint64)
            same = bool((a64 == b64).all())
            ndiff = int((a64 != b64).sum())
        else:
            same = bool((a == b).all())
            ndiff = int((a != b).sum())
        if name in gain_names:
            n_gain_diff[name] = ndiff
        elif not same:
            sys.exit(
                f"X-G1 FAIL {leg}: non-GAIN column {name} differs on {ndiff} rows "
                f"(toggle leak or environment drift) — STOP"
            )
    zero_diff = [k for k, v in n_gain_diff.items() if v == 0]
    return {"rows": new.num_rows, "gain_rows_changed": n_gain_diff, "gain_all_zero": zero_diff}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--canon", required=True, type=Path)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--legs", default=",".join(LEGS))
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    report = {
        "description": (
            "APPENDIX X BANDVIS-ON (dst-activity) EXPERIMENT extraction — "
            "NOT CANONICAL. GAIN slots f924/f929/f934/f939 use the P1.5 "
            "shipped GAIN-only combine (ZENSIM_APPEND2_DSTACT=1); every other "
            "column gated bitwise-identical to ext944-canonical-2026-08-01. "
            "NEVER column-mix with canonical tables."
        ),
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "build_commit": args.commit,
        "driver": "scripts/linbandvis/x_extract_dstact.sh (v2_ab_extract, foldapp2, ZENSIM_APPEND2_DSTACT=1)",
        "gate": "X-G1 lanes-only (this script)",
        "legs": {},
    }
    for leg in args.legs.split(","):
        csv_path = args.src / f"{leg}.csv"
        canon_path = args.canon / f"{leg}.parquet"
        out_path = args.dest / f"{leg}.parquet"
        print(f"== {leg}: promote + X-G1", flush=True)
        tbl = csv_to_table(csv_path, fix_kadid=(leg == "ext_kadid"))
        g = gate_leg(tbl, canon_path, leg)
        pq.write_table(tbl, out_path, compression="zstd", compression_level=7)
        g["sha256"] = sha256(out_path)
        g["parquet"] = str(out_path)
        report["legs"][leg] = g
        print(f"   rows={g['rows']} gain_rows_changed={g['gain_rows_changed']}", flush=True)

    # The toggle must be LIVE somewhere: at least one leg must move every GAIN
    # slot on >0 rows (identity-heavy legs may legitimately have sparse
    # movement; flat-zero across ALL legs would mean the env did not reach the
    # extractor).
    total = {f"f{i}": 0 for i in GAIN_SLOTS}
    for g in report["legs"].values():
        for k, v in g["gain_rows_changed"].items():
            total[k] += v
    report["gain_rows_changed_total"] = total
    if any(v == 0 for v in total.values()):
        sys.exit(f"X-G1 FAIL: a GAIN slot never changed across all legs: {total}")
    (args.dest / "_MANIFEST.json").write_text(json.dumps(report, indent=1))
    print(f"X-G1 PASS — manifest written to {args.dest}/_MANIFEST.json")


if __name__ == "__main__":
    main()
