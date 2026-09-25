#!/usr/bin/env python3
"""Draw 200 bank pairs, then check a fresh extractor run against Part B sidecars."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from featbank_sets import BANK


ROOT = Path("/var/tmp/partb")
SEED = 20260924
N_SAMPLE = 200
import os
GMS = os.environ.get("PARTB_GMS") == "1"  # pass 2: raw CSV from /var/tmp/partb2, also compare f1322-f1501
NEW_IDS = range(944, 1502 if GMS else 1322)  # csfw/dvifm f944-985 (features__csfw_dvifm) + C1-C4 f986-1321 (features__rev4c1c4)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def draw():
    names = sorted(d.name for d in BANK.iterdir()
                   if d.is_dir() and not d.name.startswith("_"))
    counts = [pq.read_metadata(BANK / name / "keys.parquet").num_rows
              for name in names]
    total = sum(counts)
    if len(names) != 18 or total < N_SAMPLE:
        raise SystemExit(f"unexpected bank population: {len(names)} sets, {total} rows")
    selected = sorted(np.random.default_rng(SEED).choice(
        total, N_SAMPLE, replace=False).tolist())
    records = []
    offset = 0
    for name, count in zip(names, counts):
        local = [i - offset for i in selected if offset <= i < offset + count]
        offset += count
        if not local:
            continue
        table = pq.read_table(BANK / name / "keys.parquet",
                              columns=["pair_key", "ref_path", "dist_path",
                                       "ref_pixels_sha256", "dist_pixels_sha256"])
        for rid in local:
            records.append({"set": name, "bank_row": rid,
                            **{col: table.column(col)[rid].as_py()
                               for col in table.column_names}})
    if len(records) != N_SAMPLE:
        raise SystemExit("sample count changed")
    for i, record in enumerate(records):
        record["row_id"] = i
    pairs = ROOT / "pairs/reextract_200.tsv"
    with pairs.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["ref_path", "dist_path", "human_score", "row_id"])
        for r in records:
            writer.writerow([r["ref_path"], r["dist_path"], 0, r["row_id"]])
    mapping = ROOT / "pairs/reextract_200.json"
    mapping.write_text(json.dumps(records, indent=1) + "\n")
    print(f"PARTB_REEXTRACT_DRAW seed={SEED} rows={len(records)} "
          f"pairs_sha256={sha(pairs)} mapping_sha256={sha(mapping)}")


def verify():
    records = json.loads((ROOT / "pairs/reextract_200.json").read_text())
    if len(records) != N_SAMPLE:
        raise SystemExit("mapping count changed")
    want = {}
    for name in sorted({r["set"] for r in records}):
        selected = [r for r in records if r["set"] == name]
        take = pa.array([r["bank_row"] for r in selected], type=pa.int64())
        xt = pq.read_table(BANK / name / "features__csfw_dvifm.parquet",
                           columns=[f"f{i}" for i in range(944, 986)]).take(take)
        ct = pq.read_table(BANK / name / "features__rev4c1c4.parquet",
                           columns=[f"f{i}" for i in range(986, 1322)]).take(take)
        tabs = [xt, ct]
        if GMS:
            tabs.append(pq.read_table(BANK / name / "features__gmsbank.parquet",
                                      columns=[f"f{i}" for i in range(1322, 1502)]).take(take))
        for i, r in enumerate(selected):
            want[r["row_id"]] = np.array(
                [t.column(j)[i].as_py() for t in tabs for j in range(t.num_columns)],
                dtype=np.float32)
    raw = (Path("/var/tmp/partb2") if GMS else ROOT) / "raw"
    audit_path = raw / "reextract_200.audit.jsonl"
    with audit_path.open() as f:
        audits = [json.loads(line) for line in f]
    if len(audits) != N_SAMPLE:
        raise SystemExit("fresh audit count changed")
    for i, (a, r) in enumerate(zip(audits, records)):
        if (a["reference"] != r["ref_path"] or
            a["distorted"] != r["dist_path"] or
            a["reference_pixels_sha256"] != r["ref_pixels_sha256"] or
            a["distorted_pixels_sha256"] != r["dist_pixels_sha256"]):
            raise SystemExit(f"fresh pixel audit mismatch at {i}")
    csv_path = raw / "reextract_200.csv"
    seen = set()
    mismatches = 0
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rid_col = header.index("row_id")
        positions = [header.index(f"f{i}") for i in NEW_IDS]
        for row in reader:
            rid = int(row[rid_col])
            if rid in seen or rid not in want:
                raise SystemExit(f"fresh row id duplicated/outside sample: {rid}")
            seen.add(rid)
            got = np.fromiter((float(row[j]) for j in positions),
                              dtype=np.float64, count=len(positions)).astype(np.float32)
            mismatches += int(np.count_nonzero(
                got.view(np.uint32) != want[rid].view(np.uint32)))
    if len(seen) != N_SAMPLE or mismatches:
        raise SystemExit(f"fresh sidecar mismatch: rows={len(seen)} cells={mismatches}")
    print(f"PARTB_REEXTRACT_VERIFY rows={len(seen)} new_f32_cells={len(seen)*len(positions)} "
          f"bit_mismatches={mismatches} audit_sha256={sha(audit_path)} "
          f"csv_sha256={sha(csv_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["draw", "verify"])
    args = parser.parse_args()
    draw() if args.action == "draw" else verify()
