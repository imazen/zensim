#!/usr/bin/env python3
"""Restored-cut sidecars over the 18 promoted Rev4 bank sets (Part B conventions).

Subcommands
  pairs            write label-free pair TSVs from the promoted bank keys (human_score = 0
                   placeholder, row_id = dense bank row) under $ROOT/pairs/
  bind <set>       bind one set's extractor CSV + audit JSONL to the bank keys and write one
                   f32 sidecar per family plus a manifest under $ROOT/bank/<set>/
  verify           read-only audit of every written sidecar against keys and raw CSVs
  draw / reextract 200-pair fresh-extraction sample (seed fixed) and its bit-exact check

Conventions (copied from scripts/rev4_featbank/convert_cache.py, Part B): pair_key =
sha256(ref_px_hex || dist_px_hex || 'legacy-rgb8'); f32 = round-nearest-even cast of the
f64 the extractor printed; Parquet zstd-3, BYTE_STREAM_SPLIT, no dictionary, row groups of
65,536, key order = bank key order; every audit record is bound to its key by row_id and
both pixel hashes. No label is read: the pair TSVs carry a constant human_score = 0.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BANK = Path("/var/tmp/rev4-featbank/bank")
ROOT = Path(os.environ.get("RESTORE_ROOT", "/var/tmp/restore-cuts"))
INPUT_CONTRACT = "legacy-rgb8"
ROW_GROUP = 65536
SEED = 20260924
N_SAMPLE = 200

# family -> (name, first id, width). Order = registry order (append-only after f1501).
FAMILIES = {
    "mapdev": ("features__restore_mapdev.parquet", 1502, 60),
    "z1max": ("features__restore_z1max.parquet", 1562, 228),
    "gmsnative": ("features__restore_gmsnative.parquet", 1790, 30),
    "dvifmgate": ("features__restore_dvifmgate.parquet", 1820, 5),
}
FULL_WIDTH = 1825
# z1max cell-local slots whose registry form is Difference (exactly 0 on an identity pair).
Z1_DIFF_LOCALS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18]


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pair_key(ref_hex, dist_hex):
    return hashlib.sha256((ref_hex + dist_hex + INPUT_CONTRACT).encode()).hexdigest()


def sets():
    names = sorted(d.name for d in BANK.iterdir() if d.is_dir() and not d.name.startswith("_"))
    if len(names) != 18:
        raise SystemExit(f"expected 18 bank sets, found {len(names)}")
    return names


def keys_of(name, limit=None):
    t = pq.read_table(BANK / name / "keys.parquet",
                      columns=["pair_key", "row_id", "ref_path", "dist_path",
                               "ref_pixels_sha256", "dist_pixels_sha256", "pixels_identical"])
    if limit:
        t = t.slice(0, limit)
    d = t.to_pydict()
    n = t.num_rows
    if d["row_id"] != list(range(n)):
        raise SystemExit(f"{name}: bank row_id is not dense/in order")
    if len(set(d["pair_key"])) != n:
        raise SystemExit(f"{name}: duplicate bank pair_key")
    for i in range(n):
        if pair_key(d["ref_pixels_sha256"][i], d["dist_pixels_sha256"][i]) != d["pair_key"][i]:
            raise SystemExit(f"{name}: stored pair_key disagrees at {i}")
    return d


def write_pairs(name, path, limit=None):
    k = keys_of(name, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["ref_path", "dist_path", "human_score", "row_id"])
        for i, (r, d) in enumerate(zip(k["ref_path"], k["dist_path"])):
            w.writerow([r, d, 0, i])
    print(f"RESTORE_PAIRS set={name} rows={len(k['pair_key'])} sha256={sha256_file(path)} path={path}")


def write_sidecar(path, keys, ids, arr):
    n = arr.shape[0]
    tmp = path.with_suffix(".tmp")
    schema = pa.schema([pa.field("pair_key", pa.string())] +
                       [pa.field(f"f{i}", pa.float32()) for i in ids])
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=3,
                          use_dictionary=False,
                          column_encoding={f"f{i}": "BYTE_STREAM_SPLIT" for i in ids}) as w:
        for lo in range(0, n, ROW_GROUP):
            hi = min(lo + ROW_GROUP, n)
            cols = [pa.array(keys["pair_key"][lo:hi], type=pa.string())]
            cols.extend(pa.array(arr[lo:hi, j], type=pa.float32()) for j in range(len(ids)))
            w.write_table(pa.Table.from_arrays(cols, schema=schema), row_group_size=ROW_GROUP)
    with pq.ParquetFile(tmp) as pf:
        if pf.metadata.num_rows != n or pf.schema_arrow != schema:
            raise SystemExit(f"{path}: Parquet row/schema roundtrip failed")
        lo = 0
        for batch in pf.iter_batches(batch_size=ROW_GROUP):
            hi = lo + batch.num_rows
            if batch.column(0).to_pylist() != keys["pair_key"][lo:hi]:
                raise SystemExit(f"{path}: key order changed at {lo}")
            for j in range(len(ids)):
                got = batch.column(j + 1).to_numpy()
                if not np.array_equal(got.view(np.uint32), arr[lo:hi, j].view(np.uint32)):
                    raise SystemExit(f"{path}: value changed at {lo}, f{ids[j]}")
            lo = hi
        if lo != n:
            raise SystemExit(f"{path}: coverage {lo}/{n}")
    os.replace(tmp, path)


def identity_violations(family, ident, arr):
    """Count nonzero cells the registry says must be exactly 0 (or equal) on identity rows."""
    if not ident.any():
        return 0
    a = arr[ident]
    if family == "mapdev":
        cells = a.reshape(len(a), 12, 5)
        bad = np.count_nonzero(cells[:, :, 0])
        bad += np.count_nonzero(cells[:, :, 1].view(np.uint32) != cells[:, :, 2].view(np.uint32))
        bad += np.count_nonzero(cells[:, :, 3].view(np.uint32) != cells[:, :, 4].view(np.uint32))
        return int(bad)
    if family == "z1max":
        cells = a.reshape(len(a), 12, 19)
        return int(np.count_nonzero(cells[:, :, Z1_DIFF_LOCALS]))
    return int(np.count_nonzero(a))


def bind(name, build_meta_path, binary):
    if not (ROOT / "raw").exists():
        raise SystemExit("run the extraction first")
    keys = keys_of(name)
    n = len(keys["pair_key"])
    csv_path = ROOT / "raw" / f"{name}.csv"
    audit_path = ROOT / "raw" / f"{name}.audit.jsonl"
    pairs_path = ROOT / "pairs" / f"{name}.tsv"
    producer = json.loads(Path(str(csv_path) + ".manifest.json").read_text())
    bin_sha = sha256_file(binary)
    if producer.get("producer_binary_sha256") != bin_sha:
        raise SystemExit("extractor manifest binary hash differs from the build")
    if producer.get("formula_revision") != "3" or producer.get("layout") != f"w{FULL_WIDTH}":
        raise SystemExit(f"extractor manifest has wrong formula/layout: {producer}")
    if producer.get("input_contract", INPUT_CONTRACT) != INPUT_CONTRACT:
        raise SystemExit("extractor manifest has wrong input contract")
    build_meta = json.loads(Path(build_meta_path).read_text())

    with audit_path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                raise SystemExit(f"{name}: extra audit row")
            a = json.loads(line)
            if (a["reference"] != keys["ref_path"][i] or a["distorted"] != keys["dist_path"][i]
                    or a["reference_pixels_sha256"] != keys["ref_pixels_sha256"][i]
                    or a["distorted_pixels_sha256"] != keys["dist_pixels_sha256"][i]
                    or pair_key(a["reference_pixels_sha256"], a["distorted_pixels_sha256"])
                    != keys["pair_key"][i]):
                raise SystemExit(f"{name}: audit/key binding failed at {i}")
        if i + 1 != n:
            raise SystemExit(f"{name}: audit has {i + 1} of {n} rows")

    total = sum(w for _, _, w in FAMILIES.values())
    first = min(f for _, f, _ in FAMILIES.values())
    arr = np.memmap(ROOT / "raw" / f"{name}.new.f32", mode="w+", dtype=np.float32, shape=(n, total))
    seen = np.zeros(n, dtype=np.bool_)
    with csv_path.open(newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        rid_i = hdr.index("row_id")
        pos = [hdr.index(f"f{i}") for i in range(first, FULL_WIDTH)]
        if pos != list(range(pos[0], pos[0] + total)):
            raise SystemExit("new feature columns are not contiguous")
        if hdr.index(f"f{FULL_WIDTH - 1}") != pos[-1] or f"f{FULL_WIDTH}" in hdr:
            raise SystemExit("extractor CSV width is not the registered full width")
        for row in r:
            rid = int(row[rid_i])
            if rid < 0 or rid >= n or seen[rid]:
                raise SystemExit(f"{name}: invalid/duplicate CSV row_id {rid}")
            seen[rid] = True
            v = np.fromiter((float(x) for x in row[pos[0]:pos[0] + total]), dtype=np.float64,
                            count=total)
            if not np.isfinite(v).all():
                raise SystemExit(f"{name}: non-finite value at {rid}")
            arr[rid] = v.astype(np.float32)
    if not seen.all():
        raise SystemExit(f"{name}: coverage {int(seen.sum())}/{n}")
    arr.flush()
    ident = np.asarray(keys["pixels_identical"], dtype=np.bool_)
    manifest = {"set": name, "row_count": n, "families": {}}
    off = 0
    for fam, (fname, fid, width) in FAMILIES.items():
        block = np.asarray(arr[:, off:off + width])
        off += width
        ids = list(range(fid, fid + width))
        bad = identity_violations(fam, ident, block)
        live = [int(np.count_nonzero(block[:, j])) for j in range(width)]
        print(f"RESTORE_CHECK set={name} family={fam} rows={n} finite=1 unique=1 "
              f"identity_violations={bad} dead_columns={[ids[j] for j, v in enumerate(live) if v == 0]}")
        if bad:
            raise SystemExit(f"{name}/{fam}: {bad} identity violations")
        out_dir = ROOT / "bank" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / fname
        write_sidecar(path, keys, ids, block)
        manifest["families"][fam] = {
            "file": fname, "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "populated_feature_ids": ids, "live_cells_by_feature": live,
            "identity_violations": bad, "dtype": "f32", "cast": "f64->f32 round-nearest-even",
        }
    manifest.update({
        "feature_set_id": producer["feature_set_id"], "formula_revision": "3", "root_form": "sqrt",
        "input_contract": INPUT_CONTRACT, "build_commit": build_meta["repositories"]["zensim"]["commit"],
        "binary_sha256": bin_sha, "build_meta_sha256": sha256_file(build_meta_path),
        "env": {"ZENSIM_FORMULA_REV": "3", "ZENSIM_ROOT_FORM": "sqrt", "RAYON_NUM_THREADS": "8"},
        "pairs_sha256": sha256_file(pairs_path), "audit_sha256": sha256_file(audit_path),
        "extractor_csv_sha256": sha256_file(csv_path),
        "bank_keys_sha256": sha256_file(BANK / name / "keys.parquet"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    mpath = ROOT / "bank" / name / "_MANIFEST_restore.json"
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"RESTORE_WRITTEN set={name} rows={n} manifest_sha256={sha256_file(mpath)} "
          f"feature_set_id={producer['feature_set_id']}")


def verify():
    result = {}
    for name in sets():
        d = ROOT / "bank" / name
        m = json.loads((d / "_MANIFEST_restore.json").read_text())
        keys = pq.read_table(BANK / name / "keys.parquet", columns=["pair_key", "pixels_identical"])
        pks = keys.column("pair_key").to_pylist()
        ident = keys.column("pixels_identical").to_numpy()
        n = len(pks)
        assert m["row_count"] == n and m["bank_keys_sha256"] == sha256_file(BANK / name / "keys.parquet")
        per = {}
        for fam, (fname, fid, width) in FAMILIES.items():
            path = d / fname
            assert sha256_file(path) == m["families"][fam]["sha256"], f"{name}/{fam} sha"
            pf = pq.ParquetFile(path)
            cols = ["pair_key"] + [f"f{i}" for i in range(fid, fid + width)]
            assert pf.schema_arrow.names == cols
            assert all(pf.schema_arrow.field(c).type == pa.float32() for c in cols[1:])
            assert pf.metadata.num_rows == n
            lo = 0
            bad = 0
            live = np.zeros(width, dtype=np.int64)
            for batch in pf.iter_batches(batch_size=ROW_GROUP):
                hi = lo + batch.num_rows
                assert batch.column(0).to_pylist() == pks[lo:hi], f"{name}/{fam} order {lo}"
                v = np.stack([batch.column(j + 1).to_numpy() for j in range(width)], axis=1)
                assert np.isfinite(v).all(), f"{name}/{fam} nonfinite {lo}"
                live += np.count_nonzero(v, axis=0)
                bad += identity_violations(fam, ident[lo:hi], v)
                lo = hi
            assert lo == n and bad == 0
            per[fam] = {"sha256": m["families"][fam]["sha256"], "bytes": m["families"][fam]["bytes"],
                        "dead_columns": [fid + j for j, c in enumerate(live) if c == 0]}
        result[name] = {"rows": n, "families": per, "binary_sha256": m["binary_sha256"]}
    assert len({v["binary_sha256"] for v in result.values()}) == 1
    out = ROOT / "verification.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"RESTORE_VERIFY sets={len(result)} rows={sum(v['rows'] for v in result.values())} "
          f"sha256={sha256_file(out)}")
    for name, v in result.items():
        print(name, v["rows"], {f: x["dead_columns"] for f, x in v["families"].items()})


def draw():
    names = sets()
    counts = [pq.read_metadata(BANK / nme / "keys.parquet").num_rows for nme in names]
    total = sum(counts)
    selected = sorted(np.random.default_rng(SEED).choice(total, N_SAMPLE, replace=False).tolist())
    records, offset = [], 0
    for nme, count in zip(names, counts):
        local = [i - offset for i in selected if offset <= i < offset + count]
        offset += count
        if not local:
            continue
        t = pq.read_table(BANK / nme / "keys.parquet",
                          columns=["pair_key", "ref_path", "dist_path", "ref_pixels_sha256",
                                   "dist_pixels_sha256"])
        for rid in local:
            records.append({"set": nme, "bank_row": rid,
                            **{c: t.column(c)[rid].as_py() for c in t.column_names}})
    for i, r in enumerate(records):
        r["row_id"] = i
    pairs = ROOT / "pairs/reextract_200.tsv"
    with pairs.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["ref_path", "dist_path", "human_score", "row_id"])
        for r in records:
            w.writerow([r["ref_path"], r["dist_path"], 0, r["row_id"]])
    mapping = ROOT / "pairs/reextract_200.json"
    mapping.write_text(json.dumps(records, indent=1) + "\n")
    print(f"RESTORE_REEXTRACT_DRAW seed={SEED} rows={len(records)} "
          f"pairs_sha256={sha256_file(pairs)} mapping_sha256={sha256_file(mapping)}")


def reextract():
    records = json.loads((ROOT / "pairs/reextract_200.json").read_text())
    want = {}
    for name in sorted({r["set"] for r in records}):
        sel = [r for r in records if r["set"] == name]
        take = pa.array([r["bank_row"] for r in sel], type=pa.int64())
        tabs = [pq.read_table(ROOT / "bank" / name / fname,
                              columns=[f"f{i}" for i in range(fid, fid + w)]).take(take)
                for fname, fid, w in FAMILIES.values()]
        for i, r in enumerate(sel):
            want[r["row_id"]] = np.array([t.column(j)[i].as_py() for t in tabs
                                          for j in range(t.num_columns)], dtype=np.float32)
    audits = [json.loads(line) for line in (ROOT / "raw/reextract_200.audit.jsonl").open()]
    for i, (a, r) in enumerate(zip(audits, records)):
        if (a["reference_pixels_sha256"] != r["ref_pixels_sha256"]
                or a["distorted_pixels_sha256"] != r["dist_pixels_sha256"]):
            raise SystemExit(f"fresh pixel audit mismatch at {i}")
    seen, mism = set(), 0
    with (ROOT / "raw/reextract_200.csv").open(newline="") as f:
        reader = csv.reader(f)
        hdr = next(reader)
        rid_col = hdr.index("row_id")
        pos = [hdr.index(f"f{i}") for i in range(1502, FULL_WIDTH)]
        for row in reader:
            rid = int(row[rid_col])
            if rid in seen or rid not in want:
                raise SystemExit(f"fresh row id duplicated/outside sample: {rid}")
            seen.add(rid)
            got = np.fromiter((float(row[j]) for j in pos), dtype=np.float64,
                              count=len(pos)).astype(np.float32)
            mism += int(np.count_nonzero(got.view(np.uint32) != want[rid].view(np.uint32)))
    if len(seen) != N_SAMPLE or mism:
        raise SystemExit(f"fresh sidecar mismatch: rows={len(seen)} cells={mism}")
    print(f"RESTORE_REEXTRACT_VERIFY rows={len(seen)} new_f32_cells={len(seen) * len(pos)} "
          f"bit_mismatches={mism}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["pairs", "bind", "verify", "draw", "reextract"])
    ap.add_argument("--set")
    ap.add_argument("--binary")
    ap.add_argument("--build-meta")
    a = ap.parse_args()
    if a.action == "pairs":
        for nme in sets():
            write_pairs(nme, ROOT / "pairs" / f"{nme}.tsv")
    elif a.action == "bind":
        bind(a.set, a.build_meta, a.binary)
    elif a.action == "verify":
        verify()
    elif a.action == "draw":
        draw()
    else:
        reextract()
