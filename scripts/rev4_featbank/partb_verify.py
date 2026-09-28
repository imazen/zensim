#!/usr/bin/env python3
"""Read-only audit of Part B sidecars against the promoted bank and raw CSVs."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from featbank_sets import BANK, POPULATED_IDS

ROOT = Path("/var/tmp/partb")
NAME = "features__rev4c1c4.parquet"
XNAME = "features__csfw_dvifm.parquet"
XIDS = list(range(944, 986))
BASE = json.loads(Path("benchmarks/rev4_featbank_extract_2026-09-23.json").read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_set(name):
    d = BANK / name
    manifest = json.loads((d / "_MANIFEST.json").read_text())
    side = manifest["sidecars"][NAME]
    build_meta = json.loads((ROOT / "build_meta.json").read_text())
    assert side["build_meta_sha256"] == sha(ROOT / "build_meta.json")
    assert side["build_commit"] == build_meta["repositories"]["zensim"]["commit"]
    if build_meta.get("schema") == "partb-c1c4-build-meta-v2":  # v2 records the compiled binary, so this is not circular
        assert build_meta["binary_sha256"] == side["binary_sha256"]
    assert side["binary_sha256"] == sha(ROOT / "target/release/examples/extract_features_372col")
    source_files = BASE["sets"][name]["files"]
    for base_name, expected in source_files.items():
        if base_name.startswith("labels__"):
            continue
        assert sha(d / base_name) == expected, f"{name}: baseline drift {base_name}"
    keys = pq.read_table(d / "keys.parquet", columns=["pair_key", "pixels_identical"])
    pks = keys.column("pair_key").to_pylist()
    assert len(pks) == len(set(pks)), f"{name}: duplicate key"
    identity = keys.column("pixels_identical").to_numpy()
    n = len(pks)
    path = d / NAME
    assert sha(path) == manifest["files"][NAME]["sha256"]
    pf = pq.ParquetFile(path)
    expected_cols = ["pair_key"] + [f"f{i}" for i in range(986, 1322)]
    assert pf.schema_arrow.names == expected_cols
    assert all(pf.schema_arrow.field(c).type == pa.float32() for c in expected_cols[1:])
    assert pf.metadata.num_rows == n == side["row_count"]
    c1_live = np.zeros(6, dtype=np.int64)
    c3_sat = np.zeros(4, dtype=np.int64)
    identity_bad = 0
    lo = 0
    for batch in pf.iter_batches(batch_size=65536):
        hi = lo + batch.num_rows
        assert batch.column(0).to_pylist() == pks[lo:hi], f"{name}: order {lo}"
        values = np.stack([batch.column(j + 1).to_numpy() for j in range(336)], axis=1)
        assert np.isfinite(values).all(), f"{name}: nonfinite {lo}"
        c1 = values[:, :96].reshape(-1, 12, 8)[:, :, :6]
        c1_live += np.count_nonzero(c1, axis=(0, 1))
        c3 = values[:, 168:312].reshape(-1, 12, 4, 3)
        c3_sat += np.count_nonzero(c3[:, :, :, 1] >= np.float32(1.2709334445868168), axis=(0, 1))
        if identity[lo:hi].any():
            ident = values[identity[lo:hi]]
            identity_bad += int(np.count_nonzero(ident[:, :168]))
            identity_bad += int(np.count_nonzero(ident[:, 312:]))
        lo = hi
    assert lo == n and identity_bad == 0

    # Same registered 1% seed as the producer, independently re-read from CSV.
    count = max(1, math.ceil(n / 100))
    chosen = sorted(np.random.default_rng(20260924).choice(n, count, replace=False).tolist())
    ident_rows = set(np.nonzero(identity)[0].tolist())
    chosen = sorted(set(chosen) | ident_rows)
    old_name = next(k for k in source_files if k.startswith("features__"))
    old = pq.read_table(d / old_name, columns=[f"f{i}" for i in POPULATED_IDS]).take(
        pa.array(chosen, type=pa.int64()))
    want = {rid: np.array([old.column(j)[i].as_py() for j in range(len(POPULATED_IDS))],
                          dtype=np.float32)
            for i, rid in enumerate(chosen)}
    mismatches = 0
    ident_mis = 0
    checked = 0
    with (ROOT / "raw" / f"{name}.csv").open(newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        id_col = hdr.index("row_id")
        positions = [hdr.index(f"f{i}") for i in POPULATED_IDS]
        for row in r:
            rid = int(row[id_col])
            if rid not in want:
                continue
            got = np.fromiter((float(row[j]) for j in positions), dtype=np.float64,
                              count=len(positions)).astype(np.float32)
            mm = int(np.count_nonzero(got.view(np.uint32) != want[rid].view(np.uint32)))
            if rid in ident_rows:
                ident_mis += mm
            else:
                mismatches += mm
            checked += 1
    assert checked == len(chosen) and mismatches == 0, f"{name}: old parity {checked}, {mismatches}"
    # csfw/dvifm sidecar: schema, key order, finite, manifest hash, and an independent CSV re-read
    xside = manifest["sidecars"][XNAME]
    assert xside["build_commit"] == side["build_commit"] and xside["binary_sha256"] == side["binary_sha256"]
    xpath = d / XNAME
    assert sha(xpath) == manifest["files"][XNAME]["sha256"]
    xpf = pq.ParquetFile(xpath)
    xcols = ["pair_key"] + [f"f{i}" for i in XIDS]
    assert xpf.schema_arrow.names == xcols
    assert all(xpf.schema_arrow.field(c).type == pa.float32() for c in xcols[1:])
    assert xpf.metadata.num_rows == n == xside["row_count"]
    x_live = np.zeros(len(XIDS), dtype=np.int64)
    x_ident = 0
    lo = 0
    for batch in xpf.iter_batches(batch_size=65536):
        hi = lo + batch.num_rows
        assert batch.column(0).to_pylist() == pks[lo:hi], f"{name}: csfw order {lo}"
        xv = np.stack([batch.column(j + 1).to_numpy() for j in range(len(XIDS))], axis=1)
        assert np.isfinite(xv).all(), f"{name}: csfw nonfinite {lo}"
        x_live += np.count_nonzero(xv, axis=0)
        if identity[lo:hi].any():
            x_ident += int(np.count_nonzero(xv[identity[lo:hi]]))
        lo = hi
    assert lo == n
    new_want = pq.read_table(path).take(pa.array(chosen, type=pa.int64()))
    x_want = pq.read_table(xpath).take(pa.array(chosen, type=pa.int64()))
    csv_new_mis = csv_x_mis = 0
    pos = {rid: i for i, rid in enumerate(chosen)}
    with (ROOT / "raw" / f"{name}.csv").open(newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        id_col = hdr.index("row_id")
        npos = [hdr.index(f"f{i}") for i in range(986, 1322)]
        xpos = [hdr.index(f"f{i}") for i in XIDS]
        for row in r:
            rid = int(row[id_col])
            if rid not in pos:
                continue
            i = pos[rid]
            g = np.array([float(row[j]) for j in npos], dtype=np.float64).astype(np.float32)
            w = np.array([new_want.column(j + 1)[i].as_py() for j in range(336)], dtype=np.float32)
            csv_new_mis += int(np.count_nonzero(g.view(np.uint32) != w.view(np.uint32)))
            g = np.array([float(row[j]) for j in xpos], dtype=np.float64).astype(np.float32)
            w = np.array([x_want.column(j + 1)[i].as_py() for j in range(len(XIDS))], dtype=np.float32)
            csv_x_mis += int(np.count_nonzero(g.view(np.uint32) != w.view(np.uint32)))
    assert csv_new_mis == 0 and csv_x_mis == 0, f"{name}: sidecar-vs-CSV {csv_new_mis} {csv_x_mis}"
    return {"rows": n, "old_checked": checked, "old_f32_mismatches": mismatches, "identical_rows_checked": len(ident_rows),
            "identical_old_cell_mismatches": ident_mis,
            "finite": True, "key_coverage": True, "identity_bad": identity_bad,
            "c1_live_cells_by_hat": c1_live.tolist(),
            "c3_p99_saturated_cells_by_map": c3_sat.tolist(),
            "sidecar_sha256": sha(path), "csfw_sidecar_sha256": sha(xpath),
            "csfw_live_cells_by_feature": x_live.tolist(), "csfw_identity_nonzero": x_ident,
            "sidecar_vs_csv_mismatches": csv_new_mis + csv_x_mis, "manifest_sha256": sha(d / "_MANIFEST.json"),
            "feature_set_id": side["feature_set_id"], "binary_sha256": side["binary_sha256"]}


def main():
    names = sorted(BASE["sets"])
    result = {name: verify_set(name) for name in names}
    assert len({x["feature_set_id"] for x in result.values()}) == 1
    assert len({x["binary_sha256"] for x in result.values()}) == 1
    out = ROOT / "verification.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"PARTB_VERIFY sets={len(result)} rows={sum(x['rows'] for x in result.values())} "
          f"old_checked={sum(x['old_checked'] for x in result.values())} "
          f"old_f32_mismatches={sum(x['old_f32_mismatches'] for x in result.values())} "
          f"sha256={sha(out)}")
    for name in names:
        x = result[name]
        print(f"{name} rows={x['rows']} old={x['old_checked']} mismatches=0 "
              f"c1_live={x['c1_live_cells_by_hat']} "
              f"c3_sat={x['c3_p99_saturated_cells_by_map']} sha256={x['sidecar_sha256']} "
              f"csfw_sha256={x['csfw_sidecar_sha256']} csfw_dead={[XIDS[j] for j, v in enumerate(x['csfw_live_cells_by_feature']) if v == 0]}")


if __name__ == "__main__":
    main()
