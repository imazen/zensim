#!/usr/bin/env python3
"""Read-only audit of the C8 features__gmsbank sidecars (pass 2, /var/tmp/partb2)."""
import csv, hashlib, json
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from featbank_sets import BANK

ROOT = Path("/var/tmp/partb2")
NAME = "features__gmsbank.parquet"
IDS = list(range(1322, 1502))
SETS = json.loads(Path("benchmarks/rev4_featbank_extract_2026-09-23.json").read_text())["sets"]


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(name):
    d = BANK / name
    m = json.loads((d / "_MANIFEST.json").read_text())
    side = m["sidecars"][NAME]
    assert side["binary_sha256"] == sha(ROOT / "target/release/examples/extract_features_372col")
    assert side["build_meta_sha256"] == sha(ROOT / "build_meta.json")
    keys = pq.read_table(d / "keys.parquet", columns=["pair_key", "pixels_identical"])
    pks = keys.column("pair_key").to_pylist(); n = len(pks)
    assert len(set(pks)) == n
    ident = keys.column("pixels_identical").to_numpy()
    p = d / NAME
    assert sha(p) == m["files"][NAME]["sha256"]
    pf = pq.ParquetFile(p)
    cols = ["pair_key"] + [f"f{i}" for i in IDS]
    assert pf.schema_arrow.names == cols and pf.metadata.num_rows == n == side["row_count"]
    assert all(pf.schema_arrow.field(c).type == pa.float32() for c in cols[1:])
    live = np.zeros(len(IDS), dtype=np.int64); ident_nz = 0; lo = 0
    for b in pf.iter_batches(batch_size=65536):
        hi = lo + b.num_rows
        assert b.column(0).to_pylist() == pks[lo:hi]
        v = np.stack([b.column(j + 1).to_numpy() for j in range(len(IDS))], axis=1)
        assert np.isfinite(v).all()
        live += np.count_nonzero(v, axis=0)
        if ident[lo:hi].any():
            ident_nz += int(np.count_nonzero(v[ident[lo:hi]]))
        lo = hi
    assert lo == n
    # independent re-read of raw CSV on a 2% seeded sample (seed differs from the producer's)
    chosen = sorted(np.random.default_rng(7).choice(n, max(1, n // 50), replace=False).tolist())
    tbl = pq.read_table(p).take(pa.array(chosen, type=pa.int64())); pos = {r: i for i, r in enumerate(chosen)}
    mis = checked = 0
    with (ROOT / "raw" / f"{name}.csv").open(newline="") as f:
        r = csv.reader(f); h = next(r); rid = h.index("row_id"); g0 = h.index("f1322")
        for row in r:
            i = int(row[rid])
            if i in pos:
                g = np.array(row[g0:g0 + 180], dtype=np.float64).astype(np.float32)
                w = np.array([tbl.column(j + 1)[pos[i]].as_py() for j in range(180)], dtype=np.float32)
                mis += int(np.count_nonzero(g.view(np.uint32) != w.view(np.uint32))); checked += 1
    assert checked == len(chosen) and mis == 0, (name, checked, mis)
    assert ident_nz == 0
    return {"rows": n, "csv_rows_checked": checked, "csv_bit_mismatches": mis,
            "dead_columns": [IDS[j] for j, x in enumerate(live) if x == 0],
            "identity_nonzero": ident_nz, "sidecar_sha256": sha(p),
            "manifest_sha256": sha(d / "_MANIFEST.json"),
            "old_identical_rows": side["old_f32_identical_rows_checked"],
            "old_identical_cell_mismatches": side["old_f32_identical_cell_mismatches"],
            "feature_set_id": side["feature_set_id"], "binary_sha256": side["binary_sha256"]}


def main():
    res = {n: verify(n) for n in sorted(SETS)}
    assert len({x["feature_set_id"] for x in res.values()}) == 1
    assert len({x["binary_sha256"] for x in res.values()}) == 1
    out = ROOT / "verification.json"; out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"GMS_VERIFY sets={len(res)} rows={sum(x['rows'] for x in res.values())} "
          f"csv_rows_checked={sum(x['csv_rows_checked'] for x in res.values())} "
          f"csv_bit_mismatches=0 dead_sets={sum(bool(x['dead_columns']) for x in res.values())} "
          f"identical_rows={sum(x['old_identical_rows'] for x in res.values())} "
          f"identical_old_cell_mismatches={sum(x['old_identical_cell_mismatches'] for x in res.values())} "
          f"verification_sha256={sha(out)}")
    for n, x in res.items():
        print(f"{n} rows={x['rows']} sidecar_sha256={x['sidecar_sha256']}")


if __name__ == "__main__":
    main()
