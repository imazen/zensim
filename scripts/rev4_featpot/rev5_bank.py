"""Rev5 feature bank (spec `benchmarks/rev5_spec_2026-10-04.md` §6): extract one set's basic + peaks + v2 slots and assemble
features.parquet + keys.parquet + _MANIFEST.json in the Rev4 bank's layout.

  python3 rev5_bank.py <set> --bin EXTRACTOR --build-commit SHA --revision 5 --era ERA [--out DIR] [--chunk N] [--limit N]

The pair list comes from the OLD bank's keys.parquet (ref_path, dist_path in row order, human_score 0 placeholder,
row_id = row position), exactly as the Rev4 re-extraction did (`/var/tmp/reextract/assemble_set.py`, which this generalises).
No labels file and nothing under `_sealed/` is opened.

The extractor is asked only for the Rev5 families (`--restore-cuts basic,peaks,v2` at the research layout width). The
research path emits unrequested slots as structural zeros; this assembler writes them as NaN, so an absent feature can never
pass for a measured zero. Every requested slot must be finite. The manifest records the requested slot ranges.

Identity gate (`--revision 4`): the same request at Rev4 must reproduce the Rev4 bank's requested columns bit for bit
(`verify-against`), which shows that requesting a subset does not change the subset's values.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

OLD_BANK = Path("/var/tmp/rev4-featbank/bank")
TOKENS = "basic,peaks,v2"
SLOT_RANGES = ((0, 228), (372, 720))  # Basic f0..155 + Peaks f156..227, V2 f372..719 (feature_set_id::ComputeToken)
SCHEMA = "rev5-featbank-v1"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def requested_ids() -> list[int]:
    return [i for lo, hi in SLOT_RANGES for i in range(lo, hi)]


def cmd_extract(a) -> int:
    s = a.set
    keys = pq.read_table(OLD_BANK / s / "keys.parquet")
    n = a.limit or keys.num_rows
    ref, dist = keys.column("ref_path").to_pylist(), keys.column("dist_path").to_pylist()
    pk = keys.column("pair_key").to_pylist()
    outdir = Path(a.out) / (f"_partial/{s}" if a.limit else s)
    outdir.mkdir(parents=True, exist_ok=True)
    wd = Path(a.out) / "_work" / s
    wd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ZENSIM_FORMULA_REV=str(a.revision), ZENSIM_ROOT_FORM="sqrt", RAYON_NUM_THREADS=str(a.threads))
    want = set(requested_ids())
    writer, chunk_meta, tsv_hashes, fsids, eras, width, t_all = None, [], [], set(), None, None, time.time()
    for ci, lo in enumerate(range(0, n, a.chunk)):
        hi = min(lo + a.chunk, n)
        tsv = wd / f"pairs_{ci:03d}.tsv"
        with open(tsv, "w") as f:
            f.write("ref_path\tdist_path\thuman_score\trow_id\n")
            for i in range(lo, hi):
                f.write(f"{ref[i]}\t{dist[i]}\t0\t{i}\n")
        tsv_hashes.append(sha256_file(tsv))
        csvp = wd / f"chunk_{ci:03d}.csv"
        cmd = [a.bin, "--corpus", "pairs-tsv", "--path", str(tsv), "--out", str(csvp), "--restore-cuts", TOKENS,
               "--input-contract", "legacy-rgb8", "--era-label", a.era, "--force-tier", a.tier]
        t0 = time.time()
        with open(wd / f"chunk_{ci:03d}.log", "w") as lg:
            rc = subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
        t_ext = time.time() - t0
        if rc != 0:
            sys.exit(f"extractor failed rc={rc} chunk {ci}: see {wd}/chunk_{ci:03d}.log")
        man = json.loads(Path(f"{csvp}.manifest.json").read_text())
        rm = json.loads(Path(f"{csvp}.research_manifest.json").read_text())
        fsids.add(man["feature_set_id"])
        eras = eras or rm["formula_revision_eras"]
        if eras != rm["formula_revision_eras"] or man["era_label"] != a.era or str(man["formula_revision"]) != str(a.revision):
            sys.exit(f"chunk {ci}: extractor manifest disagrees (era {man['era_label']!r}, revision {man['formula_revision']!r})")
        names = pacsv.read_csv(csvp, read_options=pacsv.ReadOptions(skip_rows_after_names=10**9)).column_names
        w = sum(1 for c in names if c.startswith("f") and c[1:].isdigit())
        width = width or w
        if w != width or not want <= set(range(w)):
            sys.exit(f"chunk {ci}: width {w} (first chunk {width}) does not cover the requested slots")
        conv = pacsv.ConvertOptions(column_types={"row_id": pa.int64(), **{f"f{i}": pa.float64() for i in range(w)}},
                                    include_columns=["row_id"] + [f"f{i}" for i in range(w)])
        tab = pacsv.read_csv(csvp, convert_options=conv)
        rid = tab.column("row_id").to_numpy()
        order = np.argsort(rid, kind="stable")
        if tab.num_rows != hi - lo or not (rid[order] == np.arange(lo, hi)).all():
            sys.exit(f"chunk {ci}: row_id join failed")
        tab = tab.take(pa.array(order))
        cols = {}
        for i in range(w):
            v = tab.column(f"f{i}").to_numpy()
            if i in want:
                if not np.isfinite(v).all():
                    sys.exit(f"chunk {ci}: non-finite value in requested slot f{i}")
                cols[f"f{i}"] = pa.array(v)
            else:
                cols[f"f{i}"] = pa.array(np.full(len(v), np.nan))  # absent, not zero
        out = pa.table({"pair_key": pa.array(pk[lo:hi]), "row_id": tab.column("row_id"), **cols})
        if writer is None:
            writer = pq.ParquetWriter(outdir / "features.parquet", out.schema, compression="zstd", compression_level=3)
        writer.write_table(out, row_group_size=5000)
        csv_bytes = csvp.stat().st_size
        csvp.unlink()
        chunk_meta.append({"chunk": ci, "rows": hi - lo, "pairs_tsv_sha256": tsv_hashes[-1], "extract_wall_s": round(t_ext, 2),
                           "csv_bytes_deleted": csv_bytes, "extractor_manifest": man})
        log(f"{s} chunk {ci} rows {lo}-{hi} extract {t_ext:.1f}s")
    writer.close()
    if len(fsids) != 1:
        sys.exit(f"feature_set_id changed between chunks: {fsids}")
    if not a.limit:
        (outdir / "keys.parquet").write_bytes((OLD_BANK / s / "keys.parquet").read_bytes())
    manifest = {
        "set": s, "schema": SCHEMA, "rows": n, "feature_width": width, "requested_slot_ranges": [list(r) for r in SLOT_RANGES],
        "absent_slots": "NaN", "restore_cuts": TOKENS, "build_commit": a.build_commit, "binary_sha256": sha256_file(a.bin),
        "formula_revision": f"Rev{a.revision}", "root_form": "sqrt", "input_contract": "legacy-rgb8", "tier_request": a.tier,
        "era_label": a.era, "feature_set_id": next(iter(fsids)), "formula_revision_eras": eras,
        "pairs_tsv_sha256": tsv_hashes, "pairs_origin": f"{OLD_BANK}/{s}/keys.parquet (ref_path,dist_path in row order)",
        "keys_sha256": sha256_file(OLD_BANK / s / "keys.parquet"),
        "features_parquet_sha256": sha256_file(outdir / "features.parquet"), "dtype": "float64",
        "chunks": chunk_meta, "wall_s_total": round(time.time() - t_all, 1),
        "assembler": {"path": "scripts/rev4_featpot/rev5_bank.py", "sha256": sha256_file(__file__)},
    }
    (outdir / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    log(f"{s} DONE rows {n} width {width} wall {manifest['wall_s_total']}s fsid {manifest['feature_set_id']}")
    return 0


def cmd_verify_against(a) -> int:
    """Requested columns of a subset-request bank equal the reference bank's columns bit for bit (row-limited ok)."""
    base = Path(a.out) / a.set
    if not base.is_dir():
        base = Path(a.out) / "_partial" / a.set
    mine = pq.read_table(base / "features.parquet")
    ref = pq.read_table(Path(a.reference) / a.set / "features.parquet")
    n = mine.num_rows
    if mine.column("pair_key").to_pylist() != ref.column("pair_key").to_pylist()[:n]:
        sys.exit("pair_key order differs")
    bad = []
    for i in requested_ids():
        x, y = mine.column(f"f{i}").to_numpy(), ref.column(f"f{i}").to_numpy()[:n]
        if not np.array_equal(x.view(np.uint64), y.view(np.uint64)):
            bad.append(i)
    absent_ok = all(np.isnan(mine.column(c).to_numpy()).all() for c in mine.column_names
                    if c.startswith("f") and c[1:].isdigit() and int(c[1:]) not in set(requested_ids()))
    print(json.dumps({"set": a.set, "rows": n, "requested": len(requested_ids()), "differing_slots": bad[:20],
                      "n_differing": len(bad), "absent_all_nan": absent_ok}))
    return 0 if not bad and absent_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "verify-against"])
    ap.add_argument("set")
    ap.add_argument("--bin")
    ap.add_argument("--build-commit", default="")
    ap.add_argument("--revision", type=int, choices=[4, 5], default=5)
    ap.add_argument("--era", default="")
    ap.add_argument("--out", default="/var/tmp/rev5-featbank")
    ap.add_argument("--reference", default="/var/tmp/rev4-featbank-r4")
    ap.add_argument("--chunk", type=int, default=25000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tier", default="native")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "extract":
        if not (a.bin and a.build_commit and a.era):
            ap.error("extract needs --bin, --build-commit and --era")
        return cmd_extract(a)
    return cmd_verify_against(a)


if __name__ == "__main__":
    sys.exit(main())
