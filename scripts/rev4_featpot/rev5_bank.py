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
ASSESSMENT_KEY_COLUMNS = {"pair_key", "row_id", "ref_path", "dist_path", "ref_group", "pixels_identical",
    "entry", "image_id", "codec", "q", "codec_param", "param_kind", "case", "panel", "block",
    "source_reference", "source_distorted", "source_role", "delivery_receipt", "source_row_id", "origin",
    "ref_pixels_sha256", "dist_pixels_sha256", "knob", "n_stimuli", "width", "height"}
SLOT_RANGES = ((0, 228), (372, 720))  # Basic f0..155 + Peaks f156..227, V2 f372..719 (feature_set_id::ComputeToken)
SCHEMA = "rev5-featbank-v1"
HDR_TEACHER_SHA = {
    "train": "deb70e775b043a578c77e0c3ff27960ebfa936e9d901497f9d74389e6ce9fbce",
    "val": "4b0f39dea0255659232f248070c8683dd7edf5916c63621cbb099b4afb6c9057",
}
HDR_TRANSFORM = "score=10*q_jod;no-clipping"


def cmd_hdr_extract(a) -> int:
    """The same bank owner, explicit native HDR research mode and original teacher order."""
    import csv
    import e21_cheap_recipe as e21

    if a.revision != 5 or a.limit:
        raise ValueError("E26 requires a complete Rev5 role; partial tables are inadmissible")
    teacher = Path(a.hdr_teacher)
    tab = pq.read_table(teacher)
    role = (tab.schema.metadata or {}).get(b"zensim.hdrteach.role", b"").decode()
    if role not in HDR_TEACHER_SHA or sha256_file(teacher) != HDR_TEACHER_SHA[role]:
        raise ValueError("teacher is not the registered immutable HDRTEACH role")
    rows = tab.to_pylist()
    if any(r["role"] != role for r in rows) or len(rows) != {"train": 7425, "val": 3900}[role]:
        raise ValueError("teacher role/coverage mismatch")
    rows = [r for r in rows if role == "val" or r["agree"]]
    if len(rows) != {"train": 7390, "val": 3900}[role]:
        raise ValueError("registered agreement population mismatch")
    ids = e21.columns("by_v2fy")
    want = set(ids)
    hashes = {}
    for r in rows:
        for prefix in ("ref", "dist"):
            path, digest = r[f"{prefix}_path"], r[f"{prefix}_sha256"]
            if path in hashes and hashes[path] != digest:
                raise ValueError("conflicting native input hash")
            hashes[path] = digest
    for path, expected in hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"native input changed: {path}")
    outdir = Path(a.out) / a.set
    outdir.mkdir(parents=True, exist_ok=False)
    wd = outdir / "_work"
    wd.mkdir()
    keys = pa.Table.from_pylist(rows)
    pq.write_table(keys, outdir / "keys.parquet", compression="zstd")
    keys_sha = sha256_file(outdir / "keys.parquet")
    width, writer, producer, chunks = None, None, None, []
    env = dict(os.environ, ZENSIM_FORMULA_REV="5", ZENSIM_ROOT_FORM="sqrt", RAYON_NUM_THREADS=str(a.threads))
    start = time.time()
    try:
        for ci, lo in enumerate(range(0, len(rows), a.chunk)):
            group = rows[lo:lo + a.chunk]
            pairs, result = wd / f"pairs_{ci:03d}.tsv", wd / f"features_{ci:03d}.tsv"
            with pairs.open("w") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["q", "ref_path", "dist_path"])
                w.writerows((r["q"], r["ref_path"], r["dist_path"]) for r in group)
            cmd = [a.bin, "--pairs", str(pairs), "--ref-root", "/", "--enc-root", "/", "--absolute-pairs",
                   "--requested-ids", ",".join(map(str, ids)), "--input-contract", "hdr-common-primaries-v2-cicp-pq10000",
                   "--threads", str(a.threads), "--out", str(result)]
            t0 = time.time()
            with (wd / f"chunk_{ci:03d}.log").open("w") as lg:
                subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT, check=True)
            man = json.loads(result.with_suffix(".manifest.json").read_text())
            if man["formula_revision"] != "Rev5" or man["requested_ids"] != ids or man["rows"] != len(group):
                raise ValueError("native extractor revision/read-set/coverage mismatch")
            if producer is not None and producer != man["research"]:
                raise ValueError("research provenance changed between chunks")
            producer = man["research"]
            width = man["feature_count"]
            features = pacsv.read_csv(result, parse_options=pacsv.ParseOptions(delimiter="\t"),
                convert_options=pacsv.ConvertOptions(column_types={"q": pa.string(), "dist_basename": pa.string(),
                    **{f"f{i}": pa.float64() for i in range(width)}}))
            if (features["dist_basename"].to_pylist() != [Path(r["dist_path"]).name for r in group]
                    or features["q"].to_pylist() != [r["q"] for r in group]):
                raise ValueError("native row order/key join failed")
            for i in range(width):
                values = features[f"f{i}"].to_numpy()
                if not (np.isfinite(values).all() if i in want else np.isnan(values).all()):
                    raise ValueError(f"measured/absent contract failed f{i}")
            features = features.select([f"f{i}" for i in range(width)])
            features = features.add_column(0, "row_id", pa.array([r["row_id"] for r in group]))
            if writer is None:
                writer = pq.ParquetWriter(outdir / "features.parquet", features.schema, compression="zstd")
            writer.write_table(features)
            chunks.append(dict(rows=len(group), pairs_sha256=sha256_file(pairs), extractor_manifest=man,
                               wall_s=time.time()-t0))
            log(f"{a.set} chunk {ci}: {len(group)} native rows in {time.time()-t0:.1f}s")
    finally:
        if writer is not None:
            writer.close()
    manifest = dict(schema=SCHEMA, study="E26", role=role, rows=len(rows), feature_width=width,
                    formula_revision="Rev5", root_form="sqrt", input_contract="hdr-common-primaries-v2-cicp-pq10000",
                    requested_ids=ids, absent_slots="NaN", target_transform=HDR_TRANSFORM,
                    teacher_sha256=HDR_TEACHER_SHA[role], keys_sha256=keys_sha,
                    features_parquet_sha256=sha256_file(outdir / "features.parquet"),
                    build_commit=a.build_commit, binary_sha256=sha256_file(a.bin), research=producer,
                    native_inputs=hashes, row_order_sha256=hashlib.sha256(json.dumps([r["row_id"] for r in rows]).encode()).hexdigest(),
                    chunks=chunks, wall_s=time.time()-start, assembler_sha256=sha256_file(__file__))
    (outdir / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1)+"\n")
    return 0


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
    assessment = getattr(a, "instrument_manifest", None)
    spec = None
    key_path = OLD_BANK / s / "keys.parquet"
    if assessment:
        from v2c_wide import assessment_path as safe_path
        from v2_common import refuse_immutable_output
        if not s or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in s):
            raise ValueError("assessment set token required")
        spec = json.loads(safe_path(assessment).read_text())
        if a.limit or spec.get("schema") != "rev5-assessment-keys-v1" or spec.get("labels_read") is not False:
            raise ValueError("explicit label-free assessment keys required")
        key_path = safe_path(spec["keys"]["path"])
        roots = [Path(v) for v in spec["immutable_roots"]] + [key_path.parent, Path(assessment).parent]
        for dest in [Path(a.out) / s, Path(a.out) / "_work" / s]:
            refuse_immutable_output(dest, roots)
            if dest.exists():
                raise ValueError("fresh assessment bank/scratch required")
        if set(pq.ParquetFile(key_path).schema_arrow.names) - ASSESSMENT_KEY_COLUMNS:
            raise ValueError("label-bearing assessment keys forbidden before payload read")
        if a.revision != 5 or sha256_file(key_path) != spec["keys"]["sha256"] or sha256_file(a.bin) != spec["extractor"]["sha256"]:
            raise ValueError("changed assessment keys/extractor")
    keys = pq.read_table(key_path)
    if spec:
        if set(keys.column_names) - ASSESSMENT_KEY_COLUMNS or keys.num_rows != spec["rows"] or keys.num_rows == 0:
            raise ValueError("assessment key columns/rows are not label-free")
        for path in keys.column("ref_path").to_pylist() + keys.column("dist_path").to_pylist():
            safe_path(path)
        pixel_columns = [c for c in ("ref_path", "dist_path", "source_reference", "source_distorted") if c in keys.column_names]
        roots += sorted({safe_path(p).resolve().parent for c in pixel_columns for p in keys.column(c).to_pylist()})
        for dest in [Path(a.out) / s, Path(a.out) / "_work" / s]:
            refuse_immutable_output(dest, roots)
        if keys.column("row_id").to_pylist() != list(range(keys.num_rows)) or len(set(keys.column("pair_key").to_pylist())) != keys.num_rows:
            raise ValueError("assessment ordinal/key order required")
    n = a.limit or keys.num_rows
    ref, dist = keys.column("ref_path").to_pylist(), keys.column("dist_path").to_pylist()
    pk = keys.column("pair_key").to_pylist()
    outdir = Path(a.out) / (f"_partial/{s}" if a.limit else s)
    outdir.mkdir(parents=True, exist_ok=True)
    wd = Path(a.out) / "_work" / s
    wd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ZENSIM_FORMULA_REV=str(a.revision), ZENSIM_ROOT_FORM="sqrt", RAYON_NUM_THREADS=str(a.threads))
    want = set(requested_ids())
    audit_records = []
    writer, chunk_meta, tsv_hashes, fsids, eras, width, t_all = None, [], [], set(), None, None, time.time()
    for ci, lo in enumerate(range(0, n, a.chunk)):
        hi = min(lo + a.chunk, n)
        tsv = wd / f"pairs_{ci:03d}.tsv"
        with open(tsv, "w") as f:
            f.write("ref_path\tdist_path\thuman_score\trow_id\n")
            for i in range(lo, hi):
                f.write(f"{ref[i]}\t{dist[i]}\t{i if spec else 0}\t{i}\n")
        tsv_hashes.append(sha256_file(tsv))
        csvp = wd / f"chunk_{ci:03d}.csv"
        cmd = [a.bin, "--corpus", "pairs-tsv", "--path", str(tsv), "--out", str(csvp), "--restore-cuts", TOKENS,
               "--input-contract", "legacy-rgb8", "--era-label", a.era, "--force-tier", a.tier]
        audit_path = wd / f"chunk_{ci:03d}.audit.jsonl"
        if spec:
            cmd += ["--audit-jsonl", str(audit_path)]
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
        audits = None
        if spec:
            audits = [json.loads(line) for line in audit_path.read_text().splitlines()]
            audits.sort(key=lambda r: r["human_score"])
            if len(audits) != hi-lo or [r["human_score"] for r in audits] != list(range(lo, hi)):
                raise ValueError("assessment audit key coverage")
            for j, audit in enumerate(audits):
                expected = lo+j
                if audit["model_inputs"] or audit["reference"] != ref[expected] or audit["distorted"] != dist[expected]:
                    raise ValueError("features-only audit must have no model and exact paths")
                if "canonical_features_f32_le_sha256" in audit and audit["canonical_features_f32_le_sha256"] != hashlib.sha256(np.array(
                        [tab.column(f"f{i}")[j].as_py() for i in range(w)], dtype="<f4").tobytes()).hexdigest():
                    raise ValueError("extractor/CSV f32 byte parity failed")
                if "delivery_receipt" in keys.column_names:
                    delivery_path = safe_path(keys.column("delivery_receipt")[expected].as_py())
                    delivery = json.loads(delivery_path.read_text())
                    if any(delivery[k] != audit[k] for k in ("width", "height", "reference_pixels_sha256", "distorted_pixels_sha256")):
                        raise ValueError("canonical steering decoded pixel parity failed")
            audit_records.extend(audits)
        csv_bytes = csvp.stat().st_size
        csvp.unlink()
        chunk_meta.append({"chunk": ci, "rows": hi - lo, "pairs_tsv_sha256": tsv_hashes[-1], "extract_wall_s": round(t_ext, 2),
                           "csv_bytes_deleted": csv_bytes, "extractor_manifest": man, **({"audit_path":str(audit_path),"audit_sha256":sha256_file(audit_path)} if spec else {})})
        log(f"{s} chunk {ci} rows {lo}-{hi} extract {t_ext:.1f}s")
    writer.close()
    if len(fsids) != 1:
        sys.exit(f"feature_set_id changed between chunks: {fsids}")
    if not a.limit:
        if spec:
            extra = {k: pa.array([v[k] for v in audit_records]) for k in (
                "pixels_identical", "width", "height", "reference_file_sha256", "distorted_file_sha256",
                "reference_pixels_sha256", "distorted_pixels_sha256")}
            delivered = pa.table({**{k: keys.column(k) for k in keys.column_names if k not in extra}, **extra})
            pq.write_table(delivered, outdir / "keys.parquet", compression="zstd")
        else:
            (outdir / "keys.parquet").write_bytes(key_path.read_bytes())
    manifest = {
        "set": s, "schema": SCHEMA, "rows": n, "feature_width": width, "requested_slot_ranges": [list(r) for r in SLOT_RANGES],
        "absent_slots": "NaN", "restore_cuts": TOKENS, "build_commit": a.build_commit, "binary_sha256": sha256_file(a.bin),
        "formula_revision": f"Rev{a.revision}", "root_form": "sqrt", "input_contract": "legacy-rgb8", "tier_request": a.tier,
        "era_label": a.era, "feature_set_id": next(iter(fsids)), "formula_revision_eras": eras,
        "pairs_tsv_sha256": tsv_hashes, "pairs_origin": f"{key_path} (ref_path,dist_path in row order)",
        "keys_sha256": sha256_file(outdir / "keys.parquet") if spec else sha256_file(key_path),
        "features_parquet_sha256": sha256_file(outdir / "features.parquet"), "dtype": "float64",
        "chunks": chunk_meta, "wall_s_total": round(time.time() - t_all, 1),
        "assembler": {"path": "scripts/rev4_featpot/rev5_bank.py", "sha256": sha256_file(__file__)},
    }
    if spec:
        manifest["assessment"] = {"features_only": True, "labels_read": False,
            "input_keys": spec["keys"], "input_manifest_sha256": sha256_file(assessment),
            "instrument": spec["instrument"], "exposure_freeze": spec["exposure_freeze"],
            "human_score_semantics":"ordinal row key only; no target", "input_manifest":str(assessment),
            "all_decoded_pixels_bound":True, "immutable_roots":[str(r.resolve()) for r in roots],
            "extractor_feature_digest_available": all("canonical_features_f32_le_sha256" in r for r in audit_records)}
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
    ap.add_argument("--instrument-manifest", type=Path, help="explicit label-free assessment keys; historical bank default unchanged")
    ap.add_argument("--hdr-teacher", help="immutable HDRTEACH role; native E26 extraction")
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
        return cmd_hdr_extract(a) if a.hdr_teacher else cmd_extract(a)
    return cmd_verify_against(a)


if __name__ == "__main__":
    sys.exit(main())
