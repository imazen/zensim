#!/usr/bin/env python3
"""Build the Rev4 f32 bank (docs/REV4_FEATURE_BANK_PLAN_2026-09-23 §2.3).

Existing-family paths convert Rev3 f64 caches or bind fresh extraction by
row_id. Part B generates label-free pairs from promoted keys and binds the
landed C1–C4 slots by row_id plus decoded pixel hashes. No label analysis.
The set registry and key contract live in featbank_sets.py and pair_key().
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from featbank_sets import (
    BANK, ERA, FEATURE_SET_ID, INPUT_CONTRACT, POPULATED_IDS, ROW_GROUP, SETS,
    SIDECAR_NAME, STRUCTURAL_ZERO_IDS, audit_row_id, basename_norm,
    derivations,
)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pair_key(ref_hex, dist_hex):
    return hashlib.sha256(
        (ref_hex + dist_hex + INPUT_CONTRACT).encode()
    ).hexdigest()


def load_audit_keyed(path):
    """row_id -> audit record, keyed by extra_targets row_id (ceiling/baseline)."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rid = audit_row_id(r)
            if rid is None:
                continue
            out[rid] = r
    return out


def load_audit_positional(path):
    """audit records in file order (extractor writes them in pairs order)."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def read_pairs(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_feats_csv(path):
    """Read f64 rows keyed by row_id; CSV is sorted by ref_basename."""
    out = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        fi = [i for i, c in enumerate(hdr) if c[0] == "f" and c[1:].isdigit()]
        assert [int(hdr[i][1:]) for i in fi] == list(range(944))
        if "row_id" not in hdr:
            raise SystemExit(f"{path}: missing row_id column (rebuild pairs)")
        rid_i = hdr.index("row_id")
        for row in r:
            rid = int(row[rid_i])
            if rid in out:
                raise SystemExit(f"{path}: duplicate row_id {rid}")
            out[rid] = np.array([float(row[i]) for i in fi], dtype=np.float64)
    return out


def write_features_sidecar(path, pair_keys, feat_rows):
    """feat_rows: list/2D-array of 944 f64 values per row (all cols)."""
    cols = {"pair_key": pa.array(pair_keys, type=pa.string())}
    m = np.asarray(feat_rows, dtype=np.float64)
    for fid in POPULATED_IDS:
        cols[f"f{fid}"] = pa.array(m[:, fid].astype(np.float32))
    t = pa.table(cols)
    pq.write_table(
        t, path,
        compression="zstd", compression_level=3, use_dictionary=False,
        column_encoding={f"f{fid}": "BYTE_STREAM_SPLIT" for fid in POPULATED_IDS},
        row_group_size=ROW_GROUP,
    )
    return t.num_rows


def verify_sidecar(path, feat_rows):
    """Every stored f32 == np.float32(source f64). Returns (bad_cells, rows)."""
    back = pq.read_table(path)
    m = np.asarray(feat_rows, dtype=np.float64)
    bad = 0
    for fid in POPULATED_IDS:
        want = m[:, fid].astype(np.float32)
        got = back.column(f"f{fid}").to_numpy()
        bad += int((got.view(np.uint32) != want.view(np.uint32)).sum())
    return bad, back.num_rows


def check_structural_zeros(feat_rows, set_name):
    m = np.asarray(feat_rows, dtype=np.float64)
    nz = [(fid, int((m[:, fid] != 0).sum())) for fid in STRUCTURAL_ZERO_IDS]
    nz = [(f, c) for f, c in nz if c]
    if nz:
        raise SystemExit(f"{set_name}: structural-zero cols non-zero: {nz[:8]}")


def emit_set(set_name, cfg, stimuli, feat_matrix):
    """Emit key-deduplicated features and source-row labels from aligned f64s."""
    t0 = time.time()
    out_dir = BANK / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_stimuli = len(stimuli)
    if n_stimuli != cfg["rows"]:
        raise SystemExit(
            f"{set_name}: {n_stimuli} stimulus rows != expected {cfg['rows']}"
        )

    seen = {}
    key_rows = []          # (pk, stimulus index of first occurrence)
    counts = Counter()
    for i, s in enumerate(stimuli):
        counts[s["pair_key"]] += 1
        if s["pair_key"] not in seen:
            seen[s["pair_key"]] = i
            key_rows.append((s["pair_key"], i))
    n_keys = len(key_rows)
    n_collapsed = n_stimuli - n_keys
    if n_collapsed:
        print(f"{set_name}: {n_collapsed} pixel-identical stimuli collapse "
              f"({n_stimuli} rows -> {n_keys} unique pair_keys)")

    pks = [pk for pk, _ in key_rows]
    first = [i for _, i in key_rows]
    key_feats = feat_matrix[np.array(first)]

    # collapsed rows must carry identical features (identical pixels ->
    # identical extraction); check every duplicate against its key row.
    if n_collapsed:
        first_pos = {pk: j for j, (pk, _) in enumerate(key_rows)}
        bad_dup = 0
        for i, s in enumerate(stimuli):
            j = first_pos[s["pair_key"]]
            if j != i and not np.array_equal(feat_matrix[i], key_feats[j], equal_nan=True):
                bad_dup += 1
        if bad_dup:
            raise SystemExit(
                f"{set_name}: {bad_dup} collapsed stimuli have differing features"
            )

    check_structural_zeros(key_feats, set_name)

    keys = pa.table({
        "pair_key": pa.array(pks, type=pa.string()),
        "row_id": pa.array(list(range(n_keys)), type=pa.int64()),
        "ref_pixels_sha256": pa.array([stimuli[i]["ref_h"] for i in first], type=pa.string()),
        "dist_pixels_sha256": pa.array([stimuli[i]["dist_h"] for i in first], type=pa.string()),
        "ref_group": pa.array([stimuli[i]["ref_group"] for i in first], type=pa.string()),
        "codec": pa.array([stimuli[i]["codec"] for i in first], type=pa.string()),
        "knob": pa.array([stimuli[i]["knob"] for i in first], type=pa.string()),
        "width": pa.array([stimuli[i]["width"] for i in first], type=pa.int32()),
        "height": pa.array([stimuli[i]["height"] for i in first], type=pa.int32()),
        "pixels_identical": pa.array([stimuli[i]["pixels_identical"] for i in first], type=pa.bool_()),
        "ref_path": pa.array([stimuli[i]["ref_path"] for i in first], type=pa.string()),
        "dist_path": pa.array([stimuli[i]["dist_path"] for i in first], type=pa.string()),
        "n_stimuli": pa.array([counts[pk] for pk, _ in key_rows], type=pa.int32()),
    })
    keys_path = out_dir / "keys.parquet"
    pq.write_table(keys, keys_path, compression="zstd", compression_level=3)

    feat_path = out_dir / SIDECAR_NAME
    write_features_sidecar(feat_path, pks, key_feats)

    label_files = write_labels(cfg, set_name, out_dir, stimuli)

    bad, back_rows = verify_sidecar(feat_path, key_feats)
    if bad or back_rows != n_keys:
        raise SystemExit(
            f"{set_name}: f32 check failed ({bad} cells, rows {back_rows})"
        )

    extra = {}
    for k in ("pairs", "audit", "feats"):
        if cfg.get(k):
            extra[k] = cfg[k]
    extra.update(cfg.get("extra_source_files", {}))
    return finish(set_name, cfg, out_dir, keys_path, feat_path, label_files,
                  n_stimuli, n_keys, n_collapsed, bad, t0, extra)


def assemble_extract(set_name, cfg):
    """Bind fresh pairs, audits and sorted feature CSV by row_id."""
    pairs = read_pairs(cfg["pairs"])
    audits = load_audit_positional(cfg["audit"])
    feats_by_rid = read_feats_csv(cfg["feats"])
    n_pairs = len(pairs)
    if not (len(audits) == len(feats_by_rid) == n_pairs):
        raise SystemExit(
            f"{set_name}: counts differ pairs={n_pairs} audit={len(audits)} feats={len(feats_by_rid)}"
        )

    # keyed bind checks (mechanical; label echoed for join check only)
    bad_bind = 0
    audits_by_rid = {}
    for i in range(n_pairs):
        a = audits[i]
        pr = pairs[i]
        rid = int(pr["row_id"])
        if audit_row_id(a) != rid:
            bad_bind += 1
            continue
        if basename_norm(a["reference"]) != basename_norm(pr["ref_path"]):
            bad_bind += 1
            continue
        if basename_norm(a["distorted"]) != basename_norm(pr["dist_path"]):
            bad_bind += 1
            continue
        if abs(float(a["human_score"]) - float(pr["human_score"])) > 1e-9:
            bad_bind += 1
            continue
        audits_by_rid[rid] = a
    if bad_bind:
        raise SystemExit(f"{set_name}: {bad_bind} keyed bind failures")
    if len(audits_by_rid) != n_pairs:
        raise SystemExit(f"{set_name}: non-unique row_id in pairs/audit")

    sel = cfg.get("select")
    stimuli, feat_rows = [], []
    for i in range(n_pairs):
        rid = int(pairs[i]["row_id"])
        a = audits_by_rid[rid]
        if sel is not None and not sel(i, pairs[i], a):
            continue
        if not a.get("distorted_pixels_sha256"):
            raise SystemExit(f"{set_name}: row {i} lacks distorted pixel hash")
        ref_h = a["reference_pixels_sha256"]
        dist_h = a["distorted_pixels_sha256"]
        codec, knob = derivations(set_name, a["distorted"])
        stimuli.append({
            "pair_key": pair_key(ref_h, dist_h),
            "src_row_id": rid,
            "ref_h": ref_h, "dist_h": dist_h,
            "ref_group": basename_norm(a["reference"]),
            "codec": codec, "knob": knob,
            "width": int(a["width"]), "height": int(a["height"]),
            "pixels_identical": bool(a["pixels_identical"]),
            "ref_path": a["reference"], "dist_path": a["distorted"],
            "score": float(a["human_score"]),
        })
        feat_rows.append(feats_by_rid[rid])
    return emit_set(set_name, cfg, stimuli, np.array(feat_rows, dtype=np.float64))


def write_labels(cfg, set_name, out_dir, stimuli):
    """Mechanical copy-through. Confirmation sets (labels=none) emit nothing —
    pixels only, ruling D4. Keeps every stimulus row (pixel-identical stimuli
    share pair_key)."""
    files = {}
    mode = cfg["labels"]
    if mode == "none":
        return files
    kind = mode.removeprefix("sealed_")
    col = "ssim2_oracle" if kind == "ssim2_oracle" else "human_score"
    cols = {
        "pair_key": pa.array([s["pair_key"] for s in stimuli], type=pa.string()),
        "source_row_id": pa.array([s["src_row_id"] for s in stimuli], type=pa.int64()),
        "codec": pa.array([s["codec"] for s in stimuli], type=pa.string()),
        "knob": pa.array([s["knob"] for s in stimuli], type=pa.string()),
        col: pa.array([s["score"] for s in stimuli], type=pa.float64()),
    }
    extra = stimuli[0].get("extra_label_cols")
    if extra:
        for k in extra:
            cols[k] = pa.array([s.get(k) for s in stimuli])
    name = f"labels__{kind}.parquet"
    lp = out_dir / name
    pq.write_table(pa.table(cols), lp, compression="zstd", compression_level=3)
    files[name] = lp
    return files


def finish(set_name, cfg, out_dir, keys_path, feat_path, label_files,
           n_stimuli, n_keys, n_collapsed, bad, t0, extra_files):
    files = {}
    for name, p in [("keys.parquet", keys_path), (SIDECAR_NAME, feat_path)] + list(
        label_files.items()
    ):
        files[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    manifest = {
        "set": set_name,
        "role": cfg["role"],
        "schema": "rev4-featbank-v1",
        "feature_set_id": FEATURE_SET_ID,
        "formula_revision": "3",
        "root_form": "sqrt",
        "input_contract": INPUT_CONTRACT,
        "era": ERA,
        "build_commit": cfg["build_commit"],
        "binary_sha256": cfg.get("binary_sha256"),
        "decoder_identities": cfg.get("decoder_identities", "canonical zen_decode route via pinned extract-native-admission"),
        "admission_sha256": cfg.get("admission_sha256"),
        "pairs_sha256": sha256_file(cfg["pairs"]) if cfg.get("pairs") else cfg.get("pairs_sha256"),
        "input_file_sha256s": cfg.get("input_files_note"),
        "source_sha256": {k: sha256_file(v) for k, v in extra_files.items()},
        "source_files": {k: str(v) for k, v in extra_files.items()},
        "pixel_hash_source": cfg["pixel_hash_source"],
        "populated_feature_ids": POPULATED_IDS,
        "structural_zero_feature_ids": STRUCTURAL_ZERO_IDS,
        "dtype": "f32",
        "cast": "f64->f32 round-nearest-even",
        "pair_key_construction": "sha256(utf8(ref_pixels_sha256_hex)||utf8(dist_pixels_sha256_hex)||utf8('legacy-rgb8')) hex",
        "row_count": n_stimuli,
        "unique_pair_keys": n_keys,
        "pixel_identical_stimuli_collapsed": n_collapsed,
        "env": {"python": sys.version.split()[0], "pyarrow": pa.__version__},
        "converter": "scripts/rev4_featbank/convert_cache.py + featbank_sets.py (quarantine/devin/featbank-extract)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
        "checks": {
            "row_count_matches_source": n_stimuli == cfg["rows"],
            "pair_key_unique": True,
            "f32_full_check_mismatched_cells": bad,
            "audit_join_complete": True,
        },
        "label_scale": cfg.get("label_scale"),
        "label_handling": cfg["label_note"],
    }
    mpath = out_dir / "_MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    dt = time.time() - t0
    bpr = files[SIDECAR_NAME]["bytes"] / n_keys
    print(
        f"{set_name}: stimuli={n_stimuli} keys={n_keys} "
        f"sidecar={files[SIDECAR_NAME]['bytes']}B "
        f"{bpr:.0f}B/row f32_mismatch={bad} ({dt:.1f}s)"
    )
    return manifest


# ---------- parquet-convert path (existing caches) ----------


def convert_parquet(set_name, cfg):
    src_tables = [pq.read_table(p) for p in cfg["parquets"]]
    src = pa.concat_tables(src_tables) if len(src_tables) > 1 else src_tables[0]
    if "select_row_ids" in cfg:
        ids = cfg["select_row_ids"]
        keep = set(ids() if callable(ids) else ids)
        mask = pa.array([r in keep for r in src.column("row_id").to_pylist()])
        src = src.filter(mask)
    order = np.argsort(src.column("row_id").to_numpy())
    src = src.take(pa.array(order))

    zids = zero_ids(src)
    if zids != STRUCTURAL_ZERO_IDS:
        raise SystemExit(f"{set_name}: structural-zero ids differ: {zids}")

    audit = load_audit_keyed(cfg["audit"])
    meta = cfg["row_meta"](src)

    # ref_basename parity: audit basename vs parquet ref_basename (normalized;
    # parquet may prefix the corpus, e.g. 'kadid:I02')
    def _norm_ref(x):
        s = str(x).split(":")[-1]
        return Path(s).stem.lower()

    rb = [_norm_ref(x) for x in src.column("ref_basename").to_pylist()]
    scores = src.column("human_score").to_pylist()
    has_oo = "original_oracle" in src.schema.names
    oo = src.column("original_oracle").to_pylist() if has_oo else [None] * len(rb)

    stimuli = []
    missing = []
    for j, r in enumerate(meta):
        a = audit.get(r["row_id"])
        if a is None or not a.get("distorted_pixels_sha256"):
            missing.append(r["row_id"])
            continue
        if basename_norm(a["reference"]) != rb[j]:
            raise SystemExit(
                f"{set_name}: ref_basename mismatch at row {r['row_id']}: "
                f"{a['reference']} vs {rb[j]}"
            )
        ref_h = a["reference_pixels_sha256"]
        dist_h = a["distorted_pixels_sha256"]
        codec = r["codec"]
        knob = r["knob"]
        if codec is None:
            dp = Path(a["distorted"])
            codec = dp.parent.name
            knob = dp.stem
        s = {
            "pair_key": pair_key(ref_h, dist_h),
            "src_row_id": r["row_id"],
            "ref_h": ref_h, "dist_h": dist_h,
            "ref_group": r["ref_group"],
            "codec": codec, "knob": knob,
            "width": int(a["width"]), "height": int(a["height"]),
            "pixels_identical": bool(a["pixels_identical"]),
            "ref_path": a["reference"], "dist_path": a["distorted"],
            "score": float(scores[j]) if scores[j] is not None else float("nan"),
        }
        if has_oo:
            s["extra_label_cols"] = ["original_oracle"]
            s["original_oracle"] = oo[j]
        stimuli.append(s)
    if missing:
        raise SystemExit(f"{set_name}: {len(missing)} rows lack audit/hash: {missing[:10]}")

    # stored ref-hash column cross-check where present
    if "reference_pixels_sha256" in src.schema.names:
        stored = src.column("reference_pixels_sha256").to_pylist()
        if [s["ref_h"] for s in stimuli] != stored:
            raise SystemExit(f"{set_name}: audit ref hash != stored column")

    feat_rows = np.stack(
        [src.column(f"f{i}").to_numpy().astype(np.float64) for i in range(944)],
        axis=1,
    )
    return emit_set(set_name, cfg, stimuli, feat_rows)


def zero_ids(src_table):
    ids = []
    for name in src_table.schema.names:
        if name.startswith("f") and name[1:].isdigit():
            f = src_table.schema.field(name)
            if str(f.type) == "int64":
                col = src_table.column(name).to_numpy()
                assert (col == 0).all(), f"{name} int64 but not all-zero"
                ids.append(int(name[1:]))
    return sorted(ids)


PARTB_NAME = "features__rev4c1c4.parquet"
PARTB_IDS = list(range(986, 1322))
CSFW_IDS = list(range(944, 986))  # csfw f944-955 + DVIFM C7 f956-985 (coordinator add 2026-09-24)
CSFW_NAME = "features__csfw_dvifm.parquet"
GMS_IDS = list(range(1322, 1502))  # C8 gmsbank
GMS_NAME = "features__gmsbank.parquet"


def partb_keys(set_name, limit=None):
    path = BANK / set_name / "keys.parquet"
    t = pq.read_table(path, columns=["pair_key", "row_id", "ref_path",
                                    "dist_path", "ref_pixels_sha256",
                                    "dist_pixels_sha256", "pixels_identical"])
    if limit is not None:
        if limit <= 0 or limit > t.num_rows:
            raise SystemExit(f"invalid Part B limit {limit} for {set_name}")
        t = t.slice(0, limit)
    d = t.to_pydict()
    n = t.num_rows
    if d["row_id"] != list(range(n)):
        raise SystemExit(f"{set_name}: bank row_id is not dense/in order")
    if len(set(d["pair_key"])) != n:
        raise SystemExit(f"{set_name}: duplicate bank pair_key")
    for i in range(n):
        if pair_key(d["ref_pixels_sha256"][i],
                    d["dist_pixels_sha256"][i]) != d["pair_key"][i]:
            raise SystemExit(f"{set_name}: stored pair_key disagrees at {i}")
    return d


def write_partb_pairs(set_name, path, limit):
    keys = partb_keys(set_name, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        # human_score is a constant 0 placeholder: the extractor audit requires the column, no label is read
        w.writerow(["ref_path", "dist_path", "human_score", "row_id"])
        for i, (ref, dist) in enumerate(zip(keys["ref_path"], keys["dist_path"])):
            w.writerow([ref, dist, 0, i])
    print(f"PARTB_PAIRS set={set_name} rows={len(keys['pair_key'])} "
          f"sha256={sha256_file(path)} path={path}")


def partb_old_sample(set_name, indices):
    old_name = next(n for n in json.loads(
        (BANK / set_name / "_MANIFEST.json").read_text())["files"]
        if n.startswith("features__"))
    path = BANK / set_name / old_name
    names = [f"f{i}" for i in POPULATED_IDS]
    t = pq.read_table(path, columns=["pair_key"] + names)
    keys = partb_keys(set_name)
    if t.column("pair_key").to_pylist() != keys["pair_key"]:
        raise SystemExit(f"{set_name}: original sidecar order disagrees with keys")
    ix = pa.array(indices, type=pa.int64())
    sampled = t.take(ix)
    result = {int(i): np.array([sampled.column(name)[j].as_py()
                                for name in names], dtype=np.float32)
              for j, i in enumerate(indices)}
    return result


def _write_sidecar(set_name, path, keys, ids, arr, n):
    tmp = path.with_suffix(".tmp")
    schema = pa.schema([pa.field("pair_key", pa.string())] +
                       [pa.field(f"f{i}", pa.float32()) for i in ids])
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=3,
                          use_dictionary=False,
                          column_encoding={f"f{i}": "BYTE_STREAM_SPLIT"
                                           for i in ids}) as writer:
        for lo in range(0, n, ROW_GROUP):
            hi = min(lo + ROW_GROUP, n)
            cols = [pa.array(keys["pair_key"][lo:hi], type=pa.string())]
            cols.extend(pa.array(arr[lo:hi, j], type=pa.float32())
                        for j in range(len(ids)))
            writer.write_table(pa.Table.from_arrays(cols, schema=schema), row_group_size=ROW_GROUP)
    with pq.ParquetFile(tmp) as pf:
        if pf.metadata.num_rows != n or pf.schema_arrow != schema:
            raise SystemExit(f"{set_name}: Parquet row/schema roundtrip failed")
        lo = 0
        for batch in pf.iter_batches(batch_size=ROW_GROUP):
            hi = lo + batch.num_rows
            if batch.column(0).to_pylist() != keys["pair_key"][lo:hi]:
                raise SystemExit(f"{set_name}: Parquet key order changed at {lo}")
            for j in range(len(ids)):
                got = batch.column(j + 1).to_numpy()
                if not np.array_equal(got.view(np.uint32),
                                      arr[lo:hi, j].view(np.uint32)):
                    raise SystemExit(f"{set_name}: Parquet value changed at {lo}, f{ids[j]}")
            lo = hi
        if lo != n:
            raise SystemExit(f"{set_name}: Parquet coverage {lo}/{n}")
    os.replace(tmp, path)


def bind_partb(set_name, csv_path, audit_path, binary, build_commit, limit):
    keys = partb_keys(set_name, limit)
    n = len(keys["pair_key"])
    pairs_path = csv_path.parent.parent / "pairs" / f"{set_name}{'_sample' if limit else ''}.tsv"
    if not pairs_path.exists():
        raise SystemExit(f"missing pairs list {pairs_path}")
    producer_path = Path(str(csv_path) + ".manifest.json")
    producer = json.loads(producer_path.read_text())
    bin_sha = sha256_file(binary)
    if producer.get("producer_binary_sha256") != bin_sha:
        raise SystemExit("extractor manifest binary hash differs from build")
    if producer.get("formula_revision") != "3" or producer.get("layout") != "w1322":
        raise SystemExit("extractor manifest has wrong formula/layout")
    if producer.get("input_contract", "legacy-rgb8") != "legacy-rgb8":
        raise SystemExit("extractor manifest has wrong input contract")
    feature_set_id = producer["feature_set_id"]
    if not feature_set_id:
        raise SystemExit("extractor did not emit a feature_set_id")
    build_meta_path = Path("/var/tmp/partb/build_meta.json")
    build_meta = json.loads(build_meta_path.read_text())
    if build_meta["repositories"]["zensim"]["commit"] != build_commit:
        raise SystemExit("build commit differs from clean-source metadata")

    with audit_path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                raise SystemExit(f"{set_name}: extra audit row")
            a = json.loads(line)
            if (audit_row_id(a) != i or
                a["reference"] != keys["ref_path"][i] or
                a["distorted"] != keys["dist_path"][i] or
                a["reference_pixels_sha256"] != keys["ref_pixels_sha256"][i] or
                a["distorted_pixels_sha256"] != keys["dist_pixels_sha256"][i] or
                pair_key(a["reference_pixels_sha256"], a["distorted_pixels_sha256"])
                    != keys["pair_key"][i]):
                raise SystemExit(f"{set_name}: audit/key binding failed at {i}")
        if i + 1 != n:
            raise SystemExit(f"{set_name}: audit has {i + 1} of {n} rows")

    rng = np.random.default_rng(20260924)
    count = n if limit else max(1, math.ceil(n / 100))
    indices = sorted(rng.choice(n, count, replace=False).tolist())
    # Ruling 2026-09-25 (user): pixels-identical pairs are checked in full but RECORDED, not gated:
    # the bank stores 0 in ~80 old-family slots for identical pairs while the extractor computes
    # nonzero values. Non-identical rows stay strictly bit-exact.
    ident_flags = np.asarray(keys["pixels_identical"], dtype=np.bool_)
    ident_set = set(np.nonzero(ident_flags)[0].tolist())
    indices = sorted(set(indices) | ident_set)
    n_ident_checked = len(ident_set)
    n_nonident_checked = len(indices) - n_ident_checked
    bad_ident = 0
    expected_old = partb_old_sample(set_name, indices)
    new = np.memmap(csv_path.parent /
                    f"{set_name}{'_sample' if limit else ''}.partb.f32", mode="w+",
                    dtype=np.float32, shape=(n, len(PARTB_IDS)))
    xnew = np.memmap(csv_path.parent /
                     f"{set_name}{'_sample' if limit else ''}.csfw.f32", mode="w+",
                     dtype=np.float32, shape=(n, len(CSFW_IDS)))
    seen = np.zeros(n, dtype=np.bool_)
    bad_old = 0
    with csv_path.open(newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        rid_i = hdr.index("row_id")
        if [hdr.index(f"f{i}") for i in PARTB_IDS] != list(
                range(hdr.index("f986"), hdr.index("f986") + len(PARTB_IDS))):
            raise SystemExit("new feature columns are not contiguous")
        if [int(x[1:]) for x in hdr if x.startswith("f") and x[1:].isdigit()] != list(range(1322)):
            raise SystemExit("extractor CSV does not contain f0-f1321")
        new_start = hdr.index("f986")
        x_start = hdr.index("f944")
        if [hdr.index(f"f{i}") for i in CSFW_IDS] != list(range(x_start, x_start + len(CSFW_IDS))):
            raise SystemExit("csfw/dvifm feature columns are not contiguous")
        old_pos = [hdr.index(f"f{i}") for i in POPULATED_IDS]
        for row in r:
            rid = int(row[rid_i])
            if rid < 0 or rid >= n or seen[rid]:
                raise SystemExit(f"{set_name}: invalid/duplicate CSV row_id {rid}")
            seen[rid] = True
            values = np.fromiter((float(x) for x in row[new_start:new_start + len(PARTB_IDS)]),
                                 dtype=np.float64, count=len(PARTB_IDS))
            if not np.isfinite(values).all():
                raise SystemExit(f"{set_name}: non-finite new value at {rid}")
            new[rid] = values.astype(np.float32)
            xvals = np.fromiter((float(x) for x in row[x_start:x_start + len(CSFW_IDS)]),
                                dtype=np.float64, count=len(CSFW_IDS))
            if not np.isfinite(xvals).all():
                raise SystemExit(f"{set_name}: non-finite csfw/dvifm value at {rid}")
            xnew[rid] = xvals.astype(np.float32)
            if not np.isfinite(xnew[rid]).all():
                raise SystemExit(f"{set_name}: non-finite csfw/dvifm f32 value at {rid}")
            if not np.isfinite(new[rid]).all():
                raise SystemExit(f"{set_name}: non-finite f32 value at {rid}")
            if rid in expected_old:
                got = np.fromiter((float(row[p]) for p in old_pos),
                                  dtype=np.float64, count=len(old_pos)).astype(np.float32)
                mm = int(np.count_nonzero(got.view(np.uint32) !=
                                          expected_old[rid].view(np.uint32)))
                if rid in ident_set:
                    bad_ident += mm
                else:
                    bad_old += mm
    if not seen.all() or bad_old:
        raise SystemExit(f"{set_name}: coverage={int(seen.sum())}/{n} "
                         f"old_f32_mismatches_nonidentical={bad_old}/{n_nonident_checked * len(POPULATED_IDS)}")
    new.flush()
    xnew.flush()
    x_live = [int(np.count_nonzero(xnew[:, j])) for j in range(len(CSFW_IDS))]
    x_ident_nonzero = int(np.count_nonzero(xnew[np.asarray(keys["pixels_identical"], dtype=np.bool_)]))

    c1 = np.asarray(new[:, :96]).reshape(n, 12, 8)[:, :, :6]
    c1_live = [int(np.count_nonzero(c1[:, :, j])) for j in range(6)]
    c3 = np.asarray(new[:, 168:312]).reshape(n, 12, 4, 3)
    top_edge = np.float32(1.2709334445868168)
    c3_sat = [int(np.count_nonzero(c3[:, :, m, 1] >= top_edge)) for m in range(4)]
    ident = np.asarray(keys["pixels_identical"], dtype=np.bool_)
    identity_bad = int(np.count_nonzero(new[ident, :168])) + int(
        np.count_nonzero(new[ident, 312:]))
    if identity_bad:
        raise SystemExit(f"{set_name}: {identity_bad} nonzero C1/C2/C4 identity cells")
    print(f"PARTB_CHECK set={set_name} rows={n} old_checked={n_nonident_checked} "
          f"old_f32_mismatches={bad_old} identical_rows_checked={n_ident_checked} "
          f"identical_old_cell_mismatches={bad_ident} finite=1 unique=1 "
          f"identity_bad={identity_bad} c1_live={c1_live} c3_p99_saturated={c3_sat}")
    print(f"PARTB_CHECK_CSFW set={set_name} rows={n} finite=1 unique=1 "
          f"identity_nonzero={x_ident_nonzero} dead_columns={[CSFW_IDS[j] for j, v in enumerate(x_live) if v == 0]}")
    if limit:
        return

    path = BANK / set_name / PARTB_NAME
    _write_sidecar(set_name, path, keys, PARTB_IDS, new, n)
    xpath = BANK / set_name / CSFW_NAME
    _write_sidecar(set_name, xpath, keys, CSFW_IDS, xnew, n)
    manifest_path = BANK / set_name / "_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("sidecars", {})[PARTB_NAME] = {
        "feature_set_id": feature_set_id, "formula_revision": "3", "root_form": "sqrt",
        "input_contract": "legacy-rgb8", "era": "ceiling_rev3", "build_commit": build_commit,
        "binary_sha256": bin_sha, "populated_feature_ids": PARTB_IDS,
        "build_meta_sha256": sha256_file(build_meta_path),
        "dtype": "f32", "cast": "f64->f32 round-nearest-even",
        "env": {"ZENSIM_FORMULA_REV": "3", "ZENSIM_ROOT_FORM": "sqrt",
                "RAYON_NUM_THREADS": "8"},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pairs_sha256": sha256_file(pairs_path), "audit_sha256": sha256_file(audit_path),
        "extractor_csv_sha256": sha256_file(csv_path), "row_count": n,
        "old_f32_checked_rows": n_nonident_checked, "old_f32_mismatches": bad_old,
        "old_f32_identical_rows_checked": n_ident_checked,
        "old_f32_identical_cell_mismatches": bad_ident,
        "old_f32_identical_note": "recorded not gated (user ruling 2026-09-25); bank stores 0 in old slots for identical pairs",
        "c1_live_cells_by_hat": c1_live, "c3_p99_saturated_cells_by_map": c3_sat,
        "identity_nonzero_c1c2c4": identity_bad,
    }
    manifest["files"][PARTB_NAME] = {"sha256": sha256_file(path),
                                      "bytes": path.stat().st_size}
    entry = dict(manifest["sidecars"][PARTB_NAME])
    for k in ("c1_live_cells_by_hat", "c3_p99_saturated_cells_by_map", "identity_nonzero_c1c2c4"):
        entry.pop(k)
    entry.update({"populated_feature_ids": CSFW_IDS, "live_cells_by_feature": x_live,
                  "identity_nonzero_cells": x_ident_nonzero,
                  "note": "csfw f944-955 + DVIFM C7 f956-985; no stored bank baseline, so no old-f32 parity"})
    entry.pop("old_f32_checked_rows"); entry.pop("old_f32_mismatches")
    manifest["sidecars"][CSFW_NAME] = entry
    manifest["files"][CSFW_NAME] = {"sha256": sha256_file(xpath), "bytes": xpath.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"PARTB_WRITTEN set={set_name} rows={n} sidecar_sha256={sha256_file(path)} "
          f"manifest_sha256={sha256_file(manifest_path)} feature_set_id={feature_set_id}")


def bind_gmsbank(set_name, csv_path, audit_path, binary, build_commit, build_meta_path):
    """C8 gmsbank pass (--full-gmsbank, width 1502): write f1322-f1501 and re-verify f0-f1321."""
    keys = partb_keys(set_name, None)
    n = len(keys["pair_key"])
    producer = json.loads(Path(str(csv_path) + ".manifest.json").read_text())
    bin_sha = sha256_file(binary)
    if producer.get("producer_binary_sha256") != bin_sha:
        raise SystemExit("extractor manifest binary hash differs from build")
    if producer.get("formula_revision") != "3" or producer.get("layout") != "w1502":
        raise SystemExit("extractor manifest has wrong formula/layout")
    if producer.get("input_contract", "legacy-rgb8") != "legacy-rgb8":
        raise SystemExit("extractor manifest has wrong input contract")
    feature_set_id = producer["feature_set_id"]
    if not feature_set_id:
        raise SystemExit("extractor did not emit a feature_set_id")
    build_meta = json.loads(build_meta_path.read_text())
    if build_meta["repositories"]["zensim"]["commit"] != build_commit:
        raise SystemExit("build commit differs from clean-source metadata")
    pairs_path = csv_path.parent.parent / "pairs" / f"{set_name}.tsv"

    with audit_path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                raise SystemExit(f"{set_name}: extra audit row")
            a = json.loads(line)
            if (audit_row_id(a) != i or
                a["reference"] != keys["ref_path"][i] or
                a["distorted"] != keys["dist_path"][i] or
                a["reference_pixels_sha256"] != keys["ref_pixels_sha256"][i] or
                a["distorted_pixels_sha256"] != keys["dist_pixels_sha256"][i] or
                pair_key(a["reference_pixels_sha256"], a["distorted_pixels_sha256"])
                    != keys["pair_key"][i]):
                raise SystemExit(f"{set_name}: audit/key binding failed at {i}")
        if i + 1 != n:
            raise SystemExit(f"{set_name}: audit has {i + 1} of {n} rows")

    d = BANK / set_name
    prev_ids = CSFW_IDS + PARTB_IDS  # f944..f1321, contiguous
    prev = np.empty((n, len(prev_ids)), dtype=np.float32)
    prev[:, :len(CSFW_IDS)] = pq.read_table(
        d / CSFW_NAME, columns=[f"f{i}" for i in CSFW_IDS]).to_pandas().to_numpy(np.float32)
    prev[:, len(CSFW_IDS):] = pq.read_table(
        d / PARTB_NAME, columns=[f"f{i}" for i in PARTB_IDS]).to_pandas().to_numpy(np.float32)
    rng = np.random.default_rng(20260924)
    count = max(1, math.ceil(n / 100))
    indices = sorted(rng.choice(n, count, replace=False).tolist())
    ident_set = set(np.nonzero(np.asarray(keys["pixels_identical"], dtype=np.bool_))[0].tolist())
    indices = sorted(set(indices) | ident_set)   # identical rows: checked, recorded, not gated (ruling 2026-09-25)
    n_ident_checked = len(ident_set)
    n_nonident_checked = len(indices) - n_ident_checked
    bad_ident = 0
    expected_old = partb_old_sample(set_name, indices)
    new = np.memmap(csv_path.parent / f"{set_name}.gms.f32", mode="w+",
                    dtype=np.float32, shape=(n, len(GMS_IDS)))
    seen = np.zeros(n, dtype=np.bool_)
    bad_old = bad_prev = 0
    with csv_path.open(newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        rid_i = hdr.index("row_id")
        if [int(x[1:]) for x in hdr if x.startswith("f") and x[1:].isdigit()] != list(range(1502)):
            raise SystemExit("extractor CSV does not contain f0-f1501")
        p0 = hdr.index("f944")
        g0 = hdr.index("f1322")
        old_pos = [hdr.index(f"f{i}") for i in POPULATED_IDS]
        for row in r:
            rid = int(row[rid_i])
            if rid < 0 or rid >= n or seen[rid]:
                raise SystemExit(f"{set_name}: invalid/duplicate CSV row_id {rid}")
            seen[rid] = True
            vals = np.fromiter((float(x) for x in row[g0:g0 + len(GMS_IDS)]),
                               dtype=np.float64, count=len(GMS_IDS))
            if not np.isfinite(vals).all():
                raise SystemExit(f"{set_name}: non-finite gmsbank value at {rid}")
            new[rid] = vals.astype(np.float32)
            pv = np.fromiter((float(x) for x in row[p0:p0 + len(prev_ids)]),
                             dtype=np.float64, count=len(prev_ids)).astype(np.float32)
            bad_prev += int(np.count_nonzero(pv.view(np.uint32) != prev[rid].view(np.uint32)))
            if rid in expected_old:
                got = np.fromiter((float(row[q]) for q in old_pos),
                                  dtype=np.float64, count=len(old_pos)).astype(np.float32)
                mm = int(np.count_nonzero(got.view(np.uint32) !=
                                          expected_old[rid].view(np.uint32)))
                if rid in ident_set:
                    bad_ident += mm
                else:
                    bad_old += mm
    if not seen.all() or bad_old or bad_prev:
        raise SystemExit(f"{set_name}: coverage={int(seen.sum())}/{n} "
                         f"old_f32_mismatches={bad_old} prev_f944_f1321_mismatches={bad_prev}")
    new.flush()
    live = [int(np.count_nonzero(new[:, j])) for j in range(len(GMS_IDS))]
    ident = np.asarray(keys["pixels_identical"], dtype=np.bool_)
    ident_nonzero = int(np.count_nonzero(new[ident]))
    print(f"PARTB_CHECK_GMS set={set_name} rows={n} finite=1 unique=1 "
          f"old_f0_f943_checked_rows={n_nonident_checked} old_mismatches={bad_old} "
          f"identical_rows_checked={n_ident_checked} identical_old_cell_mismatches={bad_ident} "
          f"prev_f944_f1321_checked_rows={n} prev_mismatches={bad_prev} "
          f"identity_nonzero={ident_nonzero} dead_columns={[GMS_IDS[j] for j, v in enumerate(live) if v == 0]}")
    path = d / GMS_NAME
    _write_sidecar(set_name, path, keys, GMS_IDS, new, n)
    manifest_path = d / "_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("sidecars", {})[GMS_NAME] = {
        "feature_set_id": feature_set_id, "formula_revision": "3", "root_form": "sqrt",
        "input_contract": "legacy-rgb8", "era": "ceiling_rev3", "build_commit": build_commit,
        "binary_sha256": bin_sha, "populated_feature_ids": GMS_IDS,
        "build_meta_sha256": sha256_file(build_meta_path),
        "dtype": "f32", "cast": "f64->f32 round-nearest-even", "flag": "--full-gmsbank",
        "env": {"ZENSIM_FORMULA_REV": "3", "ZENSIM_ROOT_FORM": "sqrt", "RAYON_NUM_THREADS": "8"},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pairs_sha256": sha256_file(pairs_path), "audit_sha256": sha256_file(audit_path),
        "extractor_csv_sha256": sha256_file(csv_path), "row_count": n,
        "old_f32_checked_rows": n_nonident_checked, "old_f32_mismatches": bad_old,
        "old_f32_identical_rows_checked": n_ident_checked,
        "old_f32_identical_cell_mismatches": bad_ident,
        "prev_sidecar_f944_f1321_rows_checked": n, "prev_sidecar_mismatches": bad_prev,
        "live_cells_by_feature": live, "identity_nonzero_cells": ident_nonzero,
    }
    manifest["files"][GMS_NAME] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"PARTB_WRITTEN_GMS set={set_name} rows={n} sidecar_sha256={sha256_file(path)} "
          f"manifest_sha256={sha256_file(manifest_path)} feature_set_id={feature_set_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--partb-pairs", type=Path, help="write label-free pairs from bank keys")
    ap.add_argument("--partb-csv", type=Path, help="bind a full-rev4 extractor CSV")
    ap.add_argument("--partb-audit", type=Path)
    ap.add_argument("--partb-binary", type=Path)
    ap.add_argument("--partb-commit")
    ap.add_argument("--gms-csv", type=Path, help="bind a --full-gmsbank extractor CSV (w1502)")
    ap.add_argument("--gms-build-meta", type=Path)
    ap.add_argument("--partb-limit", type=int, help="first N keys; validation only")
    a = ap.parse_args()
    if a.gms_csv:
        if not (a.partb_audit and a.partb_binary and a.partb_commit and a.gms_build_meta):
            raise SystemExit('--gms-csv needs --partb-audit/--partb-binary/--partb-commit/--gms-build-meta')
        bind_gmsbank(a.set, a.gms_csv, a.partb_audit, a.partb_binary, a.partb_commit, a.gms_build_meta)
        return
    if a.partb_pairs or a.partb_csv:
        if not a.set or a.all:
            ap.error("Part B requires exactly one --set")
        if a.partb_pairs:
            write_partb_pairs(a.set, a.partb_pairs, a.partb_limit)
        if a.partb_csv:
            if not (a.partb_audit and a.partb_binary and a.partb_commit):
                ap.error("Part B CSV requires audit, binary and commit")
            bind_partb(a.set, a.partb_csv, a.partb_audit,
                       a.partb_binary, a.partb_commit, a.partb_limit)
        return
    names = list(SETS) if a.all else [a.set]
    for name in names:
        cfg = SETS[name]
        if cfg["path"] == "parquet":
            convert_parquet(name, cfg)
        else:
            assemble_extract(name, cfg)


if __name__ == "__main__":
    main()
