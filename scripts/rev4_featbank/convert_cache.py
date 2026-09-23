#!/usr/bin/env python3
"""rev4 featbank-extract lane: build Rev3 f32 feature-bank sets.

Two source paths, one output layout (docs/REV4_FEATURE_BANK_PLAN_2026-09-23
section 2.3):

  * "parquet" — convert an existing Rev3 f64 944-feature cache
    (baseline-recovery cid22/safesyn, ceiling human parquets). Stored values
    are preserved exactly; only the f32 cast is applied.
  * "extract" — bind a fresh extract-native-admission run
    (feats.csv + audit.jsonl + pairs TSV, all joins keyed on row_id) into the
    same bank layout. Used where no Rev3 cache exists (the ext944 caches are
    a different feature era — see lane worklog).

Fits nothing; label columns are copied through as data only. Sets whose
role allows labels emit labels__<source>.parquet inside the set dir;
confirmation sets emit no labels file at all (pixels only, ruling D4).
Sources that replicate held-out human_score live under
/var/tmp/rev4-featbank/_sealed/ (review correction 4, option a).

pair_key = sha256(utf8(ref_pixels_sha256_hex) || utf8(dist_pixels_sha256_hex)
                  || utf8(input_contract)) -> lowercase hex.

Set registry, constants and row-meta helpers live in featbank_sets.py.

Usage:
  python3 convert_cache.py --set <name>   # one set
  python3 convert_cache.py --all          # every set in SETS order
"""
import argparse
import csv
import hashlib
import json
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
    """Return {row_id: f64 feature vector (944,)} from an extractor CSV.

    The extractor emits the CSV sorted by ref_basename, NOT in pairs order, so
    positional binding is forbidden. Extra TSV columns (row_id) are echoed into
    the CSV as extra target columns; row_id is required here.
    """
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
    """Shared emit core. stimuli: list of dicts (one per SOURCE stimulus row —
    pixel-identical stimuli share a pair_key but keep separate label rows).
    feat_matrix: (n_stimuli, 944) f64 aligned with stimuli.

    keys/features are deduplicated by pair_key (content-addressed): identical
    pixel pairs yield one key+feature row; labels keep every stimulus.
    """
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
    """Fresh-extraction path: pairs + audit.jsonl + feats.csv.

    Audit lines are emitted in pairs order AND echo pairs `row_id` in
    extra_targets; the feats CSV is sorted by ref_basename but also echoes
    `row_id`. All joins are keyed on row_id — never positional.
    """
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
        keep = set(cfg["select_row_ids"])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    names = list(SETS) if a.all else [a.set]
    for name in names:
        cfg = SETS[name]
        if cfg["path"] == "parquet":
            convert_parquet(name, cfg)
        else:
            assemble_extract(name, cfg)


if __name__ == "__main__":
    main()
