"""Admit only UPIQ-380 from metadata, then ingest its isolated HDR JOD labels.

The original mixed subjective/objective CSVs and all SDR images are forbidden.
Extraction uses upiq_pu_score's pinned label-free mode, the production Rev5
research HDR walk and zenexr native BT.709 absolute nits. No model is fitted.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from e21_cheap_recipe import columns

ROOT = Path("/mnt/v/datasets/upiq_extracted/upiq_dataset/images")
README = Path("/mnt/v/datasets/upiq/README.md")
LABEL = Path("/mnt/v/output/zenmetrics/upiq-pu/upiq_cid_jod.csv")
LABEL_SHA = "e0b23f539d46845f6cc4a65591474bafbf7eb4faf02da7175fb342caa5df80ac"
RULE = "int(SHA256(original reference EXR bytes),16)%5==0:development;else:fit"
TRANSFORM = "score=100+10*human_JOD;no-clipping;reference-JOD-zero-maps-to-100"
WIDTH = 1825


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metadata_rows(root):
    """No pixel or label payload opens: the dataset's own filename metadata."""
    rows = []
    for dataset, prefix, nc, nd, nl in [("narwaria", "n", 10, 2, 7), ("korshunov", "k", 20, 3, 4)]:
        scenes = sorted(p for p in (root / dataset).iterdir() if p.is_dir())
        if [p.name for p in scenes] != [f"{c:02}" for c in range(1, nc + 1)]:
            raise ValueError("UPIQ HDR source directories incomplete")
        for scene in scenes:
            c = int(scene.name)
            expected = {f"i{c:02}.exr"} | {f"i{c:02}_{d:02}_{l}.exr" for d in range(1, nd + 1) for l in range(1, nl + 1)}
            names = {p.name for p in scene.iterdir() if not p.name.startswith("._")}
            if names != expected:
                raise ValueError("UPIQ HDR filename inventory mismatch")
            for d in range(1, nd + 1):
                for l in range(1, nl + 1):
                    rows.append(dict(condition_id=f"{prefix}-i{c:02}-{prefix}-{d:02}-{l}", dataset=dataset,
                        content=c, distortion=d, level=l, reference_rel=f"{dataset}/{c:02}/i{c:02}.exr",
                        distorted_rel=f"{dataset}/{c:02}/i{c:02}_{d:02}_{l}.exr"))
    return rows


def admit_metadata(a):
    """Check the complete member/path declaration before payload hashing."""
    if (a.get("schema") != "upiq380-extraction-admission-v1" or a.get("role") != "train"
            or a.get("tier") != "T2" or a.get("authority") != "D3-2026-10-07"
            or a.get("formula_revision") != 5 or a.get("input_contract") != "upiq-exr-bt709-nits-v1"
            or a.get("requested_ids") != columns("by_v2fy") or a.get("split_rule") != RULE
            or a.get("target_transform") != TRANSFORM
            or a.get("label_file_approved") != str(LABEL) or a.get("label_sha256") != LABEL_SHA
            or not Path(a.get("image_root", "")).is_absolute()):
        raise ValueError("UPIQ TRAIN admission identity/role/pin mismatch")
    rows = a.get("rows", [])
    expected = metadata_rows(Path(a["image_root"]))
    names = list(expected[0])
    if [{k: r.get(k) for k in names} for r in rows] != expected:
        raise ValueError("UPIQ-380 ordered membership/path binding mismatch")
    return rows


def prepare(args):
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=False)
    root = Path(args.images)
    rows = metadata_rows(root)
    a = dict(schema="upiq380-extraction-admission-v1", role="train", tier="T2", authority="D3-2026-10-07",
        input_contract="upiq-exr-bt709-nits-v1", formula_revision=5, requested_ids=columns("by_v2fy"),
        image_root=str(root), rows=rows, split_rule=RULE, target_transform=TRANSFORM,
        label_file_approved=str(LABEL), label_sha256=LABEL_SHA, build_commit=args.build_commit,
        metadata_readme=str(README), metadata_readme_sha256=sha(README),
        legacy_label_subset_producer_commit=None, original_mixed_UPIQ_CSV_read=False)
    admit_metadata(a)
    # Persist the member set and label-independent rule before any pixel/label read.
    write(dest / "membership-before-payloads.json", a)
    hashes = {}
    for r in rows:
        for kind in ("reference", "distorted"):
            rel = r[f"{kind}_rel"]
            path = root / rel
            for component in [path, *path.parents]:
                if component == root:
                    break
                if component.is_symlink():
                    raise ValueError("symlink below admitted image root")
            if rel not in hashes:
                hashes[rel] = sha(path)
            r[f"{kind}_sha256"] = hashes[rel]
        r["split"] = "development" if int(r["reference_sha256"], 16) % 5 == 0 else "fit"
        r["pair_key"] = hashlib.sha256(("upiq380-original-byte-pair-v1\0" + r["condition_id"] + "\0"
            + r["reference_sha256"] + "\0" + r["distorted_sha256"]).encode()).hexdigest()
    write(dest / "extraction-admission.json", a)
    write(dest / "PRELABEL_SPLIT.json", {"rule": RULE, "authority": "D3-2026-10-07", "labels_opened": 0,
        "counts": {s: sum(r["split"] == s for r in rows) for s in ("fit", "development")},
        "reference_counts": {s: len({r["reference_sha256"] for r in rows if r["split"] == s}) for s in ("fit", "development")},
        "extraction_admission_sha256": sha(dest / "extraction-admission.json")})
    write(dest / "PRELABEL_READS.json", {"metadata_files": [str(README)], "payload_files": [str(root / p) for p in hashes],
        "label_files": [], "SDR_payload_files": []})


def targets(a):
    admit_metadata(a)
    if sha(LABEL) != a["label_sha256"]:
        raise ValueError("admitted HDR-only label source changed")
    with LABEL.open(newline="") as f:
        raw = list(csv.reader(f))
    allowed = {r["condition_id"] for r in a["rows"]}
    if (len(raw) != 380 or any(len(r) != 2 or r[0] not in allowed for r in raw)
            or {r[0] for r in raw} != allowed):
        raise ValueError("HDR-only label membership mismatch before target conversion")
    result = {r[0]: float(r[1]) for r in raw}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite human JOD")
    return result


def ingest(args):
    dest = Path(args.dest)
    admission = dest / "extraction-admission.json"
    a = json.loads(admission.read_text())
    rows = admit_metadata(a)
    extract = Path(args.features)
    producer = json.loads(Path(f"{extract}.manifest.json").read_text())
    if (producer.get("allowlist_sha256") != sha(admission) or producer.get("formula_revision") != 5
            or producer.get("input_contract") != a["input_contract"] or producer.get("rows") != 380
            or producer.get("requested_ids") != a["requested_ids"] or producer.get("build_commit") != args.build_commit):
        raise ValueError("extractor producer binding mismatch before labels")
    with extract.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        values = list(reader)
    if [r["condition_id"] for r in values] != [r["condition_id"] for r in rows]:
        raise ValueError("extraction row order mismatch before labels")
    matrix = np.array([[float(r[f"f{i}"]) for i in range(WIDTH)] for r in values], dtype=np.float64)
    ids = a["requested_ids"]
    if not np.isfinite(matrix[:, ids]).all() or not np.isnan(np.delete(matrix, ids, axis=1)).all():
        raise ValueError("requested feature coverage or absent slots mismatch")
    y = targets(a)
    files = {}
    for split in ("fit", "development"):
        indices = [i for i, r in enumerate(rows) if r["split"] == split]
        keys = [{**rows[i], "row_id": i, "role": "train", "source": "UPIQ-380", "authority": a["authority"]} for i in indices]
        path = dest / f"upiq380_{split}.parquet"
        key_path = path.with_suffix(".keys.parquet")
        man_path = Path(f"{path}.manifest.json")
        if any(p.exists() for p in [path, key_path, man_path]):
            raise ValueError("fresh leg output required")
        data = {"pair_key": [k["pair_key"] for k in keys], "condition_id": [k["condition_id"] for k in keys],
            "ref_basename": [k["reference_sha256"] for k in keys], "human_JOD": [y[k["condition_id"]] for k in keys],
            "human_score": [100 + 10 * y[k["condition_id"]] for k in keys]}
        data.update({f"f{i}": matrix[indices, i] for i in range(WIDTH)})
        pq.write_table(pa.table(data), path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(keys), key_path, compression="zstd")
        record = dict(schema="upiq380-v2-leg-v1", arm="uh4", source="UPIQ-380", split=split, role="train", tier="T2",
            allowed_use="HDR-recipe-training" if split == "fit" else "TRAIN-development-report-only-no-selection",
            authority=a["authority"], rows=len(keys), references=len({k["reference_sha256"] for k in keys}),
            member_set=[k["condition_id"] for k in keys], build_commit=args.build_commit, binary_sha256=sha(args.binary),
            formula_revision=5, requested_ids=ids, feature_set_id=None, feature_set_identity_scope="explicit-by_v2fy-420-native-HDR-slot-subset",
            feature_dtype="float64", absent_slots="NaN", input_contract=a["input_contract"], target_transform=TRANSFORM,
            split_rule=RULE, table_sha256=sha(path), keys_sha256=sha(key_path), admission_sha256=sha(admission),
            label_source={"path":str(LABEL),"sha256":LABEL_SHA,"producer_commit":None,"scope":"legacy HDR-only derivative; original mixed CSV unopened"},
            extractor_manifest_sha256=sha(f"{extract}.manifest.json"), qualified_provenance=False,
            missing=["legacy HDR-only label subset producer commit", "independent human HDR test"],
            research=producer["research"])
        write(man_path, record)
        files[split] = dict(table=str(path), table_sha256=record["table_sha256"], keys=str(key_path), keys_sha256=record["keys_sha256"],
            manifest=str(man_path), manifest_sha256=sha(man_path), rows=len(keys), references=record["references"])
    write(dest / "INGEST_RECEIPT.json", {"schema":"upiq380-ingest-receipt-v1", "status":"PASS", "legs":files,
        "rows":380,"references":30,"labels_read":[str(LABEL)],"original_mixed_UPIQ_CSV_read":False,
        "member_set_sha256":sha(admission),"build_commit":args.build_commit})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "ingest"])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--images", default=str(ROOT))
    ap.add_argument("--build-commit", required=True)
    ap.add_argument("--features")
    ap.add_argument("--binary")
    args = ap.parse_args()
    (prepare if args.command == "prepare" else ingest)(args)


if __name__ == "__main__":
    main()
