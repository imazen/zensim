#!/usr/bin/env python3
"""HDRVID Rev5 admissions and pooled tables (label-free).

  extract.py admit  --out OUT --set hdrvdc|avt
      DECODE_PLAN.json + receipts -> OUT/extraction/<set>-admission.json
      (the `hdrvid_extract` contract: every frame's path and SHA-256).
  extract.py tables --out OUT --set hdrvdc|avt --features TSV --binary BIN --build-commit SHA
      per-frame features -> OUT/tables/<set>.parquet (one row per video and
      display configuration, each feature the MEAN over the eight frames:
      the stored-table aggregation the external-read owner rescored bakes on),
      <set>.keys.parquet (label-free identities) and <set>.parquet.manifest.json.

No label file is read here. Labels are bound only by the exposure-frozen
report (`scripts/rev4_featpot/v40_panels.py --mode e31video`).
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rev4_featpot"))
from v2_teacher import row_keys_sha  # noqa: E402

WIDTH = 1825
CONFIGS = {"hdrvdc": ["A", "B", "C", "D", "E"], "avt": ["A"]}
VIDEOS = {"hdrvdc": 116, "avt": 195}
CONTRACT = "hdrvid-pq-png16-bt2020-display-v1"
CANDIDATES = Path(__file__).resolve().parents[3] / "benchmarks/costset2_2026-10-03.candidate_ids.json"
CANDIDATES_SHA256 = "0a6a20dc356acef3bef9deffc411f03189813e8b924fddcf7b22f7efea6b9f17"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def requested_ids():
    if sha(CANDIDATES) != CANDIDATES_SHA256:
        raise ValueError("candidate ID list changed")
    return json.loads(CANDIDATES.read_text())["candidates"]["by_v2fy"]


def write_new(path, text):
    path = Path(path)
    if path.exists():
        raise ValueError(f"fresh output required: {path}")
    path.write_text(text)


def video_rows(out, name):
    plan = json.loads((Path(out) / "DECODE_PLAN.json").read_text())
    rows = [r for r in plan["rows"] if r["set"] == name]
    refs = {r["content"]: r for r in rows if r["role"] == "reference"}
    tests = [r for r in rows if r["role"] == "test" and r["admitted"]]
    if len(tests) != VIDEOS[name]:
        raise ValueError(f"{name}: {len(tests)} admitted videos, registered {VIDEOS[name]}")
    return plan, refs, tests


def receipt(out, name, stem):
    return json.loads((Path(out) / "receipts" / name / f"{stem}.json").read_text())


def key_of(name, row, config):
    if name == "hdrvdc":
        return f"hdrvdc/{row['content']}/{row['crf']}_{row['coded']}"
    return f"avt/{row['content']}/{row['codec']}_{row['coded']}_{row['bitrate']}"


def admit(args):
    out = Path(args.out)
    _, refs, tests = video_rows(out, args.set)
    rel = lambda p: str(Path(p).relative_to(out))  # noqa: E731
    rows = []
    for row in tests:
        ref, dist = receipt(out, args.set, refs[row["content"]]["stem"]), receipt(out, args.set, row["stem"])
        for config in CONFIGS[args.set]:
            far = config in ("D", "E")
            pick = (lambda f: f["far"]) if far else (lambda f: f)
            frames = [
                dict(j=j,
                     reference=dict(rel=rel(pick(rf)["png"]), sha256=pick(rf)["png_sha256"]),
                     distorted=dict(rel=rel(pick(df)["png"]), sha256=pick(df)["png_sha256"]))
                for j, (rf, df) in enumerate(zip(ref["frames"], dist["frames"]))
            ]
            if ref["indices"] != dist["indices"]:
                raise ValueError(f"{row['stem']}: reference/test frame indices differ")
            rows.append(dict(key=key_of(args.set, row, config), content=row["content"], config=config, frames=frames))
    admission = dict(schema="hdrvid-extraction-admission-v1", set=args.set, role="external-eval",
                     input_contract=CONTRACT, formula_revision=5, frame_root=str(out.resolve()),
                     requested_ids=requested_ids(), rows=rows)
    (out / "extraction").mkdir(exist_ok=True)
    path = out / "extraction" / f"{args.set}-admission.json"
    write_new(path, json.dumps(admission, indent=1) + "\n")
    print(f"{path} {sha(path)} rows={len(rows)}")


def tables(args):
    out = Path(args.out)
    plan, refs, tests = video_rows(out, args.set)
    admission_path = out / "extraction" / f"{args.set}-admission.json"
    admission = json.loads(admission_path.read_text())
    features = Path(args.features)
    producer = json.loads(Path(f"{features}.manifest.json").read_text())
    ids = admission["requested_ids"]
    if (producer.get("admission_sha256") != sha(admission_path) or producer.get("formula_revision") != 5
            or producer.get("set") != args.set or producer.get("requested_ids") != ids
            or producer.get("rows") != len(admission["rows"]) or producer.get("build_commit") != args.build_commit
            or producer.get("labels_read") is not False):
        raise ValueError("extractor producer binding mismatch")
    with features.open(newline="") as f:
        values = list(csv.DictReader(f, delimiter="\t"))
    expected = [(r["key"], r["config"], str(j)) for r in admission["rows"] for j in range(8)]
    if [(v["key"], v["config"], v["j"]) for v in values] != expected:
        raise ValueError("extraction row order mismatch")
    frames = np.array([[float(v[f"f{i}"]) for i in range(WIDTH)] for v in values], dtype=np.float64)
    if not np.isfinite(frames[:, ids]).all() or not np.isnan(np.delete(frames, ids, axis=1)).all():
        raise ValueError("requested feature coverage or absent slots mismatch")
    pooled = frames.reshape(len(admission["rows"]), 8, WIDTH).mean(axis=1)
    by_key = {key_of(args.set, r, None): r for r in tests}
    keys = []
    for r in admission["rows"]:
        video = by_key[r["key"]]
        k = dict(pair_key=f"{r['key']}/{r['config']}", set=args.set, video=r["key"], content=r["content"],
                 config=r["config"], reference_sha256=refs[r["content"]]["sha256"], distorted_sha256=video["sha256"],
                 coded=video["coded"], frames=video["frames"], role="external-eval")
        if args.set == "hdrvdc":
            k.update(crf=video["crf"])
        else:
            k.update(codec=video["codec"], bitrate=video["bitrate"])
        keys.append(k)
    dest = out / "tables"
    dest.mkdir(exist_ok=True)
    path, key_path = dest / f"{args.set}.parquet", dest / f"{args.set}.keys.parquet"
    man_path = Path(f"{path}.manifest.json")
    if any(p.exists() for p in (path, key_path, man_path)):
        raise ValueError("fresh table output required")
    key_table = pa.Table.from_pylist(keys)
    data = {"pair_key": [k["pair_key"] for k in keys]}
    data.update({f"f{i}": pooled[:, i] for i in range(WIDTH)})
    pq.write_table(pa.table(data), path, compression="zstd")
    pq.write_table(key_table, key_path, compression="zstd")
    record = dict(
        schema="hdrvid-external-table-v1", set=args.set, data_role="assessment-only", role="external-eval",
        rows=len(keys), videos=VIDEOS[args.set], configs=CONFIGS[args.set], formula_revision="Rev5",
        requested_ids=ids, feature_set_id=None, feature_set_identity_scope="explicit-by_v2fy-420-native-HDR-slot-subset",
        feature_dtype="float64", absent_slots="NaN", aggregation="mean over the eight uniform frames per video, per feature",
        input_contract=CONTRACT,
        decoder_era="hdrvid-2026-10-10: rav1d-safe f3132ee6 (all AV1); ffmpeg 8.1.3 native hevc/vvc/ffvhuff decode only; "
                    "zenavif 85dd0d2b yuv_convert; zenresize e3975fb9 Lanczos-3; zenpng 0.1.4 16-bit PNG",
        table_sha256=sha(path), keys_sha256=sha(key_path), row_keys_sha256=row_keys_sha(key_table),
        admission_sha256=sha(admission_path), extractor_manifest_sha256=sha(f"{features}.manifest.json"),
        features_tsv_sha256=sha(features), build_commit=args.build_commit, binary_sha256=sha(args.binary),
        decode_plan_sha256=sha(out / "DECODE_PLAN.json"), decode_summary_sha256=sha(out / "DECODE_SUMMARY.json"),
        dropped_videos=[dict(stem=r["stem"], reason=r["reason"]) for r in plan["rows"]
                        if r["set"] == args.set and not r["admitted"]],
        labels_read=False, research=producer["research"],
    )
    write_new(man_path, json.dumps(record, indent=1) + "\n")
    print(f"{path} rows={len(keys)} table={record['table_sha256']} keys={record['keys_sha256']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("admit", "tables"))
    p.add_argument("--out", required=True)
    p.add_argument("--set", required=True, choices=("hdrvdc", "avt"))
    p.add_argument("--features")
    p.add_argument("--binary")
    p.add_argument("--build-commit")
    args = p.parse_args()
    {"admit": admit, "tables": tables}[args.command](args)


if __name__ == "__main__":
    main()
