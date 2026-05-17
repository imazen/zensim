#!/usr/bin/env python3
"""Export CID22, KADID-10k, and TID2013 human-rated datasets to parquet.

These are the three independent human-rated quality assessment corpora.
Schemas across them differ; we normalize to the comparison-site
convention (corpus / ref_path / dist_path / image_name / codec /
version / human_mos / human_dmos / score_* / bpp).

Source paths (per `~/.claude/CLAUDE.md`):
  CID22:    /mnt/v/dataset/cid22/CID22_validation_set.csv
            + images under  /mnt/v/dataset/cid22/CID22_validation_set/
  KADID:    /mnt/v/dataset/kadid10k/dmos.csv
            + images under  /mnt/v/dataset/kadid10k/images/
  TID2013:  /mnt/v/dataset/tid2013/mos_with_names.txt
            + reference_images/ + distorted_images/

Optional --metrics-tsv <PATH> can be passed once per per-corpus zen-metrics
batch output to merge in dssim/ssim2/butter/zensim. Same rename map as
the AIC export scripts (dssim_gpu → score_dssim etc.).

Usage:
    python3 export_human_corpora_to_parquet.py --which cid22 \\
        --out /tmp/cid22_export.parquet
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CID22_DIR  = "/mnt/v/dataset/cid22/CID22_validation_set"
CID22_CSV  = "/mnt/v/dataset/cid22/CID22_validation_set.csv"
KADID_DIR  = "/mnt/v/dataset/kadid10k/images"
KADID_CSV  = "/mnt/v/dataset/kadid10k/dmos.csv"
TID_REF    = "/mnt/v/dataset/tid2013/reference_images"
TID_DIST   = "/mnt/v/dataset/tid2013/distorted_images"
TID_TXT    = "/mnt/v/dataset/tid2013/mos_with_names.txt"

RENAME_ZEN_METRICS = {
    "dssim_gpu":             "score_dssim",
    "dssim":                 "score_dssim_cpu",
    "ssim2_gpu":             "score_ssim2_gpu",
    "ssim2":                 "score_ssim2",
    "butteraugli_max_gpu":   "score_butter_max",
    "butteraugli_pnorm3_gpu":"score_butter_p3",
    "butteraugli_max":       "score_butter_max_cpu",
    "butteraugli_pnorm3":    "score_butter_p3_cpu",
    "zensim":                "score_zensim",
}


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_cid22() -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(CID22_CSV):
        print(f"CID22 CSV not found: {CID22_CSV}", file=sys.stderr)
        return rows
    with open(CID22_CSV) as f:
        for r in csv.DictReader(f):
            ref_rel = r["reference_img"]   # e.g. "original/162520.png"
            dist_rel = r["distorted_img"]
            ref_path = f"{CID22_DIR}/{ref_rel}"
            dist_path = f"{CID22_DIR}/{dist_rel}"
            # CID22 reference rows include the reference itself as distorted_img.
            # Keep them — useful as a 100-score anchor.
            stem = Path(ref_rel).stem
            rows.append({
                "corpus": "cid22",
                "ref_path": ref_path,
                "dist_path": dist_path,
                "image_name": stem,
                "codec": r["encoder"],
                "version": r["setting"] or "",
                "bpp": _float(r.get("bpp")),
                "human_mos":  _float(r.get("MCOS")),
                "human_dmos": _float(r.get("RMOS")),
                "human_elo":  _float(r.get("Elo")),
                "nb_pc_opinions": int(r["nb_pc_opinions"]) if r.get("nb_pc_opinions") else None,
            })
    return rows


def load_kadid() -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(KADID_CSV):
        print(f"KADID CSV not found: {KADID_CSV}", file=sys.stderr)
        return rows
    with open(KADID_CSV) as f:
        for r in csv.DictReader(f):
            ref_rel = r["ref_img"]      # e.g. "I01.png"
            dist_rel = r["dist_img"]    # e.g. "I01_01_01.png"
            ref_path = f"{KADID_DIR}/{ref_rel}"
            dist_path = f"{KADID_DIR}/{dist_rel}"
            # Filename encodes distortion: <refid>_<distortion>_<level>.png
            parts = Path(dist_rel).stem.split("_")
            distortion_id = parts[1] if len(parts) >= 3 else ""
            level = parts[2] if len(parts) >= 3 else ""
            rows.append({
                "corpus": "kadid10k",
                "ref_path": ref_path,
                "dist_path": dist_path,
                "image_name": Path(ref_rel).stem,
                "codec": distortion_id,    # 01..25 distortion type code
                "version": level,           # 01..05 distortion level
                "human_dmos": _float(r.get("dmos")),
                "human_dmos_var": _float(r.get("var")),
            })
    return rows


def load_tid() -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(TID_TXT):
        print(f"TID2013 txt not found: {TID_TXT}", file=sys.stderr)
        return rows
    with open(TID_TXT) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2: continue
            mos = _float(parts[0])
            dist_name = parts[1]
            # dist filenames like "I01_01_1.bmp" or "i01_01_2.bmp" (case varies)
            # Reference like "I01.BMP" — uppercase ext on disk.
            stem_parts = Path(dist_name).stem.split("_")
            ref_id = stem_parts[0].upper()  # "I01"
            distortion_id = stem_parts[1] if len(stem_parts) >= 3 else ""
            level = stem_parts[2] if len(stem_parts) >= 3 else ""
            ref_path = f"{TID_REF}/{ref_id}.BMP"
            dist_path = f"{TID_DIST}/{dist_name}"
            rows.append({
                "corpus": "tid2013",
                "ref_path": ref_path,
                "dist_path": dist_path,
                "image_name": ref_id,
                "codec": distortion_id,
                "version": level,
                "human_mos": mos,
            })
    return rows


def merge_metric_tsv(rows: list[dict], tsv_path: str, key_cols: tuple[str, ...]) -> None:
    if not tsv_path or not os.path.exists(tsv_path):
        print(f"  skip metrics merge: {tsv_path} not found", file=sys.stderr)
        return
    by_key: dict[tuple, dict] = {}
    with open(tsv_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                k = tuple(r[c] for c in key_cols)
            except KeyError:
                continue
            by_key[k] = r
    if not by_key:
        return
    sample = next(iter(by_key.values()))
    known = set(key_cols) | {"ref_path", "dist_path"}
    metric_cols = [c for c in sample if c not in known and not c.startswith("human_")]
    print(f"  {tsv_path}: {len(by_key)} rows, metric cols: {metric_cols} → "
          f"{[RENAME_ZEN_METRICS.get(c, c) for c in metric_cols]}", file=sys.stderr)
    for row in rows:
        try:
            k = tuple(str(row[c]) for c in key_cols)
        except KeyError:
            continue
        m = by_key.get(k)
        if not m: continue
        for c in metric_cols:
            v = _float(m.get(c))
            if v is not None:
                row[RENAME_ZEN_METRICS.get(c, c)] = v


def write_parquet(rows: list[dict], out_path: str) -> None:
    if not rows:
        print("no rows to write", file=sys.stderr)
        sys.exit(1)
    col_order: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen: col_order.append(k); seen.add(k)
    cols = {c: [r.get(c) for r in rows] for c in col_order}
    fields = []
    for c in col_order:
        if c == "nb_pc_opinions":
            t = pa.uint32()
        elif c.startswith("score_") or c.startswith("human_") or c == "bpp":
            t = pa.float32()
        else:
            t = pa.string()
        fields.append(pa.field(c, t))
    table = pa.Table.from_pydict(cols, schema=pa.schema(fields))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd", compression_level=9)
    print(f"wrote {len(rows)} rows × {len(col_order)} cols to {out_path} "
          f"({os.path.getsize(out_path):,} bytes)", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["cid22", "kadid", "tid", "all"], default="all")
    ap.add_argument("--metrics-tsv", action="append", default=[])
    ap.add_argument("--out", required=True, help="output parquet path (used as a prefix when --which=all)")
    args = ap.parse_args()

    targets = {
        "cid22": (load_cid22, ("codec", "version", "image_name")),
        "kadid": (load_kadid, ("codec", "version", "image_name")),
        "tid":   (load_tid,   ("codec", "version", "image_name")),
    }
    if args.which == "all":
        for which, (loader, keys) in targets.items():
            rows = loader()
            for tsv in args.metrics_tsv:
                merge_metric_tsv(rows, tsv, keys)
            out = Path(args.out)
            if out.is_dir() or args.out.endswith("/"):
                outp = out / f"{which}.parquet"
            else:
                outp = out.with_name(f"{out.stem}_{which}{out.suffix}")
            write_parquet(rows, str(outp))
    else:
        loader, keys = targets[args.which]
        rows = loader()
        for tsv in args.metrics_tsv:
            merge_metric_tsv(rows, tsv, keys)
        write_parquet(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
