#!/usr/bin/env python3
"""Export AIC-3 CTC EPFL dataset to a single parquet for the comparison site.

Source CSV: /mnt/v/dataset/aic3_ctc_epfl/decoded/info_with_bitrates.csv
  600 rows = 10 references × 6 codecs (AVIF, HM, JPEG-1, JPEG-2000, JPEGXL, VVC)
  × 10 quality levels.

Schema (column order matches the unified parquet convention):
  corpus           : utf8  ("aic3_ctc_epfl")
  ref_path         : utf8  (absolute path to reference PNG)
  dist_path        : utf8  (absolute path to encoded variant PNG)
  image_name       : utf8  ("00001_1192x832")
  codec            : utf8  ("AVIF" / "HM" / "JPEG-1" / "JPEG-2000" / "JPEGXL" / "VVC")
  q                : uint32 (encoder parameter, the "quality.selected" column)
  quality_index    : uint32 (1-10, human-rated quality level)
  bpp              : float32 (bits per pixel)
  human_jnd        : float32 (score.jnd, the human-rated JND; 0 ≈ JND threshold)
  score_dssim      : float32 (optional, merged from zen-metrics batch output)
  score_ssim2      : float32 (optional, merged from zen-metrics batch output)
  score_butter_max : float32 (optional, merged from zen-metrics batch output)
  score_butter_p3  : float32 (optional, merged from zen-metrics batch output)
  score_zensim     : float32 (optional, merged from zen-metrics batch output)

Optional --metrics-tsv <PATH> ... can be passed multiple times; each TSV is
expected to be a `zen-metrics batch` output with the pass-through columns
intact plus a metric column. Rows are joined on (codec, image_name, quality).

Usage:
    python3 export_aic3_to_parquet.py \
        --csv /mnt/v/dataset/aic3_ctc_epfl/decoded/info_with_bitrates.csv \
        --metrics-tsv /tmp/aic3_dssim/scored_dssim.tsv \
        --metrics-tsv /tmp/aic3_dssim/scored_ssim2.tsv \
        --out /tmp/aic3_dssim/aic3_ctc_epfl.parquet
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REF_DIR = "/mnt/v/dataset/aic3_ctc_epfl/original"
DEC_DIR = "/mnt/v/dataset/aic3_ctc_epfl/decoded"


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            codec = r.get("codec", "").strip()
            if not codec:
                continue
            img_name = r["img.name"]
            quality = int(r["quality"])
            ref_path = f"{REF_DIR}/{img_name}.png"
            dist_path = f"{DEC_DIR}/{img_name}/{codec}_{img_name}_{quality}.png"
            if not (os.path.exists(ref_path) and os.path.exists(dist_path)):
                continue
            rows.append({
                "corpus": "aic3_ctc_epfl",
                "ref_path": ref_path,
                "dist_path": dist_path,
                "image_name": img_name,
                "codec": codec,
                "q": int(r["quality.selected"]),
                "quality_index": quality,
                "bpp": float(r["bpp"]),
                "human_jnd": float(r["score.jnd"]),
            })
    return rows


def merge_metric_tsv(rows: list[dict], tsv_path: str) -> None:
    """Join a zen-metrics batch output TSV into rows in place.

    The TSV must have the original pass-through cols (codec, img_name, quality)
    and one or more metric columns. Match key: (codec, img_name, quality).
    """
    if not tsv_path or not os.path.exists(tsv_path):
        print(f"  skip metrics merge: {tsv_path} not found", file=sys.stderr)
        return
    by_key: dict[tuple, dict] = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                k = (r["codec"], r["img_name"], int(r["quality"]))
            except (KeyError, ValueError):
                continue
            by_key[k] = r
    # Rename zen-metrics column names → canonical site convention (score_*).
    rename = {
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
    metric_cols = []
    if by_key:
        sample = next(iter(by_key.values()))
        known = {"ref_path", "dist_path", "codec", "img_name", "quality",
                 "quality_selected", "bpp", "score_jnd"}
        metric_cols = [c for c in sample.keys() if c not in known]
        print(f"  {tsv_path}: {len(by_key)} rows, metric cols: {metric_cols} → "
              f"{[rename.get(c, c) for c in metric_cols]}",
              file=sys.stderr)
    for row in rows:
        k = (row["codec"], row["image_name"], row["quality_index"])
        m = by_key.get(k)
        if not m:
            continue
        for c in metric_cols:
            try:
                row[rename.get(c, c)] = float(m[c])
            except (KeyError, ValueError):
                continue


def write_parquet(rows: list[dict], out_path: str) -> None:
    if not rows:
        print("no rows; nothing to write", file=sys.stderr)
        sys.exit(1)
    # Collect every column name across rows (preserve first-seen order;
    # missing values → None per pa.Table.from_pydict semantics).
    col_order: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                col_order.append(k); seen.add(k)
    cols: dict[str, list] = {c: [r.get(c) for r in rows] for c in col_order}
    fields: list[pa.Field] = []
    for k in col_order:
        if k in ("q", "quality_index"):
            t = pa.uint32()
        elif k in ("bpp", "human_jnd") or k.startswith("score_") or k.startswith("human_"):
            t = pa.float32()
        else:
            t = pa.string()
        fields.append(pa.field(k, t))
    schema = pa.schema(fields)
    table = pa.Table.from_pydict(cols, schema=schema)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd", compression_level=9)
    print(f"wrote {len(rows)} rows × {len(cols)} cols to {out_path} "
          f"({os.path.getsize(out_path):,} bytes)", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=f"{DEC_DIR}/info_with_bitrates.csv")
    ap.add_argument("--metrics-tsv", action="append", default=[],
                    help="zen-metrics batch output TSV(s); pass once per metric")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load_csv(args.csv)
    print(f"loaded {len(rows)} rows from {args.csv}", file=sys.stderr)
    for tsv in args.metrics_tsv:
        merge_metric_tsv(rows, tsv)
    write_parquet(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
