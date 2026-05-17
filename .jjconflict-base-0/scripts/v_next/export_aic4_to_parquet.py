#!/usr/bin/env python3
"""Export AIC-4 sample dataset to a single parquet for the comparison site.

Sources (per `~/.claude/CLAUDE.md` and the AIC-4 README):
  - Image dir: /mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/
      PTC_images/<NNNNN>/PTC_<NNNNN>_<CODEC>_<DLEVEL>.png  (61 each: 1 ref + 60 dist)
      full_resolution_images/<NNNNN>/...
  - Reconstructed-JND scores:
      /mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG_AIC_reconstructed_jnd_scores.csv
      columns: img_num,codec,dlevel,img_source,img_distorted,distortion,CI_min,CI_max
  - Pre-computed metric scores (PSNR-Y, SSIM, MS-SSIM, IW-SSIM, VMAF-neg, SSIMULACRA2, HDR-VDP, CVVDP):
      /mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG-AIC_metric_scores.csv

Codec ID → name (from filename inspection):
  1 = AVIF
  2 = JPEG-1
  3 = JPEG-2000
  4 = JPEG-XL
  5 = VVC
  6 = JPEG-AI

300 rows total = 5 refs × 6 codecs × 10 distortion levels.

Schema (compatible with the comparison-site convention):
  corpus           : utf8  ("aic4_sample")
  ref_path         : utf8
  dist_path        : utf8
  image_name       : utf8  ("00002" .. "00010")
  codec            : utf8  ("AVIF" / "JPEG-1" / "JPEG-2000" / "JPEG-XL" / "VVC" / "JPEG-AI")
  dlevel           : uint32 (1..10, distortion level)
  human_jnd        : float32 (distortion column from reconstructed JND CSV; mean reconstructed JND)
  human_jnd_ci_lo  : float32 (CI_min)
  human_jnd_ci_hi  : float32 (CI_max)
  score_psnr_y     : float32 (pre-computed PSNR-Y)
  score_ssim       : float32
  score_ms_ssim    : float32
  score_iw_ssim    : float32
  score_vmaf_neg   : float32
  score_ssim2_paper: float32 (the "SSIMULACRA2" col from the paper CSV)
  score_hdr_vdp_2  : float32
  score_hdr_vdp_3  : float32
  score_cvvdp      : float32
  score_dssim      : float32 (optional, merged from zen-metrics batch --metric dssim-gpu)
  score_ssim2_gpu  : float32 (optional, merged from zen-metrics batch --metric ssim2-gpu)
  score_butter_max : float32 (optional, merged from zen-metrics)
  score_butter_p3  : float32 (optional)
  score_zensim     : float32 (optional, merged from zen-metrics batch --metric zensim)

The pre-computed paper metrics are merged via `img_distorted` filename
(unique across the 300 rows). The zen-metrics outputs merge via
(codec, image_name, dlevel) — same convention as AIC-3.

Usage:
    python3 export_aic4_to_parquet.py \
        --metrics-tsv /tmp/aic4_metrics/scored_dssim.tsv \
        --metrics-tsv /tmp/aic4_metrics/scored_zensim.tsv \
        --out /tmp/aic4_metrics/aic4_sample.parquet
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATASET_DIR = "/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset"
JND_CSV     = "/mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG_AIC_reconstructed_jnd_scores.csv"
METRIC_CSV  = "/mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG-AIC_metric_scores.csv"

CODEC_NAME = {
    1: "AVIF", 2: "JPEG-1", 3: "JPEG-2000",
    4: "JPEG-XL", 5: "VVC", 6: "JPEG-AI",
}


def _float(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_metric_csv(path: str) -> dict[str, dict]:
    """Map img_distorted → metric row (the pre-computed scores from the paper CSV)."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["img_distorted"]] = r
    return out


def load_jnd_csv(path: str, metric_lookup: dict[str, dict]) -> list[dict]:
    """Walk reconstructed JND CSV; each row → one corpus row, joined with metric CSV by img_distorted."""
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            img_num = r["img_num"].zfill(5)  # "2" → "00002"
            codec_id = int(r["codec"])
            codec = CODEC_NAME[codec_id]
            dlevel = int(r["dlevel"])
            dist_name = r["img_distorted"]
            src_name  = r["img_source"]
            # PTC dir layout: PTC_images/<img_num>/<dist_name>
            ref_path  = f"{DATASET_DIR}/PTC_images/{img_num}/{src_name}"
            dist_path = f"{DATASET_DIR}/PTC_images/{img_num}/{dist_name}"
            if not (os.path.exists(ref_path) and os.path.exists(dist_path)):
                continue
            row: dict = {
                "corpus": "aic4_sample",
                "ref_path": ref_path,
                "dist_path": dist_path,
                "image_name": img_num,
                "codec": codec,
                "dlevel": dlevel,
                "human_jnd":        _float(r["distortion"]),
                "human_jnd_ci_lo":  _float(r["CI_min"]),
                "human_jnd_ci_hi":  _float(r["CI_max"]),
            }
            m = metric_lookup.get(dist_name)
            if m:
                row["score_psnr_y"]      = _float(m.get("PSNR-Y"))
                row["score_ssim"]        = _float(m.get("SSIM"))
                row["score_ms_ssim"]     = _float(m.get("MS-SSIM"))
                row["score_iw_ssim"]     = _float(m.get("IW-SSIM"))
                row["score_vmaf_neg"]    = _float(m.get("VMAF-neg"))
                row["score_ssim2_paper"] = _float(m.get("SSIMULACRA2"))
                row["score_hdr_vdp_2"]   = _float(m.get("HDR-VDP-2 Q"))
                row["score_hdr_vdp_3"]   = _float(m.get("HDR-VDP-3 Q"))
                row["score_cvvdp"]       = _float(m.get("CVVDP"))
            rows.append(row)
    return rows


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


def merge_metric_tsv(rows: list[dict], tsv_path: str) -> None:
    if not tsv_path or not os.path.exists(tsv_path):
        print(f"  skip metrics merge: {tsv_path} not found", file=sys.stderr)
        return
    by_key: dict[tuple, dict] = {}
    with open(tsv_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                k = (r["codec"], r["image_name"], int(r["dlevel"]))
            except (KeyError, ValueError):
                continue
            by_key[k] = r
    if not by_key:
        return
    known = {"ref_path", "dist_path", "codec", "image_name", "dlevel"}
    sample = next(iter(by_key.values()))
    metric_cols = [c for c in sample if c not in known and not c.startswith("human_")]
    print(f"  {tsv_path}: {len(by_key)} rows, metric cols: {metric_cols} → "
          f"{[RENAME_ZEN_METRICS.get(c, c) for c in metric_cols]}",
          file=sys.stderr)
    for row in rows:
        k = (row["codec"], row["image_name"], row["dlevel"])
        m = by_key.get(k)
        if not m:
            continue
        for c in metric_cols:
            v = _float(m.get(c))
            if v is not None:
                row[RENAME_ZEN_METRICS.get(c, c)] = v


def write_parquet(rows: list[dict], out_path: str) -> None:
    if not rows:
        print("no rows; nothing to write", file=sys.stderr)
        sys.exit(1)
    # Collect every column name across rows; missing values → None.
    cols: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                cols.append(k); seen.add(k)
    arrays: dict[str, list] = {c: [r.get(c) for r in rows] for c in cols}
    fields: list[pa.Field] = []
    for c in cols:
        if c == "dlevel":
            t = pa.uint32()
        elif c.startswith("score_") or c.startswith("human_"):
            t = pa.float32()
        else:
            t = pa.string()
        fields.append(pa.field(c, t))
    table = pa.Table.from_pydict(arrays, schema=pa.schema(fields))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd", compression_level=9)
    print(f"wrote {len(rows)} rows × {len(cols)} cols to {out_path} "
          f"({os.path.getsize(out_path):,} bytes)", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jnd-csv",    default=JND_CSV)
    ap.add_argument("--metric-csv", default=METRIC_CSV)
    ap.add_argument("--metrics-tsv", action="append", default=[],
                    help="zen-metrics batch output TSV(s); pass once per metric")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    metric_lookup = load_metric_csv(args.metric_csv)
    rows = load_jnd_csv(args.jnd_csv, metric_lookup)
    print(f"loaded {len(rows)} rows from {args.jnd_csv}", file=sys.stderr)
    for tsv in args.metrics_tsv:
        merge_metric_tsv(rows, tsv)
    write_parquet(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
