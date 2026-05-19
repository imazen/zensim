#!/usr/bin/env python3
"""EXP-METRIC-INPUTS-FIX 375-col variant: keep all 372 zenanalyze features, append metric inputs.

Same data sources as build_augmented_parquets_fixed.py, but instead of taking
the first 300 zenanalyze features and replacing f300..f302 with metric inputs,
this variant keeps the full f0..f371 zenanalyze feature block and appends the
3 metric inputs as f372 (ssim2_gpu), f373 (cvvdp_score), f374 (iwssim).

This shape is comparable with the ship bakes (which use f0..f371) — bake_compare
can score both bakes on the same parquet, picking n_inputs.min(row.len()) features.

Output:
  /mnt/v/zen/zensim-training/2026-05-18-metric-inputs-fixed/train_375/{safesyn,kadid,tid,cvvdp_iwssim_LARGE}.parquet
  /mnt/v/zen/zensim-training/2026-05-18-metric-inputs-fixed/val_375/{cid22,kadid,tid,konjnd,aic3}.parquet
"""
import csv
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

CANONICAL = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18")
OUT_ROOT = Path("/mnt/v/zen/zensim-training/2026-05-18-metric-inputs-fixed")
SSIM2_GPU = Path("/tmp/exp_metric_fix")
SCORES_CVVDP = Path("/mnt/v/zen/zensim-eval")
SCORES_IWSSIM = Path("/mnt/v/zen/zensim-training/2026-05-18-iwssim-panels")
LOCAL_SSIM2 = Path("/mnt/v/zen/zensim-training/2026-05-18-ssim2")
KONJND_PAIRS = Path("/mnt/v/zen/zensim-training/2026-05-18-extfeat/konjnd_pairs.tsv")

SAFESYN_FILL_RAW = {
    "ssim2": 68.7350,
    "cvvdp": 9.5912,
    "iwssim": 0.9725,
}


def load_score_tsv(path: Path, score_col: str) -> dict[tuple[str, str], float]:
    lookup = {}
    if not path.exists():
        return lookup
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                v = float(row[score_col])
                if np.isfinite(v):
                    lookup[(row["ref_path"], row["dist_path"])] = v
            except (ValueError, KeyError):
                pass
    return lookup


def fill_if_nan(v: float, key: str) -> float:
    if not np.isfinite(v):
        return float(SAFESYN_FILL_RAW[key])
    return float(v)


def append_metric_372(
    tbl: pa.Table, ssim2: list[float], cvvdp: list[float], iwssim: list[float]
) -> pa.Table:
    """Keep all f0..f371 columns; append f372/f373/f374 metric inputs."""
    n = tbl.num_rows
    assert len(ssim2) == n
    assert len(cvvdp) == n
    assert len(iwssim) == n
    tbl = tbl.append_column("f372", pa.array(ssim2, type=pa.float64()))
    tbl = tbl.append_column("f373", pa.array(cvvdp, type=pa.float64()))
    tbl = tbl.append_column("f374", pa.array(iwssim, type=pa.float64()))
    return tbl


def build_train_safesyn():
    print("\n=== TRAIN safesyn (375 col) ===")
    tbl = pq.read_table(str(CANONICAL / "train" / "safesyn.parquet"))
    ssim2 = np.array(tbl["ssim2_gpu"])
    cvvdp = np.array(tbl["cvvdp_score"])
    iwssim = np.array(tbl["iwssim"])
    ssim2 = np.where(np.isfinite(ssim2), ssim2, SAFESYN_FILL_RAW["ssim2"])
    cvvdp = np.where(np.isfinite(cvvdp), cvvdp, SAFESYN_FILL_RAW["cvvdp"])
    iwssim = np.where(np.isfinite(iwssim), iwssim, SAFESYN_FILL_RAW["iwssim"])
    out = append_metric_372(tbl, ssim2.tolist(), cvvdp.tolist(), iwssim.tolist())
    out_path = OUT_ROOT / "train_375" / "safesyn.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({out.num_rows} rows × {len(out.schema.names)} cols)")


def build_train_kadid_tid(name: str):
    print(f"\n=== TRAIN {name} (375 col, positional ssim2 fix) ===")
    tbl = pq.read_table(str(CANONICAL / "train" / f"{name}.parquet"))
    local = pq.read_table(str(LOCAL_SSIM2 / f"{name}_ssim2_local.parquet"))
    assert local.num_rows == tbl.num_rows
    ssim2 = np.array(local["ssim2_gpu"])
    cvvdp = np.array(tbl["cvvdp_score"])
    iwssim = np.array(tbl["iwssim"])
    ssim2 = np.where(np.isfinite(ssim2), ssim2, SAFESYN_FILL_RAW["ssim2"])
    cvvdp = np.where(np.isfinite(cvvdp), cvvdp, SAFESYN_FILL_RAW["cvvdp"])
    iwssim = np.where(np.isfinite(iwssim), iwssim, SAFESYN_FILL_RAW["iwssim"])
    out = append_metric_372(tbl, ssim2.tolist(), cvvdp.tolist(), iwssim.tolist())
    out_path = OUT_ROOT / "train_375" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({out.num_rows} rows × {len(out.schema.names)} cols)")


def build_train_cvvdp_LARGE():
    print("\n=== TRAIN cvvdp_iwssim_LARGE (375 col) ===")
    src = CANONICAL / "train" / "cvvdp_iwssim_LARGE.parquet"
    tbl = pq.read_table(str(src))
    # This parquet only has 300 features f0..f299 — pad to f0..f371 with zeros
    f_cols = [n for n in tbl.schema.names if n.startswith("f") and n[1:].isdigit()]
    print(f"  cvvdp_iwssim_LARGE has {len(f_cols)} f-cols ({f_cols[0]}..{f_cols[-1]})")
    n = tbl.num_rows
    # Add f300..f371 as zero columns
    for i in range(300, 372):
        col_name = f"f{i}"
        if col_name not in tbl.schema.names:
            tbl = tbl.append_column(col_name, pa.array([0.0] * n, type=pa.float64()))
    print(f"  padded to f0..f371: {tbl.num_rows} rows × {len(tbl.schema.names)} cols")
    # Append metric inputs
    cvvdp = np.array(tbl["cvvdp_score"])
    iwssim = np.array(tbl["iwssim"])
    ssim2 = np.full(n, SAFESYN_FILL_RAW["ssim2"], dtype=np.float64)
    cvvdp = np.where(np.isfinite(cvvdp), cvvdp, SAFESYN_FILL_RAW["cvvdp"])
    iwssim = np.where(np.isfinite(iwssim), iwssim, SAFESYN_FILL_RAW["iwssim"])
    out = append_metric_372(tbl, ssim2.tolist(), cvvdp.tolist(), iwssim.tolist())
    out_path = OUT_ROOT / "train_375" / "cvvdp_iwssim_LARGE.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({out.num_rows} rows × {len(out.schema.names)} cols)")


def build_val(name: str, pairs_tsv: Path, ssim2_tsv: Path, cvvdp_tsv: Path, iwssim_tsv: Path):
    print(f"\n=== VAL {name} (375 col) ===")
    tbl = pq.read_table(str(CANONICAL / "val" / f"{name}.parquet"))
    with open(pairs_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [(r["ref_path"], r["dist_path"]) for r in reader]
    assert len(rows) == tbl.num_rows
    with open(ssim2_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        ssim2_rows = [(r["ref_path"], r["dist_path"], float(r["ssim2_gpu"])) for r in reader]
    assert len(ssim2_rows) == tbl.num_rows
    ssim2_vals_pos = [r[2] for r in ssim2_rows]
    cvvdp_lookup = load_score_tsv(cvvdp_tsv, "cvvdp_imazen_v0_0_1")
    iwssim_lookup = load_score_tsv(iwssim_tsv, "iwssim_gpu")
    cvvdp_vals = []
    iwssim_vals = []
    for ref_path, dist_path in rows:
        c = cvvdp_lookup.get((ref_path, dist_path))
        iw = iwssim_lookup.get((ref_path, dist_path))
        cvvdp_vals.append(fill_if_nan(c if c is not None else float("nan"), "cvvdp"))
        iwssim_vals.append(fill_if_nan(iw if iw is not None else float("nan"), "iwssim"))
    ssim2_vals = [fill_if_nan(v, "ssim2") for v in ssim2_vals_pos]
    out = append_metric_372(tbl, ssim2_vals, cvvdp_vals, iwssim_vals)
    out_path = OUT_ROOT / "val_375" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({out.num_rows} rows × {len(out.schema.names)} cols)")


def build_val_konjnd():
    print("\n=== VAL konjnd (375 col, JPEG+BPG combined) ===")
    tbl = pq.read_table(str(CANONICAL / "val" / "konjnd.parquet"))
    with open(KONJND_PAIRS) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [(r["ref_path"], r["dist_path"]) for r in reader]
    assert len(rows) == tbl.num_rows
    with open(SSIM2_GPU / "konjnd_ssim2_gpu.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        ssim2_rows = [(r["ref_path"], r["dist_path"], float(r["ssim2_gpu"])) for r in reader]
    assert len(ssim2_rows) == tbl.num_rows
    ssim2_vals_pos = [r[2] for r in ssim2_rows]
    iwssim_lookup = {}
    for sub in ["konjnd_jpeg_iwssim_scores.tsv", "konjnd_bpg_iwssim_scores.tsv"]:
        iwssim_lookup.update(load_score_tsv(SCORES_IWSSIM / sub, "iwssim_gpu"))
    cvvdp_lookup = {}
    for sub in ["konjnd_jpeg_cvvdp_2026-05-17.tsv", "konjnd_bpg_cvvdp_2026-05-17.tsv"]:
        cvvdp_lookup.update(load_score_tsv(SCORES_CVVDP / sub, "cvvdp_imazen_v0_0_1"))
    cvvdp_vals = []
    iwssim_vals = []
    for ref_path, dist_path in rows:
        c = cvvdp_lookup.get((ref_path, dist_path))
        iw = iwssim_lookup.get((ref_path, dist_path))
        cvvdp_vals.append(fill_if_nan(c if c is not None else float("nan"), "cvvdp"))
        iwssim_vals.append(fill_if_nan(iw if iw is not None else float("nan"), "iwssim"))
    ssim2_vals = [fill_if_nan(v, "ssim2") for v in ssim2_vals_pos]
    out = append_metric_372(tbl, ssim2_vals, cvvdp_vals, iwssim_vals)
    out_path = OUT_ROOT / "val_375" / "konjnd.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({out.num_rows} rows × {len(out.schema.names)} cols)")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train_375").mkdir(exist_ok=True)
    (OUT_ROOT / "val_375").mkdir(exist_ok=True)

    build_train_safesyn()
    build_train_kadid_tid("kadid")
    build_train_kadid_tid("tid")
    build_train_cvvdp_LARGE()

    build_val("kadid",
              SCORES_CVVDP / "kadid_cvvdp_pairs_2026-05-17.tsv",
              SSIM2_GPU / "kadid_ssim2_gpu.tsv",
              SCORES_CVVDP / "kadid_cvvdp_scores_2026-05-17.tsv",
              SCORES_IWSSIM / "kadid_iwssim_scores.tsv")
    build_val("tid",
              SCORES_CVVDP / "tid_cvvdp_pairs_2026-05-17.tsv",
              SSIM2_GPU / "tid_ssim2_gpu.tsv",
              SCORES_CVVDP / "tid_cvvdp_scores_2026-05-17.tsv",
              SCORES_IWSSIM / "tid_iwssim_scores.tsv")
    build_val("cid22",
              SCORES_CVVDP / "cid22_cvvdp_pairs_2026-05-17.tsv",
              SSIM2_GPU / "cid22_ssim2_gpu.tsv",
              SCORES_CVVDP / "cid22_cvvdp_scores_2026-05-17.tsv",
              SCORES_IWSSIM / "cid22_iwssim_scores.tsv")
    build_val("aic3",
              SCORES_CVVDP / "aic3_cvvdp_pairs_2026-05-17.tsv",
              SSIM2_GPU / "aic3_ssim2_gpu.tsv",
              SCORES_CVVDP / "aic3_cvvdp_scores_2026-05-17.tsv",
              SCORES_IWSSIM / "aic3_iwssim_scores.tsv")
    build_val_konjnd()
    print("\nDONE.")


if __name__ == "__main__":
    main()
