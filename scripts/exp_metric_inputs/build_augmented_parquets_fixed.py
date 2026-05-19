#!/usr/bin/env python3
"""EXP-METRIC-INPUTS-FIX: build 303-feature augmented training + validation parquets.

This is the FIXED version of build_augmented_parquets.py addressing the 4 corpus
defects flagged in the EXP-METRIC-INPUTS falsification:

1. train/kadid + train/tid ssim2_gpu constant per image (upstream join bug).
   FIX: positional join with kadid_ssim2_local.parquet and tid_ssim2_local.parquet
   (which carry the correct per-pair scores, but their join keys all collide
   to a single ref). Row order matches canonical train/{kadid,tid}.parquet
   and dmos.csv / mos_with_names.txt verified.

2. konjnd-dense NaN scores.
   FIX: DROP konjnd-dense from training. The KonJND signal at inference comes
   from val/konjnd (1008 anchor pairs, see fix #3).

3. val/konjnd missing dist_path keying.
   FIX: konjnd_pairs.tsv carries the 1008 PJND-anchor pair paths. Row-aligned
   to val/konjnd.parquet. Compute ssim2-gpu via zen-metrics, join cvvdp+iwssim
   from existing TSVs.

4. Training=GPU-SSIMULACRA2, val=fast-ssim2-CPU scale mismatch.
   FIX: Computed GPU ssim2 for every val corpus via `zen-metrics batch
   --metric ssim2-gpu --gpu-runtime cuda` at /tmp/exp_metric_fix/. Use those
   exclusively. Never read fast_ssim2_score CPU column.

Output:
  /mnt/v/zen/zensim-training/2026-05-18-metric-inputs-fixed/train/{safesyn,kadid,tid,cvvdp_iwssim_LARGE}.parquet
  /mnt/v/zen/zensim-training/2026-05-18-metric-inputs-fixed/val/{cid22,kadid,tid,konjnd,aic3}.parquet
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

# Safesyn fill values for missing metric inputs (raw natural-scale)
# - ssim2 ∈ [0, 100] higher=better, cvvdp ∈ [0, 10] higher=better, iwssim ∈ [0, 1] higher=better
# Computed from /mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet
# (median for ssim2 because raw has heavy left-tail outliers down to -739)
SAFESYN_FILL_RAW = {
    "ssim2": 68.7350,
    "cvvdp": 9.5912,
    "iwssim": 0.9725,
}


def load_score_tsv(path: Path, score_col: str) -> dict[tuple[str, str], float]:
    """Load (ref_path, dist_path) → score lookup from a TSV."""
    lookup = {}
    if not path.exists():
        print(f"  WARN: {path} does not exist", file=sys.stderr)
        return lookup
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["ref_path"], row["dist_path"])
            try:
                v = float(row[score_col])
                if np.isfinite(v):
                    lookup[key] = v
            except (ValueError, KeyError):
                pass
    print(f"  loaded {len(lookup)} entries from {path.name} (score col: {score_col})")
    return lookup


def fill_if_nan(raw_value: float, key: str) -> float:
    """Pass raw score through; fill NaN with safesyn-natural-scale median/mean."""
    if not np.isfinite(raw_value):
        return float(SAFESYN_FILL_RAW[key])
    return float(raw_value)


def take_300_features(tbl: pa.Table) -> pa.Table:
    """Select non-feature cols + f0..f299, dropping f300..f371 if present."""
    n_feat_in = sum(1 for n in tbl.schema.names if n.startswith("f") and n[1:].isdigit())
    keep_features = [f"f{i}" for i in range(300)]
    other_cols = [n for n in tbl.schema.names if not (n.startswith("f") and n[1:].isdigit())]
    select_cols = other_cols + keep_features
    return tbl.select(select_cols), n_feat_in


def append_metric_cols(
    tbl: pa.Table,
    ssim2_vals: list[float],
    cvvdp_vals: list[float],
    iwssim_vals: list[float],
) -> pa.Table:
    """Append ssim2/cvvdp/iwssim as f300, f301, f302."""
    n = tbl.num_rows
    assert len(ssim2_vals) == n, f"len mismatch ssim2 {len(ssim2_vals)} != {n}"
    assert len(cvvdp_vals) == n, f"len mismatch cvvdp {len(cvvdp_vals)} != {n}"
    assert len(iwssim_vals) == n, f"len mismatch iwssim {len(iwssim_vals)} != {n}"
    tbl = tbl.append_column("f300", pa.array(ssim2_vals, type=pa.float64()))
    tbl = tbl.append_column("f301", pa.array(cvvdp_vals, type=pa.float64()))
    tbl = tbl.append_column("f302", pa.array(iwssim_vals, type=pa.float64()))
    return tbl


def stats(vals, name):
    a = np.array(vals, dtype=np.float64)
    return f"{name}: n={len(a)} finite={np.isfinite(a).sum()} mean={np.nanmean(a):.3f} std={np.nanstd(a):.3f} min={np.nanmin(a):.3f} max={np.nanmax(a):.3f}"


def build_training_safesyn():
    """safesyn already has per-pair ssim2/cvvdp/iwssim — no fix needed."""
    name = "safesyn"
    print(f"\n=== TRAIN {name} ===")
    src = CANONICAL / "train" / f"{name}.parquet"
    tbl = pq.read_table(str(src))

    ssim2 = np.array(tbl["ssim2_gpu"])
    cvvdp = np.array(tbl["cvvdp_score"])
    iwssim = np.array(tbl["iwssim"])

    print(f"  coverage: ssim2={np.isfinite(ssim2).sum()}/{tbl.num_rows} "
          f"cvvdp={np.isfinite(cvvdp).sum()}/{tbl.num_rows} "
          f"iwssim={np.isfinite(iwssim).sum()}/{tbl.num_rows}")

    ssim2 = np.where(np.isfinite(ssim2), ssim2, SAFESYN_FILL_RAW["ssim2"])
    cvvdp = np.where(np.isfinite(cvvdp), cvvdp, SAFESYN_FILL_RAW["cvvdp"])
    iwssim = np.where(np.isfinite(iwssim), iwssim, SAFESYN_FILL_RAW["iwssim"])

    print(f"  {stats(ssim2, 'ssim2')}")
    print(f"  {stats(cvvdp, 'cvvdp')}")
    print(f"  {stats(iwssim, 'iwssim')}")

    tbl_300, _ = take_300_features(tbl)
    tbl_out = append_metric_cols(tbl_300, ssim2.tolist(), cvvdp.tolist(), iwssim.tolist())

    out_path = OUT_ROOT / "train" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def build_training_kadid_tid_fixed(name: str):
    """KADID/TID: use positional join with local ssim2 parquets (row-aligned)."""
    print(f"\n=== TRAIN {name} (with positional ssim2 fix) ===")
    src = CANONICAL / "train" / f"{name}.parquet"
    tbl = pq.read_table(str(src))

    # Replace ssim2_gpu via positional row-order match
    local_ssim2_path = LOCAL_SSIM2 / f"{name}_ssim2_local.parquet"
    if local_ssim2_path.exists():
        local_tbl = pq.read_table(str(local_ssim2_path))
        assert local_tbl.num_rows == tbl.num_rows, (
            f"row mismatch {local_tbl.num_rows} != {tbl.num_rows} between "
            f"{local_ssim2_path} and {src}"
        )
        ssim2_fixed = np.array(local_tbl["ssim2_gpu"])
        n_finite = np.isfinite(ssim2_fixed).sum()
        ssim2_distinct_per_ref = "varies"  # implicit
        print(f"  fixed ssim2: {n_finite}/{tbl.num_rows} finite, "
              f"range [{np.nanmin(ssim2_fixed):.2f}, {np.nanmax(ssim2_fixed):.2f}]")
        # Sanity: ensure not all-equal within first 5 refs (means proper per-pair scores)
        refs = np.array(tbl["ref_basename"])
        for r in np.unique(refs)[:3]:
            mask = refs == r
            vals = ssim2_fixed[mask]
            assert len(np.unique(vals)) > 5, (
                f"After fix, ref={r} still has only {len(np.unique(vals))} distinct ssim2 values "
                f"(out of {mask.sum()} pairs) — positional fix didn't work"
            )
        print(f"  positional fix verified — per-ref ssim2 now varies properly")
    else:
        print(f"  FATAL: {local_ssim2_path} missing — cannot fix ssim2 for {name}", file=sys.stderr)
        sys.exit(1)

    cvvdp = np.array(tbl["cvvdp_score"])
    iwssim = np.array(tbl["iwssim"])
    print(f"  cvvdp coverage: {np.isfinite(cvvdp).sum()}/{tbl.num_rows}")
    print(f"  iwssim coverage: {np.isfinite(iwssim).sum()}/{tbl.num_rows}")

    # Fill NaN
    ssim2 = np.where(np.isfinite(ssim2_fixed), ssim2_fixed, SAFESYN_FILL_RAW["ssim2"])
    cvvdp = np.where(np.isfinite(cvvdp), cvvdp, SAFESYN_FILL_RAW["cvvdp"])
    iwssim = np.where(np.isfinite(iwssim), iwssim, SAFESYN_FILL_RAW["iwssim"])

    print(f"  {stats(ssim2, 'ssim2 (FIXED)')}")
    print(f"  {stats(cvvdp, 'cvvdp')}")
    print(f"  {stats(iwssim, 'iwssim')}")

    tbl_300, _ = take_300_features(tbl)
    tbl_out = append_metric_cols(tbl_300, ssim2.tolist(), cvvdp.tolist(), iwssim.tolist())

    out_path = OUT_ROOT / "train" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def build_training_cvvdp_LARGE():
    """cvvdp_iwssim_LARGE has cvvdp+iwssim but ssim2 all-null. Fill ssim2 with safesyn median."""
    name = "cvvdp_iwssim_LARGE"
    print(f"\n=== TRAIN {name} ===")
    src = CANONICAL / "train" / f"{name}.parquet"
    tbl = pq.read_table(str(src))

    n = tbl.num_rows
    cvvdp = np.array(tbl["cvvdp_score"])
    iwssim = np.array(tbl["iwssim"])
    ssim2 = np.full(n, SAFESYN_FILL_RAW["ssim2"], dtype=np.float64)
    print(f"  ssim2: imputed safesyn-median ({SAFESYN_FILL_RAW['ssim2']}) for all {n} rows")
    print(f"  cvvdp coverage: {np.isfinite(cvvdp).sum()}/{n}")
    print(f"  iwssim coverage: {np.isfinite(iwssim).sum()}/{n}")

    cvvdp = np.where(np.isfinite(cvvdp), cvvdp, SAFESYN_FILL_RAW["cvvdp"])
    iwssim = np.where(np.isfinite(iwssim), iwssim, SAFESYN_FILL_RAW["iwssim"])

    tbl_300, _ = take_300_features(tbl)
    tbl_out = append_metric_cols(tbl_300, ssim2.tolist(), cvvdp.tolist(), iwssim.tolist())

    out_path = OUT_ROOT / "train" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def build_val_corpus(
    name: str,
    pairs_tsv: Path,
    ssim2_tsv: Path,
    ssim2_col: str,
    cvvdp_tsv: Path,
    cvvdp_col: str,
    iwssim_tsv: Path,
    iwssim_col: str,
):
    """Generic val-corpus augmenter: positional join via pairs_tsv, key-join via score TSVs."""
    print(f"\n=== VAL {name} ===")
    src_parquet = CANONICAL / "val" / f"{name}.parquet"
    tbl = pq.read_table(str(src_parquet))

    # Load row-aligned ref/dist paths
    with open(pairs_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [(r["ref_path"], r["dist_path"]) for r in reader]
    assert len(rows) == tbl.num_rows, f"row mismatch {len(rows)} != {tbl.num_rows} in {pairs_tsv}"

    # Load ssim2 gpu via positional (same row order as pairs_tsv)
    with open(ssim2_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        ssim2_rows = []
        for r in reader:
            try:
                ssim2_rows.append((r["ref_path"], r["dist_path"], float(r[ssim2_col])))
            except (ValueError, KeyError):
                ssim2_rows.append((r["ref_path"], r["dist_path"], float("nan")))
    assert len(ssim2_rows) == tbl.num_rows, (
        f"ssim2 row mismatch {len(ssim2_rows)} != {tbl.num_rows} in {ssim2_tsv}"
    )
    # Verify positional alignment
    for i in range(min(5, tbl.num_rows)):
        assert ssim2_rows[i][:2] == rows[i], (
            f"ssim2 misalign at row {i}: ssim2={ssim2_rows[i][:2]} vs pairs={rows[i]}"
        )
    ssim2_vals_pos = [r[2] for r in ssim2_rows]

    # cvvdp/iwssim: key-join via (ref_path, dist_path)
    cvvdp_lookup = load_score_tsv(cvvdp_tsv, cvvdp_col)
    iwssim_lookup = load_score_tsv(iwssim_tsv, iwssim_col)

    cvvdp_vals = []
    iwssim_vals = []
    hit_c = hit_i = 0
    for ref_path, dist_path in rows:
        key = (ref_path, dist_path)
        c = cvvdp_lookup.get(key)
        iw = iwssim_lookup.get(key)
        if c is not None and np.isfinite(c):
            cvvdp_vals.append(float(c))
            hit_c += 1
        else:
            cvvdp_vals.append(SAFESYN_FILL_RAW["cvvdp"])
        if iw is not None and np.isfinite(iw):
            iwssim_vals.append(float(iw))
            hit_i += 1
        else:
            iwssim_vals.append(SAFESYN_FILL_RAW["iwssim"])

    # Fill ssim2 NaNs
    ssim2_vals = [fill_if_nan(v, "ssim2") for v in ssim2_vals_pos]

    n = tbl.num_rows
    n_ssim2_hit = sum(1 for v in ssim2_vals_pos if np.isfinite(v))
    print(f"  join coverage: ssim2_gpu={n_ssim2_hit}/{n} cvvdp={hit_c}/{n} iwssim={hit_i}/{n}")
    print(f"  {stats(ssim2_vals, 'ssim2')}")
    print(f"  {stats(cvvdp_vals, 'cvvdp')}")
    print(f"  {stats(iwssim_vals, 'iwssim')}")

    tbl_300, _ = take_300_features(tbl)
    tbl_out = append_metric_cols(tbl_300, ssim2_vals, cvvdp_vals, iwssim_vals)

    out_path = OUT_ROOT / "val" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def build_val_konjnd_combined():
    """konjnd val: 1008 anchor pairs (504 JPEG + 504 BPG). Combine the two iwssim TSVs."""
    name = "konjnd"
    print(f"\n=== VAL {name} (combined JPEG + BPG anchors) ===")
    src_parquet = CANONICAL / "val" / f"{name}.parquet"
    tbl = pq.read_table(str(src_parquet))

    # Load pairs (1008)
    with open(KONJND_PAIRS) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [(r["ref_path"], r["dist_path"]) for r in reader]
    assert len(rows) == tbl.num_rows, f"row mismatch {len(rows)} != {tbl.num_rows}"

    # Load combined iwssim (JPEG + BPG)
    iwssim_lookup = {}
    for sub in ["konjnd_jpeg_iwssim_scores.tsv", "konjnd_bpg_iwssim_scores.tsv"]:
        iwssim_lookup.update(load_score_tsv(SCORES_IWSSIM / sub, "iwssim_gpu"))

    # Load combined cvvdp (the dense files include anchor pairs)
    cvvdp_lookup = {}
    for sub in ["konjnd_jpeg_cvvdp_2026-05-17.tsv", "konjnd_bpg_cvvdp_2026-05-17.tsv"]:
        cvvdp_lookup.update(load_score_tsv(SCORES_CVVDP / sub, "cvvdp_imazen_v0_0_1"))

    # ssim2-gpu we just computed at /tmp/exp_metric_fix/konjnd_ssim2_gpu.tsv (positional)
    with open(SSIM2_GPU / "konjnd_ssim2_gpu.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        ssim2_rows = [(r["ref_path"], r["dist_path"], float(r["ssim2_gpu"])) for r in reader]
    assert len(ssim2_rows) == tbl.num_rows, (
        f"ssim2 row mismatch {len(ssim2_rows)} != {tbl.num_rows}"
    )
    for i in range(min(5, tbl.num_rows)):
        assert ssim2_rows[i][:2] == rows[i], (
            f"ssim2 misalign at row {i}: ssim2={ssim2_rows[i][:2]} vs pairs={rows[i]}"
        )
    ssim2_vals_pos = [r[2] for r in ssim2_rows]

    cvvdp_vals = []
    iwssim_vals = []
    hit_c = hit_i = 0
    for ref_path, dist_path in rows:
        key = (ref_path, dist_path)
        c = cvvdp_lookup.get(key)
        iw = iwssim_lookup.get(key)
        if c is not None and np.isfinite(c):
            cvvdp_vals.append(float(c))
            hit_c += 1
        else:
            cvvdp_vals.append(SAFESYN_FILL_RAW["cvvdp"])
        if iw is not None and np.isfinite(iw):
            iwssim_vals.append(float(iw))
            hit_i += 1
        else:
            iwssim_vals.append(SAFESYN_FILL_RAW["iwssim"])

    ssim2_vals = [fill_if_nan(v, "ssim2") for v in ssim2_vals_pos]
    n = tbl.num_rows
    n_ssim2_hit = sum(1 for v in ssim2_vals_pos if np.isfinite(v))
    print(f"  join coverage: ssim2_gpu={n_ssim2_hit}/{n} cvvdp={hit_c}/{n} iwssim={hit_i}/{n}")
    print(f"  {stats(ssim2_vals, 'ssim2')}")
    print(f"  {stats(cvvdp_vals, 'cvvdp')}")
    print(f"  {stats(iwssim_vals, 'iwssim')}")

    tbl_300, _ = take_300_features(tbl)
    tbl_out = append_metric_cols(tbl_300, ssim2_vals, cvvdp_vals, iwssim_vals)

    out_path = OUT_ROOT / "val" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, str(out_path), compression="zstd", compression_level=15)
    print(f"  wrote {out_path} ({tbl_out.num_rows} rows × {len(tbl_out.schema.names)} cols)")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train").mkdir(exist_ok=True)
    (OUT_ROOT / "val").mkdir(exist_ok=True)

    # Training corpora
    build_training_safesyn()
    build_training_kadid_tid_fixed("kadid")
    build_training_kadid_tid_fixed("tid")
    build_training_cvvdp_LARGE()
    # konjnd-dense intentionally dropped — all metric scores are null and no per-pair join key

    # Validation corpora
    # KADID
    build_val_corpus(
        "kadid",
        pairs_tsv=SCORES_CVVDP / "kadid_cvvdp_pairs_2026-05-17.tsv",
        ssim2_tsv=SSIM2_GPU / "kadid_ssim2_gpu.tsv",
        ssim2_col="ssim2_gpu",
        cvvdp_tsv=SCORES_CVVDP / "kadid_cvvdp_scores_2026-05-17.tsv",
        cvvdp_col="cvvdp_imazen_v0_0_1",
        iwssim_tsv=SCORES_IWSSIM / "kadid_iwssim_scores.tsv",
        iwssim_col="iwssim_gpu",
    )
    # TID
    build_val_corpus(
        "tid",
        pairs_tsv=SCORES_CVVDP / "tid_cvvdp_pairs_2026-05-17.tsv",
        ssim2_tsv=SSIM2_GPU / "tid_ssim2_gpu.tsv",
        ssim2_col="ssim2_gpu",
        cvvdp_tsv=SCORES_CVVDP / "tid_cvvdp_scores_2026-05-17.tsv",
        cvvdp_col="cvvdp_imazen_v0_0_1",
        iwssim_tsv=SCORES_IWSSIM / "tid_iwssim_scores.tsv",
        iwssim_col="iwssim_gpu",
    )
    # CID22
    build_val_corpus(
        "cid22",
        pairs_tsv=SCORES_CVVDP / "cid22_cvvdp_pairs_2026-05-17.tsv",
        ssim2_tsv=SSIM2_GPU / "cid22_ssim2_gpu.tsv",
        ssim2_col="ssim2_gpu",
        cvvdp_tsv=SCORES_CVVDP / "cid22_cvvdp_scores_2026-05-17.tsv",
        cvvdp_col="cvvdp_imazen_v0_0_1",
        iwssim_tsv=SCORES_IWSSIM / "cid22_iwssim_scores.tsv",
        iwssim_col="iwssim_gpu",
    )
    # AIC3
    build_val_corpus(
        "aic3",
        pairs_tsv=SCORES_CVVDP / "aic3_cvvdp_pairs_2026-05-17.tsv",
        ssim2_tsv=SSIM2_GPU / "aic3_ssim2_gpu.tsv",
        ssim2_col="ssim2_gpu",
        cvvdp_tsv=SCORES_CVVDP / "aic3_cvvdp_scores_2026-05-17.tsv",
        cvvdp_col="cvvdp_imazen_v0_0_1",
        iwssim_tsv=SCORES_IWSSIM / "aic3_iwssim_scores.tsv",
        iwssim_col="iwssim_gpu",
    )
    # KonJND combined
    build_val_konjnd_combined()

    print("\nDONE.")


if __name__ == "__main__":
    main()
