#!/usr/bin/env python3
"""Apply recomputed IW-SSIM + SSIMULACRA2 scores to the canonical KADID/TID
train parquets, fixing the DATA INTEGRITY BUG (2026-05-25).

The bug (root cause):
  In the upstream `2026-05-18-v24/{kadid,tid}_4target_372col.parquet` build,
  the `iwssim` column was created as a literal copy of `human_score`
  (target leak — never overwritten with real IW-SSIM), and the `ssim2_gpu`
  column was joined on `ref_basename` ALONE, so every distorted variant of a
  reference got the SAME ssim2 value (ssim2(ref, ref) ≈ 99.99 for KADID),
  giving ~0 correlation with MOS. The canonical builders
  (`build_canonical_parquets.py` → `build_canonical_2026_05_21.py`)
  faithfully propagated both corrupt columns. The `iwssim_log_norm`,
  `ssim2_log_norm`, and every `mix_*` column are derived from the corrupt
  raw values and are therefore ALSO corrupt.

This script:
  1. Reads recomputed per-pair scores (positionally row-aligned to the
     parquet, produced by `fix_kadid_tid_build_pairs.py` + `zen-metrics batch`).
  2. Replaces, in row order:
       iwssim            ← real IW-SSIM (Wang & Li 2011, [0,1])
       iwssim_log_norm   ← clip(SLOPE_IW·(-log(1-iw+1e-6))+INT_IW, 0, 100)
       ssim2_gpu         ← real SSIMULACRA2 ([-inf,100])
       ssim2_log_norm    ← max((ssim2+30)/1.3, 0)
       mix_cvX_iwY       ← X·cvvdp_log_norm + Y·iwssim_log_norm
       mix_cv33_iw33_sm33← (cvvdp_log_norm + iwssim_log_norm + ssim2_log_norm)/3
       mix_target        ← == mix_cv33_iw33_sm33 (verified equal in source)
  3. Preserves EXACT row order + every other column (human_score, cvvdp_*,
     pjnd_target, f0..f371) unchanged.
  4. Writes a `*_fixed_2026-05-25.parquet` SIBLING — never overwrites the
     original. The user promotes after review.

Transform constants are verified-exact against safesyn's real columns:
  iwssim_log_norm: build_larger_large.py SLOPE_IW=7.2837 INT_IW=0.0302 (maxerr 0.03)
  ssim2_log_norm:  v11_extract_cid22_train.py (ssim2+30)/1.3 (maxerr 1.4e-14)
  mix_* / mix_target: solved exactly from kadid source columns (maxerr <1e-13)
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# IW-SSIM log-norm constants (build_larger_large.py)
SLOPE_IW = 7.2837
INT_IW = 0.0302

OUT_DIR_DEFAULT = Path("/mnt/v/output/zensim/data-fix-2026-05-25")
CANON21 = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21/train")
CANON18 = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18/train")


def iwssim_log_norm_arr(iw: np.ndarray) -> np.ndarray:
    iw_c = np.clip(iw, 0.0, 1.0 - 1e-9)
    raw = -np.log(1.0 - iw_c + 1e-6)
    return np.clip(SLOPE_IW * raw + INT_IW, 0.0, 100.0)


def ssim2_log_norm_arr(s2: np.ndarray) -> np.ndarray:
    return np.maximum((s2 + 30.0) / 1.3, 0.0)


def load_scores(tsv_path: Path, col_candidates: list[str]) -> np.ndarray:
    rows = list(csv.DictReader(open(tsv_path), delimiter="\t"))
    fields = rows[0].keys() if rows else []
    col = next((c for c in col_candidates if c in fields), None)
    if col is None:
        raise RuntimeError(f"{tsv_path}: none of {col_candidates} present; have {list(fields)}")
    return np.array([float(r[col]) for r in rows], dtype=float)


def fix_corpus(name: str, parquet_path: Path, ssim2_tsv: Path, iwssim_tsv: Path,
               out_path: Path, ref_basenames_check: bool = True):
    print(f"\n--- fix {name}: {parquet_path}")
    tbl = pq.read_table(str(parquet_path))
    n = tbl.num_rows
    cols = tbl.schema.names

    ssim2 = load_scores(ssim2_tsv, ["ssim2", "ssim2_gpu", "ssimulacra2"])
    iwssim = load_scores(iwssim_tsv, ["iwssim", "iwssim_imazen_v0_0_1", "iwssim_gpu"])
    if len(ssim2) != n or len(iwssim) != n:
        raise RuntimeError(f"row mismatch: parquet {n}, ssim2 {len(ssim2)}, iwssim {len(iwssim)}")

    # Sanity: recomputed scores must NOT be a copy of human_score
    hs = np.asarray(tbl.column("human_score").to_numpy(zero_copy_only=False), dtype=float)
    if np.mean(np.isclose(hs, iwssim)) > 0.5:
        raise RuntimeError("recomputed iwssim is STILL ~human_score — the fix is wrong")

    # Derived columns
    iwl = iwssim_log_norm_arr(iwssim)
    s2l = ssim2_log_norm_arr(ssim2)
    cvl = np.asarray(tbl.column("cvvdp_log_norm").to_numpy(zero_copy_only=False), dtype=float)

    new_vals: dict[str, np.ndarray] = {
        "iwssim": iwssim,
        "iwssim_log_norm": iwl,
        "ssim2_gpu": ssim2,
        "ssim2_log_norm": s2l,
    }

    # Recompute mix_* columns (all depend on iwssim_log_norm)
    for c in cols:
        m2 = re.match(r"mix_cv(\d+)_iw(\d+)$", c)
        m3 = re.match(r"mix_cv(\d+)_iw(\d+)_sm(\d+)$", c)
        if m3:
            a, b, cc = int(m3.group(1)) / 100, int(m3.group(2)) / 100, int(m3.group(3)) / 100
            # source used exact 1/3 each for cv33_iw33_sm33
            if (a, b, cc) == (0.33, 0.33, 0.33):
                new_vals[c] = (cvl + iwl + s2l) / 3.0
            else:
                new_vals[c] = a * cvl + b * iwl + cc * s2l
        elif m2:
            cvw, iww = int(m2.group(1)) / 100, int(m2.group(2)) / 100
            new_vals[c] = cvw * cvl + iww * iwl
    # mix_target == mix_cv33_iw33_sm33 in source
    if "mix_target" in cols and "mix_cv33_iw33_sm33" in new_vals:
        new_vals["mix_target"] = new_vals["mix_cv33_iw33_sm33"]

    # Build the new table preserving column order
    arrays = []
    for c in cols:
        if c in new_vals:
            arrays.append(pa.array(new_vals[c].astype(np.float64)))
        else:
            arrays.append(tbl.column(c))
    fixed = pa.Table.from_arrays(arrays, names=cols)

    # Preserve + augment schema metadata
    src_meta = tbl.schema.metadata or {}
    meta = {k.decode(): v.decode() for k, v in src_meta.items()}
    note = meta.get("schema_note", "")
    fix_note = ("DATA-INTEGRITY FIX 2026-05-25: iwssim (was human_score copy) + ssim2_gpu "
                "(was joined on ref_basename only) + their *_log_norm + all mix_* columns "
                "recomputed on the correct (ref,dist) pairs via zen-metrics ssim2 + iwssim. "
                "human_score, cvvdp_*, pjnd_target, f0..fN unchanged.")
    meta["schema_note"] = f"{note}; {fix_note}" if note else fix_note
    meta["data_integrity_fix"] = "kadid_tid_iwssim_ssim2_recompute_2026-05-25"
    fixed = fixed.replace_schema_metadata({k.encode(): v.encode() for k, v in meta.items()})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(fixed, str(out_path), compression="zstd", compression_level=15)
    print(f"  WROTE {out_path} ({fixed.num_rows} rows × {fixed.num_columns} cols, "
          f"{out_path.stat().st_size/1e6:.1f} MB)")

    # Post-fix report
    from scipy.stats import spearmanr
    m = np.isfinite(ssim2) & np.isfinite(hs)
    print(f"  POST-FIX iwssim==human ident%: {100*np.mean(np.isclose(hs, iwssim)):.1f} (was 100)")
    print(f"  POST-FIX ssim2 SROCC vs human: {spearmanr(hs[m], ssim2[m]).correlation:.4f} (was ~0)")
    print(f"  POST-FIX iwssim SROCC vs human: {spearmanr(hs[m], iwssim[m]).correlation:.4f}")
    print(f"  POST-FIX ssim2 range [{ssim2.min():.2f},{ssim2.max():.2f}] "
          f"iwssim range [{iwssim.min():.4f},{iwssim.max():.4f}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["kadid", "tid", "both"], default="both")
    ap.add_argument("--canonical", choices=["2026-05-21", "2026-05-18"], default="2026-05-21")
    ap.add_argument("--scores-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    canon = CANON21 if args.canonical == "2026-05-21" else CANON18
    sd = args.scores_dir
    corpora = ["kadid", "tid"] if args.corpus == "both" else [args.corpus]
    for c in corpora:
        fix_corpus(
            c,
            canon / f"{c}.parquet",
            sd / f"{c}_ssim2.tsv",
            sd / f"{c}_iwssim.tsv",
            canon / f"{c}_fixed_2026-05-25.parquet",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
