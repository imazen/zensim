#!/usr/bin/env python3
"""Build V_24 training corpus with renamed `mix_target` columns.

DISCOVERY (2026-05-18 ex3-followup): the safesyn parquet's
`human_score` column equals `ssim2_gpu / 100` — safesyn was already
trained against ssim2 as ground truth. So safesyn's ssim2 is already
present and the prior agent's "57hr fleet" estimate was based on a
wrong premise that ssim2 wasn't available on safesyn.

This builder:

1. Takes safesyn parquet → derives `ssim2_log_norm` from human_score*100
   using the same `(x+30)/130*100` clip transform as EX-3 KADID/TID/LARGE.
2. Builds `mix_target = (cvvdp_log_norm + iwssim_log_norm + ssim2_log_norm) / 3`.
3. Renames in-place:
   - safesyn:  new `mix_target` column (3-way symmetric)
   - kadid_4target / tid_4target: `mix_cv33_iw33_sm33` → `mix_target`
   - LARGE 3-target: `mix_cv33_iw33_sm33` → `mix_target`
   - konjnd: keep `human_score` as the target column, renamed to `mix_target`.
     The 0.02 group weight in the V_22-mix-LARGE recipe handles scale mismatch
     between PJND and the codec-mix target.
4. Emits five output parquets to /mnt/v/zen/zensim-training/2026-05-18-v24/.

All transforms exactly mirror methodology doc
benchmarks/v0_24_ex3_ssim2_target_methodology_2026-05-18.md.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


OUT_DIR = "/mnt/v/zen/zensim-training/2026-05-18-v24"

SAFESYN_IN = "/mnt/v/zen/zensim-training/2026-05-17-cvvdp/safesyn_features_mix_targets_372col.parquet"
KADID_IN = "/mnt/v/zen/zensim-training/2026-05-18-ssim2/kadid_4target_372col.parquet"
TID_IN = "/mnt/v/zen/zensim-training/2026-05-18-ssim2/tid_4target_372col.parquet"
LARGE_IN = "/mnt/v/zen/zensim-training/2026-05-18-ssim2/large_3target_300feat_minus_ssim2.parquet"
KONJND_IN = "/mnt/v/zen/zensim-training/2026-05-17-cvvdp/konjnd_features_mix_targets_372col.parquet"


def ssim2_log_norm(x: np.ndarray) -> np.ndarray:
    """Canonical EX-3 ssim2 → log_norm transform.

    Maps ssim2 range [-30, 100] → [0, 100] linearly, clipped.
    Identical to the formula used in EX-3 kadid/tid/LARGE corpora.
    """
    return np.clip((x + 30.0) / 130.0 * 100.0, 0.0, 100.0)


def add_or_replace_column(t: pa.Table, name: str, arr) -> pa.Table:
    """Replace `name` column if present, otherwise append."""
    if name in t.column_names:
        idx = t.column_names.index(name)
        return t.set_column(idx, name, pa.array(arr))
    return t.append_column(name, pa.array(arr))


def build_safesyn(path: str, out_path: str) -> None:
    print(f"\n=== safesyn: {path} ===")
    t = pq.read_table(path)
    print(f"  in: rows={t.num_rows}, cols={t.num_columns}")
    # safesyn's "human_score" IS ssim2 / 100 (verified 2026-05-18)
    hs = t.column("human_score").to_numpy()
    ssim2 = hs * 100.0
    sln = ssim2_log_norm(ssim2)
    cv = t.column("cvvdp_log_norm").to_numpy()
    iw = t.column("iwssim_log_norm").to_numpy()
    mix = (cv + iw + sln) / 3.0
    t = add_or_replace_column(t, "ssim2_gpu", ssim2)
    t = add_or_replace_column(t, "ssim2_log_norm", sln)
    t = add_or_replace_column(t, "mix_target", mix)
    print(f"  ssim2: min={ssim2.min():.2f} max={ssim2.max():.2f} mean={ssim2.mean():.2f}")
    print(f"  ssim2_log_norm: min={sln.min():.2f} max={sln.max():.2f} mean={sln.mean():.2f}")
    print(f"  mix_target: min={mix.min():.2f} max={mix.max():.2f} mean={mix.mean():.2f}")
    pq.write_table(t, out_path, compression="zstd", compression_level=9)
    print(f"  wrote: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


def build_rename_mix33(path: str, out_path: str, in_name: str = "mix_cv33_iw33_sm33") -> None:
    print(f"\n=== rename {in_name} → mix_target: {path} ===")
    t = pq.read_table(path)
    print(f"  in: rows={t.num_rows}, cols={t.num_columns}")
    arr = t.column(in_name).to_numpy()
    finite = np.isfinite(arr)
    n_nan = int((~finite).sum())
    if n_nan > 0:
        print(f"  dropping {n_nan} NaN/inf rows ({n_nan*100/len(arr):.1f}%)")
        t = t.filter(pa.array(finite))
        arr = arr[finite]
    print(f"  {in_name}: min={arr.min():.2f} max={arr.max():.2f} mean={arr.mean():.2f}")
    t = add_or_replace_column(t, "mix_target", arr)
    pq.write_table(t, out_path, compression="zstd", compression_level=9)
    print(f"  wrote: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB), rows={t.num_rows}")


def build_konjnd(path: str, out_path: str) -> None:
    print(f"\n=== konjnd: keep PJND-as-target, rename: {path} ===")
    t = pq.read_table(path)
    print(f"  in: rows={t.num_rows}, cols={t.num_columns}")
    # konjnd's human_score IS PJND (in [0,100] scale). Group weight 0.02
    # handles scale mismatch with codec-mix target.
    hs = t.column("human_score").to_numpy()
    print(f"  PJND (human_score): min={hs.min():.2f} max={hs.max():.2f} mean={hs.mean():.2f}")
    t = add_or_replace_column(t, "mix_target", hs)
    pq.write_table(t, out_path, compression="zstd", compression_level=9)
    print(f"  wrote: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    build_safesyn(SAFESYN_IN, os.path.join(args.out_dir, "safesyn_4target_372col.parquet"))
    build_rename_mix33(KADID_IN, os.path.join(args.out_dir, "kadid_4target_372col.parquet"))
    build_rename_mix33(TID_IN, os.path.join(args.out_dir, "tid_4target_372col.parquet"))
    build_rename_mix33(LARGE_IN, os.path.join(args.out_dir, "large_3target_300feat_minus_ssim2.parquet"))
    build_konjnd(KONJND_IN, os.path.join(args.out_dir, "konjnd_features_mix_targets_372col.parquet"))
    print("\n[v24-corpus] done. Output dir:", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
