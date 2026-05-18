#!/usr/bin/env python3
"""Build V_24-alpha sweep training corpora for a single α value.

For each α ∈ {0.05, 0.10, 0.15, 0.20, 0.25} (and any extension):

    mix_alpha = α · ssim2_log_norm
              + (1 − α) · (0.4 · cvvdp_log_norm + 0.6 · iwssim_log_norm)

α = 0    → V_22-mix-LARGE (cvvdp/iwssim only)
α = 1/3  → V_24-full (symmetric 3-way)
α ∈ [0.05, 0.25] → this sweep, looking for the Pareto sweet spot.

Per-group behavior:

- safesyn: derive ssim2_log_norm from human_score*100 (the EX-3 discovery).
- kadid / tid: ssim2_log_norm already present (EX-3 outputs).
- LARGE: ssim2_log_norm / iwssim_log_norm / cvvdp_log_norm all present.
- konjnd: PJND-only — keep human_score as the target.

All output files use a unified `mix_target` column so the trainer
runs with `--target-column mix_target --target-scale 1.0`.

Outputs land at
`/mnt/v/zen/zensim-training/2026-05-18-v24-alpha/<group>_alpha<XX>.parquet`
with the XX encoded as percent*100 (e.g. α=0.05 → `alpha005`).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


OUT_DIR = "/mnt/v/zen/zensim-training/2026-05-18-v24-alpha"

SAFESYN_IN = "/mnt/v/zen/zensim-training/2026-05-17-cvvdp/safesyn_features_mix_targets_372col.parquet"
KADID_IN = "/mnt/v/zen/zensim-training/2026-05-18-ssim2/kadid_4target_372col.parquet"
TID_IN = "/mnt/v/zen/zensim-training/2026-05-18-ssim2/tid_4target_372col.parquet"
LARGE_IN = "/mnt/v/zen/zensim-training/2026-05-18-ssim2/large_3target_300feat_minus_ssim2.parquet"
KONJND_IN = "/mnt/v/zen/zensim-training/2026-05-17-cvvdp/konjnd_features_mix_targets_372col.parquet"


def alpha_suffix(alpha: float) -> str:
    # 0.05 -> "alpha005", 0.10 -> "alpha010", 0.15 -> "alpha015"
    return f"alpha{int(round(alpha * 100)):03d}"


def ssim2_log_norm(x: np.ndarray) -> np.ndarray:
    """Canonical EX-3 ssim2 → log_norm transform."""
    return np.clip((x + 30.0) / 130.0 * 100.0, 0.0, 100.0)


def add_or_replace_column(t: pa.Table, name: str, arr) -> pa.Table:
    if name in t.column_names:
        idx = t.column_names.index(name)
        return t.set_column(idx, name, pa.array(arr))
    return t.append_column(name, pa.array(arr))


def maybe_rename_feat_cols(t: pa.Table) -> pa.Table:
    if any(c.startswith("feat_") and c[5:].isdigit() for c in t.column_names):
        new_names = [
            "f" + c[5:] if c.startswith("feat_") and c[5:].isdigit() else c
            for c in t.column_names
        ]
        return t.rename_columns(new_names)
    return t


def compute_alpha_mix(
    cv: np.ndarray,
    iw: np.ndarray,
    sm: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """The α-weighted mix.

    Base = 0.4·cvvdp_log_norm + 0.6·iwssim_log_norm (matches V_22-mix-LARGE).
    mix_alpha = α · ssim2_log_norm + (1 − α) · base.
    """
    base = 0.4 * cv + 0.6 * iw
    return alpha * sm + (1.0 - alpha) * base


def build_safesyn(path: str, out_path: str, alpha: float) -> None:
    print(f"\n=== safesyn α={alpha:.3f}: {path} ===")
    t = pq.read_table(path)
    print(f"  in: rows={t.num_rows}, cols={t.num_columns}")
    hs = t.column("human_score").to_numpy()
    ssim2 = hs * 100.0
    sln = ssim2_log_norm(ssim2)
    cv = t.column("cvvdp_log_norm").to_numpy()
    iw = t.column("iwssim_log_norm").to_numpy()
    mix = compute_alpha_mix(cv, iw, sln, alpha)
    t = add_or_replace_column(t, "ssim2_gpu", ssim2)
    t = add_or_replace_column(t, "ssim2_log_norm", sln)
    t = add_or_replace_column(t, "mix_target", mix)
    print(f"  mix_target: min={mix.min():.2f} max={mix.max():.2f} mean={mix.mean():.2f}")
    pq.write_table(t, out_path, compression="zstd", compression_level=9)
    print(f"  wrote: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


def build_kadid_tid(path: str, out_path: str, alpha: float, name: str) -> None:
    print(f"\n=== {name} α={alpha:.3f}: {path} ===")
    t = pq.read_table(path)
    print(f"  in: rows={t.num_rows}, cols={t.num_columns}")
    cv = t.column("cvvdp_log_norm").to_numpy()
    iw = t.column("iwssim_log_norm").to_numpy()
    sm = t.column("ssim2_log_norm").to_numpy()
    finite = np.isfinite(cv) & np.isfinite(iw) & np.isfinite(sm)
    n_nan = int((~finite).sum())
    if n_nan > 0:
        print(f"  dropping {n_nan} NaN/inf rows ({n_nan * 100 / len(cv):.1f}%)")
        t = t.filter(pa.array(finite))
        cv, iw, sm = cv[finite], iw[finite], sm[finite]
    mix = compute_alpha_mix(cv, iw, sm, alpha)
    t = add_or_replace_column(t, "mix_target", mix)
    t = maybe_rename_feat_cols(t)
    print(f"  mix_target: min={mix.min():.2f} max={mix.max():.2f} mean={mix.mean():.2f}")
    pq.write_table(t, out_path, compression="zstd", compression_level=9)
    print(f"  wrote: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB), rows={t.num_rows}")


def build_large(path: str, out_path: str, alpha: float) -> None:
    print(f"\n=== large α={alpha:.3f}: {path} ===")
    t = pq.read_table(path)
    print(f"  in: rows={t.num_rows}, cols={t.num_columns}")
    cv = t.column("cvvdp_log_norm").to_numpy()
    iw = t.column("iwssim_log_norm").to_numpy()
    sm = t.column("ssim2_log_norm").to_numpy()
    finite = np.isfinite(cv) & np.isfinite(iw) & np.isfinite(sm)
    n_nan = int((~finite).sum())
    if n_nan > 0:
        print(f"  dropping {n_nan} NaN/inf rows ({n_nan * 100 / len(cv):.1f}%)")
        t = t.filter(pa.array(finite))
        cv, iw, sm = cv[finite], iw[finite], sm[finite]
    mix = compute_alpha_mix(cv, iw, sm, alpha)
    t = add_or_replace_column(t, "mix_target", mix)
    t = maybe_rename_feat_cols(t)
    print(f"  mix_target: min={mix.min():.2f} max={mix.max():.2f} mean={mix.mean():.2f}")
    pq.write_table(t, out_path, compression="zstd", compression_level=9)
    print(f"  wrote: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB), rows={t.num_rows}")


def build_konjnd(path: str, out_path: str) -> None:
    print(f"\n=== konjnd (PJND-as-target): {path} ===")
    t = pq.read_table(path)
    hs = t.column("human_score").to_numpy()
    t = add_or_replace_column(t, "mix_target", hs)
    pq.write_table(t, out_path, compression="zstd", compression_level=9)
    print(f"  wrote: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB), rows={t.num_rows}")


def build_alpha(alpha: float, out_dir: str) -> None:
    sfx = alpha_suffix(alpha)
    os.makedirs(out_dir, exist_ok=True)
    build_safesyn(SAFESYN_IN, os.path.join(out_dir, f"safesyn_{sfx}.parquet"), alpha)
    build_kadid_tid(KADID_IN, os.path.join(out_dir, f"kadid_{sfx}.parquet"), alpha, "kadid")
    build_kadid_tid(TID_IN, os.path.join(out_dir, f"tid_{sfx}.parquet"), alpha, "tid")
    build_large(LARGE_IN, os.path.join(out_dir, f"large_{sfx}.parquet"), alpha)
    # konjnd is α-independent (keep PJND as target)
    konjnd_out = os.path.join(out_dir, "konjnd.parquet")
    if not os.path.exists(konjnd_out):
        build_konjnd(KONJND_IN, konjnd_out)
    else:
        print(f"\n=== konjnd: already built at {konjnd_out} (α-independent)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument(
        "--alpha", type=float, action="append",
        help="α value(s) to build; pass multiple times for batch builds."
    )
    args = parser.parse_args()
    alphas = args.alpha or [0.05, 0.10, 0.15, 0.20, 0.25]
    print(f"[v24-alpha-sweep] building corpora for α = {alphas}")
    for a in alphas:
        build_alpha(a, args.out_dir)
    print("\n[v24-alpha-sweep] done. Output dir:", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
