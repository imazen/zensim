#!/usr/bin/env python3
"""Build the EXP-LARGER-LARGE training parquet.

Consolidates:
  - Existing iwssim sidecar: /mnt/v/zen/zensim-training/2026-05-15-cvvdp-r2/iwssim_imazen_consolidated.parquet (75,300 rows)
  - New iwssim sidecars: <new_sidecars_dir>/*.parquet (downloaded from R2)
  - Unified cvvdp+features sources: /mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_*_cvvdp.parquet (2.37M rows)

Joins on (basename, codec, q, knob_tuple_json). Output schema matches canonical
cvvdp_iwssim_LARGE: ref_basename + human_score + cvvdp_score + cvvdp_log_norm +
iwssim + iwssim_log_norm + mix_cv40_iw60 + f0..f299.

Uses exact constants derived from existing v2 LARGE:
  cvvdp_log_norm = (-log(10 - cv + 1e-6) - LO_CV) / (HI_CV - LO_CV) * 100
  iwssim_log_norm = min(100, SLOPE_IW * (-log(1 - clip(iw, 0, 1-1e-9) + 1e-6)) + INT_IW)
"""
import argparse
import os
import sys
import math
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

UNIFIED_DIR = Path('/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged')
EXISTING_IWSSIM = Path('/mnt/v/zen/zensim-training/2026-05-15-cvvdp-r2/iwssim_imazen_consolidated.parquet')
DEFAULT_OUT_DIR = Path('/mnt/v/zen/zensim-training/2026-05-18-larger-large')

# Exact constants derived empirically from cvvdp_iwssim_large_300col_v2.parquet
LO_CV = -2.118800
HI_CV = 13.815500
SLOPE_IW = 7.2837
INT_IW = 0.0302


def read_iwssim_sidecars(new_sidecars_dir: Path) -> pa.Table:
    parts = []
    if EXISTING_IWSSIM.exists():
        t = pq.read_table(str(EXISTING_IWSSIM))
        print(f"  existing iwssim: {t.num_rows} rows from {EXISTING_IWSSIM.name}")
        parts.append(t)
    if new_sidecars_dir.exists():
        files = sorted(new_sidecars_dir.glob('*.parquet'))
        print(f"  new sidecars: {len(files)} files in {new_sidecars_dir}")
        for fp in files:
            try:
                t = pq.read_table(str(fp))
                parts.append(t)
            except Exception as e:
                print(f"    WARN read {fp.name}: {e}", file=sys.stderr)
    combined = pa.concat_tables(parts, promote_options='default')
    print(f"  total iwssim rows: {combined.num_rows}")
    return combined


def cvvdp_log_norm_arr(cv_arr: np.ndarray) -> np.ndarray:
    raw = -np.log(10.0 - cv_arr + 1e-6)
    return (raw - LO_CV) / (HI_CV - LO_CV) * 100.0


def iwssim_log_norm_arr(iw_arr: np.ndarray) -> np.ndarray:
    iw_c = np.clip(iw_arr, 0, 1.0 - 1e-9)
    raw = -np.log(1.0 - iw_c + 1e-6)
    ln = SLOPE_IW * raw + INT_IW
    return np.clip(ln, 0, 100.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--new-sidecars-dir', required=True, type=Path,
                   help='Directory containing the new chunk sidecar parquets')
    p.add_argument('--out', type=Path,
                   default=DEFAULT_OUT_DIR / 'cvvdp_iwssim_LARGE_v3_300col.parquet',
                   help='Output parquet path')
    args = p.parse_args()

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== build_larger_large.py ===")
    print(f"new sidecars dir: {args.new_sidecars_dir}")
    print(f"output: {out_path}")
    print()

    print("Step 1: Load iwssim sidecars")
    iw = read_iwssim_sidecars(args.new_sidecars_dir)
    # Add basename column
    paths = iw['image_path'].to_pylist()
    iw_bn = [os.path.basename(p) if p else '' for p in paths]
    iw = iw.append_column('basename', pa.array(iw_bn, type=pa.string()))

    # Dedupe by (basename, codec, q, knob)
    iw_keys = [f"{b}|{c}|{q}|{k}"
               for b, c, q, k in zip(iw_bn, iw['codec'].to_pylist(),
                                     iw['q'].to_pylist(), iw['knob_tuple_json'].to_pylist())]
    seen = {}
    keep_idx = []
    for i, k in enumerate(iw_keys):
        if k in seen:
            continue
        seen[k] = i
        keep_idx.append(i)
    if len(keep_idx) < iw.num_rows:
        print(f"  deduped {iw.num_rows - len(keep_idx)} duplicate iwssim rows")
        iw = iw.take(keep_idx)
        iw_keys = [iw_keys[i] for i in keep_idx]
    print(f"  iwssim final: {iw.num_rows} rows")

    iw_lookup = dict(zip(iw_keys, iw['iwssim_imazen_v0_0_1'].to_pylist()))

    print("\nStep 2: Stream-join unified cvvdp+features files")
    out_chunks = []
    total_joined = 0
    for uf in sorted(UNIFIED_DIR.glob('unified_*_cvvdp.parquet')):
        print(f"  reading {uf.name}...")
        u = pq.read_table(str(uf))
        u_paths = u['image_path'].to_pylist()
        u_bn = [os.path.basename(p) for p in u_paths]
        u_keys = [f"{b}|{c}|{q}|{k}"
                  for b, c, q, k in zip(u_bn, u['codec'].to_pylist(),
                                        u['q'].to_pylist(), u['knob_tuple_json'].to_pylist())]
        keep_mask = np.array([k in iw_lookup for k in u_keys])
        n_keep = int(keep_mask.sum())
        if n_keep == 0:
            print(f"    no overlap — skip ({u.num_rows} rows)")
            continue
        u_keep = u.filter(pa.array(keep_mask))
        u_keep_paths = [u_bn[i] for i in range(len(u_bn)) if keep_mask[i]]
        u_keep_codec = [u['codec'][i].as_py() for i in range(len(u_bn)) if keep_mask[i]]
        u_keep_q = [u['q'][i].as_py() for i in range(len(u_bn)) if keep_mask[i]]
        u_keep_knob = [u['knob_tuple_json'][i].as_py() for i in range(len(u_bn)) if keep_mask[i]]
        u_keep_keys = [f"{b}|{c}|{q}|{k}"
                       for b, c, q, k in zip(u_keep_paths, u_keep_codec, u_keep_q, u_keep_knob)]
        iwssim_vals = np.array([iw_lookup[k] for k in u_keep_keys], dtype=np.float64)

        cv_arr = np.array(u_keep['cvvdp_imazen_v0_0_1'].to_pylist(), dtype=np.float64)
        cv_log_norm = cvvdp_log_norm_arr(cv_arr)
        iw_log_norm = iwssim_log_norm_arr(iwssim_vals)
        mix = 0.4 * cv_log_norm + 0.6 * iw_log_norm

        out_cols = {
            'ref_basename': pa.array(u_keep_paths, type=pa.string()),
            'human_score': pa.array(mix),
            'cvvdp_score': pa.array(cv_arr),
            'cvvdp_log_norm': pa.array(cv_log_norm),
            'iwssim': pa.array(iwssim_vals),
            'iwssim_log_norm': pa.array(iw_log_norm),
            'mix_cv40_iw60': pa.array(mix),
        }
        # Append 300 features
        for i in range(300):
            src = f'feat_{i}'
            dst = f'f{i}'
            if src in u_keep.schema.names:
                out_cols[dst] = u_keep[src].cast(pa.float64())
            else:
                out_cols[dst] = pa.array([None] * n_keep, type=pa.float64())

        out_tbl = pa.table(out_cols)
        out_chunks.append(out_tbl)
        total_joined += n_keep
        print(f"    joined chunk: {n_keep} rows × {len(out_tbl.schema.names)} cols (cum {total_joined})")

    if not out_chunks:
        print("ERROR: no joined output", file=sys.stderr)
        sys.exit(1)

    print(f"\nStep 3: Concat all joined chunks → {out_path}")
    full = pa.concat_tables(out_chunks, promote_options='default')
    print(f"  final: {full.num_rows} rows × {len(full.schema.names)} cols")
    pq.write_table(full, str(out_path), compression='zstd', compression_level=15)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nDONE. wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
