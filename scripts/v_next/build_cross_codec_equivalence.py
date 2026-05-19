#!/usr/bin/env python3
"""Build cross-codec equivalence pair parquet from per-codec butter sweeps.

Input: per-codec parquets from cross_codec_butter_features:
    /mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet
    Each row: (ref_basename, codec, q, butter_max, butter_pnorm3,
              encoded_bytes, width, height, f0..f371)

For each source, define K butter levels (default 20 spaced from 0 to 30 in
butter_pnorm3 units). For each level L:
  - For each codec C in {zenjpeg, zenwebp, zenavif, zenjxl}: pick the q
    whose butter_pnorm3(S, C, q) is closest to L.
  - Generate C(4,2) = 6 equivalence pairs per level (all pairs of codecs).

Output: a single parquet at out_path with columns:
    ref_basename, codec_a, q_a, codec_b, q_b, butter_level,
    butter_a, butter_b, row_weight, fa_0..fa_371, fb_0..fb_371

The row_weight is 1.0 / abs(butter_a - butter_b + epsilon), capped at 1.0,
so pairs that are TIGHT cross-codec equivalents pull harder than loose ones.

Filter: only keep pairs where both codecs have valid features AND butter
values, AND |butter_a - butter_b| < 0.5 (within-level tolerance).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CODECS_DEFAULT = ["zenjpeg", "zenwebp", "zenavif", "zenjxl"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--butter-dir",
        default="/mnt/v/zen/picker-training/2026-05-19/butter",
        type=Path,
        help="directory containing per-codec butter+features parquets",
    )
    ap.add_argument(
        "--out",
        default="/mnt/v/zen/picker-training/2026-05-19/cross_codec_equivalence.parquet",
        type=Path,
    )
    ap.add_argument(
        "--n-levels",
        type=int,
        default=20,
        help="number of butter levels per source (default 20)",
    )
    ap.add_argument(
        "--butter-low",
        type=float,
        default=0.5,
        help="low end of butter level grid (skip near-lossless; default 0.5)",
    )
    ap.add_argument(
        "--butter-high",
        type=float,
        default=12.0,
        help="high end of butter level grid (cap heavy distortion; default 12.0)",
    )
    ap.add_argument(
        "--max-pair-gap",
        type=float,
        default=0.5,
        help="reject equivalence pair if |butter_a - butter_b| > this (default 0.5)",
    )
    ap.add_argument(
        "--metric-col",
        default="butter_pnorm3",
        choices=["butter_max", "butter_pnorm3"],
        help="butter aggregation to use as pivot (default pnorm3)",
    )
    ap.add_argument(
        "--max-sources",
        type=int,
        default=0,
        help="0 = all sources (default)",
    )
    ap.add_argument(
        "--codecs",
        default=",".join(CODECS_DEFAULT),
        help="comma-separated codec list (default: zenjpeg,zenwebp,zenavif,zenjxl)",
    )
    return ap.parse_args()


def load_codec_parquet(path):
    print(f"  loading {path}")
    t = pq.read_table(path)
    print(f"    rows={t.num_rows}, cols={len(t.column_names)}")
    df = t.to_pandas()
    return df


def main():
    args = parse_args()
    print(f"build_cross_codec_equivalence")
    print(f"  butter_dir: {args.butter_dir}")
    print(f"  out:        {args.out}")
    print(f"  n_levels:   {args.n_levels}")
    print(f"  metric:     {args.metric_col}")
    print(f"  pair_gap:   <={args.max_pair_gap}")

    codecs_list = [c.strip() for c in args.codecs.split(",") if c.strip()]
    print(f"  codecs: {codecs_list}")
    dfs = {}
    for codec in codecs_list:
        path = args.butter_dir / f"{codec}.parquet"
        if not path.exists():
            print(f"  ERROR: missing {path}", file=sys.stderr)
            sys.exit(2)
        dfs[codec] = load_codec_parquet(path)

    # Common source basenames present in all 4 codecs.
    common = None
    for codec, df in dfs.items():
        s = set(df["ref_basename"].unique())
        if common is None:
            common = s
        else:
            common &= s
    sources = sorted(common)
    if args.max_sources > 0:
        sources = sources[: args.max_sources]
    print(f"  common sources: {len(sources)}")

    butter_levels = np.linspace(args.butter_low, args.butter_high, args.n_levels)
    print(f"  butter levels: {butter_levels.tolist()}")

    # Build per-(source, codec) DataFrame index.
    by_codec = {codec: df.set_index("ref_basename", drop=False) for codec, df in dfs.items()}

    rows_basename = []
    rows_codec_a = []
    rows_q_a = []
    rows_codec_b = []
    rows_q_b = []
    rows_level = []
    rows_butter_a = []
    rows_butter_b = []
    rows_weight = []
    rows_fa = []  # list of np arrays len 372
    rows_fb = []

    fcols = [f"f{i}" for i in range(372)]

    for src_i, basename in enumerate(sources):
        if src_i % 100 == 0:
            print(f"    source {src_i}/{len(sources)}")
        per_codec = {}
        skip = False
        for codec in codecs_list:
            df = by_codec[codec]
            if basename not in df.index:
                skip = True
                break
            rows = df.loc[basename]
            # rows may be a Series (single q) or DataFrame (multi q)
            if hasattr(rows, "to_frame") and not isinstance(rows, type(df)):
                rows = rows.to_frame().T
            per_codec[codec] = rows
        if skip:
            continue

        # For each butter level, find nearest q per codec.
        picks = {}  # codec -> (q, butter, features f0..f371)
        for L in butter_levels:
            cell_picks = {}
            for codec in codecs_list:
                sub = per_codec[codec]
                vals = sub[args.metric_col].to_numpy()
                valid_mask = np.isfinite(vals)
                if not valid_mask.any():
                    cell_picks[codec] = None
                    continue
                vsub = vals[valid_mask]
                qsub = sub["q"].to_numpy()[valid_mask]
                # Pull features (may be NaN if feature extraction failed)
                feat_arr = sub[fcols].to_numpy(dtype=np.float32)[valid_mask]
                # Skip rows whose features are all NaN.
                feat_ok = ~np.isnan(feat_arr).all(axis=1)
                if not feat_ok.any():
                    cell_picks[codec] = None
                    continue
                vsub = vsub[feat_ok]
                qsub = qsub[feat_ok]
                feat_arr = feat_arr[feat_ok]
                # Nearest butter to L
                idx = int(np.argmin(np.abs(vsub - L)))
                cell_picks[codec] = (int(qsub[idx]), float(vsub[idx]), feat_arr[idx])
            picks[L] = cell_picks

        for L, codec_picks in picks.items():
            valid_codecs = [c for c in codecs_list if codec_picks.get(c) is not None]
            if len(valid_codecs) < 2:
                continue
            for i in range(len(valid_codecs)):
                for j in range(i + 1, len(valid_codecs)):
                    ca = valid_codecs[i]
                    cb = valid_codecs[j]
                    qa, ba, fa = codec_picks[ca]
                    qb, bb, fb = codec_picks[cb]
                    gap = abs(ba - bb)
                    if gap > args.max_pair_gap:
                        continue
                    # Weight: pairs that are TIGHTLY matched get higher weight.
                    weight = float(1.0 / (gap + 0.05))
                    if weight > 20.0:
                        weight = 20.0
                    rows_basename.append(basename)
                    rows_codec_a.append(ca)
                    rows_q_a.append(qa)
                    rows_codec_b.append(cb)
                    rows_q_b.append(qb)
                    rows_level.append(float(L))
                    rows_butter_a.append(ba)
                    rows_butter_b.append(bb)
                    rows_weight.append(weight)
                    rows_fa.append(fa)
                    rows_fb.append(fb)

    print(f"  built {len(rows_basename)} equivalence pairs")
    if not rows_basename:
        print("  ERROR: no pairs survived filters", file=sys.stderr)
        sys.exit(3)

    # Build parquet.
    n = len(rows_basename)
    fields = [
        pa.field("ref_basename", pa.string()),
        pa.field("codec_a", pa.string()),
        pa.field("q_a", pa.int64()),
        pa.field("codec_b", pa.string()),
        pa.field("q_b", pa.int64()),
        pa.field("butter_level", pa.float64()),
        pa.field("butter_a", pa.float64()),
        pa.field("butter_b", pa.float64()),
        pa.field("row_weight", pa.float64()),
    ]
    for prefix in ["fa", "fb"]:
        for i in range(372):
            fields.append(pa.field(f"{prefix}_{i}", pa.float32()))

    arrays = [
        pa.array(rows_basename, type=pa.string()),
        pa.array(rows_codec_a, type=pa.string()),
        pa.array(rows_q_a, type=pa.int64()),
        pa.array(rows_codec_b, type=pa.string()),
        pa.array(rows_q_b, type=pa.int64()),
        pa.array(rows_level, type=pa.float64()),
        pa.array(rows_butter_a, type=pa.float64()),
        pa.array(rows_butter_b, type=pa.float64()),
        pa.array(rows_weight, type=pa.float64()),
    ]
    fa_stack = np.stack(rows_fa, axis=0)  # (n, 372)
    fb_stack = np.stack(rows_fb, axis=0)
    for i in range(372):
        arrays.append(pa.array(fa_stack[:, i], type=pa.float32()))
    for i in range(372):
        arrays.append(pa.array(fb_stack[:, i], type=pa.float32()))

    schema = pa.schema(fields)
    table = pa.Table.from_arrays(arrays, schema=schema)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")
    print(f"  wrote {n} rows × {len(fields)} cols → {args.out}")

    # Quick stats
    print("  stats:")
    gaps = np.abs(np.array(rows_butter_a) - np.array(rows_butter_b))
    print(f"    gap: mean={gaps.mean():.3f} p95={np.percentile(gaps, 95):.3f}")
    print(f"    weight: mean={np.mean(rows_weight):.2f} max={max(rows_weight):.2f}")
    # Distribution by codec pair
    from collections import Counter
    cpairs = Counter(
        tuple(sorted([a, b])) for a, b in zip(rows_codec_a, rows_codec_b)
    )
    print("    pair counts:")
    for p, c in sorted(cpairs.items()):
        print(f"      {p[0]:>8} ↔ {p[1]:<8}  {c:>6}")


if __name__ == "__main__":
    main()
