#!/usr/bin/env python3
"""V11-SUBSTRATE-V2 (task #190, 2026-05-20).

Rebuild V11 ssim2-anchored substrate using the R2 omni-multi-codec
sidecars (omni-multi-codec-2026-05-19 + cvvdp-v15rc-2026-05-18).

## The previous V11 agent's mistake

Claimed "ssim2 was never computed on zenwebp/zenavif/zenjxl" — this
was wrong. The local unified parquets at
`/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/` were missing
ssim2 for non-zenjpeg codecs, but the R2 omni sidecars DO have
score_ssim2_gpu populated 100% across all 4 codecs (verified by
inspecting the multi-codec parquets:
  zenwebp 1000/1000 nn, zenavif 4000/4000 nn,
  zenjxl 51200/51200 nn, zenjpeg 61600/61600 nn).

## Strategy (no need for cvvdp→ssim2 conversion)

1. **Anchor parquet (Phase 2)**: join omni + features per chunk on
   (image_path, codec, q, knob_tuple_json), keeping image_path,
   codec, q, knob_tuple_json, score_ssim2_gpu, score_cvvdp_imazen_v0_0_1,
   score_butteraugli_pnorm3_gpu, feat_0..feat_299. For each
   (image, codec, ssim2_band), find q whose ssim2 is closest to
   band's ssim2_target. Skip if dist > ±3.

2. **Cross-codec equivalence (Phase 2)**: all 4 codecs share the
   same `gen-*` image corpus in the omni-multi-codec run. For each
   image × each ssim2 reference level, find matching q per codec.
   Emit (codec_a, codec_b) pairs for codec_a < codec_b.

3. **Schema**: matches V8/V10 anchor parquet schema (V5 multi-band
   with `target_score` column, `f0..f<N-1>` features, optional
   per-row `target_score`).

## Inputs (verified pulled from R2 2026-05-20)

- Multi-codec omni: 365 parquets, 117,800 rows total
  (4000 zenavif, 61600 zenjpeg, 51200 zenjxl, 1000 zenwebp)
- Multi-codec features: 365 parquets, 117,800 rows total
- v15rc-jpeg omni: 2568 parquets, ~513,570 zenjpeg rows
- v15rc-jpeg features: 2568 parquets, ~513,570 zenjpeg rows

## Output

`/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/`
  - `unified_omni.parquet` — joined omni+features for all 4 codecs.
  - `anchors_ssim2_300col_v2.parquet` — V11 anchor parquet, V5 multi-band.
  - `cross_codec_equivalence_ssim2_v2.parquet` — V11 equiv parquet.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# (ssim2_target, V11_target_score). 10 bands per task brief.
ANCHOR_BANDS_V11_V2: list[tuple[float, float]] = [
    (100.0, 100.0),
    (95.0, 95.0),
    (90.0, 90.0),
    (75.0, 80.0),  # JND
    (60.0, 65.0),
    (45.0, 50.0),  # JOD
    (30.0, 35.0),
    (18.0, 20.0),
    (10.0, 10.0),
    (3.0, 0.0),
]

OMNI_KEY_COLS = ["image_path", "codec", "q", "knob_tuple_json"]
OMNI_METRIC_COLS = [
    "score_ssim2_gpu",
    "score_cvvdp_imazen_v0_0_1",
    "score_butteraugli_pnorm3_gpu",
]
FEATURE_COLS_300 = [f"feat_{i}" for i in range(300)]


def list_chunk_parquets(dirpath: Path) -> list[Path]:
    return sorted(dirpath.glob("*.parquet"))


def join_chunk(omni_path: Path, feat_path: Path, n_features: int) -> pd.DataFrame:
    """Load one chunk pair, join on (image_path, codec, q, knob_tuple_json)."""
    omni_cols = OMNI_KEY_COLS + OMNI_METRIC_COLS
    feat_cols = OMNI_KEY_COLS + [f"feat_{i}" for i in range(n_features)]
    omni = pq.read_table(omni_path, columns=omni_cols).to_pandas()
    feat = pq.read_table(feat_path, columns=feat_cols).to_pandas()
    return omni.merge(feat, on=OMNI_KEY_COLS, how="inner")


def build_unified_omni(
    multi_codec_omni: Path,
    multi_codec_feat: Path,
    v15rc_omni: Path | None,
    v15rc_feat: Path | None,
    out_path: Path,
    n_features: int,
) -> pd.DataFrame:
    """Concatenate joined chunks from all R2 prefixes into one parquet."""
    print(f"=== building unified omni parquet ===")
    print(f"  out: {out_path}")
    chunks = []

    multi_omni_files = list_chunk_parquets(multi_codec_omni)
    print(f"  multi-codec: {len(multi_omni_files)} chunks")
    for i, omni_p in enumerate(multi_omni_files):
        feat_p = multi_codec_feat / omni_p.name
        if not feat_p.exists():
            print(f"    skip {omni_p.name}: features missing")
            continue
        try:
            df = join_chunk(omni_p, feat_p, n_features)
        except Exception as e:
            print(f"    error {omni_p.name}: {e}")
            continue
        chunks.append(df)
        if (i + 1) % 50 == 0:
            print(f"    ...{i+1}/{len(multi_omni_files)} done, rows so far={sum(len(c) for c in chunks)}")

    if v15rc_omni is not None and v15rc_feat is not None:
        v15rc_omni_files = list_chunk_parquets(v15rc_omni)
        print(f"  v15rc-jpeg: {len(v15rc_omni_files)} chunks")
        for i, omni_p in enumerate(v15rc_omni_files):
            feat_p = v15rc_feat / omni_p.name
            if not feat_p.exists():
                continue
            try:
                df = join_chunk(omni_p, feat_p, n_features)
            except Exception:
                continue
            chunks.append(df)
            if (i + 1) % 500 == 0:
                print(f"    ...{i+1}/{len(v15rc_omni_files)} done, rows so far={sum(len(c) for c in chunks)}")

    if not chunks:
        raise SystemExit("no chunks loaded")

    print(f"  concatenating {len(chunks)} chunks...")
    df = pd.concat(chunks, ignore_index=True)
    df["ref_basename"] = df["image_path"].apply(os.path.basename)

    # Coverage audit
    print(f"  unified rows: {len(df)}")
    print(f"  unique codecs: {sorted(df['codec'].unique().tolist())}")
    print(f"  per-codec coverage:")
    for codec, g in df.groupby("codec"):
        ssim2_nn = g["score_ssim2_gpu"].notna().sum()
        cvvdp_nn = g["score_cvvdp_imazen_v0_0_1"].notna().sum()
        q_levels = sorted(g["q"].unique().tolist())
        imgs = g["ref_basename"].nunique()
        nrows = len(g)
        ssim2_pct = 100 * ssim2_nn / nrows
        cvvdp_pct = 100 * cvvdp_nn / nrows
        print(
            f"    {codec:>8s}: rows={nrows:>7d} imgs={imgs:>5d} "
            f"q_levels={q_levels[:6]}{'...' if len(q_levels) > 6 else ''} "
            f"ssim2_nn={ssim2_nn:>7d} ({ssim2_pct:5.1f}%) "
            f"cvvdp_nn={cvvdp_nn:>7d} ({cvvdp_pct:5.1f}%)"
        )

    # Write unified parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, out_path, compression="zstd", compression_level=10)
    print(f"  wrote {out_path} ({out_path.stat().st_size / (1024*1024):.0f} MiB)")
    return df


def build_anchor_rows(
    df: pd.DataFrame,
    bands: list[tuple[float, float]],
    n_features: int,
    tolerance: float = 3.0,
) -> pd.DataFrame:
    """For each (image, codec, band), find q with closest ssim2."""
    print(f"=== Phase 2: building anchor rows (ssim2 tol=±{tolerance}) ===")
    df_valid = df[df["score_ssim2_gpu"].notna()].copy()
    print(f"  valid rows: {len(df_valid)}")

    rows = []
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    per_band_stats = {b[0]: {"emit": 0, "filt": 0, "tot_d": 0.0, "target_score": b[1]} for b in bands}
    per_codec_band_stats = {}

    # Group by (ref_basename, codec)
    for (ref, codec), group in df_valid.groupby(["ref_basename", "codec"], sort=False):
        for ssim2_target, target_score in bands:
            dist = (group["score_ssim2_gpu"] - ssim2_target).abs()
            idx = dist.idxmin()
            bd = float(dist.loc[idx])
            key = (codec, ssim2_target)
            per_codec_band_stats.setdefault(
                key, {"emit": 0, "filt": 0, "tot_d": 0.0, "target_score": target_score}
            )
            if bd > tolerance:
                per_band_stats[ssim2_target]["filt"] += 1
                per_codec_band_stats[key]["filt"] += 1
                continue
            per_band_stats[ssim2_target]["emit"] += 1
            per_band_stats[ssim2_target]["tot_d"] += bd
            per_codec_band_stats[key]["emit"] += 1
            per_codec_band_stats[key]["tot_d"] += bd
            best = group.loc[idx]

            row = {
                "ref_basename": str(ref),
                "anchor_source": f"v11v2_{codec}_s{ssim2_target:.0f}_t{target_score:.0f}",
                "human_score": float(target_score),
                "anchor_weight": 1.0,
                "q": int(best["q"]),
                "ssim2_anchor": float(best["score_ssim2_gpu"]),
                "ssim2_target": float(ssim2_target),
                "cvvdp_anchor": (
                    float(best["score_cvvdp_imazen_v0_0_1"])
                    if pd.notna(best["score_cvvdp_imazen_v0_0_1"])
                    else float("nan")
                ),
                "butter_pnorm3_anchor": (
                    float(best["score_butteraugli_pnorm3_gpu"])
                    if pd.notna(best["score_butteraugli_pnorm3_gpu"])
                    else float("nan")
                ),
                "target_score": float(target_score),
                "codec": str(codec),
                "anchor_via": "ssim2_direct",
            }
            for c in feature_cols:
                v = best[c]
                row[c] = float(v) if pd.notna(v) else 0.0
            rows.append(row)

    # Print per-band summary
    print("  per-band summary:")
    for ssim2_target, _ in bands:
        s = per_band_stats[ssim2_target]
        tot = s["emit"] + s["filt"]
        mean_d = s["tot_d"] / s["emit"] if s["emit"] > 0 else 0.0
        pct = 100 * s["filt"] / tot if tot > 0 else 0
        print(
            f"    ssim2={ssim2_target:5.0f} → target={s['target_score']:5.0f}: "
            f"emit={s['emit']:>6d} filt={s['filt']:>5d} ({pct:4.1f}%) "
            f"mean_d={mean_d:.2f}"
        )

    print("  per-codec×band emit counts:")
    codecs = sorted(set(k[0] for k in per_codec_band_stats.keys()))
    header = f"    band/codec        " + "  ".join(f"{c:>10s}" for c in codecs)
    print(header)
    for ssim2_target, target_score in bands:
        row_str = f"    ssim2={ssim2_target:5.0f}→t={target_score:5.0f}  "
        for codec in codecs:
            s = per_codec_band_stats.get((codec, ssim2_target), {"emit": 0})
            row_str += f"  {s['emit']:>10d}"
        print(row_str)

    return pd.DataFrame(rows)


def write_anchor_parquet(df_rows: pd.DataFrame, out_path: Path, n_features: int) -> None:
    """Write anchor parquet with V5 schema (target_score column)."""
    print(f"=== writing anchor parquet: {out_path} ===")
    feature_cols = [f"f{i}" for i in range(n_features)]
    # rename feat_* → f*
    rename_map = {f"feat_{i}": f"f{i}" for i in range(n_features)}
    df_rows = df_rows.rename(columns=rename_map)

    field_specs = [
        ("ref_basename", pa.string()),
        ("anchor_source", pa.string()),
        ("human_score", pa.float64()),
        ("anchor_weight", pa.float64()),
        ("q", pa.int64()),
        ("ssim2_anchor", pa.float64()),
        ("ssim2_target", pa.float64()),
        ("cvvdp_anchor", pa.float64()),
        ("butter_pnorm3_anchor", pa.float64()),
        ("target_score", pa.float64()),
        ("codec", pa.string()),
        ("anchor_via", pa.string()),
    ]
    cols: dict[str, pa.Array] = {}
    for name, tp in field_specs:
        cols[name] = pa.array(df_rows[name].values, type=tp)
    for c in feature_cols:
        cols[c] = pa.array(df_rows[c].astype("float32").values, type=pa.float32())

    tbl = pa.table(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out_path, compression="zstd", compression_level=15)
    print(
        f"  wrote {tbl.num_rows} rows × {tbl.num_columns} cols, "
        f"{out_path.stat().st_size / (1024*1024):.1f} MiB"
    )


def build_cross_codec_equivalence(
    df: pd.DataFrame,
    pivot_levels: list[float],
    tolerance: float,
    n_features: int,
) -> pd.DataFrame:
    """Cross-codec equivalence pairs pivoted on ssim2.

    For each (image, ssim2_L) and each ordered codec pair (a, b)
    (a < b alphabetically), find the q per codec with ssim2 closest
    to L. Emit pair if both within tolerance and the codecs are
    different.
    """
    print(f"=== Phase 3: building cross-codec equivalence (ssim2 levels={pivot_levels}, tol=±{tolerance}) ===")
    df_valid = df[df["score_ssim2_gpu"].notna()].copy()
    print(f"  valid rows: {len(df_valid)}")
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    # Group by ref_basename, then per codec pick best q for each level
    pairs = []
    pair_count = {}
    images = sorted(df_valid["ref_basename"].unique())
    print(f"  unique images: {len(images)}")

    for i_img, ref in enumerate(images):
        if (i_img + 1) % 500 == 0:
            print(f"    img {i_img+1}/{len(images)} pairs={len(pairs)}")
        sub = df_valid[df_valid["ref_basename"] == ref]
        codecs_in_img = sorted(sub["codec"].unique())
        if len(codecs_in_img) < 2:
            continue
        per_codec_picks: dict[str, dict[float, dict]] = {}
        for codec in codecs_in_img:
            csub = sub[sub["codec"] == codec]
            per_codec_picks[codec] = {}
            for L in pivot_levels:
                dist = (csub["score_ssim2_gpu"] - L).abs()
                idx = dist.idxmin()
                bd = float(dist.loc[idx])
                if bd > tolerance:
                    continue
                best = csub.loc[idx]
                per_codec_picks[codec][L] = {
                    "q": int(best["q"]),
                    "ssim2": float(best["score_ssim2_gpu"]),
                    "cvvdp": float(best["score_cvvdp_imazen_v0_0_1"])
                    if pd.notna(best["score_cvvdp_imazen_v0_0_1"])
                    else float("nan"),
                    "butter": float(best["score_butteraugli_pnorm3_gpu"])
                    if pd.notna(best["score_butteraugli_pnorm3_gpu"])
                    else float("nan"),
                    "feat": [
                        float(best[c]) if pd.notna(best[c]) else 0.0 for c in feature_cols
                    ],
                }

        # Emit ordered pairs at each L
        for L in pivot_levels:
            valid_codecs = [c for c in codecs_in_img if L in per_codec_picks[c]]
            for i in range(len(valid_codecs)):
                for j in range(i + 1, len(valid_codecs)):
                    ca = valid_codecs[i]
                    cb = valid_codecs[j]
                    a = per_codec_picks[ca][L]
                    b = per_codec_picks[cb][L]
                    ssim2_diff = abs(a["ssim2"] - b["ssim2"])
                    weight = float(1.0 / (ssim2_diff + 0.5))
                    if weight > 4.0:
                        weight = 4.0
                    pair_count[(ca, cb)] = pair_count.get((ca, cb), 0) + 1
                    pairs.append(
                        {
                            "ref_basename": ref,
                            "codec_a": ca,
                            "q_a": a["q"],
                            "codec_b": cb,
                            "q_b": b["q"],
                            "ssim2_level": L,
                            "ssim2_a": a["ssim2"],
                            "ssim2_b": b["ssim2"],
                            "ssim2_diff": ssim2_diff,
                            "cvvdp_a": a["cvvdp"],
                            "cvvdp_b": b["cvvdp"],
                            "butter_pnorm3_a": a["butter"],
                            "butter_pnorm3_b": b["butter"],
                            "row_weight": weight,
                            "butter_level": L,  # alias for V8 schema compat
                            "butter_a": a["ssim2"],  # use ssim2 as level
                            "butter_b": b["ssim2"],
                            "fa": a["feat"],
                            "fb": b["feat"],
                        }
                    )

    print(f"  built {len(pairs)} equivalence pairs")
    print(f"  pair counts (sorted):")
    for (ca, cb), c in sorted(pair_count.items()):
        print(f"    {ca:>10s} ↔ {cb:<10s}  {c:>6d}")
    return pd.DataFrame(pairs)


def write_equivalence_parquet(df_pairs: pd.DataFrame, out_path: Path, n_features: int) -> None:
    """Write equivalence parquet matching V8 schema."""
    print(f"=== writing equivalence parquet: {out_path} ===")
    if df_pairs.empty:
        raise SystemExit("no equivalence pairs to write")

    fields = [
        ("ref_basename", pa.string()),
        ("codec_a", pa.string()),
        ("q_a", pa.int64()),
        ("codec_b", pa.string()),
        ("q_b", pa.int64()),
        ("butter_level", pa.float64()),
        ("butter_a", pa.float64()),
        ("butter_b", pa.float64()),
        ("ssim2_level", pa.float64()),
        ("ssim2_a", pa.float64()),
        ("ssim2_b", pa.float64()),
        ("ssim2_diff", pa.float64()),
        ("cvvdp_a", pa.float64()),
        ("cvvdp_b", pa.float64()),
        ("butter_pnorm3_a", pa.float64()),
        ("butter_pnorm3_b", pa.float64()),
        ("row_weight", pa.float64()),
    ]
    cols = {}
    for name, tp in fields:
        cols[name] = pa.array(df_pairs[name].values, type=tp)

    fa = np.stack(df_pairs["fa"].to_numpy(), axis=0)
    fb = np.stack(df_pairs["fb"].to_numpy(), axis=0)
    for i in range(n_features):
        cols[f"fa_{i}"] = pa.array(fa[:, i].astype("float32"), type=pa.float32())
    for i in range(n_features):
        cols[f"fb_{i}"] = pa.array(fb[:, i].astype("float32"), type=pa.float32())

    tbl = pa.table(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out_path, compression="zstd", compression_level=15)
    print(
        f"  wrote {tbl.num_rows} rows × {tbl.num_columns} cols, "
        f"{out_path.stat().st_size / (1024*1024):.1f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi-codec-omni",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-r2-omni/multi-codec/omni"),
    )
    parser.add_argument(
        "--multi-codec-feat",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-r2-omni/multi-codec/zensim_features"),
    )
    parser.add_argument(
        "--v15rc-omni",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-r2-omni/v15rc-jpeg/omni"),
    )
    parser.add_argument(
        "--v15rc-feat",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-r2-omni/v15rc-jpeg/zensim_features"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v11-substrate"),
    )
    parser.add_argument(
        "--n-features", type=int, default=300, help="(default 300, R2 omni schema)"
    )
    parser.add_argument(
        "--ssim2-tolerance", type=float, default=3.0,
        help="anchor band tolerance in ssim2 units (default ±3)",
    )
    parser.add_argument(
        "--ssim2-eq-tolerance", type=float, default=3.0,
        help="cross-codec equivalence tolerance in ssim2 units",
    )
    parser.add_argument(
        "--skip-unified", action="store_true",
        help="if --unified-omni exists, skip rebuild and load it",
    )
    parser.add_argument(
        "--unified-omni", type=Path, default=None,
        help="path to pre-built unified omni parquet",
    )
    parser.add_argument(
        "--include-v15rc-jpeg", action="store_true",
        help="if set, also include v15rc-jpeg in the unified parquet "
             "(adds 513k zenjpeg rows with 19 q levels). Default: off "
             "for substrate v2 since multi-codec already gives 5q "
             "across all 4 codecs",
    )
    parser.add_argument(
        "--skip-equivalence", action="store_true",
        help="skip Phase 3",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    unified_path = args.unified_omni or (args.out_dir / "unified_omni.parquet")

    if args.skip_unified and unified_path.exists():
        print(f"loading pre-built unified omni from {unified_path}")
        df = pq.read_table(unified_path).to_pandas()
        if "ref_basename" not in df.columns:
            df["ref_basename"] = df["image_path"].apply(os.path.basename)
    else:
        df = build_unified_omni(
            args.multi_codec_omni,
            args.multi_codec_feat,
            args.v15rc_omni if args.include_v15rc_jpeg else None,
            args.v15rc_feat if args.include_v15rc_jpeg else None,
            unified_path,
            args.n_features,
        )

    # Phase 2: anchor rows
    print()
    df_anchors = build_anchor_rows(
        df, ANCHOR_BANDS_V11_V2, args.n_features, tolerance=args.ssim2_tolerance
    )
    write_anchor_parquet(
        df_anchors,
        args.out_dir / "anchors_ssim2_300col_v2.parquet",
        args.n_features,
    )

    # Phase 3: cross-codec equivalence pairs
    if not args.skip_equivalence:
        print()
        # Use the 6 mid-range bands for equivalence: 90, 75 (JND), 60, 45 (JOD), 30, 18.
        pivot_levels = [90.0, 75.0, 60.0, 45.0, 30.0, 18.0]
        df_pairs = build_cross_codec_equivalence(
            df, pivot_levels, args.ssim2_eq_tolerance, args.n_features
        )
        write_equivalence_parquet(
            df_pairs,
            args.out_dir / "cross_codec_equivalence_ssim2_v2.parquet",
            args.n_features,
        )


if __name__ == "__main__":
    main()
