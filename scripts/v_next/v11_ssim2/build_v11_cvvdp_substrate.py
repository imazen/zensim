#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V11 cvvdp-anchored cross-validation substrate (2026-05-20).

Phase 3 of task #189: build a cvvdp-anchored anchor parquet using the same
procedure as the ssim2-anchored substrate (`build_v11_ssim2_substrate.py`)
but pivoted on cvvdp directly. This is for cross-validating that V11-A'
trained on ssim2-anchored data ALSO satisfies cross-codec consistency
when measured against cvvdp-anchored equivalence — i.e., the metric is
robust across anchor metrics, not memorizing ssim2 idiosyncrasies.

## cvvdp → target_score mapping

Per task brief: inspect the cvvdp distribution and pick a sensible
mapping. Empirical p1/p5/p25/p50/p75/p95/p99 from
canonical-2026-05-18/scores/cvvdp_imazen_v0_0_1.parquet (1.17 M rows
across 4 codecs):

| Percentile | cvvdp | Maps to target_score (proposed) |
|---|--:|---|
| p99 | 10.000 | 100 (lossless) |
| p95 |  9.996 | 95 |
| p75 |  9.939 | 90 |
| p50 |  9.844 | 80 (JND landing) |
| p25 |  9.652 | 65 |
| p10 |  9.377 | 50 (JOD) |
| p5  |  9.198 | 35 |
| p1  |  8.786 | 10 |
| p0  |  6.836 | 0 |

We anchor JND at cvvdp ≈ 9.84 (the p50 of all-corpus cvvdp). cvvdp
saturates near 10.0 for high-quality encodes, so the 95-100 bands
have very tight cvvdp tolerance. JOD is anchored at cvvdp ≈ 9.38 (the
p10) — about 1.5 cvvdp units below JND, consistent with the cvvdp
"quality scale" semantics where 1 JOD ≈ 1 unit.

The mapping uses cvvdp ABSOLUTE values for anchor bands (not
percentiles), so it's a direct lookup. The percentile origin is
documented for traceability.

## Coverage caveat (verified 2026-05-20)

cvvdp is non-null at 95-100% across all 4 codecs in the v12/v13 unified
parquets (200 imgs × 5 q-levels = 1k–32k rows/codec). zenjxl saturates
cvvdp at ≥ 9.98 across the full q range; bands below cvvdp = 9.7
will have zero rows for zenjxl. Same structural issue as ssim2.

## Output

`/mnt/v/zen/zensim-training/2026-05-20-cvvdp-anchors/anchors_cvvdp_300col.parquet`
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# (cvvdp_target, V11_target_score). 10 bands aligned to all-corpus percentiles.
ANCHOR_BANDS_CVVDP: list[tuple[float, float]] = [
    (10.000, 100.0),  # p99, lossless
    (9.996, 95.0),    # p95
    (9.939, 90.0),    # p75
    (9.84, 80.0),     # p50, JND
    (9.74, 65.0),
    (9.38, 50.0),     # p10, JOD
    (9.20, 35.0),     # p5
    (8.93, 20.0),
    (8.79, 10.0),     # p1
    (7.50, 0.0),      # well below p1, pathological
]

ZENJPEG_PRIMARY = Path(
    "/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_v13_zenjpeg_cvvdp.parquet"
)
PER_CODEC_OTHER = {
    "zenwebp": Path("/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_v12_zenwebp_cvvdp.parquet"),
    "zenavif": Path("/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_v12_zenavif_cvvdp.parquet"),
    "zenjxl": Path("/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_v12_zenjxl_cvvdp.parquet"),
}
DEFAULT_OUT_DIR = Path("/mnt/v/zen/zensim-training/2026-05-20-cvvdp-anchors")

CVVDP_ANCHOR_TOLERANCE_DEFAULT = 0.05
CVVDP_EQ_TOLERANCE_DEFAULT = 0.03


def load_codec(codec: str, path: Path, n_features: int) -> pd.DataFrame:
    print(f"loading {codec} {path}")
    t = pq.read_table(path)
    df = t.to_pandas()
    df["ref_basename"] = df["image_path"].apply(os.path.basename)
    df = df.rename(columns={
        "score_ssim2": "ssim2_anchor",
        "cvvdp_imazen_v0_0_1": "cvvdp_anchor",
    })
    for c in list(df.columns):
        if c.startswith("feat_"):
            n = c[len("feat_"):]
            df = df.rename(columns={c: f"f{n}"})
    print(f"  rows={len(df)} unique_imgs={df['ref_basename'].nunique()} "
          f"unique_q={sorted(df['q'].unique())}")
    if "ssim2_anchor" in df.columns:
        print(f"  ssim2 nn={df['ssim2_anchor'].notna().sum()} "
              f"cvvdp nn={df['cvvdp_anchor'].notna().sum()}")
    return df


def build_anchor_rows(
    codec: str,
    df_codec: pd.DataFrame,
    cvvdp_tolerance: float,
    n_features: int,
) -> tuple[list[dict], dict[float, dict[str, int]]]:
    """Build per-(image, band) anchor rows pivoted on cvvdp."""
    print(f"=== building {codec} anchor rows (cvvdp-direct) ===")
    rows: list[dict] = []
    stats: dict[float, dict[str, int]] = {}
    feature_cols = [f"f{i}" for i in range(n_features)]
    df_valid = df_codec[df_codec["cvvdp_anchor"].notna()].copy()
    print(f"  using {len(df_valid)} rows with cvvdp")

    for source, group in df_valid.groupby("ref_basename"):
        for cvvdp_target, target_score in ANCHOR_BANDS_CVVDP:
            stats.setdefault(cvvdp_target, {"emitted": 0, "filtered": 0,
                                              "tot_distance": 0.0})
            distances = (group["cvvdp_anchor"] - cvvdp_target).abs()
            idx = distances.idxmin()
            best = group.loc[idx]
            best_distance = float(distances.loc[idx])
            if best_distance > cvvdp_tolerance:
                stats[cvvdp_target]["filtered"] += 1
                continue
            stats[cvvdp_target]["emitted"] += 1
            stats[cvvdp_target]["tot_distance"] += best_distance
            row = {
                "ref_basename": str(source),
                "anchor_source": f"v11_{codec}_cv{cvvdp_target:.3f}_t{target_score:.0f}",
                "human_score": float(target_score),
                "anchor_weight": 1.0,
                "q": int(best["q"]),
                "cvvdp_anchor": float(best["cvvdp_anchor"]),
                "cvvdp_target": float(cvvdp_target),
                "ssim2_anchor": (
                    float(best["ssim2_anchor"]) if "ssim2_anchor" in best.index
                    and pd.notna(best["ssim2_anchor"]) else np.nan
                ),
                "target_score": float(target_score),
                "codec": codec,
            }
            for col in feature_cols:
                if col in best.index:
                    val = best[col]
                    row[col] = float(val) if pd.notna(val) else 0.0
                else:
                    row[col] = 0.0
            rows.append(row)
    return rows, stats


def write_parquet(rows: list[dict], out_path: Path, n_features: int) -> None:
    if not rows:
        raise SystemExit("no anchor rows")
    feature_cols = [f"f{i}" for i in range(n_features)]
    cols = {
        "ref_basename": pa.array([r["ref_basename"] for r in rows], type=pa.string()),
        "anchor_source": pa.array([r["anchor_source"] for r in rows], type=pa.string()),
        "human_score": pa.array([r["human_score"] for r in rows], type=pa.float64()),
        "anchor_weight": pa.array([r["anchor_weight"] for r in rows], type=pa.float64()),
        "q": pa.array([r["q"] for r in rows], type=pa.int64()),
        "cvvdp_anchor": pa.array([r["cvvdp_anchor"] for r in rows], type=pa.float64()),
        "cvvdp_target": pa.array([r["cvvdp_target"] for r in rows], type=pa.float64()),
        "ssim2_anchor": pa.array([r["ssim2_anchor"] for r in rows], type=pa.float64()),
        "target_score": pa.array([r["target_score"] for r in rows], type=pa.float64()),
        "codec": pa.array([r["codec"] for r in rows], type=pa.string()),
    }
    for col in feature_cols:
        cols[col] = pa.array([r[col] for r in rows], type=pa.float32())
    tbl = pa.table(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out_path, compression="zstd", compression_level=15)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB, "
          f"{tbl.num_rows} rows × {tbl.num_columns} cols)")


def build_cross_codec_equivalence_cvvdp(
    codec_dfs: dict[str, pd.DataFrame],
    cvvdp_tolerance: float,
    n_features: int,
) -> list[dict]:
    """Cross-codec equivalence pairs pivoted on cvvdp directly."""
    print("=== building cvvdp-pivoted cross-codec equivalence pairs ===")
    cvvdp_levels = [9.95, 9.84, 9.74, 9.50, 9.38, 9.20, 8.93, 8.79]
    print(f"  cvvdp pivot levels: {cvvdp_levels}")

    pairs: list[dict] = []
    feature_cols = [f"f{i}" for i in range(n_features)]
    codec_indexed = {}
    for codec, df in codec_dfs.items():
        codec_indexed[codec] = df[df["cvvdp_anchor"].notna()].copy()

    all_imgs = set()
    for d in codec_indexed.values():
        all_imgs |= set(d["ref_basename"].unique())
    all_imgs_sorted = sorted(all_imgs)
    print(f"  total imgs: {len(all_imgs_sorted)}")

    pair_counter = {}
    for img_i, ref in enumerate(all_imgs_sorted):
        if img_i % 50 == 0:
            print(f"    img {img_i}/{len(all_imgs_sorted)} pairs={len(pairs)}")
        per_codec_picks = {}
        for codec, df_c in codec_indexed.items():
            grp = df_c[df_c["ref_basename"] == ref]
            if len(grp) == 0:
                continue
            per_codec_picks[codec] = {}
            for L in cvvdp_levels:
                dist = (grp["cvvdp_anchor"] - L).abs()
                if len(dist) == 0:
                    continue
                idx = dist.idxmin()
                bd = float(dist.loc[idx])
                if bd > cvvdp_tolerance:
                    continue
                row = grp.loc[idx]
                per_codec_picks[codec][L] = {
                    "q": int(row["q"]),
                    "cvvdp": float(row["cvvdp_anchor"]),
                    "ssim2": (
                        float(row["ssim2_anchor"]) if "ssim2_anchor" in row.index
                        and pd.notna(row["ssim2_anchor"]) else None
                    ),
                    "feat": [
                        float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0
                        for c in feature_cols
                    ],
                }
        codecs_present = list(per_codec_picks.keys())
        for L in cvvdp_levels:
            valid = [c for c in codecs_present if L in per_codec_picks[c]]
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    ca, cb = valid[i], valid[j]
                    a, b = per_codec_picks[ca][L], per_codec_picks[cb][L]
                    cvvdp_diff = abs(a["cvvdp"] - b["cvvdp"])
                    weight = float(1.0 / (cvvdp_diff + 0.01))
                    if weight > 100.0:
                        weight = 100.0
                    pair_counter[(ca, cb)] = pair_counter.get((ca, cb), 0) + 1
                    pairs.append({
                        "ref_basename": ref,
                        "codec_a": ca, "q_a": a["q"],
                        "codec_b": cb, "q_b": b["q"],
                        "cvvdp_level": L,
                        "cvvdp_a": a["cvvdp"], "cvvdp_b": b["cvvdp"],
                        "cvvdp_diff": cvvdp_diff,
                        "ssim2_a": a["ssim2"] if a["ssim2"] is not None else np.nan,
                        "ssim2_b": b["ssim2"] if b["ssim2"] is not None else np.nan,
                        "row_weight": weight,
                        "fa": a["feat"], "fb": b["feat"],
                    })

    print(f"built {len(pairs)} cvvdp-equivalence pairs")
    for (ca, cb), c in sorted(pair_counter.items()):
        print(f"  {ca:>10s} ↔ {cb:<10s}  {c:6d}")
    return pairs


def write_eq_parquet(pairs: list[dict], out_path: Path, n_features: int) -> None:
    if not pairs:
        raise SystemExit("no pairs")
    fields = [
        pa.field("ref_basename", pa.string()),
        pa.field("codec_a", pa.string()), pa.field("q_a", pa.int64()),
        pa.field("codec_b", pa.string()), pa.field("q_b", pa.int64()),
        pa.field("cvvdp_level", pa.float64()),
        pa.field("cvvdp_a", pa.float64()), pa.field("cvvdp_b", pa.float64()),
        pa.field("cvvdp_diff", pa.float64()),
        pa.field("ssim2_a", pa.float64()), pa.field("ssim2_b", pa.float64()),
        pa.field("row_weight", pa.float64()),
    ]
    for prefix in ["fa", "fb"]:
        for i in range(n_features):
            fields.append(pa.field(f"{prefix}_{i}", pa.float32()))

    arrays = [
        pa.array([p["ref_basename"] for p in pairs], type=pa.string()),
        pa.array([p["codec_a"] for p in pairs], type=pa.string()),
        pa.array([p["q_a"] for p in pairs], type=pa.int64()),
        pa.array([p["codec_b"] for p in pairs], type=pa.string()),
        pa.array([p["q_b"] for p in pairs], type=pa.int64()),
        pa.array([p["cvvdp_level"] for p in pairs], type=pa.float64()),
        pa.array([p["cvvdp_a"] for p in pairs], type=pa.float64()),
        pa.array([p["cvvdp_b"] for p in pairs], type=pa.float64()),
        pa.array([p["cvvdp_diff"] for p in pairs], type=pa.float64()),
        pa.array([p["ssim2_a"] for p in pairs], type=pa.float64()),
        pa.array([p["ssim2_b"] for p in pairs], type=pa.float64()),
        pa.array([p["row_weight"] for p in pairs], type=pa.float64()),
    ]
    fa = np.stack([p["fa"] for p in pairs], axis=0)
    fb = np.stack([p["fb"] for p in pairs], axis=0)
    for i in range(n_features):
        arrays.append(pa.array(fa[:, i], type=pa.float32()))
    for i in range(n_features):
        arrays.append(pa.array(fb[:, i], type=pa.float32()))

    schema = pa.schema(fields)
    table = pa.Table.from_arrays(arrays, schema=schema)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd", compression_level=15)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB, "
          f"{table.num_rows} rows × {table.num_columns} cols)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cvvdp-tolerance", type=float,
                        default=CVVDP_ANCHOR_TOLERANCE_DEFAULT,
                        help="anchor cvvdp max distance (default 0.05)")
    parser.add_argument("--cvvdp-eq-tolerance", type=float,
                        default=CVVDP_EQ_TOLERANCE_DEFAULT,
                        help="equivalence cvvdp max distance (default 0.03)")
    parser.add_argument("--n-features", type=int, default=300)
    parser.add_argument("--skip-equivalence", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    codec_dfs = {"zenjpeg": load_codec("zenjpeg", ZENJPEG_PRIMARY, args.n_features)}
    for codec, path in PER_CODEC_OTHER.items():
        codec_dfs[codec] = load_codec(codec, path, args.n_features)

    print()
    print("=== PHASE 3: cvvdp anchor parquet ===")
    all_rows = []
    for codec, df in codec_dfs.items():
        codec_rows, stats = build_anchor_rows(
            codec, df, args.cvvdp_tolerance, args.n_features
        )
        all_rows += codec_rows
        print(f"  {codec}: {len(codec_rows)} rows")
        for cvvdp_target, target_score in ANCHOR_BANDS_CVVDP:
            s = stats.get(cvvdp_target, {})
            emit = s.get("emitted", 0)
            filt = s.get("filtered", 0)
            tot = emit + filt
            pct = 100 * filt / tot if tot else 0
            md = (s.get("tot_distance", 0) / emit) if emit else 0
            print(f"    cv={cvvdp_target:5.3f}→s={target_score:5.0f}: "
                  f"emit={emit:4d} filt={filt:4d} ({pct:5.1f}%) md={md:.4f}")

    anchor_out = args.out_dir / "anchors_cvvdp_300col.parquet"
    write_parquet(all_rows, anchor_out, args.n_features)

    if not args.skip_equivalence:
        print()
        print("=== cvvdp cross-codec equivalence pairs ===")
        pairs = build_cross_codec_equivalence_cvvdp(
            codec_dfs, args.cvvdp_eq_tolerance, args.n_features
        )
        eq_out = args.out_dir / "cross_codec_equivalence_cvvdp.parquet"
        write_eq_parquet(pairs, eq_out, args.n_features)


if __name__ == "__main__":
    main()
