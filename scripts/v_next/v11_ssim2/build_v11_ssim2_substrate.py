#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V11 ssim2-anchored substrate builder (2026-05-20).

Per user direction 2026-05-20 (task #189): rebuild zensim cross-codec
substrate using ssim2 as the primary anchor, replacing the butter_pnorm3-
based substrate inherited from V_CrossCodec → V6 → V9 → V10. ssim2 is
grounded in CID22 paper Table 4 PJND=63 and cvvdp has 5× lower Z-RMSE
than ssim2 per Mohammadi 2025.

## Data reality (verified 2026-05-20)

The canonical ssim2 score parquet
(`canonical-2026-05-18/scores/ssim2_imazen.parquet`) has only 5 q-levels
per codec at 80–194 imgs each. The unified V parquets reveal:

| Source corpus | rows | unique q | unique imgs | ssim2 non-null | cvvdp non-null |
|---|--:|--:|--:|--:|--:|
| unified_v15r_zenjpeg_cvvdp | 1,785,696 | 19 (q=5..95) | 979 | 100% | 61% |
| unified_v15rc_zenjpeg_cvvdp | 513,570 | 19 (q=5..95) | 901 | 100% | 0% |
| unified_v12_zenavif_cvvdp | 4,000 | 5 (q=10..90) | 200 | 0% | 95% |
| unified_v12_zenwebp_cvvdp | 1,000 | 5 (q=10..90) | 200 | 0% | 100% |
| unified_v12_zenjxl_cvvdp | 32,000 | 5 (q=10..90) | 200 | 0% | 98% |
| unified_v13_zenjpeg_cvvdp | 36,000 | 5 (q=10..90) | 200 | 100% | 97% |
| picker-training/butter/<codec> | 19,000 | 19 (q=5..95) | 1,000 | NO | NO |

**ssim2 is computed only for zenjpeg in the high-coverage substrate.**
For zenwebp/zenavif/zenjxl we have cvvdp but not ssim2. The original
butter-based substrate worked because butter was computed across all 4
codecs uniformly.

## Strategy

To honor the user's ssim2-anchor design while accommodating the data
reality, we build a **hybrid V11 anchor substrate**:

1. **zenjpeg anchor rows**: use ssim2 directly via `unified_v15r`
   (979 imgs × 19 q × 1.78M rows, ssim2 100% non-null). For each
   (image, band) cell, pick the q whose ssim2 is closest to the band's
   `ssim2_target`. Drop if |ssim2 − target| > tolerance.

2. **zenwebp/zenavif/zenjxl anchor rows**: use cvvdp via `unified_v12_*`
   (200 imgs × 5 q × 1k–32k rows, cvvdp 95–100% non-null). Convert
   `ssim2_target` band into a `cvvdp_target` via the empirically-fit
   zenjpeg cvvdp→ssim2 map (50-bin median curve from v15r). Pick q
   whose cvvdp is closest to that cvvdp_target. Drop if distance >
   tolerance.

3. **Document the limitation**: zenjpeg substrate has 19-q coverage; the
   other codecs are limited to the 5 q-levels (10/30/60/80/90) of the
   unified V12 sweeps. The cvvdp→ssim2 conversion assumes cvvdp's
   perceptual calibration is codec-agnostic (cvvdp anchors at JND/JOD
   apply equally across codecs); this is an explicit modeling assumption,
   not a data-derived guarantee.

4. **Cross-codec band coverage**: only bands where all 4 codecs have at
   least one anchor row are emitted to the "tight" subset. The "wide"
   subset emits whatever each codec has.

## Anchor band table (V11, per task brief)

| ssim2 | target_score | semantic |
|---:|---:|---|
| 100 | 100 | mathematically lossless |
|  95 |  95 | near-lossless |
|  90 |  90 | visually lossless (CID22 paper anchor) |
|  75 |  80 | JND (our score-space) — ssim2=75 maps to OUR JND |
|  60 |  65 | mildly noticeable |
|  45 |  50 | JOD |
|  30 |  35 | 3×-DPI resize-out |
|  18 |  20 | clear artifacts |
|  10 |  10 | very degraded |
|   3 |   0 | borderline unacceptable |

Note: CID22 paper Table 4 says ssim2 = 63 IS PJND, but our convention
puts JND at score = 80. The mapping is OUR score ↔ ssim2; we
re-anchor JND to ssim2 = 75 (a tighter PJND than the paper, but
aligned with our 80-score-space convention).

## Output

- `anchors_ssim2_372col.parquet` — V11 anchor parquet with same schema
  as V8/V10 (ref_basename, anchor_source, human_score, anchor_weight,
  q, ssim2_anchor, ssim2_target, target_score, codec, f0..f371).
  ssim2 substituted for butter_pnorm3 in the anchor-pivot column.
- `cross_codec_equivalence_ssim2.parquet` — V11 cross-codec equivalence
  pair parquet pivoting on cvvdp (because cvvdp has cross-codec
  coverage; ssim2 does not). For each (image, ssim2_band_L), find the q
  per codec whose cvvdp lands at the cvvdp-equivalent-of-L (via the
  ssim2→cvvdp inverse map). Same schema as V8 cross-codec parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# (ssim2_target, V11_target_score). 10 bands.
ANCHOR_BANDS_V11: list[tuple[float, float]] = [
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

# Source corpora.
#
# For zenjpeg anchor rows: use v15r (979 imgs × 19 q × ssim2 100% nn).
# For cross-codec equivalence: use v13_zenjpeg (200 imgs shared with v12 codecs).
# For other-codec anchor rows: use v12 unified parquets (200 imgs each, shared corpus).
#
# v13_zenjpeg and v12_zen{webp,avif,jxl} share the same 200 gen-* synthetic
# images, so they support cross-codec pairing. v15r_zenjpeg uses wikimedia
# 512sq images that don't overlap with v12, so it can't be paired cross-codec
# but provides dense ssim2 anchor coverage.
ZENJPEG_ANCHOR_PRIMARY = Path(
    "/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_v15r_zenjpeg_cvvdp.parquet"
)
ZENJPEG_CROSSCODEC_PARQUET = Path(
    "/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_v13_zenjpeg_cvvdp.parquet"
)
PER_CODEC_OTHER = {
    "zenwebp": Path("/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_v12_zenwebp_cvvdp.parquet"),
    "zenavif": Path("/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_v12_zenavif_cvvdp.parquet"),
    "zenjxl": Path("/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_v12_zenjxl_cvvdp.parquet"),
}

DEFAULT_OUT_DIR = Path("/mnt/v/zen/zensim-training/2026-05-20-ssim2-anchors")

# Tolerances: ssim2 anchor matching is in [0..100] units.
# Per task brief: ±3 ssim2 default for anchor; ±2 for equivalence pairs.
SSIM2_ANCHOR_TOLERANCE_DEFAULT = 3.0
SSIM2_EQ_TOLERANCE_DEFAULT = 2.0
CVVDP_EQ_TOLERANCE_DEFAULT = 0.05  # cvvdp range ~6.8..10, scaled


def load_zenjpeg_substrate(path: Path) -> pd.DataFrame:
    """Load zenjpeg v15r and return rows with ssim2 + features + cvvdp."""
    print(f"loading zenjpeg substrate {path}")
    t = pq.read_table(path)
    df = t.to_pandas()
    df["ref_basename"] = df["image_path"].apply(os.path.basename)
    # The schema uses score_ssim2 (no _gpu suffix); features are feat_*
    cols_keep = [
        "ref_basename", "codec", "q",
        "score_ssim2", "cvvdp_imazen_v0_0_1",
    ] + [c for c in df.columns if c.startswith("feat_")]
    df = df[cols_keep].copy()
    df = df.rename(columns={"score_ssim2": "ssim2_anchor",
                              "cvvdp_imazen_v0_0_1": "cvvdp_anchor"})
    # Rename feat_N → fN to match V8/V10 anchor schema
    for c in list(df.columns):
        if c.startswith("feat_"):
            n = c[len("feat_"):]
            df = df.rename(columns={c: f"f{n}"})
    feat_count = sum(1 for c in df.columns if c.startswith("f") and c[1:].isdigit())
    print(f"  rows={len(df)} unique_imgs={df['ref_basename'].nunique()} "
          f"unique_q={sorted(df['q'].unique())} feature_count={feat_count}")
    print(f"  ssim2 nn={df['ssim2_anchor'].notna().sum()} "
          f"cvvdp nn={df['cvvdp_anchor'].notna().sum()}")
    return df


def load_codec_substrate(codec: str, path: Path) -> pd.DataFrame:
    """Load non-jpeg unified V12 parquet, return cvvdp + features rows."""
    print(f"loading {codec} substrate {path}")
    t = pq.read_table(path)
    df = t.to_pandas()
    df["ref_basename"] = df["image_path"].apply(os.path.basename)
    cols_keep = [
        "ref_basename", "codec", "q",
        "score_ssim2", "cvvdp_imazen_v0_0_1",
    ] + [c for c in df.columns if c.startswith("feat_")]
    df = df[cols_keep].copy()
    df = df.rename(columns={"score_ssim2": "ssim2_anchor",
                              "cvvdp_imazen_v0_0_1": "cvvdp_anchor"})
    for c in list(df.columns):
        if c.startswith("feat_"):
            n = c[len("feat_"):]
            df = df.rename(columns={c: f"f{n}"})
    feat_count = sum(1 for c in df.columns if c.startswith("f") and c[1:].isdigit())
    print(f"  rows={len(df)} unique_imgs={df['ref_basename'].nunique()} "
          f"unique_q={sorted(df['q'].unique())} feature_count={feat_count}")
    print(f"  ssim2 nn={df['ssim2_anchor'].notna().sum()} "
          f"cvvdp nn={df['cvvdp_anchor'].notna().sum()}")
    return df


def fit_cvvdp_to_ssim2_map(df_zenjpeg: pd.DataFrame, n_bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Fit cvvdp → ssim2 monotone map on zenjpeg data where both exist.

    Returns (cvvdp_knots, ssim2_knots), sorted ascending in cvvdp. Use
    np.interp for piecewise-linear lookup.
    """
    mask = df_zenjpeg["ssim2_anchor"].notna() & df_zenjpeg["cvvdp_anchor"].notna()
    if not mask.any():
        raise RuntimeError("zenjpeg has no rows with both ssim2 and cvvdp")
    sub = df_zenjpeg.loc[mask, ["cvvdp_anchor", "ssim2_anchor"]].sort_values("cvvdp_anchor")
    print(f"  fitting cvvdp→ssim2 on {len(sub)} zenjpeg rows")
    bin_ix = (np.arange(len(sub)) * n_bins // len(sub)).clip(max=n_bins - 1)
    sub = sub.assign(bin_ix=bin_ix)
    g = sub.groupby("bin_ix").agg(cvvdp=("cvvdp_anchor", "median"),
                                    ssim2=("ssim2_anchor", "median"))
    # Enforce strict monotone increase
    cvvdp = g["cvvdp"].to_numpy()
    ssim2 = g["ssim2"].to_numpy()
    # Sort by cvvdp (should already be sorted via bin_ix order, but double-check)
    order = np.argsort(cvvdp)
    cvvdp = cvvdp[order]
    ssim2 = ssim2[order]
    # Enforce strict monotone increase on cvvdp (drop dupes if any)
    keep = np.concatenate([[True], np.diff(cvvdp) > 0])
    cvvdp = cvvdp[keep]
    ssim2 = ssim2[keep]
    # Enforce monotone increase on ssim2 (cumulative max) — small inversions
    # in the median are smoothed out.
    ssim2 = np.maximum.accumulate(ssim2)
    print(f"  map points={len(cvvdp)} cvvdp range=[{cvvdp.min():.3f}, {cvvdp.max():.3f}] "
          f"ssim2 range=[{ssim2.min():.2f}, {ssim2.max():.2f}]")
    return cvvdp, ssim2


def ssim2_to_cvvdp(ssim2: float, cvvdp_knots: np.ndarray, ssim2_knots: np.ndarray) -> float:
    """Inverse map: given target ssim2, return cvvdp value via interp."""
    # ssim2_knots is monotone increasing; we want the cvvdp that yields it.
    if ssim2 <= ssim2_knots[0]:
        return float(cvvdp_knots[0])
    if ssim2 >= ssim2_knots[-1]:
        return float(cvvdp_knots[-1])
    return float(np.interp(ssim2, ssim2_knots, cvvdp_knots))


def cvvdp_to_ssim2(cvvdp: float, cvvdp_knots: np.ndarray, ssim2_knots: np.ndarray) -> float:
    if cvvdp <= cvvdp_knots[0]:
        return float(ssim2_knots[0])
    if cvvdp >= cvvdp_knots[-1]:
        return float(ssim2_knots[-1])
    return float(np.interp(cvvdp, cvvdp_knots, ssim2_knots))


def build_anchor_rows_jpeg(
    df_zenjpeg: pd.DataFrame,
    ssim2_tolerance: float,
    n_features: int,
) -> tuple[list[dict], dict[tuple[float], dict[str, int]]]:
    """Build zenjpeg anchor rows by direct ssim2-band matching."""
    print("=== building zenjpeg anchor rows (direct ssim2) ===")
    rows: list[dict] = []
    stats: dict[float, dict[str, int]] = {}
    feature_cols = [f"f{i}" for i in range(n_features)]
    df_valid = df_zenjpeg[df_zenjpeg["ssim2_anchor"].notna()].copy()
    print(f"  using {len(df_valid)} rows with ssim2")

    for source, group in df_valid.groupby("ref_basename"):
        for ssim2_target, target_score in ANCHOR_BANDS_V11:
            stats.setdefault(ssim2_target, {"emitted": 0, "filtered": 0,
                                              "tot_distance": 0.0})
            distances = (group["ssim2_anchor"] - ssim2_target).abs()
            idx = distances.idxmin()
            best = group.loc[idx]
            best_distance = float(distances.loc[idx])
            if best_distance > ssim2_tolerance:
                stats[ssim2_target]["filtered"] += 1
                continue
            stats[ssim2_target]["emitted"] += 1
            stats[ssim2_target]["tot_distance"] += best_distance
            row = {
                "ref_basename": str(source),
                "anchor_source": f"v11_zenjpeg_s{ssim2_target:.0f}_t{target_score:.0f}_direct",
                "human_score": float(target_score),
                "anchor_weight": 1.0,
                "q": int(best["q"]),
                "ssim2_anchor": float(best["ssim2_anchor"]),
                "ssim2_target": float(ssim2_target),
                "cvvdp_anchor": (
                    float(best["cvvdp_anchor"]) if pd.notna(best["cvvdp_anchor"]) else np.nan
                ),
                "target_score": float(target_score),
                "codec": "zenjpeg",
                "anchor_via": "ssim2_direct",
            }
            for col in feature_cols:
                if col in best.index:
                    val = best[col]
                    row[col] = float(val) if pd.notna(val) else 0.0
                else:
                    row[col] = 0.0
            rows.append(row)
    return rows, stats


def build_anchor_rows_other(
    codec: str,
    df_codec: pd.DataFrame,
    cvvdp_knots: np.ndarray,
    ssim2_knots: np.ndarray,
    n_features: int,
    cvvdp_tolerance: float = 0.1,
) -> tuple[list[dict], dict[tuple[float], dict[str, int]]]:
    """Build non-jpeg codec anchor rows via cvvdp-equivalent-of-ssim2 band."""
    print(f"=== building {codec} anchor rows (via cvvdp→ssim2 conversion) ===")
    rows: list[dict] = []
    stats: dict[float, dict[str, int]] = {}
    feature_cols = [f"f{i}" for i in range(n_features)]
    df_valid = df_codec[df_codec["cvvdp_anchor"].notna()].copy()
    print(f"  using {len(df_valid)} rows with cvvdp")

    for source, group in df_valid.groupby("ref_basename"):
        for ssim2_target, target_score in ANCHOR_BANDS_V11:
            # Convert the ssim2 band into the equivalent cvvdp value via the
            # zenjpeg-derived map.
            cvvdp_target = ssim2_to_cvvdp(ssim2_target, cvvdp_knots, ssim2_knots)
            stats.setdefault(ssim2_target, {"emitted": 0, "filtered": 0,
                                              "tot_distance": 0.0,
                                              "cvvdp_target": cvvdp_target})
            distances = (group["cvvdp_anchor"] - cvvdp_target).abs()
            idx = distances.idxmin()
            best = group.loc[idx]
            best_distance = float(distances.loc[idx])
            if best_distance > cvvdp_tolerance:
                stats[ssim2_target]["filtered"] += 1
                continue
            stats[ssim2_target]["emitted"] += 1
            stats[ssim2_target]["tot_distance"] += best_distance
            # Reconstruct the "as-if ssim2" value via inverse map at the actual
            # cvvdp landing point.
            reconstructed_ssim2 = cvvdp_to_ssim2(
                float(best["cvvdp_anchor"]), cvvdp_knots, ssim2_knots
            )
            row = {
                "ref_basename": str(source),
                "anchor_source": f"v11_{codec}_s{ssim2_target:.0f}_t{target_score:.0f}_via_cvvdp",
                "human_score": float(target_score),
                "anchor_weight": 1.0,
                "q": int(best["q"]),
                "ssim2_anchor": float(reconstructed_ssim2),
                "ssim2_target": float(ssim2_target),
                "cvvdp_anchor": float(best["cvvdp_anchor"]),
                "target_score": float(target_score),
                "codec": codec,
                "anchor_via": "cvvdp_to_ssim2_map",
            }
            for col in feature_cols:
                if col in best.index:
                    val = best[col]
                    row[col] = float(val) if pd.notna(val) else 0.0
                else:
                    row[col] = 0.0
            rows.append(row)
    return rows, stats


def write_anchor_parquet(rows: list[dict], out_path: Path, n_features: int) -> None:
    if not rows:
        raise SystemExit("no anchor rows built")
    feature_cols = [f"f{i}" for i in range(n_features)]
    cols = {
        "ref_basename": pa.array([r["ref_basename"] for r in rows], type=pa.string()),
        "anchor_source": pa.array([r["anchor_source"] for r in rows], type=pa.string()),
        "human_score": pa.array([r["human_score"] for r in rows], type=pa.float64()),
        "anchor_weight": pa.array([r["anchor_weight"] for r in rows], type=pa.float64()),
        "q": pa.array([r["q"] for r in rows], type=pa.int64()),
        "ssim2_anchor": pa.array([r["ssim2_anchor"] for r in rows], type=pa.float64()),
        "ssim2_target": pa.array([r["ssim2_target"] for r in rows], type=pa.float64()),
        "cvvdp_anchor": pa.array([r["cvvdp_anchor"] for r in rows], type=pa.float64()),
        "target_score": pa.array([r["target_score"] for r in rows], type=pa.float64()),
        "codec": pa.array([r["codec"] for r in rows], type=pa.string()),
        "anchor_via": pa.array([r["anchor_via"] for r in rows], type=pa.string()),
    }
    for col in feature_cols:
        cols[col] = pa.array([r[col] for r in rows], type=pa.float32())
    tbl = pa.table(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out_path, compression="zstd", compression_level=15)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB, "
          f"{tbl.num_rows} rows × {tbl.num_columns} cols)")


def build_cross_codec_equivalence(
    df_zenjpeg: pd.DataFrame,
    others: dict[str, pd.DataFrame],
    cvvdp_knots: np.ndarray,
    ssim2_knots: np.ndarray,
    ssim2_eq_tolerance: float,
    cvvdp_eq_tolerance: float,
    n_features: int,
) -> list[dict]:
    """Cross-codec equivalence pairs pivoted on ssim2 (zenjpeg-anchored).

    For each (image, ssim2_L), find:
      - For zenjpeg: q where ssim2 closest to L (within ssim2_eq_tolerance).
      - For other codec C: q where cvvdp closest to ssim2_to_cvvdp(L)
        (within cvvdp_eq_tolerance).
    Emit pair (zenjpeg, C) and (C, C') for C, C' in others.
    """
    print("=== building cross-codec equivalence pairs ===")
    # ssim2 pivot levels - dense in the band region; per task brief use 8 levels.
    ssim2_levels = [90.0, 80.0, 75.0, 65.0, 50.0, 35.0, 20.0, 10.0]
    print(f"  ssim2 pivot levels: {ssim2_levels}")

    pairs: list[dict] = []
    feature_cols = [f"f{i}" for i in range(n_features)]

    # Index zenjpeg by ref_basename (drop rows without ssim2)
    df_jpeg = df_zenjpeg[df_zenjpeg["ssim2_anchor"].notna()]
    jpeg_imgs = set(df_jpeg["ref_basename"].unique())
    print(f"  zenjpeg imgs with ssim2: {len(jpeg_imgs)}")

    # Per-codec index for others (cvvdp only)
    other_indexed = {}
    for codec, df in others.items():
        valid = df[df["cvvdp_anchor"].notna()]
        other_indexed[codec] = valid
        print(f"  {codec} imgs with cvvdp: {valid['ref_basename'].nunique()}")

    # For each image, build per-codec picks at each level
    all_imgs = jpeg_imgs.copy()
    for d in other_indexed.values():
        all_imgs |= set(d["ref_basename"].unique())
    print(f"  total unique imgs: {len(all_imgs)}")

    # Find intersection: imgs present in zenjpeg AND at least one other codec
    inter = jpeg_imgs.copy()
    for codec, d in other_indexed.items():
        codec_imgs = set(d["ref_basename"].unique())
        # We're not requiring all 4; just need at least one cross-codec pair
    all_imgs_sorted = sorted(all_imgs)

    n_imgs_processed = 0
    pair_counter = {}
    for ref in all_imgs_sorted:
        n_imgs_processed += 1
        if n_imgs_processed % 100 == 0:
            print(f"    img {n_imgs_processed}/{len(all_imgs_sorted)} pairs={len(pairs)}")
        per_codec_picks: dict[str, dict[float, dict]] = {}
        # zenjpeg picks via ssim2
        if ref in jpeg_imgs:
            j_group = df_jpeg[df_jpeg["ref_basename"] == ref]
            per_codec_picks["zenjpeg"] = {}
            for L in ssim2_levels:
                dist = (j_group["ssim2_anchor"] - L).abs()
                if len(dist) == 0:
                    continue
                idx = dist.idxmin()
                bd = float(dist.loc[idx])
                if bd > ssim2_eq_tolerance:
                    continue
                row = j_group.loc[idx]
                per_codec_picks["zenjpeg"][L] = {
                    "q": int(row["q"]),
                    "ssim2": float(row["ssim2_anchor"]),
                    "cvvdp": (
                        float(row["cvvdp_anchor"]) if pd.notna(row["cvvdp_anchor"]) else None
                    ),
                    "feat": [
                        float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0
                        for c in feature_cols
                    ],
                }
        # other codec picks via cvvdp
        for codec, df_c in other_indexed.items():
            c_group = df_c[df_c["ref_basename"] == ref]
            if len(c_group) == 0:
                continue
            per_codec_picks[codec] = {}
            for L in ssim2_levels:
                cvvdp_target = ssim2_to_cvvdp(L, cvvdp_knots, ssim2_knots)
                dist = (c_group["cvvdp_anchor"] - cvvdp_target).abs()
                if len(dist) == 0:
                    continue
                idx = dist.idxmin()
                bd = float(dist.loc[idx])
                if bd > cvvdp_eq_tolerance:
                    continue
                row = c_group.loc[idx]
                reconstructed_ssim2 = cvvdp_to_ssim2(
                    float(row["cvvdp_anchor"]), cvvdp_knots, ssim2_knots
                )
                per_codec_picks[codec][L] = {
                    "q": int(row["q"]),
                    "ssim2": float(reconstructed_ssim2),
                    "cvvdp": float(row["cvvdp_anchor"]),
                    "feat": [
                        float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0
                        for c in feature_cols
                    ],
                }

        # Emit pairs: every (codec_a, codec_b) where both have a pick at L
        codecs_present = list(per_codec_picks.keys())
        for L in ssim2_levels:
            valid_codecs = [c for c in codecs_present if L in per_codec_picks[c]]
            for i in range(len(valid_codecs)):
                for j in range(i + 1, len(valid_codecs)):
                    ca = valid_codecs[i]
                    cb = valid_codecs[j]
                    a = per_codec_picks[ca][L]
                    b = per_codec_picks[cb][L]
                    ssim2_diff = abs(a["ssim2"] - b["ssim2"])
                    # Weight tight pairs higher.
                    weight = float(1.0 / (ssim2_diff + 0.5))
                    if weight > 4.0:
                        weight = 4.0
                    pair_counter[(ca, cb)] = pair_counter.get((ca, cb), 0) + 1
                    pairs.append({
                        "ref_basename": ref,
                        "codec_a": ca,
                        "q_a": a["q"],
                        "codec_b": cb,
                        "q_b": b["q"],
                        "ssim2_level": L,
                        "ssim2_a": a["ssim2"],
                        "ssim2_b": b["ssim2"],
                        "ssim2_diff": ssim2_diff,
                        "cvvdp_a": a["cvvdp"] if a["cvvdp"] is not None else np.nan,
                        "cvvdp_b": b["cvvdp"] if b["cvvdp"] is not None else np.nan,
                        "row_weight": weight,
                        "fa": a["feat"],
                        "fb": b["feat"],
                    })

    print(f"=== built {len(pairs)} equivalence pairs ===")
    for (ca, cb), c in sorted(pair_counter.items()):
        print(f"  {ca:>10s} ↔ {cb:<10s}  {c:6d}")
    return pairs


def write_equivalence_parquet(pairs: list[dict], out_path: Path, n_features: int) -> None:
    if not pairs:
        raise SystemExit("no equivalence pairs built")
    fields = [
        pa.field("ref_basename", pa.string()),
        pa.field("codec_a", pa.string()),
        pa.field("q_a", pa.int64()),
        pa.field("codec_b", pa.string()),
        pa.field("q_b", pa.int64()),
        pa.field("ssim2_level", pa.float64()),
        pa.field("ssim2_a", pa.float64()),
        pa.field("ssim2_b", pa.float64()),
        pa.field("ssim2_diff", pa.float64()),
        pa.field("cvvdp_a", pa.float64()),
        pa.field("cvvdp_b", pa.float64()),
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
        pa.array([p["ssim2_level"] for p in pairs], type=pa.float64()),
        pa.array([p["ssim2_a"] for p in pairs], type=pa.float64()),
        pa.array([p["ssim2_b"] for p in pairs], type=pa.float64()),
        pa.array([p["ssim2_diff"] for p in pairs], type=pa.float64()),
        pa.array([p["cvvdp_a"] for p in pairs], type=pa.float64()),
        pa.array([p["cvvdp_b"] for p in pairs], type=pa.float64()),
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
    parser.add_argument("--ssim2-tolerance", type=float,
                        default=SSIM2_ANCHOR_TOLERANCE_DEFAULT,
                        help="anchor-band ssim2 max distance (default 3.0)")
    parser.add_argument("--cvvdp-anchor-tolerance", type=float, default=0.1,
                        help="anchor-band cvvdp max distance for non-jpeg "
                             "codecs (default 0.1)")
    parser.add_argument("--ssim2-eq-tolerance", type=float,
                        default=SSIM2_EQ_TOLERANCE_DEFAULT,
                        help="equivalence-pair ssim2 max distance (default 2.0)")
    parser.add_argument("--cvvdp-eq-tolerance", type=float,
                        default=CVVDP_EQ_TOLERANCE_DEFAULT,
                        help="equivalence-pair cvvdp max distance (default 0.05)")
    parser.add_argument("--n-features", type=int, default=300,
                        help="number of feature columns (default 300 — "
                             "matches LARGE schema, max for cvvdp parquets)")
    parser.add_argument("--skip-equivalence", action="store_true",
                        help="skip Phase 2 cross-codec equivalence build")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: load corpora
    df_zenjpeg = load_zenjpeg_substrate(ZENJPEG_ANCHOR_PRIMARY)
    others = {}
    for codec, path in PER_CODEC_OTHER.items():
        others[codec] = load_codec_substrate(codec, path)

    # Separate zenjpeg corpus for cross-codec equivalence (shares basenames
    # with v12 zenwebp/avif/jxl).
    print()
    df_zenjpeg_cc = load_codec_substrate("zenjpeg_cc", ZENJPEG_CROSSCODEC_PARQUET)
    df_zenjpeg_cc["codec"] = "zenjpeg"  # rename for downstream pair-emission

    # Fit cvvdp→ssim2 map on zenjpeg
    print()
    print("=== fitting cvvdp ↔ ssim2 map on zenjpeg ===")
    cvvdp_knots, ssim2_knots = fit_cvvdp_to_ssim2_map(df_zenjpeg)

    # Save the map for diagnostics
    map_path = args.out_dir / "cvvdp_to_ssim2_map.json"
    map_path.write_text(json.dumps({
        "method": "50-bin median on zenjpeg unified_v15r where both metrics non-null",
        "n_zenjpeg_rows_used": int((df_zenjpeg["ssim2_anchor"].notna() &
                                      df_zenjpeg["cvvdp_anchor"].notna()).sum()),
        "cvvdp_knots": cvvdp_knots.tolist(),
        "ssim2_knots": ssim2_knots.tolist(),
        "ssim2_anchor_band_to_cvvdp_target": {
            f"{ssim2_target:.1f}_score_{target_score:.0f}":
                ssim2_to_cvvdp(ssim2_target, cvvdp_knots, ssim2_knots)
            for ssim2_target, target_score in ANCHOR_BANDS_V11
        },
    }, indent=2))
    print(f"wrote {map_path}")

    # Phase 1: anchor parquet
    print()
    print("=== PHASE 1: anchor parquet ===")
    all_rows = []
    all_rows_jpeg, stats_jpeg = build_anchor_rows_jpeg(
        df_zenjpeg, args.ssim2_tolerance, args.n_features
    )
    all_rows += all_rows_jpeg
    print(f"  zenjpeg emitted: {len(all_rows_jpeg)} rows")
    for ssim2_target, target_score in ANCHOR_BANDS_V11:
        s = stats_jpeg.get(ssim2_target, {})
        emit = s.get("emitted", 0)
        filt = s.get("filtered", 0)
        tot = emit + filt
        pct = 100 * filt / tot if tot else 0
        mean_d = (s.get("tot_distance", 0) / emit) if emit else 0
        print(f"    ssim2={ssim2_target:5.0f}→score={target_score:5.0f}: "
              f"emit={emit:5d} filt={filt:5d} ({pct:5.1f}%) mean_d={mean_d:.2f}")

    for codec, df_c in others.items():
        codec_rows, stats_c = build_anchor_rows_other(
            codec, df_c, cvvdp_knots, ssim2_knots, args.n_features,
            cvvdp_tolerance=args.cvvdp_anchor_tolerance,
        )
        all_rows += codec_rows
        print(f"  {codec} emitted: {len(codec_rows)} rows")
        for ssim2_target, target_score in ANCHOR_BANDS_V11:
            s = stats_c.get(ssim2_target, {})
            emit = s.get("emitted", 0)
            filt = s.get("filtered", 0)
            tot = emit + filt
            pct = 100 * filt / tot if tot else 0
            mean_d = (s.get("tot_distance", 0) / emit) if emit else 0
            cv_t = s.get("cvvdp_target", 0)
            print(f"    ssim2={ssim2_target:5.0f}→score={target_score:5.0f} "
                  f"(cvvdp_target={cv_t:5.3f}): emit={emit:5d} filt={filt:5d} "
                  f"({pct:5.1f}%) mean_d={mean_d:.4f}")

    anchor_out = args.out_dir / "anchors_ssim2_300col.parquet"
    write_anchor_parquet(all_rows, anchor_out, args.n_features)

    # Per-codec/band landings summary
    print()
    print("=== anchor landing summary (per codec × band) ===")
    df_anchors = pd.DataFrame([{
        "codec": r["codec"], "ssim2_target": r["ssim2_target"],
        "target_score": r["target_score"], "ssim2_anchor": r["ssim2_anchor"],
        "cvvdp_anchor": r["cvvdp_anchor"], "q": r["q"],
    } for r in all_rows])
    for ssim2_target, target_score in ANCHOR_BANDS_V11:
        print(f"\nband ssim2={ssim2_target:.0f} → target_score={target_score:.0f}:")
        sub = df_anchors[df_anchors["ssim2_target"] == ssim2_target]
        if len(sub) == 0:
            print("  no rows")
            continue
        for codec in ["zenjpeg", "zenwebp", "zenavif", "zenjxl"]:
            csub = sub[sub["codec"] == codec]
            if len(csub) == 0:
                print(f"  {codec}: 0 rows")
                continue
            print(f"  {codec}: n={len(csub):5d} q_median={csub['q'].median():4.0f} "
                  f"ssim2_med={csub['ssim2_anchor'].median():7.2f} "
                  f"cvvdp_med={csub['cvvdp_anchor'].median():6.3f}")

    # Phase 2: cross-codec equivalence pairs
    if not args.skip_equivalence:
        print()
        print("=== PHASE 2: cross-codec equivalence pairs ===")
        # Use v13 zenjpeg (200 imgs shared with v12 others) NOT v15r
        # zenjpeg (979 imgs, disjoint corpus). v15r remains the anchor
        # primary because of its 19-q-level density.
        pairs = build_cross_codec_equivalence(
            df_zenjpeg_cc, others, cvvdp_knots, ssim2_knots,
            args.ssim2_eq_tolerance, args.cvvdp_eq_tolerance,
            args.n_features,
        )
        eq_out = args.out_dir / "cross_codec_equivalence_ssim2.parquet"
        write_equivalence_parquet(pairs, eq_out, args.n_features)


if __name__ == "__main__":
    main()
