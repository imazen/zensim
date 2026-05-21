#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V12-A cvvdp-anchored substrate builder (task #199, 2026-05-20).

cvvdp-pivoted counterpart to the V11-V2 372-feat ssim2 substrate
(`build_v11_372feat_substrate.py`). The Mohammadi 2025 paper reports
cvvdp Z-RMSE 9.45 vs ssim2 47.63 (~5× better absolute calibration); the
V11 frontier was closed at the same anchor pivot but on the wrong
metric. V12-A swaps the pivot to cvvdp and retests whether the
cross-codec-eq + anchor-loss mechanism still collapses KonJND.

## cvvdp → target_score mapping

cvvdp scores are JOD-scale, 10.0 = imperceptible, lower = worse. The
V11-cvvdp probe (`build_v11_cvvdp_substrate.py`) confirmed the empirical
percentile distribution on the multi-codec corpus:

  p99 ≈ 10.00 / p95 ≈ 10.00 / p75 ≈ 9.99 / p50 ≈ 9.95 / p25 ≈ 9.82 /
  p10 ≈ 9.44 / p5  ≈ 9.18 / p1  ≈ 8.46

On the multi_codec_372col_full.parquet (117,800 rows, 4 codecs, 5 q
levels per image), the cvvdp distribution is:

  zenavif n=4000 :  p1=7.89 p10=8.92 p25=9.57 p50=9.96 p90=9.999
  zenjpeg n=61600:  p1=8.12 p10=9.24 p25=9.61 p50=9.89 p90=9.990
  zenjxl  n=51200:  p1=9.67 p10=9.90 p25=9.96 p50=9.99 p90=10.00 (SATURATED)
  zenwebp n=1000 :  p1=9.03 p10=9.63 p25=9.79 p50=9.91 p90=9.98

zenjxl saturates ABOVE 9.85 across the full q range — the low-q bands
(cvvdp ≤ 9.5) will have zero zenjxl rows. Same structural coverage
issue as V11-ssim2; preserved by design (the cross-codec-eq term only
fires when both codecs have a row at the same pivot).

## Mapping (10 bands, V10 score-space)

| cvvdp target | target_score | semantic                                |
|--:|--:|---|
| 10.00 | 100 | imperceptible / mathematically lossless |
|  9.95 |  95 | near-imperceptible (≈ p50)             |
|  9.85 |  90 | visually lossless                       |
|  9.65 |  80 | JND threshold (≈ p25-p35)              |
|  9.30 |  65 | mildly noticeable                       |
|  8.50 |  50 | JOD (just objectionable)               |
|  7.50 |  35 | 3x-DPI resize-out                       |
|  6.50 |  20 | clear artifacts                         |
|  5.00 |  10 | very degraded                           |
|  3.00 |   0 | borderline unacceptable                 |

Coverage rule: cvvdp tolerance ±0.4 per band per task brief (much
larger than the V11-cvvdp probe's ±0.05 because the per-image q grid
is sparse — 5 q levels rather than 19). Bands missed at this
tolerance silently drop from the substrate.

## Inputs

- `--input-parquet` (default the V11-DECODER-FIX 372col full):
  `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/multi_codec_372col_full.parquet`

## Outputs

- `<out-dir>/anchors_cvvdp_372col.parquet` — V5 multi-band schema,
  10 bands × 4 codecs × per-image-best-q, 372 features.
- `<out-dir>/cross_codec_equivalence_cvvdp_372col.parquet` — pairs
  across codec_a/codec_b for same image at the same cvvdp pivot.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse V11-V2 substrate helpers so write-format is byte-identical
# to the V11-A v4 anchor/equiv parquets the trainer already consumes.
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "v11_ssim2_v2"))
from build_v11_substrate_v2 import (  # noqa: E402
    write_anchor_parquet,
    write_equivalence_parquet,
)


# (cvvdp_target, V10/V11_target_score). 10 bands per task brief.
ANCHOR_BANDS_V12_CVVDP: list[tuple[float, float]] = [
    (10.00, 100.0),
    (9.95, 95.0),
    (9.85, 90.0),
    (9.65, 80.0),   # JND
    (9.30, 65.0),
    (8.50, 50.0),   # JOD
    (7.50, 35.0),
    (6.50, 20.0),
    (5.00, 10.0),
    (3.00, 0.0),
]


def load_input(input_parquet: Path, n_features: int) -> pd.DataFrame:
    """Load + rename the V11-DECODER-FIX multi-codec parquet.

    The Rust extractor emits columns f0..f371 + score_cvvdp_imazen_v0_0_1 +
    score_ssim2_gpu + score_butteraugli_pnorm3_gpu. Rename f{i} → feat_{i}
    so the V11-V2 write helpers (which expect feat_*) work unchanged.
    """
    print(f"loading {input_parquet}")
    df = pq.read_table(input_parquet).to_pandas()
    df["ref_basename"] = df["image_path"].apply(os.path.basename)

    # f{i} → feat_{i}
    rename_map = {f"f{i}": f"feat_{i}" for i in range(n_features)}
    df = df.rename(columns=rename_map)

    print(f"  rows={len(df)} codecs={sorted(df['codec'].unique())}")
    for codec, g in df.groupby("codec"):
        ssim2_nn = g["score_ssim2_gpu"].notna().sum() if "score_ssim2_gpu" in g else 0
        cvvdp_nn = g["score_cvvdp_imazen_v0_0_1"].notna().sum()
        butter_nn = g["score_butteraugli_pnorm3_gpu"].notna().sum() if "score_butteraugli_pnorm3_gpu" in g else 0
        print(
            f"  {codec:>10s}: rows={len(g):>7d} imgs={g['ref_basename'].nunique():>4d} "
            f"q={sorted(g['q'].unique().tolist())[:6]} "
            f"ssim2_nn={ssim2_nn} cvvdp_nn={cvvdp_nn} butter_nn={butter_nn}"
        )

    if "score_butteraugli_pnorm3_gpu" not in df.columns:
        df["score_butteraugli_pnorm3_gpu"] = np.nan
    if "score_ssim2_gpu" not in df.columns:
        df["score_ssim2_gpu"] = np.nan
    return df


def cvvdp_distribution_report(df: pd.DataFrame, out_path: Path | None) -> str:
    """Per-codec cvvdp distribution as markdown — emit and return."""
    lines: list[str] = []
    lines.append("# cvvdp distribution per codec (multi_codec_372col_full)")
    lines.append("")
    lines.append("| codec | n | min | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    cv_all = df["score_cvvdp_imazen_v0_0_1"].dropna()
    for codec in sorted(df["codec"].unique()):
        cv = df.loc[df["codec"] == codec, "score_cvvdp_imazen_v0_0_1"].dropna()
        if len(cv) == 0:
            continue
        ps = {p: np.percentile(cv, p) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
        lines.append(
            f"| {codec} | {len(cv)} | {cv.min():.3f} | "
            f"{ps[1]:.3f} | {ps[5]:.3f} | {ps[10]:.3f} | {ps[25]:.3f} | "
            f"{ps[50]:.3f} | {ps[75]:.3f} | {ps[90]:.3f} | {ps[95]:.3f} | "
            f"{ps[99]:.3f} | {cv.max():.3f} |"
        )
    ps = {p: np.percentile(cv_all, p) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
    lines.append(
        f"| **ALL** | **{len(cv_all)}** | {cv_all.min():.3f} | "
        f"{ps[1]:.3f} | {ps[5]:.3f} | {ps[10]:.3f} | {ps[25]:.3f} | "
        f"{ps[50]:.3f} | {ps[75]:.3f} | {ps[90]:.3f} | {ps[95]:.3f} | "
        f"{ps[99]:.3f} | {cv_all.max():.3f} |"
    )
    lines.append("")
    lines.append("Notes:")
    lines.append("- cvvdp scale: JOD 0..10, 10 = imperceptible.")
    lines.append("- zenjxl saturates above 9.85 across the q-grid — low-q bands have ~0 zenjxl rows by design.")
    lines.append("- Low-q coverage limited to zenavif/zenjpeg/zenwebp where the q-grid produces sub-9.0 cvvdp.")
    out = "\n".join(lines) + "\n"
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out)
        print(f"  wrote distribution report: {out_path}")
    return out


def build_cvvdp_anchor_rows(
    df: pd.DataFrame,
    bands: list[tuple[float, float]],
    n_features: int,
    tolerance: float,
) -> pd.DataFrame:
    """For each (image, codec, band), find q with closest cvvdp.

    Mirrors build_v11_substrate_v2.build_anchor_rows but pivots on
    score_cvvdp_imazen_v0_0_1 with the V12 cvvdp bands + the larger
    per-task-brief ±0.4 tolerance.
    """
    print(f"=== Phase 2: cvvdp anchor rows (tol=±{tolerance}) ===")
    df_valid = df[df["score_cvvdp_imazen_v0_0_1"].notna()].copy()
    print(f"  valid rows: {len(df_valid)}")

    rows = []
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    per_band_stats = {b[0]: {"emit": 0, "filt": 0, "tot_d": 0.0, "target_score": b[1]} for b in bands}
    per_codec_band_stats: dict[tuple[str, float], dict] = {}

    for (ref, codec), group in df_valid.groupby(["ref_basename", "codec"], sort=False):
        for cvvdp_target, target_score in bands:
            dist = (group["score_cvvdp_imazen_v0_0_1"] - cvvdp_target).abs()
            idx = dist.idxmin()
            bd = float(dist.loc[idx])
            key = (codec, cvvdp_target)
            per_codec_band_stats.setdefault(
                key, {"emit": 0, "filt": 0, "tot_d": 0.0, "target_score": target_score}
            )
            if bd > tolerance:
                per_band_stats[cvvdp_target]["filt"] += 1
                per_codec_band_stats[key]["filt"] += 1
                continue
            per_band_stats[cvvdp_target]["emit"] += 1
            per_band_stats[cvvdp_target]["tot_d"] += bd
            per_codec_band_stats[key]["emit"] += 1
            per_codec_band_stats[key]["tot_d"] += bd
            best = group.loc[idx]

            row = {
                "ref_basename": str(ref),
                "anchor_source": f"v12cv_{codec}_cv{cvvdp_target:.2f}_t{target_score:.0f}",
                "human_score": float(target_score),
                "anchor_weight": 1.0,
                "q": int(best["q"]),
                "ssim2_anchor": (
                    float(best["score_ssim2_gpu"])
                    if pd.notna(best.get("score_ssim2_gpu"))
                    else float("nan")
                ),
                # ssim2_target placed as NaN — the V11 schema expects a target column
                # per anchor metric, but the cvvdp-pivoted substrate doesn't have an
                # ssim2 target. The trainer doesn't read ssim2_target directly (it
                # reads human_score + target_score), so NaN is safe.
                "ssim2_target": float("nan"),
                "cvvdp_anchor": float(best["score_cvvdp_imazen_v0_0_1"]),
                "butter_pnorm3_anchor": (
                    float(best["score_butteraugli_pnorm3_gpu"])
                    if pd.notna(best.get("score_butteraugli_pnorm3_gpu"))
                    else float("nan")
                ),
                "target_score": float(target_score),
                "codec": str(codec),
                "anchor_via": "cvvdp_direct",
            }
            for c in feature_cols:
                v = best[c]
                row[c] = float(v) if pd.notna(v) else 0.0
            rows.append(row)

    # Per-band summary
    print("  per-band summary:")
    for cvvdp_target, _ in bands:
        s = per_band_stats[cvvdp_target]
        tot = s["emit"] + s["filt"]
        mean_d = s["tot_d"] / s["emit"] if s["emit"] > 0 else 0.0
        pct = 100 * s["filt"] / tot if tot > 0 else 0
        print(
            f"    cvvdp={cvvdp_target:5.2f} → target={s['target_score']:5.0f}: "
            f"emit={s['emit']:>6d} filt={s['filt']:>5d} ({pct:4.1f}%) mean_d={mean_d:.3f}"
        )

    print("  per-codec×band emit counts:")
    codecs = sorted(set(k[0] for k in per_codec_band_stats.keys()))
    header = f"    band/codec        " + "  ".join(f"{c:>10s}" for c in codecs)
    print(header)
    for cvvdp_target, target_score in bands:
        row_str = f"    cvvdp={cvvdp_target:5.2f}→t={target_score:5.0f}  "
        for codec in codecs:
            s = per_codec_band_stats.get((codec, cvvdp_target), {"emit": 0})
            row_str += f"  {s['emit']:>10d}"
        print(row_str)

    return pd.DataFrame(rows)


def build_cvvdp_cross_codec_equivalence(
    df: pd.DataFrame,
    pivot_levels: list[float],
    tolerance: float,
    n_features: int,
) -> pd.DataFrame:
    """Cross-codec equivalence pairs pivoted on cvvdp.

    For each (image, cvvdp_L) and each ordered codec pair (a, b)
    (a < b alphabetically), find the q per codec with cvvdp closest
    to L. Emit pair if both within tolerance and codecs differ.

    Writes the V8 schema columns (butter_level, butter_a, butter_b,
    ssim2_level, ...) populated with the cvvdp pivot in the *_level
    slots so the trainer's existing cross-codec-eq loss code reads
    cvvdp without modification — see build_v11_substrate_v2.build_cross_codec_equivalence
    for the schema rationale.
    """
    print(f"=== Phase 3: cvvdp cross-codec equivalence (levels={pivot_levels}, tol=±{tolerance}) ===")
    df_valid = df[df["score_cvvdp_imazen_v0_0_1"].notna()].copy()
    print(f"  valid rows: {len(df_valid)}")
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    pairs = []
    pair_count: dict[tuple[str, str], int] = {}
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
                dist = (csub["score_cvvdp_imazen_v0_0_1"] - L).abs()
                idx = dist.idxmin()
                bd = float(dist.loc[idx])
                if bd > tolerance:
                    continue
                best = csub.loc[idx]
                per_codec_picks[codec][L] = {
                    "q": int(best["q"]),
                    "cvvdp": float(best["score_cvvdp_imazen_v0_0_1"]),
                    "ssim2": float(best["score_ssim2_gpu"])
                    if pd.notna(best.get("score_ssim2_gpu"))
                    else float("nan"),
                    "butter": float(best["score_butteraugli_pnorm3_gpu"])
                    if pd.notna(best.get("score_butteraugli_pnorm3_gpu"))
                    else float("nan"),
                    "feat": [
                        float(best[c]) if pd.notna(best[c]) else 0.0 for c in feature_cols
                    ],
                }

        for L in pivot_levels:
            valid_codecs = [c for c in codecs_in_img if L in per_codec_picks[c]]
            for i in range(len(valid_codecs)):
                for j in range(i + 1, len(valid_codecs)):
                    ca = valid_codecs[i]
                    cb = valid_codecs[j]
                    a = per_codec_picks[ca][L]
                    b = per_codec_picks[cb][L]
                    cvvdp_diff = abs(a["cvvdp"] - b["cvvdp"])
                    # cvvdp scale [0..10]; weight maps inverse distance.
                    # 0.1 offset prevents 100x weights when cvvdp_diff is tiny.
                    weight = float(1.0 / (cvvdp_diff + 0.1))
                    if weight > 10.0:
                        weight = 10.0
                    pair_count[(ca, cb)] = pair_count.get((ca, cb), 0) + 1
                    # SIGN FIX (2026-05-20): the trainer's cross-codec rank-preserve
                    # term reads `butter_a` / `butter_b` and assumes the butter
                    # convention (LOWER = HIGHER quality). cvvdp is the OPPOSITE
                    # convention (HIGHER = HIGHER quality). Store NEGATED cvvdp in
                    # the butter_a/butter_b slots so the trainer sees the right
                    # sign: butter_diff = (-cvvdp_a) - (-cvvdp_b) = cvvdp_b - cvvdp_a.
                    # Δb > 0 ⇒ cvvdp_b > cvvdp_a ⇒ A is worse quality ⇒ y_a < y_b ✓.
                    pairs.append(
                        {
                            "ref_basename": ref,
                            "codec_a": ca,
                            "q_a": a["q"],
                            "codec_b": cb,
                            "q_b": b["q"],
                            "ssim2_level": L,  # V11/V8 schema slot — re-purposed for cvvdp.
                            "ssim2_a": a["cvvdp"],
                            "ssim2_b": b["cvvdp"],
                            "ssim2_diff": cvvdp_diff,
                            "cvvdp_a": a["cvvdp"],
                            "cvvdp_b": b["cvvdp"],
                            "butter_pnorm3_a": a["butter"],
                            "butter_pnorm3_b": b["butter"],
                            "row_weight": weight,
                            "butter_level": L,
                            "butter_a": -a["cvvdp"],  # NEGATED for trainer-convention.
                            "butter_b": -b["cvvdp"],  # NEGATED for trainer-convention.
                            "fa": a["feat"],
                            "fb": b["feat"],
                        }
                    )

    print(f"  built {len(pairs)} equivalence pairs")
    print(f"  pair counts:")
    for (ca, cb), c in sorted(pair_count.items()):
        print(f"    {ca:>10s} <-> {cb:<10s}  {c:>6d}")
    return pd.DataFrame(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/multi_codec_372col_full.parquet"),
        help="V11-DECODER-FIX 4-codec × 372-feat full parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate"),
    )
    parser.add_argument(
        "--n-features", type=int, default=372,
    )
    parser.add_argument(
        "--cvvdp-tolerance", type=float, default=0.4,
        help="anchor band tolerance in cvvdp units (default ±0.4 per task brief)",
    )
    parser.add_argument(
        "--cvvdp-eq-tolerance", type=float, default=0.2,
        help="cross-codec equivalence tolerance in cvvdp units",
    )
    parser.add_argument(
        "--skip-equivalence", action="store_true",
    )
    parser.add_argument(
        "--distribution-md",
        type=Path,
        default=None,
        help="if set, emit cvvdp distribution report to this markdown file",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_input(args.input_parquet, args.n_features)

    # Distribution report
    dist_md_path = args.distribution_md or (args.out_dir / "cvvdp_distribution.md")
    cvvdp_distribution_report(df, dist_md_path)

    # Phase 2: anchor rows
    print()
    df_anchors = build_cvvdp_anchor_rows(
        df, ANCHOR_BANDS_V12_CVVDP, args.n_features, tolerance=args.cvvdp_tolerance
    )
    write_anchor_parquet(
        df_anchors,
        args.out_dir / "anchors_cvvdp_372col.parquet",
        args.n_features,
    )

    # Phase 3: cross-codec equivalence pairs.
    # Pivot at mid-range cvvdp levels (excluding the saturated extremes).
    if not args.skip_equivalence:
        print()
        pivot_levels = [9.85, 9.65, 9.30, 8.50, 7.50, 6.50]
        df_pairs = build_cvvdp_cross_codec_equivalence(
            df, pivot_levels, args.cvvdp_eq_tolerance, args.n_features
        )
        write_equivalence_parquet(
            df_pairs,
            args.out_dir / "cross_codec_equivalence_cvvdp_372col.parquet",
            args.n_features,
        )


if __name__ == "__main__":
    main()
