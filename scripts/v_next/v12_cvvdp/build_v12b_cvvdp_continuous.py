#!/usr/bin/env python3
"""V12-B continuous-mapping cvvdp anchor (task #199 follow-up, 2026-05-20).

The V12-A band-snap substrate concentrated anchors in the upper bands
(target_score >= 50 only; 0/0/1/28 at scores 0-35) because the source
corpus's q-grid produces almost no sub-9 cvvdp. The trained bake learns
to predict scores ~[50, 100] only and collapses on CID22 + KonJND val
sets where the score distribution spans 0..100.

V12-B fixes this by mapping cvvdp DIRECTLY to a continuous target_score
per row (no band snap), preserving the same overall mapping shape as
V12-A but covering whatever range of cvvdp the corpus actually produces.
This gives the bake full anchor coverage at its trained cvvdp range,
and the eval validates whether the V11 cross-codec mechanism survives at
cvvdp pivot OR collapses due to anchor coverage limits.

## Mapping (continuous PCHIP-like)

Same 10 band knots as V12-A, but interpolated continuously between knots
(linear in cvvdp space) and extrapolated below the lowest cvvdp present
in the corpus by holding target_score = 0:

  cvvdp ≤ 3.00 → 0
  cvvdp = 5.00 → 10
  cvvdp = 6.50 → 20
  cvvdp = 7.50 → 35
  cvvdp = 8.50 → 50
  cvvdp = 9.30 → 65
  cvvdp = 9.65 → 80
  cvvdp = 9.85 → 90
  cvvdp = 9.95 → 95
  cvvdp ≥ 10.00 → 100

Linear interpolation between knots so every (image, codec, q) row gets
an anchor with a meaningful target_score reflecting its cvvdp value.

## Output

`/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate/anchors_cvvdp_372col_continuous.parquet`
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

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "v11_ssim2_v2"))
from build_v11_substrate_v2 import (  # noqa: E402
    write_anchor_parquet,
)

# Same knots as V12-A (cvvdp -> target_score).
KNOTS_CVVDP = np.array([3.00, 5.00, 6.50, 7.50, 8.50, 9.30, 9.65, 9.85, 9.95, 10.00])
KNOTS_SCORE = np.array([0.0, 10.0, 20.0, 35.0, 50.0, 65.0, 80.0, 90.0, 95.0, 100.0])


def cvvdp_to_target_score(cvvdp: np.ndarray) -> np.ndarray:
    """Continuous monotone mapping cvvdp -> target_score via linear knots."""
    return np.interp(cvvdp, KNOTS_CVVDP, KNOTS_SCORE, left=0.0, right=100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/multi_codec_372col_full.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate"),
    )
    parser.add_argument("--n-features", type=int, default=372)
    parser.add_argument(
        "--row-stride",
        type=int,
        default=1,
        help="anchor row stride — use 1 for all rows (~118k anchors), "
             "5 for ~24k (still > V11-A v4's 2471)",
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=0.5,
        help="per-row anchor weight — lower than V12-A's 1.0 because we "
             "have ~30-50x more anchor rows (118k vs 2471)",
    )
    args = parser.parse_args()

    print(f"loading {args.input_parquet}")
    df = pq.read_table(args.input_parquet).to_pandas()
    df["ref_basename"] = df["image_path"].apply(os.path.basename)

    rename_map = {f"f{i}": f"feat_{i}" for i in range(args.n_features)}
    df = df.rename(columns=rename_map)

    df_valid = df[df["score_cvvdp_imazen_v0_0_1"].notna()].copy()
    print(f"  total rows: {len(df)} cvvdp-valid: {len(df_valid)}")

    # Map cvvdp -> target_score continuously
    df_valid["target_score"] = cvvdp_to_target_score(
        df_valid["score_cvvdp_imazen_v0_0_1"].values
    )

    if args.row_stride > 1:
        df_valid = df_valid.iloc[::args.row_stride].copy()
        print(f"  after stride={args.row_stride}: {len(df_valid)} anchor rows")

    print(f"  target_score distribution:")
    print(df_valid["target_score"].describe())
    print()
    print("  per-codec mean target_score:")
    print(df_valid.groupby("codec")["target_score"].agg(["count", "min", "mean", "max"]))

    # Build anchor rows in V11/V12-A schema
    feature_cols = [f"feat_{i}" for i in range(args.n_features)]
    rows = []
    for _, r in df_valid.iterrows():
        row = {
            "ref_basename": str(r["ref_basename"]),
            "anchor_source": f"v12b_{r['codec']}_q{int(r['q'])}_cv{r['score_cvvdp_imazen_v0_0_1']:.2f}",
            "human_score": float(r["target_score"]),
            "anchor_weight": float(args.anchor_weight),
            "q": int(r["q"]),
            "ssim2_anchor": (
                float(r["score_ssim2_gpu"]) if pd.notna(r.get("score_ssim2_gpu")) else float("nan")
            ),
            "ssim2_target": float("nan"),
            "cvvdp_anchor": float(r["score_cvvdp_imazen_v0_0_1"]),
            "butter_pnorm3_anchor": (
                float(r["score_butteraugli_pnorm3_gpu"])
                if pd.notna(r.get("score_butteraugli_pnorm3_gpu"))
                else float("nan")
            ),
            "target_score": float(r["target_score"]),
            "codec": str(r["codec"]),
            "anchor_via": "cvvdp_continuous",
        }
        for c in feature_cols:
            v = r[c]
            row[c] = float(v) if pd.notna(v) else 0.0
        rows.append(row)

    df_rows = pd.DataFrame(rows)
    out_path = args.out_dir / "anchors_cvvdp_372col_continuous.parquet"
    write_anchor_parquet(df_rows, out_path, args.n_features)


if __name__ == "__main__":
    main()
