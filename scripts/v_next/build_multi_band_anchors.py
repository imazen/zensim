#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V5 multi-band piecewise anchor builder (2026-05-19).

V4 used a single multi-codec PJND anchor at score=63 (butter_pnorm3=1.5)
which pulled all outputs toward 63 and structurally collapsed the y_score
range — V4 best range was 35.25 vs gate 50.

V5 fix: piecewise multi-band anchor. For every (source, codec), emit
ANCHOR_BANDS rows where each row targets a different butter level
mapped to a different score target. This gives the network "calibration
landmarks" across [0, 100] while preserving the V4 cross-codec parity
mechanism at each band.

The 6 bands span [butter=0.3 → score=90] (near-lossless) through
[butter=6.0 → score=10] (heavy distortion), with butter=1.5 → score=63
preserving V4's existing PJND anchor.

For each (source, codec, band), we find the q whose butter_pnorm3 is
closest to the band's butter_target via argmin |delta|. Rows are FILTERED
when no q within the available q-sweep is within `--max-distance` of the
band target (default 0.5) — i.e., that source/codec can't achieve that
butter level. Filter rates are logged.

Output schema (same as V4 anchor parquet, plus `butter_target` column):
  ref_basename, anchor_source, human_score, anchor_weight,
  q, butter_pnorm3, butter_target, target_score, codec, f0..f371

The trainer reads `target_score` per-row when present (V5 trainer change),
falling back to the `--anchor-target-score` CLI default for V4-style
single-band parquets that don't have the column.

Run with:
    python3 scripts/v_next/build_multi_band_anchors.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# (butter_pnorm3_target, target_score) — same 6 bands as
# project_v5_piecewise_anchor_design.md.
ANCHOR_BANDS: list[tuple[float, float]] = [
    (0.3, 90.0),  # near-lossless
    (0.8, 75.0),
    (1.5, 63.0),  # PJND (preserves V4's anchor)
    (2.5, 45.0),
    (4.0, 25.0),
    (6.0, 10.0),  # heavy distortion
]

CODECS = ["zenjpeg", "zenwebp", "zenavif", "zenjxl"]
BUTTER_DIR = Path("/mnt/v/zen/picker-training/2026-05-19/butter")
OUT_PATH = Path(
    "/mnt/v/zen/zensim-training/2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet"
)


def build(out_path: Path, max_distance: float) -> None:
    rows: list[dict] = []
    feature_cols = [f"f{i}" for i in range(372)]

    # Per-(codec, band) filter stats.
    stats: dict[tuple[str, float], dict[str, int]] = {}

    for codec in CODECS:
        path = BUTTER_DIR / f"{codec}.parquet"
        if not path.exists():
            print(f"  skip {codec}: parquet missing at {path}")
            continue
        df = pq.read_table(path).to_pandas()
        n_sources = df["ref_basename"].nunique()
        print(f"{codec}: loaded {len(df)} rows ({n_sources} sources)")

        for source, group in df.groupby("ref_basename"):
            for butter_target, score_target in ANCHOR_BANDS:
                key = (codec, butter_target)
                slot = stats.setdefault(key, {"emitted": 0, "filtered": 0})

                distances = (group["butter_pnorm3"] - butter_target).abs()
                idx = distances.idxmin()
                best = group.loc[idx]
                best_distance = float(distances.loc[idx])

                if best_distance > max_distance:
                    slot["filtered"] += 1
                    continue
                slot["emitted"] += 1

                row = {
                    "ref_basename": str(source),
                    "anchor_source": f"{codec}_band_b{butter_target}_s{score_target:.0f}",
                    "human_score": score_target,  # placeholder mirror
                    "anchor_weight": 1.0,
                    "q": int(best["q"]),
                    "butter_pnorm3": float(best["butter_pnorm3"]),
                    "butter_target": float(butter_target),
                    "target_score": float(score_target),
                    "codec": codec,
                }
                for col in feature_cols:
                    if col in best.index:
                        val = best[col]
                        row[col] = float(val) if val is not None else 0.0
                    else:
                        row[col] = 0.0
                rows.append(row)

    print(f"\ntotal anchor rows: {len(rows)}")
    if not rows:
        raise SystemExit("no anchor rows built")

    cols = {
        "ref_basename": pa.array([r["ref_basename"] for r in rows], type=pa.string()),
        "anchor_source": pa.array([r["anchor_source"] for r in rows], type=pa.string()),
        "human_score": pa.array([r["human_score"] for r in rows], type=pa.float64()),
        "anchor_weight": pa.array([r["anchor_weight"] for r in rows], type=pa.float64()),
        "q": pa.array([r["q"] for r in rows], type=pa.int64()),
        "butter_pnorm3": pa.array([r["butter_pnorm3"] for r in rows], type=pa.float64()),
        "butter_target": pa.array([r["butter_target"] for r in rows], type=pa.float64()),
        "target_score": pa.array([r["target_score"] for r in rows], type=pa.float64()),
        "codec": pa.array([r["codec"] for r in rows], type=pa.string()),
    }
    for col in feature_cols:
        cols[col] = pa.array([r[col] for r in rows], type=pa.float32())
    tbl = pa.table(cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out_path, compression="zstd", compression_level=15)
    print(
        f"wrote {out_path} "
        f"({out_path.stat().st_size / 1024:.0f} KiB, {tbl.num_rows} rows × {tbl.num_columns} cols)"
    )

    # Print filter stats per (codec, band).
    print("\nper-(codec, band) filter stats:")
    print("  codec       butter_target  score_target  emitted   filtered  filter%")
    print("  -----       -------------  ------------  -------   --------  -------")
    for codec in CODECS:
        for butter_target, score_target in ANCHOR_BANDS:
            slot = stats.get((codec, butter_target))
            if slot is None:
                continue
            total = slot["emitted"] + slot["filtered"]
            pct = 100.0 * slot["filtered"] / total if total else 0.0
            print(
                f"  {codec:10s}  {butter_target:13.2f}  {score_target:12.1f}  "
                f"{slot['emitted']:7d}   {slot['filtered']:8d}  {pct:6.2f}%"
            )

    # Print achievement stats per (codec, band) — median achieved butter
    # vs target, to surface how well each codec covers each band.
    print("\nper-(codec, band) butter_pnorm3 achievement (median actual vs target):")
    print("  codec       butter_target   median_actual   median_delta")
    print("  -----       -------------   -------------   ------------")
    for codec in CODECS:
        for butter_target, _ in ANCHOR_BANDS:
            sub = [
                r["butter_pnorm3"]
                for r in rows
                if r["codec"] == codec and r["butter_target"] == butter_target
            ]
            if not sub:
                continue
            sub.sort()
            med = sub[len(sub) // 2]
            print(
                f"  {codec:10s}  {butter_target:13.2f}   {med:13.4f}   {med - butter_target:+12.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.5,
        help="drop anchor rows whose closest q's butter_pnorm3 is more than this from the band target",
    )
    args = parser.parse_args()
    build(args.out, args.max_distance)
