#!/usr/bin/env python3
"""Measure Tuner v10 cross-codec consistency baseline.

Scores both sides of every cross-codec equivalence pair in
/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet
through `predict_features_with_bake` using `zensim/weights/v_tuner_v10_2026-05-20.bin`,
then reports per-anchor (butter_level) and per-codec-pair cross-codec score deviation.

A "perfectly cross-codec-consistent" metric would yield |score_a - score_b| = 0
at every matched anchor. Real metrics produce a stddev / p50 / p90 spread per anchor.

Outputs:
- benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md  (markdown report)
- /mnt/v/output/zensim/tuner_v10_cross_codec_baseline_2026-05-24/scores.parquet  (raw)
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BAKE = ROOT / "zensim/weights/v_tuner_v10_2026-05-20.bin"
PARQUET = Path(
    "/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"
)
PREDICT = ROOT / "target/release/predict_features_with_bake"
N_FEATURES = 372


def score_block(features: np.ndarray, bake: Path) -> np.ndarray:
    """features: (n_rows, n_features) float32. Returns (n_rows,) float64 scores."""
    assert features.dtype == np.float32
    n_rows, n_features = features.shape
    buf = bytearray()
    buf += struct.pack("<II", n_features, n_rows)
    buf += features.tobytes(order="C")
    with tempfile.NamedTemporaryFile(suffix=".features.bin", delete=False) as f:
        f.write(buf)
        feats_path = f.name
    try:
        out = subprocess.check_output(
            [
                str(PREDICT),
                "--bake",
                str(bake),
                "--bake-post",
                "raw",
                "--features-file",
                feats_path,
            ],
        )
    finally:
        os.unlink(feats_path)
    scores = np.array([float(x) for x in out.decode().split()], dtype=np.float64)
    assert scores.shape == (n_rows,), f"got {scores.shape}, expected ({n_rows},)"
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", type=Path, default=BAKE)
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md",
    )
    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=Path(
            "/mnt/v/output/zensim/tuner_v10_cross_codec_baseline_2026-05-24/scores.parquet"
        ),
    )
    ap.add_argument(
        "--bake-post",
        default="clamp",
        help="bake post-processing; clamp matches what codec consumers see",
    )
    args = ap.parse_args()

    if not args.bake.exists():
        sys.exit(f"bake not found: {args.bake}")
    if not args.parquet.exists():
        sys.exit(f"parquet not found: {args.parquet}")
    if not PREDICT.exists():
        sys.exit(
            f"predict tool not built: {PREDICT}\n"
            "  cargo build --release --bin predict_features_with_bake -p zensim-validate"
        )

    print(f"[1/4] loading {args.parquet} ...")
    table = pq.read_table(args.parquet)
    n = table.num_rows
    print(f"      {n} cross-codec eq pairs, {table.num_columns} columns")

    fa_cols = [f"fa_{i}" for i in range(N_FEATURES)]
    fb_cols = [f"fb_{i}" for i in range(N_FEATURES)]
    missing = [c for c in fa_cols + fb_cols if c not in table.column_names]
    if missing:
        sys.exit(f"missing columns: {missing[:5]} ...")

    print(f"[2/4] extracting fa_/fb_ blocks → ({n}, {N_FEATURES}) each ...")
    fa = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in fa_cols]
    ).astype(np.float32, copy=False)
    fb = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in fb_cols]
    ).astype(np.float32, copy=False)
    assert fa.shape == (n, N_FEATURES) and fb.shape == (n, N_FEATURES)

    print(f"[3/4] scoring fa block through {args.bake.name} (bake-post={args.bake_post}) ...")
    # Use raw post; we apply clamp / spline interpretation downstream if needed.
    score_a = score_block(fa, args.bake)
    print(f"      score_a: mean={score_a.mean():.3f} sd={score_a.std():.3f} "
          f"min={score_a.min():.3f} max={score_a.max():.3f}")
    print(f"[3/4] scoring fb block ...")
    score_b = score_block(fb, args.bake)
    print(f"      score_b: mean={score_b.mean():.3f} sd={score_b.std():.3f} "
          f"min={score_b.min():.3f} max={score_b.max():.3f}")

    meta = table.select(
        ["ref_basename", "codec_a", "q_a", "codec_b", "q_b", "butter_level",
         "butter_a", "butter_b", "row_weight"]
    ).to_pandas()
    df = meta.assign(score_a=score_a, score_b=score_b)
    df["score_delta"] = df["score_a"] - df["score_b"]
    df["abs_delta"] = df["score_delta"].abs()
    df["codec_pair"] = df["codec_a"] + "_" + df["codec_b"]

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, compression="zstd", compression_level=15)
    print(f"      wrote raw scores: {args.out_parquet}  ({args.out_parquet.stat().st_size//1024} KB)")

    print(f"[4/4] aggregating + writing markdown ...")

    def pct(s, p):
        return float(np.percentile(s, p)) if len(s) else float("nan")

    overall = {
        "n": len(df),
        "median_abs_delta": float(df["abs_delta"].median()),
        "p90_abs_delta": pct(df["abs_delta"], 90),
        "p99_abs_delta": pct(df["abs_delta"], 99),
        "max_abs_delta": float(df["abs_delta"].max()),
        "stddev_delta": float(df["score_delta"].std()),
    }

    per_anchor = (
        df.groupby("butter_level")
          .agg(
              n=("abs_delta", "size"),
              p50=("abs_delta", "median"),
              p90=("abs_delta", lambda s: pct(s, 90)),
              p99=("abs_delta", lambda s: pct(s, 99)),
              max=("abs_delta", "max"),
              score_a_mean=("score_a", "mean"),
              score_b_mean=("score_b", "mean"),
          )
          .round(3)
          .reset_index()
          .sort_values("butter_level")
    )

    per_pair = (
        df.groupby("codec_pair")
          .agg(
              n=("abs_delta", "size"),
              p50=("abs_delta", "median"),
              p90=("abs_delta", lambda s: pct(s, 90)),
              p99=("abs_delta", lambda s: pct(s, 99)),
              max=("abs_delta", "max"),
              stddev=("score_delta", "std"),
          )
          .round(3)
          .reset_index()
          .sort_values("p50")
    )

    md = []
    md.append(f"# Tuner v10 cross-codec consistency baseline — 2026-05-24")
    md.append("")
    md.append(f"- **Bake:** `{args.bake.relative_to(ROOT)}`")
    md.append(f"- **Eq parquet:** `{args.parquet}`")
    md.append(f"- **n pairs:** {overall['n']:,}")
    md.append(f"- **bake-post:** `{args.bake_post}` (raw output, before any clamp/spline)")
    md.append("")
    md.append("## Overall cross-codec score deviation")
    md.append("")
    md.append("| stat | value |")
    md.append("|---|---:|")
    md.append(f"| n | {overall['n']:,} |")
    md.append(f"| median \\|Δ\\| | {overall['median_abs_delta']:.3f} |")
    md.append(f"| p90 \\|Δ\\| | {overall['p90_abs_delta']:.3f} |")
    md.append(f"| p99 \\|Δ\\| | {overall['p99_abs_delta']:.3f} |")
    md.append(f"| max \\|Δ\\| | {overall['max_abs_delta']:.3f} |")
    md.append(f"| stddev(Δ) | {overall['stddev_delta']:.3f} |")
    md.append("")
    md.append("Δ = score_a − score_b on matched butter-level anchor pairs. "
              "Lower is better. A perfectly cross-codec-consistent metric "
              "would give Δ=0 at every anchor.")
    md.append("")
    md.append("## Per-butter-anchor breakdown")
    md.append("")
    md.append("| butter_level | n | p50 \\|Δ\\| | p90 \\|Δ\\| | p99 \\|Δ\\| | max \\|Δ\\| | mean score_a | mean score_b |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for _, r in per_anchor.iterrows():
        md.append(
            f"| {r['butter_level']:.3f} | {int(r['n']):,} | "
            f"{r['p50']:.3f} | {r['p90']:.3f} | {r['p99']:.3f} | {r['max']:.3f} | "
            f"{r['score_a_mean']:.2f} | {r['score_b_mean']:.2f} |"
        )
    md.append("")
    md.append("## Per-codec-pair breakdown")
    md.append("")
    md.append("| codec_pair | n | p50 \\|Δ\\| | p90 \\|Δ\\| | p99 \\|Δ\\| | max \\|Δ\\| | stddev(Δ) |")
    md.append("|---|--:|--:|--:|--:|--:|--:|")
    for _, r in per_pair.iterrows():
        md.append(
            f"| {r['codec_pair']} | {int(r['n']):,} | "
            f"{r['p50']:.3f} | {r['p90']:.3f} | {r['p99']:.3f} | "
            f"{r['max']:.3f} | {r['stddev']:.3f} |"
        )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("For codec-target use (zensim-target binary-searching q for "
              "a target score), the **p90 |Δ|** is the practically-relevant "
              "number: 90% of cross-codec pairs at matched perceptual quality "
              "land within this many score units. The picker integration sees "
              "this as the floor on its cross-codec dial precision.")
    md.append("")
    md.append("V11-E falsification (`benchmarks/v11_e_per_codec_falsification_2026-05-20.md`) "
              "measured Tuner v4's per-codec α spread at 0.7 score units — small, "
              "and post-spline residual cross-codec stddev at 4.5-7.0. The numbers "
              "here are the matched-anchor analog (same input → expected same "
              "score across codecs).")
    md.append("")
    md.append("## Reproducibility")
    md.append("")
    md.append("```sh")
    md.append("python3 scripts/v_next/measure_tuner_v10_cross_codec.py \\")
    md.append("    --bake zensim/weights/v_tuner_v10_2026-05-20.bin \\")
    md.append(f"    --parquet {args.parquet} \\")
    md.append(f"    --out-parquet {args.out_parquet} \\")
    md.append(f"    --out-md {args.out_md}")
    md.append("```")
    md.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md))
    print(f"      wrote markdown: {args.out_md}")

    # Print the key number to stdout for quick reference.
    print()
    print(f"=== HEADLINE: median |Δ|={overall['median_abs_delta']:.3f}, "
          f"p90 |Δ|={overall['p90_abs_delta']:.3f}, "
          f"p99 |Δ|={overall['p99_abs_delta']:.3f} (n={overall['n']:,}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
