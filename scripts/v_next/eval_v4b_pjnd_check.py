#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V4 multi-codec PJND score check (2026-05-19).

For each V4 bake in <v4_dir>, predict score for every (image, codec) row
in the multi-codec PJND anchor parquet, then report:
  - Per-bake mean score across all 4000 anchor rows (target=63.0)
  - Per-bake per-codec mean score (4 means)
  - Per-bake cross-codec score std PER SOURCE (then averaged across sources)
  - Per-bake aggregate score std across the 4000 rows

The "cross-codec score std per source" is the key gate metric: if the
network is correctly anchored, all 4 codecs should predict ~63 at each
source's PJND-q, so the std across codecs WITHIN one source should be
small (<5 score units → pass).

Uses `ensemble_score_rows` (extended in EXP-CROSS-CODEC-V4 to honor
`zentrain.tanh_output_head` metadata) for the Rust-side scoring.

Usage:
    python3 scripts/v_next/eval_v4b_pjnd_check.py /mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ANCHOR_PATH = Path(
    "/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/anchors_multi_codec_372col.parquet"
)
SCORE_BIN = Path(
    "/home/lilith/work/zen/zensim/target/release/ensemble_score_rows"
)


def score_bake(bake_path: Path, anchor_path: Path) -> np.ndarray:
    """Run ensemble_score_rows on the anchor parquet, return per-row scores."""
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    result = subprocess.run(
        [str(SCORE_BIN), "--bake", str(bake_path), "--parquet", str(anchor_path), "--output", tmp_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"score_rows stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"ensemble_score_rows failed: {result.returncode}")
    scores: list[float] = []
    with open(tmp_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                scores.append(float(parts[2]))
            except ValueError:
                scores.append(float("nan"))
    Path(tmp_path).unlink(missing_ok=True)
    return np.asarray(scores, dtype=np.float64)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: eval_v4b_pjnd_check.py <v4_dir>", file=sys.stderr)
        return 2
    v4_dir = Path(sys.argv[1])
    bakes = sorted(v4_dir.glob("cc4v4b_*.bin"))
    if not bakes:
        print(f"no cc4v4b_*.bin under {v4_dir}", file=sys.stderr)
        return 1

    print(f"loading anchor parquet: {ANCHOR_PATH}")
    df = pq.read_table(ANCHOR_PATH).to_pandas()
    print(f"  n={len(df)}, codecs={sorted(df['codec'].unique())}, sources={df['ref_basename'].nunique()}")

    codecs_arr = df["codec"].to_numpy()
    sources_arr = df["ref_basename"].to_numpy()

    summary = []

    for bake_path in bakes:
        name = bake_path.stem
        print(f"\n=== {name} ===")
        scores = score_bake(bake_path, ANCHOR_PATH)
        if len(scores) != len(df):
            print(f"  ERROR: got {len(scores)} scores, expected {len(df)}", file=sys.stderr)
            continue

        valid = np.isfinite(scores)
        if not valid.all():
            n_bad = (~valid).sum()
            print(f"  WARN: {n_bad} non-finite scores; dropping")

        # Per-codec mean
        per_codec_mean = {}
        per_codec_std = {}
        for codec in sorted(np.unique(codecs_arr)):
            mask = (codecs_arr == codec) & valid
            if mask.sum() == 0:
                continue
            per_codec_mean[codec] = float(scores[mask].mean())
            per_codec_std[codec] = float(scores[mask].std())

        # Cross-codec std per source
        cross_codec_std_per_source = []
        for source in np.unique(sources_arr):
            src_mask = (sources_arr == source) & valid
            if src_mask.sum() < 2:
                continue
            scores_at_src = scores[src_mask]
            cross_codec_std_per_source.append(scores_at_src.std())
        cc = np.asarray(cross_codec_std_per_source) if cross_codec_std_per_source else np.array([np.nan])

        agg_mean = float(np.nanmean(scores))
        agg_std = float(np.nanstd(scores))

        cc_median = float(np.nanmedian(cc))
        cc_mean = float(np.nanmean(cc))
        cc_p25 = float(np.nanpercentile(cc, 25))
        cc_p75 = float(np.nanpercentile(cc, 75))
        cc_p95 = float(np.nanpercentile(cc, 95))

        print(f"  aggregate mean score:    {agg_mean:.3f} (target 63.000)")
        print(f"  aggregate score std:     {agg_std:.3f}")
        print(f"  per-codec mean: {' '.join(f'{c}={m:.2f}' for c, m in per_codec_mean.items())}")
        print(f"  per-codec std:  {' '.join(f'{c}={s:.2f}' for c, s in per_codec_std.items())}")
        print(f"  cross-codec score std PER SOURCE:")
        print(f"    median: {cc_median:.3f}")
        print(f"    p25:    {cc_p25:.3f}")
        print(f"    p75:    {cc_p75:.3f}")
        print(f"    p95:    {cc_p95:.3f}")
        print(f"    mean:   {cc_mean:.3f}")

        summary.append({
            "name": name,
            "agg_mean": agg_mean,
            "agg_std": agg_std,
            "per_codec_mean": per_codec_mean,
            "cc_std_median": cc_median,
            "cc_std_mean": cc_mean,
            "cc_std_p95": cc_p95,
        })

    # Write summary
    out_md = v4_dir / "v4b_pjnd_check.md"
    with open(out_md, "w") as f:
        f.write("# EXP-CROSS-CODEC-V4 multi-codec PJND score check\n\n")
        f.write("Target: each (source, codec) at PJND-q should score 63.0 ± a few units.\n")
        f.write("Gate (relaxed start, per task brief): cross-codec score std per source ≤ 5.0.\n\n")
        f.write("| bake | agg_mean | agg_std | cc_std_median | cc_std_mean | cc_std_p95 | gate (cc_std_median ≤ 5) |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | :-: |\n")
        for s in summary:
            gate = "PASS" if s["cc_std_median"] <= 5.0 else "FAIL"
            f.write(
                f"| {s['name']} | {s['agg_mean']:.2f} | {s['agg_std']:.2f} | "
                f"{s['cc_std_median']:.2f} | {s['cc_std_mean']:.2f} | {s['cc_std_p95']:.2f} | {gate} |\n"
            )
        f.write("\n## Per-codec mean per bake\n\n")
        for s in summary:
            f.write(f"### {s['name']}\n")
            f.write(f"  agg_mean = {s['agg_mean']:.3f}, agg_std = {s['agg_std']:.3f}\n\n")
            f.write(f"  per-codec mean:\n")
            for c, m in s["per_codec_mean"].items():
                f.write(f"  - {c}: {m:.3f}\n")
            f.write("\n")
    print(f"\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
