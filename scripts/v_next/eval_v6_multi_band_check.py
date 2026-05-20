#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V6 multi-band cross-codec consistency check (2026-05-19).

V6 mirrors V5's multi-band gate check but globs cc4v6_*.bin and writes
v6_multi_band_check.md. V6 increases anchor pressure (anchor_loss_weight
and step_p) — we expect the band targets to materialize as actual
output means now, while ideally retaining V5's tight cc_std at every
band.

For each V6 bake in <v6_dir>, predict score for every (image, codec,
band) row in the multi-band anchor parquet, then for each band:
  - per-codec mean score (4 means per band — target = band's target_score)
  - cross-codec score std per source (then median across sources)
  - flag PASS/FAIL on `cc_std_median ≤ 5.0`

Usage:
    python3 scripts/v_next/eval_v6_multi_band_check.py /mnt/v/zen/zensim-eval/exp_cross_codec_v6_2026-05-19
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ANCHOR_PATH = Path(
    "/mnt/v/zen/zensim-training/2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet"
)
SCORE_BIN = Path(
    "/home/lilith/work/zen/zensim--cross-codec-metric/target/release/ensemble_score_rows"
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
        print("usage: eval_v6_multi_band_check.py <v6_dir>", file=sys.stderr)
        return 2
    v6_dir = Path(sys.argv[1])
    bakes = sorted(v6_dir.glob("cc4v6_*.bin"))
    if not bakes:
        print(f"no cc4v6_*.bin under {v6_dir}", file=sys.stderr)
        return 1

    print(f"loading anchor parquet: {ANCHOR_PATH}")
    df = pq.read_table(ANCHOR_PATH).to_pandas()
    print(
        f"  n={len(df)}, codecs={sorted(df['codec'].unique())}, "
        f"sources={df['ref_basename'].nunique()}, bands={sorted(df['butter_target'].unique())}"
    )

    codecs_arr = df["codec"].to_numpy()
    sources_arr = df["ref_basename"].to_numpy()
    bands_arr = df["butter_target"].to_numpy()
    targets_arr = df["target_score"].to_numpy()

    unique_bands = sorted(df["butter_target"].unique())

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

        per_band_rows = []
        for band in unique_bands:
            band_mask = (bands_arr == band) & valid
            if band_mask.sum() == 0:
                per_band_rows.append({
                    "band": band,
                    "target": float("nan"),
                    "n": 0,
                    "achieved_mean": float("nan"),
                    "achieved_std": float("nan"),
                    "cc_std_median": float("nan"),
                    "cc_std_mean": float("nan"),
                    "cc_std_p95": float("nan"),
                    "gate": "no-data",
                })
                continue

            target = float(targets_arr[band_mask][0])
            band_scores = scores[band_mask]
            band_sources = sources_arr[band_mask]
            band_codecs = codecs_arr[band_mask]

            achieved_mean = float(band_scores.mean())
            achieved_std = float(band_scores.std())

            cc_per_src = []
            for source in np.unique(band_sources):
                src_mask = band_sources == source
                if src_mask.sum() < 2:
                    continue
                src_scores = band_scores[src_mask]
                cc_per_src.append(src_scores.std())
            cc_arr = np.asarray(cc_per_src) if cc_per_src else np.array([np.nan])

            cc_median = float(np.nanmedian(cc_arr))
            cc_mean = float(np.nanmean(cc_arr))
            cc_p95 = float(np.nanpercentile(cc_arr, 95))
            gate = "PASS" if cc_median <= 5.0 else "FAIL"

            per_band_rows.append({
                "band": band,
                "target": target,
                "n": int(band_mask.sum()),
                "achieved_mean": achieved_mean,
                "achieved_std": achieved_std,
                "cc_std_median": cc_median,
                "cc_std_mean": cc_mean,
                "cc_std_p95": cc_p95,
                "gate": gate,
            })

            print(
                f"  band butter={band:.2f} target={target:5.1f} "
                f"n={int(band_mask.sum()):5d} achieved={achieved_mean:6.2f}±{achieved_std:5.2f} "
                f"cc_std_median={cc_median:5.2f} (p95={cc_p95:5.2f}) {gate}"
            )

        passing_bands = sum(1 for r in per_band_rows if r["gate"] == "PASS")
        total_bands = sum(1 for r in per_band_rows if r["gate"] not in ("no-data",))
        all_pass = passing_bands == total_bands and total_bands > 0

        summary.append({
            "name": name,
            "passing_bands": passing_bands,
            "total_bands": total_bands,
            "all_pass": all_pass,
            "per_band": per_band_rows,
        })

    out_md = v6_dir / "v6_multi_band_check.md"
    with open(out_md, "w") as f:
        f.write("# EXP-CROSS-CODEC-V6 multi-band cross-codec consistency check\n\n")
        f.write(
            "For each anchor band (butter ∈ {0.3, 0.8, 1.5, 2.5, 4.0, 6.0}),\n"
            "measure cross-codec score std per source within that band.\n"
            "Gate: cc_std_median ≤ 5.0 at EVERY band.\n\n"
        )

        f.write("## Bake-level summary\n\n")
        f.write("| bake | passing_bands | total_bands | all_pass |\n")
        f.write("| --- | ---: | ---: | :-: |\n")
        for s in summary:
            verdict = "PASS" if s["all_pass"] else "FAIL"
            f.write(
                f"| {s['name']} | {s['passing_bands']} | {s['total_bands']} | {verdict} |\n"
            )

        f.write("\n## Per-bake per-band detail\n\n")
        for s in summary:
            f.write(f"### {s['name']}\n\n")
            f.write(
                "| band (butter) | target_score | n | achieved_mean | achieved_std | "
                "cc_std_median | cc_std_p95 | gate |\n"
            )
            f.write(
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: |\n"
            )
            for r in s["per_band"]:
                f.write(
                    f"| {r['band']:.2f} | {r['target']:.1f} | {r['n']} | "
                    f"{r['achieved_mean']:.2f} | {r['achieved_std']:.2f} | "
                    f"{r['cc_std_median']:.2f} | {r['cc_std_p95']:.2f} | {r['gate']} |\n"
                )
            f.write("\n")

    print(f"\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
