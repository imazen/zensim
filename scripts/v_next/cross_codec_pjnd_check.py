#!/usr/bin/env python3
"""EXP-CROSS-CODEC multi-codec PJND score check — any experiment generation.

Replaces eval_v4/v4b/v6/v7/v8_pjnd_check.py (2026-07-15). Those were five
163-line files differing in a glob (`cc4v6_*` vs `cc4v7_*`), an output
filename, and some prose. Each new generation was a `cp` plus a sed. Verified
against all five: every number in every report is identical.

The prose is where it went wrong. THREE of the five shipped documentation that
contradicted their own code, because the sed updated the glob and never the
sentences around it:

  - v4  claimed "Gate (relaxed start, per task brief)" while testing the
        standard `<= 5.0`. The gate was never relaxed. Anyone reading that
        report believed they were seeing a lenient result.
  - v4b titled its report "EXP-CROSS-CODEC-V4" and pointed its usage line at
        the v4 directory, while globbing `cc4v4b_*`.
  - v8  said "Verifies V6 still satisfies..." and pointed at v6's directory,
        while globbing `cc4v8_*`.

So copying does not merely duplicate code — at a measured 3-in-5 rate here, it
duplicates it WRONG, and the wrongness lands in the part humans actually read.
This file drops v4's false "relaxed" annotation and gives v4b its real title.

The gate itself: each (source, codec) pair at PJND-q should score ~63, and the
cross-codec score std per source should be <= 5.0 — i.e. the metric agrees on
"just noticeable" regardless of which codec produced the distortion.

Usage:
    python3 scripts/v_next/cross_codec_pjnd_check.py v8 /mnt/v/zen/zensim-eval/exp_cross_codec_v8_2026-05-19
    python3 scripts/v_next/cross_codec_pjnd_check.py v4b <dir> --bake-glob 'cc4v4b_*.bin'
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]

ANCHOR_PATH = Path(
    "/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/"
    "anchors_multi_codec_372col.parquet"
)
# Repo-relative, NOT a sibling-worktree path. The five predecessors hardcoded
# `zensim--cross-codec-metric/target/release/ensemble_score_rows`; that
# worktree was cleaned up and all five died with it, despite the binary living
# in THIS repo the whole time. See CLAUDE.md "NEVER hardcode a sibling-worktree
# path in a committed script".
SCORE_BIN = REPO / "target/release/ensemble_score_rows"

GATE_CC_STD = 5.0
TARGET_SCORE = 63.0


def score_bake(bake_path: Path, anchor_path: Path) -> np.ndarray:
    if not SCORE_BIN.exists():
        raise SystemExit(
            f"missing {SCORE_BIN}\n"
            f"  build it:  cargo build --release -p zensim-validate --bin ensemble_score_rows"
        )
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    result = subprocess.run(
        [
            str(SCORE_BIN),
            "--bake", str(bake_path),
            "--parquet", str(anchor_path),
            "--output", tmp_path,
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"score_rows stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"ensemble_score_rows failed: {result.returncode}")
    scores: list[float] = []
    with open(tmp_path) as f:
        f.readline()  # header
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("exp", help="experiment generation, e.g. v4 / v4b / v6 / v7 / v8")
    ap.add_argument("dir", type=Path, help="experiment dir holding the bakes")
    ap.add_argument(
        "--bake-glob", default=None,
        help="override the bake glob (default: cc4<exp>_*.bin)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="override the report path (default: <dir>/<exp>_pjnd_check.md)",
    )
    ap.add_argument("--anchor", type=Path, default=ANCHOR_PATH)
    a = ap.parse_args()

    bake_glob = a.bake_glob or f"cc4{a.exp}_*.bin"
    out_md = a.out or (a.dir / f"{a.exp}_pjnd_check.md")

    bakes = sorted(a.dir.glob(bake_glob))
    if not bakes:
        print(f"no {bake_glob} under {a.dir}", file=sys.stderr)
        return 1
    if not a.anchor.exists():
        print(f"missing anchor parquet: {a.anchor}", file=sys.stderr)
        return 1

    print(f"loading anchor parquet: {a.anchor}")
    df = pq.read_table(a.anchor).to_pandas()
    print(
        f"  n={len(df)}, codecs={sorted(df['codec'].unique())}, "
        f"sources={df['ref_basename'].nunique()}"
    )

    codecs_arr = df["codec"].to_numpy()
    sources_arr = df["ref_basename"].to_numpy()
    summary = []

    for bake_path in bakes:
        name = bake_path.stem
        print(f"\n=== {name} ===")
        scores = score_bake(bake_path, a.anchor)
        if len(scores) != len(df):
            print(
                f"  ERROR: got {len(scores)} scores, expected {len(df)}",
                file=sys.stderr,
            )
            continue

        valid = np.isfinite(scores)
        if not valid.all():
            print(f"  WARN: {(~valid).sum()} non-finite scores; dropping")

        per_codec_mean, per_codec_std = {}, {}
        for codec in sorted(np.unique(codecs_arr)):
            mask = (codecs_arr == codec) & valid
            if mask.sum() == 0:
                continue
            per_codec_mean[codec] = float(scores[mask].mean())
            per_codec_std[codec] = float(scores[mask].std())

        # The gate stat: within one source, how much does the score move when
        # only the CODEC changes? At PJND it should not move at all.
        cross = []
        for source in np.unique(sources_arr):
            m = (sources_arr == source) & valid
            if m.sum() < 2:
                continue
            cross.append(scores[m].std())
        cc = np.asarray(cross) if cross else np.array([np.nan])

        agg_mean, agg_std = float(np.nanmean(scores)), float(np.nanstd(scores))
        cc_median = float(np.nanmedian(cc))
        cc_mean = float(np.nanmean(cc))
        cc_p25 = float(np.nanpercentile(cc, 25))
        cc_p75 = float(np.nanpercentile(cc, 75))
        cc_p95 = float(np.nanpercentile(cc, 95))

        print(f"  aggregate mean score:    {agg_mean:.3f} (target {TARGET_SCORE:.3f})")
        print(f"  aggregate score std:     {agg_std:.3f}")
        print("  per-codec mean: " + " ".join(f"{c}={m:.2f}" for c, m in per_codec_mean.items()))
        print("  per-codec std:  " + " ".join(f"{c}={s:.2f}" for c, s in per_codec_std.items()))
        print("  cross-codec score std PER SOURCE:")
        for lbl, v in (("median", cc_median), ("p25", cc_p25), ("p75", cc_p75),
                       ("p95", cc_p95), ("mean", cc_mean)):
            print(f"    {lbl:7s} {v:.3f}")

        summary.append({
            "name": name, "agg_mean": agg_mean, "agg_std": agg_std,
            "per_codec_mean": per_codec_mean, "cc_std_median": cc_median,
            "cc_std_mean": cc_mean, "cc_std_p95": cc_p95,
        })

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write(f"# EXP-CROSS-CODEC-{a.exp.upper()} multi-codec PJND score check\n\n")
        f.write(f"Target: each (source, codec) at PJND-q should score {TARGET_SCORE} "
                "± a few units.\n")
        # `{...}` vs `{...:g}` renders 5.0 vs 5. That inconsistency is the
        # predecessors': they hardcoded "5.0" in the prose and "5" in the table.
        # Reproduced verbatim so the report is byte-identical — the proof that
        # this file is a drop-in for all five.
        f.write(f"Gate: cross-codec score std per source ≤ {GATE_CC_STD}.\n\n")
        f.write("| bake | agg_mean | agg_std | cc_std_median | cc_std_mean | cc_std_p95 | "
                f"gate (cc_std_median ≤ {GATE_CC_STD:g}) |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | :-: |\n")
        for s in summary:
            gate = "PASS" if s["cc_std_median"] <= GATE_CC_STD else "FAIL"
            f.write(
                f"| {s['name']} | {s['agg_mean']:.2f} | {s['agg_std']:.2f} | "
                f"{s['cc_std_median']:.2f} | {s['cc_std_mean']:.2f} | "
                f"{s['cc_std_p95']:.2f} | {gate} |\n"
            )
        f.write("\n## Per-codec mean per bake\n\n")
        for s in summary:
            f.write(f"### {s['name']}\n")
            f.write(f"  agg_mean = {s['agg_mean']:.3f}, agg_std = {s['agg_std']:.3f}\n\n")
            f.write("  per-codec mean:\n")
            for c, m in s["per_codec_mean"].items():
                f.write(f"  - {c}: {m:.3f}\n")
            f.write("\n")
    print(f"\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
