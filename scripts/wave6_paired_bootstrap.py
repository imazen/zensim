#!/usr/bin/env python3
"""Paired bootstrap over per-pair dumps — the SOTA-944 wave-6 instrument.

Two models scored on the SAME pairs must be compared with the SAME resampled
index sets; marginal per-cell CIs are the wrong test (wave 5 §"Is the crossing
real?"). This reproduces that instrument for wave 6's arms.

NO STAT MATH LIVES HERE. Every Spearman comes from `panel --batch` through
`scripts/lib/zen_stats.panel_batch_indexed` — the canonical owner in its
registered paired-bootstrap shape (the caller keeps the RNG; scipy-in-a-loop is
the banned pattern this replaces). This script only reads the dumps, asserts the
`human` column is identical across series, generates the index sets, and
reduces the returned rows to deltas.

The KonJND column is read as |SROCC| exactly as `bake_verdict` reports it, so
the deltas here are directly comparable to the campaign's KonJND rows.

    wave6_paired_bootstrap.py --dir ~/tmp/wave6/perpair --corpus cid22 \
        --series GE2 K5 EM4 SING --ref EM4 --b 2000 --seed 20260804
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.zen_stats import panel_batch_indexed  # noqa: E402


def load(p: Path) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(p, delimiter="\t", skiprows=1)
    return a[:, 0], a[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--series", nargs="+", required=True)
    ap.add_argument(
        "--ref", nargs="+", required=True, help="baseline series; every other series is compared to each"
    )
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument(
        "--band-lo", type=float, default=None,
        help="restrict to rows with human >= this (bake_verdict's 10-band cuts are "
             "width-0.10 on the [0,1] human scale, so B9 is --band-lo 0.90)")
    ap.add_argument(
        "--band-hi", type=float, default=None,
        help="restrict to rows with human < this (band B9 is closed at 1.00, so omit "
             "--band-hi for it; bake_verdict's B0..B8 are half-open [lo, hi))")
    ap.add_argument(
        "--signed", action="store_true",
        help="report SROCC with its sign (panel's `srocc_signed`) instead of |SROCC|. "
             "REQUIRED for band tails: freeze_check's F8 gate is signed, because a "
             "band whose ordering COLLAPSES must be able to score below zero.")
    a = ap.parse_args()

    human = None
    preds: dict[str, np.ndarray] = {}
    for s in a.series:
        h, p = load(a.dir / f"{s}_{a.corpus}.tsv")
        if human is None:
            human = h
        elif not np.array_equal(human, h):
            print(f"FATAL: human column of {s} differs from the first series", file=sys.stderr)
            return 2
        preds[s] = p
    assert human is not None

    # Band restriction is applied to the SHARED human column, so every series
    # keeps the same rows and the pairing the whole instrument rests on holds.
    band = ""
    if a.band_lo is not None or a.band_hi is not None:
        keep = np.ones(len(human), dtype=bool)
        if a.band_lo is not None:
            keep &= human >= a.band_lo
        if a.band_hi is not None:
            keep &= human < a.band_hi
        if not keep.any():
            print("FATAL: band selection is empty", file=sys.stderr)
            return 2
        human = human[keep]
        preds = {s: p[keep] for s, p in preds.items()}
        band = f" band=[{a.band_lo}, {a.band_hi})"
    stat = "srocc_signed" if a.signed else "srocc"
    n = len(human)
    print(f"[{a.corpus}]{band} n={n} stat={stat} series={a.series} "
          f"(human column identical across all)")

    bases = {"HUMAN": human.tolist()}
    for s in a.series:
        bases[s] = preds[s].tolist()

    rng = np.random.default_rng(a.seed)
    idx = [rng.integers(0, n, n) for _ in range(a.b)]

    jobs = [(f"{s}_full", s, "HUMAN", None) for s in a.series]
    for b, ix in enumerate(idx):
        for s in a.series:
            jobs.append((f"{s}_{b}", s, "HUMAN", ix))
    rows = panel_batch_indexed(bases, jobs, stats="srocc", timeout=7200.0)
    by = {r["label"]: float(r[stat]) for r in rows}

    print(f"\npoint estimates ({'SROCC' if a.signed else '|SROCC|'}, {a.corpus}{band}):")
    for s in a.series:
        print(f"  {s:6s} {by[s + '_full']:.6f}")

    boot = {s: np.array([by[f"{s}_{b}"] for b in range(a.b)]) for s in a.series}
    # Per-series MARGINAL interval: how well the axis resolves at all. On a thin
    # band this is the number that decides whether any delta could ever clear a
    # floor, so it is printed before the deltas rather than inferred from them.
    print(f"\nmarginal bootstrap interval per series (B={a.b}, NOT a comparison):")
    print(f"{'series':28s} {'point':>10s} {'2.5%':>10s} {'97.5%':>10s} {'sd':>10s}")
    for s in a.series:
        v = boot[s]
        print(f"{s:28s} {by[s + '_full']:+10.5f} {np.quantile(v, 0.025):+10.5f} "
              f"{np.quantile(v, 0.975):+10.5f} {v.std(ddof=1):10.5f}")
    print(f"\npaired bootstrap B={a.b} seed={a.seed} (same index sets on both sides):")
    print(f"{'comparison':28s} {'median d':>10s} {'2.5%':>10s} {'97.5%':>10s} {'P(d>0)':>8s}")
    for ref in a.ref:
        for s in a.series:
            if s == ref:
                continue
            d = boot[s] - boot[ref]
            print(
                f"{s + ' - ' + ref:28s} {np.median(d):+10.5f} {np.quantile(d, 0.025):+10.5f} "
                f"{np.quantile(d, 0.975):+10.5f} {(d > 0).mean():8.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
