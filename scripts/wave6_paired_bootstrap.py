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
    n = len(human)
    print(f"[{a.corpus}] n={n} series={a.series} (human column identical across all)")

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
    by = {r["label"]: float(r["srocc"]) for r in rows}

    print(f"\npoint estimates (|SROCC|, {a.corpus}):")
    for s in a.series:
        print(f"  {s:6s} {by[s + '_full']:.6f}")

    boot = {s: np.array([by[f"{s}_{b}"] for b in range(a.b)]) for s in a.series}
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
