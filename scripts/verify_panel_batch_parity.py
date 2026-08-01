#!/usr/bin/env python3
"""Parity + determinism gate for the `panel --batch` mode.

Companion to `scripts/verify_panel_parity.py` (which gates the AGGREGATE
panel). This gates the batch mode added for decision-surface-audit gap 4
(`benchmarks/decision_surface_audit_2026-07-31.md`): the canonical
replacement for every `scipy.stats.spearmanr`-inside-a-bootstrap-loop
call site must agree with the scipy midrank reference before those call
sites are migrated (`scripts/hdr/upiq_panel.py` being the named first).

## The gates (exit 0 iff ALL pass)

1. **SROCC vs scipy midrank <= 1e-12** on every fixture pair, including
   tie-heavy cases (>=50 % duplicated values) — both the signed
   (`srocc_signed`) and `.abs()` (`srocc`) columns. scipy's `spearmanr`
   is the tie-correct midrank reference; `panel::ranks` implements the
   same midrank averaging.
2. **plcc_raw vs scipy `pearsonr` <= 1e-12** (signed, no logistic).
3. **Indexed == explicit**: the same resample expressed as an index set
   over `#def` bases and as a materialized pair must produce IDENTICAL
   rows (string equality — the indexed path gathers then computes).
4. **Determinism**: two invocations on the same manifest are
   byte-identical (the binary is RNG-free by design).
5. **Batch full == aggregate**: batch `--stats full` on a single pair
   matches `panel --input --json` on the same data to <= 1e-9 on every
   shared stat (they share `compute_panel`; the JSON prints {:.10}).
6. **Bootstrap-shape smoke**: 2,000 seeded index-set resamples of a
   140-point pair in ONE process; every row gated vs scipy on the
   materialized resample (<= 1e-12).

Degenerate (constant-vector) pairs are asserted against the OWNER's
documented convention (srocc = 0.0 where scipy returns NaN) — that is a
definitional difference, reported not gated, same treatment as OR /
Z-RMSE in verify_panel_parity.py.

Usage:
  python3 scripts/verify_panel_batch_parity.py [--bin path/to/panel] [--tol 1e-12]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts.lib import zen_stats  # noqa: E402


def make_fixtures(rng: np.random.Generator):
    """(name, x, y) fixture pairs; >= half are tie-heavy."""
    fixtures = []
    n = 37
    x = rng.normal(size=n)
    fixtures.append(("clean_monotone", np.sort(x), np.arange(n, dtype=float)))
    fixtures.append(("clean_anti", np.sort(x), -np.arange(n, dtype=float)))
    fixtures.append(("clean_noise", rng.normal(size=n), rng.normal(size=n)))
    # Tie-heavy: quantized to few levels (exact repeated values).
    for levels in (2, 3, 5, 8):
        xq = np.round(rng.normal(size=61) * levels) / levels
        yq = np.round(xq * 2.0 + rng.normal(size=61) * 0.7, 1)
        fixtures.append((f"ties_q{levels}", xq, yq))
    # Tie-heavy: integer scores (MOS-like 1..5) vs continuous metric.
    mos = rng.integers(1, 6, size=80).astype(float)
    met = mos + rng.normal(size=80) * 1.2
    fixtures.append(("ties_mos5", met, mos))
    # Tie-heavy: >50% duplicates on BOTH sides.
    xa = np.repeat(rng.normal(size=10), 6)
    ya = np.repeat(rng.normal(size=10), 6)
    rng.shuffle(ya)
    fixtures.append(("ties_heavy_both", xa, ya))
    # Tiny n + ties.
    fixtures.append(("tiny_ties", np.array([1.0, 1.0, 2.0]), np.array([3.0, 3.0, 1.0])))
    # NaN rows (drop policy — scipy comparison on the finite subset).
    xn = rng.normal(size=30)
    yn = rng.normal(size=30)
    xn[3] = np.nan
    yn[7] = np.inf
    fixtures.append(("nonfinite_rows", xn, yn))
    return fixtures


def finite_pair(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=None, help="path to the panel binary")
    ap.add_argument("--tol", type=float, default=1e-12)
    a = ap.parse_args()
    if a.bin:
        os.environ["ZEN_PANEL_BIN"] = a.bin

    rng = np.random.default_rng(20260731)
    fixtures = make_fixtures(rng)
    fails = []
    max_ds = 0.0
    max_dp = 0.0

    # ---- gates 1+2: explicit batch vs scipy ----
    jobs = [(name, x, y) for name, x, y in fixtures]
    rows = zen_stats.panel_batch(jobs, stats="full")
    for (name, x, y), row in zip(fixtures, rows):
        xf, yf = finite_pair(np.asarray(x, float), np.asarray(y, float))
        s_ref = float(spearmanr(xf, yf).statistic)
        p_ref = float(pearsonr(xf, yf).statistic)
        ds = abs(row["srocc_signed"] - s_ref)
        da = abs(row["srocc"] - abs(s_ref))
        dp = abs(row["plcc_raw"] - p_ref)
        max_ds = max(max_ds, ds, da)
        max_dp = max(max_dp, dp)
        if ds > a.tol or da > a.tol:
            fails.append(f"{name}: srocc diff {max(ds, da):.3e} (rust "
                         f"{row['srocc_signed']!r} scipy {s_ref!r})")
        if dp > a.tol:
            fails.append(f"{name}: plcc_raw diff {dp:.3e}")
        if row["n"] != len(xf):
            fails.append(f"{name}: n {row['n']} != finite rows {len(xf)}")
    print(f"gate 1+2: {len(fixtures)} fixtures ({sum(1 for n_, _, _ in fixtures if n_.startswith(('ties', 'tiny')))} "
          f"tie-heavy) | max srocc diff {max_ds:.3e} | max plcc_raw diff {max_dp:.3e}")

    # ---- degenerate convention (reported, not gated vs scipy) ----
    deg = zen_stats.panel_batch(
        [("const_x", [2.0] * 8, list(range(8))),
         ("const_both", [1.0] * 5, [3.0] * 5)], stats="srocc")
    for row in deg:
        if row["srocc"] != 0.0 or row["srocc_signed"] != 0.0:
            fails.append(f"degenerate {row['label']}: expected owner convention 0.0, "
                         f"got {row['srocc_signed']!r}")
    print("degenerate constant-vector pairs -> srocc 0.0 (owner convention; "
          "scipy returns NaN — definitional, documented)")

    # ---- gates 3+4: indexed == explicit, determinism ----
    base_x = rng.normal(size=64)
    base_y = np.round(base_x + rng.normal(size=64), 1)  # with ties
    idx_sets = [rng.integers(0, 64, 64) for _ in range(50)]
    manifest = ["#def X\t" + ",".join(repr(float(v)) for v in base_x),
                "#def Y\t" + ",".join(repr(float(v)) for v in base_y)]
    for i, idx in enumerate(idx_sets):
        manifest.append(f"ix{i}\t@X:@Y\t" + ",".join(str(int(j)) for j in idx))
        manifest.append(f"ex{i}\t"
                        + ",".join(repr(float(base_x[j])) for j in idx) + "\t"
                        + ",".join(repr(float(base_y[j])) for j in idx))
    text = "\n".join(manifest) + "\n"
    bin_path = zen_stats._find_panel_bin()
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        o1 = subprocess.run([bin_path, "--batch", tmp, "--stats", "full"],
                            capture_output=True, text=True, check=True).stdout
        o2 = subprocess.run([bin_path, "--batch", tmp, "--stats", "full"],
                            capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(tmp)
    if o1 != o2:
        fails.append("determinism: two runs differ")
    lines = o1.splitlines()[1:]
    for i in range(50):
        ix = lines[2 * i].split("\t", 1)[1]
        ex = lines[2 * i + 1].split("\t", 1)[1]
        if ix != ex:
            fails.append(f"indexed != explicit at resample {i}")
            break
    print("gate 3+4: 50 indexed-vs-explicit resamples identical; two runs byte-identical")

    # ---- gate 5: batch full == aggregate panel on the same pair ----
    xp, yp = fixtures[3][1], fixtures[3][2]  # a tie-heavy pair
    agg = zen_stats.panel(xp, yp)
    brow = zen_stats.panel_batch([("agg", xp, yp)], stats="full")[0]
    for k in ("srocc", "plcc", "krocc", "pwrc", "z_rmse"):
        d = abs(agg[k] - brow[k])
        if d > 1e-9:  # aggregate JSON prints {:.10}
            fails.append(f"batch-vs-aggregate {k}: diff {d:.3e}")
    if abs(agg["or"] - brow["or"]) > 1e-9:
        fails.append("batch-vs-aggregate or")
    print("gate 5: batch --stats full == aggregate panel (<=1e-9, JSON precision)")

    # ---- gate 6: bootstrap-shape smoke (one process, 2k resamples) ----
    n = 140
    da = rng.normal(size=n)
    dj = np.round(da * 1.5 + rng.normal(size=n), 1)
    boot_rng = np.random.default_rng(20260714)
    boots = [boot_rng.integers(0, n, n) for _ in range(2000)]
    rows = zen_stats.panel_batch_indexed(
        {"A": da, "J": dj},
        [(f"b{i}", "A", "J", idx) for i, idx in enumerate(boots)],
        stats="srocc")
    worst = 0.0
    for row, idx in zip(rows, boots):
        ref = float(spearmanr(da[idx], dj[idx]).statistic)
        d = abs(row["srocc_signed"] - ref)
        worst = max(worst, d)
        if d > a.tol:
            fails.append(f"bootstrap {row['label']}: diff {d:.3e}")
            break
    print(f"gate 6: 2,000 seeded index-set resamples in ONE process | "
          f"max srocc diff vs scipy {worst:.3e}")

    if fails:
        print("\nFAIL:")
        for f_ in fails:
            print("  " + f_)
        return 1
    print(f"\nALL GATES PASS (tol {a.tol:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
