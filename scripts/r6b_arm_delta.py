#!/usr/bin/env python3
"""R6b controls CB2/CB4 and gates H3/H5/H6: what each F17 arm actually MOVES.

Compares every arm's 372-col table against the revision-1 arm's, cell by cell,
on tables extracted from the SAME pixels by the SAME binary — so a difference is
the `hf_energy_gain` form and nothing else.

**Why this is not `r6_arm_delta.py` with a flag.** That file answers F4's gates,
and two of them do not transfer:

* its **C2 pathology detector** is the `clamp` ARM — `clamp` is exact wherever
  `D² ≤ 1`, so "clamp moved this row" IS "this row holds a pathological pixel".
  F17 has no such arm: every bounded form moves every cell where the feature is
  nonzero, which is 12–52 % of them. CB2 is therefore a VALUE threshold on the
  revision-1 table (the gold holdout's own p99.9), pre-registered in §11.6.
* it has no **H5**: F4's chosen arm sacrificed severity order because nothing
  else could bound `d`; F17 has arms that do not have to, so order preservation
  is a gate here and did not exist there.

Its `load()` is imported rather than re-written, and no statistic is computed
here — H5's ordering check is exact combinatorics on sorted values, not a
correlation estimate.

Gates implemented (`docs/PLAN_FEATURE_REV2_2026-09-05.md` §11.6, §11.8):

* **CB4 containment** — an arm must differ from revision 1 in EXACTLY the twelve
  `contrast_inc` slots and nowhere else. This is what makes the fits an A/B on
  one feature rather than on a rebuild, so it is reported first and a violation
  is fatal to the comparison, not a footnote.
* **CB2 pathology** — cells whose REVISION-1 value exceeds the bar; a row is
  flagged if any of its twelve is. A corpus that flags nothing cannot
  discriminate on pathology and is reported as such.
* **H3 outlier removal** — `max f` over the twelve slots against the arm's own
  declared structural bound. Revision 1's value is printed beside it.
* **H5 order preservation** — over every cell with a nonzero revision-1 value:
  inversions (a pair the arm orders backwards) and NEW ties (a pair revision 1
  separates and the arm does not). Both must be 0 for an order-preserving arm.
  An arm passes iff it is a function of the RATIO alone (`ARM_RATIO_ONLY`); an
  arm that reads the magnitude too does not bound the shipped statistic, it
  replaces it.
* **H6 healthy-cell perturbation** — on CB2-unflagged rows: moved cells,
  max |Δ|, median |Δ| over moved cells, and the count exceeding 1e-4. RANKS the
  arms; does not gate them (§11.8 states why, in advance).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r6_arm_delta import load  # noqa: E402  — one owner for the table read

BASE_ARM = "ratio"
# The twelve `contrast_inc` slots at n_scales=4: basic block-local 12 of each
# (scale, channel) cell. Mirrors zensim::feature_defs::def_at; asserted against
# the measured moved set rather than trusted.
F17_SLOTS = [c * 13 + 12 for c in range(12)]
# §11.6: the gold photographic holdout's own p99.9 over those slots.
CB2_BAR = 0.3468747063945118
H6_REPORT_BAR = 1e-4
# Declared structural bounds, from zensim::hf_gain_form::HfGainForm::upper_bound.
ARM_BOUND = {
    "ratio": None,
    "bexcess": 1.0,
    "log1p": None,
    "satexcess": 1.0,
    "cap": 1.0,
}
# Mirrors zensim::hf_gain_form::HfGainForm::depends_only_on_ratio — the property
# H5 measures. NOT `preserves_order`, which is the LOCAL question (monotone in
# var_dst at fixed var_src) and is true for `bexcess` while H5 is not: `max(0,
# a-b)/(a+b+C)` depends on the MAGNITUDE of b, so two cells with the same ratio
# and different var_src get different values and the population re-ranks.
ARM_RATIO_ONLY = {
    "ratio": True,
    "bexcess": False,
    "log1p": True,
    "satexcess": True,
    "cap": True,
}


def order_stats(base: np.ndarray, arm: np.ndarray) -> tuple[int, int]:
    """(inversions, new_ties) over the flattened cells, EXACTLY.

    Sort by the revision-1 value; a strictly-increasing arm must then be
    non-decreasing, with no equal neighbours where revision 1 differs. Both
    counts are over ADJACENT pairs in that order, which is zero iff the full
    O(n²) pairwise count is zero — an exact test, not a sampled one.
    """
    o = np.argsort(base, kind="stable")
    b, a = base[o], arm[o]
    sep = b[1:] > b[:-1]                       # revision 1 strictly separates
    inversions = int(np.count_nonzero(sep & (a[1:] < a[:-1])))
    new_ties = int(np.count_nonzero(sep & (a[1:] == a[:-1])))
    return inversions, new_ties


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="/mnt/v/output/zensim/rev2-2026-09-05/r6b/tables")
    ap.add_argument("--base-tables", default=None,
                    help="where the revision-1 arm's tables live, if not under --tables")
    ap.add_argument("--arms", default="bexcess,log1p,satexcess,cap")
    ap.add_argument("--corpora", default="cid22,kadid,tid,konjnd,aic3,csiq,live,safesyn")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base_dir = a.base_tables or os.path.join(a.tables, BASE_ARM)
    arms = a.arms.split(",")
    report: dict = {
        "base_arm": BASE_ARM, "f17_slots": F17_SLOTS, "cb2_bar": CB2_BAR,
        "h6_report_bar": H6_REPORT_BAR, "arm_bound": ARM_BOUND,
        "cb4_violations": [], "corpora": {},
    }
    print(f"{'corpus':9s} {'arm':10s} {'CB4':>4s} {'CB2rows':>9s} {'moved':>11s} "
          f"{'H3 rev1':>11s} {'H3 arm':>10s} {'H5 inv':>8s} {'H5 tie':>9s} "
          f"{'H6 cells':>10s} {'H6 max|d|':>10s} {'H6 >1e-4':>10s}")
    for corpus in a.corpora.split(","):
        bp = os.path.join(base_dir, f"{corpus}.csv")
        if not os.path.exists(bp):
            print(f"{corpus:9s} -- absent at the revision-1 arm, skipped")
            continue
        base, names = load(bp)
        idx = np.asarray(F17_SLOTS)
        b17 = base[:, idx]
        path_cell = b17 > CB2_BAR
        path_row = path_cell.any(axis=1)
        rep_c: dict = {
            "rows": int(base.shape[0]),
            "cb2_pathological_rows": int(path_row.sum()),
            "cb2_pathological_cells": int(path_cell.sum()),
            "rev1_max_over_f17": float(b17.max()),
            "rev1_max_over_all_372": float(np.abs(base).max()),
            "rev1_nonzero_f17_cells": int((b17 > 0).sum()),
            "arms": {},
        }
        for arm in arms:
            p = os.path.join(a.tables, arm, f"{corpus}.csv")
            if not os.path.exists(p):
                continue
            m, nm = load(p)
            if nm != names or m.shape != base.shape:
                raise SystemExit(f"{corpus}/{arm}: shape/schema mismatch vs the revision-1 arm")
            diff = m != base
            moved_cols = sorted(int(names[i][1:]) for i in np.nonzero(diff.any(axis=0))[0])
            cb4_ok = set(moved_cols) <= set(F17_SLOTS)
            if not cb4_ok:
                report["cb4_violations"].append(
                    {"corpus": corpus, "arm": arm,
                     "unexpected_slots": sorted(set(moved_cols) - set(F17_SLOTS))})
            a17 = m[:, idx]
            d17 = np.abs(a17 - b17)
            bound = ARM_BOUND[arm]
            h3_arm = float(a17.max())
            nz = b17 > 0
            inv, ties = order_stats(b17[nz].ravel(), a17[nz].ravel())
            healthy = ~path_row
            dh = d17[healthy]
            moved_h = dh > 0
            rep = {
                "cb4_only_f17_slots": bool(cb4_ok),
                "moved_slots": moved_cols,
                "moved_cells_f17": int((d17 > 0).sum()),
                "h3_arm_max": h3_arm,
                "h3_declared_bound": bound,
                "h3_pass": bool(bound is not None and h3_arm <= bound + 1e-12),
                "h5_inversions": inv,
                "h5_new_ties": ties,
                "h5_pass": bool(inv == 0 and ties == 0),
                "h5_declared": ARM_RATIO_ONLY[arm],
                "h6_healthy_rows": int(healthy.sum()),
                "h6_moved_cells": int(moved_h.sum()),
                "h6_max_abs_delta": float(dh.max()) if healthy.any() else 0.0,
                "h6_median_abs_delta_over_moved": (
                    float(np.median(dh[moved_h])) if moved_h.any() else 0.0),
                "h6_cells_over_bar": int((dh > H6_REPORT_BAR).sum()),
            }
            rep_c["arms"][arm] = rep
            print(f"{corpus:9s} {arm:10s} {'ok' if cb4_ok else 'BAD':>4s} "
                  f"{int(path_row.sum()):9d} {rep['moved_cells_f17']:11d} "
                  f"{float(b17.max()):11.5g} {h3_arm:10.5g} {inv:8d} {ties:9d} "
                  f"{rep['h6_moved_cells']:10d} {rep['h6_max_abs_delta']:10.4g} "
                  f"{rep['h6_cells_over_bar']:10d}")
        report["corpora"][corpus] = rep_c

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    if report["cb4_violations"]:
        print(f"\n⛔ CB4 VIOLATED on {len(report['cb4_violations'])} (corpus, arm) pairs — "
              "an arm moved a slot outside the twelve; the fits are NOT a one-feature A/B")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
