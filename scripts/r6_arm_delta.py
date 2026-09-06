#!/usr/bin/env python3
"""R6 controls C2/C3 and gates G3/G4: what each F4 arm actually MOVES.

Compares every arm's 372-col table against the `ssim2` (revision-1) arm's, cell
by cell, on tables extracted from the SAME pixels by the SAME binary — so a
difference is the luminance form and nothing else.

What each number is FOR (`docs/PLAN_FEATURE_REV2_2026-09-05.md` §7.3, §7.5):

* **C2 pathology detector.** `clamp` is `max(0, 1 - D^2)`: exact wherever
  `D^2 <= 1` and different only above it. So a row where CLAMP moves is a row
  holding at least one pixel in F4's pathological regime, and a corpus where
  clamp moves nothing cannot discriminate an arm on pathology at all. That
  makes the clamp column a corpus PROPERTY, not just an arm result.
* **G3 outlier removal.** `max |f_j|` over the moved slots. Revision 1's value
  is the thing F4 is about (5.8e6 on the worst content on record); a bounded
  arm's must sit at its structural bound (`d in [0,2]`).
* **G4 healthy-cell perturbation.** The same comparison restricted to rows C2
  does NOT flag. An arm that moves healthy content is doing something other
  than removing an outlier, whatever its rank.

NO statistics live here — this is exact cell arithmetic (counts, max |delta|).
Rank statistics come from `panel` / `bake_verdict`, their owners.

The moved-slot SET is measured, never assumed: the registry's own count is
printed beside it so a disagreement is visible rather than asserted away.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.csv as pacsv

BASE_ARM = "ssim2"
REGISTERED_F4_SLOTS = 132  # benchmarks/feature_rev2_2026-09-05.md §1.4


def load(path: str) -> tuple[np.ndarray, list[str]]:
    t = pacsv.read_csv(path)
    names = [n for n in t.schema.names if n.startswith("f") and n[1:].isdigit()]
    names.sort(key=lambda n: int(n[1:]))
    cols = [t.column(n).to_numpy(zero_copy_only=False).astype(np.float64) for n in names]
    return np.column_stack(cols), names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="/mnt/v/output/zensim/rev2-2026-09-05/r6/tables")
    ap.add_argument("--arms", default="c1,lorentz,clamp")
    ap.add_argument("--corpora", default="cid22,kadid,tid,konjnd,aic3,csiq,live,safesyn")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = a.arms.split(",")
    report: dict = {"base_arm": BASE_ARM, "registered_f4_slots": REGISTERED_F4_SLOTS,
                    "corpora": {}}
    print(f"{'corpus':10s} {'arm':8s} {'rows±':>13s} {'cells':>12s} {'slots':>5s} "
          f"{'max|Δ|':>12s} {'G3 max|f|':>13s} {'G4 cells':>9s} {'G4 max|Δ|':>11s}")
    for corpus in a.corpora.split(","):
        bp = os.path.join(a.tables, BASE_ARM, f"{corpus}.csv")
        if not os.path.exists(bp):
            print(f"{corpus:10s} -- absent at base arm, skipped")
            continue
        base, names = load(bp)
        n_rows = base.shape[0]
        per_arm: dict = {}
        mats: dict = {}
        for arm in arms:
            p = os.path.join(a.tables, arm, f"{corpus}.csv")
            if not os.path.exists(p):
                continue
            m, nm = load(p)
            if nm != names or m.shape != base.shape:
                raise SystemExit(f"{corpus}/{arm}: shape/schema mismatch vs base")
            mats[arm] = m
        # C2: the clamp arm defines which rows hold a pathological pixel.
        clamp_rows = None
        if "clamp" in mats:
            clamp_rows = np.any(mats["clamp"] != base, axis=1)
        rep_corpus = {
            "rows": int(n_rows),
            "c2_pathological_rows": (int(clamp_rows.sum()) if clamp_rows is not None else None),
            "rev1_max_abs_over_all_slots": float(np.abs(base).max()),
            "arms": {},
        }
        for arm, m in mats.items():
            diff = m != base
            moved_cells = int(diff.sum())
            slot_moved = np.any(diff, axis=0)
            slots = [names[i] for i in np.nonzero(slot_moved)[0]]
            d = np.abs(m - base)
            mx = float(d.max()) if moved_cells else 0.0
            # G3 is read on the slots F4 actually reaches in this population.
            g3_cols = np.nonzero(slot_moved)[0]
            g3_arm = float(np.abs(m[:, g3_cols]).max()) if len(g3_cols) else 0.0
            g3_rev1 = float(np.abs(base[:, g3_cols]).max()) if len(g3_cols) else 0.0
            # G4: healthy rows only (C2-unflagged). Undefined without clamp.
            if clamp_rows is not None:
                healthy = ~clamp_rows
                g4_cells = int(diff[healthy].sum())
                g4_mx = float(d[healthy].max()) if healthy.any() else 0.0
            else:
                g4_cells, g4_mx = None, None
            rep = {
                "moved_rows": int(np.any(diff, axis=1).sum()),
                "moved_cells": moved_cells,
                "moved_slots_n": len(slots),
                "moved_slots": slots,
                "max_abs_delta": mx,
                "g3_arm_max_abs_on_moved_slots": g3_arm,
                "g3_rev1_max_abs_on_moved_slots": g3_rev1,
                "g4_healthy_moved_cells": g4_cells,
                "g4_healthy_max_abs_delta": g4_mx,
            }
            per_arm[arm] = rep
            print(f"{corpus:10s} {arm:8s} {rep['moved_rows']:6d}/{n_rows:<6d} "
                  f"{moved_cells:12d} {len(slots):5d} {mx:12.6g} {g3_arm:13.6g} "
                  f"{str(g4_cells):>9s} {('%.4g' % g4_mx) if g4_mx is not None else '--':>11s}")
        rep_corpus["arms"] = per_arm
        report["corpora"][corpus] = rep_corpus

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
