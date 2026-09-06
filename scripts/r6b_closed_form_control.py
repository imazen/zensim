#!/usr/bin/env python3
"""R6b control CB5: an extracted ratio-only arm must equal the CLOSED FORM of
the revision-1 table, cell for cell.

Three of F17's five arms are pure functions of the shipped feature value
`g = max(0, var_dst/var_src - 1)`:

    log1p      ln(1 + g)
    satexcess  g / (g + 1)
    cap        min(g, 1)

so their tables are PREDICTABLE from revision 1's without extracting anything.
Checking the prediction against the real extraction is what proves the runtime
arm switch actually fired: a typo in the env-var match would fall through to the
shipped form and produce a table identical to the control, which every gate
downstream would happily report as "this arm moves nothing".

`bexcess` is deliberately NOT predictable here — `max(0, a-b)/(a+b+C_HF)` needs
the magnitudes, which the stored table does not carry. That asymmetry is itself
the finding H5 reports, so it is named rather than worked around.

Tolerance is a few ULP: the extractor computes the arm in f64 from the pooled
moments, and this recomputes it in f64 from the STORED revision-1 value, which
is one extra decimal round-trip through the CSV. Exact equality is not the
claim; agreement far below any measurement bar is.

**The liveness half is per ARM, not per (arm, corpus)** — corrected 2026-09-06
after the first run reported a FAIL that was the control being wrong, not the
data. `cap` is exact below `g = 1` and AIC-3's largest `contrast_inc` is
0.61517, so `cap` correctly moves ZERO cells there; a per-corpus "must move
something" rule calls that a dead switch. What actually proves the switch fired
is that the arm moves cells SOMEWHERE in the population, and the per-corpus
assertion is the exact one: the extracted table equals the closed form.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r6_arm_delta import load  # noqa: E402

F17_SLOTS = [c * 13 + 12 for c in range(12)]
CLOSED_FORM = {
    "log1p": np.log1p,
    "satexcess": lambda g: g / (g + 1.0),
    "cap": lambda g: np.minimum(g, 1.0),
}
TOL = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="/mnt/v/output/zensim/rev2-2026-09-05/r6b/tables")
    ap.add_argument("--base", default="ratio")
    ap.add_argument("--corpora", default="cid22,kadid,tid,konjnd,aic3,csiq,live,safesyn")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    idx = np.asarray(F17_SLOTS)
    rep: dict = {"tolerance": TOL, "base_arm": a.base, "arms": {}}
    ok = True
    live: dict = {}
    print(f"{'corpus':9s} {'arm':10s} {'cells':>10s} {'max|pred-got|':>14s} "
          f"{'moved vs rev1':>14s} {'verdict':>9s}")
    for corpus in a.corpora.split(","):
        bp = os.path.join(a.tables, a.base, f"{corpus}.csv")
        if not os.path.exists(bp):
            continue
        base, _ = load(bp)
        g = base[:, idx]
        for arm, fn in CLOSED_FORM.items():
            p = os.path.join(a.tables, arm, f"{corpus}.csv")
            if not os.path.exists(p):
                continue
            m, _ = load(p)
            pred = fn(g)
            got = m[:, idx]
            err = float(np.abs(pred - got).max())
            moved = int((got != g).sum())
            # Per corpus the claim is EXACTNESS. Liveness is pooled, below:
            # an arm exact below a threshold legitimately moves nothing on a
            # corpus that never reaches it.
            good = err <= TOL
            ok &= good
            live[arm] = live.get(arm, 0) + moved
            rep["arms"].setdefault(arm, {})[corpus] = {
                "cells": int(g.size), "max_abs_pred_minus_got": err,
                "moved_cells_vs_rev1": moved, "pass": good}
            print(f"{corpus:9s} {arm:10s} {g.size:10d} {err:14.4g} {moved:14d} "
                  f"{'PASS' if good else 'FAIL':>9s}")
    print(f"\n{'arm':10s} {'moved cells, POOLED':>20s}   liveness")
    for arm in CLOSED_FORM:
        n = live.get(arm, 0)
        if arm in live:
            print(f"{arm:10s} {n:20d}   {'PASS' if n > 0 else 'FAIL — switch did not fire'}")
            ok &= n > 0
    rep["pass"] = ok
    rep["pooled_moved_cells"] = live
    if not ok:
        print("\n⛔ CB5 FAILED — an arm disagrees with its own closed form, or moved "
              "NOTHING anywhere (the runtime switch did not fire)")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
