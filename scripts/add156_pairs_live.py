#!/usr/bin/env python3
"""Classify appendix-U cells LIVE vs ZERO on the CANDIDATE COEFFICIENTS.

A cell offered `base ∪ C` is only a real intervention if the solver actually
spends budget on `C`. Testing that on the bake sha does NOT work: offering one
extra coordinate perturbs the coordinate-descent path, so a cell whose
candidate lands at exactly 0.0 still differs from base by ~5e-11 on the other
weights (below the 1e-10 `tol`) and therefore by a few bake bytes. Measured on
arm A: 7 of 12 G-U1 gate cells — whose candidates are provably no-ops by the
KKT restriction argument — had a different sha while `w[candidate]` was exactly
`-0.0`.

So the test is `any(w[i] != 0 for i in C)`, read from the fit npz.

Also emits, per cell, the candidate weights themselves and how far the BASE
coefficients moved, which is what separates "the pair added information" from
"the pair displaced base mass" (appendix U §U.8).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

BASE28 = [6, 8, 11, 14, 17, 19, 22, 24, 26, 34, 37, 89, 91, 93, 94, 116, 120,
          121, 122, 124, 128, 136, 137, 138, 140, 146, 150, 155]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--fits", required=True, type=Path)
    ap.add_argument("--tag", required=True, help="e.g. 2e3s200000")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    base = {}
    for arm in ("A", "B"):
        p = a.fits / f"U_{arm}{a.tag}_BASE.npz"
        if not p.exists():
            print(f"FATAL: missing base fit {p}", file=sys.stderr)
            return 2
        base[arm] = np.load(p)["w"]

    rows, missing = [], 0
    with open(a.manifest) as fh:
        for m in csv.DictReader(fh, delimiter="\t"):
            arm, cid = m["arm"], m["cell_id"]
            p = a.fits / f"U_{arm}{a.tag}_{cid}.npz"
            if not p.exists():
                missing += 1
                continue
            w = np.load(p)["w"]
            cand = [int(i) for i in m["indices"].split(",")]
            cw = [float(w[i]) for i in cand]
            live = any(v != 0.0 for v in cw)
            b = base[arm]
            # how much of the change is displacement of the EXISTING model
            base_shift = float(np.abs(w[BASE28] - b[BASE28]).max())
            rows.append((cid, arm, "LIVE" if live else "ZERO",
                         ";".join(f"{v:+.6e}" for v in cw),
                         sum(1 for v in cw if v != 0.0), len(cand),
                         f"{base_shift:.3e}", int((w != 0).sum())))

    with open(a.out, "w") as f:
        f.write("cell_id\tarm\tstatus\tcand_w\tn_cand_live\tn_cand\t"
                "base_max_shift\tn_active\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    n = {}
    for r in rows:
        n[f"{r[1]}/{r[2]}"] = n.get(f"{r[1]}/{r[2]}", 0) + 1
    print(f"wrote {a.out}")
    for k in sorted(n):
        print(f"  {k} {n[k]}")
    if missing:
        print(f"  WARNING: {missing} cells had no fit npz", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
