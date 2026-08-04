#!/usr/bin/env python3
"""m3a_cheap_grid_agreement.py — agreement between the FULL 27-cell M3a grid
and the REGISTERED 9-cell balanced Latin square (campaign appendix E.5).

The cheap grid is a strict SUBSET of the full grid, so its value is derived
from the SAME per-cell measurements — no re-measurement, and no chance of a
run-to-run confound. Feed it the per-cell TSVs `scripts/m3a_sweep.sh --tsv`
already wrote for a full-grid sweep.

Registered subset (frozen BEFORE any agreement number existed, so it cannot
be tuned to agree): `q_index = (content_index + size_index) mod 3` over
content (city, dog, girl) x size (576, 384, 256) x q (20, 50, 75). Every
content, size and q appears exactly 3x -- balanced on all three axes.

Registered gate: SROCC(cheap, full) >= 0.90 AND max|cheap - full| <= 0.02.

Stats come from `zen_stats` (-> the Rust `panel` binary) ONLY. No stat math
lives here, per the no-duplication rule.

usage: m3a_cheap_grid_agreement.py <dir-of-*.tsv> [--full-col m3a]
"""

import sys
import csv
import glob
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from zen_stats import srocc  # noqa: E402

CONTENT = ["city", "dog", "girl"]
SIZES = ["576", "384", "256"]
QS = ["20", "50", "75"]

# The registered 9 cells.
CHEAP = {
    (CONTENT[ci], SIZES[si], QS[(ci + si) % 3]) for ci in range(3) for si in range(3)
}


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    d = sys.argv[1]
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.tsv"))):
        cells = list(csv.DictReader(open(p), delimiter="\t"))
        if not cells:
            continue
        label = cells[0]["label"]
        full, cheap = [], []
        for c in cells:
            if not c.get("m3a"):
                continue
            v = float(c["m3a"])
            full.append(v)
            if (c["content"], c["size"], c["q"]) in CHEAP:
                cheap.append(v)
        if len(full) != 27 or len(cheap) != 9:
            print(
                f"SKIP {label}: {len(full)} full / {len(cheap)} cheap cells "
                "(expected 27 / 9)",
                file=sys.stderr,
            )
            continue
        rows.append((label, mean(full), mean(cheap)))

    if not rows:
        print("no usable TSVs", file=sys.stderr)
        return 3

    f = [r[1] for r in rows]
    c = [r[2] for r in rows]
    s = srocc(c, f)
    maxabs = max(abs(a - b) for a, b in zip(c, f))
    meanabs = mean([abs(a - b) for a, b in zip(c, f)])

    print("label\tfull_27\tcheap_9\tdiff")
    for label, fv, cv in sorted(rows, key=lambda r: -abs(r[2] - r[1])):
        print(f"{label}\t{fv:.6f}\t{cv:.6f}\t{cv - fv:+.6f}")
    print()
    print(f"n                = {len(rows)}")
    print(f"SROCC(cheap,full)= {s:.4f}   [registered gate >= 0.90]")
    print(f"max |cheap-full| = {maxabs:.4f}   [registered gate <= 0.02]")
    print(f"mean|cheap-full| = {meanabs:.4f}")
    ok = s >= 0.90 and maxabs <= 0.02
    print(f"VERDICT          = {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
