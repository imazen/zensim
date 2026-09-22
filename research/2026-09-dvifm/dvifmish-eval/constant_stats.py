#!/usr/bin/env python3
"""Ranges of the fitted constants across the dvifmish presets (docs/CONSTANTS.md,
"What we learned"). Prior-protocol presets are left out: their constants are
set, not fitted. A plane-level's share is its normalised plane weight times
its level weight; the integer path's range rule looks at shares of 5% or more.

Usage: constant_stats.py <dvifmish presets dir>
"""
import glob
import json
import sys


def rng(xs):
    return f"{min(xs):.3g} .. {max(xs):.3g}" if xs else "-"


def main():
    cells, betas = [], {}
    for f in sorted(glob.glob(f"{sys.argv[1]}/*.json")):
        p = json.load(open(f))
        name = p["name"]
        if name.endswith("-prior"):
            continue
        total = sum(pl["weight"] for pl in p["planes"])
        for pl in p["planes"]:
            for i, lv in enumerate(pl["levels"]):
                cells.append(dict(preset=name, plane=pl["channel"], level=i,
                                  share=pl["weight"] / total * lv["weight"], **lv))
                if lv["visibility"] == "curve":
                    betas.setdefault(name, set()).add(round(lv["beta"], 3))
    for heavy in (True, False):
        sel = [c for c in cells if (c["share"] >= 0.05) == heavy]
        print(f"share {'>=' if heavy else '<'} 5%: {len(sel)} plane-levels")
        print(f"  P    {rng([c['p'] for c in sel])}; outside [1/4, 2]: "
              f"{sum(1 for c in sel if not 0.25 <= c['p'] <= 2)}")
        print(f"  g    {rng([c['g'] for c in sel if c['visibility'] != 'off'])}")
        print(f"  knee {rng([c['knee'] for c in sel if c['visibility'] != 'off'])}")
        print(f"  beta {rng([c['beta'] for c in sel if c['visibility'] == 'curve'])}")
        print(f"  L    {rng([c['l'] for c in sel])}")
    for name, b in sorted(betas.items()):
        kind = "shared" if len(b) == 1 else f"{len(b)} values"
        print(f"beta {name}: {kind} {sorted(b) if len(b) < 4 else rng(sorted(b))}")


if __name__ == "__main__":
    main()
