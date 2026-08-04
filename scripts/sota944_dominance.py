#!/usr/bin/env python3
"""sota944_dominance.py — strict same-class Pareto dominance over the balanced
matrix (board-integrity pass 2026-08-04, directive 2).

Reads ONLY owner-produced numbers (the `freeze_check --profile ... --tsv`
matrix; this script computes set comparisons, never a statistic), computes
strict Pareto dominance WITHIN each class, and (with --apply) writes
`dominated_by` marks into the affected board fullevals via the committed
promoter (`promote_fulleval.py --mark-dominated`). Files are NEVER deleted;
the board renders dominated cells dimmed + default-off behind a filter chip.

REGISTERED RULE `strict-pareto-2026-08-04`:
  D dominates C (same class only) iff, over the 8 floor axes + the registered
  balanced_composite:
   * every axis MEASURED on C is also measured on D, with D >= C on all of
     them (<= for tied_pct) and D strictly better on >= 1 — i.e. the
     dominator must COVER the dominated cell's measured axes: an absent axis
     never dominates and is never dominated (a cell can't lose on an axis
     nobody measured on it, and can't be beaten via an axis its dominator
     lacks).
   * F4 (dial mono/tied) compares only within the same annotation status
     (spline-bearing dial-unit vs spline-less raw-unit — the
     `dial-mono-raw-unit` registry entry): mixed status makes F4
     not-comparable, which BLOCKS domination of a cell whose F4 is measured
     (coverage rule above). Packaging twins are therefore never trimmed
     against their parents by the flattered raw-unit numbers.
   * F5 (span sanity) is a window, not a monotone axis: compared as the
     owner's pass/fail token (pass > fail).
   * B3/B9 compare SIGNED (collapse must hurt), per the F8 registration.

Usage:
  sota944_dominance.py --matrix <matrix.tsv> [--fulleval-dir DIR] [--apply]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

FE_DEFAULT = Path("/mnt/v/output/zensim/reports/fulleval")
RULE = "strict-pareto-2026-08-04"


def fnum(s: str):
    return None if s in ("-", "", None) else float(s)


def axes_of(r: dict):
    """-> {axis: (value, higher_is_better)} for measured axes; None value = absent.
    F4 carries the unit status in the axis NAME so mixed-status pairs are
    structurally not-comparable (coverage rule blocks domination)."""
    ax = {}
    for c in ("cid22", "konjnd_abs", "nonphoto", "csiq", "live", "hfnl_perref",
              "b3", "b9", "bal_composite"):
        v = fnum(r[c])
        if v is not None:
            ax[c] = (v, True)
    unit = "dialunit" if r["spline"] == "present" else "rawunit"
    mono, tied = fnum(r["mono"]), fnum(r["tied"])
    if mono is not None:
        ax[f"mono@{unit}"] = (mono, True)
    if tied is not None:
        ax[f"tied@{unit}"] = (tied, False)
    if fnum(r["dynrange"]) is not None:
        fails = set(r["fails"].split(",")) if r["fails"] != "-" else set()
        ax["f5_pass"] = (0.0 if "dialrange" in fails else 1.0, True)
    return ax


def dominates(d: dict, c: dict) -> bool:
    """True iff cell d dominates cell c under the registered rule."""
    axd, axc = axes_of(d), axes_of(c)
    strict = False
    for name, (vc, hib) in axc.items():
        if name not in axd:
            return False  # coverage: dominator must measure everything c measures
        vd = axd[name][0]
        better = vd > vc if hib else vd < vc
        equal = vd == vc
        if not (better or equal):
            return False
        strict = strict or better
    return strict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, type=Path)
    ap.add_argument("--fulleval-dir", default=FE_DEFAULT, type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="write dominated_by marks via promote_fulleval.py (else report only)")
    a = ap.parse_args()

    with open(a.matrix) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)

    # name -> fulleval path (stem is the board name for every promoted cell;
    # verify by reading `name` where the stem doesn't match).
    fe_by_name = {}
    for p in sorted(a.fulleval_dir.glob("*.fulleval.json")):
        stem = p.name[: -len(".fulleval.json")]
        fe_by_name[stem] = p
    promoter = Path(__file__).resolve().parent / "promote_fulleval.py"

    total = 0
    for cls, cells in sorted(by_class.items()):
        dominated = {}
        for c in cells:
            doms = sorted(d["name"] for d in cells
                          if d["name"] != c["name"] and dominates(d, c))
            if doms:
                dominated[c["name"]] = doms
        survivors = [c["name"] for c in cells if c["name"] not in dominated]
        print(f"\n== class {cls}: {len(cells)} cells, {len(dominated)} dominated, "
              f"{len(survivors)} survivors")
        for n, doms in sorted(dominated.items()):
            print(f"  DOMINATED {n}  by {','.join(doms[:6])}"
                  + (f" (+{len(doms)-6} more)" if len(doms) > 6 else ""))
        total += len(dominated)
        if a.apply:
            for n, doms in sorted(dominated.items()):
                fe = fe_by_name.get(n)
                if fe is None:
                    print(f"  !! no fulleval file for {n} — NOT marked", file=sys.stderr)
                    continue
                # sanity: the file's `name` must match the matrix row.
                if json.loads(fe.read_text()).get("name") != n:
                    print(f"  !! {fe.name} name mismatch — NOT marked", file=sys.stderr)
                    continue
                subprocess.run([sys.executable, str(promoter), "--mark-dominated", str(fe),
                                "--dominated-by", ",".join(doms),
                                "--dominance-rule", RULE], check=True)
    print(f"\ntotal dominated across classes: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
