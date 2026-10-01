"""Emit the zenfleet fit-spec cells for v2_confirm_fit.py (amendment R2): specs x heads x 10 seeds.

Same JSON shape as `build_fit_spec.py --grid v2` writes (`{program_sha, data_sha, cells: [{name, argv}]}`), declared with
`zenfleet-ctl declare-fits`. The specs are r0 plus every frozen candidate arm and its three permuted controls; the
human-leg weight selected under amendment R3 rides on every spec as `@h<w>` (so job ids differ per weight).

  python v2c_grid.py --kind calibration-screen --human-weight 32 ...   # R2.3 (a): 900 LODO cells
  python v2c_grid.py --kind full-arms --arm c1 --arm csfw --human-weight 32 ...   # R2.3 (b): 400 LODO cells per arm
  python v2c_grid.py --program-sha H --data-sha H --arm c1 --arm c3 --human-weight 2 \
      --root /var/tmp/rev4-featpot/v2c --out fit-spec-v2c-confirm.json

NOTE (not this repo): the executor `zenmetrics/scripts/jobsys/fit_cell_exec.py` approves programs through its SCRIPTS
table (program -> output root under /var/tmp/rev4-featpot). `v2_confirm_fit.py` needs an entry there, and its
`--root`-relative output (`<root>/confirm/cells/<spec>__<head>/full_s<i>`) must map to the same directory, e.g.
`"v2_confirm_fit.py": ("v2c/confirm/cells", None)` with `--root /var/tmp/rev4-featpot/v2c`.
"""

import argparse
import json
import sys
from pathlib import Path

from v2_common import CALIBRATION, HEADS, N_PERMS, SCREEN, SOURCE_ORDER

SEEDS = range(10)
CALIBRATION_SPECS = ("r0", *CALIBRATION, *(f"oracle_lo~p{k}" for k in range(1, N_PERMS + 1)))  # = build_fit_spec V2_CALIBRATION


def specs_for(arms: list[str], human_weight: float | None) -> list[str]:
    base = ["r0"]
    for arm in arms:
        base += [arm, *(f"{arm}~p{k}" for k in range(1, N_PERMS + 1))]
    if human_weight is None:
        return base
    return [f"{spec}@h{human_weight:g}" for spec in base]


def cells(arms: list[str], human_weight: float | None, root: str) -> list[dict]:
    out = []
    for spec in specs_for(arms, human_weight):
        for head in HEADS:
            for seed in SEEDS:
                argv = ["v2_confirm_fit.py", "--spec", spec, "--head", head, "--seed-index", str(seed), "--root", root]
                out.append({"name": f"{spec}__{head}/full_s{seed}", "argv": argv})
    return out


def lodo_cells(specs: list[str], human_weight: float | None, root: str) -> list[dict]:
    """v2_lodo_mlp cells: spec x head x held-out source x seed, on the canon root (`--root` in argv, so job ids differ from Rev3's)."""
    out = []
    for spec in (specs if human_weight is None else [f"{x}@h{human_weight:g}" for x in specs]):
        for head in HEADS:
            for source in SOURCE_ORDER:
                for seed in SEEDS:
                    out.append({"name": f"{spec}__{head}/without_{source}_s{seed}",
                                "argv": ["v2_lodo_mlp.py", "--spec", spec, "--head", head, "--heldout", source,
                                         "--seed-index", str(seed), "--root", root]})
    return out


def grid_calibration_screen(human_weight: float, root: str) -> list[dict]:
    """R2.3 (a): canon calibration (7 specs) + the two screen specs, both heads, five sources, 10 seeds = 900 cells."""
    return lodo_cells([*CALIBRATION_SPECS, *SCREEN], human_weight, root)


def grid_full_arms(arms: list[str], human_weight: float, root: str) -> list[dict]:
    """R2.3 (b): full arms for a family list: arm + its three permuted controls (r0 comes from (a)), 400 cells per arm."""
    return lodo_cells([s for a in arms for s in (a, *[f"{a}~p{k}" for k in range(1, N_PERMS + 1)])], human_weight, root)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program-sha", required=True)
    ap.add_argument("--data-sha", required=True)
    ap.add_argument("--kind", choices=["confirm", "calibration-screen", "full-arms"], default="confirm",
                    help="confirm: v2_confirm_fit cells (default); calibration-screen: R2.3 (a), 900 v2_lodo_mlp cells; "
                         "full-arms: R2.3 (b), 400 cells per --arm")
    ap.add_argument("--arm", action="append", default=[], help="a candidate arm (repeatable); confirm adds r0 itself")
    ap.add_argument("--human-weight", type=float, default=None, help="R3 human-leg weight; appended as @h<w>")
    ap.add_argument("--root", required=True, help="canon instrument root passed to every cell as --root")
    ap.add_argument("--existing-root", type=Path, default=None, help="skip cells whose result.json exists under here")
    ap.add_argument("--include-complete", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for label, value in (("program", args.program_sha), ("data", args.data_sha)):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            ap.error(f"{label} SHA-256 must be lowercase hex")
    if args.kind == "confirm":
        allcells = cells(args.arm, args.human_weight, args.root)
        assert len(allcells) == (1 + (1 + N_PERMS) * len(args.arm)) * len(HEADS) * len(SEEDS)
    else:
        if args.human_weight is None:
            ap.error("--human-weight is required for the LODO grids (R3 selected 32)")
        if args.kind == "calibration-screen":
            allcells = grid_calibration_screen(args.human_weight, args.root)
            assert len(allcells) == 900
        else:
            allcells = grid_full_arms(args.arm, args.human_weight, args.root)
            assert len(allcells) == 400 * len(args.arm)
    done = set()
    if args.existing_root is not None:
        done = {c["name"] for c in allcells if (args.existing_root / c["name"] / "result.json").is_file()}
    chosen = allcells if args.include_complete else [c for c in allcells if c["name"] not in done]
    args.out.write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": chosen}, indent=2) + "\n")
    print(json.dumps({"total": len(allcells), "completed_local": len(done), "declared": len(chosen), "out": str(args.out)}))


if __name__ == "__main__":
    sys.exit(main())
