"""Amendment R7 (2026-10-04): the set-compare confirmatory read of the two adopted candidates.

  python3 v2_setcompare.py grid --root ROOT --out SPEC.json --program-sha P --data-sha D   # 4 entries x head N x seeds 0-9

Entries (head N, full-data `v2_confirm_fit` cells under <root>/confirm/cells; R7 in benchmarks/rev4_featpot_v2_amendment_2026-09-30.md):
A = v2 + basic at cv16:cf98, B = by_v2fy at cv16:cf98, C = v2 + basic uncurated, D = R0 at H128. The fits read no label.
The read itself is `v2_confirm_read.py --set-compare` against the R7 pin.
"""

import argparse
import json
import sys
from pathlib import Path

from v2_common import V2, selection_id

REPO = Path(__file__).resolve().parents[2]
HEAD, SEEDS = "N", range(10)


def by_v2fy_columns() -> list:
    d = json.loads((REPO / "benchmarks/costset2_2026-10-03.candidate_ids.json").read_text())
    return d.get("candidates", d)["by_v2fy"]


def entries() -> dict:
    cols = by_v2fy_columns()
    return {"A": ("set:v2+basic@h32:H128:cv16:cf98", None),
            "B": (f"sel:{selection_id(cols)}@h32:H128:cv16:cf98", cols),
            "C": ("set:v2+basic@h32:H128", None),
            "D": ("r0@h32:H128", None)}


def cmd_grid(args) -> int:
    cells = []
    for label, (spec, cols) in entries().items():
        for i in SEEDS:
            argv = ["v2_confirm_fit.py", "--spec", spec, "--head", HEAD, "--seed-index", str(i), "--root", str(V2)]
            if cols:
                argv += ["--columns", ",".join(map(str, cols))]
            cells.append({"name": f"{spec}__{HEAD}/full_s{i}", "argv": argv})
    todo = [c for c in cells if not (V2 / "confirm" / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "entries": {k: v[0] for k, v in entries().items()}}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid"])
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    args = ap.parse_args()
    return cmd_grid(args)


if __name__ == "__main__":
    sys.exit(main())
