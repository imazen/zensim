"""Design log E12 (registered 2026-10-02 22:50 MT): cheaper steering-supported sets on the Rev4 featpot instrument.

  python3 e12_costset.py grid  --root ROOT --candidates benchmarks/e12_costset_candidates_2026-10-02.json --out SPEC.json \\
                               --program-sha P --data-sha D
  python3 e12_costset.py score --root ROOT --candidates ...       # -> <root>/compare/e12_costset.json

Each candidate runs as `sel:<id>@h32:H128` (columns in argv), head N, seeds 0-4, five design folds; scored as the seed-paired
held-out signed-SROCC Δ against v2 + basic (E9″'s final set) and against R0. Rule: "as good as v2 + basic" iff the mean Δ is
>= -0.002 and Δ >= -0.005 on every source.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from e10_multistart import find, index, signed as signed_at
from v2_common import SOURCE_ORDER, V2, selection_id

SEEDS, HEAD, RECIPE = range(5), "N", ":H128"
REFS = {"v2basic": ["v2", "basic"]}
R0 = "r0@h32:H128"
MEAN_FLOOR, SOURCE_FLOOR = -0.002, -0.005


def candidates(path: str) -> dict:
    return json.loads(Path(path).read_text())["candidates"]


def sel_spec(cols) -> str:
    return f"sel:{selection_id(cols)}@h32{RECIPE}"


def cmd_grid(args) -> int:
    cells = []
    for name, cols in candidates(args.candidates).items():
        sp = sel_spec(cols)
        for s in SOURCE_ORDER:
            for i in SEEDS:
                cells.append({"name": f"{sp}__{HEAD}/without_{s}_s{i}",
                              "argv": ["v2_lodo_mlp.py", "--spec", sp, "--head", HEAD, "--heldout", s, "--seed-index", str(i),
                                       "--root", str(V2), "--columns", ",".join(map(str, cols))]})
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "widest": max(len(v) for v in candidates(args.candidates).values())}))
    return 0


def result(spec: str, source: str, seed: int):
    p = V2 / "cells" / f"{spec}__{HEAD}" / f"without_{source}_s{seed}" / "result.json"
    return p if p.is_file() else None


def cmd_score(args) -> int:
    idx = index(RECIPE)
    rows, missing = {}, 0
    for name, cols in candidates(args.candidates).items():
        sp = sel_spec(cols)
        row = {"columns": len(cols), "spec": sp}
        for ref, how in (("v2basic", "set"), ("r0", "r0")):
            per = {}
            for s in SOURCE_ORDER:
                vals = []
                for i in SEEDS:
                    a = result(sp, s, i)
                    b = find(idx, REFS["v2basic"], RECIPE, s, i) if how == "set" else result(R0, s, i)
                    if a is None or b is None:
                        missing += 1
                        continue
                    vals.append(signed_at(a, "set:basic+v2@h32:H128", s) - signed_at(b, "set:basic+v2@h32:H128", s))
                per[s] = {"delta": float(np.mean(vals)) if vals else None,
                          "se": float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else None}
            d = [p["delta"] for p in per.values()]
            row[f"vs_{ref}"] = {"per_source": per, "mean": float(np.mean(d)) if all(x is not None for x in d) else None,
                                "se": (math.sqrt(sum(p["se"] ** 2 for p in per.values())) / len(per)
                                       if all(p["se"] is not None for p in per.values()) else None)}
        v = row["vs_v2basic"]
        row["as_good"] = bool(v["mean"] is not None and v["mean"] >= MEAN_FLOOR
                              and all(p["delta"] is not None and p["delta"] >= SOURCE_FLOOR for p in v["per_source"].values()))
        rows[name] = row
    out = {"schema": "rev4-featpot-e12-costset-v1", "rule": {"mean_floor": MEAN_FLOOR, "source_floor": SOURCE_FLOOR},
           "rows": rows, "missing_cells": missing, "status": "INCOMPLETE" if missing else "complete"}
    (V2 / "compare" / "e12_costset.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, r in sorted(rows.items(), key=lambda kv: -(kv[1]["vs_v2basic"]["mean"] or -9)):
        a, b = r["vs_v2basic"], r["vs_r0"]
        print(f"  {name:14s} cols {r['columns']:4d}  vs v2+basic {a['mean'] if a['mean'] is not None else float('nan'):+.4f} ± "
              f"{a['se'] if a['se'] is not None else float('nan'):.4f}  vs R0 {b['mean'] if b['mean'] is not None else float('nan'):+.4f}"
              + ("  AS GOOD" if r["as_good"] else ""))
    print(json.dumps({"status": out["status"], "missing_cells": missing}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "score"])
    ap.add_argument("--root")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    args = ap.parse_args()
    return cmd_grid(args) if args.cmd == "grid" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
