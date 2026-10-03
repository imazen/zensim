"""Design log "E9″ near-miss confirmation": seeds 5-9 for the base set and its closest round-4 additions; scores the
10-seed seed-paired gain (seeds 0-9) per source with its standard error.

  python3 e9_nearmiss.py grid  --root ROOT --out SPEC.json --program-sha P --data-sha D
  python3 e9_nearmiss.py score --root ROOT          # -> <root>/compare/e9_nearmiss.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from e9_forward import HEAD, signed, spec_of
from v2_common import SOURCE_ORDER, V2, arm_columns

BASE = ["v2", "basic"]
NEAR = ["masked", "c4", "c8n", "iw", "p1", "p3"]
NEW_SEEDS, ALL_SEEDS = range(5, 10), range(10)
RECIPE = ":H128"


def cmd_grid(args) -> int:
    specs = [spec_of(BASE, RECIPE), *(spec_of([*BASE, g], RECIPE) for g in NEAR)]
    cells = [{"name": f"{sp}__{HEAD}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", sp, "--head", HEAD, "--heldout", s, "--seed-index", str(i), "--root", str(V2)]}
             for sp in specs for s in SOURCE_ORDER for i in NEW_SEEDS]
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "widest": max(len(arm_columns(sp)[2]) for sp in specs)}))
    return 0


def cmd_score(args) -> int:
    base = spec_of(BASE, RECIPE)
    rows, missing = {}, 0
    for g in NEAR:
        spec = spec_of([*BASE, g], RECIPE)
        per = {}
        for s in SOURCE_ORDER:
            vals = []
            for i in ALL_SEEDS:
                a, b = signed(spec, s, i), signed(base, s, i)
                if a is None or b is None:
                    missing += 1
                    continue
                vals.append(a - b)
            per[s] = {"gain": float(np.mean(vals)) if vals else None,
                      "se": float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else None, "n": len(vals)}
        g_mean = float(np.mean([p["gain"] for p in per.values()])) if all(p["gain"] is not None for p in per.values()) else None
        se = math.sqrt(sum(p["se"] ** 2 for p in per.values())) / len(per) if all(p["se"] is not None for p in per.values()) else None
        pos = sum(p["gain"] is not None and p["gain"] > 0 for p in per.values())
        rows[g] = {"per_source": per, "mean_gain": g_mean, "se": se, "positive_sources": pos,
                   "candidate": bool(g_mean is not None and g_mean >= 0.002 and pos >= 3)}
    out = {"schema": "rev4-featpot-e9-nearmiss-v1", "base": base, "seeds": list(ALL_SEEDS), "rows": rows,
           "missing_cells": missing, "status": "INCOMPLETE" if missing else "complete"}
    (V2 / "compare" / "e9_nearmiss.json").write_text(json.dumps(out, indent=1) + "\n")
    for g, r in sorted(rows.items(), key=lambda kv: -(kv[1]["mean_gain"] or -9)):
        m, se = r["mean_gain"], r["se"]
        print(f"  {g:7s} gain {m if m is not None else float('nan'):+.4f} ± {se if se is not None else float('nan'):.4f}  positive {r['positive_sources']}/5"
              + ("  CANDIDATE" if r["candidate"] else ""))
    print(json.dumps({"status": out["status"], "missing_cells": missing}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "score"])
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    args = ap.parse_args()
    return cmd_grid(args) if args.cmd == "grid" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
