"""Design log E9′ method 1: group-level forward selection from the 228-column core, equal footing for every legacy
block and every registered arm.

  python3 e9_forward.py grid  --recipe ':H128' --selected '' --out STEP.json      # cells of one step (base + candidates)
  python3 e9_forward.py score --recipe ':H128' --selected '' --root /var/tmp/rev4-featpot/v2c

Step k evaluates `core+S+X` for every remaining group X against `core+S` (S = groups chosen so far, in order), seeds 0-4,
the five design folds, head N. gain(X) = mean over sources of the seed-paired signed-SROCC difference. The step's rule
(registered in E9′): choose the best X if gain >= 0.002 and the difference is > 0 on >= 3 of 5 sources; else stop.
`score` writes <root>/compare/e9_step<k>.json and prints the decision. Groups whose table family conflicts with S (the
aux-only peer arm p3 against main-table arms) are skipped and listed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from v2_common import CANDIDATES, LEGACY_BLOCKS, SOURCE_ORDER, V2, arm_columns, extra_arms

SEEDS = range(5)
HEAD = "N"
MIN_GAIN, MIN_SOURCES = 0.002, 3


def groups() -> list[str]:
    arms = [a for a in (*CANDIDATES, *extra_arms()["arms"]) if a not in ("all", "rall")]
    return [b for b in LEGACY_BLOCKS if b not in ("basic", "peaks")] + arms


def spec_of(selected: list[str], extra: str | None, recipe: str) -> str:
    parts = [*selected, *([extra] if extra else [])]
    return ("core+" + "+".join(parts) if parts else "core") + f"@h32{recipe}"


def compatible(selected: list[str], x: str) -> bool:
    try:
        arm_columns(spec_of(selected, x, ""))
        return True
    except ValueError:
        return False


def signed(spec: str, source: str, seed: int) -> float | None:
    path = V2 / "cells" / f"{spec}__{HEAD}" / f"without_{source}_s{seed}" / "result.json"
    if not path.is_file():
        return None
    fam = arm_columns(spec)[0]
    y = pq.read_table(V2 / "wide" / fam / "real" / f"{source}.keys.parquet", columns=["target"]).column(0).to_numpy()
    return float(spearmanr(json.loads(path.read_text())["prediction"], y)[0])


def cmd_grid(args, selected: list[str]) -> int:
    specs = [spec_of(selected, None, args.recipe)] + [spec_of(selected, x, args.recipe) for x in groups()
                                                       if x not in selected and compatible(selected, x)]
    cells = [{"name": f"{sp}__{HEAD}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", sp, "--head", HEAD, "--heldout", s, "--seed-index", str(i),
                       "--root", str(V2)]}
             for sp in specs for s in SOURCE_ORDER for i in SEEDS]
    if args.skip_base:
        cells = [c for c in cells if not c["name"].startswith(specs[0] + "__")]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": cells}, indent=1))
    print(json.dumps({"step": len(selected) + 1, "specs": len(specs), "cells": len(cells)}))
    return 0


def cmd_score(args, selected: list[str]) -> int:
    base = spec_of(selected, None, args.recipe)
    rows, skipped, missing = {}, [], 0
    for x in groups():
        if x in selected:
            continue
        if not compatible(selected, x):
            skipped.append(x)
            continue
        spec = spec_of(selected, x, args.recipe)
        per = {}
        for s in SOURCE_ORDER:
            d = []
            for i in SEEDS:
                a, b = signed(spec, s, i), signed(base, s, i)
                if a is None or b is None:
                    missing += 1
                    continue
                d.append(a - b)
            per[s] = {"delta": float(np.mean(d)) if d else None, "se": float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
                      "n": len(d)}
        deltas = [v["delta"] for v in per.values() if v["delta"] is not None]
        rows[x] = {"per_source": per, "gain": float(np.mean(deltas)) if len(deltas) == len(SOURCE_ORDER) else None,
                   "positive_sources": sum(d > 0 for d in deltas), "columns": len(arm_columns(spec)[2])}
    out = {"schema": "rev4-featpot-e9-forward-v1", "step": len(selected) + 1, "selected_before": selected,
           "recipe": args.recipe, "base": base, "candidates": rows, "skipped_incompatible": skipped, "missing_cells": missing}
    if missing:
        out["decision"] = {"status": "INCOMPLETE"}
    else:
        ranked = sorted(rows, key=lambda x: -rows[x]["gain"])
        best = ranked[0]
        ok = rows[best]["gain"] >= MIN_GAIN and rows[best]["positive_sources"] >= MIN_SOURCES
        out["ranking"] = [(x, round(rows[x]["gain"], 5), rows[x]["positive_sources"]) for x in ranked]
        out["decision"] = {"status": "ADD" if ok else "STOP", "group": best if ok else None,
                           "best_gain": rows[best]["gain"], "best_positive_sources": rows[best]["positive_sources"]}
    dest = V2 / "compare" / f"e9_step{len(selected) + 1}{args.recipe.replace(':', '_')}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["decision"]))
    for x, g, n in out.get("ranking", [])[:12]:
        print(f"  {x:10s} gain {g:+.4f}  positive {n}/5  cols {rows[x]['columns']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "score"])
    ap.add_argument("--recipe", default="", help="E8 recipe tokens, e.g. ':H128'")
    ap.add_argument("--selected", default="", help="comma-separated groups chosen so far, in order")
    ap.add_argument("--root", help="instrument root; read by v2_common from argv")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    ap.add_argument("--skip-base", action="store_true", help="omit the base spec's cells (already run as the previous winner)")
    args = ap.parse_args()
    selected = [x for x in args.selected.split(",") if x]
    return cmd_grid(args, selected) if args.cmd == "grid" else cmd_score(args, selected)


if __name__ == "__main__":
    sys.exit(main())
