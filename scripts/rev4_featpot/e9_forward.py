"""Design log E9″ method 1: bidirectional stepwise group selection from the EMPTY set, equal footing for every group —
all seven legacy blocks (basic and peaks included: nothing is exempt) and the 17 registered arms (their own columns).

  python3 e9_forward.py grid  --phase add  --recipe ':H128' --selected 'v2,c3' --out ROUND.json   # cells of one move
  python3 e9_forward.py score --phase add  --recipe ':H128' --selected 'v2,c3' --root ROOT
  python3 e9_forward.py grid  --phase drop --recipe ':H128' --selected 'v2,c3,iw' --out ROUND.json
  python3 e9_forward.py score --phase drop --recipe ':H128' --selected 'v2,c3,iw' --root ROOT

Rules (registered in E9″; head N, seeds 0-4, five design folds, signed held-out SROCC):
  add,  S empty : every group alone; choose the highest mean SROCC over the five sources.
  add,  S given : for each remaining compatible X, gain = mean seed-paired Δ(S+X − S); choose the best if gain >= 0.002 and
                  Δ > 0 on >= 3 of 5 sources, else STOP.
  drop, |S| >= 2: for each g in S, loss = mean seed-paired Δ(S − S∖g); drop the smallest-loss g if loss < 0.001, else KEEP.
Spec of a set = `set:` + the groups in selection order, so a move's winning spec is the next move's base by name.
`score` writes <root>/compare/e9_<phase><k>.json and prints the decision.
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
MIN_GAIN, MIN_SOURCES, MAX_DROP_LOSS = 0.002, 3, 0.001


def groups() -> list[str]:
    return [*LEGACY_BLOCKS, *[a for a in (*CANDIDATES, *extra_arms()["arms"]) if a not in ("all", "rall")]]


def spec_of(parts: list[str], recipe: str) -> str:
    return "set:" + "+".join(parts) + f"@h32{recipe}"


def compatible(parts: list[str]) -> bool:
    try:
        arm_columns(spec_of(parts, ""))
        return True
    except ValueError:
        return False


def candidates(phase: str, selected: list[str]) -> dict[str, list[str]]:
    """{label: group list} the move evaluates (the base set itself is not included)."""
    if phase == "add":
        return {x: [*selected, x] for x in groups() if x not in selected and compatible([*selected, x])}
    return {g: [x for x in selected if x != g] for g in selected}


def signed(spec: str, source: str, seed: int) -> float | None:
    path = V2 / "cells" / f"{spec}__{HEAD}" / f"without_{source}_s{seed}" / "result.json"
    if not path.is_file():
        return None
    fam = arm_columns(spec)[0]
    y = pq.read_table(V2 / "wide" / fam / "real" / f"{source}.keys.parquet", columns=["target"]).column(0).to_numpy()
    return float(spearmanr(json.loads(path.read_text())["prediction"], y)[0])


def cmd_grid(args, selected: list[str]) -> int:
    specs = [spec_of(parts, args.recipe) for parts in candidates(args.phase, selected).values()]
    if args.include_base and selected:
        specs.insert(0, spec_of(selected, args.recipe))
    cells = [{"name": f"{sp}__{HEAD}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", sp, "--head", HEAD, "--heldout", s, "--seed-index", str(i), "--root", str(V2)]}
             for sp in specs for s in SOURCE_ORDER for i in SEEDS]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": cells}, indent=1))
    print(json.dumps({"phase": args.phase, "selected": selected, "specs": len(specs), "cells": len(cells)}))
    return 0


def cmd_score(args, selected: list[str]) -> int:
    base = spec_of(selected, args.recipe) if selected else None
    rows, missing = {}, 0
    for label, parts in candidates(args.phase, selected).items():
        spec = spec_of(parts, args.recipe)
        per = {}
        for s in SOURCE_ORDER:
            vals = []
            for i in SEEDS:
                a = signed(spec, s, i)
                b = signed(base, s, i) if base else 0.0
                if a is None or b is None:
                    missing += 1
                    continue
                # add: Δ = S+X − S (or the absolute SROCC from empty); drop: Δ = S − S∖g, so a positive value is a loss
                vals.append(a - b if args.phase == "add" else b - a)
            per[s] = {"value": float(np.mean(vals)) if vals else None,
                      "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None, "n": len(vals)}
        v = [x["value"] for x in per.values() if x["value"] is not None]
        rows[label] = {"spec": spec, "per_source": per, "mean": float(np.mean(v)) if len(v) == len(SOURCE_ORDER) else None,
                       "positive_sources": sum(x > 0 for x in v), "columns": len(arm_columns(spec)[2])}
    k = len(selected) + (1 if args.phase == "add" else 0)
    out = {"schema": "rev4-featpot-e9-stepwise-v1", "phase": args.phase, "selected_before": selected, "recipe": args.recipe,
           "base": base, "candidates": rows, "missing_cells": missing}
    if missing:
        out["decision"] = {"status": "INCOMPLETE"}
    elif args.phase == "add":
        ranked = sorted(rows, key=lambda x: -rows[x]["mean"])
        best = ranked[0]
        if not selected:
            ok = True
        else:
            ok = rows[best]["mean"] >= MIN_GAIN and rows[best]["positive_sources"] >= MIN_SOURCES
        out["ranking"] = [(x, rows[x]["mean"], rows[x]["positive_sources"], rows[x]["columns"]) for x in ranked]
        out["decision"] = {"status": "ADD" if ok else "STOP", "group": best if ok else None,
                           "selected_after": [*selected, best] if ok else selected, "best": rows[best]["mean"]}
    else:
        ranked = sorted(rows, key=lambda x: rows[x]["mean"])
        best = ranked[0]
        ok = len(selected) >= 2 and rows[best]["mean"] < MAX_DROP_LOSS
        out["ranking"] = [(x, rows[x]["mean"], rows[x]["positive_sources"], rows[x]["columns"]) for x in ranked]
        out["decision"] = {"status": "DROP" if ok else "KEEP", "group": best if ok else None,
                           "selected_after": [x for x in selected if x != best] if ok else selected, "smallest_loss": rows[best]["mean"]}
    dest = V2 / "compare" / f"e9_{args.phase}{k}_{'-'.join(selected) or 'empty'}{args.recipe.replace(':', '_')}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["decision"]))
    for x, m, n, c in out.get("ranking", [])[:10]:
        print(f"  {x:10s} {'gain' if args.phase == 'add' else 'loss'} {m:+.4f}  positive {n}/5  cols {c}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "score"])
    ap.add_argument("--phase", choices=["add", "drop"], default="add")
    ap.add_argument("--recipe", default="", help="E8 recipe tokens, e.g. ':H128'")
    ap.add_argument("--selected", default="", help="comma-separated groups in S, in selection order")
    ap.add_argument("--root", help="instrument root; read by v2_common from argv")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    ap.add_argument("--include-base", action="store_true", help="also emit the base set's cells (normally run by the previous move)")
    args = ap.parse_args()
    selected = [x for x in args.selected.split(",") if x]
    return cmd_grid(args, selected) if args.cmd == "grid" else cmd_score(args, selected)


if __name__ == "__main__":
    sys.exit(main())
