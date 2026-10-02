"""Design log E10: multi-start bidirectional stepwise selection (E9″'s rules from several forced starting groups).

  python3 e10_multistart.py init  --root ROOT --state STATE.json --starts append,c3,b2,basic,p3,masked
  python3 e10_multistart.py grid  --root ROOT --state STATE.json --out SPEC.json --program-sha P --data-sha D
  python3 e10_multistart.py score --root ROOT --state STATE.json        # applies one move to every active path

A cell is identified by its group SET: the keep list (sorted columns), seeds and fold depend only on the set, so a result
stored under any group order (E9″ named sets in selection order) is reused. New cells use sorted group names. Paths advance
together: `grid` emits the union of every active path's next-move cells that have no result under any order; `score`
applies each path's move with the E9″ rules (e9_forward MIN_GAIN, MIN_SOURCES, MAX_DROP_LOSS) and records it in the state
and in <root>/compare/e10_<start>_<move>.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from e9_forward import HEAD, MAX_DROP_LOSS, MIN_GAIN, MIN_SOURCES, SEEDS, groups, compatible
from v2_common import SOURCE_ORDER, V2, arm_columns

MAX_ADDS = 10


def canonical(parts, recipe: str) -> str:
    return "set:" + "+".join(sorted(parts)) + f"@h32{recipe}"


def index(recipe: str) -> dict:
    """frozenset(groups) -> existing spec names (any order) under <root>/cells with this recipe and head."""
    out = {}
    suffix = f"@h32{recipe}__{HEAD}"
    for d in (V2 / "cells").glob(f"set:*{suffix}"):
        spec = d.name[: -len(f"__{HEAD}")]
        core = spec[len("set:"): -len(f"@h32{recipe}")]
        if "@" in core:  # another recipe whose tokens extend this one
            continue
        out.setdefault(frozenset(core.split("+")), []).append(spec)
    return out


def result_path(spec: str, source: str, seed: int) -> Path:
    return V2 / "cells" / f"{spec}__{HEAD}" / f"without_{source}_s{seed}" / "result.json"


def find(idx: dict, parts, recipe: str, source: str, seed: int):
    for spec in idx.get(frozenset(parts), []):
        p = result_path(spec, source, seed)
        if p.is_file():
            return p
    return None


_TARGETS = {}


def signed(path: Path, spec_for_family: str, source: str) -> float:
    fam = arm_columns(spec_for_family)[0]
    if (fam, source) not in _TARGETS:
        _TARGETS[(fam, source)] = pq.read_table(V2 / "wide" / fam / "real" / f"{source}.keys.parquet",
                                                columns=["target"]).column(0).to_numpy()
    return float(spearmanr(json.loads(path.read_text())["prediction"], _TARGETS[(fam, source)])[0])


def move_sets(path: dict) -> dict:
    """{label: group list} this path's next move evaluates."""
    s = path["selected"]
    if path["phase"] == "add":
        return {x: [*s, x] for x in groups() if x not in s and compatible([*s, x])}
    return {g: [x for x in s if x != g] for g in s}


def cmd_init(args) -> int:
    starts = [x for x in args.starts.split(",") if x]
    bad = [x for x in starts if x not in groups()]
    if bad:
        raise SystemExit(f"unknown groups {bad}")
    state = {"schema": "rev4-featpot-e10-v1", "recipe": args.recipe, "round": 0,
             "paths": {g: {"selected": [g], "phase": "add", "adds": 0, "status": "active", "history": []} for g in starts}}
    Path(args.state).write_text(json.dumps(state, indent=1) + "\n")
    print(json.dumps({"paths": starts}))
    return 0


def cmd_grid(args) -> int:
    state = json.loads(Path(args.state).read_text())
    recipe, idx = state["recipe"], index(state["recipe"])
    need = {}
    for path in state["paths"].values():
        if path["status"] != "active":
            continue
        for parts in [path["selected"], *move_sets(path).values()]:
            if not parts:
                continue
            for s in SOURCE_ORDER:
                for i in SEEDS:
                    if find(idx, parts, recipe, s, i) is None:
                        need.setdefault(canonical(parts, recipe), set()).add((s, i))
    cells = [{"name": f"{sp}__{HEAD}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", sp, "--head", HEAD, "--heldout", s, "--seed-index", str(i), "--root", str(V2)]}
             for sp in sorted(need, key=lambda sp: -len(arm_columns(sp)[2])) for s, i in sorted(need[sp])]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": cells}, indent=1))
    width = max((len(arm_columns(sp)[2]) for sp in need), default=0)
    print(json.dumps({"specs": len(need), "cells": len(cells), "widest": width}))
    return 0


def cmd_score(args) -> int:
    state = json.loads(Path(args.state).read_text())
    recipe, idx = state["recipe"], index(state["recipe"])
    summary = {}
    for start, path in state["paths"].items():
        if path["status"] != "active":
            continue
        base = path["selected"]
        rows, missing = {}, 0
        for label, parts in move_sets(path).items():
            per = {}
            for s in SOURCE_ORDER:
                vals = []
                for i in SEEDS:
                    pa, pb = find(idx, parts, recipe, s, i), find(idx, base, recipe, s, i)
                    if pa is None or pb is None:
                        missing += 1
                        continue
                    a = signed(pa, canonical(parts, recipe), s)
                    b = signed(pb, canonical(base, recipe), s)
                    vals.append(a - b if path["phase"] == "add" else b - a)
                per[s] = {"value": float(np.mean(vals)) if vals else None, "n": len(vals)}
            v = [x["value"] for x in per.values() if x["value"] is not None]
            rows[label] = {"per_source": per, "mean": float(np.mean(v)) if len(v) == len(SOURCE_ORDER) else None,
                           "positive_sources": sum(x > 0 for x in v), "columns": len(arm_columns(canonical(parts, recipe))[2])}
        if missing:
            summary[start] = {"status": "INCOMPLETE", "missing": missing}
            continue
        if path["phase"] == "add":
            best = max(rows, key=lambda x: rows[x]["mean"])
            ok = rows[best]["mean"] >= MIN_GAIN and rows[best]["positive_sources"] >= MIN_SOURCES
            dec = {"status": "ADD" if ok else "STOP", "group": best if ok else None, "best": rows[best]["mean"]}
        else:
            best = min(rows, key=lambda x: rows[x]["mean"])
            ok = len(base) >= 2 and rows[best]["mean"] < MAX_DROP_LOSS
            dec = {"status": "DROP" if ok else "KEEP", "group": best if ok else None, "smallest_loss": rows[best]["mean"]}
        k = len(path["history"]) + 1
        rec = {"schema": "rev4-featpot-e10-move-v1", "start": start, "move": k, "phase": path["phase"], "selected_before": base,
               "candidates": rows, "decision": dec}
        (V2 / "compare").mkdir(parents=True, exist_ok=True)
        (V2 / "compare" / f"e10_{start}_{k:02d}.json").write_text(json.dumps(rec, indent=1) + "\n")
        path["history"].append({"phase": path["phase"], "before": base, **dec})
        st = dec["status"]
        if st == "ADD":
            path["selected"] = [*base, dec["group"]]
            path["adds"] += 1
            path["phase"] = "drop"
        elif st == "DROP":
            path["selected"] = [x for x in base if x != dec["group"]]
            path["phase"] = "drop" if len(path["selected"]) >= 2 else "add"
        elif st == "KEEP":
            path["phase"] = "add"
        else:  # STOP
            path["status"] = "done"
        if path["status"] == "active" and path["phase"] == "add" and path["adds"] >= MAX_ADDS:
            path["status"] = "done"
        summary[start] = {**dec, "selected_after": path["selected"], "path_status": path["status"]}
    if any(m.get("status") != "INCOMPLETE" for m in summary.values()):
        state["round"] += 1
    Path(args.state).write_text(json.dumps(state, indent=1) + "\n")
    print(json.dumps({"round": state["round"], "moves": summary,
                      "active": [g for g, p in state["paths"].items() if p["status"] == "active"]}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["init", "grid", "score"])
    ap.add_argument("--root", help="instrument root; read by v2_common from argv")
    ap.add_argument("--state", required=True)
    ap.add_argument("--starts", default="append,c3,b2,basic,p3,masked")
    ap.add_argument("--recipe", default=":H128")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    args = ap.parse_args()
    return {"init": cmd_init, "grid": cmd_grid, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
