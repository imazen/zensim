"""Design log E9′ method 2 (λ grid recalibrated 2026-10-02 01:10 MT): column-level group-lasso stability selection on all
1,853 main columns, core included — the penalty treats every column alike.

  python3 e9_lasso.py fits      --program-sha P --data-sha D --out SPEC.json   # stage 1: penalized fits + reference cells
  python3 e9_lasso.py survivors --root ROOT                                     # per-fold subsets -> <root>/compare/e9_lasso_subsets.json
  python3 e9_lasso.py refits    --root ROOT --program-sha P --data-sha D --out SPEC.json   # stage 2: unpenalized refits
  python3 e9_lasso.py score     --root ROOT                                     # -> <root>/compare/e9_lasso.json

Registered rules: fits `screen_main@h32:H128:gl<λ>`, λ ∈ {0.7, 1, 1.4, 2, 2.8}, head N, seeds 0-9, five design folds. A
column survives a fit when its layer-0 row is non-zero in the selected (final-epoch) bake. For held-out source s and
threshold t ∈ {0.5, 0.8}, the subset is the columns surviving in >= t of fold s's 10 seeds (fits that never saw s). Each
subset is refit without the penalty (`sel:<id>@h32:H128`, seeds 0-4) and scored on s. Reported per (λ, t): the seed-paired
held-out gain over core (`set:basic+peaks`) and over R0 (`r0`), both at H128, per source and mean; positive-source count;
columns per fold; family composition. No sealed label is read.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from v2_common import LEGACY_BLOCKS, SOURCE_ORDER, V2, selection_id

LAMBDAS = ("0.7", "1", "1.4", "2", "2.8")
THRESHOLDS = (0.5, 0.8)
FIT_SEEDS, REFIT_SEEDS = range(10), range(5)
HEAD, RECIPE = "N", ":H128"
BASES = {"core": "set:basic+peaks@h32:H128", "r0": "r0@h32:H128"}


def fit_spec(lam: str) -> str:
    return f"screen_main@h32{RECIPE}:gl{lam}"


def cell(spec: str, source: str, seed: int, columns: list[int] | None = None) -> dict:
    argv = ["v2_lodo_mlp.py", "--spec", spec, "--head", HEAD, "--heldout", source, "--seed-index", str(seed), "--root", str(V2)]
    if columns is not None:
        argv += ["--columns", ",".join(map(str, columns))]
    return {"name": f"{spec}__{HEAD}/without_{source}_s{seed}", "argv": argv}


def write_spec(args, cells: list[dict]) -> None:
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "out": args.out}))


def cmd_fits(args) -> int:
    cells = [cell(fit_spec(lam), s, i) for lam in LAMBDAS for s in SOURCE_ORDER for i in FIT_SEEDS]
    cells += [cell(BASES["core"], s, i) for s in SOURCE_ORDER for i in REFIT_SEEDS]
    cells += [cell(BASES["r0"], s, i) for s in SOURCE_ORDER for i in REFIT_SEEDS]  # seeds 0-2 exist from E8
    write_spec(args, cells)
    return 0


def survivors_of(result_path: Path) -> list[int]:
    from bake_survivors import survivors
    bake = json.loads(result_path.read_text())["selected_bake"]
    return survivors(str(result_path.parent / "refit" / Path(bake).name))


def cmd_survivors(args) -> int:
    out, missing = {"schema": "rev4-featpot-e9-lasso-subsets-v1", "subsets": {}, "survivors": {}}, 0
    for lam in LAMBDAS:
        for s in SOURCE_ORDER:
            per_seed = []
            for i in FIT_SEEDS:
                path = V2 / "cells" / f"{fit_spec(lam)}__{HEAD}" / f"without_{s}_s{i}" / "result.json"
                if not path.is_file():
                    missing += 1
                    continue
                per_seed.append(survivors_of(path))
            out["survivors"][f"gl{lam}/{s}"] = [len(x) for x in per_seed]
            if len(per_seed) != len(FIT_SEEDS):
                continue
            counts = np.zeros(max((max(x) for x in per_seed if x), default=0) + 1, dtype=int)
            for x in per_seed:
                counts[x] += 1
            for t in THRESHOLDS:
                cols = [int(c) for c in np.flatnonzero(counts >= t * len(FIT_SEEDS))]
                out["subsets"][f"gl{lam}_t{t}/{s}"] = {"id": selection_id(cols) if cols else None, "columns": cols}
    out["missing_fits"] = missing
    dest = V2 / "compare" / "e9_lasso_subsets.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"subsets": len(out["subsets"]), "missing_fits": missing, "out": str(dest)}))
    return 0 if not missing else 1


def subsets() -> dict:
    record = json.loads((V2 / "compare" / "e9_lasso_subsets.json").read_text())
    if record["missing_fits"]:
        raise ValueError("survivor subsets were built from incomplete fits")
    return record["subsets"]


def cmd_refits(args) -> int:
    cells, seen = [], set()
    for key, sub in subsets().items():
        s = key.split("/")[1]
        if not sub["columns"]:
            continue  # an empty subset has nothing to refit; score reports it
        for i in REFIT_SEEDS:
            c = cell(f"sel:{sub['id']}@h32{RECIPE}", s, i, sub["columns"])
            if c["name"] not in seen:
                seen.add(c["name"])
                cells.append(c)
    write_spec(args, cells)
    return 0


def signed(spec: str, source: str, seed: int) -> float | None:
    path = V2 / "cells" / f"{spec}__{HEAD}" / f"without_{source}_s{seed}" / "result.json"
    if not path.is_file():
        return None
    y = pq.read_table(V2 / "wide" / "main" / "real" / f"{source}.keys.parquet", columns=["target"]).column(0).to_numpy()
    return float(spearmanr(json.loads(path.read_text())["prediction"], y)[0])


def composition(cols: list[int]) -> dict:
    lists = json.loads((V2 / "wide" / "keep_lists.json").read_text())["specs"]
    owner = {}
    for name, block in LEGACY_BLOCKS.items():
        for c in block:
            owner[c] = name
    for arm, entry in lists.items():  # an arm's own columns are its keep list beyond the 944-column bank
        if "~p" in arm or arm in ("r0", "all", "rall", "screen_main", "screen_aux", "minus_basic") or arm.startswith("oracle"):
            continue
        for c in entry["keep"]:
            if c >= 944:
                owner.setdefault(c, arm)
    out = {}
    for c in cols:
        out[owner.get(c, "other")] = out.get(owner.get(c, "other"), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def cmd_score(args) -> int:
    subs, rows, missing = subsets(), {}, 0
    for lam in LAMBDAS:
        for t in THRESHOLDS:
            label = f"gl{lam}_t{t}"
            per = {}
            for s in SOURCE_ORDER:
                sub = subs[f"{label}/{s}"]
                entry = {"columns": len(sub["columns"]), "id": sub["id"]}
                for base, bspec in BASES.items():
                    vals = []
                    for i in REFIT_SEEDS:
                        if not sub["columns"]:  # an empty subset has no refit by design: reported, not missing
                            continue
                        a = signed(f"sel:{sub['id']}@h32{RECIPE}", s, i)
                        b = signed(bspec, s, i)
                        if a is None or b is None:
                            missing += 1
                            continue
                        vals.append(a - b)
                    entry[f"gain_vs_{base}"] = float(np.mean(vals)) if vals else None
                    entry[f"se_vs_{base}"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None
                per[s] = entry
            row = {"per_source": per, "columns_mean": float(np.mean([p["columns"] for p in per.values()]))}
            for base in BASES:
                v = [p[f"gain_vs_{base}"] for p in per.values()]
                row[f"mean_gain_vs_{base}"] = float(np.mean(v)) if all(x is not None for x in v) else None
                row[f"positive_vs_{base}"] = sum(x is not None and x > 0 for x in v)
            union = sorted({c for s in SOURCE_ORDER for c in subs[f"{label}/{s}"]["columns"]})
            row["composition_union"] = composition(union)
            rows[label] = row
    out = {"schema": "rev4-featpot-e9-lasso-v1", "head": HEAD, "recipe": RECIPE, "bases": BASES, "rows": rows,
           "missing_cells": missing, "status": "INCOMPLETE" if missing else "complete"}
    dest = V2 / "compare" / "e9_lasso.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for label, r in sorted(rows.items(), key=lambda kv: -(kv[1]["mean_gain_vs_r0"] or -9)):
        print(f"  {label:10s} cols {r['columns_mean']:7.1f}  vs R0 {r['mean_gain_vs_r0'] if r['mean_gain_vs_r0'] is not None else float('nan'):+.4f}"
              f" ({r['positive_vs_r0']}/5)  vs core {r['mean_gain_vs_core'] if r['mean_gain_vs_core'] is not None else float('nan'):+.4f}"
              f" ({r['positive_vs_core']}/5)")
    print(json.dumps({"status": out["status"], "missing_cells": missing, "out": str(dest)}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fits", "survivors", "refits", "score"])
    ap.add_argument("--root", help="instrument root; read by v2_common from argv")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    args = ap.parse_args()
    return {"fits": cmd_fits, "survivors": cmd_survivors, "refits": cmd_refits, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
