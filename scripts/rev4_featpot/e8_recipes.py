"""Design log E8: training-recipe sweep on the canon instrument. For each recipe (A = H32, the registered canon cells;
B = H64; C = H128; D = H128 + group-L1 1e-4; E = H128 + group-L1 1e-3) and head: R0's mean held-out signed SROCC over the
five design sources (seeds 0-2) and absorption = mean over sources of the seed-paired (screen_main - R0) difference.
Applies E8's selection rule (head N decides: among recipes whose R0 mean is within 0.003 of the best, the largest
absorption; ties within 0.001 -> smaller H, then no group-L1) and writes <root>/compare/e8_recipes.json.

  python3 e8_recipes.py --root /var/tmp/rev4-featpot/v2c
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from v2_common import SOURCE_ORDER, V2

RECIPES = {"A": ("", 32, 0.0), "B": (":H64", 64, 0.0), "C": (":H128", 128, 0.0), "D": (":H128:gl0.0001", 128, 1e-4),
           "E": (":H128:gl0.001", 128, 1e-3)}
SEEDS = range(3)
FAMILY = {"r0": "main", "screen_main": "main"}


def signed_srocc(spec: str, head: str, source: str, seed: int) -> float | None:
    path = V2 / "cells" / f"{spec}__{head}" / f"without_{source}_s{seed}" / "result.json"
    if not path.is_file():
        return None
    pred = json.loads(path.read_text())["prediction"]
    y = pq.read_table(V2 / "wide" / FAMILY[spec.split("@")[0]] / "real" / f"{source}.keys.parquet",
                      columns=["target"]).column(0).to_numpy()
    return float(spearmanr(pred, y)[0])


def main() -> int:
    out = {"schema": "rev4-featpot-e8-recipes-v1", "seeds": list(SEEDS), "recipes": {}}
    for key, (tok, hidden, gl) in RECIPES.items():
        rec = {"tokens": tok, "hidden": hidden, "group_l1": gl}
        for head in ("N", "F"):
            r0, absorb, missing = {}, {}, 0
            for s in SOURCE_ORDER:
                a = [signed_srocc(f"r0@h32{tok}", head, s, i) for i in SEEDS]
                b = [signed_srocc(f"screen_main@h32{tok}", head, s, i) for i in SEEDS]
                missing += sum(x is None for x in a + b)
                pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
                r0[s] = float(np.mean([x for x in a if x is not None])) if any(x is not None for x in a) else None
                absorb[s] = float(np.mean([y - x for x, y in pairs])) if pairs else None
            rec[head] = {"r0_by_source": r0, "absorption_by_source": absorb, "missing_cells": missing,
                         "r0_mean": None if None in r0.values() else float(np.mean(list(r0.values()))),
                         "absorption_mean": None if None in absorb.values() else float(np.mean(list(absorb.values())))}
        out["recipes"][key] = rec
    n = {k: v["N"] for k, v in out["recipes"].items()}
    complete = all(v["missing_cells"] == 0 for v in n.values())
    if complete:
        best_r0 = max(v["r0_mean"] for v in n.values())
        ok = [k for k, v in n.items() if v["r0_mean"] >= best_r0 - 0.003]
        top = max(n[k]["absorption_mean"] for k in ok)
        tied = [k for k in ok if n[k]["absorption_mean"] >= top - 0.001]
        tied.sort(key=lambda k: (RECIPES[k][1], RECIPES[k][2]))
        out["selection"] = {"status": "SELECTED", "recipe": tied[0], "tokens": RECIPES[tied[0]][0],
                            "eligible_by_r0": ok, "best_r0_mean_N": best_r0, "tied": tied,
                            "absorption_N": n[tied[0]]["absorption_mean"],
                            "absorption_nonnegative": n[tied[0]]["absorption_mean"] >= 0}
    else:
        out["selection"] = {"status": "INCOMPLETE"}
    dest = V2 / "compare" / "e8_recipes.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["recipes"].items():
        print(k, f"{v['tokens'] or ':H32':16s}", " ".join(
            f"{h}: R0 {v[h]['r0_mean'] if v[h]['r0_mean'] is None else round(v[h]['r0_mean'], 4)} "
            f"absorb {v[h]['absorption_mean'] if v[h]['absorption_mean'] is None else round(v[h]['absorption_mean'], 4)}"
            for h in ("N", "F")))
    print(json.dumps(out["selection"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
