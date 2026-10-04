"""Design log E23 (registered 2026-10-03 23:20 MT, before any E23 cell): is by_v2fy as good as v2 + basic in the product form,
a 5-member seed ensemble, under the adopted recipe cv16:cf98?

  python3 e23_ensemble.py grid  --root ROOT --out SPEC.json --program-sha P --data-sha D   # both arms, seeds 10-19
  python3 e23_ensemble.py score --root ROOT                                               # -> <root>/compare/e23_decision.json

Arms: `sel:59f0bbc2f290@h32:H128:cv16:cf98` (by_v2fy) and `set:v2+basic@h32:H128:cv16:cf98`; seeds 0-9 exist (E16/E17/E21),
seeds 10-19 are new here (100 cells). An ensemble = the mean of five seed models' held-out predictions (the method of the
2026-10-03 best-seeds analysis, `seed_selection_analysis.py`), scored with e13's signed SROCC convention.
Primary: per fold, the four disjoint 5-seed groups {0-4, 5-9, 10-14, 15-19}; Δ = by_v2fy ensemble - v2 + basic ensemble on the same
group; per-source mean over the four groups. AS GOOD iff the mean over sources >= -0.002 AND every source >= -0.005 (E12's rule).
Descriptive: ensemble W2 Δ (worst three distortion types, KADID/TID), the 20-seed single-model paired Δ (e13.score_arms), and the
per-cell external NITS / LIVE / MCIQA Δ over seeds 0-19.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from v2_common import SOURCE_ORDER, V2, selection_id

REPO = Path(__file__).resolve().parents[2]
RECIPE = ":H128:cv16:cf98"
HEAD = "N"
NEW_SEEDS, ALL_SEEDS = range(10, 20), range(20)
GROUPS = [range(0, 5), range(5, 10), range(10, 15), range(15, 20)]
MEAN_FLOOR, SOURCE_FLOOR = -0.002, -0.005


def by_v2fy_columns() -> list:
    d = json.loads((REPO / "benchmarks/costset2_2026-10-03.candidate_ids.json").read_text())
    return d.get("candidates", d)["by_v2fy"]


ARM = f"sel:{selection_id(by_v2fy_columns())}@h32{RECIPE}"
CONTROL = f"set:v2+basic@h32{RECIPE}"


def cell(spec: str, source: str, seed: int) -> Path:
    return V2 / "cells" / f"{spec}__{HEAD}" / f"without_{source}_s{seed}" / "result.json"


def cmd_grid(args) -> int:
    cols = by_v2fy_columns()
    cells = []
    for sp in (ARM, CONTROL):
        for s in SOURCE_ORDER:
            for i in NEW_SEEDS:
                argv = ["v2_lodo_mlp.py", "--spec", sp, "--head", HEAD, "--heldout", s, "--seed-index", str(i), "--root", str(V2)]
                if sp.startswith("sel:"):
                    argv += ["--columns", ",".join(map(str, cols))]
                cells.append({"name": f"{sp}__{HEAD}/without_{s}_s{i}", "argv": argv})
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "arm": ARM, "control": CONTROL}))
    return 0


def ensemble_stats(spec: str, source: str, seeds, meta) -> dict | None:
    """Signed SROCC and W2 of the mean of the seed models' held-out predictions (None if any member is missing)."""
    import e13_teacher as e13
    paths = [cell(spec, source, i) for i in seeds]
    if not all(p.is_file() for p in paths):
        return None
    preds = [np.asarray(json.loads(p.read_text())["prediction"], dtype=np.float64) for p in paths]
    pred, y = np.mean(preds, axis=0), meta.target.to_numpy(dtype=np.float64)
    sign = 1.0 if e13.spearman(pred, y) >= 0 else -1.0  # e13.worst_case's orientation rule
    out = {"signed": sign * e13.spearman(pred, y)}
    if "dtype" in meta:
        per_type = sorted(sign * e13.spearman(pred[g], y[g]) for g in meta.groupby("dtype").indices.values())
        out["w2"] = float(np.mean(per_type[:3]))
    return out


def cmd_score(args) -> int:
    import e13_teacher as e13
    groups = [g for g in GROUPS if max(g) < args.max_seed + 1]
    per, w2d, missing = {}, [], 0
    for s in SOURCE_ORDER:
        meta = e13.heldout_meta(s)
        deltas = []
        for g in groups:
            a, b = ensemble_stats(ARM, s, g, meta), ensemble_stats(CONTROL, s, g, meta)
            if a is None or b is None:
                missing += 1
                continue
            deltas.append(a["signed"] - b["signed"])
            if "w2" in a:
                w2d.append(a["w2"] - b["w2"])
        per[s] = {"delta": float(np.mean(deltas)) if deltas else None,
                  "se": float(np.std(deltas, ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else None, "n": len(deltas)}
    d = [per[s]["delta"] for s in SOURCE_ORDER]
    complete = not missing and all(x is not None for x in d)
    out = {"arm": ARM, "control": CONTROL, "groups": [list(g) for g in groups], "per_source": per,
           "mean": float(np.mean(d)) if complete else None,
           "se": (math.sqrt(sum(per[s]["se"] ** 2 for s in SOURCE_ORDER)) / len(SOURCE_ORDER)
                  if complete and all(per[s]["se"] is not None for s in SOURCE_ORDER) else None),
           "w2_delta": float(np.mean(w2d)) if w2d else None,
           "w2_se": float(np.std(w2d, ddof=1) / math.sqrt(len(w2d))) if len(w2d) > 1 else None,
           "missing_groups": missing, "status": "complete" if complete else "INCOMPLETE"}
    out["as_good"] = bool(complete and out["mean"] >= MEAN_FLOOR and min(d) >= SOURCE_FLOOR)
    if args.dry_run:
        print(json.dumps(out))
        return 0
    e13.SEEDS, e13.BASE = ALL_SEEDS, CONTROL
    e13.score_arms([("by_v2fy", ARM)], "e23_single20", args.monotonicity)
    single = json.loads((V2 / "compare" / "e23_single20.json").read_text())["rows"]["by_v2fy"]
    out["single20"] = {"signed_mean": single["signed"]["mean"], "signed_se": single["signed"]["se"],
                       "signed_worst": single["signed"]["worst"], "w2": single["w2_type_worst3"]["mean"]}
    subprocess.run([sys.executable, str(Path(__file__).with_name("external_sets.py")), "score", "--root", str(V2),
                    "--specs", f"{CONTROL},{ARM}", "--seeds", "0-19", "--out", "e23_external"], check=True, capture_output=True)
    ext = json.loads((V2 / "compare" / "e23_external.json").read_text())["specs"][ARM]
    out["external_single20"] = {st: {"delta": ext[st]["all"]["delta"], "se": ext[st]["all"]["se"]} for st in ("nits", "live", "mciqa")}
    (V2 / "compare" / "e23_decision.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return 0 if complete else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "score"])
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    ap.add_argument("--monotonicity", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="primary statistic only, no writes (code check on existing seeds)")
    ap.add_argument("--max-seed", type=int, default=19, help="dry-run code check on a prefix of the seed groups")
    args = ap.parse_args()
    return {"grid": cmd_grid, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
