"""Design log E21 (registered 2026-10-03 20:30 MT, before any E21 cell): do the two cheap "as good" sets stay as good as
v2 + basic under the adopted coverage recipe cv16:cf98?

  python3 e21_cheap_recipe.py grid  --root ROOT --out SPEC.json --program-sha P --data-sha D   # 2 arms x 5 folds x seeds 0-9
  python3 e21_cheap_recipe.py score --root ROOT                                               # -> <root>/compare/e21_decision.json

Arms: by_v2fy (COSTSET2, 420 columns) and b228_v2s123 (COSTSET, 489 columns) as `sel:<id>@h32:H128:cv16:cf98`, head N.
Control: v2 + basic at the same recipe (`set:v2+basic@h32:H128:cv16:cf98`, E16/E17 cells, seeds 0-9).
Primary, per arm, seed-paired over seeds 0-9: E12's "as good" rule (mean signed Δ >= -0.002 and every source >= -0.005)
AND the worst-three-types W2 Δ not worse than -2 SE. Descriptive (reported, not gating): per-source Δ, W1, and the external
NITS / LIVE / MCIQA Δ against the control on seeds 0-9. Costs are COSTPOT's (benchmarks/costpot_2026-10-03.md).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from v2_common import SOURCE_ORDER, V2, selection_id

REPO = Path(__file__).resolve().parents[2]
RECIPE = ":H128:cv16:cf98"
CONTROL = f"set:v2+basic@h32{RECIPE}"
HEAD, SEEDS = "N", range(10)
MEAN_FLOOR, SOURCE_FLOOR = -0.002, -0.005
ARMS = {  # label -> (candidate file, key)
    "by_v2fy": ("benchmarks/costset2_2026-10-03.candidate_ids.json", "by_v2fy"),
    "b228_v2s123": ("benchmarks/costset_2026-10-02.candidate_ids.json", "b228_v2s123"),
}


def columns(label: str) -> list:
    f, key = ARMS[label]
    d = json.loads((REPO / f).read_text())
    return d.get("candidates", d)[key]


def spec(label: str) -> str:
    return f"sel:{selection_id(columns(label))}@h32{RECIPE}"


def cmd_grid(args) -> int:
    cells = [{"name": f"{spec(a)}__{HEAD}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", spec(a), "--head", HEAD, "--heldout", s, "--seed-index", str(i),
                       "--root", str(V2), "--columns", ",".join(map(str, columns(a)))]}
             for a in ARMS for s in SOURCE_ORDER for i in SEEDS]
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "specs": {a: spec(a) for a in ARMS}}))
    return 0


def cmd_score(args) -> int:
    import e13_teacher as e13
    e13.SEEDS, e13.BASE = SEEDS, CONTROL
    rc = e13.score_arms([(a, spec(a)) for a in ARMS], "e21_cheap_recipe", args.monotonicity)
    rows = json.loads((V2 / "compare" / "e21_cheap_recipe.json").read_text())["rows"]
    out = {}
    for a in ARMS:
        r = rows[a]
        sig, w2 = r["signed"], r["w2_type_worst3"]
        out[a] = {"spec": spec(a), "signed_mean": sig["mean"], "signed_se": sig["se"], "signed_worst": sig["worst"],
                  "per_source": {s: r["per_source"][s]["signed"] for s in SOURCE_ORDER}, "w2": w2["mean"], "w2_se": w2["se"],
                  "as_good": bool(sig["mean"] >= MEAN_FLOOR and sig["worst"] >= SOURCE_FLOOR and w2["mean"] > -2 * w2["se"])}
    subprocess.run([sys.executable, str(Path(__file__).with_name("external_sets.py")), "score", "--root", str(V2),
                    "--specs", ",".join([CONTROL, *(spec(a) for a in ARMS)]), "--seeds", "0-9", "--out", "e21_external"],
                   check=True, capture_output=True)
    ext = json.loads((V2 / "compare" / "e21_external.json").read_text())["specs"]
    for a in ARMS:
        out[a]["external"] = {st: {"delta": ext[spec(a)][st]["all"]["delta"], "se": ext[spec(a)][st]["all"]["se"]}
                              for st in ("nits", "live", "mciqa")}
    (V2 / "compare" / "e21_decision.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "score"])
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    ap.add_argument("--monotonicity", action="store_true")
    args = ap.parse_args()
    return {"grid": cmd_grid, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
