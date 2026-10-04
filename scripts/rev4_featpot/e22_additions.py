"""Design log E22 (registered 2026-10-03 22:55 MT, before any E22 cell): under the adopted coverage recipe cv16:cf98, does adding
a worst-type group to v2 + basic still add value?

  python3 e22_additions.py grid  --root ROOT --out SPEC.json --program-sha P --data-sha D   # 3 arms x 5 folds x seeds 0-9
  python3 e22_additions.py score --root ROOT                                               # -> <root>/compare/e22_decision.json

Arms (head N, seeds 0-9): v2 + basic + p3 (gmsbank + gmsd/gmsm peers), + iw, + masked, each `set:v2+basic+<g>@h32:H128:cv16:cf98`.
Uncurated-recipe evidence that picked them (seeds 0-4, `v2c/compare/costpot_add1.json`): p3 W2 +0.037 +/- 0.018 (worst source
-0.0015), iw W2 +0.019 +/- 0.016, masked mean +0.0019 (E9 near-miss). Rev4 cost (COSTPOT, 1 MP score): p3 ~6x, iw and masked
~1.2x v2 + basic; none of the three is steerable.
Control: v2 + basic at the same recipe (existing E16/E17 cells, seeds 0-9).
Primary, per arm, seed-paired over seeds 0-9: the arm ADDS VALUE iff signed mean delta >= 0 AND every source >= -0.003 AND
(W2 delta >= +2 SE OR signed mean delta >= +2 SE). Descriptive: per-source delta, W1, external NITS/LIVE/MCIQA on seeds 0-9.
No adoption follows from E22 alone: an added group costs extraction time and steering, so adoption is the user's decision.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from v2_common import SOURCE_ORDER, V2

RECIPE = ":H128:cv16:cf98"
CONTROL = f"set:v2+basic@h32{RECIPE}"
HEAD, SEEDS = "N", range(10)
ARMS = ("p3", "iw", "masked")


def spec(g: str) -> str:
    return f"set:v2+basic+{g}@h32{RECIPE}"


def cmd_grid(args) -> int:
    cells = [{"name": f"{spec(g)}__{HEAD}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", spec(g), "--head", HEAD, "--heldout", s, "--seed-index", str(i),
                       "--root", str(V2)]}
             for g in ARMS for s in SOURCE_ORDER for i in SEEDS]
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo)}))
    return 0


def cmd_score(args) -> int:
    import e13_teacher as e13
    e13.SEEDS, e13.BASE = SEEDS, CONTROL
    rc = e13.score_arms([(g, spec(g)) for g in ARMS], "e22_additions", args.monotonicity)
    rows = json.loads((V2 / "compare" / "e22_additions.json").read_text())["rows"]
    out = {}
    for g in ARMS:
        r = rows[g]
        sig, w2 = r["signed"], r["w2_type_worst3"]
        out[g] = {"spec": spec(g), "signed_mean": sig["mean"], "signed_se": sig["se"], "signed_worst": sig["worst"],
                  "per_source": {s: r["per_source"][s]["signed"] for s in SOURCE_ORDER}, "w2": w2["mean"], "w2_se": w2["se"],
                  "adds_value": bool(sig["mean"] >= 0 and sig["worst"] >= -0.003
                                     and (w2["mean"] >= 2 * w2["se"] or sig["mean"] >= 2 * sig["se"]))}
    subprocess.run([sys.executable, str(Path(__file__).with_name("external_sets.py")), "score", "--root", str(V2),
                    "--specs", ",".join([CONTROL, *(spec(g) for g in ARMS)]), "--seeds", "0-9", "--out", "e22_external"],
                   check=True, capture_output=True)
    ext = json.loads((V2 / "compare" / "e22_external.json").read_text())
    for g in ARMS:
        if spec(g) in ext["specs"]:
            out[g]["external"] = {st: {"delta": ext["specs"][spec(g)][st]["all"]["delta"], "se": ext["specs"][spec(g)][st]["all"]["se"]}
                                  for st in ("nits", "live", "mciqa")}
        else:
            out[g]["external"] = "refused (reads columns the external tables do not carry)"
    (V2 / "compare" / "e22_decision.json").write_text(json.dumps(out, indent=1) + "\n")
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
