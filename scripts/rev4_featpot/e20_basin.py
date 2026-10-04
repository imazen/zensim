"""Design log E20 (registered 2026-10-03 18:25 MT, before any E20 cell): does the E10 basin [append, masked] hold its tie with
v2 + basic on fresh seeds?

  python3 e20_basin.py grid  --root ROOT --out SPEC.json --program-sha P --data-sha D   # set:append+masked, seeds 5-9
  python3 e20_basin.py score --root ROOT                                               # -> <root>/compare/e20_decision.json

Primary: E12's unchanged "as good" rule on seeds 5-9 against v2 + basic (mean signed delta >= -0.002 and every source >= -0.005).
Registered descriptive read (reported, not gating): W2 delta; per-source deltas pooled over seeds 0-9; external NITS / LIVE /
MCIQA deltas on seeds 5-9. No adoption follows from E20 alone.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from v2_common import SOURCE_ORDER, V2

ARM = "set:append+masked@h32:H128"
HEAD = "N"
NEW_SEEDS, ALL_SEEDS = range(5, 10), range(10)
MEAN_FLOOR, SOURCE_FLOOR = -0.002, -0.005  # E12's registered "as good" rule


def cmd_grid(args) -> int:
    cells = [{"name": f"{ARM}__{HEAD}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", ARM, "--head", HEAD, "--heldout", s, "--seed-index", str(i),
                       "--root", str(V2)]}
             for s in SOURCE_ORDER for i in NEW_SEEDS]
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo)}))
    return 0


def cmd_score(args) -> int:
    import e13_teacher as e13
    out = {}
    rc = 0
    for name, seeds in (("e20_seeds59", NEW_SEEDS), ("e20_seeds09", ALL_SEEDS)):
        e13.SEEDS = seeds
        rc |= e13.score_arms([("append_masked", ARM)], name, args.monotonicity)
        row = json.loads((V2 / "compare" / f"{name}.json").read_text())["rows"]["append_masked"]
        out[name] = {"signed_mean": row["signed"]["mean"], "signed_se": row["signed"]["se"],
                     "signed_worst": row["signed"]["worst"],
                     "per_source": {s: row["per_source"][s]["signed"] for s in SOURCE_ORDER},
                     "w2": row["w2_type_worst3"]["mean"], "w2_se": row["w2_type_worst3"]["se"]}
    p = out["e20_seeds59"]
    out["primary_as_good"] = bool(p["signed_mean"] >= MEAN_FLOOR and p["signed_worst"] >= SOURCE_FLOOR)
    subprocess.run([sys.executable, str(Path(__file__).with_name("external_sets.py")), "score", "--root", str(V2),
                    "--specs", f"set:v2+basic@h32:H128,{ARM}", "--seeds", "5-9", "--out", "e20_external"],
                   check=True, capture_output=True)
    ext = json.loads((V2 / "compare" / "e20_external.json").read_text())["specs"][ARM]
    out["external_seeds59"] = {st: {"delta": ext[st]["all"]["delta"], "se": ext[st]["all"]["se"]}
                               for st in ("nits", "live", "mciqa")}
    (V2 / "compare" / "e20_decision.json").write_text(json.dumps(out, indent=1) + "\n")
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
