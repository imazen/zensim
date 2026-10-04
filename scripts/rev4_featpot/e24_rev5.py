"""Design log E24 (registered 2026-10-04 12:05 UTC, before any Rev5 cell or table exists): is by_v2fy at formula Rev5 as
good as by_v2fy at Rev4, both at the adopted recipe cv16:cf98?

  python3 e24_rev5.py grid  --root REV5_ROOT --out SPEC.json --program-sha P --data-sha D   # 2 arms x 5 folds x seeds 0-9
  python3 e24_rev5.py score --root REV5_ROOT --rev4-root REV4_ROOT                          # -> REV5_ROOT/compare/e24_rev5.json

Spec: `benchmarks/rev5_spec_2026-10-04.md` (Rev5 = f32 local-window blur, 16-lane canon, FMA, stable central moments, exact
identity zeros; basic + peaks + v2 only). The Rev5 instrument root is built from the Rev5 bank (`rev5_bank.py`,
`v2c_wide.py --revision 5`) with the same keys, labels, legs, coverage pool pairs and seeds as the Rev4 root, so a Rev5 cell and
the Rev4 cell with the same (held-out source, seed) differ only in feature arithmetic.

Arms (Rev5 root): by_v2fy (`sel:59f0bbc2f290`, 420 columns) and v2 + basic, both `@h32:H128:cv16:cf98`, head N, seeds 0-9.
Control: by_v2fy at Rev4 (the Rev4 root's E21 cells, seeds 0-9) — the set the R7 sealed read found as good as v2 + basic.
Primary (by_v2fy@Rev5 vs control), seed-paired over seeds 0-9: E12's "as good" rule (mean signed Δ >= -0.002 and every
source >= -0.005) AND the worst-three-types W2 Δ not worse than -2 SE. Secondary (reported, not gating): v2 + basic@Rev5 vs the
same control; per-source Δ, W1; the external NITS / LIVE / MCIQA Δ; steering panels and CHROMAQ on the Rev5 bakes (separate
records). Exposure: exploratory sources only (as E21/E23). Confirmation of a Rev5 model needs a new holdout (spec §6).
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

import e21_cheap_recipe as e21
from v2_common import SOURCE_ORDER, V2

RECIPE = e21.RECIPE
HEAD, SEEDS = "N", range(10)
ARMS = {"rev5_by_v2fy": "by_v2fy", "rev5_v2basic": None}
CONTROL = e21.spec("by_v2fy")          # sel:59f0bbc2f290@h32:H128:cv16:cf98, read from the Rev4 root
PREFIX = "rev5::"                      # routes an arm's cells to the Rev5 root inside the shared scorer


def arm_spec(label: str) -> str:
    base = ARMS[label]
    return e21.spec(base) if base else f"set:v2+basic@h32{RECIPE}"


def arm_columns(label: str) -> list | None:
    base = ARMS[label]
    return e21.columns(base) if base else None


def cmd_grid(args) -> int:
    cells = []
    for a in ARMS:
        cols = arm_columns(a)
        for s in SOURCE_ORDER:
            for i in SEEDS:
                argv = ["v2_lodo_mlp.py", "--spec", arm_spec(a), "--head", HEAD, "--heldout", s, "--seed-index", str(i),
                        "--root", str(V2)]
                if cols:
                    argv += ["--columns", ",".join(map(str, cols))]
                cells.append({"name": f"{arm_spec(a)}__{HEAD}/without_{s}_s{i}", "argv": argv})
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "specs": {a: arm_spec(a) for a in ARMS}}))
    return 0


def same_heldout_keys(rev4: Path, rev5: Path) -> None:
    """The held-out tables of both roots must be the same keys in the same order, or no pairing is valid."""
    for s in SOURCE_ORDER:
        a = pq.read_table(rev4 / "wide" / "main" / "real" / f"{s}.keys.parquet").to_pandas()
        b = pq.read_table(rev5 / "wide" / "main" / "real" / f"{s}.keys.parquet").to_pandas()
        if not a.equals(b):
            raise ValueError(f"{s}: held-out keys differ between the Rev4 and Rev5 roots")


def cmd_score(args) -> int:
    import e13_teacher as e13
    rev5, rev4 = Path(V2), Path(args.rev4_root)
    same_heldout_keys(rev4, rev5)

    def cell_of(spec_: str, source: str, seed: int) -> Path:
        root, s = (rev5, spec_[len(PREFIX):]) if spec_.startswith(PREFIX) else (rev4, spec_)
        return root / "cells" / f"{s}__{HEAD}" / f"without_{source}_s{seed}" / "result.json"

    e13.cell_of, e13.V2, e13.SEEDS, e13.BASE = cell_of, rev4, SEEDS, CONTROL
    e13.MEAN_FLOOR, e13.SOURCE_FLOOR = e21.MEAN_FLOOR, e21.SOURCE_FLOOR
    rc = e13.score_arms([(a, PREFIX + arm_spec(a)) for a in ARMS], "e24_rev5", False)
    src = rev4 / "compare" / "e24_rev5.json"
    rows = json.loads(src.read_text())
    (rev5 / "compare").mkdir(exist_ok=True)
    (rev5 / "compare" / "e24_rev5.json").write_text(json.dumps({**rows, "rev4_root": str(rev4), "rev5_root": str(rev5)},
                                                                indent=1) + "\n")
    src.unlink()
    decision = {}
    for a, row in rows["rows"].items():
        if "signed" not in row:
            decision[a] = {"status": "INCOMPLETE"}
            continue
        sig, w2 = row["signed"], row["w2_type_worst3"]   # E21's "as good" rule, not E13's adopt flag
        decision[a] = {"spec": arm_spec(a), "signed_mean": sig["mean"], "signed_se": sig["se"], "signed_worst": sig["worst"],
                       "per_source": {s: row["per_source"][s]["signed"] for s in SOURCE_ORDER}, "w2": w2["mean"],
                       "w2_se": w2["se"], "as_good": bool(sig["mean"] >= e21.MEAN_FLOOR and sig["worst"] >= e21.SOURCE_FLOOR
                                                         and w2["mean"] > -2 * w2["se"])}
    (rev5 / "compare" / "e24_decision.json").write_text(json.dumps({"control": CONTROL, "control_root": str(rev4),
                                                                    "arms": decision}, indent=1) + "\n")
    print(json.dumps(decision))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "score"])
    ap.add_argument("--root")
    ap.add_argument("--rev4-root", default="/var/tmp/rev4-featpot/v2c")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    args = ap.parse_args()
    return {"grid": cmd_grid, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
