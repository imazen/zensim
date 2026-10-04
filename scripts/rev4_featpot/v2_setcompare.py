"""Amendment R7 (2026-10-04): the set-compare confirmatory read of the two adopted candidates.

  python3 v2_setcompare.py grid --root ROOT --out SPEC.json --program-sha P --data-sha D   # 4 entries x head N x seeds 0-9

Entries (head N, full-data `v2_confirm_fit` cells under <root>/confirm/cells; R7 in benchmarks/rev4_featpot_v2_amendment_2026-09-30.md):
A = v2 + basic at cv16:cf98, B = by_v2fy at cv16:cf98, C = v2 + basic uncurated, D = R0 at H128. The fits read no label.
The read itself is `v2_confirm_read.py --set-compare` against the R7a pin (R7 plus the KonFiG/MCL-JCI clean primary).
"""

import argparse
import json
import sys
from pathlib import Path

from v2_common import V2, selection_id

REPO = Path(__file__).resolve().parents[2]
HEAD, SEEDS = "N", range(10)


def by_v2fy_columns() -> list:
    d = json.loads((REPO / "benchmarks/costset2_2026-10-03.candidate_ids.json").read_text())
    return d.get("candidates", d)["by_v2fy"]


def entries() -> dict:
    cols = by_v2fy_columns()
    return {"A": ("set:v2+basic@h32:H128:cv16:cf98", None),
            "B": (f"sel:{selection_id(cols)}@h32:H128:cv16:cf98", cols),
            "C": ("set:v2+basic@h32:H128", None),
            "D": ("r0@h32:H128", None)}


def cmd_grid(args) -> int:
    cells = []
    for label, (spec, cols) in entries().items():
        for i in SEEDS:
            argv = ["v2_confirm_fit.py", "--spec", spec, "--head", HEAD, "--seed-index", str(i), "--root", str(V2)]
            if cols:
                argv += ["--columns", ",".join(map(str, cols))]
            cells.append({"name": f"{spec}__{HEAD}/full_s{i}", "argv": argv})
    todo = [c for c in cells if not (V2 / "confirm" / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "entries": {k: v[0] for k, v in entries().items()}}))
    return 0


PROGRAM_V24 = "dfffd0912699541e3dc0735d2f97393e23f98bc3f3f2f29a3f259494a6eb447b"
CONFIRM_DATA = "cf9b83179376c78471905c8e97e3280b3d45ea008123f9febdbd660f9046f99d"
OLD_PIN = REPO / "benchmarks/rev4_featpot_effaudit/v2c_confirm_pin_2026-10-01.json"
REGISTRATION = REPO / "benchmarks/rev4_featpot_R7_registration_2026-10-04.md"
AMENDMENT_R7A = REPO / "benchmarks/rev4_featpot_R7a_registration_2026-10-04.md"  # KonFiG/MCL-JCI contamination guard
BIN_DIR = Path("/var/tmp/fitv2/bin-v7")  # program v21-v24 binaries (trainer 605d20e0, predictor 56da0529, panel f76b85a7)


def cmd_pin(args) -> int:
    """The R7 pin: frozen root, binaries, program/data, code hashes, the label specs of pin 1223712b byte-for-byte, the entries
    and comparisons of R7, and the registration copy's sha256. Label files are hashed by preflight, never parsed here."""
    import v2_confirm_read as cr
    from v2_common import load_frozen, sha
    frozen, frozen_sha = load_frozen(V2)
    old = json.loads(OLD_PIN.read_text())
    ents = {k: v[0] for k, v in entries().items()}
    pin = {"schema": cr.SC_PIN_SCHEMA, "entries": ents, "head": HEAD,
           "superiority": [{"arm": "A", "reference": "C", "question": "Q1: the adopted coverage recipe out of sample"},
                           {"arm": "A", "reference": "D", "question": "Q2: the selected set vs the R0 bank"}],
           "noninferiority": [{"arm": "B", "reference": "A", "question": "Q3: by_v2fy as an equal-quality substitute"}],
           "ensemble_pairs": [{"arm": "B", "reference": "A"}, {"arm": "A", "reference": "C"}],
           "frozen_sha256": frozen_sha, "wide_receipts": frozen["wide_receipts"],
           "confirm_receipt_sha256": frozen["confirm_receipt_sha256"], "keep_lists_sha256": frozen["keep_lists_sha256"],
           "binaries": {n: sha(BIN_DIR / n) for n in ("zensim_mlp_train", "bake_dial_refit", "panel")},
           "program_sha": PROGRAM_V24, "data_sha": CONFIRM_DATA,
           "code": {rel: sha(path) for rel, path in cr.CODE_FILES.items()},
           "labels": old["labels"], "pixel_hashes": old["pixel_hashes"],
           "labels_from_pin": {"path": str(OLD_PIN), "sha256": sha(OLD_PIN)},
           "registration": {"path": str(REGISTRATION), "sha256": sha(REGISTRATION)},
           "amendment_r7a": {"path": str(AMENDMENT_R7A), "sha256": sha(AMENDMENT_R7A)}}
    Path(args.out).write_text(json.dumps(pin, indent=1) + "\n")
    print(json.dumps({"pin": args.out, "frozen": frozen_sha[:12], "binaries": {k: v[:8] for k, v in pin["binaries"].items()}}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "pin"])
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    args = ap.parse_args()
    return {"grid": cmd_grid, "pin": cmd_pin}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
