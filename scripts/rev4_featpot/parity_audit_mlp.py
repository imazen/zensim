"""Bit-parity audit: pre-rule vs clean-main `zensim_mlp_train` on stored fit tables.

Re-trains the outer-fit (refit) run of already-completed P0 MLP cells with the
current-main trainer into a scratch dir and compares every dumped checkpoint
byte for byte against the stored pre-rule checkpoints. Cells are fixed by the
list below (registered before running; never chosen by outcome). It reads only
tables that the original cells already read; it writes nothing to fits/.

Run under the shared heavy wrapper. POTENTIAL - ceiling, not a model score.
"""

import argparse
import json
import sys
from pathlib import Path

import mlp_probe
from linear_probe import ROOT, sha
from mlp_probe import train

# Fixed audit cells: (set, arm, hidden, tag, rep). One H32 outer fold, one
# H128 outer fold, and one full in-sample refit if present.
CELLS = [
    ("kadid_train", "r0", 32, "o0", 0),
    ("konfig_val", "minus_basic", 128, "o1", 4),
]
SCRATCH = ROOT / "parity/mlp"
PRE_RULE_TRAINER = ROOT / "main_build_20260924/pre_current_main_binaries/zensim_mlp_train"


def main() -> None:
    global SCRATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-pre-rule-trainer", action="store_true",
                    help="repeat with the pre-rule trainer: run-to-run determinism control")
    args = ap.parse_args()
    if args.control_pre_rule_trainer:
        mlp_probe.TRAINER = PRE_RULE_TRAINER
        SCRATCH = ROOT / "parity/mlp_prerule_control"
    TRAINER = mlp_probe.TRAINER
    report = {"schema": "rev4-featpot-mlp-parity-v1", "trainer": str(TRAINER.resolve()),
              "trainer_sha256": sha(TRAINER.resolve()), "cells": []}
    ok = True
    for name, arm, hidden, tag, rep in CELLS:
        src = ROOT / "fits" / f"POT_{name}_{arm}_mlp{hidden}" / f"{tag}_r{rep}"
        result = json.loads((src / "result.json").read_text())
        dest = SCRATCH / f"{name}_{arm}_h{hidden}_{tag}_r{rep}"
        slice_file = src / "keep_f228_to_f943.txt"
        train(src / "refit" / "fit.parquet", None, dest, hidden, arm,
              result["init_seed"], result["sample_seed"], slice_file, dump=True)
        rows = []
        for old in sorted((src / "refit").glob("ckpt_epoch*.bin")):
            new = dest / old.name
            same = new.is_file() and sha(new) == sha(old)
            rows.append({"file": old.name, "old_sha256": sha(old),
                         "new_sha256": sha(new) if new.is_file() else None, "identical": same})
        cell_ok = bool(rows) and all(r["identical"] for r in rows)
        ok = ok and cell_ok
        report["cells"].append({"cell": str(src), "selected_epoch": result["selected_epoch"],
                                "checkpoints": rows, "all_identical": cell_ok})
        print(json.dumps({"cell": src.name, "checkpoints": len(rows), "all_identical": cell_ok}),
              flush=True)
    report["all_identical"] = ok
    out = SCRATCH / "parity_mlp.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(out), "sha256": sha(out), "all_identical": ok}), flush=True)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
