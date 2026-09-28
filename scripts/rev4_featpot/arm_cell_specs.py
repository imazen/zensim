"""Enumerate the MLP cells of the candidate and restore arms for the fleet AVX2 era.

POTENTIAL - ceiling, not a model score. Writes a JSONL cell list (one cell per line:
argv, result path, weight H32=1 / H128=4, suggested priority) plus a small summary
record. Enumeration only: it reads no data and runs nothing. Grid shape follows the
P0 grid: D1 = 8 sets x {H32,H128} x {outer 0-4, full} x 5 reps per arm variant
(real and `_perm`); D2 = 7 held-out sets x {H32,H128} x 5 reps per arm variant.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path("/var/tmp/rev4-featpot")
SPEC = ROOT / "fleet_specs/arm_cells_2026-09-25.jsonl"
SUMMARY = Path("benchmarks/rev4_featpot_arm_cells_2026-09-25.json")
D1_SETS = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
           "cid22_a25", "aic3", "kadid_select", "konfig_val")
D2_HELDOUT = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
              "cid22_a25", "aic3", "kadid_select")
# Suggested execution order (a planning suggestion, not part of the preregistration).
ARM_ORDER = ("p1", "p3", "c7", "csfw", "a1", "a1w", "b2", "a1m", "b2m", "c1", "c2", "c3", "c4",
             "all", "b1", "b1s", "c8n", "rall")


def cell(kind: str, argv: list[str], result: str, arm: str, hidden: int, extra: dict) -> dict:
    line = " ".join(argv)
    return {"cell_id": hashlib.sha256(line.encode()).hexdigest()[:16], "kind": kind,
            "arm": arm, "hidden": hidden, "weight": 1 if hidden == 32 else 4,
            "argv": argv, "result": result, **extra}


def main() -> None:
    cells = []
    for rank, base in enumerate(ARM_ORDER):
        for variant in (base, f"{base}_perm"):
            for hidden in (32, 128):
                for name in D1_SETS:
                    for outer in (None, 0, 1, 2, 3, 4):
                        for rep in range(5):
                            tag = "full" if outer is None else f"o{outer}"
                            argv = ["python", "scripts/rev4_featpot/candidate_mlp.py", "--set", name,
                                    "--arm", variant, "--hidden", str(hidden), "--rep", str(rep)]
                            if outer is not None:
                                argv += ["--outer", str(outer)]
                            res = (f"candidates/mlp/POT_{name}_{variant}_mlp{hidden}/{tag}_r{rep}/result.json")
                            cells.append(cell("d1_mlp", argv, res, variant, hidden,
                                              {"set": name, "view": tag, "rep": rep, "priority": rank}))
                for held in D2_HELDOUT:
                    for rep in range(5):
                        argv = ["python", "scripts/rev4_featpot/candidate_lodo_mlp.py", "--arm", variant,
                                "--hidden", str(hidden), "--heldout", held, "--rep", str(rep)]
                        res = f"candidates/d2_mlp/LODO_{variant}_mlp{hidden}/without_{held}_r{rep}/result.json"
                        cells.append(cell("d2_mlp", argv, res, variant, hidden,
                                          {"heldout": held, "rep": rep, "priority": rank}))
    ids = [c["cell_id"] for c in cells]
    assert len(ids) == len(set(ids))
    SPEC.parent.mkdir(parents=True, exist_ok=True)
    SPEC.write_text("".join(json.dumps(c) + "\n" for c in cells))
    digest = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    per_arm = {}
    for c in cells:
        base = c["arm"].removesuffix("_perm")
        row = per_arm.setdefault(base, {"d1_cells": 0, "d2_cells": 0, "weight": 0})
        row["d1_cells" if c["kind"] == "d1_mlp" else "d2_cells"] += 1
        row["weight"] += c["weight"]
    summary = {"schema": "rev4-featpot-arm-cells-v1", "label": "POTENTIAL — ceiling, not a model score",
               "spec_file": str(SPEC), "spec_sha256": digest, "cells": len(cells),
               "d1_cells": sum(c["kind"] == "d1_mlp" for c in cells),
               "d2_cells": sum(c["kind"] == "d2_mlp" for c in cells),
               "total_weight_H32_1_H128_4": sum(c["weight"] for c in cells),
               "arm_order_suggestion": list(ARM_ORDER), "per_arm": per_arm,
               "note": "enumeration only; no cell has been run; smoke first"}
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("spec_sha256", "cells", "d1_cells", "d2_cells",
                                              "total_weight_H32_1_H128_4")}))


if __name__ == "__main__":
    main()
