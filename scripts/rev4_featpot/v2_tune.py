#!/usr/bin/env python3
"""Instrument v2 retune (amendment R3): human-leg weight sweep and the registered selection rule.

Reads the sweep cells (`<spec>@h<w>__<head>/without_<fold>_s<seed>/result.json`) and writes
`/var/tmp/rev4-featpot/v2/compare/tune.json`. The rule is applied to head N only; head F and oracle_lo are reported.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from v2_common import V2

WEIGHTS = (0.5, 2.0, 8.0, 32.0)
FOLDS = ("kadid", "konfig", "aic3")
SEEDS = (0, 1, 2)
SPECS = ("r0", "oracle_hi", "oracle_hi~p1", "oracle_lo")
HEADS = ("N", "F")
ACCURACY_SLACK = 0.01
CENTRING_LIMIT = 0.01
TIE = 0.002
MIN_DETECTION = 0.02


def tag(w: float) -> str:
    return f"{w:g}"


def srocc(spec: str, w: float, head: str, fold: str, seed: int) -> float | None:
    res = V2 / "cells" / f"{spec}@h{tag(w)}__{head}" / f"without_{fold}_s{seed}" / "result.json"
    if not res.is_file():
        return None
    cell = json.loads(res.read_text())
    if cell.get("human_nominal_weight") != w:
        raise ValueError(f"{res}: human_nominal_weight {cell.get('human_nominal_weight')} != {w}")
    return float(cell["score"]["srocc"])


def summarise(head: str) -> tuple[dict, list[str]]:
    rows, missing = {}, []
    for w in WEIGHTS:
        fold_means = {}
        for spec in SPECS:
            per_fold = []
            for fold in FOLDS:
                vals = [srocc(spec, w, head, fold, s) for s in SEEDS]
                missing += [f"{spec}@h{tag(w)}__{head}/without_{fold}_s{s}" for s, v in zip(SEEDS, vals) if v is None]
                per_fold.append(None if None in vals else float(np.mean(vals)))
            fold_means[spec] = per_fold
        if any(None in v for v in fold_means.values()):
            continue
        f = {k: np.asarray(v) for k, v in fold_means.items()}
        rows[tag(w)] = {
            "E": float(np.mean(f["oracle_hi"] - f["oracle_hi~p1"])),
            "A": float(np.mean(f["r0"])),
            "C": float(np.mean(f["oracle_hi~p1"] - f["r0"])),
            "E_lo": float(np.mean(f["oracle_lo"] - f["r0"])),
            "per_fold": {k: dict(zip(FOLDS, v)) for k, v in fold_means.items()},
        }
    return rows, missing


def select(rows: dict) -> dict:
    if len(rows) != len(WEIGHTS):
        return {"status": "INCOMPLETE"}
    best_a = max(r["A"] for r in rows.values())
    admissible = [w for w in WEIGHTS if rows[tag(w)]["A"] >= best_a - ACCURACY_SLACK
                  and abs(rows[tag(w)]["C"]) <= CENTRING_LIMIT]
    if not admissible:
        return {"status": "NO_RECIPE", "reason": "no admissible weight", "best_A": best_a}
    best_e = max(rows[tag(w)]["E"] for w in admissible)
    chosen = min(w for w in admissible if rows[tag(w)]["E"] >= best_e - TIE)
    if best_e < MIN_DETECTION:
        return {"status": "NO_RECIPE", "reason": f"best E {best_e:.4f} < {MIN_DETECTION}",
                "admissible": admissible, "best_A": best_a}
    return {"status": "SELECTED", "human_nominal_weight": chosen, "admissible": admissible,
            "best_A": best_a, "best_E": best_e}


def main() -> int:
    out = {"rule": "amendment R3", "weights": WEIGHTS, "folds": FOLDS, "seeds": SEEDS}
    for head in HEADS:
        rows, missing = summarise(head)
        out[head] = {"by_weight": rows, "missing": missing}
    out["selection_head_N"] = select(out["N"]["by_weight"])
    dest = V2 / "compare" / "tune.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    for head in HEADS:
        for w, r in out[head]["by_weight"].items():
            print(f"{head} w={w:>4}  E={r['E']:+.4f}  A={r['A']:.4f}  C={r['C']:+.4f}  E_lo={r['E_lo']:+.4f}")
        print(f"{head} missing {len(out[head]['missing'])}")
    print(json.dumps(out["selection_head_N"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
