"""Prediction-level parity of the MLP trainer across binaries (companion to parity_audit_mlp.py).

Byte-equality of checkpoints failed even for the pre-rule trainer against its own stored
checkpoints, so bit parity is not achievable. This compares, on the SAME stored test table,
predictions of (a) the stored pre-rule checkpoint, (b) a pre-rule-trainer rerun, and
(c) a clean-main-trainer rerun: max |pred diff| between pairs and panel SROCC vs the label.
Same two fixed cells as parity_audit_mlp.py. Run under the heavy wrapper.
POTENTIAL - ceiling, not a model score.
"""

import json
from pathlib import Path

import numpy as np

from data import load
from linear_probe import ROOT, panel_batch, sha
from mlp_probe import predict
from parity_audit_mlp import CELLS

RUNS = {"stored": None, "prerule_rerun": ROOT / "parity/mlp_prerule_control",
        "cleanmain_rerun": ROOT / "parity/mlp"}


def main() -> None:
    out = {"schema": "rev4-featpot-mlp-parity-pred-v1", "cells": []}
    for name, arm, hidden, tag, rep in CELLS:
        src = ROOT / "fits" / f"POT_{name}_{arm}_mlp{hidden}" / f"{tag}_r{rep}"
        res = json.loads((src / "result.json").read_text())
        epoch = res["selected_epoch"]
        test = src / "refit/test.parquet"
        data, _ = load(name)
        y = data.loc[res["test_index"], "target"].to_numpy(dtype=np.float64)
        preds, scores = {}, {}
        for label, base in RUNS.items():
            ck = (src / "refit" if base is None else base / f"{name}_{arm}_h{hidden}_{tag}_r{rep}") \
                / f"ckpt_epoch{epoch:03}.bin"
            dest = ROOT / "parity/pred" / f"{name}_{arm}_h{hidden}_{tag}_r{rep}_{label}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            preds[label] = predict(ck, test, dest.with_suffix(".tsv"))
            scores[label] = panel_batch([(label, preds[label], y)], stats="full")[0]
        diffs = {f"{a}_vs_{b}": float(np.max(np.abs(preds[a] - preds[b])))
                 for a, b in (("stored", "prerule_rerun"), ("stored", "cleanmain_rerun"),
                              ("prerule_rerun", "cleanmain_rerun"))}
        out["cells"].append({"cell": str(src), "epoch": epoch, "n_test": len(y),
                             "max_abs_pred_diff": diffs, "scores": scores})
        print(json.dumps({"cell": src.name, "max_abs_pred_diff": diffs}), flush=True)
    dest = ROOT / "parity/parity_pred.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"report": str(dest), "sha256": sha(dest)}), flush=True)


if __name__ == "__main__":
    main()
