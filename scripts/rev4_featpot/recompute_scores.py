"""Recompute completed baseline SROCCs from bank labels and stored predictions.

This audits the result receipts with the canonical panel owner. It reads target
values only through data.load, which pins each role-allowed bank labels file.
"""

import json

import numpy as np

from data import load
from linear_probe import ROOT, panel_batch, sha
from summarize import ARMS, MODELS, SETS


def score(prediction: list[float], target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    if len(pred) != len(target) or not np.isfinite(pred).all():
        raise ValueError("prediction/label mismatch")
    return float(panel_batch([("audit", pred, target)], stats="srocc")[0]["srocc"])


def main() -> None:
    audited = 0
    for name in SETS:
        data, _ = load(name, features=False)
        target = data.target.to_numpy(dtype=np.float64)
        for model in MODELS:
            for arm in ARMS:
                path = ROOT / "fits" / f"POT_{name}_{arm}_{model}" / "result.json"
                if not path.is_file():
                    print(json.dumps({"MISSING": str(path)}), flush=True)
                    continue
                value = json.loads(path.read_text())
                nested = score(value["nested"]["prediction"], target)
                insample = score(value["in_sample"]["prediction"], target)
                if abs(nested - value["nested"]["score"]["srocc"]) > 1e-12:
                    raise ValueError(f"{path}: nested score mismatch")
                if abs(insample - value["in_sample"]["score"]["srocc"]) > 1e-12:
                    raise ValueError(f"{path}: in-sample score mismatch")
                audited += 1
                print(json.dumps({"cell": f"{name}/{model}/{arm}",
                                  "rows": len(target), "nested_srocc": nested,
                                  "in_sample_srocc": insample,
                                  "source_sha256": sha(path)}), flush=True)
    for model in MODELS:
        path = ROOT / "lodo" / f"LODO_r0_{model}" / "result.json"
        if not path.is_file():
            print(json.dumps({"MISSING": str(path)}), flush=True)
            continue
        value = json.loads(path.read_text())
        for heldout, fold in value["folds"].items():
            data, _ = load(fold["eval_set"], features=False)
            target = data.target.to_numpy(dtype=np.float64)
            measured = score(fold["prediction"], target)
            if abs(measured - fold["score"]["srocc"]) > 1e-12:
                raise ValueError(f"{path}/{heldout}: transfer score mismatch")
            print(json.dumps({"lodo": f"{model}/{heldout}", "rows": len(target),
                              "srocc": measured, "source_sha256": sha(path)}), flush=True)
    print(json.dumps({"audited_fit_cells": audited,
                      "label": "POTENTIAL — ceiling, not a model score"}), flush=True)


if __name__ == "__main__":
    main()
