"""Seven-by-seven R0 LODO cross-evaluation matrix for BVLS or lasso.

Rows are the six-source fitted fold models; columns are the D2 evaluation
views. The receipt distinguishes source membership from evaluation rows seen
in fitting. Scores and bootstrap inputs go through the panel owner.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data import FEATURES, load
from linear_probe import ROOT, panel_batch, predict_path, sha
from lodo_bvls import EVAL, SOURCES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["bvls", "linear"], required=True)
    args = ap.parse_args()
    root = ROOT / "lodo" / f"LODO_r0_{args.model}"
    source = root / "result.json"
    result = json.loads(source.read_text())
    if set(result["folds"]) != set(SOURCES):
        raise ValueError("D2 fold set changed")
    views = {}
    for name in SOURCES:
        eval_set = EVAL.get(name, name)
        data, label_meta = load(eval_set)
        views[name] = {"eval_set": eval_set, "label_source": label_meta,
                       "x": data[FEATURES].to_numpy(dtype=np.float64, copy=False),
                       "y": data.target.to_numpy(dtype=np.float64),
                       "refs": data.ref_basename.astype(str).to_numpy()}
    matrix = {"schema": "rev4-featpot-lodo-cross-matrix-v1",
              "label": "POTENTIAL — ceiling, not a model score",
              "model": args.model, "source_result_sha256": sha(source),
              "B": 2000, "seed": 20260923, "unit": "reference",
              "rows": {}}
    for heldout in SOURCES:
        fold = result["folds"][heldout]
        if args.model == "bvls":
            fit_path = root / f"fit_without_{heldout}.npz"
            if sha(fit_path) != fold["fit_sha256"]:
                raise ValueError(f"{heldout}: BVLS fit hash changed")
            with np.load(fit_path) as fit:
                mu = fit["mu"].copy()
                sd = fit["sd"].copy()
                w = fit["w"].copy()
                bias = float(fit["bias"].item())

            def predict(x: np.ndarray) -> np.ndarray:
                return (x - mu) / sd @ w + bias

        else:
            fit_path = root / f"without_{heldout}_fit.json"
            if sha(fit_path) != fold["path_sha256"]:
                raise ValueError(f"{heldout}: lasso path hash changed")
            path = json.loads(fit_path.read_text())
            selected = fold["selection"]["selected_index"]

            def predict(x: np.ndarray) -> np.ndarray:
                return predict_path(path, x)[:, selected]

        matrix["rows"][heldout] = {"fit_source_sets": fold["source_sets"],
                                   "fit_path": str(fit_path), "fit_sha256": sha(fit_path),
                                   "columns": {}}
        for source_set in SOURCES:
            view = views[source_set]
            y = view["y"]
            pred = predict(view["x"])
            point = panel_batch([(source_set, pred, y)], stats="srocc")[0]
            refs = view["refs"]
            groups = [np.flatnonzero(refs == ref) for ref in sorted(set(refs))]
            rng = np.random.default_rng(20260923)
            jobs = [("POINT", "p", "y", None)]
            for b in range(2000):
                draw = rng.integers(0, len(groups), len(groups))
                jobs.append((f"B{b}", "p", "y",
                             np.concatenate([groups[i] for i in draw])))
            panel_rows = panel_batch_indexed({"p": pred, "y": y}, jobs,
                                             stats="srocc", timeout=7200)
            by = {row["label"]: row["srocc_signed"] for row in panel_rows}
            boot = np.asarray([by[f"B{b}"] for b in range(2000)], dtype=np.float64)
            finite = boot[np.isfinite(boot)]
            if len(finite) < 1900:
                raise ValueError(f"{heldout}/{source_set}: only {len(finite)} finite bootstraps")
            signed = float(by["POINT"])
            if abs(signed - point["srocc_signed"]) > 1e-12:
                raise ValueError(f"{heldout}/{source_set}: owner score mismatch")
            if source_set == heldout and abs(signed - fold["score"]["srocc_signed"]) > 1e-12:
                raise ValueError(f"{heldout}: diagonal differs from held-out receipt")
            column = {"eval_set": view["eval_set"],
                      "label_source": view["label_source"],
                      "source_in_fit": source_set in fold["source_sets"],
                      "eval_rows_seen_in_fit": (source_set in fold["source_sets"]
                                                and view["eval_set"] == source_set),
                      "rows": len(y), "references": len(groups),
                      "srocc_signed": signed,
                      "ci95_srocc_signed": np.quantile(finite, [0.025, 0.975]).tolist(),
                      "bootstrap_finite": len(finite)}
            matrix["rows"][heldout]["columns"][source_set] = column
            print(json.dumps({"heldout": heldout, "column": source_set,
                              "rows": len(y), "srocc_signed": signed,
                              "ci95": column["ci95_srocc_signed"]}), flush=True)
    output = root / "transfer_matrix.json"
    output.write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output),
                      "cells": len(SOURCES) ** 2}), flush=True)


if __name__ == "__main__":
    main()
