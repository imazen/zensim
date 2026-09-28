"""Seven-fold R0 lasso LODO POTENTIAL transfer with source-level inner CV.

For each D2 held-out source, the other six contribute equal total weight.
The 50-lambda path's index is chosen by leaving each of those six sources
out in turn, using the preregistered one-SE rule. Source Grammars are the
separately normalized, hash-checked R0 BVLS LODO inputs; no test target enters
any fit or lambda choice. Run through the shared heavy wrapper.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from data import FEATURES, load
from linear_probe import BIN, ROOT, panel_batch, predict_path, run, sha
from lodo_bvls import EVAL, SOURCES
from stability_lasso import FAMILIES


def path_fit(sources: list[str], source_grams: dict, output: Path) -> dict:
    cmd = [str(BIN), "fit-lasso", "--space", "raw", "--target", "target__mm01",
           "--solver", "lasso", "--lam", "0", "--path-out", str(output)]
    for name in sources:
        cmd += ["--gram", source_grams[name]["path"]]
    for name in sources:
        cmd += ["--weight", repr(1.0 / source_grams[name]["rows"])]
    cmd += ["--out", str(output.with_suffix(".unused.bin"))]
    run(cmd, output.with_suffix(".log"))
    result = json.loads(output.read_text())
    if result["schema"] != "rev4-featpot-lasso-path-v1" or len(result["fits"]) != 50:
        raise ValueError(f"{output}: not the registered 50-lambda path")
    return result


def choose_1se(score_rows: list[list[float]]) -> tuple[int, dict]:
    values = np.asarray(score_rows, dtype=np.float64)
    if values.shape != (6, 50) or not np.isfinite(values).all():
        raise ValueError(f"invalid six-source score matrix {values.shape}")
    means = values.mean(axis=0)
    best = int(np.argmax(means))
    se = values[:, best].std(ddof=1) / np.sqrt(6)
    selected = int(np.flatnonzero(means >= means[best] - se)[0])
    return selected, {"best_index": best, "selected_index": selected,
                      "best_mean": float(means[best]), "one_se": float(se),
                      "mean_source_srocc_by_index": means.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["r0"], required=True)
    args = ap.parse_args()
    base = ROOT / "lodo" / "LODO_r0_bvls" / "result.json"
    bvls = json.loads(base.read_text())
    source_grams = bvls["source_grams"]
    if set(source_grams) != set(SOURCES):
        raise ValueError("BVLS source Gram set differs from D2 ruling")
    for name, record in source_grams.items():
        if sha(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"{name}: source Gram hash changed")
    dest = ROOT / "lodo" / "LODO_r0_linear"
    dest.mkdir(parents=True, exist_ok=True)
    result = {"schema": "rev4-featpot-lodo-lasso-v1", "arm": args.arm,
              "label": "POTENTIAL — ceiling, not a model score",
              "source_gram_manifest_sha256": sha(base), "folds": {}}
    for heldout in SOURCES:
        included = [name for name in SOURCES if name != heldout]
        scores = []
        inner_receipts = []
        for validation in included:
            inner_fit = [name for name in included if name != validation]
            path_file = dest / f"without_{heldout}_inner_{validation}.json"
            path = path_fit(inner_fit, source_grams, path_file)
            data, label_meta = load(validation)
            x = data[FEATURES].to_numpy(dtype=np.float64, copy=False)
            y = data.target.to_numpy(dtype=np.float64)
            preds = predict_path(path, x)
            rows = panel_batch([(f"lambda_{i}", preds[:, i], y) for i in range(50)],
                               stats="srocc")
            scores.append([float(row["srocc"]) for row in rows])
            inner_receipts.append({"validation_source": validation,
                                   "validation_rows": len(y), "label_source": label_meta,
                                   "fit_sources": inner_fit, "path_sha256": sha(path_file)})
        selected, selection = choose_1se(scores)
        path_file = dest / f"without_{heldout}_fit.json"
        path = path_fit(included, source_grams, path_file)
        eval_name = EVAL.get(heldout, heldout)
        test, label_meta = load(eval_name)
        x = test[FEATURES].to_numpy(dtype=np.float64, copy=False)
        y = test.target.to_numpy(dtype=np.float64)
        pred = predict_path(path, x)[:, selected]
        score = panel_batch([(heldout, pred, y)], stats="full")[0]
        weights = np.asarray(path["fits"][selected]["w"], dtype=np.float64)
        selected_ids = np.flatnonzero(np.abs(weights) > 1e-10).tolist()
        result["folds"][heldout] = {"eval_set": eval_name,
                                    "eval_target": label_meta["target"],
                                    "label_source": label_meta,
                                    "rows": len(y),
                                    "references": test.ref_basename.nunique(),
                                    "source_sets": included,
                                    "inner": inner_receipts,
                                    "selection": selection,
                                    "selected_lambda": path["fits"][selected]["lambda"],
                                    "selected_feature_ids": selected_ids,
                                    "score": score, "prediction": pred.tolist(),
                                    "path_sha256": sha(path_file)}
        print(json.dumps({"heldout": heldout, "eval": eval_name,
                          "rows": len(y), "references": test.ref_basename.nunique(),
                          "selected_index": selected, "srocc": score["srocc"]}), flush=True)
    result["family_fold_frequency"] = {
        name: sum(any(feature in family for feature in fold["selected_feature_ids"])
                  for fold in result["folds"].values()) / len(SOURCES)
        for name, family in FAMILIES.items()
    }
    output = dest / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(output), "sha256": sha(output),
                      "family_fold_frequency": result["family_fold_frequency"]}), flush=True)


if __name__ == "__main__":
    main()
