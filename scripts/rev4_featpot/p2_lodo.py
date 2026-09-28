"""Registered seven-fold D2 transfer for the reviewed P2 peer arm and control."""

import argparse
import json
import numpy as np

from linear_probe import BIN, REPO, ROOT, panel_batch, predict_path, run
from lodo_bvls import EVAL, SOURCES
from lodo_lasso import choose_1se
from p2_data import load, sha


COLS = [f"f{i}" for i in range(946)]


def matrix(name, arm):
    data, meta = load(name, arm)
    values = data.rename(columns={"gmsd": "f944", "gmsm": "f945"})
    return values[COLS].to_numpy(dtype=np.float64), values.target.to_numpy(dtype=np.float64), values, meta


def audit_source_gram(file, expected_rows, expected_refs):
    raw = file.with_name("full_raw.npz")
    refs = sorted(file.with_suffix(".refs").glob("ref_*.npz"))
    if len(refs) != expected_refs:
        raise ValueError(f"{file}: {len(refs)} reference grams, expected {expected_refs}")
    with np.load(raw) as source:
        whole = {key: source[key].copy() for key in source.files}
    if whole["raw__n"].item() != expected_rows:
        raise ValueError(f"{file}: source Gram row count mismatch")
    summation = {key: np.zeros_like(value) for key, value in whole.items()}
    for reference in refs:
        with np.load(reference) as part:
            if set(part.files) != set(whole):
                raise ValueError(f"{reference}: raw Gram schema mismatch")
            for key in whole:
                summation[key] += part[key]
    errors = {}
    for key in whole:
        diff = np.abs(summation[key] - whole[key])
        errors[key] = float(np.max(diff))
        if not np.allclose(summation[key], whole[key], rtol=1e-10, atol=1e-9):
            raise ValueError(f"{file}: per-reference sum differs from raw whole Gram at {key}")
    return {"reference_grams": len(refs), "raw_sha256": sha(raw),
            "maximum_absolute_error": errors}


def path_fit(sources, grams, path):
    cmd = [str(BIN), "fit-lasso", "--space", "raw", "--target", "target__mm01",
           "--solver", "lasso", "--lam", "0", "--path-out", str(path)]
    for name in sources:
        cmd += ["--gram", str(grams[name]["path"])]
    for name in sources:
        cmd += ["--weight", repr(1.0 / grams[name]["rows"])]
    cmd += ["--out", str(path.with_suffix(".unused.bin"))]
    run(cmd, path.with_suffix(".log"))
    value = json.loads(path.read_text())
    if value["schema"] != "rev4-featpot-lasso-path-v1" or len(value["fits"]) != 50:
        raise ValueError("LODO P2 lasso path is incomplete")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["p2", "p2_perm"], required=True)
    parser.add_argument("--model", choices=["linear", "bvls"], required=True)
    args = parser.parse_args()
    root = ROOT / "p2" / "d2" / f"LODO_{args.arm}_{args.model}"
    root.mkdir(parents=True, exist_ok=True)
    grams = {}
    audits = {}
    for name in SOURCES:
        file = ROOT / "p2" / "d1" / f"POT_{name}_{args.arm}_bvls" / "full.npz"
        if not file.is_file():
            raise FileNotFoundError(f"run P2 D1 BVLS full-source Gram first: {file}")
        _, y, data, _ = matrix(name, args.arm)
        grams[name] = {"path": str(file), "sha256": sha(file), "rows": len(y)}
        audits[name] = audit_source_gram(file, len(y), data.ref_basename.nunique())
    result = {"schema": "rev4-featpot-p2-lodo-v1", "arm": args.arm,
              "model": args.model, "label": "POTENTIAL — ceiling, not a model score",
              "source_grams": grams, "source_gram_audits": audits, "folds": {}}
    for heldout in SOURCES:
        included = [name for name in SOURCES if name != heldout]
        selection = None
        if args.model == "linear":
            scores = []
            for validation in included:
                fit_names = [name for name in included if name != validation]
                path_file = root / f"without_{heldout}_inner_{validation}.json"
                path = path_fit(fit_names, grams, path_file)
                x, y, _, _ = matrix(validation, args.arm)
                preds = predict_path(path, x)
                rows = panel_batch([(f"lambda_{i}", preds[:, i], y)
                                    for i in range(50)], stats="srocc")
                scores.append([row["srocc"] for row in rows])
            selected, selection = choose_1se(scores)
            path_file = root / f"fit_without_{heldout}.json"
            path = path_fit(included, grams, path_file)
            weights = np.asarray(path["fits"][selected]["w"], dtype=np.float64)
            fit_sha = sha(path_file)
        else:
            cmd = [str(BIN), "fit-lasso", "--space", "raw", "--target", "target__mm01",
                   "--solver", "bvls", "--lam", "0", "--bounds-tsv",
                   str(REPO / "benchmarks/feature_sign_mask_2026-05-26.tsv")]
            for name in included:
                cmd += ["--gram", grams[name]["path"]]
            for name in included:
                cmd += ["--weight", repr(1.0 / grams[name]["rows"])]
            fit_file = root / f"fit_without_{heldout}.npz"
            cmd += ["--emit-fit-npz", str(fit_file), "--diagnostic-fit-only",
                    "--out", str(fit_file.with_suffix(".unused.bin"))]
            run(cmd, fit_file.with_suffix(".log"))
            with np.load(fit_file) as fit:
                values = {key: fit[key].copy() for key in ("w", "mu", "sd", "bias")}
            weights = values["w"]
            fit_sha = sha(fit_file)
        eval_name = EVAL.get(heldout, heldout)
        x, y, data, label_meta = matrix(eval_name, args.arm)
        if args.model == "linear":
            pred = predict_path(path, x)[:, selected]
        else:
            pred = (x - values["mu"]) / values["sd"] @ values["w"] + values["bias"].item()
        score = panel_batch([(heldout, pred, y)], stats="full")[0]
        result["folds"][heldout] = {"eval_set": eval_name, "rows": len(y),
                                    "references": data.ref_basename.nunique(),
                                    "label_source": label_meta, "source_sets": included,
                                    "selection": selection, "score": score,
                                    "prediction": pred.tolist(), "fit_sha256": fit_sha,
                                    "peer_coefficients": weights[-2:].tolist(),
                                    "selected_feature_ids": np.flatnonzero(np.abs(weights) > 1e-10).tolist()}
        print(json.dumps({"heldout": heldout, "eval_set": eval_name,
                          "rows": len(y), "srocc": score["srocc"]}), flush=True)
    output = root / "result.json"
    output.write_text(json.dumps(result) + "\n")
    print(json.dumps({"result": str(output), "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    main()
