"""P2/paired-permutation D1 lasso and BVLS via the registered Rust owners."""

import argparse
import json
from pathlib import Path

import numpy as np

import linear_probe as lp
from bvls_probe import fit as bvls_fit
from p2_data import ROOT, load, sha


COLS = [f"f{i}" for i in range(946)]
lp.FEATURES = COLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True)
    parser.add_argument("--arm", choices=["p2", "p2_perm"], required=True)
    parser.add_argument("--model", choices=["linear", "bvls"], required=True)
    args = parser.parse_args()
    folds = json.loads((ROOT / "folds.json").read_text())["sets"][args.set]
    source, label_meta = load(args.set, args.arm)
    data = source.rename(columns={"gmsd": "f944", "gmsm": "f945"})
    if data[COLS].shape[1] != 946 or not np.isfinite(data[COLS].to_numpy()).all():
        raise ValueError("P2 feature table is not finite and 946 wide")
    x = data[COLS].to_numpy(dtype=np.float64)
    y = data.target.to_numpy(dtype=np.float64)
    refs = data.ref_basename.astype(str).to_numpy()
    dest = ROOT / "p2" / "d1" / f"POT_{args.set}_{args.arm}_{args.model}"
    dest.mkdir(parents=True, exist_ok=True)
    result = {"schema": "rev4-featpot-p2-d1-v1", "set": args.set, "arm": args.arm,
              "model": args.model, "label_source": label_meta,
              "peer_manifest_sha256": label_meta["peer_manifest_sha256"],
              "label": "POTENTIAL — ceiling, not a model score", "outer": []}

    def indices(ref_list):
        return np.flatnonzero(np.isin(refs, ref_list))

    def prepare(stem: Path, fit_idx: np.ndarray, per_ref: bool = False):
        bounds = lp.fold_table(data, fit_idx, stem.with_suffix(".parquet"))
        lp.gram(stem.with_suffix(".parquet"), stem.with_suffix(".npz"),
                per_ref=per_ref, n_features=946)
        return bounds

    def score(pred, test_idx, tag):
        return lp.panel_batch([(tag, pred, y[test_idx])], stats="full")[0]

    def bvls_one(tag, fit_refs, test_idx, per_ref=False):
        fit_idx = indices(fit_refs)
        stem = dest / tag
        bounds = prepare(stem, fit_idx, per_ref=per_ref)
        fit_path = stem.with_suffix(".fit.npz")
        values = bvls_fit(stem.with_suffix(".npz"), fit_path, args.arm,
                          Path("/dev/null"))
        pred = (x[test_idx] - values["mu"]) / values["sd"] @ values["w"] + values["bias"].item()
        return {"fit_rows": len(fit_idx), "test_rows": len(test_idx), "bounds": bounds,
                "test_index": test_idx.tolist(), "prediction": pred.tolist(),
                "score": score(pred, test_idx, tag), "gram_sha256": sha(stem.with_suffix(".npz")),
                "fit_sha256": sha(fit_path), "peer_coefficients": values["w"][-2:].tolist()}

    def lasso_one(tag, fit_refs, inner_folds, test_idx, per_ref=False):
        fit_idx = indices(fit_refs)
        inner_scores = []
        for inner_i, val_refs in enumerate(inner_folds):
            train_refs = sorted(set(fit_refs) - set(val_refs))
            train_idx = indices(train_refs)
            val_idx = indices(val_refs)
            stem = dest / f"{tag}_i{inner_i}"
            prepare(stem, train_idx)
            path = lp.lasso_path(stem.with_suffix(".npz"), stem.with_suffix(".json"),
                                 args.arm, Path("/dev/null"))
            predictions = lp.predict_path(path, x[val_idx])
            rows = lp.panel_batch([(f"lambda_{i}", predictions[:, i], y[val_idx])
                                   for i in range(50)], stats="srocc")
            inner_scores.append([row["srocc"] for row in rows])
        selected, selection = lp.choose_1se(inner_scores)
        stem = dest / f"{tag}_fit"
        bounds = prepare(stem, fit_idx, per_ref=per_ref)
        path_file = stem.with_suffix(".json")
        path = lp.lasso_path(stem.with_suffix(".npz"), path_file, args.arm, Path("/dev/null"))
        pred = lp.predict_path(path, x[test_idx])[:, selected]
        selected_w = np.asarray(path["fits"][selected]["w"], dtype=np.float64)
        return {"fit_rows": len(fit_idx), "test_rows": len(test_idx), "bounds": bounds,
                "selection": selection, "selected_lambda": path["fits"][selected]["lambda"],
                "selected_feature_ids": np.flatnonzero(np.abs(selected_w) > 1e-10).tolist(),
                "test_index": test_idx.tolist(), "prediction": pred.tolist(),
                "score": score(pred, test_idx, tag), "gram_sha256": sha(stem.with_suffix(".npz")),
                "path_sha256": sha(path_file), "peer_coefficients": selected_w[-2:].tolist()}

    for outer_i, outer in enumerate(folds["outer"]):
        fit_refs = sorted(set(folds["full_refs"]) - set(outer["test_refs"]))
        test_idx = indices(outer["test_refs"])
        if args.model == "bvls":
            value = bvls_one(f"o{outer_i}", fit_refs, test_idx)
        else:
            value = lasso_one(f"o{outer_i}", fit_refs, outer["inner_val_refs"], test_idx)
        result["outer"].append(value)
        print(json.dumps({"outer": outer_i, "srocc": value["score"]["srocc"]}), flush=True)
    oof = np.full(len(data), np.nan)
    for value in result["outer"]:
        oof[np.asarray(value["test_index"], dtype=np.int64)] = value["prediction"]
    if not np.isfinite(oof).all():
        raise ValueError("incomplete P2 outer-fold coverage")
    result["nested"] = {"score": score(oof, np.arange(len(y)), "nested"),
                        "prediction": oof.tolist()}
    if args.model == "bvls":
        result["in_sample"] = bvls_one("full", folds["full_refs"],
                                       np.arange(len(data)), per_ref=True)
    else:
        result["in_sample"] = lasso_one("full", folds["full_refs"],
                                        folds["full_inner_val_refs"],
                                        np.arange(len(data)), per_ref=True)
    result["gap_srocc"] = (result["in_sample"]["score"]["srocc"] -
                           result["nested"]["score"]["srocc"])
    output = dest / "result.json"
    output.write_text(json.dumps(result) + "\n")
    print(json.dumps({"result": str(output), "sha256": sha(output),
                      "nested_srocc": result["nested"]["score"]["srocc"],
                      "in_sample_srocc": result["in_sample"]["score"]["srocc"]}), flush=True)


if __name__ == "__main__":
    main()
