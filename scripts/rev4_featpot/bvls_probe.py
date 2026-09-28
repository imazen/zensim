"""Sign-masked BVLS Rev4 POTENTIAL cell using the Rust diagnostic fit owner.

Run the whole script under ``~/tmp/devin/heavy``. Input is only the admitted
diagnostic table; every fit uses a fold-local target transform and emits f64
coefficients under /var/tmp, never a deployable bake.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import restore_data
from data import load
from linear_probe import BIN, FEATURES, REPO, ROOT, fold_table, panel_batch, run, sha


def fit(gram_path: Path, output: Path, arm: str, slice_file: Path) -> dict:
    bounds = REPO / "benchmarks/feature_sign_mask_2026-05-26.tsv"
    if arm.removesuffix("_perm") in restore_data.ARM_KEYS:
        bounds = restore_data.bounds_tsv(arm)  # registry direction for the packed columns
    cmd = [str(BIN), "fit-lasso", "--gram", str(gram_path), "--space", "raw",
           "--target", "target__mm01", "--solver", "bvls", "--lam", "0",
           "--bounds-tsv", str(bounds),
           "--emit-fit-npz", str(output), "--diagnostic-fit-only",
           "--out", str(output.with_suffix(".unused.bin"))]
    if arm == "minus_basic" or arm.removesuffix("_perm") == "a1w":
        cmd += ["--slice-file", str(slice_file)]
    run(cmd, output.with_suffix(".log"))
    with np.load(output) as values:
        return {name: values[name].copy() for name in ("w", "mu", "sd", "bias")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--arm", choices=["r0", "minus_basic"], required=True)
    args = ap.parse_args()
    if not BIN.is_file():
        raise FileNotFoundError(f"build {BIN} through the heavy runner first")
    folds = json.loads((ROOT / "folds.json").read_text())["sets"][args.set]
    admitted = ROOT / "admitted" / f"POT_{args.set}_rev3_944.parquet"
    data, label_meta = load(args.set)
    label = label_meta["target"]
    x = data[FEATURES].to_numpy(dtype=np.float64, copy=False)
    y = data["target"].to_numpy(dtype=np.float64)
    refs = data["ref_basename"].astype(str).to_numpy()
    dest = ROOT / "fits" / f"POT_{args.set}_{args.arm}_bvls"
    dest.mkdir(parents=True, exist_ok=True)
    slice_file = dest / "keep_f228_to_f943.txt"
    slice_file.write_text("\n".join(map(str, range(228, 944))) + "\n")

    def one(name: str, fit_refs: list[str], test_idx: np.ndarray) -> dict:
        fit_idx = np.flatnonzero(np.isin(refs, fit_refs))
        stem = dest / name
        bounds = fold_table(data, fit_idx, stem.with_suffix(".parquet"))
        from linear_probe import gram
        gram(stem.with_suffix(".parquet"), stem.with_suffix(".gram.npz"), per_ref=name == "full")
        values = fit(stem.with_suffix(".gram.npz"), stem.with_suffix(".fit.npz"), args.arm, slice_file)
        pred = (x[test_idx] - values["mu"]) / values["sd"] @ values["w"] + values["bias"].item()
        score = panel_batch([(name, pred, y[test_idx])], stats="full")[0]
        return {"fit_rows": len(fit_idx), "test_rows": len(test_idx), "bounds": bounds,
                "score": score, "prediction": pred.tolist(), "test_index": test_idx.tolist(),
                "gram_sha256": sha(stem.with_suffix(".gram.npz")),
                "fit_sha256": sha(stem.with_suffix(".fit.npz"))}

    result = {"schema": "rev4-featpot-bvls-v1", "set": args.set, "arm": args.arm,
              "input_sha256": sha(admitted), "target": label,
              "label_source": label_meta,
              "label": "POTENTIAL — ceiling, not a model score", "outer": []}
    for i, outer in enumerate(folds["outer"]):
        fit_refs = sorted(set(folds["full_refs"]) - set(outer["test_refs"]))
        test = np.flatnonzero(np.isin(refs, outer["test_refs"]))
        value = one(f"o{i}", fit_refs, test)
        result["outer"].append(value)
        print(json.dumps({"outer": i, "test_rows": value["test_rows"],
                          "srocc": value["score"]["srocc"]}), flush=True)
    oof = np.full(len(data), np.nan, dtype=np.float64)
    for fold in result["outer"]:
        oof[np.asarray(fold["test_index"], dtype=np.int64)] = fold["prediction"]
    if not np.isfinite(oof).all():
        raise ValueError("outer folds did not cover every row exactly once")
    result["nested"] = {"score": panel_batch([("nested", oof, y)], stats="full")[0],
                        "prediction": oof.tolist()}
    result["in_sample"] = one("full", folds["full_refs"], np.arange(len(data)))
    result["gap_srocc"] = result["in_sample"]["score"]["srocc"] - result["nested"]["score"]["srocc"]
    out = dest / "result.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(out), "sha256": sha(out),
                      "nested_srocc": result["nested"]["score"]["srocc"],
                      "in_sample_srocc": result["in_sample"]["score"]["srocc"],
                      "gap_srocc": result["gap_srocc"]}), flush=True)


if __name__ == "__main__":
    main()
