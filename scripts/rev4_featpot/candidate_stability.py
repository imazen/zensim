"""C1-C4 reference-half-sample lasso stability on frozen 1-SE lambda.

Run one set under the shared heavy wrapper. Every target value comes from the
role-allowed bank labels file through data.load. The lambda is selected by the
set's full-data four-inner-fold R0 fit and is then frozen before resampling.
This is a diagnostic POTENTIAL selection frequency, not a model score.
"""

import argparse
import json

import numpy as np

import linear_probe as lp
from linear_probe import BIN, ROOT, run, sha
import restore_data
from candidate_data import ARMS, FAMILIES as CANDIDATE_FAMILIES, columns, load


BASE_FAMILIES = {
    "basic": range(0, 156), "peaks": range(156, 228),
    "masked_iw": range(228, 372), "v2": range(372, 720),
    "append": range(720, 924), "append2": range(924, 944),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--arm", choices=[*ARMS, *(f"{name}_perm" for name in ARMS)], required=True)
    ap.add_argument("--limit", type=int, default=200, help="for instrument smoke only; full run is 200")
    args = ap.parse_args()
    if not 1 <= args.limit <= 200:
        raise ValueError("limit must be in 1..200")
    cols, canonical = columns(args.arm)
    lp.FEATURES = cols
    id_to_packed = {feature_id: 944 + j for j, feature_id in enumerate(canonical)}
    families = dict(BASE_FAMILIES)
    for name, ids in CANDIDATE_FAMILIES.items():
        packed = [id_to_packed[feature_id] for feature_id in ids if feature_id in id_to_packed]
        if packed:
            families[name] = packed
    families["added_all"] = list(range(944, 944 + len(canonical)))  # every column the arm adds
    source = ROOT / "candidates/d1" / f"POT_{args.set}_{args.arm}_linear"
    full = json.loads((source / "result.json").read_text())
    full_path = json.loads((source / "full_fit.json").read_text())
    one_se_index = full["in_sample"]["selection"]["selected_index"]
    fixed_lambda = float(full_path["fits"][one_se_index]["lambda"])
    if full["in_sample"]["path_sha256"] != sha(source / "full_fit.json"):
        raise ValueError("full-fit lasso path hash changed")
    draws = json.loads((ROOT / "stability_subsamples.json").read_text())["sets"][args.set]
    data, label_meta = load(args.set, args.arm)
    refs = data.ref_basename.astype(str).to_numpy()
    dest = ROOT / "candidates/stability" / f"POT_{args.set}_{args.arm}_lasso"
    dest.mkdir(parents=True, exist_ok=True)
    # A1w: the full-population drop list applies to every half-sample of the set (features-only list)
    slice_file = (restore_data.a1w_slice(f"D1/{args.set}/full", len(cols))
                  if restore_data.is_a1w(args.arm) else dest / "unused_slice.txt")
    result = {"schema": "rev4-featpot-c1c4-lasso-stability-v1", "set": args.set,
              "arm": args.arm,
              "canonical_candidate_ids": canonical,
              "label": "POTENTIAL — ceiling, not a model score",
              "draws_completed": args.limit, "draws_registered": 200,
              "fixed_lambda": fixed_lambda, "full_one_se_index": one_se_index,
              "full_path_sha256": sha(source / "full_fit.json"),
              "subsamples_sha256": sha(ROOT / "stability_subsamples.json"),
              "label_source": label_meta, "draws": []}
    for draw_i, sample in enumerate(draws["samples"][:args.limit]):
        fit_idx = np.flatnonzero(np.isin(refs, sample))
        stem = dest / f"s{draw_i:03}"
        bounds = lp.fold_table(data, fit_idx, stem.with_suffix(".parquet"))
        gram_path = stem.with_suffix(".gram.npz")
        path_file = stem.with_suffix(".path.json")
        lp.gram(stem.with_suffix(".parquet"), gram_path, n_features=len(cols))
        path = lp.lasso_path(gram_path, path_file, args.arm, slice_file)
        weights = np.asarray([fit["w"] for fit in path["fits"]], dtype=np.float64)
        nz = np.abs(weights) > 1e-10
        entry = np.where(nz.any(axis=0), nz.argmax(axis=0), -1)
        fit_file = stem.with_suffix(".fixed.npz")
        run([str(BIN), "fit-lasso", "--gram", str(gram_path), "--space", "raw",
             "--target", "target__mm01", "--solver", "lasso", "--lam", repr(fixed_lambda),
             "--emit-fit-npz", str(fit_file), "--diagnostic-fit-only",
             "--out", str(stem.with_suffix(".unused.bin"))]
            + (["--slice-file", str(slice_file)] if restore_data.is_a1w(args.arm) else []),
            stem.with_suffix(".fixed.log"))
        with np.load(fit_file) as fitted:
            selected = np.flatnonzero(np.abs(fitted["w"]) > 1e-10).tolist()
        result["draws"].append({"draw": draw_i, "reference_count": len(sample),
                                "rows": len(fit_idx), "bounds": bounds,
                                "selected_feature_ids": selected,
                                "path_entry_index": entry.tolist(),
                                "gram_sha256": sha(gram_path),
                                "path_sha256": sha(path_file),
                                "fixed_fit_sha256": sha(fit_file)})
        print(json.dumps({"draw": draw_i, "references": len(sample),
                          "rows": len(fit_idx), "selected": len(selected)}), flush=True)
    result["family_frequency"] = {
        name: sum(any(feature in family for feature in draw["selected_feature_ids"])
                  for draw in result["draws"]) / args.limit
        for name, family in families.items()
    }
    output = dest / ("result.json" if args.limit == 200 else f"pilot_{args.limit}.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(output), "sha256": sha(output),
                      "draws_completed": args.limit,
                      "family_frequency": result["family_frequency"]}), flush=True)


if __name__ == "__main__":
    main()
