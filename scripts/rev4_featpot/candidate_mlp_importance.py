"""Outer-fold family permutation importance for a C1-C4 MLP diagnostic.

This reads evaluation targets only through the role-allowed bank labels file.
For each family, it shuffles the same rows across all its columns within each
reference, preserving the reference and feature-column covariance. The Rust
predictor and panel owner compute predictions and SROCC respectively.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from linear_probe import ROOT, panel_batch, sha
from candidate_data import ARMS, FAMILIES as CANDIDATE_FAMILIES, columns, load
from candidate_mlp import predict
from stability_lasso import FAMILIES as BASE_FAMILIES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--arm", choices=[*ARMS, *(f"{name}_perm" for name in ARMS)], required=True)
    ap.add_argument("--hidden", type=int, choices=[32, 128], required=True)
    ap.add_argument("--outer", type=int, choices=range(5), required=True)
    ap.add_argument("--rep", type=int, choices=range(5), required=True)
    args = ap.parse_args()
    dest = ROOT / "candidates/mlp" / f"POT_{args.set}_{args.arm}_mlp{args.hidden}" / f"o{args.outer}_r{args.rep}"
    source = dest / "result.json"
    cell = json.loads(source.read_text())
    if cell["outer"] != args.outer or cell["rep"] != args.rep:
        raise ValueError("replicate identity mismatch")
    data, label_meta = load(args.set, args.arm)
    features, canonical = columns(args.arm)
    id_to_packed = {feature_id: 944 + j for j, feature_id in enumerate(canonical)}
    families = dict(BASE_FAMILIES)
    for name, ids in CANDIDATE_FAMILIES.items():
        packed = [id_to_packed[feature_id] for feature_id in ids if feature_id in id_to_packed]
        if packed:
            families[name] = packed
    ii = np.asarray(cell["test_index"], dtype=int)
    if len(ii) != cell["test_rows"] or len(set(ii)) != len(ii):
        raise ValueError("test index mismatch")
    refs = data.loc[ii, "ref_basename"].astype(str).to_numpy()
    keys = data.loc[ii, "pair_key"].to_numpy()
    y = data.loc[ii, "target"].to_numpy(dtype=np.float64)
    original = np.asarray(cell["prediction"], dtype=np.float64)
    baseline = panel_batch([("baseline", original, y)], stats="srocc")[0]["srocc"]
    if abs(baseline - cell["score"]["srocc"]) > 1e-12:
        raise ValueError("baseline panel score differs from replicate")
    bake = Path(cell["selected_bake"])
    if sha(bake) != cell["selected_bake_sha256"]:
        raise ValueError("selected bake hash changed")
    x = data.loc[ii, features].to_numpy(dtype=np.float32, copy=True)
    lo, hi = cell["bounds"]
    target_for_predictor = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
    output = {"schema": "rev4-featpot-c1c4-mlp-outer-permutation-v1",
              "label": "POTENTIAL — ceiling, not a model score",
              "set": args.set, "arm": args.arm, "hidden": args.hidden,
              "outer": args.outer, "rep": args.rep,
              "source_result_sha256": sha(source), "label_source": label_meta,
              "rows": len(ii), "references": len(set(refs)),
              "baseline_srocc": baseline, "family": {}}
    ref_groups = [np.flatnonzero(refs == ref) for ref in sorted(set(refs))]
    for family_i, (name, feature_ids) in enumerate(families.items()):
        rng_seed = 20260923 + args.outer * 100 + args.rep * 10 + family_i
        rng = np.random.default_rng(rng_seed)
        xp = x.copy()
        ids = list(feature_ids)
        for group in ref_groups:
            unique, first = np.unique(keys[group], return_index=True)
            values = x[np.ix_(group[first], ids)]
            mapped = dict(zip(unique, values[rng.permutation(len(unique))]))
            xp[np.ix_(group, ids)] = np.stack([mapped[key] for key in keys[group]])
        # The prediction owner requires the training target column in its
        # Parquet schema. Reconstruct it from this role-allowed bank read;
        # never load the earlier fold table's copied target values.
        frame = pa.table({"ref_basename": pa.array(refs),
                          "human_score": pa.array(target_for_predictor),
                          **{feature: pa.array(xp[:, j]) for j, feature in enumerate(features)}})
        table_path = dest / f"perm_{name}.parquet"
        pq.write_table(frame, table_path, compression="zstd")
        Path(f"{table_path}.manifest.json").write_text(json.dumps({
            "source_bank_feature_set_id": "basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#b782e349",
            "composite": "C1-C4 POTENTIAL diagnostic: bank f0-f943 plus packed candidate IDs",
            "canonical_candidate_ids": canonical,
            "formula_revision": 3,
        }) + "\n")
        p = predict(bake, table_path, dest / f"perm_{name}_preds.tsv")
        score = panel_batch([(name, p, y)], stats="srocc")[0]["srocc"]
        output["family"][name] = {"columns": ids, "seed": rng_seed,
                                  "permuted_srocc": score,
                                  "importance_srocc": baseline - score,
                                  "table_sha256": sha(table_path)}
        print(json.dumps({"family": name, "baseline_srocc": baseline,
                          "permuted_srocc": score,
                          "importance_srocc": baseline - score}), flush=True)
    result = dest / "importance.json"
    result.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"result": str(result), "sha256": sha(result)}), flush=True)


if __name__ == "__main__":
    main()
