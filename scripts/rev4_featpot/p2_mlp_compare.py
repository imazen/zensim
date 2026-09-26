"""Paired P2/P0 and matched-permutation five-seed MLP deltas from saved panel draws."""

import argparse
import json

import numpy as np

from p2_data import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True)
    parser.add_argument("--hidden", type=int, choices=[32, 128], required=True)
    args = parser.parse_args()
    stem = f"POT_{args.set}_"
    paths = {"p0": ROOT / "mlp_aggregate" / f"{stem}r0_mlp{args.hidden}" / "result.json",
             "p2": ROOT / "p2/mlp_aggregate" / f"{stem}p2_mlp{args.hidden}" / "result.json",
             "perm": ROOT / "p2/mlp_aggregate" / f"{stem}p2_perm_mlp{args.hidden}" / "result.json"}
    values = {key: json.loads(path.read_text()) for key, path in paths.items()}
    if any(value["B"] != 2000 or value["seed"] != 20260923 or len(value["replicates"]) != 5
           for value in values.values()):
        raise ValueError("incomplete or non-paired bootstrap protocol")
    for rep in range(5):
        signatures = [(value["replicates"][rep]["init_seed"],
                       tuple(value["replicates"][rep]["outer_sample_seeds"]))
                      for value in values.values()]
        if len(set(signatures)) != 1:
            raise ValueError(f"rep {rep}: Latin seed mismatch")
    per_rep = {}
    for key, value in values.items():
        per_rep[key] = np.asarray([rep["bootstrap_srocc"] for rep in value["replicates"]],
                                  dtype=np.float64)
        if per_rep[key].shape != (5, 2000):
            raise ValueError(f"{key}: bootstrap shape mismatch")
    result = {"schema": "rev4-featpot-p2-mlp-compare-v1", "set": args.set,
              "hidden": args.hidden, "B": 2000, "seed": 20260923,
              "unit": "reference", "source_sha256": {key: sha(path) for key, path in paths.items()},
              "points": {key: value["nested_mean_srocc"] for key, value in values.items()},
              "deltas": {}}
    for key in ("p2", "perm"):
        boot = np.mean(per_rep[key] - per_rep["p0"], axis=0)
        finite = boot[np.isfinite(boot)]
        if len(finite) < 1900:
            raise ValueError(f"{key}: only {len(finite)} finite paired bootstraps")
        result["deltas"][key] = {"point_delta_srocc": values[key]["nested_mean_srocc"] -
                                values["p0"]["nested_mean_srocc"],
                                "ci95": np.quantile(finite, [0.025, 0.975]).tolist(),
                                "bootstrap_finite": len(finite), "bootstrap_deltas": boot.tolist()}
    out = ROOT / "p2/mlp_compare" / f"POT_{args.set}_mlp{args.hidden}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result) + "\n")
    print(json.dumps({"result": str(out), "sha256": sha(out),
                      "p2_delta": result["deltas"]["p2"]["point_delta_srocc"],
                      "p2_ci95": result["deltas"]["p2"]["ci95"],
                      "perm_ci95": result["deltas"]["perm"]["ci95"]}), flush=True)


if __name__ == "__main__":
    main()
