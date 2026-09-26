"""Paired reference-bootstrap C1-C4/P0 and matched-control deltas via panel."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from lodo_bvls import EVAL
from candidate_data import ARMS, ROOT, sha
from data import load

TMP = ROOT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP)
tempfile.tempdir = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed  # noqa: E402


def compare(name, predictions, phase):
    data, meta = load(name, features=False)
    refs = data.ref_basename.astype(str).to_numpy()
    y = data.target.to_numpy(dtype=np.float64)
    if any(len(pred) != len(y) for pred in predictions.values()):
        raise ValueError(f"{name}/{phase}: prediction length mismatch")
    groups = [np.flatnonzero(refs == ref) for ref in sorted(set(refs))]
    rng = np.random.default_rng(20260923)
    jobs = [(f"{key}_point", key, "y", None) for key in predictions]
    for b in range(2000):
        draw = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[i] for i in draw])
        jobs.extend((f"{key}_{b}", key, "y", indices) for key in predictions)
    rows = panel_batch_indexed({**predictions, "y": y}, jobs,
                               stats="srocc", timeout=7200)
    by = {row["label"]: row["srocc"] for row in rows}
    point = {key: by[f"{key}_point"] for key in predictions}
    deltas = {}
    for key in ("candidate", "perm"):
        boot = np.asarray([by[f"{key}_{b}"] - by[f"p0_{b}"] for b in range(2000)], dtype=np.float64)
        finite = boot[np.isfinite(boot)]
        if len(finite) < 1900:
            raise ValueError(f"{name}/{phase}/{key}: only {len(finite)} finite deltas")
        deltas[key] = {"point_delta_srocc": float(point[key] - point["p0"]),
                       "ci95": np.quantile(finite, [0.025, 0.975]).tolist(),
                       "bootstrap_finite": len(finite), "bootstrap_deltas": boot.tolist()}
    return {"set": name, "phase": phase, "rows": len(y), "references": len(groups),
            "label_source": meta, "point_srocc": point, "deltas": deltas,
            "permutation_instrument_sensitive": bool(deltas["perm"]["ci95"][0] > 0 or
                                                    deltas["perm"]["ci95"][1] < 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["d1", "d2"], required=True)
    parser.add_argument("--model", choices=["linear", "bvls"], required=True)
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    parser.add_argument("--set", help="required for D1")
    args = parser.parse_args()
    if args.scope == "d1":
        if not args.set:
            raise ValueError("D1 requires --set")
        paths = {"p0": ROOT / "fits" / f"POT_{args.set}_r0_{args.model}" / "result.json",
                 "candidate": ROOT / "candidates/d1" / f"POT_{args.set}_{args.arm}_{args.model}" / "result.json",
                 "perm": ROOT / "candidates/d1" / f"POT_{args.set}_{args.arm}_perm_{args.model}" / "result.json"}
        values = {key: json.loads(path.read_text()) for key, path in paths.items()}
        phases = {}
        for phase in ("nested", "in_sample"):
            preds = {key: np.asarray(value[phase]["prediction"], dtype=np.float64)
                     for key, value in values.items()}
            phases[phase] = compare(args.set, preds, phase)
            for key in preds:
                if abs(phases[phase]["point_srocc"][key] - values[key][phase]["score"]["srocc"]) > 1e-12:
                    raise ValueError(f"{args.set}/{phase}/{key}: point differs from source result")
            print(json.dumps({"set": args.set, "phase": phase,
                              "candidate_delta": phases[phase]["deltas"]["candidate"]["point_delta_srocc"],
                              "candidate_ci95": phases[phase]["deltas"]["candidate"]["ci95"],
                              "perm_ci95": phases[phase]["deltas"]["perm"]["ci95"]}), flush=True)
        output = ROOT / "candidates/d1" / f"POT_{args.set}_{args.arm}_{args.model}_compare.json"
        result = {"schema": "rev4-featpot-c1c4-compare-v1", "scope": "d1", "model": args.model,
                  "arm": args.arm,
                  "B": 2000, "seed": 20260923, "unit": "reference",
                  "sources": {key: sha(path) for key, path in paths.items()}, "phases": phases}
    else:
        paths = {"p0": ROOT / "lodo" / f"LODO_r0_{args.model}" / "result.json",
                 "candidate": ROOT / "candidates/d2" / f"LODO_{args.arm}_{args.model}" / "result.json",
                 "perm": ROOT / "candidates/d2" / f"LODO_{args.arm}_perm_{args.model}" / "result.json"}
        values = {key: json.loads(path.read_text()) for key, path in paths.items()}
        folds = {}
        for heldout in values["p0"]["folds"]:
            name = EVAL.get(heldout, heldout)
            preds = {key: np.asarray(value["folds"][heldout]["prediction"], dtype=np.float64)
                     for key, value in values.items()}
            folds[heldout] = compare(name, preds, heldout)
            for key in preds:
                if abs(folds[heldout]["point_srocc"][key] - values[key]["folds"][heldout]["score"]["srocc"]) > 1e-12:
                    raise ValueError(f"{heldout}/{key}: point differs from source result")
            print(json.dumps({"heldout": heldout,
                              "candidate_delta": folds[heldout]["deltas"]["candidate"]["point_delta_srocc"],
                              "candidate_ci95": folds[heldout]["deltas"]["candidate"]["ci95"],
                              "perm_ci95": folds[heldout]["deltas"]["perm"]["ci95"]}), flush=True)
        output = ROOT / "candidates/d2" / f"LODO_{args.arm}_{args.model}_compare.json"
        result = {"schema": "rev4-featpot-c1c4-compare-v1", "scope": "d2", "model": args.model,
                  "arm": args.arm,
                  "B": 2000, "seed": 20260923, "unit": "reference",
                  "sources": {key: sha(path) for key, path in paths.items()}, "folds": folds}
    output.write_text(json.dumps(result) + "\n")
    print(json.dumps({"result": str(output), "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    main()
