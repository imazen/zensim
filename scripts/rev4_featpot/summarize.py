"""Collect existing POTENTIAL result receipts without opening label files.

The model and panel owners computed every statistic. This script only copies
their numbers, hashes each result and lists unrun preregistered cells.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/var/tmp/rev4-featpot")
SETS = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
        "cid22_a25", "aic3", "kadid_select", "konfig_val")
MODELS = ("bvls", "linear")
ARMS = ("r0", "minus_basic")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "baseline_summary.json")
    args = ap.parse_args()
    folds = json.loads((ROOT / "folds.json").read_text())["sets"]
    summary = {"schema": "rev4-featpot-baseline-summary-v1",
               "label": "POTENTIAL — ceiling, not a model score",
               "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "fold_manifest_sha256": sha(ROOT / "folds.json"),
               "admission_receipt_sha256": sha(ROOT / "admit_bank.jsonl"),
               "cells": [], "controls": [], "shams": [], "lodo": [],
               "transfer_matrices": [], "stability": [], "stats": [],
               "mlp_replicates": [], "missing": [],
               "deferred": ["C1-C4 and GMSD candidate arms: reviewed Part B sidecars pending",
                            "MCL-JCI: D3 fitting role pending"]}
    for name in SETS:
        for model in MODELS:
            for arm in ARMS:
                path = ROOT / "fits" / f"POT_{name}_{arm}_{model}" / "result.json"
                if not path.is_file():
                    summary["missing"].append(f"{name}/{model}/{arm}: nested and in-sample")
                    continue
                value = json.loads(path.read_text())
                ci_path = path.parent / "ci.json"
                ci = json.loads(ci_path.read_text()) if ci_path.is_file() else None
                if ci and ci["source_result_sha256"] != sha(path):
                    raise ValueError(f"{ci_path}: CI source result hash changed")
                summary["cells"].append({"set": name, "model": model, "arm": arm,
                                         "rows": folds[name]["rows"],
                                         "references": folds[name]["references"],
                                         "target": value["target"],
                                         "label_source": value["label_source"],
                                         "nested_srocc": value["nested"]["score"]["srocc"],
                                         "in_sample_srocc": value["in_sample"]["score"]["srocc"],
                                         "gap_srocc": value["gap_srocc"],
                                         "nested_ci95_srocc": ci["phases"]["nested"]["ci95_srocc"] if ci else None,
                                         "in_sample_ci95_srocc": ci["phases"]["in_sample"]["ci95_srocc"] if ci else None,
                                         "outer_folds": len(value["outer"]),
                                         "path": str(path), "sha256": sha(path),
                                         "ci_sha256": sha(ci_path) if ci else None})
                if not ci:
                    summary["missing"].append(f"{name}/{model}/{arm}: reference-clustered CI")
                stats_path = ROOT / "stats" / f"{name}_{arm}_{model}.json"
                if stats_path.is_file():
                    stats_value = json.loads(stats_path.read_text())
                    summary["stats"].append({"set": name, "model": model, "arm": arm,
                                             "path": str(stats_path),
                                             "sha256": sha(stats_path),
                                             "phases": list(stats_value["phases"])})
                else:
                    summary["missing"].append(f"{name}/{model}/{arm}: E1/pairwise statistics")
            path = ROOT / "fits" / f"POT_{name}_{model}_control.json"
            if not path.is_file():
                summary["missing"].append(f"{name}/{model}: paired positive control")
            else:
                value = json.loads(path.read_text())
                summary["controls"].append({"set": name, "model": model,
                                            "rows": value["rows"], "references": value["references"],
                                            "delta_srocc": value["bootstrap"]["point_delta_srocc"],
                                            "ci95": value["bootstrap"]["ci95"],
                                            "sensitive": value["instrument_sensitive"],
                                            "path": str(path), "sha256": sha(path)})
            sham_path = ROOT / "fits" / f"POT_{name}_{model}_sham_control.json"
            if sham_path.is_file():
                sham = json.loads(sham_path.read_text())
                if sham["kind"] != "p0_zero_column_sham":
                    raise ValueError(f"{sham_path}: wrong sham kind")
                summary["shams"].append({"set": name, "model": model,
                                         "delta_srocc": sham["bootstrap"]["point_delta_srocc"],
                                         "ci95": sham["bootstrap"]["ci95"],
                                         "path": str(sham_path), "sha256": sha(sham_path)})
            else:
                summary["missing"].append(f"{name}/{model}: P0 zero-column sham")
        path = ROOT / "stability" / f"POT_{name}_r0_lasso" / "result.json"
        if path.is_file():
            value = json.loads(path.read_text())
            summary["stability"].append({"set": name, "draws": value["draws_completed"],
                                         "family_frequency": value["family_frequency"],
                                         "path": str(path), "sha256": sha(path)})
        else:
            summary["missing"].append(f"{name}: 200-reference-half-sample lasso stability")
    for model in MODELS:
        root = ROOT / "lodo" / f"LODO_r0_{model}"
        path, ci_path = root / "result.json", root / "ci.json"
        if not path.is_file():
            summary["missing"].append(f"R0/{model}: seven-fold LODO")
            continue
        value = json.loads(path.read_text())
        ci = json.loads(ci_path.read_text()) if ci_path.is_file() else None
        if ci and ci["source_result_sha256"] != sha(path):
            raise ValueError(f"{ci_path}: LODO CI source result hash changed")
        for heldout, fold in value["folds"].items():
            summary["lodo"].append({"model": model, "heldout": heldout,
                                    "eval_set": fold["eval_set"],
                                    "rows": fold["rows"], "references": fold["references"],
                                    "target": fold["eval_target"],
                                    "srocc": fold["score"]["srocc"],
                                    "ci95_srocc": ci["folds"][heldout]["ci95_srocc"] if ci else None,
                                    "result_sha256": sha(path),
                                    "ci_sha256": sha(ci_path) if ci else None})
        if not ci:
            summary["missing"].append(f"R0/{model}: LODO reference-clustered CIs")
        matrix_path = root / "transfer_matrix.json"
        if matrix_path.is_file():
            matrix = json.loads(matrix_path.read_text())
            if matrix["source_result_sha256"] != sha(path):
                raise ValueError(f"{matrix_path}: source LODO result hash changed")
            summary["transfer_matrices"].append({"model": model,
                                                 "cells": sum(len(r["columns"])
                                                              for r in matrix["rows"].values()),
                                                 "path": str(matrix_path),
                                                 "sha256": sha(matrix_path)})
        else:
            summary["missing"].append(f"R0/{model}: 7x7 cross-evaluation transfer matrix")
    for name in SETS:
        for arm in ARMS:
            for hidden in (32, 128):
                base = ROOT / "fits" / f"POT_{name}_{arm}_mlp{hidden}"
                for outer in [None, *range(5)]:
                    for rep in range(5):
                        tag = "full" if outer is None else f"o{outer}"
                        path = base / f"{tag}_r{rep}" / "result.json"
                        if not path.is_file():
                            continue
                        value = json.loads(path.read_text())
                        importance_path = path.parent / "importance.json"
                        if importance_path.is_file():
                            importance = json.loads(importance_path.read_text())
                            if importance["source_result_sha256"] != sha(path):
                                raise ValueError(f"{importance_path}: MLP source hash changed")
                        ci_path = path.parent / "ci.json"
                        ci = json.loads(ci_path.read_text()) if ci_path.is_file() else None
                        if ci and ci["source_result_sha256"] != sha(path):
                            raise ValueError(f"{ci_path}: MLP CI source hash changed")
                        summary["mlp_replicates"].append({
                            "set": name, "arm": arm, "hidden": hidden,
                            "outer": outer, "rep": rep,
                            "rows": value["test_rows"], "selected_epoch": value["selected_epoch"],
                            "srocc": value["score"]["srocc"],
                            "references": ci["references"] if ci else None,
                            "ci95_srocc": ci["ci95_srocc"] if ci else None,
                            "ci_sha256": sha(ci_path) if ci else None,
                            "path": str(path), "sha256": sha(path),
                            "importance_sha256": sha(importance_path)
                            if importance_path.is_file() else None,
                        })
    summary["mlp_expected_outer"] = len(SETS) * len(ARMS) * 2 * 5 * 5
    summary["mlp_expected_full"] = len(SETS) * len(ARMS) * 2 * 5
    if len(summary["mlp_replicates"]) < summary["mlp_expected_outer"] + summary["mlp_expected_full"]:
        summary["missing"].append(
            f"H32/H128 MLP replicates: {len(summary['mlp_replicates'])}/"
            f"{summary['mlp_expected_outer'] + summary['mlp_expected_full']}")
    mlp_ci_count = sum(r["ci_sha256"] is not None for r in summary["mlp_replicates"])
    if mlp_ci_count < summary["mlp_expected_outer"] + summary["mlp_expected_full"]:
        summary["missing"].append(
            f"MLP replicate reference CIs: {mlp_ci_count}/"
            f"{summary['mlp_expected_outer'] + summary['mlp_expected_full']}")
    outer_replicates = [r for r in summary["mlp_replicates"] if r["outer"] is not None]
    if any(r["importance_sha256"] is None for r in outer_replicates) or len(outer_replicates) < summary["mlp_expected_outer"]:
        summary["missing"].append(
            f"MLP outer-fold permutation importance: "
            f"{sum(r['importance_sha256'] is not None for r in outer_replicates)}/"
            f"{summary['mlp_expected_outer']}")
    summary["missing"] += ["H32 and H128 non-negative-head MLP LODO",
                           "incumbent/peer LODO rows and JPEG residual analysis"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "sha256": sha(args.out),
                      "cells": len(summary["cells"]), "controls": len(summary["controls"]),
                      "shams": len(summary["shams"]),
                      "lodo_folds": len(summary["lodo"]), "stability_sets": len(summary["stability"]),
                      "transfer_matrices": len(summary["transfer_matrices"]),
                      "stats_cells": len(summary["stats"]),
                      "mlp_replicates": len(summary["mlp_replicates"]),
                      "mlp_replicate_cis": mlp_ci_count,
                      "missing": len(summary["missing"])}))


if __name__ == "__main__":
    main()
