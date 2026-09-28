"""Verify registered P0 baseline core-gate receipts without opening targets."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from summarize import ROOT, sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=ROOT / "baseline_summary.json")
    args = ap.parse_args()
    path = args.summary
    summary = json.loads(path.read_text())
    cells = summary["cells"]
    mlp = summary["mlp_replicates"]
    outer = [row for row in mlp if row["outer"] is not None]
    checks = {
        "deterministic_fit_cells": len(cells) == 32,
        "deterministic_reference_cis": sum(row["ci_sha256"] is not None for row in cells) == 32,
        "positive_controls": len(summary["controls"]) == 16,
        "zero_column_shams": len(summary["shams"]) == 16 and all(
            row["delta_srocc"] == 0 and row["ci95"] == [0, 0]
            for row in summary["shams"]),
        "e1_pairwise_cells": len(summary["stats"]) == 32,
        "lasso_stability_sets": len(summary["stability"]) == 8 and all(
            row["draws"] == 200 for row in summary["stability"]),
        "deterministic_lodo_folds": len(summary["lodo"]) == 14,
        "transfer_matrices": len(summary["transfer_matrices"]) == 2 and all(
            row["cells"] == 49 for row in summary["transfer_matrices"]),
        "mlp_fit_replicates": len(mlp) == summary["mlp_expected_outer"] + summary["mlp_expected_full"],
        "mlp_reference_cis": sum(row["ci_sha256"] is not None for row in mlp) == summary["mlp_expected_outer"] + summary["mlp_expected_full"],
        "mlp_outer_importance": len(outer) == summary["mlp_expected_outer"] and all(
            row["importance_sha256"] is not None for row in outer),
    }
    result = {"schema": "rev4-featpot-core-gates-v1",
              "label": "POTENTIAL — ceiling, not a model score",
              "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "summary_sha256": sha(path), "checks": checks,
              "all_pass": all(checks.values()),
              "deferred": ["MLP LODO and incumbent/peer LODO",
                           "peer-relative JPEG residuals",
                           "reviewed Part B candidate and GMSD arms",
                           "MCL-JCI D3 role"]}
    output = ROOT / ("core_gates_done.json" if result["all_pass"]
                     else "core_gates_status.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output),
                      "all_pass": result["all_pass"], "checks": checks}))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
