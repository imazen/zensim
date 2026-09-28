"""B=2000 reference-clustered SROCC intervals for one POTENTIAL fit cell.

The panel owner computes all correlations; this driver only supplies seeded
reference-index resamples and summarizes the resulting distribution.
"""

import argparse
import json

import numpy as np

from data import load
from linear_probe import ROOT, sha

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--model", choices=["bvls", "linear"], required=True)
    ap.add_argument("--arm", choices=["r0", "minus_basic"], required=True)
    args = ap.parse_args()
    root = ROOT / "fits" / f"POT_{args.set}_{args.arm}_{args.model}"
    source = root / "result.json"
    cell = json.loads(source.read_text())
    data, label_meta = load(args.set, features=False)
    refs = data.ref_basename.astype(str).to_numpy()
    target = data.target.to_numpy(dtype=np.float64)
    groups = [np.flatnonzero(refs == ref) for ref in sorted(set(refs))]
    rng = np.random.default_rng(20260923)
    samples = [[groups[i] for i in rng.integers(0, len(groups), len(groups))]
               for _ in range(2000)]
    result = {"schema": "rev4-featpot-cell-ci-v1", "set": args.set,
              "model": args.model, "arm": args.arm,
              "label": "POTENTIAL — ceiling, not a model score",
              "source_result_sha256": sha(source), "label_source": label_meta,
              "rows": len(target), "references": len(groups),
              "B": 2000, "seed": 20260923, "unit": "reference", "phases": {}}
    for phase in ("nested", "in_sample"):
        pred = np.asarray(cell[phase]["prediction"], dtype=np.float64)
        if len(pred) != len(target) or not np.isfinite(pred).all():
            raise ValueError(f"{phase}: prediction row/nonfinite mismatch")
        jobs = [("POINT", "p", "y", None)]
        jobs.extend((f"B{b}", "p", "y", np.concatenate(sample))
                    for b, sample in enumerate(samples))
        rows = panel_batch_indexed({"p": pred, "y": target}, jobs,
                                   stats="srocc", timeout=7200)
        by = {row["label"]: row["srocc"] for row in rows}
        boot = np.asarray([by[f"B{b}"] for b in range(2000)], dtype=np.float64)
        finite = boot[np.isfinite(boot)]
        if len(finite) < 1900:
            raise ValueError(f"{phase}: only {len(finite)} finite bootstrap scores")
        point = float(by["POINT"])
        if abs(point - cell[phase]["score"]["srocc"]) > 1e-12:
            raise ValueError(f"{phase}: panel point differs from stored fit score")
        ci = np.quantile(finite, [0.025, 0.975]).tolist()
        result["phases"][phase] = {"srocc": point, "ci95_srocc": ci,
                                    "bootstrap_finite": len(finite),
                                    "bootstrap_srocc": boot.tolist()}
        print(json.dumps({"phase": phase, "rows": len(target),
                          "references": len(groups), "srocc": point,
                          "ci95_srocc": ci}), flush=True)
    output = root / "ci.json"
    output.write_text(json.dumps(result) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    main()
