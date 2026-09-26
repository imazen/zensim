"""Reference-clustered B=2000 CIs for quarantined R0 BVLS LODO folds.

The statistic is always computed by the existing panel owner; this script
only constructs the preregistered reference resamples and summarizes them.
"""

import argparse
import json

import numpy as np

from data import load
from linear_probe import ROOT, panel_batch, sha

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["r0", "minus_basic"], required=True)
    ap.add_argument("--model", choices=["bvls", "linear"], default="bvls")
    args = ap.parse_args()
    root = ROOT / "lodo" / f"LODO_{args.arm}_{args.model}"
    source = root / "result.json"
    value = json.loads(source.read_text())
    result = {"schema": "rev4-featpot-lodo-ci-v1", "arm": args.arm,
              "model": args.model,
              "label": "POTENTIAL — ceiling, not a model score",
              "source_result_sha256": sha(source), "B": 2000, "seed": 20260923,
              "unit": "reference", "folds": {}}
    for heldout, fold in value["folds"].items():
        data, label_meta = load(fold["eval_set"], features=False)
        refs = data.ref_basename.astype(str).to_numpy()
        y = data.target.to_numpy(dtype=np.float64)
        pred = np.asarray(fold["prediction"], dtype=np.float64)
        if len(y) != len(pred) or len(y) != fold["rows"]:
            raise ValueError(f"{heldout}: result/label row mismatch")
        groups = [np.flatnonzero(refs == ref) for ref in sorted(set(refs))]
        rng = np.random.default_rng(20260923)
        jobs = [("POINT", "p", "y", None)]
        for b in range(2000):
            draw = rng.integers(0, len(groups), len(groups))
            jobs.append((f"B{b}", "p", "y", np.concatenate([groups[i] for i in draw])))
        rows = panel_batch_indexed({"p": pred, "y": y}, jobs, stats="srocc", timeout=7200)
        by = {row["label"]: row["srocc"] for row in rows}
        boot = np.asarray([by[f"B{b}"] for b in range(2000)], dtype=np.float64)
        finite = boot[np.isfinite(boot)]
        if len(finite) < 1900:
            raise ValueError(f"{heldout}: only {len(finite)} finite bootstrap values")
        point = panel_batch([(heldout, pred, y)], stats="full")[0]
        ci = np.quantile(finite, [0.025, 0.975]).tolist()
        result["folds"][heldout] = {"eval_set": fold["eval_set"],
                                    "label_source": label_meta, "rows": len(y),
                                    "references": len(groups), "point": point,
                                    "ci95_srocc": ci, "bootstrap_finite": len(finite),
                                    "bootstrap_srocc": boot.tolist()}
        print(json.dumps({"heldout": heldout, "rows": len(y), "references": len(groups),
                          "srocc": point["srocc"], "ci95_srocc": ci}), flush=True)
    output = root / "ci.json"
    output.write_text(json.dumps(result) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    main()
