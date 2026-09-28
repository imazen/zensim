"""Standalone reviewed GMSD/GMSM rows on the D1/D2 populations."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from p2_data import ROOT, load, sha

TMP = ROOT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP)
tempfile.tempdir = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch, panel_batch_indexed  # noqa: E402


SETS = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
        "konjnd_bpg_val", "cid22_a25", "aic3", "kadid_select", "konfig_val")


def main():
    result = {"schema": "rev4-featpot-p2-standalone-v1", "B": 2000,
              "seed": 20260923, "unit": "reference", "rows": {}}
    for name in SETS:
        data, meta = load(name)
        y = data.target.to_numpy(dtype=np.float64)
        refs = data.ref_basename.astype(str).to_numpy()
        predictions = {"gmsd": -data.gmsd.to_numpy(dtype=np.float64),
                       "gmsm": data.gmsm.to_numpy(dtype=np.float64)}
        groups = [np.flatnonzero(refs == ref) for ref in sorted(set(refs))]
        rng = np.random.default_rng(20260923)
        jobs = []
        for b in range(2000):
            draw = rng.integers(0, len(groups), len(groups))
            indices = np.concatenate([groups[i] for i in draw])
            for col in predictions:
                jobs.append((f"{col}_{b}", col, "y", indices))
        batch = panel_batch_indexed({**predictions, "y": y}, jobs, stats="srocc", timeout=7200)
        by = {row["label"]: row["srocc"] for row in batch}
        points = panel_batch([(col, pred, y) for col, pred in predictions.items()], stats="full")
        measurements = {}
        for col, point in zip(predictions, points):
            boot = np.array([by[f"{col}_{b}"] for b in range(2000)], dtype=np.float64)
            finite = boot[np.isfinite(boot)]
            if len(finite) < 1900:
                raise ValueError(f"{name}/{col}: only {len(finite)} finite bootstrap values")
            measurements[col] = {"score": point, "ci95_srocc": np.quantile(finite, [0.025, 0.975]).tolist(),
                                 "bootstrap_finite": len(finite), "bootstrap_srocc": boot.tolist(),
                                 "prediction_orientation": "negative raw gmsd" if col == "gmsd" else "raw gmsm"}
        result["rows"][name] = {"rows": len(y), "references": len(groups),
                                "target": meta["target"], "label_source": meta,
                                "measurements": measurements}
        print(json.dumps({"set": name, "rows": len(y), "references": len(groups),
                          "target": meta["target"],
                          "gmsd_srocc": measurements["gmsd"]["score"]["srocc"],
                          "gmsd_ci95": measurements["gmsd"]["ci95_srocc"],
                          "gmsm_srocc": measurements["gmsm"]["score"]["srocc"]}), flush=True)
    dest = ROOT / "p2" / "standalone.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result) + "\n")
    print(json.dumps({"result": str(dest), "sha256": sha(dest)}), flush=True)


if __name__ == "__main__":
    main()
