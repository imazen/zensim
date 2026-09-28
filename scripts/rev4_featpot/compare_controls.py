"""Paired reference-clustered Rev4 POTENTIAL control CI via panel --batch.

The caller owns only the RNG and cluster index sets; every SROCC is computed
by the canonical Rust panel owner through zen_stats.panel_batch_indexed.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from data import load

ROOT = Path("/var/tmp/rev4-featpot")
TMP = ROOT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP)
tempfile.tempdir = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--model", choices=["linear", "bvls"], required=True)
    ap.add_argument("--sham", action="store_true",
                    help="P0 zero-column permutation: compare R0 predictions to themselves")
    args = ap.parse_args()
    paths = [ROOT / "fits" / f"POT_{args.set}_{arm}_{args.model}" / "result.json"
             for arm in ("r0", "r0" if args.sham else "minus_basic")]
    results = [json.loads(path.read_text()) for path in paths]
    base, label_meta = load(args.set, features=False)
    target = label_meta["target"]
    refs = base["ref_basename"].astype(str).to_numpy()
    y = base["target"].to_numpy(dtype=np.float64)
    p0, pminus = [np.asarray(result["nested"]["prediction"], dtype=np.float64)
                  for result in results]
    if not (len(refs) == len(y) == len(p0) == len(pminus)):
        raise ValueError("prediction/target length mismatch")
    groups = [np.flatnonzero(refs == ref) for ref in sorted(set(refs))]
    rng = np.random.default_rng(20260923)
    jobs = [("p0_full", "p0", "y", None), ("minus_full", "minus", "y", None)]
    for b in range(2000):
        draw = rng.integers(0, len(groups), size=len(groups))
        ix = np.concatenate([groups[i] for i in draw])
        jobs.append((f"p0_{b}", "p0", "y", ix))
        jobs.append((f"minus_{b}", "minus", "y", ix))
    rows = panel_batch_indexed({"p0": p0, "minus": pminus, "y": y}, jobs,
                               stats="srocc", timeout=7200)
    by = {row["label"]: row["srocc"] for row in rows}
    deltas = np.asarray([by[f"p0_{b}"] - by[f"minus_{b}"] for b in range(2000)])
    point = by["p0_full"] - by["minus_full"]
    ci = np.quantile(deltas, [0.025, 0.975])
    if args.sham and (point != 0 or np.any(deltas != 0)):
        raise ValueError("P0 zero-column sham must be exactly identical")
    suffix = "sham_control" if args.sham else "control"
    out = ROOT / "fits" / f"POT_{args.set}_{args.model}_{suffix}.json"
    result = {"schema": "rev4-featpot-paired-control-v1", "set": args.set,
              "model": args.model, "label": "POTENTIAL — ceiling, not a model score",
              "kind": "p0_zero_column_sham" if args.sham else "r0_minus_basic_positive",
              "target": target, "label_source": label_meta,
              "rows": len(y), "references": len(groups),
              "bootstrap": {"B": 2000, "seed": 20260923, "unit": "reference",
                            "point_delta_srocc": float(point), "ci95": ci.tolist(),
                            "deltas": deltas.tolist()},
              "instrument_sensitive": bool(point > 0 and ci[0] > 0)}
    out.write_text(json.dumps(result) + "\n")
    print(json.dumps({"result": str(out), "rows": len(y), "references": len(groups),
                      "delta_srocc": point, "ci95": ci.tolist(),
                      "instrument_sensitive": result["instrument_sensitive"]}))


if __name__ == "__main__":
    main()
