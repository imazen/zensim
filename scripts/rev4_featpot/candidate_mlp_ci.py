"""B=2,000 reference-clustered SROCC interval for one C1-C4 MLP replicate.

Targets come only from the admitted bank labels adapter. The panel statistics
owner computes every correlation; this driver supplies reference resamples.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from linear_probe import ROOT, sha
from candidate_data import ARMS, load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--arm", choices=[*ARMS, *(f"{name}_perm" for name in ARMS)], required=True)
    ap.add_argument("--hidden", type=int, choices=[32, 128], required=True)
    ap.add_argument("--outer", type=int, choices=range(5))
    ap.add_argument("--rep", type=int, choices=range(5), required=True)
    args = ap.parse_args()
    tag = "full" if args.outer is None else f"o{args.outer}"
    root = (ROOT / "candidates/mlp" / f"POT_{args.set}_{args.arm}_mlp{args.hidden}"
            / f"{tag}_r{args.rep}")
    source = root / "result.json"
    cell = json.loads(source.read_text())
    expected = (args.set, args.arm, args.hidden, args.outer, args.rep)
    actual = (cell["set"], cell["arm"], cell["hidden"],
              cell["outer"], cell["rep"])
    if actual != expected:
        raise ValueError(f"MLP result identity {actual} != {expected}")
    data, label_meta = load(args.set, args.arm, features=False)
    ii = np.asarray(cell["test_index"], dtype=int)
    if (len(ii) != cell["test_rows"] or len(set(ii)) != len(ii)
            or np.any(ii < 0) or np.any(ii >= len(data))):
        raise ValueError("invalid MLP test index")
    refs = data.loc[ii, "ref_basename"].astype(str).to_numpy()
    target = data.loc[ii, "target"].to_numpy(dtype=np.float64)
    pred = np.asarray(cell["prediction"], dtype=np.float64)
    if len(pred) != len(ii) or not np.isfinite(pred).all():
        raise ValueError("MLP prediction row/nonfinite mismatch")
    groups = [np.flatnonzero(refs == ref) for ref in sorted(set(refs))]
    rng = np.random.default_rng(20260923)
    jobs = [("POINT", "p", "y", None)]
    for b in range(2000):
        sample = rng.integers(0, len(groups), len(groups))
        jobs.append((f"B{b}", "p", "y",
                     np.concatenate([groups[i] for i in sample])))
    rows = panel_batch_indexed({"p": pred, "y": target}, jobs,
                               stats="srocc", timeout=7200)
    by = {row["label"]: row["srocc"] for row in rows}
    boot = np.asarray([by[f"B{b}"] for b in range(2000)], dtype=np.float64)
    finite = boot[np.isfinite(boot)]
    if len(finite) < 1900:
        raise ValueError(f"only {len(finite)} finite bootstrap scores")
    point = float(by["POINT"])
    if abs(point - cell["score"]["srocc"]) > 1e-12:
        raise ValueError("panel point differs from stored MLP score")
    ci = np.quantile(finite, [0.025, 0.975]).tolist()
    output = {"schema": "rev4-featpot-c1c4-mlp-replicate-ci-v1",
              "label": "POTENTIAL — ceiling, not a model score",
              "set": args.set, "arm": args.arm, "hidden": args.hidden,
              "outer": args.outer, "rep": args.rep,
              "source_result_sha256": sha(source), "label_source": label_meta,
              "rows": len(ii), "references": len(groups),
              "B": 2000, "seed": 20260923, "unit": "reference",
              "srocc": point, "ci95_srocc": ci,
              "bootstrap_finite": len(finite),
              "bootstrap_srocc": boot.tolist()}
    path = root / "ci.json"
    path.write_text(json.dumps(output) + "\n")
    print(json.dumps({"output": str(path), "sha256": sha(path),
                      "rows": len(ii), "references": len(groups),
                      "srocc": point, "ci95_srocc": ci}), flush=True)


if __name__ == "__main__":
    main()
