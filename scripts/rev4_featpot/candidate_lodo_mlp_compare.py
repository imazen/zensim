"""Paired five-seed P0/C1-C4/control D2 MLP transfer with panel bootstrap."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data import load
from candidate_data import ARMS, ROOT, sha
from lodo_bvls import EVAL, SOURCES
from mlp_probe import INIT_SEEDS, SAMPLE_SEEDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    parser.add_argument("--hidden", type=int, choices=[32, 128], required=True)
    parser.add_argument("--heldout", choices=SOURCES, required=True)
    args = parser.parse_args()
    eval_name = EVAL.get(args.heldout, args.heldout)
    data, meta = load(eval_name, features=False)
    y = data.target.to_numpy(dtype=np.float64)
    refs = data.ref_basename.astype(str).to_numpy()
    groups = [np.flatnonzero(refs == ref) for ref in sorted(set(refs))]
    rng = np.random.default_rng(20260923)
    samples = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
               for _ in range(2000)]
    arms = ("r0", args.arm, f"{args.arm}_perm")
    values = {}
    for arm in arms:
        rows = []
        for rep in range(5):
            source_root = ROOT / ("p2/d2_mlp" if arm == "r0" else "candidates/d2_mlp")
            path = (source_root / f"LODO_{arm}_mlp{args.hidden}" /
                    f"without_{args.heldout}_r{rep}" / "result.json")
            cell = json.loads(path.read_text())
            expected_seed = SAMPLE_SEEDS[(rep + SOURCES.index(args.heldout)) % 5]
            if ((cell["arm"], cell["hidden"], cell["heldout"], cell["eval_set"], cell["rep"])
                    != (arm, args.hidden, args.heldout, eval_name, rep)
                    or cell["init_seed"] != INIT_SEEDS[rep] or cell["sample_seed"] != expected_seed):
                raise ValueError(f"{path}: identity/Latin seed mismatch")
            prediction = np.asarray(cell["prediction"], dtype=np.float64)
            if len(prediction) != len(y) or not np.isfinite(prediction).all():
                raise ValueError(f"{path}: prediction row/nonfinite mismatch")
            jobs = [("POINT", "p", "y", None)]
            jobs.extend((f"B{b}", "p", "y", sample) for b, sample in enumerate(samples))
            scores = panel_batch_indexed({"p": prediction, "y": y}, jobs,
                                         stats="srocc", timeout=7200)
            by = {score["label"]: score["srocc"] for score in scores}
            if abs(by["POINT"] - cell["score"]["srocc"]) > 1e-12:
                raise ValueError(f"{path}: stored panel score differs")
            boot = np.asarray([by[f"B{b}"] for b in range(2000)], dtype=np.float64)
            rows.append({"source": str(path), "source_sha256": sha(path),
                         "point_srocc": float(by["POINT"]), "bootstrap": boot})
            print(json.dumps({"arm": arm, "rep": rep,
                              "srocc": by["POINT"]}), flush=True)
        values[arm] = rows
    points = {arm: float(np.mean([r["point_srocc"] for r in rows]))
              for arm, rows in values.items()}
    boots = {arm: np.asarray([r["bootstrap"] for r in rows], dtype=np.float64).mean(axis=0)
             for arm, rows in values.items()}
    result = {"schema": "rev4-featpot-c1c4-d2-mlp-compare-v1", "arm": args.arm,
              "heldout": args.heldout, "eval_set": eval_name, "hidden": args.hidden,
              "rows": len(y), "references": len(groups), "label_source": meta,
              "B": 2000, "seed": 20260923, "unit": "reference",
              "points_mean_srocc": points, "arms": {}, "deltas": {}}
    for arm, rows in values.items():
        finite = boots[arm][np.isfinite(boots[arm])]
        if len(finite) < 1900:
            raise ValueError(f"{arm}: only {len(finite)} finite bootstrap means")
        result["arms"][arm] = {"seed_srocc": [r["point_srocc"] for r in rows],
                               "seed_min_srocc": min(r["point_srocc"] for r in rows),
                               "seed_max_srocc": max(r["point_srocc"] for r in rows),
                               "ci95_mean_srocc": np.quantile(finite, [0.025, 0.975]).tolist(),
                               "source_sha256": [r["source_sha256"] for r in rows]}
    for arm in (args.arm, f"{args.arm}_perm"):
        delta = boots[arm] - boots["r0"]
        finite = delta[np.isfinite(delta)]
        if len(finite) < 1900:
            raise ValueError(f"{arm}: only {len(finite)} finite paired deltas")
        result["deltas"][arm] = {"point_delta_srocc": points[arm] - points["r0"],
                                 "ci95": np.quantile(finite, [0.025, 0.975]).tolist(),
                                 "bootstrap_finite": len(finite),
                                 "bootstrap_deltas": delta.tolist()}
    out = ROOT / "candidates/d2_mlp_compare" / f"LODO_{args.heldout}_{args.arm}_mlp{args.hidden}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result) + "\n")
    print(json.dumps({"result": str(out), "sha256": sha(out),
                      "candidate_delta": result["deltas"][args.arm]["point_delta_srocc"],
                      "candidate_ci95": result["deltas"][args.arm]["ci95"],
                      "perm_ci95": result["deltas"][f"{args.arm}_perm"]["ci95"]}), flush=True)


if __name__ == "__main__":
    main()
