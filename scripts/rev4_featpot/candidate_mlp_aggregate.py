"""Pool registered C1-C4 MLP outer-fold predictions and report all five seeds.

Each reference appears in one outer fold. The panel owner scores pooled
out-of-fold predictions and the B=2,000 reference resamples; this driver
only assembles saved predictions from the admitted bank row order.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from linear_probe import ROOT, sha
from candidate_data import ARMS, load
from candidate_mlp import INIT_SEEDS, SAMPLE_SEEDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--arm", choices=[*ARMS, *(f"{name}_perm" for name in ARMS)], required=True)
    ap.add_argument("--hidden", type=int, choices=[32, 128], required=True)
    args = ap.parse_args()
    base = ROOT / "candidates/mlp" / f"POT_{args.set}_{args.arm}_mlp{args.hidden}"
    data, label_meta = load(args.set, args.arm, features=False)
    target = data.target.to_numpy(dtype=np.float64)
    refs = data.ref_basename.astype(str).to_numpy()
    groups = [np.flatnonzero(refs == ref) for ref in sorted(set(refs))]
    fold_manifest = json.loads((ROOT / "folds.json").read_text())["sets"][args.set]
    rng = np.random.default_rng(20260923)
    samples = [np.concatenate([groups[i] for i in rng.integers(
        0, len(groups), len(groups))]) for _ in range(2000)]
    reps = []
    fold_seed_scores = []
    bootstrap_by_rep = []
    for rep in range(5):
        pred = np.full(len(data), np.nan, dtype=np.float64)
        seen = np.zeros(len(data), dtype=bool)
        sources = []
        for outer in range(5):
            source = base / f"o{outer}_r{rep}" / "result.json"
            cell = json.loads(source.read_text())
            actual = (cell["set"], cell["arm"], cell["hidden"], cell["outer"], cell["rep"])
            if actual != (args.set, args.arm, args.hidden, outer, rep):
                raise ValueError(f"{source}: identity mismatch")
            if (cell["init_seed"] != INIT_SEEDS[rep] or
                    cell["sample_seed"] != SAMPLE_SEEDS[(rep + outer) % 5]):
                raise ValueError(f"{source}: Latin-square seeds mismatch")
            ii = np.asarray(cell["test_index"], dtype=int)
            pp = np.asarray(cell["prediction"], dtype=np.float64)
            if (len(ii) != len(pp) or len(ii) != cell["test_rows"]
                    or np.any(ii < 0) or np.any(ii >= len(data))
                    or len(set(ii)) != len(ii) or seen[ii].any()
                    or not np.isfinite(pp).all()):
                raise ValueError(f"{source}: invalid test rows or predictions")
            if set(refs[ii]) != set(fold_manifest["outer"][outer]["test_refs"]):
                raise ValueError(f"{source}: reference fold mismatch")
            pred[ii] = pp
            seen[ii] = True
            fold_seed_scores.append({"outer": outer, "rep": rep,
                                     "init_seed": INIT_SEEDS[rep],
                                     "sample_seed": SAMPLE_SEEDS[(rep + outer) % 5],
                                     "srocc": cell["score"]["srocc"]})
            sources.append({"path": str(source), "sha256": sha(source)})
        if not seen.all() or not np.isfinite(pred).all():
            raise ValueError(f"rep {rep}: outer folds do not partition all rows")
        full_source = base / f"full_r{rep}" / "result.json"
        full = json.loads(full_source.read_text())
        full_actual = (full["set"], full["arm"], full["hidden"], full["outer"], full["rep"])
        if full_actual != (args.set, args.arm, args.hidden, None, rep):
            raise ValueError(f"{full_source}: full result identity mismatch")
        if full["init_seed"] != INIT_SEEDS[rep] or full["sample_seed"] != SAMPLE_SEEDS[rep]:
            raise ValueError(f"{full_source}: full seeds mismatch")
        jobs = [("POINT", "p", "y", None)]
        jobs.extend((f"B{b}", "p", "y", ii) for b, ii in enumerate(samples))
        rows = panel_batch_indexed({"p": pred, "y": target}, jobs,
                                   stats="srocc", timeout=7200)
        by = {row["label"]: row["srocc"] for row in rows}
        boot = np.asarray([by[f"B{b}"] for b in range(2000)], dtype=np.float64)
        finite = boot[np.isfinite(boot)]
        if len(finite) < 1900:
            raise ValueError(f"rep {rep}: only {len(finite)} finite bootstraps")
        point = float(by["POINT"])
        full_srocc = float(full["score"]["srocc"])
        reps.append({"rep": rep, "init_seed": INIT_SEEDS[rep],
                     "outer_sample_seeds": [SAMPLE_SEEDS[(rep + o) % 5] for o in range(5)],
                     "nested_srocc": point,
                     "nested_ci95_srocc": np.quantile(finite, [0.025, 0.975]).tolist(),
                     "in_sample_srocc": full_srocc,
                     "gap_srocc": full_srocc - point,
                     "bootstrap_srocc": boot.tolist(),
                     "outer_sources": sources,
                     "full_source": {"path": str(full_source), "sha256": sha(full_source)}})
        bootstrap_by_rep.append(boot)
        print(json.dumps({"rep": rep, "nested_srocc": point,
                          "in_sample_srocc": full_srocc,
                          "gap_srocc": full_srocc - point}), flush=True)
    seed_scores = np.asarray([row["nested_srocc"] for row in reps])
    full_scores = np.asarray([row["in_sample_srocc"] for row in reps])
    gap_scores = full_scores - seed_scores
    boot_array = np.asarray(bootstrap_by_rep)
    finite_draws = np.isfinite(boot_array).all(axis=0)
    if finite_draws.sum() < 1900:
        raise ValueError("fewer than 1900 paired finite mean-seed bootstraps")
    mean_boot = boot_array[:, finite_draws].mean(axis=0)
    init_means = {str(seed): float(np.mean([x["srocc"] for x in fold_seed_scores
                                            if x["init_seed"] == seed]))
                  for seed in INIT_SEEDS}
    sample_means = {str(seed): float(np.mean([x["srocc"] for x in fold_seed_scores
                                              if x["sample_seed"] == seed]))
                    for seed in SAMPLE_SEEDS}
    output = {"schema": "rev4-featpot-c1c4-mlp-pooled-v1",
              "label": "POTENTIAL — ceiling, not a model score",
              "set": args.set, "arm": args.arm, "hidden": args.hidden,
              "rows": len(data), "references": len(groups),
              "label_source": label_meta,
              "fold_manifest_sha256": sha(ROOT / "folds.json"),
              "B": 2000, "seed": 20260923, "unit": "reference",
              "replicates": reps,
              "nested_mean_srocc": float(np.mean(seed_scores)),
              "nested_min_srocc": float(np.min(seed_scores)),
              "nested_max_srocc": float(np.max(seed_scores)),
              "nested_mean_ci95_srocc": np.quantile(
                  mean_boot, [0.025, 0.975]).tolist(),
              "nested_mean_bootstrap_finite": int(finite_draws.sum()),
              "in_sample_mean_srocc": float(np.mean(full_scores)),
              "in_sample_min_srocc": float(np.min(full_scores)),
              "in_sample_max_srocc": float(np.max(full_scores)),
              "gap_mean_srocc": float(np.mean(gap_scores)),
              "gap_min_srocc": float(np.min(gap_scores)),
              "gap_max_srocc": float(np.max(gap_scores)),
              "fold_srocc_by_seed": fold_seed_scores,
              "fold_mean_srocc_by_init_seed": init_means,
              "fold_mean_srocc_by_sample_seed": sample_means,
              "init_spread_fold_mean_srocc": max(init_means.values()) - min(init_means.values()),
              "sample_spread_fold_mean_srocc": max(sample_means.values()) - min(sample_means.values())}
    dest = ROOT / "candidates/mlp_aggregate" / f"POT_{args.set}_{args.arm}_mlp{args.hidden}"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "result.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(path), "sha256": sha(path),
                      "rows": len(data), "references": len(groups),
                      "nested_mean_srocc": output["nested_mean_srocc"],
                      "nested_mean_ci95_srocc": output["nested_mean_ci95_srocc"]}),
          flush=True)


if __name__ == "__main__":
    main()
