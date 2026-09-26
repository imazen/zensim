"""Mechanical summary of the deterministic candidate/restore arm comparisons.

POTENTIAL - ceiling, not a model score. Reads only the `*_compare.json` files written by
candidate_compare.py (paired reference-clustered bootstrap, B=2000, seed 20260923) and
tabulates their numbers. It draws no verdict: a "flag" is an arithmetic condition on a
number (e.g. delta >= 0.005 with a CI lower bound > 0), never an adoption outcome. The D5
stability condition has not been run for these arms, so no flag here is an adoption bar.
`control_ci_excludes_zero` is the literal `permutation_instrument_sensitive` field: True means
the permuted control's own delta CI excludes zero, i.e. the control shows an effect.
"""

import csv
import glob
import json
from pathlib import Path

ROOT = Path("/var/tmp/rev4-featpot/candidates")
FULL = Path("/var/tmp/rev4-featpot/candidates/arms_summary_full_2026-09-26.tsv")
BRIEF = Path("benchmarks/rev4_featpot_arms_summary_2026-09-26.tsv")
TRAIN_ROLE = {"kadid_train", "tid2013", "konfig_train"}
COLS = ["scope", "model", "arm", "set", "rows", "refs", "srocc_r0", "srocc_arm", "srocc_perm",
        "delta", "delta_lo", "delta_hi", "perm_delta", "perm_lo", "perm_hi",
        "control_ci_excludes_zero", "gain_ge_0p005_ci_gt0"]


def rows_of(path: Path):
    doc = json.loads(path.read_text())
    scope = doc["scope"]
    items = ([(f"d1_{name}", p) for name, p in doc["phases"].items()] if scope == "d1"
             else [("d2", p) for p in doc["folds"].values()])
    for phase, p in items:
        d, c = p["deltas"]["candidate"], p["deltas"]["perm"]
        yield {"scope": phase, "model": doc["model"], "arm": doc["arm"],
               "set": p["set"], "rows": p["rows"], "refs": p["references"],
               "srocc_r0": p["point_srocc"]["p0"], "srocc_arm": p["point_srocc"]["candidate"],
               "srocc_perm": p["point_srocc"]["perm"], "delta": d["point_delta_srocc"],
               "delta_lo": d["ci95"][0], "delta_hi": d["ci95"][1],
               "perm_delta": c["point_delta_srocc"], "perm_lo": c["ci95"][0], "perm_hi": c["ci95"][1],
               "control_ci_excludes_zero": p["permutation_instrument_sensitive"],
               "gain_ge_0p005_ci_gt0": bool(d["point_delta_srocc"] >= 0.005 and d["ci95"][0] > 0)}


def main() -> None:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "d1/*_compare.json")) + glob.glob(str(ROOT / "d2/*_compare.json"))):
        rows.extend(rows_of(Path(path)))
    with FULL.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    brief = {}
    # D1 rows carry both nested and in_sample phases; keep nested only in the brief
    for r in rows:
        if r["scope"] == "d1_in_sample":
            continue
        kind = "d2_lodo" if r["scope"] == "d2" else "d1_nested"
        cell = brief.setdefault((r["model"], r["arm"], kind), {"n": 0, "gain": [], "control_fail": []})
        cell["n"] += 1
        if r["gain_ge_0p005_ci_gt0"]:
            cell["gain"].append(r["set"])
        if r["control_ci_excludes_zero"]:
            cell["control_fail"].append(r["set"])
    with BRIEF.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["model", "arm", "scope", "n_sets", "n_gain_ge_0p005_ci_gt0", "sets_with_gain",
                         "n_control_ci_excludes_zero", "sets_where_control_shows_effect"])
        for (model, arm, kind), cell in sorted(brief.items()):
            writer.writerow([model, arm, kind, cell["n"], len(cell["gain"]), ",".join(cell["gain"]),
                             len(cell["control_fail"]), ",".join(cell["control_fail"])])
    print(json.dumps({"full_rows": len(rows), "brief_rows": len(brief)}))


if __name__ == "__main__":
    main()
