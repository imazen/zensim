"""E33 registration section 6 / 10.7 positive control: the E33 assessment program reproduces V40's control predictions.

For each of the 40 frozen V40 control cells (pins `V40_CONTROL_PINS.json`), the E33 program's `bake_dial_refit`
densifies the frozen final119 bake and predicts the cell's held-out table in score units (`--score-units`, the units the
E33 assessment uses for all three arms; trainer bakes carry no spline, so these equal raw `pin - g`). Both the dense
bytes and the prediction file must equal V40's stored assessment artifacts byte for byte. The predictor reads features
only and computes no statistic; the held-out populations are the D1 sources already exposed by V40's assessment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25")
SPEC = "sel:59f0bbc2f290@h32:H128:cv16:cf98"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True, help="frozen V40 bundle")
    p.add_argument("--v40-results", type=Path, required=True, help="harvested V40 control results")
    p.add_argument("--fitbin", type=Path, required=True, help="E33 program bake_dial_refit")
    p.add_argument("--dest", type=Path, required=True)
    a = p.parse_args()
    a.dest.mkdir(parents=True, exist_ok=False)
    if sha(a.bundle / "V40_CONTROL_PINS.json") != "4d7acfc887603b123f9631f38435df5477b6e628a372596cb8beb6128bddc84c":
        raise ValueError("V40 control pins changed")
    pins = json.loads((a.bundle / "V40_CONTROL_PINS.json").read_text())["cells"]
    rows, mismatches = [], 0
    for fold in FOLDS:
        table = a.bundle / "v2e29/wide/main/real" / f"{fold}.parquet"
        for seed in range(10):
            key = f"{fold}_s{seed}"
            cell = a.v40_results / "cells" / f"{SPEC}__N" / f"without_{fold}_s{seed}"
            bake = cell / "refit/last.bin"
            if sha(bake) != pins[key]["bake_sha256"] or sha(cell / "result.json") != pins[key]["result_sha256"]:
                raise ValueError(f"{key}: V40 cell differs from the frozen pin")
            stored_dir = a.bundle / "assessment-e29/control" / key
            stored_dense = next((stored_dir / "dense").glob("*.bin"))
            stored_pred = stored_dir / "pred.tsv"
            out = a.dest / key
            out.mkdir()
            dense = out / "dense.bin"
            log = subprocess.run([str(a.fitbin), "densify", "--in", str(bake), "--out", str(dense)],
                                 capture_output=True, text=True, check=True)
            (out / "densify.log").write_text(log.stdout + log.stderr)
            pred = out / "pred.tsv"
            log = subprocess.run([str(a.fitbin), "predict", "--bake", str(dense), "--corpus", str(table),
                                  "--score-units", "--out", str(pred)], capture_output=True, text=True, check=True)
            (out / "pred.log").write_text(log.stdout + log.stderr)
            row = dict(cell=key, v40_bake_sha256=sha(bake), dense_sha256=sha(dense),
                       stored_dense_sha256=sha(stored_dense), dense_identical=dense.read_bytes() == stored_dense.read_bytes(),
                       pred_sha256=sha(pred), stored_pred_sha256=sha(stored_pred),
                       pred_identical=pred.read_bytes() == stored_pred.read_bytes(),
                       rows=sum(1 for _ in pred.open()) - 1)
            mismatches += not (row["dense_identical"] and row["pred_identical"])
            rows.append(row)
            print(json.dumps(row), flush=True)
    others = {}
    for study in ("assessment-e32", "assessment-e31"):
        same = sum(sha(a.bundle / study / "control" / r["cell"] / "pred.tsv") == r["stored_pred_sha256"] for r in rows)
        others[study] = f"{same}/{len(rows)} stored control predictions identical to assessment-e29"
    report = dict(schema="e33-v40-predictor-parity-v1", status="PASS" if mismatches == 0 and len(rows) == 40 else "MISMATCH",
                  cells=len(rows), mismatches=mismatches, units="score (--score-units)", fitbin_sha256=sha(a.fitbin),
                  v40_control_pins_sha256=sha(a.bundle / "V40_CONTROL_PINS.json"),
                  stored_source="assessment-e29/control (V40 assessment program)", cross_study_controls=others,
                  rows=rows)
    (a.dest / "PREDICTOR_PARITY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
