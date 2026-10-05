"""Design log E24 (registered 2026-10-04 12:05 UTC, before any Rev5 cell or table exists): is by_v2fy at formula Rev5 as
good as by_v2fy at Rev4, both at the adopted recipe cv16:cf98?

  python3 e24_rev5.py grid  --root REV5_ROOT --out SPEC.json --program-sha P --data-sha D   # 2 arms x 5 folds x seeds 0-9
  python3 e24_rev5.py score --root REV5_ROOT --rev4-root REV4_ROOT                          # -> REV5_ROOT/compare/e24_rev5.json
  python3 e24_rev5.py e25   --root REV5_ROOT --rev4-root REV4_ROOT --out E25_ROOT            # Addendum D, exact gate first

Spec: `benchmarks/rev5_spec_2026-10-04.md` (Rev5 = f32 local-window blur, 16-lane canon, FMA, stable central moments, exact
identity zeros; basic + peaks + v2 only). The Rev5 instrument root is built from the Rev5 bank (`rev5_bank.py`,
`v2c_wide.py --revision 5`) with the same keys, labels, legs, coverage pool pairs and seeds as the Rev4 root, so a Rev5 cell and
the Rev4 cell with the same (held-out source, seed) differ only in feature arithmetic.

Arms (Rev5 root): by_v2fy (`sel:59f0bbc2f290`, 420 columns) and v2 + basic, both `@h32:H128:cv16:cf98`, head N, seeds 0-9.
Control: by_v2fy at Rev4 (the Rev4 root's E21 cells, seeds 0-9) — the set the R7 sealed read found as good as v2 + basic.
Primary (by_v2fy@Rev5 vs control), seed-paired over seeds 0-9: E12's "as good" rule (mean signed Δ >= -0.002 and every
source >= -0.005) AND the worst-three-types W2 Δ not worse than -2 SE. Secondary (reported, not gating): v2 + basic@Rev5 vs the
same control; per-source Δ, W1; the external NITS / LIVE / MCIQA Δ; steering panels and CHROMAQ on the Rev5 bakes (separate
records). Exposure: exploratory sources only (as E21/E23). Confirmation of a Rev5 model needs a new holdout (spec §6).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import e21_cheap_recipe as e21
from v2_common import FITBIN, PANEL, SOURCE_ORDER, V2, dense_bake, sha, table_revision

RECIPE = e21.RECIPE
HEAD, SEEDS = "N", range(10)
ARMS = {"rev5_by_v2fy": "by_v2fy", "rev5_v2basic": None}
CONTROL = e21.spec("by_v2fy")          # sel:59f0bbc2f290@h32:H128:cv16:cf98, read from the Rev4 root
PREFIX = "rev5::"                      # routes an arm's cells to the Rev5 root inside the shared scorer


def arm_spec(label: str) -> str:
    base = ARMS[label]
    return e21.spec(base) if base else f"set:v2+basic@h32{RECIPE}"


def arm_columns(label: str) -> list | None:
    base = ARMS[label]
    return e21.columns(base) if base else None


def cmd_grid(args) -> int:
    cells = []
    for a in ARMS:
        cols = arm_columns(a)
        for s in SOURCE_ORDER:
            for i in SEEDS:
                argv = ["v2_lodo_mlp.py", "--spec", arm_spec(a), "--head", HEAD, "--heldout", s, "--seed-index", str(i),
                        "--root", str(V2)]
                if cols:
                    argv += ["--columns", ",".join(map(str, cols))]
                cells.append({"name": f"{arm_spec(a)}__{HEAD}/without_{s}_s{i}", "argv": argv})
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "specs": {a: arm_spec(a) for a in ARMS}}))
    return 0


def same_heldout_keys(rev4: Path, rev5: Path) -> None:
    """The held-out tables of both roots must be the same keys in the same order, or no pairing is valid."""
    for s in SOURCE_ORDER:
        a = pq.read_table(rev4 / "wide" / "main" / "real" / f"{s}.keys.parquet").to_pandas()
        b = pq.read_table(rev5 / "wide" / "main" / "real" / f"{s}.keys.parquet").to_pandas()
        if not a.equals(b):
            raise ValueError(f"{s}: held-out keys differ between the Rev4 and Rev5 roots")


def cmd_score(args) -> int:
    import e13_teacher as e13
    rev5, rev4 = Path(V2), Path(args.rev4_root)
    same_heldout_keys(rev4, rev5)

    def cell_of(spec_: str, source: str, seed: int) -> Path:
        root, s = (rev5, spec_[len(PREFIX):]) if spec_.startswith(PREFIX) else (rev4, spec_)
        return root / "cells" / f"{s}__{HEAD}" / f"without_{source}_s{seed}" / "result.json"

    e13.cell_of, e13.V2, e13.SEEDS, e13.BASE = cell_of, rev4, SEEDS, CONTROL
    e13.MEAN_FLOOR, e13.SOURCE_FLOOR = e21.MEAN_FLOOR, e21.SOURCE_FLOOR
    rc = e13.score_arms([(a, PREFIX + arm_spec(a)) for a in ARMS], "e24_rev5", False)
    src = rev4 / "compare" / "e24_rev5.json"
    rows = json.loads(src.read_text())
    (rev5 / "compare").mkdir(exist_ok=True)
    (rev5 / "compare" / "e24_rev5.json").write_text(json.dumps({**rows, "rev4_root": str(rev4), "rev5_root": str(rev5)},
                                                                indent=1) + "\n")
    src.unlink()
    decision = {}
    for a, row in rows["rows"].items():
        if "signed" not in row:
            decision[a] = {"status": "INCOMPLETE"}
            continue
        decision[a] = as_good(row, arm_spec(a))
    (rev5 / "compare" / "e24_decision.json").write_text(json.dumps({"control": CONTROL, "control_root": str(rev4),
                                                                    "arms": decision}, indent=1) + "\n")
    print(json.dumps(decision))
    return rc


def as_good(row: dict, spec_: str) -> dict:
    """E21's unchanged rule, shared by E24 and registered E25 (not E13's adopt flag)."""
    sig, w2 = row["signed"], row["w2_type_worst3"]
    return {"spec": spec_, "signed_mean": sig["mean"], "signed_se": sig["se"], "signed_worst": sig["worst"],
            "per_source": {s: row["per_source"][s]["signed"] for s in SOURCE_ORDER}, "w2": w2["mean"],
            "w2_se": w2["se"], "as_good": bool(sig["mean"] >= e21.MEAN_FLOOR and sig["worst"] >= e21.SOURCE_FLOOR
                                              and w2["mean"] > -2 * w2["se"])}


def cmd_e25(args) -> int:
    """Addendum D: frozen Rev4 weights on Rev5 tables, after all 50 exact Rev4 gates.

    Prediction and full held-out metrics call v2_lodo_mlp's own functions. W1/W2,
    seed pairing and external reads retain their existing owners. No training or
    production arithmetic changes, no new scorer, no Rev5 reads before the gate.
    """
    import v2_lodo_mlp as trainer

    rev4, rev5 = Path(args.rev4_root), Path(V2)
    out = Path(args.out) if args.out else rev5 / "e25"
    if out.resolve() in (rev4.resolve(), rev5.resolve()):
        raise ValueError("E25 needs its own output root")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("E25 requires a new empty output root; preserve earlier evidence")
    os.environ["ZEN_PANEL_BIN"] = str(PANEL)
    tools = {str(p): sha(p) for p in (FITBIN, PANEL)}
    r4path = rev4 / "wide/main/real/receipt.json"
    r4receipt = json.loads(r4path.read_text())
    jobs = [(s, i) for s in SOURCE_ORDER for i in SEEDS]

    def checked_input(root: Path, receipt: dict, source: str):
        record = receipt["legs"][source]
        table = root / record["full"]["rel"]
        keys = root / "wide/main/real" / f"{source}.keys.parquet"
        for p, want in [(table, record["full"]["sha256"]),
                        (Path(f"{table}.manifest.json"), record["full"]["manifest_sha256"]),
                        (keys, record["keys_sha256"])]:
            if sha(p) != want:
                raise ValueError(f"{p}: changed after its receipt")
        return table, pq.read_table(keys).to_pandas()

    # Only Rev4 inputs/results are opened in this phase. Materialize dense models
    # for both phases so the gate tests exactly the E25 prediction path.
    r4inputs = {s: checked_input(rev4, r4receipt, s) for s in SOURCE_ORDER}

    def gate_one(job):
        s, i = job
        cell = rev4 / "cells" / f"{CONTROL}__{HEAD}" / f"without_{s}_s{i}"
        result_path = cell / "result.json"
        result = json.loads(result_path.read_text())
        if (result["spec"], result["head"], result["heldout"], result["seed_index"]) != (CONTROL, HEAD, s, i):
            raise ValueError(f"{cell}: control identity mismatch")
        bake = Path(result["selected_bake"])
        if sha(bake) != result["selected_bake_sha256"]:
            raise ValueError(f"{bake}: selected weights changed")
        keep = cell / "keep_features.txt"
        if [int(x) for x in keep.read_text().split()] != e21.columns("by_v2fy"):
            raise ValueError(f"{cell}: unexpected kept feature IDs")
        table, keys = r4inputs[s]
        if result["keys_sha256"] != r4receipt["legs"][s]["keys_sha256"]:
            raise ValueError(f"{cell}: held-out keys mismatch")
        dest = out / "gate" / f"without_{s}_s{i}"
        dest.mkdir(parents=True, exist_ok=True)
        dense = dense_bake(bake, out / "models")
        pred = trainer.predict(dense, table, dest / "eval_preds.tsv")
        if not np.array_equal(pred, np.asarray(result["prediction"], dtype=np.float64)):
            raise ValueError(f"{cell}: Rev4 prediction gate FAILED (no E25 read)")
        score = trainer.panel_batch([(s, pred, keys.target.to_numpy(dtype=np.float64))], stats="full")[0]
        if score != result["score"]:
            raise ValueError(f"{cell}: exact Rev4 metric gate FAILED (no E25 read): {score} != {result['score']}")
        gate = {"source": s, "seed": i, "result": str(result_path), "result_sha256": sha(result_path),
                "source_bake": str(bake), "source_bake_sha256": sha(bake), "dense_bake": str(dense),
                "keep_features": str(keep), "keep_features_sha256": sha(keep),
                "dense_bake_sha256": sha(dense), "table_sha256": sha(table),
                "prediction_sha256": sha(dest / "eval_preds.tsv"), "stored_binaries": result["binaries"],
                "score": score, "predictions_exact": True, "metrics_exact": True}
        (dest / "gate.json").write_text(json.dumps(gate, indent=1) + "\n")
        print(f"Rev4 exact gate: {s} s{i}", flush=True)
        return gate

    with ThreadPoolExecutor(args.jobs) as executor:
        gates = list(executor.map(gate_one, jobs))
    if len(gates) != 50:
        raise ValueError("incomplete Rev4 gate")
    gate_path = out / "rev4_gate.json"
    gate_path.write_text(json.dumps({"schema": "featpot-e25-rev4-gate-v1", "status": "PASS", "cells": gates,
                                    "tools": tools, "rev4_receipt_sha256": sha(r4path),
                                    "e25_numbers_read": False}, indent=1) + "\n")
    print("PASS: all 50 Rev4 predictions and full metric dictionaries exactly reproduced; E25 reads now admitted", flush=True)

    # No Rev5 input or E25 output is read before the complete gate above.
    same_heldout_keys(rev4, rev5)
    r5path = rev5 / "wide/main/real/receipt.json"
    r5receipt = json.loads(r5path.read_text())
    r5inputs = {s: checked_input(rev5, r5receipt, s) for s in SOURCE_ORDER}
    if any(table_revision(p) != 5 for p, _ in r5inputs.values()):
        raise ValueError("E25 needs Rev5 held-out tables")
    gate_by_job = {(g["source"], g["seed"]): g for g in gates}

    def arm_one(job):
        s, i = job
        g = gate_by_job[job]
        original = json.loads(Path(g["result"]).read_text())
        if sha(Path(g["result"])) != g["result_sha256"] or sha(Path(g["dense_bake"])) != g["dense_bake_sha256"]:
            raise ValueError("input changed since the exact gate")
        table, keys = r5inputs[s]
        dest = out / "cells" / f"{CONTROL}__{HEAD}" / f"without_{s}_s{i}"
        dest.mkdir(parents=True, exist_ok=True)
        keep = Path(g["keep_features"])
        if sha(keep) != g["keep_features_sha256"]:
            raise ValueError("kept feature IDs changed since the exact gate")
        (dest / "keep_features.txt").write_bytes(keep.read_bytes())
        # The trainer's Rev5 branch densifies its source bake. Densify refuses
        # already-dense input, so pass the original and verify the generated
        # dense bytes against the exact-gate model before prediction.
        source_bake = Path(g["source_bake"])
        if sha(source_bake) != g["source_bake_sha256"]:
            raise ValueError("selected weights changed since the exact gate")
        dense = dense_bake(source_bake, dest)
        if sha(dense) != g["dense_bake_sha256"]:
            raise ValueError("Rev5 path changed the gated dense model")
        pred = trainer.predict(source_bake, table, dest / "eval_preds.tsv")
        score = trainer.panel_batch([(s, pred, keys.target.to_numpy(dtype=np.float64))], stats="full")[0]
        result = {**original, "label": "E25 POTENTIAL — Rev4-trained weights on Rev5 held-out features",
                  "prediction": pred.tolist(), "score": score, "wide_receipt_sha256": sha(r5path),
                  "table_receipt_sha256": sha(r5path), "e25": {"rev4_result_sha256": g["result_sha256"],
                  "rev4_gate_sha256": sha(gate_path), "rev5_table_sha256": sha(table), "tools": tools,
                  "dense_bake": str(dense), "dense_bake_sha256": sha(dense)}}
        (dest / "result.json").write_text(json.dumps(result) + "\n")
        print(f"E25 predicted: {s} s{i}", flush=True)
        return {"source": s, "seed": i, "result_sha256": sha(dest / "result.json")}

    with ThreadPoolExecutor(args.jobs) as executor:
        predictions = list(executor.map(arm_one, jobs))

    def link(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            if dst.resolve() != src.resolve():
                raise ValueError(f"{dst}: wrong input link")
        elif dst.exists():
            raise ValueError(f"{dst}: expected an immutable input link")
        else:
            dst.symlink_to(src.resolve())

    link(rev5 / "wide/main/real", out / "wide/main/real")
    import e13_teacher as e13
    e13.V2, e13.SEEDS = out, SEEDS
    e13.MEAN_FLOOR, e13.SOURCE_FLOOR = e21.MEAN_FLOOR, e21.SOURCE_FLOOR
    label, arm = "rev4_weights_rev5", "e25::" + CONTROL

    def compare(control_root: Path, name: str):
        def cell_of(sp, s, i):
            root, spec_ = (out, CONTROL) if sp == arm else (control_root, CONTROL)
            return root / "cells" / f"{spec_}__{HEAD}" / f"without_{s}_s{i}" / "result.json"
        e13.cell_of, e13.BASE = cell_of, CONTROL
        e13.score_arms([(label, arm)], name, False)
        report = json.loads((out / "compare" / f"{name}.json").read_text())
        if report["status"] != "complete" or report["missing_cells"]:
            raise ValueError(f"{name}: incomplete comparison")
        return report["rows"][label]

    primary = compare(rev4, "e25_primary")
    secondary = compare(rev5, "e25_vs_retrained")
    # Use the existing external owner with a model/table view, not a second
    # external scorer. The view has Rev4 weights and Rev5 external tables.
    import external_sets as external
    for st in external.PAIRS:
        for suffix in (".parquet", ".parquet.manifest.json", ".keys.parquet"):
            link(rev5 / "external" / f"{st}{suffix}", out / "external" / f"{st}{suffix}")
    external.V2 = out
    external.cmd_score(argparse.Namespace(specs=CONTROL + "," + CONTROL, seeds="0-9", jobs=args.jobs,
                                         control_root=str(rev4), out="e25_external"))
    decision = {"schema": "featpot-e25-decision-v1", "registration": "rev5_spec_2026-10-04.md Addendum D / 399acb59",
                "status": "COMPLETE_UNQUALIFIED", "rev4_root": str(rev4), "rev5_root": str(rev5),
                "rev4_gate_sha256": sha(gate_path), "prediction_cells": predictions,
                "primary": as_good(primary, CONTROL), "vs_retrained_reported_only": secondary,
                "external_reported_only": str(out / "compare/e25_external.json")}
    (out / "compare/e25_decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    print(json.dumps(decision["primary"]), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "score", "e25"])
    ap.add_argument("--root")
    ap.add_argument("--rev4-root", default="/var/tmp/rev4-featpot/v2c")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    return {"grid": cmd_grid, "score": cmd_score, "e25": cmd_e25}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
