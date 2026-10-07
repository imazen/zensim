"""E30 adapters: 40 nA3 cells; reuse E24 control, no AIC fold or label.

Training stays in the strict owner. Post-harvest scoring calls the existing E21
statistics owner; E30 reports removal cost and cannot reverse D1.
"""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path

from v2_human_role import PRODUCTION_SOURCES, preflight_recipe
from v2_common import sha

SPEC = "sel:59f0bbc2f290@h32:H128:cv16:cf98"


def completed_control_pins(results, bundle):
    """Freeze E30's exact 40 nA3 cells, without discovering historical cells.

    This is the control for E31/E32, distinct from E30's E24 comparison.
    The canonical model inspector verifies each final checkpoint's embedded
    receipt. No assessment, table payload load, or outcome selection occurs.
    """
    manifest_path = bundle / "fit-manifest-fitv2e30-20261007.json"
    jobs = json.loads(manifest_path.read_text())
    contract_name = "benchmarks/shippath_qualified_fit_contract_2026-10-07.json"
    inventory = json.loads((bundle / "PROGRAM_INVENTORY.json").read_text())
    artifacts = json.loads((bundle / "ARTIFACT_PINS.json").read_text())
    expected = {f"{SPEC}__N/without_{source}_s{seed}" for source in PRODUCTION_SOURCES for seed in range(10)}
    names = [job["cell"]["image_path"] for job in jobs]
    if len(names) != 40 or set(names) != expected:
        raise ValueError("completed E30 control requires exactly the four-source 40-cell manifest")
    by_name = dict(zip(names, jobs))
    program = bundle / "image-context/program.tar.gz"
    if sha(program) != artifacts["program_sha"] or sha(bundle / "d1-fit-data.tar.gz") != artifacts["data_sha"]:
        raise ValueError("E30 program/data archive changed")
    with tarfile.open(program) as archive:
        contract = json.load(archive.extractfile(contract_name))
        for name, digest in inventory.items():
            import hashlib
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != digest:
                raise ValueError(f"E30 packed program inventory changed: {name}")
    inspector = bundle / "bin/inspect_qualified_checkpoint"
    if sha(inspector) != inventory["bin/inspect_qualified_checkpoint"]:
        raise ValueError("E30 canonical inspector changed")
    pins = {}
    for source in PRODUCTION_SOURCES:
        for seed in range(10):
            name = f"{SPEC}__N/without_{source}_s{seed}"
            cell = results / name
            record = json.loads((cell / "result.json").read_text())
            fleet = json.loads((cell / "fleet_receipt.json").read_text())
            selection = record["selection"]
            if (record["schema"] != "rev5-qualified-training-cell-v1"
                    or record["execution_contract"] != "registered-fit" or record["training_only"] is not True
                    or record["spec"] != SPEC or record["head"] != "N" or record["heldout"] != source
                    or record["seed_index"] != seed or record["epochs"] != 120 or record["pairs_per_epoch"] != 50000
                    or selection["epoch_rule"] != "last" or selection["selected_epoch"] != 119
                    or record["human_sources"] != list(PRODUCTION_SOURCES)):
                raise ValueError(f"E30 control identity/budget differs: {name}")
            for key in ("wide_receipt_sha256", "frozen_sha256", "data_role_decision_sha256"):
                if record[key] != contract[key]:
                    raise ValueError(f"E30 control {key} differs: {name}")
            admission = selection["strict_table_admission"]
            if sorted(admission, key=lambda r: r["name"]) != contract["routes"][source]:
                raise ValueError(f"E30 control fit-table receipts differ: {name}")
            columns = [int(x) for x in (cell / "keep_features.txt").read_text().split()]
            if columns != contract["columns"] or record["kept_features"] != len(columns):
                raise ValueError(f"E30 control consumed feature order differs: {name}")
            job = by_name[name]
            kind = job["kind"]
            if (fleet["cell"] != name or fleet["program_sha"] != artifacts["program_sha"]
                    or fleet["data_sha"] != artifacts["data_sha"] or fleet["argv_sha"] != kind["argv_sha"]
                    or kind["program_sha"] != fleet["program_sha"] or kind["data_sha"] != fleet["data_sha"]
                    or fleet["tier"]["effective"] != "v3"):
                raise ValueError(f"E30 fleet execution identity differs: {name}")
            for rel, digest in fleet["files"].items():
                if Path(rel).is_absolute() or ".." in Path(rel).parts:
                    raise ValueError("unsafe E30 fleet result member")
                if sha(cell / rel) != digest:
                    raise ValueError(f"E30 result bytes changed: {name}/{rel}")
            bake = cell / "refit/last.bin"
            if (sha(bake) != record["selected_bake_sha256"] or sha(bake) != fleet["selected_bake_sha"]
                    or sha(cell / "result.json") != fleet["result_sha"]):
                raise ValueError(f"E30 selected result/model binding changed: {name}")
            inspected = json.loads(subprocess.run([str(inspector), str(bake)], check=True, text=True,
                                                  capture_output=True).stdout)
            repro = inspected["repro"]
            if (int(repro["checkpoint_epoch"]) != 119 or repro["epochs"] != 120
                    or repro["pairs_per_epoch"] != 50000 or repro["pair_sampling"] != "uniform"
                    or repro["init_seed"] != record["init_seed"] or repro["sample_seed"] != record["sample_seed"]
                    or repro["init_seed"] != contract["init_seeds"][seed]
                    or repro["sample_seed"] != contract["sample_seeds"][(seed + contract["fold_order"].index(source)) % 10]
                    or repro["keep_features_n"] != 420 or repro["max_features"] != record["width"]
                    or repro["effective_minibatch"] != 1):
                raise ValueError(f"E30 embedded training contract differs: {name}")
            admitted_by_name = {r["name"]: r["table_sha256"] for r in admission}
            if {r["name"]: r["sha256"] for r in repro["inputs"]} != admitted_by_name:
                raise ValueError(f"E30 embedded input bytes differ: {name}")
            pins[f"{source}_s{seed}"] = {
                "result_sha256": sha(cell / "result.json"), "selected_bake_sha256": sha(bake),
                "feature_list_sha256": sha(cell / "keep_features.txt"), "feature_ids": columns,
                "fleet_receipt_sha256": sha(cell / "fleet_receipt.json"), "files": fleet["files"],
                "job": job, "fit_receipt": record, "checkpoint_receipt": inspected,
            }
    return {"schema": "e30-completed-control-freeze-v1", "cells": pins, "cell_count": 40,
            "manifest_sha256": sha(manifest_path), "program_sha256": sha(program),
            "data_sha256": artifacts["data_sha"], "program_inventory": inventory,
            "fit_contract_sha256": inventory[contract_name]}


def grid(root, results, columns, production=False):
    cells = []
    for heldout in ([None] if production else PRODUCTION_SOURCES):
        for seed in (range(3) if production else range(10)):
            name = f"{SPEC}__N/" + (f"full_s{seed}" if production else f"without_{heldout}_s{seed}")
            argv = ["v2_confirm_fit.py" if production else "v2_lodo_mlp.py", "--spec", SPEC, "--head", "N",
                    "--seed-index", str(seed), "--root", str(root), "--columns", ",".join(map(str, columns)),
                    "--strict-admission", "--train-only", "--data-role-decision", str(root / "human_role_decision.json"),
                    "--dest", str(results / name)]
            argv += ["--pack-production"] if production else ["--heldout", heldout]
            cells.append({"name": name, "argv": argv})
    return cells


def control_pins(control):
    """Bind only the 40 registered E24 control cells; never enumerate AIC cells."""
    pins = {}
    for s in PRODUCTION_SOURCES:
        for seed in range(10):
            cell = control / "cells" / f"{SPEC}__N/without_{s}_s{seed}"
            record = json.loads((cell / "result.json").read_text())
            bake = cell / "refit/last.bin"
            if (record["spec"] != SPEC or record["head"] != "N" or record["heldout"] != s
                    or record["seed_index"] != seed or record["selected_epoch"] != 119
                    or sha(bake) != record["selected_bake_sha256"]):
                raise ValueError("E30 reused E24 control differs from its registered cell")
            pins[f"{s}_s{seed}"] = {"result_sha256": sha(cell / "result.json"), "bake_sha256": sha(bake)}
    return pins


def score(root, results, control, out, pins):
    """Explicit post-harvest assessment; read four released populations only."""
    import numpy as np
    import pyarrow.parquet as pq
    import e13_teacher as e13
    from v2_lodo_mlp import predict, strict_training_groups
    from lib.zen_stats import panel_batch
    from v2_common import admission_input_roots, refuse_immutable_output
    refuse_immutable_output(out, (*admission_input_roots(root), control))
    if control_pins(control) != json.loads(pins.read_text()):
        raise ValueError("E30 control pins changed")
    preflight_recipe(root, root / "human_role_decision.json")
    for source in PRODUCTION_SOURCES:
        for seed in range(10):
            cell = results / f"{SPEC}__N/without_{source}_s{seed}"
            r = json.loads((cell / "result.json").read_text())
            if r["selection"]["selected_epoch"] != 119 or not r["selection"]["strict_table_admission"]:
                raise ValueError("E30 assessment requires completed strict final-epoch fits")
            bake = cell / "refit/last.bin"
            if sha(bake) != r["selected_bake_sha256"]:
                raise ValueError("E30 selected model changed")
            table = root / f"wide/main/real/{source}.parquet"
            strict_training_groups([(source, table, 0., 0., "withinref,rank")], root / "human_role_decision.json")
            keys = pq.read_table(table.with_suffix(".keys.parquet"))
            oldkeys = pq.read_table(control / f"wide/main/real/{source}.keys.parquet")
            if any(keys[c].to_pylist() != oldkeys[c].to_pylist() for c in keys.column_names):
                raise ValueError("E30 control/arm row identity differs")
            # Existing owner emits dense predictions; no AIC table can reach it.
            dst = out / "cells" / f"{SPEC}__N/without_{source}_s{seed}"
            dst.mkdir(parents=True)
            pred = predict(bake, table, dst / "eval_preds.tsv")
            e13.V2 = control
            released = e13.heldout_meta(source)
            stats = panel_batch([(source, pred, released.target.to_numpy(dtype=np.float64))], stats="full")[0]
            (dst / "result.json").write_text(json.dumps({**r, "prediction": np.asarray(pred).tolist(), "score": stats}) + "\n")
    original_meta = e13.heldout_meta
    def meta(source):
        if source not in PRODUCTION_SOURCES:
            raise ValueError("E30 forbids AIC fold metadata")
        previous = e13.V2
        try:
            e13.V2 = control
            return original_meta(source)
        finally:
            e13.V2 = previous
    e13.heldout_meta = meta
    e13.SOURCE_ORDER, e13.SEEDS, e13.BASE, e13.V2 = PRODUCTION_SOURCES, range(10), SPEC, out
    e13.cell_of = lambda spec, s, i: (control if spec == SPEC else out) / "cells" / f"{SPEC}__N/without_{s}_s{i}/result.json"
    e13.score_arms([("nA3", "nA3")], "e30_four_source", False)
    path = out / "compare/e30_four_source.json"
    report = json.loads(path.read_text())
    report.pop("rule")
    for row in report["rows"].values():
        row.pop("adopt", None)
    report["scope"] = "Report removal cost only; D1 fixed; no adoption rule; no AIC fold"
    path.write_text(json.dumps(report, indent=2) + "\n")


def external_score(root, results, control, out, pins):
    """Explicit later NITS/LIVE/MCIQA read via the existing external owner."""
    import shutil
    import external_sets as external
    from e24_rev5 import e26_bound_bake
    from v2_common import admission_input_roots, refuse_immutable_output
    if control_pins(control) != json.loads(pins.read_text()):
        raise ValueError("E30 control pins changed")
    refuse_immutable_output(out, (*admission_input_roots(root), control))
    preflight_recipe(root, root / "human_role_decision.json")
    if out.exists():
        raise ValueError("E30 external assessment output must be fresh")
    (out / "external").mkdir(parents=True)
    for name in ("nits", "live", "mciqa"):
        for suffix in (".parquet", ".keys.parquet", ".parquet.manifest.json"):
            src = control / f"external/{name}{suffix}"
            # Only these three named evaluation tables, never confirmation/T0.
            from lib.assessment_identity import safe_path
            shutil.copyfile(safe_path(src), out / "external" / src.name)
    for source in PRODUCTION_SOURCES:
        for seed in range(10):
            name = f"without_{source}_s{seed}"
            for label, srcroot in (("control", control / "cells"), ("nA3", results)):
                src = srcroot / f"{SPEC}__N" / name
                record = json.loads((src / "result.json").read_text())
                if label == "control":
                    model, _ = e26_bound_bake(src, out / "bindings")
                else:
                    model = src / "refit/last.bin"
                    if record["selection"]["selected_epoch"] != 119 or sha(model) != record["selected_bake_sha256"]:
                        raise ValueError("E30 external read requires strict final epoch")
                dst = out / "cells" / f"{label}__N" / name
                dst.mkdir(parents=True)
                (dst / "result.json").write_text(json.dumps({**record, "selected_bake": str(model)}) + "\n")
                shutil.copyfile(src / "keep_features.txt", dst / "keep_features.txt")
    external.V2, external.SOURCE_ORDER = out, PRODUCTION_SOURCES
    external.cmd_score(argparse.Namespace(specs="control,nA3", seeds="0-9", control_root=None, jobs=1, out="e30_external"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("grid", "production-grid", "control-pins", "completed-control-pins", "score", "external-score"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--results", type=Path)
    p.add_argument("--control-root", type=Path)
    p.add_argument("--control-pins", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--program-sha")
    p.add_argument("--data-sha")
    p.add_argument("--bundle", type=Path)
    a = p.parse_args()
    if a.cmd == "completed-control-pins":
        # Exclusive creation keeps a prior freeze immutable.
        frozen = completed_control_pins(a.results, a.bundle)
        with a.out.open("x") as output:
            output.write(json.dumps(frozen, indent=2) + "\n")
    elif a.cmd == "control-pins":
        a.out.write_text(json.dumps(control_pins(a.control_root), indent=2) + "\n")
    elif a.cmd in ("score", "external-score"):
        (score if a.cmd == "score" else external_score)(a.root, a.results, a.control_root, a.out, a.control_pins)
    else:
        import e21_cheap_recipe as e21
        cells = grid(a.root, a.results, e21.columns("by_v2fy"), a.cmd == "production-grid")
        a.out.write_text(json.dumps({"program_sha": a.program_sha, "data_sha": a.data_sha, "cells": cells}, indent=2) + "\n")


if __name__ == "__main__":
    main()
