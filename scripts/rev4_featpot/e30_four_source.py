"""E30 adapters: 40 nA3 cells; reuse E24 control, no AIC fold or label.

Training stays in the strict owner. Post-harvest scoring calls the existing E21
statistics owner; E30 reports removal cost and cannot reverse D1.
"""
import argparse
import json
from pathlib import Path

from v2_human_role import PRODUCTION_SOURCES, preflight_recipe
from v2_common import sha

SPEC = "sel:59f0bbc2f290@h32:H128:cv16:cf98"


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
    p.add_argument("cmd", choices=("grid", "production-grid", "control-pins", "score", "external-score"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--results", type=Path)
    p.add_argument("--control-root", type=Path)
    p.add_argument("--control-pins", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--program-sha")
    p.add_argument("--data-sha")
    a = p.parse_args()
    if a.cmd == "control-pins":
        a.out.write_text(json.dumps(control_pins(a.control_root), indent=2) + "\n")
    elif a.cmd in ("score", "external-score"):
        (score if a.cmd == "score" else external_score)(a.root, a.results, a.control_root, a.out, a.control_pins)
    else:
        import e21_cheap_recipe as e21
        cells = grid(a.root, a.results, e21.columns("by_v2fy"), a.cmd == "production-grid")
        a.out.write_text(json.dumps({"program_sha": a.program_sha, "data_sha": a.data_sha, "cells": cells}, indent=2) + "\n")


if __name__ == "__main__":
    main()
