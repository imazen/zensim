"""Frozen V40 assessment: complete budget-bound cells before any target read.

The default commands cover only D1's four SDR sources. HDR and external panels
remain separate registered reads, with their original exposure boundaries.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

from e24_rev5 import e29_seed_stat, _e29_signed_w2
from v2_common import sha, admission_input_roots, refuse_immutable_output
from v2_human_role import PRODUCTION_SOURCES
from v40_package import PALETTE_SPEC, SPEC


def sdr_decision(delta, w2, study):
    """Exactly ten equal-four-source seed units, plus paired two-source W2."""
    if study not in ("e29", "e31", "e32"):
        raise ValueError("unregistered statistic")
    signed = e29_seed_stat(delta)
    w2 = np.asarray(w2, dtype=np.float64)
    if w2.shape != (10, 2) or not np.isfinite(w2).all():
        raise ValueError("INCOMPLETE: ten complete KADID/TID W2 seed pairs required")
    w2_stat = e29_seed_stat(np.repeat(w2.mean(axis=1)[:, None], 4, axis=1))
    per_source = dict(zip(PRODUCTION_SOURCES, np.asarray(delta).mean(axis=0).tolist()))
    guards = dict(
        mean=signed["delta"] >= -0.002,
        each_source=min(per_source.values()) >= -0.005,
        w2=w2_stat["delta"] > -2 * w2_stat["se"],
    )
    out = dict(
        study=study,
        signed=signed,
        per_source=per_source,
        w2=w2_stat,
        guards={k: bool(v) for k, v in guards.items()},
        as_good=all(guards.values()),
        source_seed_deltas=np.asarray(delta).tolist(),
        w2_source_seed_deltas=w2.tolist(),
    )
    if study == "e32":
        statistic = signed["delta"] / signed["se"]
        p = float(student_t.sf(statistic, df=9))
        out.update(
            t=statistic,
            df=9,
            p_one_sided=p,
            adopt=bool(out["as_good"] and signed["delta"] > 0.002 and p < 0.05),
        )
    return out


def upiq_report(pred, target, keys):
    """Report-only UPIQ TRAIN fit/development geometry; never an adoption rule."""
    from lib.zen_stats import panel_batch

    pred, target = np.asarray(pred), np.asarray(target)
    if (
        len(keys) != len(pred)
        or len(pred) != len(target)
        or not np.isfinite([pred, target]).all()
    ):
        raise ValueError("INCOMPLETE: UPIQ report row/finite identity")
    if set(keys["role"]) != {"train"} or not set(keys["split"]) <= {
        "fit",
        "development",
    }:
        raise ValueError("UPIQ report refuses test/foreign roles")
    panels = {}
    for split, group in keys.groupby("split", sort=True):
        ix = group.index.to_numpy()
        panels[split] = dict(
            pooled=panel_batch([(split, pred[ix], target[ix])], stats="full")[0],
            per_study={
                study: panel_batch(
                    [(study, pred[g.index], target[g.index])], stats="full"
                )[0]
                for study, g in group.groupby("dataset")
            },
            within_reference={
                ref: panel_batch([(ref, pred[g.index], target[g.index])], stats="full")[
                    0
                ]
                for ref, g in group.groupby("reference_sha256")
            },
            scatter=dict(prediction=pred[ix].tolist(), target=target[ix].tolist()),
        )
    return dict(
        schema="e31-upiq-training-report-v1",
        panels=panels,
        independent_test=False,
        shipping_adoption_authorized=False,
    )


def complete(bundle, study, results, control, tools, *, only_control=False):
    """Verify retained receipts and canonical checkpoint provenance for every cell."""
    sys.path.insert(0, str(tools))
    from qualified_fit_contract import trusted_contract, verify_training

    program = bundle / "program.tar.gz"
    inspector = bundle / "bin/inspect_qualified_checkpoint"
    rows = {}
    arms = {"e29": ("hb4", "hc4"), "e31": ("uh4",), "e32": ("palette",)}[study]
    for label in ("control",) if only_control else ("control", *arms):
        jobset = "control" if label == "control" else study
        manifest = bundle / f"fit-manifest-fitv40-{jobset}-20261007.json"
        if not manifest.is_file():
            raise ValueError("INCOMPLETE: owner-blocked/unprepared manifest")
        jobs = json.loads(manifest.read_text())
        spec = (
            SPEC
            if label == "control"
            else PALETTE_SPEC
            if label == "palette"
            else SPEC + ":" + label
        )
        jobs = [
            j
            for j in jobs
            if j["kind"]["argv"][j["kind"]["argv"].index("--spec") + 1] == spec
        ]
        if len(jobs) != 40:
            raise ValueError("INCOMPLETE: registered forty-cell grid required")
        rows[label] = {}
        for job in jobs:
            argv, kind = job["kind"]["argv"], job["kind"]
            fold = argv[argv.index("--heldout") + 1]
            seed = int(argv[argv.index("--seed-index") + 1])
            if (
                fold not in PRODUCTION_SOURCES
                or not 0 <= seed < 10
                or (fold, seed) in rows[label]
            ):
                raise ValueError("INCOMPLETE: duplicate/foreign fold or seed")
            cell = (
                (control if label == "control" else results)
                / "cells"
                / job["cell"]["image_path"]
            )
            rp, model, fleet = (
                cell / "result.json",
                cell / "refit/last.bin",
                cell / "fleet_receipt.json",
            )
            if not all(p.is_file() for p in (rp, model, fleet)):
                raise ValueError(f"INCOMPLETE: missing {label}/{fold}/s{seed}")
            receipt = json.loads(fleet.read_text())
            if (
                receipt["program_sha"] != kind["program_sha"]
                or receipt["data_sha"] != kind["data_sha"]
                or receipt["argv_sha"] != kind["argv_sha"]
                or receipt["tier"]["effective"] != "v3"
            ):
                raise ValueError("INCOMPLETE: fleet identity differs")
            for rel, pin in receipt["files"].items():
                if (
                    Path(rel).is_absolute()
                    or ".." in Path(rel).parts
                    or sha(cell / rel) != pin
                ):
                    raise ValueError("INCOMPLETE: frozen cell file changed")
            r = json.loads(rp.read_text())
            expected = trusted_contract(kind, program, inspector)
            if (
                verify_training(r, model, kind, expected, inspector) != "registered-fit"
                or r.get("spec") != spec
                or r.get("heldout") != fold
                or r.get("head") != "N"
                or sha(model) != r.get("selected_bake_sha256")
                or sha(model) != receipt["selected_bake_sha"]
                or sha(rp) != receipt["result_sha"]
            ):
                raise ValueError("INCOMPLETE: result/model or full budget differs")
            columns = [int(v) for v in (cell / "keep_features.txt").read_text().split()]
            if columns != expected["columns"]:
                raise ValueError("INCOMPLETE: feature order differs")
            rows[label][fold, seed] = cell
    return rows


def score(bundle, study, results, control, root, tools, out, pins):
    from v2_lodo_mlp import strict_training_groups, predict
    from v2_teacher import key_path
    from lib.zen_stats import panel_batch
    import e13_teacher as e13
    import pyarrow.parquet as pq
    import pandas as pd

    cells = complete(bundle, study, results, control, tools)
    frozen = json.loads(pins.read_text())
    if frozen.get("control_choice") != "fresh-matched-v40" or frozen.get(
        "program_sha"
    ) != sha(bundle / "program.tar.gz"):
        raise ValueError("INCOMPLETE: coordinator-frozen fresh V40 control required")
    for (fold, seed), cell in cells["control"].items():
        if frozen["cells"].get(f"{fold}_s{seed}") != {
            "result_sha256": sha(cell / "result.json"),
            "bake_sha256": sha(cell / "refit/last.bin"),
        }:
            raise ValueError("INCOMPLETE: frozen control changed")
    refuse_immutable_output(out, (*admission_input_roots(root), results, control))
    if out.exists():
        raise ValueError("fresh assessment output required")
    receipt = json.loads((root / "wide/main/real/receipt.json").read_text())
    groups = [
        (fold, root / receipt["legs"][fold]["full"]["rel"], 0.0, 0.0, "withinref,rank")
        for fold in PRODUCTION_SOURCES
    ]
    strict_training_groups(groups, root / "human_role_decision.json")
    # No target read above this boundary: all cells and source populations admitted.
    out.mkdir(parents=True)
    panels = {label: {} for label in cells}
    for fold, table, _, _, _ in groups:
        meta = pq.read_table(key_path(table)).to_pandas()
        y = pq.read_table(table, columns=["human_score"])["human_score"].to_numpy()
        meta["target"] = y
        if fold in e13.TYPE_SOURCES:
            paths = pd.concat(
                [
                    pq.read_table(
                        e13.BANK / m / "keys.parquet", columns=["pair_key", "dist_path"]
                    ).to_pandas()
                    for m in e13.TYPE_SOURCES[fold]
                ]
            ).drop_duplicates("pair_key")
            ix = pd.Index(paths.pair_key).get_indexer(meta.pair_key)
            if (ix < 0).any():
                raise ValueError("distortion metadata join differs")
            meta["dtype"] = [
                Path(p).stem.split("_")[1] for p in paths.dist_path.to_numpy()[ix]
            ]
        for label, grid in cells.items():
            for seed in range(10):
                dest = out / label / f"{fold}_s{seed}"
                dest.mkdir(parents=True)
                pred = predict(
                    grid[fold, seed] / "refit/last.bin", table, dest / "pred.tsv"
                )
                panel = panel_batch([(fold, pred, y)], stats="full")[0]
                rp = dest / "result.json"
                rp.write_text(
                    json.dumps(dict(prediction=pred.tolist(), score=panel)) + "\n"
                )
                summary = e13.worst_case(rp, meta)
                if "dtype" in meta:
                    types = {
                        str(k): e13.spearman(pred[ix], y[ix])
                        for k, ix in meta.groupby("dtype").indices.items()
                    }
                    summary.update(
                        _e29_signed_w2(list(types.values())), signed_types=types
                    )
                if any(
                    v is None or not np.isfinite(v)
                    for k, v in summary.items()
                    if k != "signed_types"
                ):
                    raise ValueError("INCOMPLETE: nonfinite/undefined panel")
                panels[label][fold, seed] = summary
    decisions = {}
    for arm in set(cells) - {"control"}:
        delta = [
            [
                panels[arm][f, s]["signed"] - panels["control"][f, s]["signed"]
                for f in PRODUCTION_SOURCES
            ]
            for s in range(10)
        ]
        w2 = [
            [
                panels[arm][f, s]["w2_type_worst3"]
                - panels["control"][f, s]["w2_type_worst3"]
                for f in ("kadid", "tid2013")
            ]
            for s in range(10)
        ]
        decisions[arm] = sdr_decision(delta, w2, study)
    report = dict(
        schema=f"{study}-v40-sdr-assessment-v1",
        decisions=decisions,
        panels={
            a: {f"{f}_s{s}": v for (f, s), v in p.items()} for a, p in panels.items()
        },
        control_pins_sha256=sha(pins),
        program_sha256=sha(bundle / "program.tar.gz"),
    )
    (out / "decision.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    # Existing E29 HDR owner takes precisely this SDR guard object.
    if study == "e29":
        (out / "e29_sdr_decision.json").write_text(
            json.dumps(decisions, indent=2) + "\n"
        )
    print(json.dumps(decisions))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze-control", action="store_true")
    for name in (
        "bundle",
        "results",
        "control",
        "root",
        "tools",
        "out",
        "control-pins",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--study", choices=("e29", "e31", "e32"), required=True)
    a = p.parse_args()
    if a.freeze_control:
        rows = complete(
            a.bundle, a.study, a.results, a.control, a.tools, only_control=True
        )["control"]
        pins = dict(
            schema="v40-complete-control-freeze-v1",
            control_choice="fresh-matched-v40",
            program_sha=sha(a.bundle / "program.tar.gz"),
            cells={
                f"{f}_s{s}": {
                    "result_sha256": sha(c / "result.json"),
                    "bake_sha256": sha(c / "refit/last.bin"),
                }
                for (f, s), c in rows.items()
            },
        )
        with a.control_pins.open("x") as file:
            file.write(json.dumps(pins, indent=2) + "\n")
        with a.control_pins.with_name("E29_CONTROL_PINS.json").open("x") as file:
            file.write(
                json.dumps(
                    {**pins, "study": "E29", "control_choice": "fresh-matched"},
                    indent=2,
                )
                + "\n"
            )
        return
    score(
        a.bundle, a.study, a.results, a.control, a.root, a.tools, a.out, a.control_pins
    )


if __name__ == "__main__":
    main()
