"""Prepare combined fit contracts/specs through the existing strict and fleet owners.

This command creates local artifacts. It never queues work or grants approval.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess

from v2_common import REPO, sha, selection_id
from v2_human_role import PRODUCTION_SOURCES, preflight_recipe
from e30_four_source import SPEC
from e32_palette import ARM_IDS, FEATURE_SET_ID

PALETTE_SPEC = f"sel:{selection_id(ARM_IDS)}@h32:H128:cv16:cf98"
V40_CONTRACT = "benchmarks/v40_fit_contract_2026-10-07.json"
DECLARATION_KEYS = (
    "feature_set_id",
    "formula_revision",
    "decoder_era",
    "decoder_revision",
    "sampling",
    "research_palette",
    "data_role",
    "data_role_decision_required",
    "human_sources",
    "table_sha256",
    "keys_sha256",
    "row_keys_sha256",
    "row_selection_sha256",
    "row_selection",
    "rows",
    "rows_kept",
    "observations",
)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def contracts(bundle, e29_data, palette_data):
    """Use D1's existing pin contract and the strict palette table owner."""
    from v2_lodo_mlp import strict_training_groups
    from v2_teacher import coverage_leg

    base = json.loads(
        (
            REPO / "benchmarks/shippath_qualified_fit_contract_2026-10-07.json"
        ).read_text()
    )
    e29 = json.loads((REPO / "benchmarks/e29_fit_contract_2026-10-07.json").read_text())
    if sha(e29_data) != e29["data_sha"]:
        raise ValueError("frozen E29/D1 input archive changed")
    import pyarrow.parquet as pq
    from v2_teacher import key_path
    from v2_common import acceptance_weight

    e30_root = Path("/var/tmp/rev4-featpot/e30-results")
    input_contracts, weights = {}, {}
    for fold in PRODUCTION_SOURCES:
        cell = e30_root / "cells" / f"{SPEC}__N/without_{fold}_s0"
        r = json.loads((cell / "result.json").read_text())
        decoded = json.loads(
            subprocess.check_output(
                [
                    str(bundle / "bin/inspect_qualified_checkpoint"),
                    str(cell / "refit/last.bin"),
                ],
                text=True,
            )
        )
        weights[fold] = r["train_weights"]
        input_contracts[fold] = {
            t["name"]: {
                k: t[k]
                for k in (
                    "loss_mode",
                    "n_features",
                    "rows",
                    "train_w",
                    "val_w",
                    "within_ref",
                )
            }
            for t in decoded["repro"]["inputs"]
        }
    hdr_receipt = json.loads(
        (bundle / "v2e29/wide/main/real/receipt.json").read_text()
    )["legs"]["hdr_consensus"]["hb4"]["fit"]["rel"]
    hdr_refs = pq.read_table(
        key_path(bundle / "v2e29" / hdr_receipt), columns=["ref_basename"]
    )["ref_basename"].to_pylist()
    hdr_weight = acceptance_weight(4.0, hdr_refs)
    variants = []
    for arm in ("control", "hb4", "hc4"):
        c = {**copy.deepcopy(base), **copy.deepcopy(e29)}
        c.update(
            schema="v40-research-fit-contract-v1",
            name=arm,
            launchable=True,
            root="/var/tmp/rev4-featpot/v2e29",
            spec=SPEC + (":" + arm if arm != "control" else ""),
            research_hdr=arm != "control",
            input_contracts=copy.deepcopy(input_contracts),
            train_weights=copy.deepcopy(weights),
        )
        if arm != "control":
            for fold in PRODUCTION_SOURCES:
                c["train_weights"][fold]["hdr"] = hdr_weight
                c["input_contracts"][fold]["hdr"] = dict(
                    loss_mode="Rank",
                    n_features=1853,
                    rows=7390,
                    train_w=hdr_weight,
                    val_w=0.0,
                    within_ref=False,
                )
            c["routes"] = {
                fold: sorted([*tables, c["hdr_tables"][arm]], key=lambda t: t["name"])
                for fold, tables in c["routes"].items()
                if fold != "production"
            }
        variants.append(c)
    root = bundle / "v2e32"
    for fold in (None, *PRODUCTION_SOURCES):
        preflight_recipe(root, root / "human_role_decision.json", fold)
    r = json.loads((root / "wide/main/real/receipt.json").read_text())
    scratch = bundle / "contract-inputs"
    scratch.mkdir(exist_ok=False)
    coverage, _ = coverage_leg(152, scratch, admitted_root=root)
    c = copy.deepcopy(base)
    c.update(
        schema="v40-research-fit-contract-v1",
        name="palette",
        launchable=True,
        root="/var/tmp/rev4-featpot/v2e32",
        spec=PALETTE_SPEC,
        research_hdr=False,
        data_sha=sha(palette_data),
        feature_set_id=FEATURE_SET_ID,
        columns=ARM_IDS,
        routes={},
        table_declarations={},
        input_contracts=copy.deepcopy(input_contracts),
        train_weights=copy.deepcopy(weights),
        wide_receipt_sha256=sha(root / "wide/main/real/receipt.json"),
        frozen_sha256=sha(root / "wide/frozen.json"),
    )
    for fold in PRODUCTION_SOURCES:
        paths = [
            (
                name,
                root / r["legs"][name.removesuffix("_development")][split]["rel"],
                split,
            )
            for name, split in [
                ("safesyn", "fit"),
                ("safesyn_development", "dev"),
                ("cid22", "fit"),
                ("cid22_development", "dev"),
            ]
        ]
        paths += [
            (
                "human" + ("_development" if split == "dev" else ""),
                root / r["legs"][f"human_without_{fold}"][split]["rel"],
                split,
            )
            for split in ("fit", "dev")
        ]
        paths += [("coverage", coverage, "fit")]
        c["routes"][fold] = sorted(
            strict_training_groups(
                [
                    (n, p, int(split == "fit"), int(split == "dev"), "withinref,rank")
                    for n, p, split in paths
                ],
                root / "human_role_decision.json",
            ),
            key=lambda t: t["name"],
        )
        for _, path, _ in paths:
            d = json.loads(Path(f"{path}.manifest.json").read_text())
            c["table_declarations"][d["table_sha256"]] = {
                k: d[k] for k in DECLARATION_KEYS if k in d
            }
    for inputs in c["input_contracts"].values():
        for table in inputs.values():
            table["n_features"] = 1867
    variants.append(c)
    output = bundle / "v40-fit-contract.json"
    write(
        output,
        dict(
            schema="v40-research-fit-package-v1",
            variants=variants,
            E31=dict(
                name="uh4",
                launchable=False,
                reason="owner disposition of UPIQ legacy JOD producer gap required",
            ),
        ),
    )
    return output


def specs(bundle, program, e29_data, palette_data, ctl):
    """Canonical declare-fits produces IDs and argv hashes; no remote declaration."""
    package = json.loads((bundle / "v40-fit-contract.json").read_text())
    by_name = {v["name"]: v for v in package["variants"]}
    for study, arms in [
        ("control", ("control",)),
        ("e29", ("hb4", "hc4")),
        ("e32", ("palette",)),
    ]:
        cells = []
        for arm in arms:
            c = by_name[arm]
            for fold in PRODUCTION_SOURCES:
                for seed in range(10):
                    name = f"{c['spec']}__N/without_{fold}_s{seed}"
                    argv = [
                        "v2_lodo_mlp.py",
                        "--spec",
                        c["spec"],
                        "--head",
                        "N",
                        "--seed-index",
                        str(seed),
                        "--root",
                        c["root"],
                        "--columns",
                        ",".join(map(str, c["columns"])),
                        "--strict-admission",
                        "--train-only",
                        "--data-role-decision",
                        c["root"] + "/human_role_decision.json",
                        "--dest",
                        f"/var/tmp/rev4-featpot/v40-{study}-results/cells/{name}",
                        "--heldout",
                        fold,
                    ]
                    cells.append(dict(name=name, argv=argv))
        jobset = f"fitv40-{study}-20261007"
        path = bundle / f"fit-spec-{jobset}.json"
        write(
            path,
            dict(
                program_sha=sha(program),
                data_sha=by_name[arms[0]]["data_sha"],
                cells=cells,
            ),
        )
        subprocess.run(
            [
                str(ctl),
                "declare-fits",
                "--spec",
                str(path),
                "--out",
                str(bundle / f"fit-manifest-{jobset}.json"),
            ],
            check=True,
        )
    pending_data = bundle / "e31-pending-fit-data.tar.gz"
    cells = []
    for fold in PRODUCTION_SOURCES:
        for seed in range(10):
            name = f"{SPEC}:uh4__N/without_{fold}_s{seed}"
            root = "/var/tmp/rev4-featpot/v2d1"
            cells.append(
                dict(
                    name=name,
                    argv=[
                        "v2_lodo_mlp.py",
                        "--spec",
                        SPEC + ":uh4",
                        "--head",
                        "N",
                        "--seed-index",
                        str(seed),
                        "--root",
                        root,
                        "--columns",
                        ",".join(map(str, by_name["control"]["columns"])),
                        "--strict-admission",
                        "--train-only",
                        "--data-role-decision",
                        root + "/human_role_decision.json",
                        "--upiq380-fit",
                        "/var/tmp/rev4-featpot/upiq380-fit/upiq380_fit.parquet",
                        "--upiq-label-disposition",
                        "/var/tmp/rev4-featpot/upiq380-fit/OWNER_DISPOSITION_REQUIRED.json",
                        "--heldout",
                        fold,
                        "--dest",
                        f"/var/tmp/rev4-featpot/v40-e31-results/cells/{name}",
                    ],
                )
            )
    path = bundle / "prepared-spec-E31-NOT-LAUNCHABLE.json"
    write(path, dict(program_sha=sha(program), data_sha=sha(pending_data), cells=cells))
    subprocess.run(
        [
            str(ctl),
            "declare-fits",
            "--spec",
            str(path),
            "--out",
            str(bundle / "prepared-manifest-E31-NOT-LAUNCHABLE.json"),
        ],
        check=True,
    )
    # E31 remains an explicit, unlaunchable preparation inventory, never a declaration.
    write(
        bundle / "E31_PENDING.json",
        dict(
            schema="v40-owner-blocked-arm-v1",
            arm="uh4",
            launchable=False,
            cell_count=40,
            folds=list(PRODUCTION_SOURCES),
            seeds=list(range(10)),
            spec=SPEC + ":uh4",
            required_disposition="e31-upiq-label-disposition-v1",
            fit_source="/mnt/v/output/zensim/upiq380-rev5-r2-2026-10-07/upiq380_fit.parquet",
            development_payload_included=False,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("contracts", "specs"))
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--e29-data", type=Path, required=True)
    p.add_argument("--palette-data", type=Path, required=True)
    p.add_argument("--program", type=Path)
    p.add_argument("--ctl", type=Path)
    a = p.parse_args()
    if a.mode == "contracts":
        contracts(a.bundle, a.e29_data, a.palette_data)
    else:
        if a.program is None or a.ctl is None:
            p.error("specs requires --program and --ctl")
        specs(a.bundle, a.program, a.e29_data, a.palette_data, a.ctl)


if __name__ == "__main__":
    main()
