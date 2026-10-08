"""Synthetic controller products, never a full-budget training verifier override."""

import json


import v40_score as owner
from v2_human_role import PRODUCTION_SOURCES

ARMS = {"e29": ("hb4", "hc4"), "e31": ("uh4",), "e32": ("palette",)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def fixture(bundle, checkpoint=b"SYNTHETIC CHECKPOINT IDENTITY"):
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "program.tar.gz").write_bytes(b"SYNTHETIC PROGRAM IDENTITY")
    grids = {}
    for label in ("control", "hb4", "hc4", "uh4", "palette"):
        grid = grids[label] = {}
        for fold in PRODUCTION_SOURCES:
            for seed in range(10):
                cell = bundle / "synthetic-cells" / label / f"{fold}_s{seed}"
                (cell / "refit").mkdir(parents=True)
                (cell / "refit/last.bin").write_bytes(checkpoint)
                write(
                    cell / "result.json",
                    dict(
                        scope="SYNTHETIC SCORING FIXTURE ONLY",
                        fold=fold,
                        seed=seed,
                        label=label,
                    ),
                )
                grid[fold, seed] = cell
    root = bundle / "synthetic-root"
    write(
        root / "wide/main/real/receipt.json",
        dict(legs={f: dict(full=dict(rows=24)) for f in PRODUCTION_SOURCES}),
    )
    return grids, root


def complete_fixture(grids, study, only_control=False):
    return {
        a: grids[a]
        for a in (("control",) if only_control else ("control", *ARMS[study]))
    }


def freeze(bundle, grids):
    pins = owner._control_record(bundle, {"control": grids["control"]})
    write(bundle / "V40_CONTROL_PINS.json", pins)
    write(
        bundle / "E29_CONTROL_PINS.json",
        {**pins, "study": "E29", "control_choice": "fresh-matched"},
    )
    return pins


def assessment(bundle, study, grids, out):
    cells = complete_fixture(grids, study)
    panels = {label: {} for label in cells}
    out.mkdir(parents=True)
    for label in cells:
        for fi, fold in enumerate(PRODUCTION_SOURCES):
            for seed in range(10):
                delta = 0 if label == "control" else 0.003 + seed / 10000 + fi / 100000
                signed = 0.7 + seed / 10000 + delta
                tail = -0.2 + (0 if label == "control" else 0.002 + seed / 10000)
                types = {
                    str(i): v for i, v in enumerate((tail - 0.01, tail, tail + 0.01))
                }
                summary = dict(
                    signed=signed,
                    w1_ref_p10=0.4,
                    w3_z_rmse=0.2,
                    w3_or=0.1,
                    w4_neg_share=0.0,
                )
                if fi < 2:
                    summary.update(
                        owner._e29_signed_w2(list(types.values())), signed_types=types
                    )
                key = f"{fold}_s{seed}"
                panels[label][key] = summary
                dest = out / label / key
                pred = list(range(24))
                write(
                    dest / "result.json",
                    dict(
                        prediction=pred,
                        score=dict(srocc_signed=signed, z_rmse=0.2, **{"or": 0.1}),
                    ),
                )
                (dest / "pred.tsv").write_text(
                    "row_idx\tpred\n" + "".join(f"{i}\t{i}\n" for i in pred)
                )
    decisions = {}
    for arm in ARMS[study]:
        delta = [
            [
                panels[arm][f"{f}_s{s}"]["signed"]
                - panels["control"][f"{f}_s{s}"]["signed"]
                for f in PRODUCTION_SOURCES
            ]
            for s in range(10)
        ]
        w2 = [
            [
                panels[arm][f"{f}_s{s}"]["w2_type_worst3"]
                - panels["control"][f"{f}_s{s}"]["w2_type_worst3"]
                for f in PRODUCTION_SOURCES[:2]
            ]
            for s in range(10)
        ]
        decisions[arm] = owner.sdr_decision(delta, w2, study)
    if study == "e29":
        write(out / "e29_sdr_decision.json", decisions)
    report = dict(
        schema=f"{study}-v40-sdr-assessment-v1",
        decisions=decisions,
        panels=panels,
        cells=owner._cell_pins(cells),
        observation_counts={f: 24 for f in PRODUCTION_SOURCES},
        control_pins_sha256=owner.sha(bundle / "V40_CONTROL_PINS.json"),
        program_sha256=owner.sha(bundle / "program.tar.gz"),
        artifacts={
            str(p.relative_to(out)): owner.sha(p) for p in out.rglob("*") if p.is_file()
        },
    )
    write(out / "decision.json", report)
    return report


def complete_from_disk(bundle, study, only_control=False):
    labels = ("control",) if only_control else ("control", *ARMS[study])
    return {
        a: {
            (f, s): bundle / "synthetic-cells" / a / f"{f}_s{s}"
            for f in PRODUCTION_SOURCES
            for s in range(10)
        }
        for a in labels
    }
