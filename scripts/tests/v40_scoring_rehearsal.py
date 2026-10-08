"""Bounded authorization -> actual executor -> canonical refusal -> final scoring.

Synthetic 40-cell scoring grids never become registered fits. The production
budget verifier is unchanged and refuses the bounded blob. Only the fixture
complete/admission/prediction hooks are patched; statistics, panel owner,
scorer products and final artifact/controller validation are real.
"""

# ruff: noqa: E402
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO / "scripts"),
    str(REPO / "scripts/rev4_featpot"),
    str(REPO / "scripts/tests"),
]
import v40_launch
import v40_score
import v40_postfit_fixture as fixture
from v2_common import sha
from v2_human_role import PRODUCTION_SOURCES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=77)
    a = parser.parse_args()
    a.dest.mkdir(parents=True, exist_ok=False)
    events = []
    b = a.bundle
    probe = a.dest / "SYNTHETIC_AUTHORIZATION_PROBE"
    probe.mkdir()
    jobset = v40_launch.JOBSETS[0]
    for name in (
        "program.tar.gz",
        f"fit-manifest-{jobset}.json",
        "postfit.sh",
        "jobset_caps.json",
    ):
        (probe / name).symlink_to(b / name)
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", a.image], text=True
    ).strip()
    ids = dict(
        program_sha=sha(b / "program.tar.gz"),
        image=a.image,
        image_id=image_id,
        files={
            name: sha(probe / name)
            for name in ("program.tar.gz", f"fit-manifest-{jobset}.json", "postfit.sh")
        },
    )
    fixture.write(probe / f"AUTHORIZATION_REQUIRED-{jobset}.json", dict(identities=ids))
    fixture.write(
        probe / f"LAUNCH_AUTHORIZATION-{jobset}.json",
        dict(
            schema="v40-coordinator-launch-v1",
            coordinator_message="SYNTHETIC LOCAL BOUNDED REHEARSAL ONLY; NOT A FLEET AUTHORIZATION",
            source_landed=True,
            pins_pushed=True,
            reviewed=True,
            E30_completed=True,
            control_choice_frozen=True,
            identities=ids,
        ),
    )
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/var/tmp/fitv2/jobset_caps.json":
            return original_read(b / "jobset_caps.json")
        return original_read(path, *args, **kwargs)

    with patch.object(Path, "read_text", read):
        assert v40_launch.gate(probe, jobset) == ids
    events.append(
        dict(
            step="authorization",
            status="PASS",
            scope="isolated synthetic gate only",
            live_caps_modified=False,
            fleet_actions=0,
        )
    )
    command = [
        sys.executable,
        str(Path(__file__).with_name("v40_executor_smoke.py")),
        "--bundle",
        str(b),
        "--image",
        a.image,
        "--arm",
        "control",
        "--mode",
        "bounded",
        "--attempt",
        str(a.attempt),
    ]
    with (a.dest / "executor.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    key = f"control-bounded-{a.attempt}"
    executor = json.loads(
        (b / f"smoke-{key}/EXECUTOR_SHORT_PATH_PASS.json").read_text()
    )
    assert (
        executor["image_id"] == image_id
        and executor["program_sha"] == ids["program_sha"]
    )
    events.append(
        dict(
            step="actual-executor",
            status="PASS",
            image_id=image_id,
            program_sha256=ids["program_sha"],
            receipt_sha256=sha(b / f"smoke-{key}/EXECUTOR_SHORT_PATH_PASS.json"),
        )
    )
    stage = a.dest / "canonical-harvest"
    command = [
        sys.executable,
        str(b / "harvest_driver_v40.py"),
        "--bundle",
        str(b),
        "offline",
        str(b / f"smoke-manifest-{key}.json"),
        "--dry-run-blob",
        str(b / f"smoke-{key}/EXECUTOR_SHORT_OUTPUT.tar.gz"),
        "--scratch",
        str(stage),
        "--install",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    (a.dest / "harvest.log").write_text(result.stdout + result.stderr)
    assert (
        result.returncode == 1
        and "local smoke cannot install as a registered full-budget cell"
        in result.stderr
    )
    job = json.loads((b / f"smoke-manifest-{key}.json").read_text())[0]
    relative = Path(
        job["kind"]["argv"][job["kind"]["argv"].index("--dest") + 1]
    ).relative_to("/var/tmp/rev4-featpot")
    cell = stage / relative
    model = cell / "refit/last.bin"
    events.append(
        dict(
            step="canonical-harvest",
            status="PASS",
            outcome="verified smoke refused installation",
            model_sha256=sha(model),
            installed=False,
        )
    )
    synthetic = a.dest / "SYNTHETIC_SCORING_ONLY"
    grids, root = fixture.fixture(synthetic, checkpoint=model.read_bytes())
    shutil.copyfile(b / "program.tar.gz", synthetic / "program.tar.gz")
    fixture.freeze(synthetic, grids)
    os.environ["ZEN_PANEL_BIN"] = str(b / "bin/panel")
    import v2_lodo_mlp as loader
    import v2_common as common

    # Bind the real retained predictor to generated features only, then use its
    # vector as the basis of explicit synthetic seed/arm perturbations.
    keep = [int(v) for v in (cell / "keep_features.txt").read_text().split()]
    count = 120
    rng = np.random.default_rng(4040)
    columns = {
        f"f{i}": rng.uniform(0.01, 1.0, count) if i in keep else np.full(count, np.nan)
        for i in range(1853)
    }
    feature = a.dest / "synthetic-features.parquet"
    pq.write_table(
        pa.table(
            {
                "ref_basename": [f"synthetic-ref-{i // 20}" for i in range(count)],
                # The canonical Parquet predictor requires this loader column;
                # these generated zeros are fixtures, never human labels.
                "human_score": np.zeros(count),
                **columns,
            }
        ),
        feature,
        compression="zstd",
    )
    fixture.write(
        Path(str(feature) + ".manifest.json"),
        dict(
            formula_revision=5,
            feature_set_id="basic+peaks+v2@w1825/rev5_localwin#36c3f3af",
        ),
    )
    with (
        patch.object(common, "FITBIN", b / "bin/bake_dial_refit"),
        patch.object(loader, "FITBIN", b / "bin/bake_dial_refit"),
    ):
        native = loader.predict(model, feature, a.dest / "native-pred.tsv")
    if native.std() == 0:
        raise ValueError("synthetic native prediction must vary")
    base = (native - native.mean()) / native.std()
    y = base + np.sin(np.arange(count) * 1.7) * 0.03
    type_members = {}
    receipt = dict(legs={})
    for fold in PRODUCTION_SOURCES:
        table = root / "wide/main/real" / f"{fold}.parquet"
        keys = pa.table(
            dict(
                pair_key=[f"synthetic-{fold}-{i}" for i in range(count)],
                ref_basename=[f"synthetic-ref-{i // 20}" for i in range(count)],
            )
        )
        pq.write_table(keys, table.with_suffix(".keys.parquet"))
        pq.write_table(
            pa.table(
                {"ref_basename": keys["ref_basename"], "human_score": y, **columns}
            ),
            table,
        )
        receipt["legs"][fold] = dict(
            full=dict(rel=str(table.relative_to(root)), rows=count)
        )
        if fold in ("kadid", "tid2013"):
            path = synthetic / f"{fold}-type-keys.parquet"
            pq.write_table(
                pa.table(
                    dict(
                        pair_key=keys["pair_key"],
                        dist_path=[f"synthetic_{i % 5}_{i}.png" for i in range(count)],
                    )
                ),
                path,
            )
            member = "kadid_train" if fold == "kadid" else "tid2013"
            type_members[member] = dict(path=str(path), sha256=sha(path), rows=count)
    empty = synthetic / "kadid-select-type-keys.parquet"
    pq.write_table(
        pa.table(
            dict(
                pair_key=pa.array([], type=pa.string()),
                dist_path=pa.array([], type=pa.string()),
            )
        ),
        empty,
    )
    type_members["kadid_select"] = dict(path=str(empty), sha256=sha(empty), rows=0)
    fixture.write(
        synthetic / "W2_KEY_PINS.json",
        dict(schema="v40-w2-label-free-keys-v1", members=type_members),
    )
    fixture.write(root / "wide/main/real/receipt.json", receipt)

    def complete(*args, only_control=False, **kwargs):
        return fixture.complete_fixture(grids, args[1], only_control)

    def admit(groups, decision):
        assert len(groups) == 4 and all(root in Path(g[1]).parents for g in groups)
        return {"scope": "SYNTHETIC SCORING FIXTURE ONLY"}

    def predict(bake, table, out, **kwargs):
        parts = out.parent.parts
        label, seed_name = parts[-2:]
        fold, seed = seed_name.rsplit("_s", 1)
        seed = int(seed)
        fold_index = list(PRODUCTION_SOURCES).index(fold)
        label_index = ["control", "hb4", "hc4", "uh4", "palette"].index(label)
        noise = np.random.default_rng(
            40400 + seed * 100 + fold_index * 10 + label_index
        )
        values = base + noise.normal(0, 0.6 if label == "control" else 0.5, count)
        pd.DataFrame(dict(row_idx=np.arange(count), pred=values)).to_csv(
            out, sep="\t", index=False
        )
        return pd.read_csv(out, sep="\t").pred.to_numpy()

    with (
        patch.object(v40_score, "complete", complete),
        patch.object(v40_score, "admission_input_roots", lambda _: (root,)),
        patch.object(loader, "strict_training_groups", admit),
        patch.object(loader, "predict", predict),
    ):
        for study in fixture.ARMS:
            out = a.dest / f"assessment-{study}"
            stdout = a.dest / f"score-{study}.stdout.json"
            with stdout.open("w") as log, contextlib.redirect_stdout(log):
                v40_score.score(
                    synthetic,
                    study,
                    synthetic,
                    synthetic,
                    root,
                    b / "committed-tools",
                    out,
                    synthetic / "V40_CONTROL_PINS.json",
                )
            checked = v40_score.validate_artifacts(
                synthetic,
                study,
                synthetic,
                synthetic,
                b / "committed-tools",
                out,
                synthetic / "V40_CONTROL_PINS.json",
                stdout,
                root=root,
            )
            events.append(
                dict(
                    step="final-scoring",
                    study=study,
                    status="PASS",
                    result=checked,
                    decision_sha256=sha(out / "decision.json"),
                    scope="synthetic seed/fold outcomes only",
                )
            )
    env = {
        **os.environ,
        "V40_POSTFIT_TEST_OUTPUT": str(a.dest / "controller-cases"),
        "PYTHONPATH": ":".join(
            [
                str(REPO / "scripts"),
                str(REPO / "scripts/rev4_featpot"),
                str(REPO / "scripts/tests"),
            ]
        ),
    }
    with (a.dest / "controller-cases.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "scripts.tests.test_v40_postfit_artifacts",
            ],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    events.append(
        dict(
            step="controller-validation",
            status="PASS",
            scenarios=28,
            studies=["control", "e29", "e31", "e32"],
            empty_success_refused=True,
            credential_reads=0,
            installation_actions=0,
        )
    )
    record = dict(
        schema="v40-bounded-final-scoring-rehearsal-v1",
        status="PASS",
        program_sha256=ids["program_sha"],
        image_id=image_id,
        trainer_sha256=sha(b / "bin/zensim_mlp_train"),
        scorer_source_sha256=sha(REPO / "scripts/rev4_featpot/v40_score.py"),
        events=events,
        fixture_only_patches=[
            "complete (synthetic grids)",
            "strict_training_groups (synthetic population)",
            "admission_input_roots (synthetic roots)",
            "predict (native-vector-based explicit synthetic perturbations)",
        ],
        production_budget_verifier_unchanged=True,
        registered_cells_installed=0,
        fleet_actions=0,
        actual_native_prediction_sha256=sha(a.dest / "native-pred.tsv"),
        scientific_outcomes=False,
    )
    fixture.write(a.dest / "RESULT.json", record)
    fixture.write(b / "FINAL_SCORING_REHEARSAL.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
