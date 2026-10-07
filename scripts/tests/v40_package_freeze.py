"""Freeze local V40 runtime, refusal evidence and inert authorization templates.

No network, Docker publication, labels, queue edits or authorization creation.
Run once against the already verified prepared bundle.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open("x") as file:
        file.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze(bundle, source, source_commit, metrics_commit):
    resolved = subprocess.check_output(["jj", "log", "-r", source_commit, "--no-graph", "-T", "commit_id"], cwd=source, text=True).strip()
    if resolved != source_commit or len(metrics_commit) != 40:
        raise ValueError("freeze requires exact verified source and metrics commit IDs")
    pins = json.loads((bundle / "PACKAGE_PINNED.json").read_text())
    if sha(bundle / "program.tar.gz") != pins["program_sha"]:
        raise ValueError("fit program changed")
    runtime = bundle / "runtime"
    runtime.mkdir()
    with tarfile.open(bundle / "program.tar.gz") as tar:
        for item in tar.getmembers():
            name = Path(item.name)
            if not item.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe runtime archive")
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(item).read())
            if name.parts[0] == "bin":
                path.chmod(0o755)
    # Freeze the label-free distortion-type join used by the registered W2 owner.
    w2_pin = bundle / "W2_KEY_PINS.json"
    if w2_pin.exists():
        w2 = json.loads(w2_pin.read_text())["members"]
        for pin in w2.values():
            if sha(Path(pin["path"])) != pin["sha256"]:
                raise ValueError("frozen W2 join changed")
    else:
        import pyarrow.parquet as pq
        w2 = {}
        for member in ("kadid_train", "kadid_select", "tid2013"):
            original = Path("/var/tmp/rev4-featbank/bank") / member / "keys.parquet"
            table = pq.read_table(original, columns=["pair_key", "dist_path"])
            target = bundle / "w2-keys" / f"{member}.parquet"
            target.parent.mkdir(exist_ok=True)
            with target.open("xb") as stream:
                pq.write_table(table, stream)
            w2[member] = dict(path=str(target), sha256=sha(target), rows=len(table),
                original_path=str(original), original_sha256=sha(original))
        write(w2_pin, dict(schema="v40-w2-label-free-keys-v1", members=w2))
    # Keep the tested fit runtime intact. Freeze assessment separately, including
    # import dependencies absent from the deliberately small fit-only archive.
    assessment = bundle / "assessment-runtime"
    assessment.mkdir()
    inventory = {}
    for path in sorted((source / "scripts").rglob("*.py")):
        rel = path.relative_to(source)
        dest = assessment / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        inventory[str(rel)] = sha(dest)
    for path in sorted((runtime / "benchmarks").iterdir()):
        dest = assessment / "benchmarks" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        inventory[str(dest.relative_to(assessment))] = sha(dest)
    write(
        bundle / "ASSESSMENT_SOURCE.json",
        dict(
            schema="v40-assessment-source-freeze-v1",
            source_commit=source_commit,
            files=inventory,
            labels_read=False,
        ),
    )
    with tarfile.open(bundle / "assessment-program.tar.gz", "w:gz") as tar:
        for rel in sorted(inventory):
            raw = (assessment / rel).read_bytes()
            item = tarfile.TarInfo(rel)
            item.size, item.mode, item.mtime = len(raw), 0o644, 0
            tar.addfile(item, io.BytesIO(raw))
    decisions = source / "benchmarks/e29_e31_e32_shared_control_decision_2026-10-07.md"
    (bundle / "CONTROL_DECISION.md").write_bytes(decisions.read_bytes())
    caps = json.loads((bundle / "jobset_caps.json").read_text())
    for entry in caps.values():
        entry["build_commit"] = source_commit
    (bundle / "jobset_caps.pre-freeze.json").write_bytes(
        (bundle / "jobset_caps.json").read_bytes()
    )
    (bundle / "jobset_caps.json").write_text(json.dumps(caps, indent=2) + "\n")
    smokes = []
    selection = json.loads((bundle / "SMOKE_SELECTION.json").read_text())
    if set(selection) != {"control", "hb4", "hc4", "palette", "uh4"}:
        raise ValueError("exact arm smoke selection required")
    for arm in selection:
        for mode in ("bounded", "first-epoch"):
            use_attempt = selection[arm][mode]
            folder = bundle / f"smoke-{arm}-{mode}-{use_attempt}"
            files = list(folder.glob("*_PATH_PASS.json"))
            if len(files) != 1:
                raise ValueError("missing executor smoke")
            r = json.loads(files[0].read_text())
            if (
                r["status"] != "PASS"
                or r["program_sha"] != pins["program_sha"]
                or r["memory_peak_bytes"] >= 6 * 1024**3
                or not r["extraction_verified"]
            ):
                raise ValueError("unverified/out-of-cap executor smoke")
            smokes.append(
                dict(
                    arm=arm,
                    mode=mode,
                    receipt=str(files[0].relative_to(bundle)),
                    sha256=sha(files[0]),
                    memory_peak_bytes=r["memory_peak_bytes"],
                )
            )
    write(
        bundle / "EXECUTOR_SMOKES.json",
        dict(schema="v40-ten-executor-smokes-v1", smokes=smokes),
    )
    files = [
        "program.tar.gz",
        "build-meta.json",
        "PACKAGE_PINNED.json",
        "jobset_caps.json",
        "ASSESSMENT_SOURCE.json",
        "assessment-program.tar.gz",
        "EXECUTOR_SMOKES.json",
        "SMOKE_SELECTION.json",
        "HARVEST_REFUSALS.json",
        "CONTROL_DECISION.md",
        "E30_COMPLETE_PINS.json",
        "v40-fit-contract.json",
        "WORKER_BUILD.json",
        "W2_KEY_PINS.json",
        "SOURCE_BINDINGS.json",
        "parity-kadid/PARITY.json",
        "parity-tid2013/PARITY.json",
        "IMAGE_RECIPE.json",
        "bin/inspect_qualified_checkpoint",
        "harvest_driver_v40.py",
        "postfit.sh",
        "upiq380-fit/owner_disposition.json",
        "E31_OWNER_ADMISSION.json",
    ]
    files += [s["receipt"] for s in smokes]
    files += [str(Path(v["path"]).relative_to(bundle)) for v in w2.values()]
    common = {rel: sha(bundle / rel) for rel in files}
    for jobset, manifest_sha in pins["manifests"].items():
        manifest = f"fit-manifest-{jobset}.json"
        jobs = json.loads((bundle / manifest).read_text())
        if sha(bundle / manifest) != manifest_sha or len(jobs) != (
            80 if "-e29-" in jobset else 40
        ):
            raise ValueError("registered grid changed")
        data_file = (
            "palette-fit-data.tar.gz" if "-e32-" in jobset else
            "e31-fit-data.tar.gz" if "-e31-" in jobset else "e29-fit-data.tar.gz"
        )
        data_sha = sha(bundle / data_file)
        if data_sha not in pins["data_shas"] or any(
            j["kind"]["data_sha"] != data_sha for j in jobs
        ):
            raise ValueError("registered transport changed")
        identities = dict(
            program_sha=pins["program_sha"],
            data_sha=data_sha,
            data_file=data_file,
            image=pins["image"],
            image_id=pins["image_id"],
            worker_build=pins["worker_build"],
            source_commit=source_commit,
            zenmetrics_commit=metrics_commit,
            files={**common, manifest: manifest_sha, data_file: data_sha},
        )
        write(
            bundle / f"AUTHORIZATION_REQUIRED-{jobset}.json",
            dict(identities=identities),
        )
        write(
            bundle / f"AUTHORIZATION_TEMPLATE-{jobset}.json",
            dict(
                schema="v40-coordinator-launch-v1",
                coordinator_message="",
                reviewed=False,
                source_landed=False,
                pins_pushed=False,
                E30_completed=False,
                control_choice_frozen=False,
                identities=identities,
            ),
        )
    for mode in ("hdr", "external", "upiq"):
        write(
            bundle / f"EXPOSURE_TEMPLATE-{mode}.json",
            dict(
                schema="v40-assessment-exposure-freeze-v1",
                mode=mode,
                label_read_authorized=False,
                coordinator_message="",
                assessment_source_sha256=sha(
                    assessment / "scripts/rev4_featpot/v40_panels.py"
                ),
                program_sha256=pins["program_sha"],
                control_pins_sha256=None,
                objects={},
                tables={},
                required_action="Freeze exact assessment declarations/payload/tool pins after complete fits; no protected population is permitted.",
            ),
        )
    # These wrappers run only on an explicit invocation. No authorization file
    # or execution process is created by this preparation owner.
    wrappers = {
        "launch.py": "v40_launch.py",
        "score.py": "v40_score.py",
        "panels.py": "v40_panels.py",
    }
    for filename, owner in wrappers.items():
        directory = "runtime" if owner == "v40_launch.py" else "assessment-runtime"
        text = f'''#!/usr/bin/env python3
import os, runpy, sys
from pathlib import Path
b = Path(__file__).resolve().parent
os.environ.setdefault("REV4_V2_BIN_DIR", str(b / "bin"))
os.environ.setdefault("ZEN_PANEL_BIN", str(b / "bin/panel"))
os.environ.setdefault("TMPDIR", str(Path.home() / "tmp/v40"))
r = b / "{directory}"
sys.path[:0] = [str(r), str(r / "scripts"), str(r / "scripts/rev4_featpot")]
runpy.run_path(str(r / "scripts/rev4_featpot/{owner}"), run_name="__main__")
'''
        with (bundle / filename).open("x") as f:
            f.write(text)
        (bundle / filename).chmod(0o755)
    d1 = bundle / "v2d1"
    upiq = bundle / "upiq380-fit"
    write(
        bundle / "LOCAL_ROOTS.json",
        dict(
            v2e29=str((bundle / "v2e29").resolve()),
            v2e32=str((bundle / "v2e32").resolve()),
            v2d1=str(d1.resolve()),
            upiq380_fit_only=str(upiq),
            development_staged=False,
        ),
    )
    print(
        "PASS: local runtime, ten smoke pins, four inert authorization templates, owner-disposed E31"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--metrics-commit", required=True)
    a = p.parse_args()
    freeze(a.bundle, a.source, a.source_commit, a.metrics_commit)


if __name__ == "__main__":
    main()
