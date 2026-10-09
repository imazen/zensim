"""Amend only V40 placement and launch metadata; never touch live caps or jobs."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import subprocess

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "scripts/rev4_featpot"))
launch_owner = importlib.import_module("v40_launch")
JOBSETS, placement = launch_owner.JOBSETS, launch_owner.placement


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def amend(bundle, reference_caps, reference_jobset, source_commit):
    if bundle.resolve() == Path("/var/tmp/fitv2").resolve():
        raise ValueError("live caps directory cannot be amended")
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise ValueError("exact source commit required")
    caps_path = bundle / "jobset_caps.json"
    if caps_path.resolve() == reference_caps.resolve():
        raise ValueError("reference/live caps cannot be amended")
    if caps_path.is_symlink():
        raise ValueError("packet caps must be a local regular file")
    reference_raw = reference_caps.read_bytes()
    reference = json.loads(reference_raw)[reference_jobset]
    placement(reference)
    caps_raw = caps_path.read_bytes()
    caps = json.loads(caps_raw)
    if set(caps) != set(JOBSETS):
        raise ValueError("exact four V40 caps entries required")
    for jobset in JOBSETS:
        caps[jobset]["hosts"] = reference["hosts"].copy()
        placement(caps[jobset])
    launcher_rel = "launch-runtime/v40_launch.py"
    launcher = (SOURCE / "scripts/rev4_featpot/v40_launch.py").read_bytes()
    wrapper = (bundle / "launch.py").read_bytes()
    old = b'runpy.run_path(str(r / "scripts/rev4_featpot/v40_launch.py"), run_name="__main__")'
    if wrapper.count(old) != 1:
        raise ValueError("expected original frozen launcher wrapper")
    wrapper = wrapper.replace(
        old,
        b'runpy.run_path(str(b / "launch-runtime/v40_launch.py"), run_name="__main__")',
    )
    record = dict(
        schema="v40-caps-placement-amendment-v1",
        source_commit=source_commit,
        reference_caps_sha256=digest(reference_raw),
        reference_jobset=reference_jobset,
        previous_caps_sha256=digest(caps_raw),
        corrected_caps_sha256=digest(encoded(caps)),
        launcher_sha256=digest(launcher),
        jobsets=list(JOBSETS),
        scope="placement and launch overlay only; original program, image, assessment and cell pins retained",
    )
    updates = {
        "jobset_caps.json": encoded(caps),
        launcher_rel: launcher,
        "launch.py": wrapper,
        "CAPS_FIX.json": encoded(record),
    }
    launch_pins = {rel: digest(raw) for rel, raw in updates.items()}
    for jobset in JOBSETS:
        required_name = f"AUTHORIZATION_REQUIRED-{jobset}.json"
        template_name = f"AUTHORIZATION_TEMPLATE-{jobset}.json"
        required = json.loads((bundle / required_name).read_bytes())
        template = json.loads((bundle / template_name).read_bytes())
        ids = required["identities"]
        if template["identities"] != ids or ids["files"]["jobset_caps.json"] != digest(
            caps_raw
        ):
            raise ValueError("original caps authorization pins differ")
        if (
            any(
                template[k] is not False
                for k in (
                    "reviewed",
                    "source_landed",
                    "pins_pushed",
                    "E30_completed",
                    "control_choice_frozen",
                )
            )
            or template["coordinator_message"]
        ):
            raise ValueError("inert authorization template required")
        ids["caps_fix_source_commit"] = source_commit
        ids["files"].update(launch_pins)
        template["identities"] = ids
        updates[required_name] = encoded(required)
        updates[template_name] = encoded(template)
    # Validate everything before mutation; retain every replaced record verbatim.
    backup = bundle / "capsfix-before-2026-10-09"
    if (
        backup.exists()
        or (bundle / launcher_rel).exists()
        or (bundle / "CAPS_FIX.json").exists()
    ):
        raise ValueError("fresh caps amendment required")
    backup.mkdir()
    for rel in updates:
        path = bundle / rel
        if path.exists():
            saved = backup / rel
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(path.read_bytes())
    for rel, raw in updates.items():
        path = bundle / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--reference-caps", type=Path, required=True)
    p.add_argument("--reference-jobset", required=True)
    p.add_argument("--source-commit", required=True)
    a = p.parse_args()
    rel = "scripts/rev4_featpot/v40_launch.py"
    committed = subprocess.check_output(
        ["jj", "file", "show", "-r", a.source_commit, rel], cwd=SOURCE
    )
    if committed != (SOURCE / rel).read_bytes():
        raise ValueError("launch overlay differs from declared source commit")
    print(
        json.dumps(
            amend(a.bundle, a.reference_caps, a.reference_jobset, a.source_commit),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
