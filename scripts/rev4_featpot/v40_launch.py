"""Append reviewed V40 work to the existing fleet queue after explicit approval."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from shippath10_launch import sha

JOBSETS = tuple(f"fitv40-{s}-20261007" for s in ("control", "e29", "e32", "e31"))


def placement(entry, fleet_hosts=None):
    """Require capacity on an exact registered placement key before queueing.

    launch_v2.sh uses exact keys, including any SSH qualification. The local
    amendment binds private fleet keys; fresh packages can use public aliases.
    """
    fleet = (
        {"i265", "i270", "r3500", "r3800x", "tower"}
        if fleet_hosts is None
        else set(fleet_hosts)
    )
    hosts = entry.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise ValueError("jobset placement refused: empty fleet host map")
    if any(type(slots) is not int or slots <= 0 for slots in hosts.values()):
        raise ValueError(
            "jobset placement refused: positive integer host slots required"
        )
    if not fleet.intersection(hosts):
        raise ValueError("jobset placement refused: no registered fleet host")


def gate(bundle, jobset):
    if jobset not in JOBSETS:
        raise PermissionError("unregistered V40 jobset")
    authorization = bundle / f"LAUNCH_AUTHORIZATION-{jobset}.json"
    if not authorization.is_file():
        raise PermissionError(
            "NOT AUTHORIZED: exact coordinator authorization file required"
        )
    a = json.loads(authorization.read_text())
    required = json.loads(
        (bundle / f"AUTHORIZATION_REQUIRED-{jobset}.json").read_text()
    )
    if (
        a.get("schema") != "v40-coordinator-launch-v1"
        or not a.get("coordinator_message")
        or any(
            a.get(k) is not True
            for k in (
                "source_landed",
                "pins_pushed",
                "reviewed",
                "E30_completed",
                "control_choice_frozen",
            )
        )
        or a.get("identities") != required["identities"]
    ):
        raise PermissionError(
            "authorization differs from reviewed V40/E30/control freeze"
        )
    ids = required["identities"]
    for filename, pin in ids["files"].items():
        if sha(bundle / filename) != pin:
            raise ValueError(f"V40 input changed: {filename}")
    caps = json.loads(Path("/var/tmp/fitv2/jobset_caps.json").read_text())
    if (
        caps.get(jobset)
        != json.loads((bundle / "jobset_caps.json").read_text())[jobset]
    ):
        raise ValueError("live jobset cap differs from reviewed local entry")
    fleet_hosts = None
    if "CAPS_FIX.json" in ids["files"]:
        fleet_hosts = json.loads((bundle / "CAPS_FIX.json").read_text())["fleet_hosts"]
        if not isinstance(fleet_hosts, list) or not fleet_hosts:
            raise ValueError("registered fleet placement keys required")
    placement(caps[jobset], fleet_hosts)
    # Use the existing complete E30 freeze owner; no label payload is read.
    from e30_four_source import completed_control_pins

    proof = completed_control_pins(
        Path("/var/tmp/rev4-featpot/e30-results/cells"),
        Path("/mnt/v/output/zensim/shippath11-2026-10-07"),
    )
    if proof["cell_count"] != 40:
        raise ValueError("complete E30 prerequisite missing")
    if jobset != JOBSETS[0]:
        from v40_score import complete

        control = Path("/var/tmp/rev4-featpot/v40-control-results")
        study = jobset.split("-")[1]
        cells = complete(
            bundle,
            study,
            control,
            control,
            bundle / "committed-tools",
            only_control=True,
        )["control"]
        frozen = json.loads((bundle / "V40_CONTROL_PINS.json").read_text())
        if (
            frozen.get("control_choice") != "fresh-matched-v40"
            or frozen.get("program_sha") != ids["program_sha"]
        ):
            raise ValueError("fresh V40 control must be frozen before arm launch")
        for (fold, seed), cell in cells.items():
            if frozen["cells"].get(f"{fold}_s{seed}") != dict(
                result_sha256=sha(cell / "result.json"),
                bake_sha256=sha(cell / "refit/last.bin"),
            ):
                raise ValueError("fresh V40 control changed")
    return ids


def launch(bundle, jobset):
    ids = gate(bundle, jobset)  # No side effects before approval and all pins.
    actual = subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", ids["image"]], text=True
    ).strip()
    if actual != ids["image_id"]:
        raise ValueError("local image changed")
    queue = Path("/var/tmp/fitv2/fleet_queue")
    if any(
        line.split() and line.split()[0] == jobset
        for line in queue.read_text().splitlines()
    ):
        raise ValueError("jobset already queued")
    subprocess.run(["docker", "push", ids["image"]], check=True)
    control = bundle / f"control-{jobset}.json"
    control.write_text(
        json.dumps(
            dict(
                paused=False,
                drain=False,
                note="reviewed V40; explicit coordinator approval",
            )
        )
        + "\n"
    )
    subprocess.run(
        [
            "bash",
            "-c",
            """set -euo pipefail
. ~/.config/zen/s3env.sh >/dev/null 2>&1
s5cmd --endpoint-url "$EP" cp "$1/fit-manifest-$2.json" "s3://zentrain/jobs/$2/manifest.json"
s5cmd --endpoint-url "$EP" cp "$1/control-$2.json" "s3://zentrain/jobs/$2/control.json"
s5cmd --endpoint-url "$EP" cp "$1/$4" "s3://zentrain/jobs/$2/inputs/$3"
""",
            "--",
            str(bundle),
            jobset,
            ids["data_sha"],
            ids["data_file"],
        ],
        check=True,
    )
    queue.with_name(f"fleet_queue.{jobset}.before").write_bytes(queue.read_bytes())
    staging = queue.with_name(f"fleet_queue.{jobset}.new")
    staging.write_text(
        queue.read_text().rstrip()
        + f"\n{jobset} {bundle / f'fit-manifest-{jobset}.json'} {ids['image']}\n"
    )
    os.replace(staging, queue)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--jobset", choices=JOBSETS, required=True)
    a = p.parse_args()
    launch(a.bundle, a.jobset)


if __name__ == "__main__":
    main()
