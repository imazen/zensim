"""Check the amended shipped launcher with synthetic controls; no fleet actions."""

import argparse
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    b = args.bundle
    args.dest.mkdir(parents=True, exist_ok=False)
    probe = args.dest / "SYNTHETIC_GATE_ONLY"
    probe.mkdir()
    for path in b.iterdir():
        if (
            path.name.startswith("LAUNCH_AUTHORIZATION-")
            or path.name == "V40_CONTROL_PINS.json"
        ):
            continue
        (probe / path.name).symlink_to(path, target_is_directory=path.is_dir())
    runtime = b / "runtime"
    sys.path[:0] = [
        str(runtime),
        str(runtime / "scripts"),
        str(runtime / "scripts/rev4_featpot"),
    ]
    # Exercise the exact wrapper routing; suppress only the overlay's CLI entry.
    original_run = runpy.run_path
    routed = []

    def load_overlay(path, **kwargs):
        assert Path(path) == b / "launch-runtime/v40_launch.py"
        assert kwargs == dict(run_name="__main__")
        routed.append(path)
        return original_run(path, run_name="capsfix_probe")

    with patch.object(runpy, "run_path", load_overlay):
        original_run(str(b / "launch.py"), run_name="capsfix_probe")
    assert len(routed) == 1
    owner = original_run(routed[0], run_name="capsfix_probe")
    import e30_four_source
    import v40_score

    caps = json.loads((b / "jobset_caps.json").read_bytes())
    fleet_hosts = json.loads((b / "CAPS_FIX.json").read_bytes())["fleet_hosts"]
    for jobset in owner["JOBSETS"]:
        assert set(caps[jobset]["hosts"]) == set(fleet_hosts)
        for host in fleet_hosts:
            owner["placement"](
                {"hosts": {host: caps[jobset]["hosts"][host]}}, fleet_hosts
            )
    cells, pins = {}, {}
    for fold in ("kadid", "tid2013", "konfig", "cid22_a25"):
        for seed in range(10):
            cell = args.dest / "synthetic-cells" / f"{fold}_s{seed}"
            (cell / "refit").mkdir(parents=True)
            (cell / "result.json").write_text(
                json.dumps(dict(scope="SYNTHETIC ONLY", fold=fold, seed=seed))
            )
            (cell / "refit/last.bin").write_bytes(f"synthetic {fold} {seed}".encode())
            cells[fold, seed] = cell
            pins[f"{fold}_s{seed}"] = dict(
                result_sha256=owner["sha"](cell / "result.json"),
                bake_sha256=owner["sha"](cell / "refit/last.bin"),
            )
    ids0 = json.loads(
        (b / f"AUTHORIZATION_REQUIRED-{owner['JOBSETS'][0]}.json").read_text()
    )["identities"]
    (probe / "V40_CONTROL_PINS.json").write_text(
        json.dumps(
            dict(
                control_choice="fresh-matched-v40",
                program_sha=ids0["program_sha"],
                cells=pins,
            )
        )
    )
    original_read = Path.read_text

    def read(path, *a, **kw):
        if str(path) == "/var/tmp/fitv2/jobset_caps.json":
            return json.dumps(caps)
        return original_read(path, *a, **kw)

    checks = []
    with (
        patch.object(Path, "read_text", read),
        patch.object(
            e30_four_source, "completed_control_pins", return_value=dict(cell_count=40)
        ),
        patch.object(
            v40_score, "complete", return_value=dict(control=cells)
        ) as control,
        patch("subprocess.run", side_effect=AssertionError("fleet action")),
        patch("subprocess.check_output", side_effect=AssertionError("fleet read")),
    ):
        for jobset in owner["JOBSETS"]:
            required = json.loads(
                (probe / f"AUTHORIZATION_REQUIRED-{jobset}.json").read_text()
            )
            approval = dict(
                schema="v40-coordinator-launch-v1",
                coordinator_message="SYNTHETIC ONLY; NO FLEET AUTHORITY",
                source_landed=True,
                pins_pushed=True,
                reviewed=True,
                E30_completed=True,
                control_choice_frozen=True,
                identities=required["identities"],
            )
            path = probe / f"LAUNCH_AUTHORIZATION-{jobset}.json"
            path.write_text(json.dumps(approval))
            assert owner["gate"](probe, jobset) == required["identities"]
            checks.append(
                dict(
                    jobset=jobset,
                    outcome="amended caps and all actual file hashes passed; synthetic prerequisite/control",
                )
            )
            if jobset != owner["JOBSETS"][0]:
                control.side_effect = ValueError("INCOMPLETE: missing control/kadid/s0")
                try:
                    owner["gate"](probe, jobset)
                except ValueError as exc:
                    assert str(exc) == "INCOMPLETE: missing control/kadid/s0"
                else:
                    raise AssertionError("fresh-control requirement bypassed")
                control.side_effect = None
            path.unlink()
    # Actual harvest configuration runs offline, with both caps reads forbidden.
    spec = importlib.util.spec_from_file_location(
        "capsfix_harvest", b / "harvest_driver_v40.py"
    )
    harvest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harvest)
    original_open = Path.open

    def open_checked(path, *a, **kw):
        if path.name == "jobset_caps.json":
            raise AssertionError("postfit/harvest opened placement caps")
        return original_open(path, *a, **kw)

    with patch.object(Path, "open", open_checked):
        for jobset in owner["JOBSETS"]:
            with patch.object(
                sys,
                "argv",
                [
                    "harvest_driver_v40.py",
                    "--bundle",
                    str(b),
                    jobset,
                    str(b / f"fit-manifest-{jobset}.json"),
                    "--check-config",
                ],
            ):
                harvest.main()
    postfit = (b / "postfit.sh").read_text()
    assert "jobset_caps" not in postfit
    report = dict(
        status="PASS",
        scope="synthetic gate controls with actual amended packet hashes and offline harvest configuration",
        fleet_actions=0,
        production_authorizations_created=0,
        real_control_completion_claimed=False,
        caps_sha256=owner["sha"](b / "jobset_caps.json"),
        launcher_sha256=owner["sha"](b / "launch-runtime/v40_launch.py"),
        gates=checks,
        harvest_caps_opens=0,
        placement_keys_verified=len(fleet_hosts),
    )
    (args.dest / "CAPS_FIX_GATE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
