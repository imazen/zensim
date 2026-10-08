"""Rehearse valid V40 authorizations through gate() only, without fleet actions.

Run inside the frozen image, network disabled and production inputs read-only.
Synthetic authorization files exist only in the separately named test directory.
E30 uses its actual installed cells; missing fresh controls must block every arm.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=False)
    probe = args.dest / "SYNTHETIC_AUTHORIZATION_PROBE"
    probe.mkdir()
    for path in args.bundle.iterdir():
        (probe / path.name).symlink_to(path, target_is_directory=path.is_dir())
    runtime = args.bundle / "runtime"
    sys.path[:0] = [str(runtime), str(runtime / "scripts"), str(runtime / "scripts/rev4_featpot")]
    os.environ["REV4_V2_BIN_DIR"] = str(args.bundle / "bin")
    os.environ["ZEN_PANEL_BIN"] = str(args.bundle / "bin/panel")
    import v40_launch as owner
    caps = (args.bundle / "jobset_caps.json").read_text()
    real_read = Path.read_text

    def read(path, *a, **kw):
        if str(path) == "/var/tmp/fitv2/jobset_caps.json":
            return caps
        return real_read(path, *a, **kw)

    results = []
    # No launch() call: the real E30 owner invokes only the checkpoint inspector.
    with patch.object(Path, "read_text", read):
        for jobset in owner.JOBSETS:
            required = json.loads((probe / f"AUTHORIZATION_REQUIRED-{jobset}.json").read_text())
            approval = dict(schema="v40-coordinator-launch-v1",
                            coordinator_message="SYNTHETIC GATE PROBE; DOES NOT AUTHORIZE FLEET",
                            source_landed=True, pins_pushed=True, reviewed=True,
                            E30_completed=True, control_choice_frozen=True,
                            identities=required["identities"])
            authorization = probe / f"LAUNCH_AUTHORIZATION-{jobset}.json"
            authorization.write_text(json.dumps(approval))
            try:
                owner.gate(probe, jobset)
            except ValueError as error:
                if jobset == owner.JOBSETS[0] or "INCOMPLETE: missing control/kadid/s0" not in str(error):
                    raise
                results.append(dict(jobset=jobset, status="PASS", outcome=str(error)))
            else:
                if jobset != owner.JOBSETS[0]:
                    raise AssertionError("arm gate passed before the fresh control")
                results.append(dict(jobset=jobset, status="PASS", outcome="valid control gate passed"))
            # The positive path was reached; now prove each boolean/identity binds.
            for mutation in ("reviewed", "E30_completed", "identities"):
                changed = {**approval, mutation: {} if mutation == "identities" else False}
                authorization.write_text(json.dumps(changed))
                try:
                    owner.gate(probe, jobset)
                except PermissionError:
                    results.append(dict(jobset=jobset, status="PASS", refusal=mutation))
                else:
                    raise AssertionError(f"authorization mutation accepted: {mutation}")
            authorization.unlink()
    with (args.dest / "RESULT.json").open("x") as stream:
        stream.write(json.dumps(dict(status="PASS", fleet_actions=0, cases=results), indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
