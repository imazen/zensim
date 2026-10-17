"""Admit one isolated nontraining Docker check under its caller's heavy scope."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    own = Path("/proc/self/cgroup").read_text().strip().split("/")[-1]
    scopes = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "list-units",
            "--type=scope",
            "--state=running",
            "--no-legend",
            "--no-pager",
        ],
        text=True,
    ).splitlines()
    other = [r for r in scopes if r.strip().startswith("run-") and r.split()[0] != own]
    if other:
        raise RuntimeError(
            "E36 runtime check refuses concurrent heavy scopes: " + repr(other)
        )
    if (
        spec["schema"] != "e36-nontraining-docker-check-v1"
        or spec["fits"] != 0
        or spec["optimizer_steps"] != 0
    ):
        raise ValueError("nontraining check declaration required")
    subprocess.run(spec["argv"], check=True)


if __name__ == "__main__":
    main()
