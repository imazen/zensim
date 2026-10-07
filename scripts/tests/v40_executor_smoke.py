"""Drive the existing real-entry container smoke, one local V40 arm at a time."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--arm", choices=("control", "hb4", "hc4", "palette", "uh4"), required=True)
    p.add_argument("--mode", choices=("bounded", "first-epoch"), required=True)
    p.add_argument("--attempt", type=int, default=1)
    a = p.parse_args()
    b = a.bundle
    study = {"control": "control", "hb4": "e29", "hc4": "e29", "palette": "e32", "uh4": "e31"}[a.arm]
    spec = json.loads((b / f"fit-spec-fitv40-{study}-20261007.json").read_text())
    cell = next(
        c
        for c in spec["cells"]
        if a.arm in ("control", "palette")
        or c["argv"][c["argv"].index("--spec") + 1].endswith(":" + a.arm)
    )
    key = f"{a.arm}-{a.mode}-{a.attempt}"
    cell["argv"][cell["argv"].index("--dest") + 1] = (
        f"/var/tmp/rev4-featpot/v40-smoke-{key}/cells/{cell['name']}"
    )
    if a.mode == "bounded":
        cell["argv"] += ["--local-smoke-budget", "2:128"]
    spec_path = b / f"smoke-spec-{key}.json"
    with spec_path.open("x") as file:
        file.write(json.dumps({**spec, "cells": [cell]}, indent=2) + "\n")
    manifest = b / f"smoke-manifest-{key}.json"
    subprocess.run(
        [
            str(b / "bin/zenfleet-ctl"),
            "declare-fits",
            "--spec",
            str(spec_path),
            "--out",
            str(manifest),
        ],
        check=True,
    )
    jobs = json.loads(manifest.read_text())
    assert len(jobs) == 1
    job = b / f"smoke-job-{key}.json"
    with job.open("x") as file:
        file.write(json.dumps(jobs[0], indent=2) + "\n")
    out = b / f"smoke-{key}"
    out.mkdir(exist_ok=False)
    scratch = b / "container-scratch"
    scratch.mkdir(exist_ok=True)
    driver = Path(__file__).with_name("e28_executor_smoke.py")
    data = {"palette": "palette-fit-data.tar.gz", "uh4": "e31-fit-data.tar.gz"}.get(a.arm, "e29-fit-data.tar.gz")
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--cpus=1",
            "--memory=6g",
            "--memory-swap=6g",
            "--entrypoint",
            "python3",
            "-e",
            "TMPDIR=/scratch/tmp",
            "-e",
            "PYTHONPYCACHEPREFIX=/scratch/pycache",
            "-e",
            "ZENSIM_SAMPLE_DIGEST=1",
            "-e",
            "ZEN_FIT_DATA_LOCAL=/v40-data.tar.gz",
            "-v",
            f"{(b / data).resolve()}:/v40-data.tar.gz:ro",
            "-v",
            f"{b}:/v40",
            "-v",
            f"{scratch}:/scratch",
            "-v",
            "/usr/bin/time:/e28-time:ro",
            "-v",
            f"{driver}:/v40-driver.py:ro",
            a.image,
            "/v40-driver.py",
            "--mode",
            a.mode,
            "--job",
            f"/v40/{job.name}",
            "--out",
            f"/v40/{out.name}",
            "--deadline",
            "300",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
