"""E33 real-entry container smoke plus harvest verification of its output (registration section 10.1).

One declared cell of an E33 jobset, with an explicit local smoke budget, runs through the image's actual
`fit-cell-exec` entry (the V40 executor-smoke driver, `e28_executor_smoke.py`), under the fleet envelope
(one CPU, 6 GiB, no swap, no network). The executor's output blob is then checked by the program archive's
own harvest owner (`harvest_fit_cells.verify_blob`) against the manifest-bound program and inspector, with
local smokes allowed for verification only. No upload, queue or installation.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--jobset", choices=("control", "a", "c", "full"), required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--budget", default="2:128")
    p.add_argument("--inspector", type=Path, required=True)
    p.add_argument("--attempt", type=int, default=1)
    a = p.parse_args()
    b = a.bundle
    spec_doc = json.loads(next(b.glob(f"fit-spec-fite33-{a.jobset}-*.json")).read_text())
    cell = dict(spec_doc["cells"][a.index])
    argv = list(cell["argv"])
    key = f"{a.jobset}-{a.index}-{a.attempt}"
    # Same rooted layout as the declared cell (fit_paths: cells/ or confirm/cells/), under a smoke-only results dir.
    dest = Path(argv[argv.index("--dest") + 1])
    rel = dest.relative_to("/var/tmp/rev4-featpot")
    argv[argv.index("--dest") + 1] = str(Path("/var/tmp/rev4-featpot", f"e33-smoke-{key}", *rel.parts[1:]))
    argv += ["--local-smoke-budget", a.budget]
    spec_path = b / f"smoke-spec-{key}.json"
    with spec_path.open("x") as f:
        f.write(json.dumps({**spec_doc, "cells": [dict(name=cell["name"], argv=argv)]}, indent=2) + "\n")
    manifest = b / f"smoke-manifest-{key}.json"
    ctl = next(p for p in (b / "bin/zenfleet-ctl", b.parent / "bin-final/zenfleet-ctl") if p.is_file())
    subprocess.run([str(ctl), "declare-fits", "--spec", str(spec_path), "--out", str(manifest)], check=True)
    jobs = json.loads(manifest.read_text())
    assert len(jobs) == 1
    job = b / f"smoke-job-{key}.json"
    with job.open("x") as f:
        f.write(json.dumps(jobs[0], indent=2) + "\n")
    out = b / f"smoke-{key}"
    out.mkdir(exist_ok=False)
    scratch = b / f"container-scratch-{a.attempt}"
    scratch.mkdir(exist_ok=True)
    driver = Path(__file__).with_name("e28_executor_smoke.py")
    subprocess.run(["docker", "run", "--rm", "--network=none", "--user", f"{os.getuid()}:{os.getgid()}",
                    "--cpus=1", "--memory=6g", "--memory-swap=6g", "--entrypoint", "python3",
                    "-e", "TMPDIR=/scratch/tmp", "-e", "PYTHONPYCACHEPREFIX=/scratch/pycache",
                    "-e", "ZENSIM_SAMPLE_DIGEST=1", "-e", "ZEN_FIT_DATA_LOCAL=/e33-data.tar.gz",
                    "-v", f"{(b / 'e33-fit-data.tar.gz').resolve()}:/e33-data.tar.gz:ro",
                    "-v", f"{b}:/e33", "-v", f"{scratch}:/scratch", "-v", "/usr/bin/time:/e28-time:ro",
                    "-v", f"{driver}:/e33-driver.py:ro", a.image, "/e33-driver.py", "--mode", "bounded",
                    "--job", f"/e33/{job.name}", "--out", f"/e33/{out.name}", "--deadline", "900"], check=True)
    receipt = next(out.glob("*_PATH_PASS.json"))
    blob = next(out.glob("*_OUTPUT.tar.gz"))
    # Harvest with the program archive's own owner, exactly as the fleet harvester would.
    runtime = b / f"smoke-runtime-{key}"
    runtime.mkdir()
    with tarfile.open(b / "program.tar.gz") as tar:
        for name in ("harvest_fit_cells.py", "qualified_fit_contract.py", "fit_paths.py"):
            (runtime / name).write_bytes(tar.extractfile(name).read())
    sys.path.insert(0, str(runtime))
    from harvest_fit_cells import verify_blob
    stage = b / f"smoke-harvest-{key}"
    stage.mkdir()
    verified = verify_blob(blob, stage, jobs[0]["cell"]["image_path"], jobs[0]["kind"],
                           program_archive=b / "program.tar.gz", checkpoint_inspector=a.inspector,
                           allow_local_smoke=True)
    record = json.loads(receipt.read_text())
    record.update(image=a.image, harvest=dict(status="VERIFIED", execution_contract=verified.get("execution_contract"),
                                              owner="program-archive harvest_fit_cells.verify_blob"))
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("status", "cell", "program_sha", "memory_peak_bytes", "harvest")}))


if __name__ == "__main__":
    main()
