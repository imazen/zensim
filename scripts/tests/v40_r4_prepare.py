"""Rebind the unchanged V40 recipe to locally rebuilt tools; never enqueue."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

SOURCE = Path(__file__).resolve().parents[2]
METRICS = SOURCE.parent / "zenmetrics--v40"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--producer", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--metrics-commit", required=True)
    p.add_argument("--image", required=True)
    a = p.parse_args()
    b = a.bundle

    def run(args, cwd=SOURCE):
        subprocess.run([str(x) for x in args], cwd=cwd, check=True)

    run(["just", "v40-source-bindings", b, a.producer])
    metadata = json.loads((b / "build-meta.packer-input.json").read_text())
    metadata.update(
        fit_scripts_commit=a.source_commit,
        zensim_lane_commit=a.source_commit,
        zenmetrics_lane_commit=a.metrics_commit,
        source_bindings_sha256=sha(b / "SOURCE_BINDINGS.json"),
    )
    (b / "build-meta.packer-input.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    run(
        [
            "just",
            "v40-program",
            SOURCE,
            b / "bin",
            b / "build-meta.packer-input.json",
            b / "v40-fit-contract.json",
            b / "program.tar.gz",
        ],
        METRICS,
    )
    shutil.copy2(b / "program.tar.gz", b / "image-context/program.tar.gz")
    (b / "committed-tools").mkdir()
    with tarfile.open(b / "program.tar.gz") as archive:
        (b / "build-meta.json").write_bytes(
            archive.extractfile("build_meta.json").read()
        )
        for name in (
            "harvest_fit_cells.py",
            "qualified_fit_contract.py",
            "fit_paths.py",
        ):
            (b / "committed-tools" / name).write_bytes(archive.extractfile(name).read())
    run(
        [
            sys.executable,
            "scripts/rev4_featpot/v40_package.py",
            "specs",
            "--bundle",
            b,
            "--program",
            b / "program.tar.gz",
            "--e29-data",
            b / "e29-fit-data.tar.gz",
            "--palette-data",
            b / "palette-fit-data.tar.gz",
            "--ctl",
            b / "bin/zenfleet-ctl",
        ]
    )
    run(["just", "v40-image", b, a.image], METRICS)
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", a.image], text=True
    ).strip()
    (b / "IMAGE_ID.txt").write_text(image_id + "\n")
    jobsets = ["fitv40-" + s + "-20261007" for s in ("control", "e29", "e32", "e31")]
    pins = dict(
        data_shas=[
            sha(b / n)
            for n in (
                "e29-fit-data.tar.gz",
                "palette-fit-data.tar.gz",
                "e31-fit-data.tar.gz",
            )
        ],
        image=a.image,
        image_id=image_id,
        inspector_sha=sha(b / "bin/inspect_qualified_checkpoint"),
        program_sha=sha(b / "program.tar.gz"),
        worker_build=json.loads((b / "WORKER_BUILD.json").read_text()),
        manifests={j: sha(b / f"fit-manifest-{j}.json") for j in jobsets},
    )
    (b / "PACKAGE_PINNED.json").write_text(json.dumps(pins, indent=2) + "\n")
    (b / "SMOKE_SELECTION.json").write_text(
        json.dumps(
            {
                arm: {"bounded": 1, "first-epoch": 1}
                for arm in ("control", "hb4", "hc4", "palette", "uh4")
            },
            indent=2,
        )
        + "\n"
    )
    caps = json.loads((b.parent / "v40r3-2026-10-08/jobset_caps.json").read_text())
    (b / "jobset_caps.json").write_text(json.dumps(caps, indent=2) + "\n")
    owner = json.loads((b / "E31_OWNER_ADMISSION.json").read_text())

    # Refresh only job-level pins; the UPIQ source/manifest/decision stays frozen.
    owner["program_sha256"] = pins["program_sha"]
    owner["manifest_sha256"] = pins["manifests"]["fitv40-e31-20261007"]
    (b / "E31_OWNER_ADMISSION.json").write_text(json.dumps(owner, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", **pins)))


if __name__ == "__main__":
    main()
