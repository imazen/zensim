"""Fresh R3 packet from the preserved reviewed packet and new local binaries."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--bin-dir", type=Path, required=True)
    p.add_argument("--producer", required=True)
    p.add_argument("--quiet-start", type=Path, required=True)
    p.add_argument("--quiet-override", type=Path, required=True)
    p.add_argument("--release-authority", type=Path, required=True)
    a = p.parse_args()
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("v40_r2_stage.py")),
            "--previous",
            str(a.previous),
            "--bundle",
            str(a.bundle),
            "--quiet-start",
            str(a.quiet_start),
            "--quiet-release",
            str(a.quiet_override),
        ],
        check=True,
    )
    b = a.bundle
    binaries = (
        "zensim_mlp_train",
        "bake_dial_refit",
        "panel",
        "predict_features_with_bake",
        "inspect_qualified_checkpoint",
    )
    for name in binaries:
        source = (
            a.bin_dir
            / ("examples/" if name == "inspect_qualified_checkpoint" else "")
            / name
        )
        shutil.copy2(source, b / "bin" / name)
    (b / "bin/examples").mkdir()
    shutil.copy2(
        b / "bin/inspect_qualified_checkpoint",
        b / "bin/examples/inspect_qualified_checkpoint",
    )
    shutil.copy2(a.previous / "v40-fit-contract.json", b / "v40-fit-contract.json")
    shutil.copy2(a.release_authority, b / "QUIET_RELEASE_AUTHORITY.md")
    boundary = json.loads((b / "QUIET_WINDOW_RELEASE.json").read_text())
    boundary.update(
        final_release_authority=str(a.release_authority),
        authority_sha256=sha(a.release_authority),
        heavy_wrapper="flock heavy.lock run-heavy --mem 16G --jobs 8",
        preparation_round=3,
    )
    (b / "QUIET_WINDOW_RELEASE.json").write_text(json.dumps(boundary, indent=2) + "\n")
    metadata = json.loads((a.previous / "build-meta.packer-input.json").read_text())
    for key in (
        "trainer_producer_commit",
        "zensim_source_commit",
        "zensim_lane_commit",
    ):
        metadata[key] = a.producer
    metadata.pop("source_bindings_sha256", None)
    metadata["note"] = (
        "Fresh native complete-inventory rebuild; unchanged registered budgets, selection and data archives. Source bindings, actual image and final-scoring rehearsal are reverified in round 3."
    )
    (b / "build-meta.packer-input.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (b / "PINNED_ARTIFACTS.json").write_text(
        json.dumps(
            dict(
                source_commit=a.producer,
                trainer_build_commit=a.producer,
                files={
                    "bin/" + name: dict(
                        sha256=sha(b / "bin" / name), producer_commit=a.producer
                    )
                    for name in binaries
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                status="PASS",
                bundle=str(b),
                producer=a.producer,
                prior_packet_preserved=True,
            )
        )
    )


if __name__ == "__main__":
    main()
