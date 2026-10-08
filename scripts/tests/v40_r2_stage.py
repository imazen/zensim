"""Preserve the reviewed V40 bundle; stage only its pinned TRAIN transports."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--quiet-start", type=Path, required=True)
    args = parser.parse_args()
    boundary = json.loads(args.quiet_start.read_text())
    receipt = Path(boundary["path"])
    if receipt.stat().st_mtime_ns == boundary["mtime_ns"] or sha(receipt) == boundary["sha256"]:
        raise ValueError("SPEEDQ quiet window has not released; staging/builds forbidden")
    args.bundle.mkdir(parents=True, exist_ok=False)
    (args.bundle / "QUIET_WINDOW_RELEASE.json").write_text(json.dumps(dict(
        initial=boundary, released_mtime_ns=receipt.stat().st_mtime_ns,
        released_sha256=sha(receipt)), indent=2) + "\n")
    pins = json.loads((args.previous / "PACKAGE_PINNED.json").read_text())
    copied = {}
    for name in ("e29-fit-data.tar.gz", "palette-fit-data.tar.gz", "e31-fit-data.tar.gz"):
        source = args.previous / name
        if sha(source) not in pins["data_shas"]:
            raise ValueError(f"previous frozen transport changed: {name}")
        shutil.copy2(source, args.bundle / name)
        copied[name] = sha(args.bundle / name)
        if copied[name] != sha(source):
            raise ValueError(f"transport copy mismatch: {name}")
    # These roots remain frozen. No recursive traversal of the historical D1
    # root (which also contains excluded source data) is performed here.
    for name in ("v2d1", "v2e29", "v2e32", "upiq380-fit"):
        (args.bundle / name).symlink_to((args.previous / name).resolve(), target_is_directory=True)
    for name in ("E30_COMPLETE_PINS.json", "E31_OWNER_ADMISSION.json", "WORKER_BUILD.json", "IMAGE_RECIPE.json"):
        shutil.copy2(args.previous / name, args.bundle / name)
    (args.bundle / "bin").mkdir()
    shutil.copy2(args.previous / "bin/zenfleet-ctl", args.bundle / "bin/zenfleet-ctl")
    (args.bundle / "image-context").mkdir()
    for name in ("Dockerfile", "zenfleet-worker", "fleet-entrypoint.sh", "tmpdir_discipline.sh"):
        shutil.copy2(args.previous / "image-context" / name, args.bundle / "image-context" / name)
    worker = json.loads((args.bundle / "WORKER_BUILD.json").read_text())
    if sha(args.bundle / "image-context/zenfleet-worker") != worker["binary_sha256"]:
        raise ValueError("retained current-master worker changed")
    w2 = json.loads((args.previous / "W2_KEY_PINS.json").read_text())
    (args.bundle / "w2-keys").mkdir()
    for member, record in w2["members"].items():
        source = Path(record["path"])
        if sha(source) != record["sha256"]:
            raise ValueError("frozen label-free W2 join changed")
        destination = args.bundle / "w2-keys" / f"{member}.parquet"
        shutil.copy2(source, destination)
        record["path"] = str(destination)
    (args.bundle / "W2_KEY_PINS.json").write_text(json.dumps(w2, indent=2) + "\n")
    (args.bundle / "STAGED_FROM.json").write_text(json.dumps(dict(
        previous_bundle=str(args.previous), transport_shas=copied,
        historical_program_sha=pins["program_sha"],
        historical_image_id=pins["image_id"], labels_read=False), indent=2) + "\n")
    print(json.dumps(dict(status="PASS", bundle=str(args.bundle), data_shas=copied)))


if __name__ == "__main__":
    main()
