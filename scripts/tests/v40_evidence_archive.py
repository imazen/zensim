"""Archive a completed, locally prepared V40 packet without following root links.

Run under the shared heavy lock. Only the caller-owned log tree and packet are
copied; frozen input roots remain links. Cargo cleanup requires mirrored binaries
and no process using the caller's target directory. No labels or fleet actions.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, record):
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")


def excluded(path):
    return (
        any(
            p.startswith("container-scratch-") or p == "__pycache__" for p in path.parts
        )
        or path.suffix == ".pyc"
    )


def archive(bundle, logs, mirror):
    if bundle.resolve() == mirror.resolve():
        raise ValueError("distinct mirror required")
    snapshot = bundle / "verification-logs"
    snapshot.mkdir(exist_ok=False)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--no-owner",
            "--no-group",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=archive.log",
            str(logs) + "/",
            str(snapshot) + "/",
        ],
        check=True,
    )
    files = {
        str(p.relative_to(snapshot)): sha(p)
        for p in sorted(snapshot.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }
    write(
        bundle / "VERIFICATION_LOGS.json",
        dict(schema="v40-verification-log-snapshot-v1", source=str(logs), files=files),
    )
    mirror.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--no-owner",
            "--no-group",
            "--exclude=container-scratch-*",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            str(bundle) + "/",
            str(mirror) + "/",
        ],
        check=True,
    )
    checked = {}
    links = {}
    for p in sorted(bundle.rglob("*")):
        rel = p.relative_to(bundle)
        if excluded(rel):
            continue
        if p.is_symlink():
            target = os.readlink(p)
            if not (mirror / rel).is_symlink() or os.readlink(mirror / rel) != target:
                raise ValueError(f"mirror link differs: {rel}")
            links[str(rel)] = target
        elif p.is_file():
            value = sha(p)
            if sha(mirror / rel) != value:
                raise ValueError(f"mirror bytes differ: {rel}")
            checked[str(rel)] = value
    record = dict(
        schema="v40-verified-mirror-v1",
        status="PASS",
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        source=str(bundle),
        mirror=str(mirror),
        files=checked,
        links=links,
        verification_log_files=len(files),
        followed_input_root_links=False,
    )
    write(bundle / "MIRROR_CHECK.json", record)
    write(mirror / "MIRROR_CHECK.json", record)
    print(
        json.dumps(
            dict(
                status="PASS",
                files=len(checked),
                links=len(links),
                verification_log_files=len(files),
            )
        )
    )


def cleanup(bundle, mirror, source):
    assert json.loads((bundle / "MIRROR_CHECK.json").read_text())["status"] == "PASS"
    bindings = json.loads((bundle / "SOURCE_BINDINGS.json").read_text())["binaries"]
    for name, record in bindings.items():
        assert (
            sha(bundle / "bin" / name) == sha(mirror / "bin" / name) == record["sha256"]
        )
    target = source / "target"
    assert not target.is_symlink() and (target / ".rustc_info.json").is_file()
    assert (target / "release/deps").is_dir()
    users = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = (process / "cmdline").read_bytes().split(b"\0")
            cwd = (process / "cwd").resolve()
            name = Path(os.fsdecode(argv[0])).name if argv[0] else ""
            active = str(target).encode() in b" ".join(argv)
            active |= name in ("cargo", "rustc", "clippy-driver") and cwd == source
            active |= any(
                str(fd.resolve()).startswith(str(target) + "/")
                for fd in (process / "fd").iterdir()
            )
            if active:
                users.append(int(process.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if users:
        raise ValueError(f"target still used by processes: {users}")
    allocated = int(
        subprocess.check_output(["du", "-s", "-B1", str(target)], text=True).split()[0]
    )
    shutil.rmtree(target)
    for name, record in bindings.items():
        assert sha(bundle / "bin" / name) == record["sha256"]
    record = dict(
        schema="v40-own-cargo-output-cleanup-v1",
        status="PASS",
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        removed=str(target),
        allocated_bytes_removed=allocated,
        active_target_users=users,
        retained_binary_bindings=bindings,
    )
    write(bundle / "CARGO_TARGET_CLEANUP.json", record)
    write(mirror / "CARGO_TARGET_CLEANUP.json", record)
    print(json.dumps(dict(status="PASS", allocated_bytes_removed=allocated)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("archive", "cleanup"))
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--mirror", type=Path, required=True)
    p.add_argument("--logs", type=Path)
    p.add_argument("--source", type=Path)
    a = p.parse_args()
    if a.mode == "archive" and a.logs:
        archive(a.bundle, a.logs, a.mirror)
    elif a.mode == "cleanup" and a.source:
        cleanup(a.bundle, a.mirror, a.source.resolve())
    else:
        p.error("archive requires --logs; cleanup requires --source")


if __name__ == "__main__":
    main()
