#!/usr/bin/env python3
"""Seal, verify, and preserve local PERF2 evidence without replacing existing data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

MANIFEST = "MEASUREMENT_MANIFEST.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, record):
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


def seal(root):
    output = root / MANIFEST
    assert not output.exists()
    rows, cache = [], {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        before = path.stat()
        key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if key not in cache:
            cache[key] = sha(path)
        after = path.stat()
        assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns), path
        rows.append(dict(file=str(path.relative_to(root)), bytes=before.st_size, sha256=cache[key]))
    write(output, dict(status="PASS", time_utc=datetime.now(timezone.utc).isoformat(), files=rows))
    return dict(files=len(rows), bytes=sum(row["bytes"] for row in rows), manifest_sha256=sha(output))


def verify(root, expected):
    manifest = root / MANIFEST
    assert sha(manifest) == expected, "manifest hash mismatch"
    record = json.loads(manifest.read_text())
    assert record["status"] == "PASS"
    cache = {}
    for row in record["files"]:
        relative = Path(row["file"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = root / relative
        stat = path.stat()
        assert stat.st_size == row["bytes"], row["file"]
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = sha(path)
        assert cache[key] == row["sha256"], row["file"]
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    assert actual == {row["file"] for row in record["files"]} | {MANIFEST}
    return dict(status="PASS", files=len(record["files"]),
                bytes=sum(row["bytes"] for row in record["files"]), manifest_sha256=expected)


def preserve_ignored(workspaces, output):
    assert all(not output.resolve().is_relative_to(workspace.resolve()) for workspace in workspaces)
    output.mkdir(exist_ok=False)
    records = []
    for root in workspaces:
        root = root.resolve()
        assert any(root.name.endswith(suffix) and root.name.removesuffix(suffix) == "zensim"
                   for suffix in ["--perf2", "--perf2base"])
        assert " codex-perf2 " in (root / ".workongoing").read_text()
        tracked = set(subprocess.check_output(["jj", "file", "list"], cwd=root, text=True).splitlines())
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in [".jj", ".git", "target", "target-native"]]
            for name in list(dirs):
                if (Path(directory) / name).is_symlink():
                    files.append(name)
                    dirs.remove(name)
            for name in files:
                path = Path(directory) / name
                relative = path.relative_to(root)
                if str(relative) in tracked or name == ".workongoing":
                    continue
                target = output / root.name / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if path.is_symlink():
                    target.symlink_to(os.readlink(path))
                    records.append(dict(workspace=root.name, file=str(relative), symlink=os.readlink(path)))
                else:
                    shutil.copy2(path, target)
                    digest = sha(path)
                    assert sha(target) == digest
                    records.append(dict(workspace=root.name, file=str(relative), sha256=digest))
    write(output / "MANIFEST.json", dict(status="PASS", files=records))
    return dict(status="PASS", files=len(records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["seal", "verify", "preserve-ignored"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--workspace", type=Path, action="append")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "seal":
        result = seal(args.root)
    elif args.mode == "verify":
        assert args.manifest_sha256
        result = verify(args.root, args.manifest_sha256)
    else:
        assert args.workspace
        result = preserve_ignored(args.workspace, args.root)
    if args.receipt:
        write(args.receipt, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
