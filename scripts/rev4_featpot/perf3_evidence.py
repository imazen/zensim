#!/usr/bin/env python3
"""Seal PERF3 evidence and preserve its owned workspace before removal."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import perf2_evidence as base


def preserve(workspace, output, commit):
    workspace = workspace.resolve()
    assert workspace.name == "zensim--perf3"
    assert " codex-perf3 " in (workspace / ".workongoing").read_text()
    assert not output.resolve().is_relative_to(workspace)
    output.mkdir(exist_ok=False)
    tracked = set(subprocess.check_output(
        ["jj", "file", "list"], cwd=workspace, text=True).splitlines())
    records, targets = [], []
    for directory, dirs, files in os.walk(workspace, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if name in [".jj", ".git"]:
                dirs.remove(name)
            elif name in ["target", "target-native"]:
                indicators = [p for p in ["CACHEDIR.TAG", ".rustc_info.json",
                    "debug/.cargo-lock", "release/.cargo-lock"] if (path / p).exists()]
                assert indicators, ("unverified target directory", path)
                targets.append(dict(path=str(path.relative_to(workspace)), indicators=indicators))
                dirs.remove(name)
            elif path.is_symlink():
                files.append(name)
                dirs.remove(name)
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(workspace)
            if str(relative) in tracked or name in [".workongoing", ".git", ".jj"]:
                continue
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                target.symlink_to(os.readlink(path))
                records.append(dict(file=str(relative), symlink=os.readlink(path)))
            else:
                shutil.copy2(path, target)
                digest = base.sha(path)
                assert base.sha(target) == digest
                records.append(dict(file=str(relative), sha256=digest))
    base.write(output / "MANIFEST.json", dict(status="PASS", build_commit=commit,
        files=records, cargo_targets=targets))
    return dict(status="PASS", files=len(records), cargo_targets=len(targets))


def preserve_cargo(workspace, output, commit):
    workspace = workspace.resolve()
    assert workspace.name == "zensim--perf3"
    assert " codex-perf3 " in (workspace / ".workongoing").read_text()
    targets = [workspace / name for name in ["target", "target-native"]]
    for target in targets:
        assert target.is_dir() and not target.is_symlink()
        assert (target / "CACHEDIR.TAG").read_text().splitlines()[0] == "Signature: 8a477f597d28d172789f06886806bc55"
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            exe = (process / "exe").resolve(strict=True)
            cwd = (process / "cwd").resolve(strict=True)
            args = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (OSError, RuntimeError):
            continue
        assert not any(exe.is_relative_to(t) or cwd.is_relative_to(t) for t in targets), process.name
        if exe.name in ["cargo", "rustc"]:
            assert not cwd.is_relative_to(workspace), process.name
            assert not any(str(t) in arg for t in targets for arg in args), process.name
    root = output.parent
    binaries = {}
    for path in root.iterdir():
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                magic = stream.read(4)
            if magic == b"\x7fELF":
                binaries.setdefault(base.sha(path), []).append(path.name)
    bindings = []
    for artifact in sorted(root.glob("*.artifact.json")):
        record = json.loads(artifact.read_text())
        digest = record["binary_sha256"]
        assert digest in binaries, ("unpreserved measured binary", artifact, digest)
        bindings.append(dict(artifact=artifact.name, sha256=digest, preserved=binaries[digest]))
    assert bindings
    output.mkdir(exist_ok=False)
    data = []
    suffixes = {".bin", ".csv", ".tsv", ".json", ".jsonl", ".png", ".jpg", ".webp",
                ".parquet", ".npy", ".npz", ".svg", ".pdf", ".log", ".txt", ".data"}
    for target in targets:
        for path in target.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(target)
            if any(part in ["build", ".fingerprint", "incremental", "doc"] for part in relative.parts):
                continue
            if path.name == ".rustc_info.json" or path.suffix not in suffixes:
                continue
            saved = output / target.name / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
            digest = base.sha(path)
            assert base.sha(saved) == digest
            data.append(dict(file=str(saved.relative_to(output)), sha256=digest))
    base.write(output / "MANIFEST.json", dict(status="PASS", build_commit=commit,
        measured_binary_bindings=bindings, protected_data=data,
        targets=[str(p) for p in targets],
        scope="Every measured artifact digest has a frozen ELF outside Cargo; non-Cargo data suffixes are preserved. Other inactive Cargo output is reproducible cache."))
    return dict(status="PASS", measured_artifacts=len(bindings), data_files=len(data))


def seal(root, commit):
    assert not (root / base.MANIFEST).exists()
    rows, links, cache = [], [], {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            links.append(dict(file=str(path.relative_to(root)), target=os.readlink(path)))
            continue
        if not path.is_file():
            continue
        before = path.stat()
        key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if key not in cache:
            cache[key] = base.sha(path)
        after = path.stat()
        assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns), path
        rows.append(dict(file=str(path.relative_to(root)), bytes=before.st_size,
            sha256=cache[key]))
    base.write(root / base.MANIFEST, dict(status="PASS", build_commit=commit,
        time_utc=datetime.now(timezone.utc).isoformat(), files=rows, links=links))
    return dict(files=len(rows), bytes=sum(row["bytes"] for row in rows),
        manifest_sha256=base.sha(root / base.MANIFEST))


def verify(root, expected):
    manifest = root / base.MANIFEST
    assert base.sha(manifest) == expected, "manifest hash mismatch"
    record = json.loads(manifest.read_text())
    assert record["status"] == "PASS" and record["build_commit"]
    cache = {}
    for row in record["files"]:
        relative = Path(row["file"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = root / relative
        assert not path.is_symlink()
        stat = path.stat()
        assert stat.st_size == row["bytes"], row["file"]
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = base.sha(path)
        assert cache[key] == row["sha256"], row["file"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and not p.is_symlink()}
    assert actual == {r["file"] for r in record["files"]} | {base.MANIFEST}
    links = {str(p.relative_to(root)): os.readlink(p)
             for p in root.rglob("*") if p.is_symlink()}
    assert links == {r["file"]: r["target"] for r in record["links"]}
    return dict(status="PASS", files=len(record["files"]),
        manifest_sha256=expected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["seal", "verify", "preserve-ignored", "preserve-cargo"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workspace", type=Path)
    p.add_argument("--build-commit")
    p.add_argument("--manifest-sha256")
    a = p.parse_args()
    if a.mode == "verify":
        assert a.manifest_sha256
        result = verify(a.root, a.manifest_sha256)
    elif a.mode == "seal":
        assert a.build_commit
        result = seal(a.root, a.build_commit)
    else:
        assert a.workspace and a.build_commit
        result = (preserve_cargo if a.mode == "preserve-cargo" else preserve)(
            a.workspace, a.root, a.build_commit)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
