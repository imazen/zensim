"""Build the Rev4 diagnostic binaries from verified current-main snapshots.

Run this program through ~/tmp/devin/heavy. It refuses a stale main ref or
any resolved local/git source outside the immutable snapshot tree.
"""

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/var/tmp/rev4-featpot/main_build_20260924")
SOURCE = ROOT / "zen/zensim"
META = ROOT / "build_meta.json"
BINS = ("bake_dial_refit", "panel", "zensim_mlp_train")
LEGACY_BIN_DIR = Path("/var/tmp/rev4-featpot/target/debug")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_current_main_links(binaries: dict) -> dict:
    """Switch existing runners at the next heavy-job boundary.

    This function runs while the shared heavy lock is held, after all three
    binaries have been built and hashed. Existing jobs finish before this
    point; the next queued job sees the new executables at the old paths.
    """
    backup_dir = ROOT / "pre_current_main_binaries"
    backup_dir.mkdir(exist_ok=True)
    staged = {}
    for name, item in binaries.items():
        source = Path(item["path"])
        legacy = LEGACY_BIN_DIR / name
        backup = backup_dir / name
        if not source.is_file() or sha(source) != item["sha256"]:
            raise ValueError(f"{name}: clean-main binary changed before install")
        if legacy.is_symlink() and legacy.resolve() != source.resolve():
            raise ValueError(f"{name}: unexpected existing link {legacy}")
        if legacy.exists() and not legacy.is_symlink() and backup.exists():
            raise FileExistsError(f"refuse to replace pre-main backup: {backup}")
        staged[name] = (source, legacy, backup)

    for source, legacy, backup in staged.values():
        legacy.parent.mkdir(parents=True, exist_ok=True)
        if legacy.exists() and not legacy.is_symlink():
            shutil.copy2(legacy, backup)

    installed = {}
    for name, item in binaries.items():
        source, legacy, backup = staged[name]
        old_sha = sha(backup) if backup.exists() else None
        temp = legacy.with_name(f".{name}.current-main-{os.getpid()}")
        temp.symlink_to(source)
        os.replace(temp, legacy)
        installed[name] = {"legacy_path": str(legacy),
                           "current_main_path": str(source),
                           "current_main_sha256": item["sha256"],
                           "pre_main_backup": str(backup) if old_sha else None,
                           "pre_main_sha256": old_sha}
    return installed


def verify_remote_mains(meta: dict) -> None:
    for name, item in meta["repos"].items():
        subprocess.run(["jj", "git", "fetch"], cwd=item["source"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        current = subprocess.check_output(
            ["git", "rev-parse", f"refs/remotes/origin/{item['branch']}"],
            cwd=item["source"], text=True).strip()
        if current != item["commit"]:
            raise ValueError(f"{name}: main moved after snapshot {item['commit']} -> {current}")


def verify_home_disk_floor() -> None:
    available = shutil.disk_usage("/home").free
    if available < 20 * (1 << 30):
        raise ValueError(f"/home below 20 GiB free: {available / (1 << 30):.2f} GiB")


def main() -> None:
    meta = json.loads(META.read_text())
    if meta["schema"] != "rev4-featpot-main-build-v1":
        raise ValueError("unexpected source snapshot receipt")
    verify_home_disk_floor()
    verify_remote_mains(meta)
    env = dict(os.environ)
    env.update({"CARGO_TARGET_DIR": str(ROOT / "target"),
                "CARGO_HOME": "/var/tmp/rev4-featpot/cargo_home",
                "TMPDIR": "/var/tmp/rev4-featpot/tmp"})
    metadata = subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1"], cwd=SOURCE, env=env)
    (ROOT / "cargo_metadata.json").write_bytes(metadata)
    graph = json.loads(metadata)
    local = []
    git = []
    snapshots = [(name, Path(item["snapshot"]).resolve(), item["commit"])
                 for name, item in meta["repos"].items()]
    for package in graph["packages"]:
        source = package["source"]
        if source is None:
            manifest = Path(package["manifest_path"]).resolve()
            if not manifest.is_relative_to(ROOT):
                raise ValueError(f"{package['name']}: local source outside snapshots: {manifest}")
            matches = [(name, commit) for name, root, commit in snapshots
                       if manifest.is_relative_to(root)]
            if len(matches) != 1:
                raise ValueError(f"{package['name']}: no unique current-main snapshot for {manifest}")
            name, commit = matches[0]
            local.append({"name": package["name"], "manifest": str(manifest),
                          "source_repo": name, "current_main_commit": commit})
        elif source.startswith("git+"):
            repository = source.split("?", 1)[0].rsplit("/", 1)[-1].removesuffix(".git")
            resolved_commit = source.rsplit("#", 1)[-1]
            expected = meta["repos"].get(repository, {}).get("commit")
            if resolved_commit != expected:
                raise ValueError(f"{package['name']}: git source is not {repository} current main: {source}")
            git.append({"name": package["name"], "source": source,
                        "source_repo": repository,
                        "current_main_commit": expected})
    cmd = ["cargo", "build", "-p", "zensim-validate"]
    for binary in BINS:
        cmd.extend(["--bin", binary])
    subprocess.run(cmd, cwd=SOURCE, env=env, check=True)
    verify_home_disk_floor()
    verify_remote_mains(meta)
    binaries = {name: {"path": str(ROOT / "target/debug" / name),
                       "sha256": sha(ROOT / "target/debug" / name)} for name in BINS}
    links = install_current_main_links(binaries)
    result = {"schema": "rev4-featpot-current-main-binaries-v1",
              "built_utc": datetime.now(timezone.utc).isoformat(),
              "source_meta_sha256": sha(META), "cargo_metadata_sha256": sha(ROOT / "cargo_metadata.json"),
              "resolved_local_packages": local, "resolved_git_packages": git,
              "binaries": binaries, "legacy_path_links": links}
    output = ROOT / "binary_meta.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"binary_meta": str(output), "sha256": sha(output),
                      "resolved_local_packages": len(local), "binaries": binaries}), flush=True)


if __name__ == "__main__":
    main()
