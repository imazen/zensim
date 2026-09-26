"""Archive current remote mains for a clean, provenance-pinned diagnostic build.

Run after `jj git fetch` in every named source repository. This never reads a
working-copy source file: git archives immutable remote-main commits. The
single bake_dial_refit overlay is the lane's unlanded diagnostic extension.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/var/tmp/rev4-featpot/main_build_20260924")
ZEN = (Path.home() / "work/zen")
REPOS = {
    "zensim": (ZEN / "zensim", "main"),
    "zenmetrics": (ZEN / "zenmetrics", "master"),
    "zenjpeg": (ZEN / "zenjpeg", "main"),
    "zenwebp": (ZEN / "zenwebp", "main"),
    "zenavif": (ZEN / "zenavif", "main"),
    "zenjxl": (ZEN / "zenjxl", "main"),
    "zenpng": (ZEN / "zenpng", "main"),
    "zenanalyze": (ZEN / "zenanalyze", "main"),
    "zenpixels": (ZEN / "zenpixels", "main"),
    "jxl-encoder": (ZEN / "jxl-encoder", "main"),
    "zenjxl-decoder": (ZEN / "zenjxl-decoder", "main"),
    "zenresize": (ZEN / "zenresize", "main"),
    "zenbench": (ZEN / "zenbench", "main"),
    "zenextras": (ZEN / "zenextras", "main"),
    "butteraugli": ((Path.home() / "work/butteraugli"), "main"),
}
LANE = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*argv: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, (source, branch) in REPOS.items():
        subprocess.run(["jj", "git", "fetch"], cwd=source, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        commit = command("git", "rev-parse", "--verify", f"refs/remotes/origin/{branch}", cwd=source)
        dest = ROOT / ("butteraugli" if name == "butteraugli" else f"zen/{name}")
        if dest.exists():
            raise FileExistsError(f"refuse to overlay existing clean snapshot: {dest}")
        dest.mkdir(parents=True)
        archive = subprocess.Popen(["git", "archive", "--format=tar", commit],
                                   cwd=source, stdout=subprocess.PIPE)
        extract = subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout,
                                 check=False)
        assert archive.stdout is not None
        archive.stdout.close()
        if archive.wait() or extract.returncode:
            raise RuntimeError(f"{name}: git archive or tar failed")
        entries[name] = {"branch": branch, "commit": commit, "source": str(source),
                         "snapshot": str(dest)}
        print(json.dumps({"repo": name, "branch": branch, "commit": commit}), flush=True)
    patch = ROOT / "bake_dial_refit_lane.patch"
    diff = subprocess.run(["jj", "diff", "--from", "main@origin", "--to", "@",
                           "--git", "--", "zensim-validate/src/bin/bake_dial_refit.rs"],
                          cwd=LANE, stdout=subprocess.PIPE, check=True)
    patch.write_bytes(diff.stdout)
    if not diff.stdout:
        raise ValueError("missing diagnostic-fit Rust overlay")
    snapshot = ROOT / "zen/zensim"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=snapshot, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=snapshot, check=True)
    config = snapshot / ".cargo/config.toml"
    if config.exists():
        raise FileExistsError(f"refuse to replace tracked Cargo config: {config}")
    config.parent.mkdir(exist_ok=True)
    config.write_text(
        '[patch."https://github.com/imazen/zenanalyze"]\n'
        'zenpredict = { path = "../zenanalyze/zenpredict" }\n'
        'zenpredict-bake = { path = "../zenanalyze/zenpredict-bake" }\n'
        '[patch."https://github.com/imazen/zenmetrics"]\n'
        'zenstats = { path = "../zenmetrics/crates/zenstats" }\n'
        '[patch."https://github.com/imazen/zenresize"]\n'
        'zenresize = { path = "../zenresize" }\n'
        '[patch."https://github.com/imazen/zenbench"]\n'
        'zenbench = { path = "../zenbench" }\n'
        '[patch."https://github.com/imazen/zenextras"]\n'
        'zenexr = { path = "../zenextras/zenexr" }\n'
    )
    meta = {"schema": "rev4-featpot-main-build-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "repos": entries, "lane_overlay_patch": str(patch),
            "lane_overlay_sha256": sha(patch),
            "cargo_config_sha256": sha(config),
            "purpose": "main snapshots plus unlanded bake_dial_refit diagnostic extension"}
    output = ROOT / "build_meta.json"
    output.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"build_meta": str(output), "sha256": sha(output),
                      "repos": len(entries), "overlay_sha256": sha(patch)}), flush=True)


if __name__ == "__main__":
    main()
