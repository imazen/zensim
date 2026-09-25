#!/usr/bin/env python3
"""Record what an extractor build resolved (CODEX_NOTE crates-on-main rule).

Reads $ROOT/snapshot_revs.tsv (sibling `git archive` commits), the zensim commit the
snapshot was built from, and `cargo metadata` of the built workspace: every non-registry
package must live under $ROOT/src (no git source, no path outside the snapshot).
Usage: write_build_meta.py <zensim-commit> <zensim-base-commit>
"""
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/var/tmp/restore-cuts")
src = ROOT / "src"
zensim_commit, base_commit = sys.argv[1:3]
repos = {"zensim": {"branch": "quarantine/claude/restore-cuts", "commit": zensim_commit,
                    "base_main_origin": base_commit, "snapshot": str(src / "zensim")}}
for line in (ROOT / "snapshot_revs.tsv").read_text().splitlines():
    name, branch, rev = line.split("\t")
    repos[name] = {"branch": branch, "commit": rev, "snapshot": str(src / name)}
meta = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--format-version", "1", "--features", "training,zen-decode,verify-all",
     "--filter-platform", "x86_64-unknown-linux-gnu"],
    cwd=src / "zensim/zensim-bench", env={**__import__("os").environ, "CARGO_HOME": str(ROOT / "cargo-home")},
    text=True))
local, registry, bad = [], 0, []
for p in meta["packages"]:
    if p["source"] is None:
        m = pathlib.Path(p["manifest_path"]).resolve()
        if not m.is_relative_to(src.resolve()):
            bad.append(p["name"])
        else:
            local.append({"name": p["name"], "version": p["version"], "repo": m.relative_to(src.resolve()).parts[0],
                          "manifest": str(m)})
    elif p["source"].startswith("git+"):
        bad.append(f"{p['name']} {p['source']}")
    else:
        registry += 1
assert not bad, f"packages outside the clean snapshot / git sources: {bad}"
out = {"repositories": repos, "local_packages": local, "registry_packages": registry,
       "cargo_lock_sha256": hashlib.sha256((src / "zensim/zensim-bench/Cargo.lock").read_bytes()).hexdigest()
       if (src / "zensim/zensim-bench/Cargo.lock").exists() else
       hashlib.sha256((src / "zensim/Cargo.lock").read_bytes()).hexdigest()}
(ROOT / "build_meta.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"BUILD_META repositories={len(repos)} local={len(local)} registry={registry} git_sources=0 "
      f"outside_snapshot=0")
