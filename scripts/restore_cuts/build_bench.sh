#!/usr/bin/env bash
# Build the zenbench cost harness (`extract_paths_bench`) from the SAME clean snapshot tree as the
# extractor ($ROOT/src/zensim, synced by `build.sh cand`), with the workspace root's git-pinned
# sibling dependencies patched to the fetched-main snapshots (crates-on-main rule). Prints the
# executable path on the last line and records it in $ROOT/bin/bench_path.
set -euo pipefail
ROOT=/var/tmp/restore-cuts
export CARGO_HOME=$ROOT/cargo-home CARGO_TARGET_DIR=$ROOT/target-bench
cd "$ROOT/src/zensim"
if ! grep -q 'restore-cuts bench overlay' Cargo.toml; then
cat >> Cargo.toml <<'OVERLAY'

# restore-cuts bench overlay: clean archives of fetched main commits.
[patch."https://github.com/imazen/zenanalyze"]
zenpredict = { path = "../zenanalyze/zenpredict" }
zenpredict-bake = { path = "../zenanalyze/zenpredict-bake" }

[patch."https://github.com/imazen/zenmetrics"]
zenstats = { path = "../zenmetrics/crates/zenstats" }

[patch."https://github.com/imazen/zenbench"]
zenbench = { path = "../zenbench" }

[patch."https://github.com/imazen/zenresize"]
zenresize = { path = "../zenresize" }
OVERLAY
fi
cargo bench -p zensim --all-features --bench extract_paths_bench --no-run --message-format=json \
    > "$ROOT/logs/bench_build.jsonl"
exe=$(python3 -c 'import json,sys; a=[json.loads(s) for s in open(sys.argv[1]) if s.startswith("{")]; print(next(r["executable"] for r in a if r.get("reason")=="compiler-artifact" and r.get("target",{}).get("name")=="extract_paths_bench" and r.get("executable")))' "$ROOT/logs/bench_build.jsonl")
sha256sum "$exe"
echo "$exe" | tee "$ROOT/bin/bench_path"
