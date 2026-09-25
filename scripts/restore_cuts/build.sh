#!/usr/bin/env bash
# Clean-snapshot build of the extractor (CODEX_NOTE crates-on-main rule).
# Sibling repos are `git archive`s of their fetched mains under $ROOT/src (see
# $ROOT/snapshot_revs.tsv); zensim itself is a copy of THIS workspace. No sibling
# working copy is read or written. Usage: build.sh [example ...] (default: the extractor).
set -euo pipefail
ROOT=/var/tmp/restore-cuts
WS=/home/lilith/work/zen/zensim--restore-cuts
export CARGO_HOME=$ROOT/cargo-home CARGO_TARGET_DIR=$ROOT/target-rel
# An extraction binary must come from COMMITTED source: a dirty tree still builds (dev use)
# but writes no build_meta.json, and bank_sidecar.py refuses a set without it.
if (cd "$WS" && jj status --ignore-working-copy >/dev/null 2>&1; jj status | grep -q 'The working copy has no changes'); then
    CLEAN=1
else
    CLEAN=0
    rm -f "$ROOT/build_meta.json"
    echo "WARNING: uncommitted changes; DEV BUILD, no build_meta.json will be written"
fi
mkdir -p "$ROOT/src/zensim"
rsync -a --delete --exclude '/target' --exclude '.jj' --exclude '.git' --exclude '.workongoing' \
    "$WS/" "$ROOT/src/zensim/"
cat >> "$ROOT/src/zensim/zensim-bench/Cargo.toml" <<'OVERLAY'

# restore-cuts local build overlay: clean archives of fetched main commits.
[patch."https://github.com/imazen/zenanalyze"]
zenpredict = { path = "../../zenanalyze/zenpredict" }
zenpredict-bake = { path = "../../zenanalyze/zenpredict-bake" }

[patch."https://github.com/imazen/rav1d-safe"]
rav1d-safe = { path = "../../rav1d-safe" }
rav1d-disjoint-mut = { path = "../../rav1d-safe/crates/rav1d-disjoint-mut" }

[patch."https://github.com/imazen/zenbench"]
zenbench = { path = "../../zenbench" }

[patch."https://github.com/imazen/zenresize"]
zenresize = { path = "../../zenresize" }

[patch."https://github.com/imazen/codec-corpus"]
corruption-corpus = { path = "../../codec-corpus/crate/corruption-corpus" }
OVERLAY
cd "$ROOT/src/zensim/zensim-bench"
examples=("$@")
[ ${#examples[@]} -gt 0 ] || examples=(extract_features_372col)
args=()
for e in "${examples[@]}"; do args+=(--example "$e"); done
cargo build --release -p zensim-bench "${args[@]}" --features training,zen-decode,verify-all
sha256sum "$CARGO_TARGET_DIR"/release/examples/extract_features_372col
[ "$CLEAN" = 1 ] || exit 0
# Provenance: the workspace commit this tree is.
cd "$WS"
zc=$(jj log -r @- --no-graph -T 'commit_id' --ignore-working-copy)
base=$(jj log -r 'main@origin' --no-graph -T 'commit_id' --ignore-working-copy)
python3 "$WS/scripts/restore_cuts/write_build_meta.py" "$zc" "$base"
