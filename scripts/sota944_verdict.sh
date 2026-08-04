#!/usr/bin/env bash
# sota944_verdict.sh <bake.bin> <stem> [extra bake_verdict args] — the ONE
# bake_verdict invocation of the SOTA-944 campaign (pre-reg §0), shared by
# every arm/wave so no cell can drift corpora/grids.
#
# Since 2026-08-04 the ENTIRE invocation (ext944 features root, 944 dial +
# corruption grids, kadis-944 per-pair source, the frozen 12-corpus list) IS
# `bake_verdict --regime 944` — resolved inside the binary and pinned by its
# `regime_944_*` tests — so this wrapper structurally cannot drift from the
# preset. Hand-assembled wrapper drift is the class that produced the wrong
# published EM4 HF-NL cell (campaign doc, "Corrections" section).
set -euo pipefail
BAKE=${1:?bake}; STEM=${2:?stem}; shift 2
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BV=${ZL_BV:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_verdict}
VD=/mnt/v/output/zensim/bakes/sota944/verdicts
mkdir -p "$VD"
nice -n 19 ionice -c 3 "$BV" --bake "$BAKE" --regime 944 \
    --name "$STEM" --full-json "$VD/$STEM.full.json" --output "$VD/$STEM.verdict.md" \
    "$@" >/dev/null
echo "$VD/$STEM.full.json"
