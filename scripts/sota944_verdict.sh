#!/usr/bin/env bash
# sota944_verdict.sh <bake.bin> <stem> [--corruption-head <head.bin>] — the
# ONE bake_verdict invocation of the SOTA-944 campaign (pre-reg §0), shared
# by every arm so no cell can drift corpora/grids. Verdict JSON+md land in
# the campaign verdicts dir under <stem>.
set -euo pipefail
BAKE=${1:?bake}; STEM=${2:?stem}; shift 2
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BV=${ZL_BV:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_verdict}
E=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
V=/mnt/v/output/zensim/v2-eval-944-2026-08-01
VD=/mnt/v/output/zensim/bakes/sota944/verdicts
mkdir -p "$VD"
nice -n 19 ionice -c 3 "$BV" --bake "$BAKE" --regime 720 \
    --features-root "$E" \
    --dial-grid "$V/dial_grid_944col_2026-08-01.parquet" \
    --corruption-grid "$V/corruption_grid_944col_2026-08-01.parquet" \
    --corpora cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25,imazen26,nonphoto,hfnlproxy \
    --perpair-metrics /nonexistent-skip-kadis-block \
    --name "$STEM" --full-json "$VD/$STEM.full.json" --output "$VD/$STEM.verdict.md" \
    "$@" >/dev/null
echo "$VD/$STEM.full.json"
