#!/usr/bin/env bash
# r1_chain.sh <raw-bake.bin> <cellname> — appendix R arm R1: the §3d packaging
# chain on one group-lasso cell (sota944 campaign, REGISTERED APPENDIX R).
#
#   1. add-spline  (anchor944_dial, target_score)          -> R1_<cell>_dial.bin
#   2. gate        (G-RANGE on ext_cid22val)               -> recorded in log
#   3. pack        (--neg-tail --zerobias-bulk 0, f16)     -> R1_<cell>_packed.bin
#      ^^^^^^^^^^^ THE registered GL deviation: default zerobias 0.005 wipes
#      lasso-shrunken survivors (J.R3 measured 57->3). QUANTIZE-then-CALIBRATE
#      holds: pack refits the spline ON the packed net, on the same anchor.
#   4. instrument verdict of the _dial bake -> ~/tmp/sparsehf (rank-invariance
#      check vs the committed raw verdict; NOT a campaign cell, per §3d step 5)
# Harvest (verdict + fulleval w/ M3a) of the _packed bake is the CALLER's job
# (scripts/harvest_bakes.sh), so this script stays fast and single-purpose.
set -euo pipefail
RAW=${1:?raw bake}; CELL=${2:?cell name}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BDR=${ZL_BDR:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_dial_refit}
BV=${ZL_BV:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_verdict}
E=${SOTA944_E:-/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01}
OUT=${SPARSEHF_OUT:-/mnt/v/output/zensim/bakes/sparsehf}
LOG=${SPARSEHF_LOG:-$HOME/tmp/sparsehf}
mkdir -p "$OUT" "$LOG"
DIAL="$OUT/R1_${CELL}_dial.bin"
PACKED="$OUT/R1_${CELL}_packed.bin"

echo "[r1 $CELL] add-spline"
nice -n 19 ionice -c 3 "$BDR" add-spline --in "$RAW" --out "$DIAL" \
    --anchor "$E/anchor944_dial.parquet" --target-col target_score
echo "[r1 $CELL] gate (dial bake) — RECORD step per appendix R.2 (G-RANGE FAIL is a
# recorded row, not a chain abort: the §3d incumbents fail it too at 0.09-0.56%;
# ship-eligibility on this row needs the amendment-2 near-top densification lever)"
nice -n 19 ionice -c 3 "$BDR" gate --bake "$DIAL" --corpus "$E/ext_cid22val.parquet" \
    | tee "$LOG/R1_${CELL}_gate_dial.txt" || true
echo "[r1 $CELL] pack (--neg-tail --zerobias-bulk 0)"
nice -n 19 ionice -c 3 "$BDR" pack --in "$DIAL" --out "$PACKED" \
    --neg-tail --zerobias-bulk 0 \
    --anchor "$E/anchor944_dial.parquet" --target-col target_score \
    --verify "$E/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
    | tee "$LOG/R1_${CELL}_pack.txt"
echo "[r1 $CELL] instrument verdict of _dial -> $LOG"
nice -n 19 ionice -c 3 "$BV" --bake "$DIAL" --regime 944 --name "R1_${CELL}_dial" \
    --full-json "$LOG/R1_${CELL}_dial.full.json" \
    --output "$LOG/R1_${CELL}_dial.verdict.md" >/dev/null
echo "[r1 $CELL] done: $(stat -c%s "$DIAL") B dial, $(stat -c%s "$PACKED") B packed"
