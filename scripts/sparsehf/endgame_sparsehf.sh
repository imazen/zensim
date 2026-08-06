#!/usr/bin/env bash
# endgame_sparsehf.sh — appendix-R chain endgame (committed, idempotent).
# Runs from await_artifacts.sh --then when all 8 R2 raw fullevals exist:
#   1. package every CS cell through the R1 chain (add-spline -> pack zb0)
#   2. harvest the packed twins (verdict + fulleval w/ M3a), inline
#   3. emit the R.R tables (benchmarks/sparsehf/*.tsv)
# Idempotent: every step skips work whose artifact already exists (r1_chain is
# skipped when the packed bake exists; harvest_bakes/tables are re-runnable).
set -uo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/tmp/zensimsh-target}
export ZL_BV=${ZL_BV:-$CARGO_TARGET_DIR/release/bake_verdict}
BK=/mnt/v/output/zensim/bakes/sparsehf
LOG=${SPARSEHF_LOG:-$HOME/tmp/sparsehf}
mkdir -p "$LOG"
rc=0

for lam in 0p3 1 2 4; do
  for s in 2501 2503; do
    cell="CS${lam}_s${s}"
    raw="$BK/$cell.bin"
    packed="$BK/R1_${cell}_packed.bin"
    [ -s "$raw" ] || { echo "endgame: missing raw $raw" | tee -a "$LOG/endgame.log"; rc=1; continue; }
    [ -s "$packed" ] && continue
    if ! "$REPO_ROOT/scripts/sparsehf/r1_chain.sh" "$raw" "$cell" >> "$LOG/endgame.log" 2>&1; then
      echo "endgame: r1_chain FAILED for $cell" | tee -a "$LOG/endgame.log"; rc=1
    fi
  done
done

# Inline harvest of every packed twin that exists (idempotent; counts what's there).
n_packed=$(ls "$BK"/R1_CS*_packed.bin 2>/dev/null | wc -l)
if [ "$n_packed" -gt 0 ]; then
  "$REPO_ROOT/scripts/harvest_bakes.sh" --glob "$BK/R1_CS*_packed.bin" \
      --count "$n_packed" --regime 944 --timeout 7200 \
      --heartbeat "$LOG/harvest_cs_packed" >> "$LOG/endgame.log" 2>&1 || rc=1
fi

python3 "$REPO_ROOT/scripts/sparsehf/sparsehf_tables.py" >> "$LOG/endgame.log" 2>&1 || rc=1
echo "endgame_sparsehf: rc=$rc (packed=$n_packed)" | tee -a "$LOG/endgame.log"
exit $rc
