#!/usr/bin/env bash
# sparsehf_queue.sh <arm:seed> [...] — run appendix-R R2 cells SERIALLY in one
# lane (same shape as featsub_queue.sh: run-heavy cgroup cap + heartbeat log).
set -uo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/tmp/zensimsh-target}
LOG=${SPARSEHF_LOG:-$HOME/tmp/sparsehf}
mkdir -p "$LOG"
Q="$LOG/queue$$.log"
for cell in "$@"; do
  arm=${cell%%:*}; seed=${cell##*:}
  out=/mnt/v/output/zensim/bakes/sparsehf/${arm}_s${seed}.bin
  if [[ -f "$out" ]]; then
    echo "$(date -u +%H:%M:%S) SKIP $cell (bake exists)" | tee -a "$Q"; continue
  fi
  echo "$(date -u +%H:%M:%S) START $cell" | tee -a "$Q"
  t0=$SECONDS
  ~/work/zen/scripts/run-heavy --mem 24G --jobs 6 -- \
      "$REPO_ROOT/scripts/sparsehf/sparsehf_seed.sh" "$arm" "$seed" \
      > "$LOG/${arm}_s${seed}.log" 2>&1
  rc=$?
  dt=$((SECONDS-t0))
  if [[ $rc -eq 0 && -f "$out" ]]; then
    echo "$(date -u +%H:%M:%S) DONE  $cell rc=$rc ${dt}s $(stat -c%s "$out") B" | tee -a "$Q"
  else
    echo "$(date -u +%H:%M:%S) FAIL  $cell rc=$rc ${dt}s (see $LOG/${arm}_s${seed}.log)" | tee -a "$Q"
  fi
done
echo "$(date -u +%H:%M:%S) QUEUE COMPLETE" | tee -a "$Q"
