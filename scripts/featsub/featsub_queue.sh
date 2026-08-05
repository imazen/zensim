#!/usr/bin/env bash
# featsub_queue.sh <arm:seed> [<arm:seed> ...] — run appendix-J cells SERIALLY.
#
# RAM is the binding constraint on this box: each run of the arm-H recipe
# resident-sets ~10-11 GB, so cells are run one at a time under run-heavy
# (nice/ionice + a cgroup memory ceiling) and every cell streams a heartbeat
# line into the log the supervisor tails.
set -uo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/tmp/zensimfs-target}
LOG=${FEATSUB_LOG:-$HOME/tmp/featsub}
mkdir -p "$LOG"
Q="$LOG/queue.log"
for cell in "$@"; do
  arm=${cell%%:*}; seed=${cell##*:}
  out=/mnt/v/output/zensim/bakes/featsub/${arm}_s${seed}.bin
  if [[ -f "$out" ]]; then
    echo "$(date -u +%H:%M:%S) SKIP $cell (bake exists)" | tee -a "$Q"; continue
  fi
  echo "$(date -u +%H:%M:%S) START $cell" | tee -a "$Q"
  t0=$SECONDS
  ~/work/zen/scripts/run-heavy --mem 24G --jobs 6 -- \
      "$REPO_ROOT/scripts/featsub/featsub_seed.sh" "$arm" "$seed" \
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
