#!/usr/bin/env bash
# featsub_queue2.sh <arm:seed> ... — appendix-J cell runner, N-worker safe.
#
# Two coordination properties the first-cut serial runner lacked:
#  1. ATOMIC PER-CELL LOCK (`mkdir`), so several instances can share one cell
#     list without ever starting the same cell twice. `[[ -f bake ]]` alone
#     races: a cell in flight has no bake yet.
#  2. A RAM GATE. Each run of the arm-H recipe resident-sets ~11 GB on this
#     box and several agents share it, so a worker waits for MemAvailable
#     >= FEATSUB_MIN_AVAIL_MB (default 14000) before claiming the next cell
#     rather than pushing the box into swap.
# Every state change is a timestamped line in queue.log — the file the
# supervisor tails.
set -uo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/tmp/zensimfs-target}
LOG=${FEATSUB_LOG:-$HOME/tmp/featsub}
LOCKS=$LOG/locks
MIN_AVAIL=${FEATSUB_MIN_AVAIL_MB:-14000}
# Max OTHER-plus-own trainers already running before we add ours (so the box
# never exceeds MAX_TRAINERS+1 concurrent runs of this recipe).
MAX_TRAINERS=${FEATSUB_MAX_TRAINERS:-3}
WHO=${FEATSUB_WORKER:-w$$}
mkdir -p "$LOG" "$LOCKS"
Q="$LOG/queue.log"
say() { echo "$(date -u +%H:%M:%S) [$WHO] $*" | tee -a "$Q"; }

# Two conditions, both necessary. MemAvailable alone is not enough: the box
# holds 4 concurrent runs of this recipe (~11.6 GB each on 58 GB) and not 5,
# and OTHER agents' lanes appear between our check and our launch. Measured
# 2026-08-04: 5 concurrent runs took MemAvailable to 134 MB, pushed the box
# into swap, and left a run stuck at epoch 0 thrashing. So we also cap the
# GLOBAL trainer count — every zensim_mlp_train on the box, not just ours.
wait_for_slot() {
  local waited=0
  while true; do
    local avail nproc_t
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    nproc_t=$(pgrep -xc zensim_mlp_trai 2>/dev/null || echo 0)
    if [[ $avail -ge $MIN_AVAIL && $nproc_t -le $MAX_TRAINERS ]]; then
      [[ $waited -gt 0 ]] && say "slot ok (${avail}MB avail, ${nproc_t} trainers) after ${waited}s"
      return 0
    fi
    [[ $((waited % 600)) -eq 0 ]] &&       say "waiting for a slot (${avail}MB avail / need ${MIN_AVAIL}; ${nproc_t} trainers / need <= ${MAX_TRAINERS})"
    sleep 30; waited=$((waited+30))
  done
}

for cell in "$@"; do
  arm=${cell%%:*}; seed=${cell##*:}
  out=/mnt/v/output/zensim/bakes/featsub/${arm}_s${seed}.bin
  [[ -f "$out" ]] && { say "SKIP $cell (bake exists)"; continue; }
  mkdir "$LOCKS/$cell" 2>/dev/null || { say "SKIP $cell (locked by another worker)"; continue; }
  [[ -f "$out" ]] && { say "SKIP $cell (bake appeared)"; continue; }
  # GLOBAL START MUTEX. Without it two workers can both observe the same
  # MemAvailable and both launch — measured 2026-08-04: two claims at 17 GB
  # avail each grew toward 11 GB and drove the box to 2.8 GB, OOM-killing one
  # run inside the parquet reader ("Cannot allocate memory (os error 12)").
  # The mutex serializes the check->launch window and is held while the new
  # process's working set materializes, so the NEXT worker's check sees
  # reality rather than the pre-launch number.
  while ! mkdir "$LOCKS/.start" 2>/dev/null; do sleep 10; done
  wait_for_slot
  say "START $cell"
  ( sleep "${FEATSUB_START_HOLD:-180}"; rmdir "$LOCKS/.start" 2>/dev/null ) &
  t0=$SECONDS
  ~/work/zen/scripts/run-heavy --mem 24G --jobs 6 -- \
      "$REPO_ROOT/scripts/featsub/featsub_seed.sh" "$arm" "$seed" \
      > "$LOG/${arm}_s${seed}.log" 2>&1
  rc=$?; dt=$((SECONDS-t0))
  if [[ $rc -eq 0 && -f "$out" ]]; then
    say "DONE  $cell rc=$rc ${dt}s $(stat -c%s "$out")B"
  else
    say "FAIL  $cell rc=$rc ${dt}s (see $LOG/${arm}_s${seed}.log)"
    rmdir "$LOCKS/$cell" 2>/dev/null
  fi
done
say "QUEUE COMPLETE"
