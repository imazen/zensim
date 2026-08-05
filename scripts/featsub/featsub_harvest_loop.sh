#!/usr/bin/env bash
# featsub_harvest_loop.sh — verdict every appendix-J bake the moment it lands.
# Exists because the repo's own cycle audit found finished compute sitting
# unharvested for >2 h twice; bake_verdict is ~10 s, so there is no reason for
# a bake to wait. Exits when --expect bakes have verdicts (or --timeout).
set -uo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXPECT=${1:-99}
TIMEOUT=${2:-43200}
LOG=${FEATSUB_LOG:-$HOME/tmp/featsub}
VD=/mnt/v/output/zensim/bakes/sota944/verdicts
t0=$SECONDS
while :; do
  for b in /mnt/v/output/zensim/bakes/featsub/*.bin; do
    [[ -e "$b" ]] || continue
    stem="FS_$(basename "$b" .bin)"
    [[ -f "$VD/$stem.full.json" ]] && continue
    # only verdict a settled bake (spec sidecar written = trainer finished)
    [[ -f "$b.spec.json" ]] || continue
    echo "$(date -u +%H:%M:%S) [harvest] verdict $stem" | tee -a "$LOG/queue.log"
    "$REPO_ROOT/scripts/sota944_verdict.sh" "$b" "$stem" >/dev/null 2>&1 \
      && echo "$(date -u +%H:%M:%S) [harvest] OK $stem" | tee -a "$LOG/queue.log" \
      || echo "$(date -u +%H:%M:%S) [harvest] FAIL $stem" | tee -a "$LOG/queue.log"
  done
  n=$(ls "$VD"/FS_*.full.json 2>/dev/null | wc -l)
  [[ $n -ge $EXPECT ]] && { echo "$(date -u +%H:%M:%S) [harvest] $n verdicts — done" | tee -a "$LOG/queue.log"; break; }
  [[ $((SECONDS-t0)) -gt $TIMEOUT ]] && { echo "$(date -u +%H:%M:%S) [harvest] TIMEOUT at $n" | tee -a "$LOG/queue.log"; break; }
  sleep 60
done
