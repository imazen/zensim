#!/usr/bin/env bash
# featsub_verdicts.sh [<bake.bin> ...] — run the campaign's ONE verdict
# invocation (scripts/sota944_verdict.sh = bake_verdict --regime 944) over
# appendix-J bakes. With no arguments, every bake in the featsub dir that has
# no verdict yet. Stems are prefixed FS_ so they share the campaign verdict
# store without colliding with any other wave.
set -uo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/tmp/zensimfs-target}
VD=/mnt/v/output/zensim/bakes/sota944/verdicts
BAKES=("$@")
if [[ ${#BAKES[@]} -eq 0 ]]; then
  mapfile -t BAKES < <(ls /mnt/v/output/zensim/bakes/featsub/*.bin 2>/dev/null)
fi
for b in "${BAKES[@]}"; do
  stem="FS_$(basename "$b" .bin)"
  [[ -f "$VD/$stem.full.json" ]] && { echo "have $stem"; continue; }
  echo "verdict $stem"
  "$REPO_ROOT/scripts/sota944_verdict.sh" "$b" "$stem" || echo "FAILED $stem"
done
