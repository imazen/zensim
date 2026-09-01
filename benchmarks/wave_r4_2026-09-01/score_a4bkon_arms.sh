#!/bin/bash
# a4bkon lane: score K1/K2 bakes through the EXISTING owner (score_arm.sh),
# pointed at this lane's own output dir. No new scoring logic -- reuses the
# wave-r4 build's bake_block_profile / bake_dial_refit pack / bake_verdict
# exactly as score_arm.sh already wires them.
set -euo pipefail
REPO=/home/lilith/work/zen/zensim--a4bkon
cd "$REPO"
export WR4_SCORE=/mnt/v/output/zensim/a4bkon-2026-09-01
BAKES=/mnt/v/output/zensim/a4bkon-2026-09-01/bakes
SCORE="$REPO/benchmarks/wave_r4_2026-09-01/score_arm.sh"

for f in "$BAKES"/K1_w1.8_s4004.bin "$BAKES"/K1_w1.8_s4005.bin \
         "$BAKES"/K1_w2.4_s4004.bin "$BAKES"/K1_w2.4_s4005.bin \
         "$BAKES"/K2_s4004.bin "$BAKES"/K2_s4005.bin; do
  name="$(basename "$f" .bin)"
  if [ -f "$WR4_SCORE/$name.fulleval.json" ]; then
    echo "== SKIP (already scored): $name"
    continue
  fi
  if [ ! -f "$f" ]; then
    echo "== MISSING bake, skipping: $f"
    continue
  fi
  echo "=========================================================="
  echo "SCORING $name  $(date -u +%H:%M:%SZ)"
  echo "=========================================================="
  "$SCORE" "$f" "$name" 944
done
echo "SCORING PASS DONE $(date -u +%H:%M:%SZ)"
