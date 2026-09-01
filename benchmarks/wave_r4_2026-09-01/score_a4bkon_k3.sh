#!/bin/bash
set -euo pipefail
REPO=/home/lilith/work/zen/zensim--a4bkon
cd "$REPO"
export WR4_SCORE=/mnt/v/output/zensim/a4bkon-2026-09-01
BAKES=/mnt/v/output/zensim/a4bkon-2026-09-01/bakes
SCORE="$REPO/benchmarks/wave_r4_2026-09-01/score_arm.sh"
for SEED in 4004 4005; do
  f="$BAKES/K3_s${SEED}.bin"; name="K3_s${SEED}"
  if [ -f "$WR4_SCORE/$name.fulleval.json" ]; then echo "SKIP $name"; continue; fi
  echo "== SCORING $name $(date -u +%H:%M:%SZ)"
  "$SCORE" "$f" "$name" 944
done
echo "K3 SCORING DONE $(date -u +%H:%M:%SZ)"
