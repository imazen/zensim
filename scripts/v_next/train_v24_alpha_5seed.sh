#!/usr/bin/env bash
# 5-seed CI for a single α value.
# Usage: ./train_v24_alpha_5seed.sh <pct>
# Trains seeds 1..5 in parallel; runs bake_compare for each vs V_22 baseline.

set -euo pipefail

ALPHA_PCT="${1:?need alpha pct}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SFX=$(printf "alpha%03d" "$ALPHA_PCT")

echo "[$(date -Iseconds)] 5-seed CI for α=${ALPHA_PCT}%"
for s in 1 2 3 4 5; do
  bin="/mnt/v/zen/zensim-eval/v24_alpha_2026-05-18/v24_${SFX}_s${s}_h128.bin"
  if [ -f "$bin" ]; then
    echo "  seed=$s already trained at $bin"
    continue
  fi
  echo "  launching seed=$s"
  "$SCRIPT_DIR/train_v24_alpha_sweep.sh" "$ALPHA_PCT" "$s" > /tmp/v24_${SFX}_s${s}.stdout 2>&1 &
done
wait
echo "[$(date -Iseconds)] training done"

for s in 1 2 3 4 5; do
  A="/mnt/v/zen/zensim-eval/v24_alpha_2026-05-18/v24_${SFX}_s${s}_h128.bin"
  B="/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s${s}_h128.bin"
  OUT_MD="/mnt/v/zen/zensim-eval/v24_alpha_2026-05-18/v24_${SFX}_s${s}_vs_v22mixLARGE_s${s}.md"
  OUT_JSON="/mnt/v/zen/zensim-eval/v24_alpha_2026-05-18/v24_${SFX}_s${s}_vs_v22mixLARGE_s${s}.json"
  if [ -f "$OUT_JSON" ]; then echo "  compare s${s} already done"; continue; fi
  /home/lilith/work/zen/zensim/target/release/bake_compare \
    --a "$A" --b "$B" \
    --bootstrap-resamples 1000 --seed 42 \
    --output "$OUT_MD" --json "$OUT_JSON" \
    > /tmp/v24_${SFX}_s${s}_compare.log 2>&1 &
done
wait
echo "[$(date -Iseconds)] compares done"
ls -la /mnt/v/zen/zensim-eval/v24_alpha_2026-05-18/v24_${SFX}_s*_h128.bin
