#!/usr/bin/env bash
# Run bake_compare for each α-bake vs V_22-mix-LARGE+iwssim seed=3.
# Outputs per-α markdown + JSON report.

set -euo pipefail

BAKE_COMPARE="/home/lilith/work/zen/zensim/target/release/bake_compare"
BASELINE_B="/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128.bin"
A_DIR="/mnt/v/zen/zensim-eval/v24_alpha_2026-05-18"
SEED="${SEED:-42}"
RESAMPLES="${RESAMPLES:-1000}"

PCTS="${PCTS:-5 10 15 20 25}"

for pct in $PCTS; do
  sfx=$(printf "alpha%03d" "$pct")
  A_BAKE="$A_DIR/v24_${sfx}_s3_h128.bin"
  if [ ! -f "$A_BAKE" ]; then
    echo "[skip] $A_BAKE not built yet"
    continue
  fi
  OUT_MD="$A_DIR/v24_${sfx}_vs_v22mixLARGE.md"
  OUT_JSON="$A_DIR/v24_${sfx}_vs_v22mixLARGE.json"
  echo "[$(date -Iseconds)] comparing α=${pct}% ($A_BAKE) vs V_22-mix-LARGE+iwssim s3"
  "$BAKE_COMPARE" \
    --a "$A_BAKE" --b "$BASELINE_B" \
    --bootstrap-resamples "$RESAMPLES" \
    --seed "$SEED" \
    --output "$OUT_MD" \
    --json "$OUT_JSON" \
    > /tmp/v24_${sfx}_compare.log 2>&1
  echo "[$(date -Iseconds)] DONE α=${pct}%; verdict:"
  grep -E "ADecisivelyBeatsB|BDecisivelyBeatsA|PromisingNotDecisive|Tied|Noisy|Overall winner" "$OUT_MD" | head -10
done
