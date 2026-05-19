#!/usr/bin/env bash
# Run 5 seeds of the EXP-LARGER-LARGE-V2 training in parallel.
# Usage: scripts/exp_larger_large/run_5_seeds.sh <out_dir> <large_parquet>
set -euo pipefail
OUT_DIR="${1:?out_dir}"
LARGE_PARQUET="${2:?large_parquet}"
mkdir -p "$OUT_DIR"

SEED_SCRIPT="$(dirname "$0")/run_exp_larger_large_seed.sh"
HIDDEN=128

for SEED in 1 2 3 4 5; do
    LOG="$OUT_DIR/larger_large_v2_s${SEED}_h${HIDDEN}.parallel.log"
    bash "$SEED_SCRIPT" "$SEED" "$HIDDEN" "$OUT_DIR" "$LARGE_PARQUET" > "$LOG" 2>&1 &
    echo "Started seed=$SEED PID=$!"
done

# Wait for all parallel trainings to complete
wait
echo "All 5 seeds completed."
ls -la "$OUT_DIR/larger_large_s"*"_h${HIDDEN}".bin 2>&1 | head -10
