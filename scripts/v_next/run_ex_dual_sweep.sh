#!/usr/bin/env bash
# EX-DUAL: λ-sweep at single seed=3 across {0.0, 0.01, 0.05, 0.1, 0.3, 1.0}.
# λ=0.0 is the SINGLE-HEAD CONTROL (dual-target architecture but
# auxiliary loss zero → behaves exactly like RankNet on 5-group mix).
# Run sequentially (each run ~2.5 min on the 7950X).
set -euo pipefail
OUT_DIR="${1:?out_dir}"
mkdir -p "$OUT_DIR"
SEED=3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for L in 0.0 0.01 0.05 0.1 0.3 1.0; do
  echo "=== λ=$L seed=$SEED ==="
  "$SCRIPT_DIR/run_ex_dual_seed.sh" "$SEED" "$L" "$OUT_DIR"
done
echo "Sweep complete: $OUT_DIR"
ls -la "$OUT_DIR"
