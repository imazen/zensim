#!/usr/bin/env bash
# EX-DUAL sweep with retry-on-OOM. Skip bakes that already exist.
set -uo pipefail
OUT_DIR="${1:?out_dir}"
mkdir -p "$OUT_DIR"
SEED=3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for L in 0.0 0.01 0.05 0.1 0.3 1.0; do
  BAKE="$OUT_DIR/exdual_l${L}_seed${SEED}.bin"
  if [ -f "$BAKE" ]; then
    echo "=== SKIP λ=$L (exists: $BAKE) ==="
    continue
  fi
  echo "=== λ=$L seed=$SEED ==="
  for attempt in 1 2 3; do
    if "$SCRIPT_DIR/run_ex_dual_seed.sh" "$SEED" "$L" "$OUT_DIR"; then
      break
    fi
    echo "attempt $attempt failed for λ=$L; backing off 60s..."
    sleep 60
  done
  if [ ! -f "$BAKE" ]; then
    echo "FAILED λ=$L after 3 attempts; continuing to next" >&2
  fi
done
echo "Sweep done: $OUT_DIR"
ls -la "$OUT_DIR"/*.bin 2>/dev/null
