#!/usr/bin/env bash
# Eval all 5 seeds of V_24-persample-konjnd010 via bake_verdict.
# Computes Mohammadi panel aggregate + 10-band per corpus.
set -euo pipefail

OUT_DIR=/mnt/v/zen/zensim-eval/v24_persample_konjnd010_2026-05-18
BAKE_VERDICT=/home/lilith/work/zen/zensim--persample-konjnd010/target/release/bake_verdict

for S in 1 2 3 4 5; do
  BAKE="$OUT_DIR/persample_konjnd010_seed${S}.bin"
  VERDICT="$OUT_DIR/verdict_seed${S}.md"
  if [[ ! -f "$BAKE" ]]; then
    echo "SKIP seed=$S — bake missing: $BAKE"
    continue
  fi
  echo "EVAL seed=$S → $VERDICT"
  "$BAKE_VERDICT" --bake "$BAKE" --output "$VERDICT" 2>&1 | tail -3
done

# Cross-seed summary — extract aggregate SROCCs
echo ""
echo "=== Cross-seed aggregate SROCC ==="
for S in 1 2 3 4 5; do
  VERDICT="$OUT_DIR/verdict_seed${S}.md"
  [[ -f "$VERDICT" ]] || continue
  echo "--- seed=$S ---"
  grep -E "^\| [A-Z][a-zA-Z0-9_-]+ " "$VERDICT" | head -5
done
