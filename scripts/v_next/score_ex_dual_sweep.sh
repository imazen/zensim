#!/usr/bin/env bash
# Score every bake from the EX-DUAL sweep against the full validation
# panel (cid22 + kadid + tid + konjnd + aic3) via bake_verdict.
set -euo pipefail
OUT_DIR="${1:?out_dir}"
VERDICT="/home/lilith/work/zen/zensim--dual-target/target/release/bake_verdict"

for B in "$OUT_DIR"/exdual_l*.bin; do
  base=$(basename "$B" .bin)
  echo "=== Scoring $base ==="
  "$VERDICT" --bake "$B" --corpora cid22,kadid,tid,konjnd,aic3 \
    --output "$OUT_DIR/$base.verdict.md" 2>&1 | tee "$OUT_DIR/$base.verdict.log"
done
echo "Scoring complete"
ls "$OUT_DIR"/*.verdict.md
