#!/usr/bin/env bash
# Poll for new bakes and score them as they arrive.
set -uo pipefail
OUT_DIR="${1:?out_dir}"
VERDICT="/home/lilith/work/zen/zensim--dual-target/target/release/bake_verdict"

while true; do
  done_count=$(ls "$OUT_DIR"/exdual_l*.bin 2>/dev/null | wc -l)
  for B in "$OUT_DIR"/exdual_l*.bin; do
    [ -f "$B" ] || continue
    base=$(basename "$B" .bin)
    VERDICT_MD="$OUT_DIR/$base.verdict.md"
    if [ ! -f "$VERDICT_MD" ]; then
      echo "SCORE $base at $(date +%H:%M:%S)"
      "$VERDICT" --bake "$B" --corpora cid22,kadid,tid,konjnd,aic3 \
        --output "$VERDICT_MD" 2>&1 | tail -2
    fi
  done
  if [ "$done_count" -ge 6 ]; then
    # Check all are scored.
    scored=$(ls "$OUT_DIR"/exdual_l*.verdict.md 2>/dev/null | wc -l)
    if [ "$scored" -ge 6 ]; then
      echo "ALL_SCORED at $(date +%H:%M:%S)"
      break
    fi
  fi
  sleep 30
done
