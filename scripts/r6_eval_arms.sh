#!/usr/bin/env bash
# R6 gate G1/G2: score every arm's bake on that arm's OWN eval root.
#
# The comparison is deliberately whole-metric: an arm changes the FEATURES and
# the fit together, so "what rank would we get if we shipped arm X" is
# `bake(arm X) scored on features(arm X)`. Scoring one arm's bake on another
# arm's root would answer a question nobody is asking.
#
# One `bake_verdict` per (bake, corpus) because `--per-pair-output` dumps the
# LAST corpus only, and the paired bootstrap (G1) needs a per-pair dump per
# corpus with the human column identical across arms.
#
# Usage: r6_eval_arms.sh [ROOT]
set -euo pipefail
ROOT="${1:-/mnt/v/output/zensim/rev2-2026-09-05/r6}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BV="$REPO/target/release/bake_verdict"
CORPORA="${R6_CORPORA:-cid22 konjnd aic3 csiq live tid kadid}"
ARMS="${R6_ARMS:-ssim2 c1 lorentz clamp}"
VARIANTS="${R6_VARIANTS:-s156_lasso s156_bvls s228_lasso s228_bvls}"
mkdir -p "$ROOT"/{verdicts,perpair}

for arm in $ARMS; do
  for v in $VARIANTS; do
    B="$ROOT/bakes/${arm}_${v}.bin"
    [ -f "$B" ] || { echo "MISSING BAKE $B" >&2; continue; }
    for c in $CORPORA; do
      nice -n19 ionice -c3 "$BV" --bake "$B" --corpora "$c" \
          --features-root "$ROOT/evalroot/$arm" \
          --per-pair-output "$ROOT/perpair/${arm}_${v}_${c}.tsv" \
          --json "$ROOT/verdicts/${arm}_${v}_${c}.json" \
          --output "$ROOT/verdicts/${arm}_${v}_${c}.md" >/dev/null 2>>"$ROOT/verdicts/bv.err" \
        || echo "  FAILED ${arm}_${v}_${c}" >&2
    done
    printf '[%s] scored %s_%s\n' "$(date -u +%H:%M:%S)" "$arm" "$v"
  done
done
echo "eval done -> $ROOT/verdicts + $ROOT/perpair"
