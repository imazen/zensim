#!/usr/bin/env bash
# R6 gate G5: the DIAL side of the F4-arm decision, on each arm's own instruments.
#
# Three legs, and each is only meaningful IN-ERA:
#   * the LADDER dial grid, re-extracted at the arm (the 2026-09-05 floor-dense
#     instrument's own pairs list, 9,593 cells over 5 codec ladders);
#   * a NEGATIVE-TAIL probe cut from the arm's own safesyn table by the probe's
#     registered selection rule (ssim2 < 0), carrying `ssim2_gpu` so A8r's
#     reachability guard has the instrument's truth;
#   * the arm's IDENTITY leg (400 self-pairs).
#
# The mentor's per-cell ssim2 truth is a property of the PIXELS, so ONE truth TSV
# serves every arm — that is what makes `--floor-rule resolvable` (whose window
# and bar are both live-computed from the mentor) comparable across arms at all.
#
# The arm probes are NOT registered instruments, so the registry-pinned A7-A9
# rows read NOT MEASURED by design. That is the honest state: a bar measured on
# revision-1 pixels is not a bar for a revision-2 dial (G-ADDR doc §14).
#
# Usage: r6_dial_arms.sh <extract|grade> [ROOT]
set -euo pipefail
MODE="${1:-grade}"
ROOT="${2:-/mnt/v/output/zensim/rev2-2026-09-05/r6}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EX="$REPO/zensim-bench/target/release/examples/extract_features_372col"
BV="$REPO/target/release/bake_verdict"
LADDER_PAIRS="${LADDER_PAIRS:-$HOME/tmp/ladder_instr/ladder/ladder_pairs.tsv}"
TRUTH=/mnt/v/output/zensim/ladder-2026-09-05/instruments/dialcells_ssim2_ladder.tsv
ARMS="${R6_ARMS:-ssim2 c1 lorentz clamp}"
VARIANTS="${R6_VARIANTS:-s156_lasso s156_bvls s228_lasso s228_bvls s372_lasso s372_bvls}"
mkdir -p "$ROOT"/{dial,instruments}

if [ "$MODE" = extract ]; then
  [ -f "$LADDER_PAIRS" ] || { echo "missing ladder pairs: $LADDER_PAIRS" >&2; exit 2; }
  for a in $ARMS; do
    T="$ROOT/tables/$a"
    printf '[%s] %s ladder\n' "$(date -u +%H:%M:%S)" "$a"
    ZENSIM_SSIM_LUMA=$a nice -n19 ionice -c3 "$EX" --corpus pairs-tsv \
        --path "$LADDER_PAIRS" --out "$T/ladder.csv" >/dev/null
  done
  python3 "$REPO/scripts/r6_build_dial_instruments.py" "$ROOT"
  exit 0
fi

for a in $ARMS; do
  for v in $VARIANTS; do
    B="$ROOT/bakes/${a}_${v}.bin"
    [ -f "$B" ] || continue
    L="${a}_${v}"
    nice -n19 ionice -c3 "$BV" --bake "$B" \
        --features-root "$ROOT/evalroot/$a" \
        --dial-grid "$ROOT/instruments/${a}_ladder.parquet" \
        --gaddr-grid-truth "$TRUTH" \
        --negtail-probe "$ROOT/instruments/${a}_negtail.parquet" \
        --identity-probe "$ROOT/instruments/${a}_identity.parquet" \
        --corpora cid22,konjnd,aic3 \
        --floor-rule resolvable \
        --gaddr-tail-pins product \
        --gaddr-json "$ROOT/dial/gaddr_${L}.json" \
        --full-json "$ROOT/dial/verdict_${L}.json" \
        > "$ROOT/dial/${L}.log" 2>&1 || echo "  grade FAILED $L (see $ROOT/dial/${L}.log)"
    printf '[%s] graded %s\n' "$(date -u +%H:%M:%S)" "$L"
  done
done
echo "dial done -> $ROOT/dial"
