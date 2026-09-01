#!/bin/bash
# wave-r4: pack + score one arm on the wave's OWN root, and emit the pieces the
# exam needs. Computes NO statistic itself -- every number comes from the owners
# (bake_dial_refit for the pack, bake_verdict for rank/dial/corruption panels).
#
# Usage: score_arm.sh <bake.bin> <NAME> [regime]
set -euo pipefail
BAKE="${1:?bake required}"; NAME="${2:?name required}"; REGIME="${3:-944}"
BIN=/mnt/v/zen/cargo-targets/waver4/release
R4="${WR4_ROOT:-/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01}"
OUT="${WR4_SCORE:-/mnt/v/output/zensim/wave-r4-2026-09-01}"
DIAL="${WR4_DIAL:-/mnt/v/output/zensim/wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet}"
mkdir -p "$OUT"

# 0. the block profile decides --regime; a mismatch is the OPEN silent-mis-score bug
echo "== block profile for $NAME"
"$BIN/bake_block_profile" --bake "$BAKE" | tee "$OUT/$NAME.blockprofile.txt"

# 1. pack (dead-column pruning on by default; identity-gated by the owner)
PACKED="${BAKE%.bin}_packed.bin"
if [ ! -f "$PACKED" ]; then
  echo "== pack $NAME"
  "$BIN/bake_dial_refit" pack --in "$BAKE" --out "$PACKED" --neg-tail --anchor "${WR4_ANCHOR:-$R4/anchor944_pools_dial.parquet}" --target-col "${WR4_ANCHOR_COL:-target_score}" --verify "$R4/ext_cid22val.parquet" --verify-col human_score 2>&1 | tail -20
fi

# 2. the full verdict on the WAVE root (the arm's native features)
echo "== verdict $NAME on the wave-r4 root"
"$BIN/bake_verdict" --bake "$PACKED" --regime "$REGIME" \
  --features-root "$R4" --dial-grid "$DIAL" --name "$NAME" \
  --full-json "$OUT/$NAME.fulleval.json" --output "$OUT/$NAME.verdict.md" 2>&1 | tail -25

# 3. per-pair dumps for the five pairable corpora (LIVE + AIC-4 are pairable at
#    944 width -- the old exclusion was root-scoped, max |d| 0.0 on both roots)
for c in cid22 csiq aic3 live aic4; do
  "$BIN/bake_verdict" --bake "$PACKED" --regime "$REGIME" \
    --features-root "$R4" --corpora "$c" \
    --per-pair-output "$OUT/pp_${NAME}_${c}.tsv" --per-pair-refs \
    --output /dev/null >/dev/null 2>&1 || echo "  (per-pair $c unavailable)"
done
echo "SCORED $NAME -> $OUT/$NAME.fulleval.json"
