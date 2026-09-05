#!/usr/bin/env bash
# Build + evaluate Profile D's companion corruption head, one arm per feature
# slice. THE reproduction entry point for benchmarks/corruption_head_d_2026-09-05.md.
#
# Why these slices: shipped D reads 28 of 156 BASIC lines and zero pool lines, so
# its walk runs V1PoolsMode::Peaks — which fold_engine.rs documents costs the SAME
# as Off, while masked/IW would force Full. f0..155 and f0..227 are therefore both
# FREE at D's runtime and f228..371 is not. The 2026-07-24 head reads all 372 and
# its own ablation puts top features in mask/iw/peak f255-334, so it is not a D
# companion at any price.
#
#   d156      f0..155   negrich + ladder codec negatives + matched anchors
#   d228      f0..227   same negatives, the peaks slice (free at the same cost)
#   d228nb    f0..227   NO broad-honest — the ablation that prices the codec negatives
#
# Usage: scripts/corrhead_arms.sh [arm ...]      (default: all three)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${CORRHEAD_OUT:-/mnt/v/output/zensim/corruption-head-2026-09-05}
CORPUS=$OUT/im26_corruption_372_postC.parquet
NEGRICH=$OUT/negrich_372_postC.parquet
LADDER=${CORRHEAD_LADDER:-/mnt/v/output/zensim/ladder-2026-09-05/instruments/dial_grid_372col_ladder.parquet}
TRAIN=$ROOT/scripts/v_next/train_corruption_head.py
BAKEBIN=${ZENPREDICT_BAKE:-$HOME/work/zen/zenanalyze/target/release/zenpredict-bake}

for f in "$CORPUS" "$NEGRICH" "$LADDER" "$BAKEBIN"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 2; }
done

run_arm() {
  local name=$1 range=$2; shift 2
  local d=$OUT/$name; mkdir -p "$d"
  echo "=== arm $name (f$range) ==="
  python3 "$TRAIN" \
    --corpus "$CORPUS" --negrich "$NEGRICH" \
    --feat-range "$range" --thresholds 0.5,0.9,0.95 \
    --out "$d/corruption_head_$name.json" \
    --bake-out "$d/corruption_head_$name.bin" \
    --bake-extra-width 944 \
    --bake-bin "$BAKEBIN" \
    --split-out "$d/split.tsv" \
    "$@" 2>&1 | tee "$d/train.log"
}

want=("$@"); [ ${#want[@]} -eq 0 ] && want=(d156 d228 d228nb)
for a in "${want[@]}"; do
  case $a in
    d156)   run_arm d156   0:156 --broad-honest "ladder:$LADDER:image_id" ;;
    d228)   run_arm d228   0:228 --broad-honest "ladder:$LADDER:image_id" ;;
    d228nb) run_arm d228nb 0:228 --no-broad-honest ;;
    *) echo "unknown arm: $a" >&2; exit 2 ;;
  esac
done
