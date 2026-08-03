#!/usr/bin/env bash
#
# sota944_corruption_head.sh <seed> — train the 944 corruption HEAD
# (benchmarks/sota944_campaign_2026-08-03.md §6): rank-only ordering head on
# negrich_944 (w1.0) + kadis700k_944 (w0.5), target score_zensim_gpu. No ZNPR
# corruption head existed at any width before this (the 372-era head is
# sklearn-JSON, unusable by `bake_verdict --corruption-head`). Registered
# seeds {13, 42}; selection by trainer-internal best_val, NEVER by the
# corruption grid.
set -euo pipefail
SEED=${1:?usage: sota944_corruption_head.sh <seed>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
K=/mnt/v/zen/zensim-training/kadis-944-2026-08-01
OUT=/mnt/v/output/zensim/bakes/sota944/bakes
mkdir -p "$OUT"
BAKE="$OUT/corrhead944_s${SEED}.bin"
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }
[[ -x "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 2; }

exec "$TRAIN" \
  --group "negrich:$K/kadis_negrich_944.parquet:1.0:1.0:rank" \
  --group "kadis:$K/kadis700k_944.parquet:0.5:1.0:rank" \
  --target-column score_zensim_gpu --target-scale 1 \
  --epochs 60 --pairs-per-epoch 50000 --seed "$SEED" \
  --max-features 944 --allow-narrow-features \
  --out "$BAKE"
