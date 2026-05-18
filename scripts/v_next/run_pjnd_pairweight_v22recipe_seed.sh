#!/usr/bin/env bash
# Variant (b): V_22-mix-LARGE recipe + PJND-aware pair weighting (NO
# per-sample α), single seed. Isolates the pair-weighting effect from
# the per-sample-α architecture.
# Args: <seed> <out_dir>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR="${2:?out_dir}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/pjnd_v22recipe_seed${SEED}.bin"
LOG="$OUT_DIR/pjnd_v22recipe_seed${SEED}.log"
TRAINER=/home/lilith/work/zen/zensim--pjnd-pairweighting/target/release/zensim_mlp_train

# V_22-mix-LARGE recipe + PJND pair weighting; no per-sample-α.
# Same 5-group corpus as variant (a) so the only deltas are:
#   - per-sample-α: OFF (was ON in variant a)
#   - PJND weighting: ON (same as variant a)
"$TRAINER" \
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet:0.02:1.0 \
  --group cvvdp_iwssim_large:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --pjnd-aware-pair-weighting --pjnd-target-group konjnd \
  --pjnd-threshold 45.0 --pjnd-sigma-mid 8.0 \
  --pjnd-gap-anchor 27.0 --pjnd-sigma-gap 10.0 \
  --pjnd-normalization-z 0.329719 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
