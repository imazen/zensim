#!/usr/bin/env bash
# Per-sample α head training — single seed. Args: <seed> <out_dir>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR="${2:?out_dir}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/persample_seed${SEED}.bin"
LOG="$OUT_DIR/persample_seed${SEED}.log"
TRAINER_DIR=/home/lilith/work/zen/zensim--ex2-persample-alpha
TRAINER="$TRAINER_DIR/target/release/zensim_mlp_train"

# Mirror the V_22-LARGE / V_24-hybrid-NiN recipe exactly except
# for the head flag.
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
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
