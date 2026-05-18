#!/usr/bin/env bash
# V_24 per-sample-α + KonJND GENTLE finetune (smaller perturbation).
# Reduced from the v1 (konjnd_w=0.50 LR=1e-4 30ep) which destroyed CID22:
#   - konjnd train_w: 0.50 → 0.10 (matches V_24 konjnd010 experiment)
#   - LR: 1e-4 → 3e-5 (3× smaller)
#   - epochs: 30 → 15 (1/2)
#
# Args: <seed> <out_dir> <input_bake>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR="${2:?out_dir}"
INPUT_BAKE="${3:?input_bake}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/persample_konjnd_gentle_seed${SEED}.bin"
LOG="$OUT_DIR/persample_konjnd_gentle_seed${SEED}.log"
TRAINER=/home/lilith/work/zen/zensim--persample-finetune/target/release/zensim_mlp_train

"$TRAINER" \
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet:0.10:1.0 \
  --group cvvdp_iwssim_large:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden 128 --epochs 15 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --continue-from "$INPUT_BAKE" \
  --lr 3e-5 --seed "$SEED" --log-every 3 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
