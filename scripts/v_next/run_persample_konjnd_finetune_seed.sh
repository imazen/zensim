#!/usr/bin/env bash
# V_24 per-sample-α + KonJND finetune
# Hypothesis: starting from V_24 per-sample-α 5-seed bake (CID22 0.86 /
# KonJND 0.81 / AIC-3 0.81 — a CID22+AIC-3 specialist), a BRIEF
# finetune (30 epochs, LR=1e-4) with KonJND train_w boosted from 0.02
# → 0.50 should lift KonJND toward V_22 ship's 0.89 while preserving
# CID22 via init.
#
# Args: <seed> <out_dir> <input_bake>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR="${2:?out_dir}"
INPUT_BAKE="${3:?input_bake}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/persample_konjnd_finetune_seed${SEED}.bin"
LOG="$OUT_DIR/persample_konjnd_finetune_seed${SEED}.log"
TRAINER=/home/lilith/work/zen/zensim--persample-finetune/target/release/zensim_mlp_train

# Mirror V_22-LARGE / per-sample-α recipe except:
#  - --continue-from <input_bake>  (warm-init from persample seed bake)
#  - konjnd train_w: 0.02 → 0.50   (heavy KonJND emphasis)
#  - --lr 1e-4                     (10× smaller than baseline)
#  - --epochs 30                   (brief, 1/10 of baseline)
"$TRAINER" \
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet:0.50:1.0 \
  --group cvvdp_iwssim_large:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden 128 --epochs 30 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --continue-from "$INPUT_BAKE" \
  --lr 1e-4 --seed "$SEED" --log-every 5 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
