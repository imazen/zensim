#!/usr/bin/env bash
# V_24 per-sample-α + KonJND ULTRAGENTLE finetune.
# v1 (lr=1e-4, konjnd_w=0.5, 30ep, 50k pairs/ep): destroyed CID22 (0.86 → 0.71).
# v2-gentle (lr=3e-5, konjnd_w=0.1, 15ep, 50k pairs/ep): CID22 preserved at
#   0.84 (-0.01) but KonJND collapsed to 0.37 (vs 0.82 init).
#
# Root cause analysis: Adam's first step with m=v=0 produces updates of
# magnitude ~LR regardless of gradient — so 50k pairs at LR=3e-5 ≈ 50k×3e-5 =
# 1.5 worst-case weight drift, large vs init weight scale ~0.1-1.2. The val
# selection saves the BEST during training, but the warm-init itself is never
# considered as a candidate (best_val_score = -inf at start).
#
# v3 (this): LR=1e-5 (3× smaller), pairs/epoch=5000 (10× smaller), 30ep
# (longer), konjnd_w=0.10. Net per-epoch perturbation budget:
# 5000 × 1e-5 = 0.05 per epoch × 30 = 1.5 total — same total budget as
# v2 but spread over more epochs (more steps for Adam moments to stabilize).
#
# Args: <seed> <out_dir> <input_bake>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR="${2:?out_dir}"
INPUT_BAKE="${3:?input_bake}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/persample_konjnd_ultragentle_seed${SEED}.bin"
LOG="$OUT_DIR/persample_konjnd_ultragentle_seed${SEED}.log"
TRAINER=/home/lilith/work/zen/zensim--persample-finetune/target/release/zensim_mlp_train

"$TRAINER" \
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet:0.10:1.0 \
  --group cvvdp_iwssim_large:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden 128 --epochs 30 --pairs-per-epoch 5000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --continue-from "$INPUT_BAKE" \
  --lr 1e-5 --seed "$SEED" --log-every 2 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
