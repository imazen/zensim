#!/usr/bin/env bash
# EXP-CHUNKC-PERGROUP — V_24 per-sample-α recipe + 19 EX-4 Chunk C
# CVVDP-shape per-pair features (f324..f342), but trained on PER-GROUP
# STANDARDIZED parquets. Each training group's f324..f342 is z-scored
# in its own (mu, sigma); zero-fill corpora (safesyn, cvvdp_iwssim_large)
# stay zero.
#
# Identical recipe to scripts/exp_chunkc_perpair/run_chunkc_perpair_seed.sh
# except the --group paths point at per-group-standardized parquets.
#
# Args: <seed>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR=/mnt/v/zen/zensim-eval/exp_chunkc_pergroup_2026-05-18
LOG_DIR=/tmp/exp_chunkc_pergroup_logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

BAKE="$OUT_DIR/pergroup_s${SEED}_h128.bin"
LOG="$LOG_DIR/pergroup_s${SEED}.log"
TRAINER=/home/lilith/work/zen/zensim--exp-chunkc-pergroup/target/release/zensim_mlp_train
DATA=/mnt/v/zen/zensim-training/2026-05-18-chunkc-pergroup

"$TRAINER" \
  --group safesyn:"$DATA/safesyn_per_group_std.parquet":1.0:0.0 \
  --group kadid:"$DATA/kadid_per_group_std.parquet":0.3:1.0 \
  --group tid:"$DATA/tid_per_group_std.parquet":0.3:1.0 \
  --group konjnd:"$DATA/konjnd_per_group_std.parquet":0.02:1.0 \
  --group cvvdp_iwssim_large:"$DATA/cvvdp_iwssim_large_per_group_std.parquet":0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 343 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
