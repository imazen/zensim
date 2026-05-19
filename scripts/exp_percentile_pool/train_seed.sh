#!/usr/bin/env bash
# EX-PERCENTILE-POOL: per-seed training on KADID+TID+KonJND with
# P²-pooled Block B features. Mirrors V_22-LARGE+iwssim recipe minus
# safesyn + cvvdp_iwssim_large (those weren't re-extracted with P²;
# scope limitation documented in the experiment doc).
#
# Args: <seed> <out_dir>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR="${2:?out_dir}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/p2pool_seed${SEED}.bin"
LOG="$OUT_DIR/p2pool_seed${SEED}.log"
TRAINER=/home/lilith/work/zen/zensim--exp-percentile-pool/target/release/zensim_mlp_train

P2_DIR=/mnt/v/zen/zensim-training/2026-05-18-percentile-pool

# Same recipe shape as V_22-LARGE+iwssim, scoped to small corpora only.
# Target: mix_cv40_iw60 (the canonical V_22 / V_24 target).
"$TRAINER" \
  --group kadid:$P2_DIR/kadid_mix_300col_p2.parquet:0.3:1.0 \
  --group tid:$P2_DIR/tid_mix_300col_p2.parquet:0.3:1.0 \
  --group konjnd:$P2_DIR/konjnd_mix_300col_p2.parquet:0.02:1.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
