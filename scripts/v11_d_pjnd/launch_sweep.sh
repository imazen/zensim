#!/usr/bin/env bash
# EXP-V11-D-PJND-DOMINANT sweep launcher (2026-05-20, task #198).
#
# 3 PJND-passthrough weights × 5 seeds = 15 GPU bakes.
# All other hparams matching V11-A' v4 clean recipe.
set -euo pipefail

DATA=/mnt/v/zen/zensim-training/canonical-2026-05-21/train
ANCHOR=/mnt/v/zen/zensim-training/2026-05-20-v11-substrate
OUTDIR=/mnt/v/output/zensim/exp_v11_d_pjnd_2026-05-20
TRAINER=$(realpath "$(dirname "$0")/../../target/release/zensim_mlp_train")
LOGDIR=$OUTDIR/logs
mkdir -p "$OUTDIR" "$LOGDIR"

run_bake() {
    local pjnd_w=$1
    local seed=$2
    local tag="pjnd${pjnd_w}_s${seed}"
    local out="$OUTDIR/cc4v11d_${tag}.bin"
    local log="$LOGDIR/cc4v11d_${tag}.log"
    if [[ -f "$out" ]]; then
        echo "skip $tag — exists ($(stat -c%s "$out") bytes)"
        return
    fi
    echo "running $tag → $out"
    "$TRAINER" \
        --group safesyn:$DATA/safesyn.parquet:1.0:0.0 \
        --group kadid:$DATA/kadid.parquet:0.6:0.4 \
        --group tid:$DATA/tid.parquet:0.6:0.4 \
        --group konjnd:$DATA/konjnd-dense.parquet:0.6:0.0 \
        --group cid22_train:$DATA/cid22_train.parquet:0.5:0.0 \
        --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size 32 \
        --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
        --max-features 372 --target-column mix_cv35_iw65 \
        --per-sample-alpha-head --tanh-output-head-scale 20.0 \
        --ranknet-weight 0.0 --mse-weight 1.0 --monotonicity-reg 1.0 \
        --anchor-parquet $ANCHOR/anchors_ssim2_372col_v4.parquet \
        --anchor-loss-weight 1.0 --anchor-step-p 0.30 \
        --cross-codec-eq-parquet $ANCHOR/cross_codec_equivalence_ssim2_372col_v4.parquet \
        --cross-codec-eq-weight 0.5 \
        --cross-codec-rank-preserve-weight 0.2 \
        --dynamic-range-floor-weight 0.3 --dynamic-range-sigma-threshold 25.0 --dynamic-range-step-p 0.05 \
        --pjnd-passthrough-parquet $DATA/konjnd-dense.parquet \
        --pjnd-passthrough-weight $pjnd_w --pjnd-passthrough-step-p 0.30 \
        --pjnd-passthrough-target-score 80.0 \
        --gpu-runtime cuda \
        --seed $seed --out "$out" > "$log" 2>&1
    echo "  done $tag ($(stat -c%s "$out") bytes)"
}

# 3 weights × 5 seeds = 15 bakes.
for pjnd_w in 2.0 5.0 10.0; do
    for seed in 1 2 3 4 5; do
        run_bake "$pjnd_w" "$seed"
    done
done
echo "all bakes complete"
