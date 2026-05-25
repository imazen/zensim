#!/usr/bin/env bash
# Architecture evaluation matrix — trains multiple MLP variants on the
# same data and compares their LightPanel stats + goal scores.
#
# Usage:
#   bash scripts/arch_eval_matrix.sh [EPOCHS] [PAIRS]
#
# Default: 50 epochs × 10000 pairs/epoch (~15s per variant on 7950X).
# Results written to /tmp/arch_eval_matrix_<date>.tsv
#
# Variants tested:
#   baseline    — 372→128→heads (current production shape)
#   skip        — 372→128→heads + 372→1 linear skip
#   deep        — 372→128→64→heads (2 hidden layers)
#   deep_skip   — 372→128→64→heads + skip
#   wide        — 372→256→heads
#   narrow      — 372→64→heads

set -euo pipefail

EPOCHS=${1:-50}
PAIRS=${2:-10000}
SEED=42
OUT_DIR="/tmp/arch_eval_matrix_$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$OUT_DIR"

TRAIN_BINARY="./target/release/zensim_mlp_train"
if [ ! -f "$TRAIN_BINARY" ]; then
    echo "Building trainer..."
    cargo build --release -p zensim-validate --bin zensim_mlp_train
fi

COMMON_ARGS=(
    --group "safesyn:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet:1.0:0.0"
    --group "kadid:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/kadid.parquet:0.5:0.0"
    --group "tid:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/tid.parquet:0.5:0.0"
    --group "konjnd_dense:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/konjnd-dense.parquet:0.3:0.0"
    --epochs "$EPOCHS" --pairs-per-epoch "$PAIRS" --lr 5.66e-3 --l2 1e-5
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0
    --max-features 372 --minibatch-size 32
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32
    --per-sample-alpha-head --tanh-output-head-scale 30.0
    --ranknet-weight 0.0 --mse-weight 1.0
    --monotonicity-reg 1.0 --monotonicity-margin 0.0
    --seed "$SEED" --log-every "$EPOCHS"
    --val-aggregate geomean3
)

TSV="$OUT_DIR/results.tsv"
echo -e "variant\thidden\tlayers\tskip\tepoch_time_s\tlast_val\tlast_loss" > "$TSV"

run_variant() {
    local name=$1 hidden=$2 layers=$3 skip=$4
    local out="$OUT_DIR/${name}.bin"
    local log="$OUT_DIR/${name}.log"

    local extra_args=()
    extra_args+=(--hidden "$hidden")
    extra_args+=(--n-hidden-layers "$layers")
    if [ "$skip" = "true" ]; then
        extra_args+=(--skip-connection)
    fi

    echo ">>> $name: h=$hidden layers=$layers skip=$skip"
    /usr/bin/time -f "%e" "$TRAIN_BINARY" \
        "${COMMON_ARGS[@]}" "${extra_args[@]}" \
        --out "$out" 2>&1 | tee "$log"

    # Parse last epoch line
    local last_epoch
    last_epoch=$(grep "^  epoch" "$log" | tail -1)
    local val_score loss epoch_time
    val_score=$(echo "$last_epoch" | grep -oP 'val\([^)]+\)=\K[0-9.]+' || echo "?")
    loss=$(echo "$last_epoch" | grep -oP 'loss=\K[0-9.]+' || echo "?")
    epoch_time=$(echo "$last_epoch" | grep -oP 't=\K[0-9.]+' || echo "?")

    echo -e "${name}\t${hidden}\t${layers}\t${skip}\t${epoch_time}\t${val_score}\t${loss}" >> "$TSV"
}

run_variant "baseline"   128 1 false
run_variant "skip"       128 1 true
run_variant "deep"       128 2 false
run_variant "deep_skip"  128 2 true
run_variant "wide"       256 1 false
run_variant "narrow"     64  1 false

echo ""
echo "=== Results ==="
column -t -s $'\t' "$TSV"
echo ""
echo "Full logs in: $OUT_DIR/"
