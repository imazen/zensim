#!/usr/bin/env bash
# Run a named training experiment + Mohammadi eval, append to results TSV.
#
# Usage: bash scripts/run_experiment.sh <name> [extra_args...]
# Output: /mnt/v/output/zensim/marathon-2026-05-25/<name>.bin + <name>.log + results.tsv

set -euo pipefail
NAME=${1:?usage: run_experiment.sh <name> [extra_args...]}
shift
OUT_DIR="/mnt/v/output/zensim/marathon-2026-05-25"
mkdir -p "$OUT_DIR"
BAKE="$OUT_DIR/${NAME}.bin"
LOG="$OUT_DIR/${NAME}.log"
TSV="$OUT_DIR/results.tsv"

# Create TSV header if first run
if [ ! -f "$TSV" ]; then
    echo -e "name\tval_geomean3\taic3_srocc_all\taic3_srocc_hf\taic3_srocc_mf\taic3_zrmse_mf" > "$TSV"
fi

echo ">>> $NAME: training..."
./target/release/zensim_mlp_train "$@" \
    --out "$BAKE" --log-every 200 > "$LOG" 2>&1

VAL=$(grep "best val" "$LOG" | tail -1 | grep -oP 'best val SROCC = \K[0-9.]+' || echo "?")
echo "  best val: $VAL"

echo ">>> $NAME: Mohammadi eval..."
EVAL=$(python3 scripts/mohammadi_eval.py "$BAKE" 2>&1)
SROCC_ALL=$(echo "$EVAL" | grep "^  All" | awk '{print $4}' || echo "?")
SROCC_HF=$(echo "$EVAL" | grep "^  HF" | awk '{print $4}' || echo "?")
SROCC_MF=$(echo "$EVAL" | grep "^  MF" | awk '{print $4}' || echo "?")
ZRMSE_MF=$(echo "$EVAL" | grep "^  MF" | awk '{print $NF}' || echo "?")

echo -e "${NAME}\t${VAL}\t${SROCC_ALL}\t${SROCC_HF}\t${SROCC_MF}\t${ZRMSE_MF}" >> "$TSV"
echo "  AIC-3 All SROCC: $SROCC_ALL  HF: $SROCC_HF  MF: $SROCC_MF  MF Z-RMSE: $ZRMSE_MF"
echo ""
