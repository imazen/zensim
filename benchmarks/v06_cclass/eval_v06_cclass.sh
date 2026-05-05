#!/usr/bin/env bash
# Evaluate V0_6+content_class on KADID/TID/CID22/KonJND.
#
# Usage: bash eval_v06_cclass.sh <bake.bin>

set -euo pipefail

BAKE=${1:-}
[ -n "$BAKE" ] || { echo "usage: $0 <bake.bin>"; exit 1; }
[ -f "$BAKE" ] || { echo "missing bake $BAKE"; exit 1; }

ROOT=${ROOT:-/home/lilith/work/zen/zensim}
EVAL=$ROOT/target/release/examples/dataset_metric_baseline
TSV=/mnt/v/output/zensim/synthetic-v2/zenanalyze_union_v1_cclass.tsv
KADID=/mnt/v/dataset/kadid10k
TID=/mnt/v/dataset/tid2013
CID22=/mnt/v/dataset/cid22/CID22_validation_set
KONJND=/mnt/v/datasets/KonJND-1k/KonJND-1k

DCT_HF_CCLASS_FEATS=dct_compressibility_y,dct_compressibility_uv,high_freq_energy_ratio,cclass_photo,cclass_screen,cclass_lineart,cclass_synthetic,cclass_document

[ -x "$EVAL" ] || { cd "$ROOT" && cargo build --release -p zensim-bench --example dataset_metric_baseline; }

TS=$(date -u +%Y%m%dT%H%M%S)
EVAL_DIR=/tmp/eval_v06_cclass_${TS}
mkdir -p "$EVAL_DIR"
echo "[$(date +%H:%M:%S)] eval $BAKE → $EVAL_DIR"

"$EVAL" \
  --kadid $KADID --tid $TID --cid22 $CID22 --konjnd $KONJND \
  --v04-bake "$BAKE" \
  --zenanalyze-tsv "$TSV" --zenanalyze-features "$DCT_HF_CCLASS_FEATS" \
  --max-pairs 1500 --per-pair-output "$EVAL_DIR/perpair.csv" \
  > "$EVAL_DIR/eval.log" 2>&1

echo "[$(date +%H:%M:%S)] done"
echo "  perpair: $EVAL_DIR/perpair.csv"
echo "  log:     $EVAL_DIR/eval.log"
echo
echo "=== summary table ==="
grep -E "^\| (KADIK10k|TID2013|CID22|KonJND)" "$EVAL_DIR/eval.log" || tail -30 "$EVAL_DIR/eval.log"
