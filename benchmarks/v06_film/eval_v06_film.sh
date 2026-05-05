#!/usr/bin/env bash
# Evaluate V0_6+FiLM on KADID/TID/CID22/KonJND. Routes each pair
# through the per-class bake selected by its reference's cclass.
#
# Usage: bash eval_v06_film.sh <bake.bin>
#   <bake.bin> is the primary bake (any of the per-class .bin works
#   for argument passing; the manifest lives next to it as
#   <bake-stem>.film_manifest.tsv).

set -euo pipefail

BAKE=${1:-}
[ -n "$BAKE" ] || { echo "usage: $0 <bake.bin>"; exit 1; }
[ -f "$BAKE" ] || { echo "missing bake $BAKE"; exit 1; }

# Find the manifest. Bake path: .../v06_film_TS.bin → .../v06_film_TS.film_manifest.tsv
STEM=$(echo "$BAKE" | sed -E 's/(\.c[0-9]+_[a-z]+)?\.bin$//')
MANIFEST=${STEM}.film_manifest.tsv
[ -f "$MANIFEST" ] || { echo "missing manifest $MANIFEST"; exit 1; }
echo "FiLM manifest: $MANIFEST"

ROOT=${ROOT:-/home/lilith/work/zen/zensim--v06-film}
EVAL=$ROOT/target/release/examples/dataset_metric_baseline
TSV=/mnt/v/output/zensim/synthetic-v2/zenanalyze_union_v1_cclass.tsv
KADID=/mnt/v/dataset/kadid10k
TID=/mnt/v/dataset/tid2013
CID22=/mnt/v/dataset/cid22/CID22_validation_set
KONJND=/mnt/v/datasets/KonJND-1k/KonJND-1k

DCT_HF_CCLASS_FEATS=dct_compressibility_y,dct_compressibility_uv,high_freq_energy_ratio,cclass_photo,cclass_screen,cclass_lineart,cclass_synthetic,cclass_document

[ -x "$EVAL" ] || { cd "$ROOT" && cargo build --release -p zensim-bench --example dataset_metric_baseline; }

TS=$(date -u +%Y%m%dT%H%M%S)
EVAL_DIR=/tmp/eval_v06_film_${TS}
mkdir -p "$EVAL_DIR"
echo "[$(date +%H:%M:%S)] eval FiLM (manifest=$MANIFEST) → $EVAL_DIR"

"$EVAL" \
  --kadid $KADID --tid $TID --cid22 $CID22 --konjnd $KONJND \
  --film-manifest "$MANIFEST" \
  --zenanalyze-tsv "$TSV" --zenanalyze-features "$DCT_HF_CCLASS_FEATS" \
  --max-pairs 1500 --per-pair-output "$EVAL_DIR/perpair.csv" \
  > "$EVAL_DIR/eval.log" 2>&1

echo "[$(date +%H:%M:%S)] done"
echo "  perpair: $EVAL_DIR/perpair.csv"
echo "  log:     $EVAL_DIR/eval.log"
echo
echo "=== summary table ==="
grep -E "^\| (KADIK10k|TID2013|CID22|KonJND)" "$EVAL_DIR/eval.log" || tail -30 "$EVAL_DIR/eval.log"
