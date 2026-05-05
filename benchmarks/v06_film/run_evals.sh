#!/usr/bin/env bash
# Run paired evals for V0_6 baseline rebake AND V0_6+FiLM on
# KADID/TID/CID22/KonJND. Persists per-pair CSVs and summary tables.
#
# Usage: bash run_evals.sh <film_primary_bake.bin> [<rebake_bin>]

set -euo pipefail

FILM_BAKE=${1:-}
REBAKE=${2:-/mnt/v/output/zensim/synthetic-v2/runs/v06_baseline_rebake_20260505T202738.bin}
[ -n "$FILM_BAKE" ] || { echo "usage: $0 <film_primary_bake.bin>"; exit 1; }
[ -f "$FILM_BAKE" ] || { echo "missing $FILM_BAKE"; exit 1; }
[ -f "$REBAKE" ] || { echo "missing rebake $REBAKE"; exit 1; }

STEM=$(echo "$FILM_BAKE" | sed -E 's/(\.c[0-9]+_[a-z]+)?\.bin$//')
MANIFEST=${STEM}.film_manifest.tsv
[ -f "$MANIFEST" ] || { echo "missing manifest $MANIFEST"; exit 1; }

ROOT=${ROOT:-/home/lilith/work/zen/zensim--v06-film}
EVAL=$ROOT/target/release/examples/dataset_metric_baseline
TSV=/mnt/v/output/zensim/synthetic-v2/zenanalyze_union_v1_cclass.tsv
KADID=/mnt/v/dataset/kadid10k
TID=/mnt/v/dataset/tid2013
CID22=/mnt/v/dataset/cid22/CID22_validation_set
KONJND=/mnt/v/datasets/KonJND-1k/KonJND-1k

DCT_HF_FEATS_NOCCLASS=dct_compressibility_y,dct_compressibility_uv,high_freq_energy_ratio
DCT_HF_CCLASS_FEATS=dct_compressibility_y,dct_compressibility_uv,high_freq_energy_ratio,cclass_photo,cclass_screen,cclass_lineart,cclass_synthetic,cclass_document

ARTIFACTS=$ROOT/benchmarks/v06_film/artifacts
mkdir -p "$ARTIFACTS"
TS=$(date -u +%Y-%m-%d)

echo "[$(date +%H:%M:%S)] === eval V0_6 baseline rebake (no cclass) ==="
"$EVAL" \
  --kadid $KADID --tid $TID --cid22 $CID22 --konjnd $KONJND \
  --v04-bake "$REBAKE" \
  --zenanalyze-tsv "$TSV" --zenanalyze-features "$DCT_HF_FEATS_NOCCLASS" \
  --max-pairs 1500 --per-pair-output "$ARTIFACTS/v06_rebake_perpair_${TS}.csv" \
  > "$ARTIFACTS/v06_rebake_eval_${TS}.log" 2>&1
echo "  rebake done: $ARTIFACTS/v06_rebake_eval_${TS}.log"

echo "[$(date +%H:%M:%S)] === eval V0_6+FiLM (manifest dispatch) ==="
"$EVAL" \
  --kadid $KADID --tid $TID --cid22 $CID22 --konjnd $KONJND \
  --film-manifest "$MANIFEST" \
  --zenanalyze-tsv "$TSV" --zenanalyze-features "$DCT_HF_CCLASS_FEATS" \
  --max-pairs 1500 --per-pair-output "$ARTIFACTS/v06_film_perpair_${TS}.csv" \
  > "$ARTIFACTS/v06_film_eval_${TS}.log" 2>&1
echo "  film done:   $ARTIFACTS/v06_film_eval_${TS}.log"

# Also evaluate just the photo class bake (FiLM class 0) without manifest
# to isolate the FiLM contribution (i.e., does any bake at all change
# the score?).
PHOTO_BAKE=$(awk -F'\t' '$2=="photo" {print $3}' "$MANIFEST")
if [ -f "$PHOTO_BAKE" ]; then
  echo "[$(date +%H:%M:%S)] === eval V0_6+FiLM (photo-class only, single bake) ==="
  "$EVAL" \
    --kadid $KADID --tid $TID --cid22 $CID22 --konjnd $KONJND \
    --v04-bake "$PHOTO_BAKE" \
    --zenanalyze-tsv "$TSV" --zenanalyze-features "$DCT_HF_CCLASS_FEATS" \
    --max-pairs 1500 --per-pair-output "$ARTIFACTS/v06_film_photo_perpair_${TS}.csv" \
    > "$ARTIFACTS/v06_film_photo_eval_${TS}.log" 2>&1
  echo "  film photo done: $ARTIFACTS/v06_film_photo_eval_${TS}.log"
fi

echo
echo "=== summary ==="
for tag in v06_rebake v06_film v06_film_photo; do
  LOG="$ARTIFACTS/${tag}_eval_${TS}.log"
  [ -f "$LOG" ] || continue
  echo
  echo "--- $tag ---"
  grep -E "^\| (KADIK10k|TID2013|CID22|KonJND)" "$LOG" || tail -10 "$LOG"
done
