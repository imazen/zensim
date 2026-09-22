#!/usr/bin/env bash
# verdict-lane peer scoring (X1).
# Emits scores/<peer>__on__<leg>.csv in the metrics.py layout
# (ref_path,dist_path,target,score,E) for:
#   fastssim2     — extractor audit channel (same decoded RGB8 buffers)
#   bake_prof_b   — zensim profile B byid_2026-09-06 bake
#   bake_prof_d   — zensim profile D byid_2026-09-06 bake
#   bake_r915_basic228 / bake_r915_y60 — frozen Rev3 ensembles (5 bakes)
# Legs: the four TRAIN development tables (existing parquets; R915 per-row
# scores REUSED from recovery/development where present) + kadid_dev,
# konfig_val, cid22b (features csv -> parquet here).
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-verdict-2026-09-20
S2D=/mnt/v/output/zensim/dvifm-screen2d-2026-09-19
VALBIN=/home/lilith/work/zen/zensim/target/release
W=/home/lilith/work/zen/zensim/zensim/weights
REC=/var/tmp/zensim-validation-2026-09-15/recovery
DEV=/var/tmp/zensim-validation-2026-09-15/recovery/development
R2S=$S2D/tools/rows_to_scores.py
C2P=$S2D/tools/csv_to_parquet.py
export RAYON_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
fails=0
mkdir -p "$OUT/scores" "$OUT/parquets"

# leg -> (pairs tsv, parquet)
declare -A TSV PQ
TSV[safesyn_dev]=$OUT/pairs/safesyn_dev.tsv
TSV[cid22_dev]=$OUT/pairs/cid22_dev.tsv
TSV[codec_dev]=$OUT/pairs/codec_dev.tsv
TSV[human_dev]=$OUT/pairs/human_dev.tsv
TSV[kadid_dev]=$OUT/pairs/kadid_dev.tsv
TSV[konfig_val]=$OUT/pairs/konfig_val.tsv
TSV[cid22b]=$OUT/pairs/cid22b.tsv
# after the single unseal, score CSVs must carry real B labels
[ -s "$OUT/pairs/cid22b_unsealed.tsv" ] && \
  TSV[cid22b]=$OUT/pairs/cid22b_unsealed.tsv
PQ[safesyn_dev]=$REC/dedup-tables/safesyn_development.parquet
PQ[cid22_dev]=$REC/dedup-tables/cid22_development.parquet
PQ[codec_dev]=$REC/dedup-tables/codec_development.parquet
PQ[human_dev]=$REC/dedup-tables/human_development.parquet
for leg in kadid_dev konfig_val cid22b; do
  PQ[$leg]=$OUT/parquets/${leg}.parquet
  if [ ! -s "${PQ[$leg]}" ]; then
    python3 "$C2P" "$OUT/features/${leg}.csv" "${PQ[$leg]}" \
      || { echo "PARQUET-FAIL $leg"; fails=$((fails+1)); }
  fi
done

emit_scores() { # tag leg bakes...
  local tag=$1 leg=$2; shift 2
  local o=$OUT/scores/${tag}__on__${leg}.tsv
  local c=$OUT/scores/${tag}__on__${leg}.csv
  [ -s "$c" ] && return 0
  [ -s "$o" ] || "$VALBIN/ensemble_score_rows" \
    $(printf -- "--bake %q " "$@") --parquet "${PQ[$leg]}" --output "$o" \
    || { echo "BAKE-FAIL $tag $leg"; fails=$((fails+1)); return 1; }
  python3 "$R2S" --pairs "${TSV[$leg]}" --bake-tsv "$o" --out "$c" \
    || { echo "JOIN-FAIL $tag $leg"; fails=$((fails+1)); }
}

B=$W/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin
D=$W/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin
R228=($REC/calibrated/R915_basic228_h128_s17101.bin
      $REC/calibrated/R915_basic228_h128_s17103.bin
      $REC/calibrated/R915_basic228_h128_s17107.bin
      $REC/calibrated/R915_basic228_h128_s17111.bin
      $REC/calibrated/R915_basic228_h128_s17113.bin)
R60=($REC/calibrated/R915_y60_h32_s17101.bin
     $REC/calibrated/R915_y60_h32_s17103.bin
     $REC/calibrated/R915_y60_h32_s17107.bin
     $REC/calibrated/R915_y60_h32_s17111.bin
     $REC/calibrated/R915_y60_h32_s17113.bin)

for leg in safesyn_dev cid22_dev codec_dev human_dev; do
  emit_scores bake_prof_b "$leg" "$B"
  emit_scores bake_prof_d "$leg" "$D"
  # Rev3 ensembles: reuse frozen per-row scores where they exist (idx =
  # parquet row = pairs row)
  short=${leg%_dev}
  for ens in basic228 y60; do
    src=$DEV/R915_${ens}_h128_ens5.${short}.tsv
    [ "$ens" = y60 ] && src=$DEV/R915_y60_h32_ens5.${short}.tsv
    c=$OUT/scores/bake_r915_${ens}__on__${leg}.csv
    if [ -s "$src" ] && [ ! -s "$c" ]; then
      cp "$src" "$OUT/scores/bake_r915_${ens}__on__${leg}.tsv"
      python3 "$R2S" --pairs "${TSV[$leg]}" --bake-tsv "$src" --out "$c" \
        || { echo "JOIN-FAIL r915_$ens $leg"; fails=$((fails+1)); }
    fi
  done
done
for leg in kadid_dev konfig_val cid22b; do
  # cid22b must not be scored for its table until the single unseal;
  # skip it entirely while sealed (target column would be placeholders)
  if [ "$leg" = cid22b ] && [ ! -s "$OUT/pairs/cid22b_unsealed.tsv" ]; then
    echo "SEALED cid22b — skipping until the single read"
    continue
  fi
  emit_scores bake_prof_b "$leg" "$B"
  emit_scores bake_prof_d "$leg" "$D"
  emit_scores bake_r915_basic228 "$leg" "${R228[@]}"
  emit_scores bake_r915_y60 "$leg" "${R60[@]}"
done

# fast-ssim2 from the audit channel (jsonl emitted during extraction)
for leg in safesyn_dev cid22_dev codec_dev human_dev kadid_dev konfig_val cid22b; do
  if [ "$leg" = cid22b ] && [ ! -s "$OUT/pairs/cid22b_unsealed.tsv" ]; then
    continue
  fi
  c=$OUT/scores/fastssim2__on__${leg}.csv
  [ -s "$c" ] && continue
  python3 "$R2S" --pairs "${TSV[$leg]}" \
    --audit-jsonl "$OUT/scores/${leg}_audit_ssim2.jsonl" --out "$c" \
    || { echo "SSIM2-FAIL $leg"; fails=$((fails+1)); }
done
echo "PEER SCORING DONE fails=$fails"
