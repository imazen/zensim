#!/usr/bin/env bash
# Byte-identity gate for the sampling-owner extraction: the pre-extraction
# binary (@-) and the post-extraction binary (@) must produce IDENTICAL bake
# bytes on every sampling path.
set -euo pipefail
V=${VIEWS:-/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views}
# Both binaries are supplied by the caller: OLD must be built from the commit
# BEFORE the sampling-owner extraction, NEW from the commit after.
OLD=${OLD_TRAIN:?set OLD_TRAIN to a pre-extraction zensim_mlp_train}
NEW=${NEW_TRAIN:?set NEW_TRAIN to a post-extraction zensim_mlp_train}
OUT=${1:-/mnt/v/output/zensim/subset-study-2026-09-04/gate}
mkdir -p "$OUT"

run () {  # $1=label  $2=binary  $3..=extra args
  local label=$1 bin=$2; shift 2
  "$bin" \
    --group a:"$V/ext_aic4.parquet":1.0:1.0 \
    --group b:"$V/ext_konjnd_jpeg_val.parquet":0.5:1.0 \
    --group v:"$V/ext_sdr25.parquet":0.0:1.0 \
    --max-features 944 --hidden 16 --epochs 3 --pairs-per-epoch 2000 \
    --seed 4004 --out "$OUT/$label.bin" "$@" > "$OUT/$label.log" 2>&1
}

declare -A ARMS=(
  [plain]=""
  [withinref]="--group-withinref"
  [hqboost]="--high-q-boost 3.0"
)
fail=0
for arm in plain withinref hqboost; do
  extra=${ARMS[$arm]}
  if [ "$arm" = withinref ]; then
    # withinref is a 5th field on the group spec, not a flag
    "$OLD" --group a:"$V/ext_aic4.parquet":1.0:1.0:withinref \
           --group b:"$V/ext_konjnd_jpeg_val.parquet":0.5:1.0:withinref \
           --group v:"$V/ext_sdr25.parquet":0.0:1.0 \
           --max-features 944 --hidden 16 --epochs 3 --pairs-per-epoch 2000 \
           --seed 4004 --out "$OUT/old_$arm.bin" > "$OUT/old_$arm.log" 2>&1
    "$NEW" --group a:"$V/ext_aic4.parquet":1.0:1.0:withinref \
           --group b:"$V/ext_konjnd_jpeg_val.parquet":0.5:1.0:withinref \
           --group v:"$V/ext_sdr25.parquet":0.0:1.0 \
           --max-features 944 --hidden 16 --epochs 3 --pairs-per-epoch 2000 \
           --seed 4004 --out "$OUT/new_$arm.bin" > "$OUT/new_$arm.log" 2>&1
  else
    run "old_$arm" "$OLD" $extra
    run "new_$arm" "$NEW" $extra
  fi
  a=$(sha256sum "$OUT/old_$arm.bin" | cut -d' ' -f1)
  b=$(sha256sum "$OUT/new_$arm.bin" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "PASS  $arm  $a"; else echo "FAIL  $arm  old=$a new=$b"; fail=1; fi
done
exit $fail
