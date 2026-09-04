#!/usr/bin/env bash
# Behavioural gate for --init-seed / --sample-seed.
#
#   1. omitting both is byte-identical to plain --seed  (backward compat)
#   2. same --sample-seed, different --init-seed  -> SAME draws, DIFFERENT model
#   3. different --sample-seed, same --init-seed  -> DIFFERENT draws
#
# Draws are compared by the sample-sequence digest (ZENSIM_SAMPLE_DIGEST=1);
# models by the bake bytes with the deliberately volatile zentrain.repro
# block stripped (timestamp/argv/cwd make whole-file identity impossible).
set -euo pipefail
V=${VIEWS:-/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views}
T=${TRAIN:?set TRAIN to a zensim_mlp_train}
S=${STRIP:?set STRIP to a bake_dial_refit}
O=${1:-/mnt/v/output/zensim/subset-study-2026-09-04/splitgate}
mkdir -p "$O"

go () { # $1 = label, rest = seed flags
  local l=$1; shift
  ZENSIM_SAMPLE_DIGEST=1 "$T" \
    --group a:"$V/ext_aic4.parquet":1.0:1.0 \
    --group b:"$V/ext_konjnd_jpeg_val.parquet":0.5:1.0 \
    --group v:"$V/ext_sdr25.parquet":0.0:1.0 \
    --max-features 944 --hidden 16 --epochs 3 --pairs-per-epoch 2000 \
    --out "$O/$l.bin" "$@" > "$O/$l.log" 2>&1
  "$S" strip --in "$O/$l.bin" --out "$O/${l}_nr.bin" --key zentrain.repro >/dev/null 2>&1
}
dg () { grep '^ZENSIM_SAMPLE_DIGEST ' "$O/$1.log" | awk '{print $2}'; }
bk () { sha256sum "$O/${1}_nr.bin" | cut -d' ' -f1; }

go A --seed 100
go B --init-seed 999 --sample-seed 100
go C --init-seed 100 --sample-seed 999
go D --seed 100

fail=0
chk () { if [ "$2" = "$3" ]; then [ "$4" = same ] && echo "PASS $1" || { echo "FAIL $1 (unexpectedly equal)"; fail=1; }
         else [ "$4" = diff ] && echo "PASS $1" || { echo "FAIL $1 ($2 != $3)"; fail=1; }; fi }
chk "1 --seed is reproducible (bake)"                  "$(bk A)" "$(bk D)" same
chk "1 --seed is reproducible (draws)"                 "$(dg A)" "$(dg D)" same
chk "2 same sample-seed -> SAME draws"                 "$(dg A)" "$(dg B)" same
chk "2 different init-seed -> DIFFERENT model"         "$(bk A)" "$(bk B)" diff
chk "3 different sample-seed -> DIFFERENT draws"       "$(dg A)" "$(dg C)" diff
chk "3 same init-seed alone -> DIFFERENT model"        "$(bk A)" "$(bk C)" diff
exit $fail
