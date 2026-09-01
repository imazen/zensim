#!/bin/bash
# wave-r4 ARM A1 — the W10L9PH_s4004 recipe VERBATIM on the radius-4 root.
#
# The argv below is the shipped Profile-C bake's own embedded `zentrain.repro`
# argv, with ONLY the --group paths swapped to the wave-r4 root and the seed /
# --out varied. Nothing else is edited: the 64 --feature-transform flags, the
# per-group weights and loss modes, --epochs, --pairs-per-epoch, --coarse-decay
# and --max-features are byte-for-byte the recipe of record.
#   read it back with: zenpredict inspect zensim/weights/c_sdr_purity944_2026-08-29.bin
#
# A1 is registered as TEACHER / UPPER BOUND, not a W4 candidate (registration
# §4.2.1): under W4 AMENDMENT B2 no full-944 model can pass the speed clause.
#
# Usage: train_a1_flagship.sh <seed> [out.bin]
set -euo pipefail
SEED="${1:?seed required}"
R4="${WR4_ROOT:-/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01}"
V="${WR4_VIEWS:-$R4/recipe_views}"
OUTDIR="${WR4_OUT:-/mnt/v/output/zensim/wave-r4-2026-09-01/bakes}"
OUT="${2:-$OUTDIR/A1_r4_s${SEED}.bin}"
TRAIN="${ZL_TRAIN:?ZL_TRAIN must point at the zensim_mlp_train binary}"
KADIS="${WR4_KADIS:-}"     # empty = the kadis leg is ABSENT; see registration §3.1
mkdir -p "$OUTDIR"

for f in "$V/safesyn_pure.parquet" "$R4/ext_cid22_train201.parquet" \
         "$R4/ext_kadid.parquet" "$R4/ext_tid.parquet" \
         "$V/tbig_944_200k_pure.parquet" "$V/safesyn_teacher944_pure.parquet" \
         "$V/tbig_teacher944_pure.parquet" "$R4/ext_konjnd_bpg_train.parquet" \
         "$R4/ext_konjnd_bpg_val.parquet" "$V/tbig_hf_pure.parquet"; do
  [ -f "$f" ] || { echo "ABORT: missing training input $f"; exit 1; }
done

GROUPS=(
  --group "safesyn:$V/safesyn_pure.parquet:1.0:0.5:both"
  --group "cid22_train:$R4/ext_cid22_train201.parquet:1.0:2.0:both"
  --group "kadid:$R4/ext_kadid.parquet:0.5:1.0:rank"
  --group "tid:$R4/ext_tid.parquet:0.5:1.0:rank"
  --group "bigcodec:$V/tbig_944_200k_pure.parquet:0.5:1.0:both"
  --group "tsafesyn:$V/safesyn_teacher944_pure.parquet:0.5:1.0:both"
  --group "ttbig:$V/tbig_teacher944_pure.parquet:0.5:1.0:both"
  --group "konjnd_bpg:$R4/ext_konjnd_bpg_train.parquet:1.2:0.0:both"
  --group "konjnd_bpg_val:$R4/ext_konjnd_bpg_val.parquet:0.0:1.5:both"
)
if [ -n "$KADIS" ]; then
  [ -f "$KADIS" ] || { echo "ABORT: WR4_KADIS set but missing: $KADIS"; exit 1; }
  GROUPS+=( --group "kadis:$KADIS:0.15:1.0:both" )
  echo "== kadis leg PRESENT: $KADIS"
else
  echo "== kadis leg ABSENT (recipe deviation, registered) — arm is A1-nokadis"
fi

# the 64 feature transforms, VERBATIM and IN ARGV ORDER from the embedded repro
TF=(
 "winsor_p99:100:1.46128e-06,0.000106967"  "winsor_p99:139:4.06965e-07,4.32286e-05"
 "signed_cbrt:61:"  "winsor_p99:152:2.44617e-16,4.53981e-05"
 "winsor_p99:126:5.45232e-16,5.2143e-05"  "winsor_p99:113:2.55688e-16,6.55178e-05"
 "signed_cbrt:22:"  "winsor_p99:87:1.33566e-15,7.64692e-05"
 "signed_cbrt:132:"  "signed_cbrt:130:"
 "signed_cbrt:131:"  "winsor_p99:74:3.01356e-16,8.6061e-05"
 "winsor_p99:134:0.000331953,0.00794472"  "winsor_p99:48:4.53039e-15,0.000103628"
 "winsor_p99:155:0,0.163769"  "winsor_p99:35:4.23601e-16,0.000101019"
 "winsor_p99:135:0.000199307,0.00491572"  "winsor_p99:129:0,0.203516"
 "winsor_p99:9:4.04794e-07,4.9929e-05"  "winsor_p99:90:0,0.442516"
 "winsor_p99:116:0,0.329577"  "winsor_p99:133:3.24358e-05,0.00573569"
 "winsor_p99:142:0,0.0740237"  "winsor_p99:51:0,0.381636"
 "signed_cbrt:92:"  "winsor_p99:77:0,0.285164"
 "signed_cbrt:93:"  "winsor_p99:12:0,0.318713"
 "signed_cbrt:137:"  "signed_cbrt:38:"
 "signed_cbrt:91:"  "winsor_p99:95:0.000610845,0.0131861"
 "winsor_p99:64:0,0.000272277"  "winsor_p99:141:0,0.0431522"
 "winsor_p99:96:0.000119094,0.0137795"  "winsor_p99:118:1.4128e-05,0.0366709"
 "winsor_p99:144:3.92265e-05,0.0499891"  "winsor_p99:103:0,0.0600395"
 "winsor_p99:102:0,0.0443022"  "winsor_p99:119:5.98459e-06,0.0301378"
 "winsor_p99:731:0,0"  "winsor_p99:748:0,0"
 "winsor_p99:765:0,0"  "winsor_p99:782:0,0"
 "winsor_p99:799:0,0"  "winsor_p99:816:0,0"
 "winsor_p99:833:0,0"  "winsor_p99:850:0,0"
 "winsor_p99:867:0,0"  "winsor_p99:884:0,0"
 "winsor_p99:901:0,0"  "winsor_p99:918:0,0"
 "winsor_p99:732:0,0"  "winsor_p99:749:0,0"
 "winsor_p99:766:0,0"  "winsor_p99:783:0,0"
 "winsor_p99:800:0,0"  "winsor_p99:817:0,0"
 "winsor_p99:834:0,0"  "winsor_p99:851:0,0"
 "winsor_p99:868:0,0"  "winsor_p99:885:0,0"
 "winsor_p99:902:0,0"  "winsor_p99:919:0,0"
)
TFARGS=(); for t in "${TF[@]}"; do TFARGS+=( --feature-transform "$t" ); done

echo "== A1 seed=$SEED out=$OUT groups=${#GROUPS[@]} transforms=${#TF[@]} $(date -u +%H:%M:%SZ)"
exec "$TRAIN" "${GROUPS[@]}" \
  --n-hidden-layers 0 --target-column human_score --target-scale 100 \
  --epochs 120 --pairs-per-epoch 50000 --max-features 944 --allow-narrow-features \
  --coarse-decay 1e-5 "${TFARGS[@]}" \
  --group "tbig_hf:$V/tbig_hf_pure.parquet:1.0:0.0:both" \
  --seed "$SEED" --out "$OUT"
