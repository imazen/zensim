#!/bin/bash
# NOTE: the group array is TRAIN_GROUPS, never GROUPS -- $GROUPS is a bash
# READONLY builtin (the primary group id, 1000 here), so assigning to it silently
# fails and the trainer receives a bare "1000" ("error: unexpected argument '1000'
# found"). Hit and fixed 2026-09-01; documented in zensim/CLAUDE.md "Shell
# scripting gotchas".
# wave-r4 ARMS A3 / A4 — the 156-compute-set student lane (the wave's CANDIDATE
# lane; A1 is a teacher, see registration §4.2.1).
#
# Same recipe skeleton as A1, but --max-features 156 so the model reads ONLY the
# basic block f0..f155 and therefore prices at the `fold156_basic` walk, which is
# the class W4 AMENDMENT B2 measures against (<=1.25x, 1T and 8T).
#
# Only the f0..155 feature transforms carry over; the 24 append2 winsor guards
# (indices 731..919) are DROPPED because those inputs do not exist at width 156.
# That is a real recipe difference from A1 and is recorded, not hidden.
#
#   A3  MODE=human    : targets human_score  (the plain 156 retrain)
#   A4  MODE=distill  : the teacher legs are re-targeted at A1's OWN outputs,
#                       so the student fits the 944 flagship rather than the
#                       labels (WR4_TEACHER must point at the A1 bake).
#
# Usage: train_156_student.sh <MODE:human|distill> <seed> [out.bin]
set -euo pipefail
MODE="${1:?MODE required: human|distill}"
SEED="${2:?seed required}"
R4="${WR4_ROOT:-/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01}"
V="${WR4_VIEWS:-$R4/recipe_views}"
OUTDIR="${WR4_OUT:-/mnt/v/output/zensim/wave-r4-2026-09-01/bakes}"
TRAIN="${ZL_TRAIN:?ZL_TRAIN must point at the zensim_mlp_train binary}"
KADIS="${WR4_KADIS:-}"
mkdir -p "$OUTDIR"

case "$MODE" in
  human)   ARM=A3; TSAFE="$V/safesyn_teacher944_pure.parquet"; TTBIG="$V/tbig_teacher944_pure.parquet" ;;
  distill) ARM=A4
           TSAFE="${WR4_DISTILL_SAFESYN:-$V/safesyn_distill_a1.parquet}"
           TTBIG="${WR4_DISTILL_TBIG:-$V/tbig_distill_a1.parquet}"
           for f in "$TSAFE" "$TTBIG"; do
             [ -f "$f" ] || { echo "ABORT: distill target table missing: $f
  build it first with benchmarks/wave_r4_2026-09-01/make_distill_targets.sh <A1.bin>"; exit 1; }
           done ;;
  *) echo "ABORT: MODE must be human|distill"; exit 1 ;;
esac
OUT="${3:-$OUTDIR/${ARM}_156_s${SEED}.bin}"

TRAIN_GROUPS=(
  --group "safesyn:$V/safesyn_pure.parquet:1.0:0.5:both"
  --group "cid22_train:$R4/ext_cid22_train201.parquet:1.0:2.0:both"
  --group "kadid:$R4/ext_kadid.parquet:0.5:1.0:rank"
  --group "tid:$R4/ext_tid.parquet:0.5:1.0:rank"
  --group "bigcodec:$V/tbig_944_200k_pure.parquet:0.5:1.0:both"
  --group "tsafesyn:$TSAFE:0.5:1.0:both"
  --group "ttbig:$TTBIG:0.5:1.0:both"
  --group "konjnd_bpg:$R4/ext_konjnd_bpg_train.parquet:1.2:0.0:both"
  --group "konjnd_bpg_val:$R4/ext_konjnd_bpg_val.parquet:0.0:1.5:both"
  --group "tbig_hf:$V/tbig_hf_pure.parquet:1.0:0.0:both"
)
if [ -n "$KADIS" ]; then TRAIN_GROUPS+=( --group "kadis:$KADIS:0.15:1.0:both" ); fi
for g in "${TRAIN_GROUPS[@]}"; do case "$g" in --group) ;; *) p="${g#*:}"; p="${p%%:*}"; [ -f "$p" ] || { echo "ABORT: missing $p"; exit 1; };; esac; done

# f0..155 transforms only — the append2 guards (731..919) have no input at 156
TF=(
 "winsor_p99:100:1.46128e-06,0.000106967"  "winsor_p99:139:4.06965e-07,4.32286e-05"
 "signed_cbrt:61:"                          "winsor_p99:152:2.44617e-16,4.53981e-05"
 "winsor_p99:126:5.45232e-16,5.2143e-05"    "winsor_p99:113:2.55688e-16,6.55178e-05"
 "signed_cbrt:22:"                          "winsor_p99:87:1.33566e-15,7.64692e-05"
 "signed_cbrt:132:"                         "signed_cbrt:130:"
 "signed_cbrt:131:"                         "winsor_p99:74:3.01356e-16,8.6061e-05"
 "winsor_p99:134:0.000331953,0.00794472"    "winsor_p99:48:4.53039e-15,0.000103628"
 "winsor_p99:155:0,0.163769"                "winsor_p99:35:4.23601e-16,0.000101019"
 "winsor_p99:135:0.000199307,0.00491572"    "winsor_p99:129:0,0.203516"
 "winsor_p99:9:4.04794e-07,4.9929e-05"      "winsor_p99:90:0,0.442516"
 "winsor_p99:116:0,0.329577"                "winsor_p99:133:3.24358e-05,0.00573569"
 "winsor_p99:142:0,0.0740237"               "winsor_p99:51:0,0.381636"
 "signed_cbrt:92:"                          "winsor_p99:77:0,0.285164"
 "signed_cbrt:93:"                          "winsor_p99:12:0,0.318713"
 "signed_cbrt:137:"                         "signed_cbrt:38:"
 "signed_cbrt:91:"                          "winsor_p99:95:0.000610845,0.0131861"
 "winsor_p99:64:0,0.000272277"              "winsor_p99:141:0,0.0431522"
 "winsor_p99:96:0.000119094,0.0137795"      "winsor_p99:118:1.4128e-05,0.0366709"
 "winsor_p99:144:3.92265e-05,0.0499891"     "winsor_p99:103:0,0.0600395"
 "winsor_p99:102:0,0.0443022"               "winsor_p99:119:5.98459e-06,0.0301378"
)
TFARGS=(); for t in "${TF[@]}"; do TFARGS+=( --feature-transform "$t" ); done

echo "== $ARM MODE=$MODE seed=$SEED out=$OUT transforms=${#TF[@]} (append2 guards dropped) $(date -u +%H:%M:%SZ)"
exec "$TRAIN" "${TRAIN_GROUPS[@]}" \
  --n-hidden-layers 0 --target-column human_score --target-scale 100 \
  --epochs 120 --pairs-per-epoch 50000 --max-features 156 --allow-narrow-features \
  --coarse-decay 1e-5 "${TFARGS[@]}" --seed "$SEED" --out "$OUT"
