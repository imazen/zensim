#!/bin/bash
# fastclass2 — the SERVABLE lane: a 372-LAYOUT fast-class student.
#
# WHY THIS EXISTS (kernel lane, commit 8817f379): no `V1FreeExtras` slot is
# reachable from `Zensim::compute` today — `feature_v2.rs:7532` hard-codes
# `free_extras: Off` and `fold_engine.rs:158` truncates the emitted vector to
# 372. So a 265- or 289-wide bake trains fine and CANNOT be served, while a
# 156- or 228-wide bake **at the v1-372 layout** is servable with zero kernel
# work (`V1PoolsMode::Peaks` is the mode Profile D already resolves to).
# The 944-layout arms of this campaign answer "would the wider sets be worth
# building the serving path for"; THIS lane answers "is the servable one good
# enough", which is the ship question.
#
# HOW IT DIFFERS FROM THE 944 LANE, stated rather than buried — it is NOT the
# same recipe, because the 372 layout has no version of three of its legs:
#   * `tbig_hf` (the near-lossless ladder) — ABSENT at 372. The D2/D3
#     within-ref lever therefore has nothing to act on here.
#   * `safesyn_distill_hya_r4` / `tbig_teacher944` (the distillation teachers)
#     — ABSENT at 372. This lane trains on human/metric targets only.
#   * `konjnd_bpg_{train,val}` — ABSENT; the 372 layout has the older
#     `konjnd-dense` (20,160 rows), used TRAIN-ONLY at the same 1.2 weight the
#     944 recipe gives its konjnd leg, with no val twin (so no train==val).
# Everything else — epochs, pairs/epoch, group weights, loss modes, the 34
# f0..155 feature transforms, --coarse-decay — is carried over verbatim.
#
# ERA, stated per the coordinator's instruction: the training tables are the
# v1pre-era `canonical-2026-05-21` set. Their masked/IW blocks are the
# known-drifted pre-fix ones — and a <=228 slice NEVER READS THEM (f0..227
# is basic+peaks). The basic block never drifted. The eval root is the current
# `2026-08-30-full-features-372` default; the flip lane's own 372 era A/B puts
# the rank skew at <= 7e-4.
#
# Usage: train_372_student.sh <slice-file|full> <seed> <out.bin>
set -euo pipefail
KEEP="${1:?slice file, or the literal 'full'}"
SEED="${2:?seed}"
OUT="${3:?out.bin}"
TRAIN="${ZL_TRAIN:?ZL_TRAIN must point at zensim_mlp_train}"
C=/mnt/v/zen/zensim-training/canonical-2026-05-21/train
TBIG=/mnt/v/zen/zensim-training/tbig_372_200k.parquet
HIDDEN="${WR4_HIDDEN:-}"
ALPHA_HEAD="${WR4_ALPHA_HEAD:-}"
SKIPC="${WR4_SKIP:-}"
NHL="${WR4_N_HIDDEN_LAYERS:-0}"
CDECAY="${WR4_COARSE_DECAY:-1e-5}"
if [ -n "$ALPHA_HEAD" ] || [ -n "${WR4_NO_COARSE_DECAY:-}" ]; then
  echo "== NOTE: --coarse-decay DROPPED (silent no-op on the alpha head, or WR4_NO_COARSE_DECAY set)"
  CDECAY=""
fi
mkdir -p "$(dirname "$OUT")"

TRAIN_GROUPS=(
  --group "safesyn:$C/safesyn.parquet:1.0:0.5:both"
  --group "cid22_train:$C/cid22_train.parquet:1.0:2.0:both"
  --group "kadid:$C/kadid.parquet:0.5:1.0:rank"
  --group "tid:$C/tid.parquet:0.5:1.0:rank"
  --group "bigcodec:$TBIG:0.5:1.0:both"
  --group "konjnd:$C/konjnd-dense.parquet:1.2:0.0:both"
)
for g in "${TRAIN_GROUPS[@]}"; do case "$g" in --group) ;; *) p="${g#*:}"; p="${p%%:*}"; [ -f "$p" ] || { echo "ABORT: missing $p"; exit 1; };; esac; done

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

WIDTH_ARGS=(--max-features 372)
if [ "$KEEP" != "full" ]; then
  [ -f "$KEEP" ] || { echo "ABORT: slice file missing: $KEEP"; exit 1; }
  WIDTH_ARGS=(--max-features 372 --keep-features "$KEEP")
fi
EXTRA=()
[ -n "$HIDDEN" ]     && EXTRA+=( --hidden "$HIDDEN" )
[ -n "$ALPHA_HEAD" ] && EXTRA+=( --per-sample-alpha-head )
[ -n "$SKIPC" ]      && EXTRA+=( --skip-connection )

echo "== 372 student slice=$(basename "$KEEP") seed=$SEED hidden=${HIDDEN:-default128} alpha=${ALPHA_HEAD:-0} nhl=$NHL coarse_decay=${CDECAY:-DROPPED} out=$OUT $(date -u +%H:%M:%SZ)"
exec "$TRAIN" "${TRAIN_GROUPS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
  --n-hidden-layers "$NHL" --target-column human_score --target-scale 100 \
  --epochs 120 --pairs-per-epoch 50000 "${WIDTH_ARGS[@]}" \
  ${CDECAY:+--coarse-decay "$CDECAY"} "${TFARGS[@]}" --seed "$SEED" --out "$OUT"
