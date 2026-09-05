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
#   A3b/A4b (2026-09-01 amendment, closes the free-features coordination gap
#            registered §14.3 of the wave doc as "would need --keep-features
#            wired" -- WRONG, that flag already exists, added 2026-08-04
#            appendix J, well before this wave): set WR4_KEEP to
#            scripts/sota944/slice_basic156_free.txt (265 coords, f0..155 +
#            72 peaks + 37 raw-moment slots, max idx 941) and this script
#            switches to --max-features 944 --keep-features "$WR4_KEEP",
#            dropping --allow-narrow-features (944 already clears the >=372
#            floor). None of the free-set's 109 non-basic indices intersect
#            A1's 24 append2 winsor(0,0) guard columns (731..919) -- checked
#            by direct set intersection, empty -- so the SAME 34-entry TF
#            array below (f0..155 only) is correct unchanged for A3b/A4b;
#            every free-set index gets default scaling, same as it would in
#            A1's own 944-wide run for any index outside its 58 guarded ones.
#   A3b/A4b kon-weight sweep (a4bkon follow-up lane, 2026-09-01): WR4_KONJND_
#            TRAIN_W overrides the konjnd_bpg group's train weight (mirrors
#            train_a1_flagship.sh's own WR4_KONJND_TRAIN_W lever exactly, same
#            env var name, same default 1.2 = A1/A3b/A4b's own verbatim
#            value). Ported here unmodified otherwise -- see
#            benchmarks/a4bkon_2026-09-01.md for the registration.
#   A3b/A4b fast-class distillation levers (fastclass lane, 2026-09-04):
#            four NEW env vars, every one of which is a NO-OP when unset, so an
#            unset invocation reproduces A4b's bake BYTE-IDENTICALLY (gate G1 of
#            benchmarks/fastclass_distill_wave_2026-09-04.md, verified before
#            any arm was fit):
#              WR4_HIGH_Q_BOOST=<f>   -> --high-q-boost <f>  (B3 row-weight
#                                        boost, target>=90, in RankNet pair
#                                        sampling; flag omitted when unset,
#                                        which is the trainer's own 1.0 default)
#              WR4_KON_WITHINREF=1    -> konjnd_bpg group gains ",withinref"
#                                        (403 refs x 20 rows = a per-ref ladder)
#              WR4_HF_WITHINREF=1     -> tbig_hf group gains ",withinref"
#                                        (1,973 refs x 6.1 rows = the
#                                        near-lossless ladder the trainer's own
#                                        --group doc names as the motivating
#                                        case: uniform pairing leaves it ~1/7th
#                                        of the gradient)
#              WR4_N_HIDDEN_LAYERS=<n> -> --n-hidden-layers <n> (default 0,
#                                        A4b's own value; 2 = 944->128->64)
#   A3b/A4b MATRIX-SHAPE levers (fastclass2 lane, 2026-09-05): three MORE
#            env vars, every one a NO-OP when unset, so an unset invocation
#            still reproduces A4b/D3 byte-identically (gate G1 of
#            benchmarks/fastclass2_campaign_2026-09-05.md):
#              WR4_HIDDEN=<n>       -> --hidden <n>   (hidden-layer width; the
#                                      trainer's own default is 128, which is
#                                      what every A3b/A4b/FC_* bake carries, so
#                                      omitting the flag when unset keeps that
#                                      exact argv)
#              WR4_ALPHA_HEAD=1     -> --per-sample-alpha-head. Needed because
#                                      --keep-features is implemented on the
#                                      plain n_features->n_hidden->1 path AND on
#                                      --per-sample-alpha-head only; a 2-layer
#                                      student of this class is UNBUILDABLE
#                                      without it (fastclass wave arm E1 died on
#                                      exactly that refusal, its section 7.6).
#                                      Changes the HEAD as well as the depth, so
#                                      a depth-1 alpha-head control is mandatory
#                                      to read a depth effect apart from a head
#                                      effect.
#              WR4_SKIP=1           -> --skip-connection (input->output linear
#                                      skip alongside the MLP, ~n_in extra
#                                      params; a shape lever that adds no walk
#                                      cost at all)
#            WR4_KADIS was already here and is used unchanged by that lane's
#            KADIS-role arm.
#   A4  MODE=distill  : the tsafesyn leg is re-targeted at the TEACHER's
#                       output (registration §3.0.2 point 3: HYA_w084 =
#                       0.84*W10L9PH_s4004 + 0.16*Q7b, era-1-trained, grafted
#                       onto wave-r4 r4 features by build_teacher944.py
#                       --graft-from — the row-identity join, NOT a re-forward
#                       over new features; A1 is the secondary/comparison
#                       teacher, not the primary). WR4_DISTILL_SAFESYN must
#                       point at the grafted safesyn table.
#
#                       WR4_DISTILL_TBIG is OPTIONAL (2026-09-01 amendment):
#                       when unset, the ttbig group is DROPPED rather than
#                       falling back to the old default self-teacher target —
#                       that would silently mix two different teachers across
#                       the two legs. Building an HYA-graft for tbig needs a
#                       row-matched era-1-pools subset of the 208,169-row raw
#                       tbig_pools944 table down to wave-r4's 192,714-row
#                       filtered recipe view; not done this session. So A4
#                       (as currently run) isolates ONE variable: does the
#                       tsafesyn leg's teacher-vs-human target matter on r4
#                       features, holding bigcodec at its human target as A3
#                       does. This is a scoped-down A4, stated rather than
#                       hidden — a fuller version with a tbig teacher leg is
#                       registered as follow-up work, not claimed as done.
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
WR4_KEEP="${WR4_KEEP:-}"
KONW="${WR4_KONJND_TRAIN_W:-1.2}"
KONFLAGS="both"; if [ -n "${WR4_KON_WITHINREF:-}" ]; then KONFLAGS="both,withinref"; fi
HFFLAGS="both";  if [ -n "${WR4_HF_WITHINREF:-}" ];  then HFFLAGS="both,withinref"; fi
NHL="${WR4_N_HIDDEN_LAYERS:-0}"
HIDDEN="${WR4_HIDDEN:-}"
ALPHA_HEAD="${WR4_ALPHA_HEAD:-}"
SKIPC="${WR4_SKIP:-}"
mkdir -p "$OUTDIR"

TTBIG=""
case "$MODE" in
  human)   ARM=A3; TSAFE="$V/safesyn_teacher944_pure.parquet"; TTBIG="$V/tbig_teacher944_pure.parquet" ;;
  distill) ARM=A4
           TSAFE="${WR4_DISTILL_SAFESYN:?WR4_DISTILL_SAFESYN must point at the HYA-grafted safesyn distill table}"
           [ -f "$TSAFE" ] || { echo "ABORT: distill target table missing: $TSAFE"; exit 1; }
           if [ -n "${WR4_DISTILL_TBIG:-}" ]; then
             TTBIG="$WR4_DISTILL_TBIG"
             [ -f "$TTBIG" ] || { echo "ABORT: WR4_DISTILL_TBIG set but missing: $TTBIG"; exit 1; }
           else
             echo "== ttbig leg DROPPED (WR4_DISTILL_TBIG unset) — bigcodec stays human-target-only, see header note"
           fi ;;
  *) echo "ABORT: MODE must be human|distill"; exit 1 ;;
esac
if [ -n "$WR4_KEEP" ]; then ARM="${ARM}b"; fi
OUT="${3:-$OUTDIR/${ARM}_156_s${SEED}.bin}"

TRAIN_GROUPS=(
  --group "safesyn:$V/safesyn_pure.parquet:1.0:0.5:both"
  --group "cid22_train:$R4/ext_cid22_train201.parquet:1.0:2.0:both"
  --group "kadid:$R4/ext_kadid.parquet:0.5:1.0:rank"
  --group "tid:$R4/ext_tid.parquet:0.5:1.0:rank"
  --group "bigcodec:$V/tbig_944_200k_pure.parquet:0.5:1.0:both"
  --group "tsafesyn:$TSAFE:0.5:1.0:both"
  --group "konjnd_bpg:$R4/ext_konjnd_bpg_train.parquet:$KONW:0.0:$KONFLAGS"
  --group "konjnd_bpg_val:$R4/ext_konjnd_bpg_val.parquet:0.0:1.5:both"
  --group "tbig_hf:$V/tbig_hf_pure.parquet:1.0:0.0:$HFFLAGS"
)
if [ -n "$TTBIG" ]; then TRAIN_GROUPS+=( --group "ttbig:$TTBIG:0.5:1.0:both" ); fi
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

WIDTH_ARGS=(--max-features 156 --allow-narrow-features)
if [ -n "$WR4_KEEP" ]; then
  [ -f "$WR4_KEEP" ] || { echo "ABORT: WR4_KEEP set but missing: $WR4_KEEP"; exit 1; }
  WIDTH_ARGS=(--max-features 944 --keep-features "$WR4_KEEP")
fi

EXTRA=()
if [ -n "${WR4_HIGH_Q_BOOST:-}" ]; then EXTRA+=( --high-q-boost "$WR4_HIGH_Q_BOOST" ); fi
# Matrix-shape levers. Each flag is OMITTED when its env var is unset, so the
# emitted argv (and therefore the embedded zentrain.repro) is byte-identical to
# a pre-2026-09-05 run -- that is what gate G1 checks.
if [ -n "$HIDDEN" ];     then EXTRA+=( --hidden "$HIDDEN" ); fi
if [ -n "$ALPHA_HEAD" ]; then EXTRA+=( --per-sample-alpha-head ); fi
if [ -n "$SKIPC" ];      then EXTRA+=( --skip-connection ); fi

echo "== $ARM MODE=$MODE seed=$SEED out=$OUT transforms=${#TF[@]} width=(${WIDTH_ARGS[*]}) konjnd_train_w=$KONW kon_flags=$KONFLAGS hf_flags=$HFFLAGS n_hidden_layers=$NHL hidden=${HIDDEN:-default128} alpha_head=${ALPHA_HEAD:-0} skip=${SKIPC:-0} extra=(${EXTRA[*]+"${EXTRA[*]}"}) $(date -u +%H:%M:%SZ)"
exec "$TRAIN" "${TRAIN_GROUPS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
  --n-hidden-layers "$NHL" --target-column human_score --target-scale 100 \
  --epochs 120 --pairs-per-epoch 50000 "${WIDTH_ARGS[@]}" \
  --coarse-decay 1e-5 "${TFARGS[@]}" --seed "$SEED" --out "$OUT"
