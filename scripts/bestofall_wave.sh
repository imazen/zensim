#!/usr/bin/env bash
# BEST-OF-ALL wave: the constrained 228-slot MLP, its control, and the ladder arm.
#
# Plan: docs/PLAN_BEST_OF_ALL_2026-09-06.md. Record:
# benchmarks/best_of_all_2026-09-06.md.
#
# Repo-relative by construction (never a sibling-worktree path): WS is derived
# from this script's own location, so it works from the main checkout and from
# any workspace, and survives the mandatory workspace cleanup.
#
#   bestofall_wave.sh cell <arm> <seed>   # one cell, end to end
#   bestofall_wave.sh all                 # the whole wave, serialized
#
# Arms:
#   A_plain      the fastclass2 winner's recipe, unchanged (the CONTROL)
#   B_nonneg     + --nonneg-distance
#   C_lad05      + --nonneg-distance + the ladder hinge at tv-weight 0.5
#   D_lad20      + --nonneg-distance + the ladder hinge at tv-weight 2.0
#   E_plainlad   the control + the ladder hinge at 0.5 (isolates the LOSS from
#                the ARCHITECTURE — without it a win cannot be attributed)
#   F_nonneg32   B at --hidden 32 (bytes/speed variant of the constrained arm)
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ZL_BIN:-$WS/target/release}"
OUT="${ZL_OUT:-/mnt/v/output/zensim/best-of-all-2026-09-06}"
INSTR="$OUT/instruments"
CANON=/mnt/v/zen/zensim-training/canonical-2026-05-21/train
POSTC=/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC
LADDER=/mnt/v/output/zensim/ladder-2026-09-05/instruments
PROBES=/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments
IDANCHOR=/mnt/v/output/zensim/did100-2026-09-04/work/identity_anchor_sg_n21.parquet
ANCHOR=$INSTR/negrich_plus_identity21_anchor.parquet
TVPAIRS=$INSTR/ladder_tv_pairs_safesyn.tsv
COMMON=/mnt/v/output/zensim/rev2-refit-2026-09-06/fastclass/common_args.txt
SLICE="$WS/scripts/sota944/slice_basic156_peaks.txt"

mkdir -p "$OUT"/{bakes,verdicts,gaddr,logs}
HB="$OUT/heartbeat.txt"
say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$HB"; }

# `cell` is invoked under `||` in the wave loop, which SUPPRESSES `set -e`
# inside it. Without an explicit check every step's failure would fall through
# and the cell would still print DONE — which it did, once, on 2026-09-06.
# `step` checks the exit status AND that the artifact it was supposed to produce
# exists, and returns non-zero so the caller's `||` fires.
step() {
  local what="$1" artifact="$2"; shift 2
  if ! "$@"; then
    say "STEP FAILED ($what) rc=$? — see $OUT/logs/"
    return 1
  fi
  if [[ -n "$artifact" && ! -s "$artifact" ]]; then
    say "STEP FAILED ($what) — produced no $artifact"
    return 1
  fi
  return 0
}

# The six canonical training legs, verbatim from the winner's embedded repro.
groups() {
  printf '%s\n' \
    --group "safesyn:$CANON/safesyn.parquet:1.0:0.5:both" \
    --group "cid22_train:$CANON/cid22_train_norm.parquet:1.0:2.0:both" \
    --group "kadid:$CANON/kadid.parquet:0.5:1.0:rank" \
    --group "tid:$CANON/tid.parquet:0.5:1.0:rank" \
    --group "bigcodec:/mnt/v/zen/zensim-training/tbig_372_200k.parquet:0.5:1.0:both" \
    --group "konjnd:$CANON/konjnd-dense-norm.parquet:1.2:0.0:both"
}

# Everything after the groups, from the frozen common-args file, with the dead
# sibling-worktree `--keep-features` path repointed at this checkout's copy
# (same file, same bytes — verified by sha in the record).
common() {
  local prev=""
  while IFS= read -r a || [[ -n "$a" ]]; do
    if [[ "$prev" == "--keep-features" ]]; then printf '%s\n' "$SLICE"; else printf '%s\n' "$a"; fi
    prev="$a"
  done < "$COMMON"
}

arm_extra() {
  case "$1" in
    A_plain)    : ;;
    B_nonneg)   printf '%s\n' --nonneg-distance ;;
    F_nonneg32) printf '%s\n' --nonneg-distance --hidden 32 ;;
    C_lad05)    printf '%s\n' --nonneg-distance --tv-pairs-file "$TVPAIRS" --tv-weight 0.5 \
                              --tv-margin 0.25 --tv-apply-every 50 --tv-batch 16 \
                              --tv-band-weights 1.5,0.5,0.5,0.5 ;;
    D_lad20)    printf '%s\n' --nonneg-distance --tv-pairs-file "$TVPAIRS" --tv-weight 2.0 \
                              --tv-margin 0.25 --tv-apply-every 50 --tv-batch 16 \
                              --tv-band-weights 6.0,2.0,2.0,2.0 ;;
    E_plainlad) printf '%s\n' --tv-pairs-file "$TVPAIRS" --tv-weight 0.5 \
                              --tv-margin 0.25 --tv-apply-every 50 --tv-batch 16 \
                              --tv-band-weights 1.5,0.5,0.5,0.5 ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

cell() {
  local arm="$1" seed="$2" name="${1}_s${2}"
  local raw="$OUT/bakes/${name}.bin"
  local packed="$OUT/bakes/${name}_packed.bin"
  local byid="$OUT/bakes/${name}_byid.bin"

  if [[ -f "$OUT/verdicts/${name}.fulleval.json" ]]; then say "SKIP $name (already harvested)"; return 0; fi

  say "TRAIN $name"
  local -a argv=()
  mapfile -t -O "${#argv[@]}" argv < <(groups)
  mapfile -t -O "${#argv[@]}" argv < <(common)
  mapfile -t -O "${#argv[@]}" argv < <(arm_extra "$arm")
  argv+=(--seed "$seed" --out "$raw")
  step train "$raw" "${ZL_RUNHEAVY:-$HOME/work/zen/scripts/run-heavy}" --mem 16G --jobs 8 -- \
     "$BIN/zensim_mlp_train" "${argv[@]}" > "$OUT/logs/${name}.train.log" 2>&1 || return 1

  # ONE pack step, against the negrich dial anchor CONCATENATED with the 21-row
  # identity anchor (built by instruments/negrich_plus_identity21_anchor.parquet's
  # manifest), so `fit_spline_knots` gets a knot at (raw(identity), 100) in the
  # same pass that quantizes and prunes — QUANTIZE-then-CALIBRATE, preserved.
  #
  # `shared-anchor` would be the more elegant place for the identity anchor (it
  # takes --anchor repeatably) but it asserts a SINGLE-LAYER linear bake, and
  # these are 228 -> H -> 1 MLPs. Merging the anchors up front is the same fit.
  #
  # 21 of 2,021 rows = 1.04 %, which owns fit_spline_knots' >= p99 top bin
  # exactly as the id100 lane sized it; n = 38 spills into the next bin and
  # displaces the top real knot.
  #
  # Under --nonneg-distance raw(identity) is the ARGMAX by construction, so that
  # knot is the top one and identity lands at exactly 100. The control gets the
  # identical chain — its raw(identity) is NOT the argmax, and what that costs
  # it is the measurement.
  # ONE pack step, against the negrich dial anchor CONCATENATED with the 21-row
  # identity anchor, so `fit_spline_knots` gets a knot at (raw(identity), 100)
  # in the same pass that quantizes and prunes — QUANTIZE-then-CALIBRATE kept.
  #
  # `shared-anchor` is the more elegant home for a second anchor (it takes
  # --anchor repeatably) but it asserts a SINGLE-LAYER linear bake and these are
  # 228 -> H -> 1 MLPs. Merging the anchors up front is the same fit.
  #
  # 21 of 2,021 rows = 1.04 %, which owns fit_spline_knots' >= p99 top bin
  # exactly as the id100 lane sized it; n = 38 spills and displaces the top real
  # knot. Under --nonneg-distance raw(identity) is the ARGMAX by construction,
  # so that knot is the top one. The control gets the identical chain — its
  # raw(identity) is NOT the argmax, and what that costs it is the measurement.
  say "PACK $name"
  step pack "$packed" "$BIN/bake_dial_refit" pack --in "$raw" --out "$packed" --neg-tail \
      --anchor "$ANCHOR" --target-col ssim2_gpu \
      > "$OUT/logs/${name}.pack.log" 2>&1 || return 1

  # densify is the ONLY thing that writes `zentrain.feature_ids`, and on a
  # contiguous-prefix read set (f0..f227) it collapses the caller width 372 ->
  # 228 and declares an IDENTITY layout.
  #
  # ⛔ THAT MAKES IT UNSCOREABLE BY THE PINNED PROBES. `bake_verdict` scores a
  # probe only when its column count equals the bake's `caller_input_width`, and
  # every registered negtail/identity probe is 372-wide — so a densified 228-wide
  # bake reads C3/C4/C5/C6 as NOT MEASURED, which is the entire contract tier and
  # the entire point of this lane. MEASURED on the first wave cell: contract
  # "INCOMPLETE (not a pass)" with all four rows unmeasured.
  #
  # So the SCORING path uses the packed bake, which keeps the 372 caller width
  # (144 `Drop` transforms) exactly like the campaign's own published 228 bakes,
  # and densify runs alongside as a SERVABILITY artifact. Its failure is
  # reported, not fatal — a bake that cannot densify still has a valid verdict.
  say "DENSIFY $name (servability artifact — NOT the scored bake)"
  if ! "$BIN/bake_dial_refit" densify --in "$packed" --out "$byid" --gate-rows 512 \
       > "$OUT/logs/${name}.densify.log" 2>&1; then
    say "DENSIFY FAILED $name (non-fatal; see logs)"
  fi

  say "VERDICT $name"
  step verdict "$OUT/verdicts/${name}.fulleval.json" \
      "$BIN/bake_verdict" --bake "$packed" --features-root "$POSTC" \
      --name "$name" --full-json "$OUT/verdicts/${name}.fulleval.json" \
      --output "$OUT/verdicts/${name}.verdict.md" \
      > "$OUT/logs/${name}.verdict.log" 2>&1 || return 1

  say "GADDR $name"
  step gaddr "$OUT/gaddr/gaddr_${name}.json" \
      "$BIN/bake_verdict" --bake "$packed" --features-root "$POSTC" \
      --dial-grid "$LADDER/dial_grid_372col_ladder.parquet" \
      --gaddr-grid-truth "$LADDER/dialcells_ssim2_ladder.tsv" \
      --floor-rule resolvable --floor-margin 0.5 \
      --reference-truth "$LADDER/reference_truth_ladder_pnorm3.tsv:pnorm3" \
      --negtail-probe "$PROBES/negtail_probe_372_postC_2026-09-05.parquet" \
      --identity-probe "$PROBES/identity_probe_372_postC_2026-09-05.parquet" \
      --name "${name}@ladder" --gaddr-json "$OUT/gaddr/gaddr_${name}.json" \
      --output /dev/null > "$OUT/logs/${name}.gaddr.log" 2>&1 || return 1
  say "DONE $name"
}

case "${1:-all}" in
  cell) cell "$2" "$3" ;;
  all)
    rm -f "$OUT/WAVE.done"
    say "WAVE START"
    for arm in A_plain B_nonneg C_lad05 D_lad20 E_plainlad F_nonneg32; do
      for seed in 4004 4005 4006; do
        cell "$arm" "$seed" || say "CELL FAILED $arm $seed (continuing)"
      done
    done
    say "WAVE COMPLETE"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/WAVE.done"
    ;;
  *) echo "usage: $0 {cell <arm> <seed>|all}" >&2; exit 2 ;;
esac
