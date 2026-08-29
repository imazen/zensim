#!/usr/bin/env bash
# W12-U battery stage: harvest fullevals for the M3a-ranked lodestar cells +
# aligned per-pair rescores (cid22@ROOT, hfnlproxy@VROOT, tid@ROOT) for the
# cells + references (north-anchor A, gray-tower incumbent).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
VROOT=/mnt/v/zen/zensim-training/valsel-2026-08-28/root
WD=$HOME/tmp/w12ubat; mkdir -p "$WD"
HB=$WD/heartbeat
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
declare -A CELLS=(
  [lstar4021_final]=$OUT/LSTAR_s4021_packed.bin
  [lstar4021_e080]=$OUT/LSTAR_s4021_ckpts/ckpt_epoch080_s4021_packed.bin
  [lstar4022_final]=$OUT/LSTAR_s4022_packed.bin
  [lstar4022_e070]=$OUT/LSTAR_s4022_ckpts/ckpt_epoch070_s4022_packed.bin
  [lstar4022_e080]=$OUT/LSTAR_s4022_ckpts/ckpt_epoch080_s4022_packed.bin
  [lstar4023_e070]=$OUT/LSTAR_s4023_ckpts/ckpt_epoch070_s4023_packed.bin
)
FAILS=0
for tag in "${!CELLS[@]}"; do
  b=${CELLS[$tag]}
  [ -f "$b" ] || { say "MISSING $tag $b"; FAILS=$((FAILS+1)); continue; }
  case "$tag" in *final) ;; *)
    if [ ! -f "/mnt/v/output/zensim/reports/fulleval/$tag.fulleval.json" ]; then
      say "harvest $tag"
      "$REPO/scripts/harvest_bakes.sh" --bake "$b" --stem "$tag" --regime 944 \
        >> "$WD/harvest.log" 2>&1 || { say "HARVEST FAIL $tag"; FAILS=$((FAILS+1)); }
    fi ;;
  esac
done
declare -A REFS=(
  [A]=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin
  [incumbent]=/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin
)
for tag in "${!CELLS[@]}" "${!REFS[@]}"; do
  b=${CELLS[$tag]:-${REFS[$tag]}}
  for spec in "cid22:$ROOT" "hfnlproxy:$VROOT" "tid:$ROOT"; do
    ax=${spec%%:*}; fr=${spec#*:}
    o="$WD/pp_${ax}_${tag}.tsv"
    [ -f "$o" ] && continue
    nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "$b" \
      --regime 944 --cross-regime --corpora $ax --features-root "$fr" \
      --per-pair-output "$o" > /dev/null 2>&1 || { say "RESCORE FAIL $tag $ax"; FAILS=$((FAILS+1)); }
  done
  say "rescored $tag"
done
say "BATTERY-STAGE DONE fails=$FAILS"
[ "$FAILS" = 0 ] || exit 6
