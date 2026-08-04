#!/usr/bin/env bash
# wave6_konjnd_ensemble.sh — the SOTA-944 WAVE 6 arm G-E driver.
#
# KonJND-aware ensembles over wave 5's UNCHANGED 64-bake 944-MLP pool, each run
# through the ONE frozen §0 verdict invocation (scripts/sota944_verdict.sh) with
# `--ensemble`, so every cell is comparable to all 64 single-bake cells and to
# wave 5's six arms.
#
# Membership is FROZEN in SOTA-944 amendment 6 §6.2 (G-E); this script only
# names it. No list is derived at run time.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
B=/mnt/v/output/zensim/bakes/sota944/bakes
export ZL_BV=${ZL_BV:-$HOME/tmp/zensimw6-target/release/bake_verdict}

# --- frozen member lists (amendment 6 §6.2) --------------------------------
GE1=(C_co3a_s1301 C_co4_s1307)
GE2=(C_co3a_s1301 C_co3a_s1307 C_em944_s31)
GE3=(C_co3a_s1301 C_co3a_s1319 C_co3a_s1307 C_em944_s31 C_co4_s1307)
GE4=(C_em944_s31 C_co3a_s1307 C_co4_s1307 C_co4_s1301 C_co3a_s1327)
GE5=(C_co3a_s1301 C_co2a_s1307 C_co3a_s1319 C_co1b_s1303 C_em944_s31 \
     C_co3a_s1307 C_co4_s1307 C_em944_s127)

join_paths() { local out=""; for m in "$@"; do out+="${out:+,}$B/$m.bin"; done; printf '%s' "$out"; }

run_arm() {   # run_arm <stem> <member...>
    local stem=$1; shift
    local list; list=$(join_paths "$@")
    echo "=== $stem  (k=$#) ==="
    "$REPO_ROOT/scripts/sota944_verdict.sh" "$B/$1.bin" "$stem" --ensemble "$list"
}

case "${1:-all}" in
  ge1) run_arm W6_GE1_konpair    "${GE1[@]}" ;;
  ge2) run_arm W6_GE2_trio       "${GE2[@]}" ;;
  ge3) run_arm W6_GE3_balanced5  "${GE3[@]}" ;;
  ge4) run_arm W6_GE4_konfloor5  "${GE4[@]}" ;;
  ge5) run_arm W6_GE5_w5plus3    "${GE5[@]}" ;;
  all) for a in ge1 ge2 ge3 ge4 ge5; do "$0" "$a"; done ;;
  *) echo "usage: $0 [ge1|ge2|ge3|ge4|ge5|all]" >&2; exit 2 ;;
esac
