#!/usr/bin/env bash
# wave5_ensemble.sh — the SOTA-944 WAVE 5 (seed-ensemble) driver.
#
# Runs each pre-registered arm (amendment 5 §5.4) through the ONE frozen §0
# verdict invocation (scripts/sota944_verdict.sh) with `--ensemble`, so an
# ensemble cell is evaluated by exactly the same program, corpora, grids and
# stats as every single-bake cell in this campaign.
#
# Membership is FROZEN in the registration; this script only names it.
# E3's 51 members are read from benchmarks/wave5_e3_members.txt (committed
# with the registration) so the list cannot drift.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
B=/mnt/v/output/zensim/bakes/sota944/bakes
export ZL_BV=${ZL_BV:-$HOME/tmp/zensimw5-target/release/bake_verdict}

# Published-CID22 order, frozen at registration (§5.4).
TOPK=(C_co3a_s1301 C_co2a_s1307 C_co3a_s1319 C_co1b_s1303 C_em944_s31 \
      C_co1a_s1303 C_co3a_s1409 C_co3a_s1307)
# Diverse-5: one per config family, each family's CID22-best (§5.4).
DIVERSE=(C_co3a_s1301 C_co2a_s1307 C_co1b_s1303 C_em944_s31 C_nt944lo_s211)

join_paths() { local out=""; for m in "$@"; do out+="${out:+,}$B/$m.bin"; done; printf '%s' "$out"; }

run_arm() {   # run_arm <stem> <member...>
    local stem=$1; shift
    local list; list=$(join_paths "$@")
    echo "=== $stem  (k=$#) ==="
    "$REPO_ROOT/scripts/sota944_verdict.sh" "$B/$1.bin" "$stem" --ensemble "$list"
}

case "${1:-all}" in
  e1) for k in 2 3 5 8; do run_arm "W5_E1_k$k" "${TOPK[@]:0:$k}"; done ;;
  e2) run_arm W5_E2_diverse5 "${DIVERSE[@]}" ;;
  e3) mapfile -t E3 < <(grep -v '^#' "$REPO_ROOT/benchmarks/wave5_e3_members.txt" | cut -f1)
      echo "E3 members: ${#E3[@]}"
      run_arm W5_E3_all51 "${E3[@]}" ;;
  all) "$0" e1; "$0" e2; "$0" e3 ;;
  *) echo "usage: $0 [e1|e2|e3|all]" >&2; exit 2 ;;
esac
