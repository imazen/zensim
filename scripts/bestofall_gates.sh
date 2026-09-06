#!/usr/bin/env bash
# BEST-OF-ALL Phase C gates. Reads the wave's artifacts; computes nothing that
# an owner already computes.
#
#   bestofall_gates.sh select        # freeze_check --select --seed-group --min-k 2
#   bestofall_gates.sh corruption <bake.bin> <label>
#   bestofall_gates.sh servable <bake.bin>     # read set + feature-set id
#   bestofall_gates.sh all <bake.bin> <label>
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ZL_OUT:-/mnt/v/output/zensim/best-of-all-2026-09-06}"
BIN="${ZL_BIN:-$OUT/bin}"
POSTC=/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC
CH=/mnt/v/output/zensim/corruption-head-2026-09-05
mkdir -p "$OUT/gates"

case "${1:?}" in
  select)
    # The registered k-seed selection rule. --seed-group is not optional here:
    # without it --select ranks INDIVIDUAL cells and picks the lucky draw.
    "$BIN/freeze_check" --select "$OUT"/verdicts/*.fulleval.json \
        --seed-group --min-k 2 --floor-basis all \
        2>&1 | tee "$OUT/gates/select.txt"
    ;;
  corruption)
    bake="${2:?}"; label="${3:?}"
    # The ZCTH tree head reads f0..f227 — exactly this candidate's slice — so it
    # is servable by the same V1PoolsMode::Peaks walk and attaching it cannot
    # widen the extraction. `pass_q20` is the informative column: all 672 gate
    # triples come from ONE reference, so q10 and q20 are two encodes repeated,
    # and `pass_q10` collapses to the dial-alone value whenever the q10 anchor's
    # dial is below zero.
    "$BIN/bake_verdict" --bake "$bake" --features-root "$POSTC" \
        --corruption-head "$CH/d228hgb/corruption_head_d228hgb.zcth" \
        --corruption-grid "$CH/corruption_grid_372col_postC_2026-09-05.parquet" \
        --name "${label}@corr" \
        --full-json "$OUT/gates/${label}.corrjoint.json" \
        --output "$OUT/gates/${label}.corrjoint.md" \
        2>&1 | tail -20 | tee "$OUT/gates/${label}.corrjoint.log"
    python3 - "$OUT/gates/${label}.corrjoint.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for key in ("corruption_head", "corruption", "corruption_deploy"):
    b = d.get(key)
    if isinstance(b, dict):
        print(f"{key}: " + ", ".join(f"{k}={v}" for k, v in sorted(b.items())
                                     if isinstance(v, (int, float, str))))
PY
    ;;
  servable)
    # The W4 argument is structural, not statistical: a 228-slice bake serves
    # through V1PoolsMode::Peaks, the mode ZensimProfile::D already resolves to,
    # so its WALK is the production D walk and the walk delta is ~0 by
    # construction. The MLP forward is below the speed instrument's noise floor
    # (MEASURED: with ZEN_S2_EXTRACT_ONLY the extract-only arms read SLOWER than
    # their full siblings). This check is what makes the structural claim
    # checkable: the read set must be exactly f0..f227.
    "$BIN/bake_block_profile" --bake "${2:?}" 2>&1 | tee -a "$OUT/gates/servable.txt"
    ;;
  all)
    "$0" servable "${2:?}"
    "$0" corruption "${2:?}" "${3:?}"
    "$0" select || true
    ;;
  *) echo "usage: $0 {select|corruption <bake> <label>|servable <bake>|all <bake> <label>}" >&2; exit 2 ;;
esac
