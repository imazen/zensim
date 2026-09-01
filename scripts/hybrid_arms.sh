#!/usr/bin/env bash
# hybrid_arms.sh — the HYBRID lane driver (benchmarks/hybrid_candidate_2026-09-01.md).
#
# Runs the 33 frozen arms of §4 through the ONE owner (`bake_verdict`) on the
# ONE substrate (the keyed pools-944 root), then hands the per-pair dumps to
# the exam's own paired reference-clustered bootstrap. It computes NO statistic
# and emits NO bake: every number comes from `bake_verdict` (rank + dial) or
# from `benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py` (which is itself
# `panel --batch` = zenstats). Membership and weights are FROZEN here so the
# board's blend weight can never disagree with what scored.
#
#   $0 score     bake_verdict --full-json + per-pair dumps, every arm
#   $0 peerdial  the opponent's DIAL on the SAME (pools) grid the arms use
#   $0 boot      the paired bootstraps (cid22 pooled/within, band, csiq, aic3)
#
# Env: ZL_BV (bake_verdict), HY_OUT (artifact dir).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BV=${ZL_BV:-$REPO/target/release/bake_verdict}
OUT=${HY_OUT:-/mnt/v/output/zensim/hybrid-2026-09-01}
POOLS=/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30
PGRID=/mnt/v/output/zensim/wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet
S2BAR=$REPO/benchmarks/ssim2_bar_2026-08-31

# The four parent bakes, by their exact bytes (§4).
M=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin
M2=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9P_s4005_packed.bin
L=/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.2_a0.2_b0.97.bin
LP=/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.25_a0.2_b0.97.bin

WEIGHTS=(0 10 20 30 40 50 60 70 80 90 100)
# H-A1 (declared before it ran): step-0.02 refinement between the two
# FAILING neighbours of the single W1-passing coarse cell, to measure the
# window's WIDTH. HY-A/HY-B only; HY-C's kon is monotone in w.
REFINE=(72 74 76 78 82 84 86 88)
mkdir -p "$OUT"

verdict() {  # verdict <name> <members-csv> <weights-csv>
    local name=$1 members=$2 w=$3
    nice -n19 ionice -c3 "$BV" \
        --ensemble "$members" --ensemble-weights "$w" \
        --regime 944 --features-root "$POOLS" --dial-grid "$PGRID" \
        --name "$name" \
        --full-json "$OUT/$name.fulleval.json" \
        --output "$OUT/$name.verdict.md" >"$OUT/$name.log" 2>&1
    # Per-pair dumps for the three corpora the exam can PAIR against ssim2.
    for c in cid22 csiq aic3; do
        nice -n19 ionice -c3 "$BV" \
            --ensemble "$members" --ensemble-weights "$w" \
            --regime 944 --features-root "$POOLS" --corpora "$c" \
            --per-pair-output "$OUT/pp_${name}_${c}.tsv" --per-pair-refs \
            --output /dev/null >>"$OUT/$name.log" 2>&1
    done
    printf '%s\n' "$name"
}

case "${1:-score}" in
score)
    for w in "${WEIGHTS[@]}"; do
        a=$(python3 -c "print(f'{$w/100:.4f}')"); b=$(python3 -c "print(f'{1-$w/100:.4f}')")
        verdict "HYA_w$(printf %03d "$w")" "$M,$L"  "$a,$b"
        verdict "HYB_w$(printf %03d "$w")" "$M,$LP" "$a,$b"
        # HY-C splits the flagship half between the two 944 seeds.
        h=$(python3 -c "print(f'{$w/200:.4f}')")
        verdict "HYC_w$(printf %03d "$w")" "$M,$M2,$L" "$h,$h,$b"
    done ;;
refine)
    for w in "${REFINE[@]}"; do
        a=$(python3 -c "print(f'{$w/100:.4f}')"); b=$(python3 -c "print(f'{1-$w/100:.4f}')")
        verdict "HYA_w$(printf %03d "$w")" "$M,$L"  "$a,$b"
        verdict "HYB_w$(printf %03d "$w")" "$M,$LP" "$a,$b"
    done ;;
peerdial)
    # The opponent's ladder, on the arms' OWN grid. `--dial-peer-scores`
    # refuses --full-json/--fulleval, so this is a markdown-only read.
    nice -n19 ionice -c3 "$BV" --bake "$M" --regime 944 \
        --features-root "$POOLS" --dial-grid "$PGRID" --corpora cid22 \
        --dial-peer-scores "ssim2=/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_944grid.tsv" \
        --output "$OUT/dial_peer_ssim2_poolsgrid.md" 2>&1 | tail -3 ;;
boot)
    shift
    export ZEN_PANEL_BIN=$REPO/target/release/panel O=$OUT
    export ARMS="${ARMS:?set ARMS to the space-separated arm names}"
    for spec in "$@"; do
        c=${spec%%:*}; band=${spec#*:}
        if [ "$band" = "$spec" ]; then
            CORPUS=$c python3 "$S2BAR/paired_perref_boot.py" > "$OUT/boot_${c}.txt"
            echo "wrote $OUT/boot_${c}.txt"
        else
            CORPUS=$c BAND_LO=$band python3 "$S2BAR/paired_perref_boot.py" > "$OUT/boot_${c}_band${band}.txt"
            echo "wrote $OUT/boot_${c}_band${band}.txt"
        fi
    done ;;
*) echo "usage: $0 {score|peerdial|boot <corpus[:bandlo>]...}" >&2; exit 2 ;;
esac
