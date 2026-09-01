#!/usr/bin/env bash
# hybrid_distill.sh — PART II of the hybrid lane: distil the 944 teacher into
# the 156 compute set (benchmarks/hybrid_candidate_2026-09-01.md §10).
#
# Every step is an OWNER invocation; this file computes nothing and emits no
# bake of its own:
#   teacher  scripts/canonical_corpus/build_teacher944.py  (-> bake_dial_refit predict)
#   gram     bake_dial_refit gram --max-feat 372
#   fit      bake_dial_refit fit-lasso --slice-file scripts/sota944/slice_basic156.txt
#   eval     bake_verdict (the 372 root; basic-only bakes are era-independent — gated)
#
#   $0 gates | teacher | gram | fit | eval
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BDR=${ZL_BDR:-$REPO/target/release/bake_dial_refit}
BV=${ZL_BV:-$REPO/target/release/bake_verdict}
BBP=${ZL_BBP:-$REPO/target/release/bake_block_profile}
OUT=${HY_OUT:-/mnt/v/output/zensim/hybrid-2026-09-01}
D=$OUT/distill
POOLS=/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30
R372=/mnt/v/zen/zensim-training/2026-08-30-full-features-372
SLICE=$REPO/scripts/sota944/slice_basic156.txt
# The teacher = PART I's selected hybrid, by its exact members and weights.
M=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin
L=/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.2_a0.2_b0.97.bin
TW=0.84,0.16
SAFE=$POOLS/ext_safesyn_full.parquet
mkdir -p "$D"

case "${1:-gates}" in
gates)
    echo "== G-P1  predict endpoint identities (teacher forward == the evaluation's forward)"
    "$BDR" predict --ensemble "$M,$L" --ensemble-weights 1,0 --corpus "$POOLS/ext_cid22val.parquet" --out "$D/g_w10.tsv"
    "$BDR" predict --bake "$M"                                --corpus "$POOLS/ext_cid22val.parquet" --out "$D/g_solo.tsv"
    "$BDR" predict --ensemble "$M,$L" --ensemble-weights 0.5,0.5 --corpus "$POOLS/ext_cid22val.parquet" --out "$D/g_uni.tsv"
    "$BDR" predict --ensemble "$M,$L"                            --corpus "$POOLS/ext_cid22val.parquet" --out "$D/g_unw.tsv"
    cmp -s "$D/g_w10.tsv" "$D/g_solo.tsv" && echo "  G-P1a w=(1,0) == --bake member0: BIT-IDENTICAL" || { echo "  G-P1a FAIL"; exit 3; }
    cmp -s "$D/g_uni.tsv" "$D/g_unw.tsv" && echo "  G-P1b uniform == unweighted: BIT-IDENTICAL" || echo "  G-P1b differs (report the max delta)"
    echo "== G-E  era independence of a basic-only bake across THIS lane's two roots"
    for root in "$R372" "$POOLS"; do
        printf '  %-56s ' "$(basename "$root")"
        EXTRA=""; [ "$root" = "$POOLS" ] && EXTRA="--regime 944"
        # shellcheck disable=SC2086
        "$BV" --bake /mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin $EXTRA \
            --features-root "$root" --corpora cid22 --full-json "$D/gE_$(basename "$root").json" \
            --output /dev/null >/dev/null 2>&1
        JF="$D/gE_$(basename "$root").json" python3 -c 'import json,os;d=json.load(open(os.environ["JF"]));print("cid22 srocc_signed=%.10f" % d["rank"]["cid22"]["srocc_signed"])'
    done ;;
teacher)
    python3 "$REPO/scripts/canonical_corpus/build_teacher944.py" \
        --tag hyw084 --members "$M,$L" --weights "$TW" \
        --twin "safesyn=$SAFE" --affine-twin safesyn \
        --bdr "$BDR" --out-dir "$D/teach" ;;
gram)
    # The STUDENT is 372-wide (ADD156's shape). --max-feat 372 makes the pools
    # table's leading 372 columns the fit space; --slice-file then zeroes
    # f156..371 so the block profile reads uses_f156_371=false.
    # --target-clip-min -100 is the REGISTERED E-LIN policy (MSE magnitude
    # protection for catastrophic tails) and it is load-bearing here: the
    # safesyn human target reaches -739 at the x100 scale, so a handful of rows
    # otherwise dominate the least-squares. The teacher target is already
    # affine'd into [0,100], so the same flag is a NO-OP on that gram and the
    # two stay symmetric. The unclipped grams are kept as the declared control.
    "$BDR" gram --parquet "$SAFE" --target human_score --target-scale 100 \
        --space raw --max-feat 372 --out "$D/g_safe_human.npz"
    "$BDR" gram --parquet "$SAFE" --target human_score --target-scale 100 --target-clip-min -100 \
        --space raw --max-feat 372 --out "$D/g_safe_human_c100.npz"
    "$BDR" gram --parquet "$D/teach/safesyn_teacher944.parquet" --target human_score \
        --target-scale 100 --space raw --max-feat 372 --out "$D/g_safe_teacher.npz"
    "$BDR" gram --parquet "$D/teach/safesyn_teacher944.parquet" --target human_score \
        --target-scale 100 --target-clip-min -100 \
        --space raw --max-feat 372 --out "$D/g_safe_teacher_c100.npz" ;;
fit)
    for lam in 0.0003 0.001 0.003; do
        "$BDR" fit-lasso --gram "$D/g_safe_human_c100.npz" --space raw --target human_score \
            --lam "$lam" --slice-file "$SLICE" \
            --anchor-parquet "$SAFE" --anchor-target human_score --anchor-scale 100 --anchor-stride 37 --anchor-prefix \
            --embed-repro --out "$D/SADD_H_l$lam.bin"
        "$BDR" fit-lasso --gram "$D/g_safe_teacher_c100.npz" --space raw --target human_score \
            --lam "$lam" --slice-file "$SLICE" \
            --anchor-parquet "$SAFE" --anchor-target human_score --anchor-scale 100 --anchor-stride 37 --anchor-prefix \
            --embed-repro --out "$D/SADD_T_l$lam.bin"
        for w in 0.25 0.5 1.0 2.0; do
            "$BDR" fit-lasso --gram "$D/g_safe_human_c100.npz" --gram "$D/g_safe_teacher_c100.npz" \
                --weight 1.0 --weight "$w" --space raw --target human_score \
                --lam "$lam" --slice-file "$SLICE" \
                --anchor-parquet "$SAFE" --anchor-target human_score --anchor-scale 100 --anchor-stride 37 --anchor-prefix \
                --embed-repro --out "$D/SADD_HT_w${w}_l$lam.bin"
        done
    done ;;
eval)
    # PRIMARY = the POOLS root, so the students sit on the SAME substrate as
    # PART I's arms and both parents (one ruler for the whole lane). The 372
    # root is scored too, as the era cross-check and because ADD156's board
    # cell lives there. Dial: the pools grid for the pools read.
    PG=/mnt/v/output/zensim/wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet
    CORP=cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25,imazen26,nonphoto,hfnlproxy
    for b in "$D"/S*.bin; do
        n=$(basename "$b" .bin)
        "$BBP" --bake "$b" --json > "$D/$n.blockprofile.json" 2>/dev/null || true
        nice -n19 ionice -c3 "$BV" --bake "$b" --regime 944 --features-root "$POOLS" \
            --dial-grid "$PG" --corpora "$CORP" \
            --name "$n" --full-json "$D/$n.fulleval.json" --output "$D/$n.verdict.md" \
            >"$D/$n.log" 2>&1 || echo "  eval FAILED (pools) $n — see $D/$n.log"
        for c in cid22 csiq aic3 live aic4; do
            nice -n19 ionice -c3 "$BV" --bake "$b" --regime 944 --features-root "$POOLS" \
                --corpora "$c" --per-pair-output "$OUT/pp_${n}_$c.tsv" --per-pair-refs \
                --output /dev/null >>"$D/$n.log" 2>&1 || true
        done
        nice -n19 ionice -c3 "$BV" --bake "$b" --features-root "$R372" --corpora "$CORP" \
            --name "${n}@372" --full-json "$D/$n.era372.fulleval.json" --output /dev/null \
            >>"$D/$n.log" 2>&1 || echo "  eval FAILED (372) $n"
        echo "  $n"
    done ;;
*) echo "usage: $0 {gates|teacher|gram|fit|eval}" >&2; exit 2 ;;
esac
