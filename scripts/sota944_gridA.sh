#!/usr/bin/env bash
#
# sota944_gridA.sh — the pre-registered SOTA-944 ARM-A (additive) grid runner
# (benchmarks/sota944_campaign_2026-08-03.md §3; protocol committed BEFORE
# this ran). Cells:
#   shaped lasso:  slices {P, X, Bplus} × mixes {AM1, AM2, AM5} × λ {1e-3, 3e-3, 1e-2}
#   raw control:   slice X × AM5 × the same λ
#   bvls variant:  raw space, mm01 per-corpus targets, slices {P, X} × {AM1, AM5}
# Ship form per cell: fit-lasso (--embed-repro) -> add-winsor (--compose for
# shaped) -> bake_verdict (944 invocation incl imazen26/nonphoto/hfnlproxy).
# Idempotent; sha-recorded; stage-2 λ densify is run by the same script with
# SOTA944_EXTRA_LAMS after the registered top-3-by-sdr25 rule fires.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BDR=${ZL_BIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_dial_refit}
BV=${ZL_BV:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_verdict}
E=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
V=/mnt/v/output/zensim/v2-eval-944-2026-08-01
OUT=/mnt/v/output/zensim/bakes/sota944
G=$OUT/grams
B=$OUT/bakes
VD=$OUT/verdicts
SL=$REPO_ROOT/scripts/sota944
SCR=$SL/screen944_monotone.tsv
SIGNMASK=$REPO_ROOT/benchmarks/feature_sign_mask_2026-05-26.tsv
mkdir -p "$B" "$VD"
NI=(nice -n 19 ionice -c 3)

command -v jq >/dev/null || { echo "jq required" >&2; exit 2; }
[[ -x "$BDR" && -x "$BV" ]] || { echo "missing binaries $BDR / $BV" >&2; exit 2; }

LAMS=(${SOTA944_EXTRA_LAMS:-1e-3 3e-3 1e-2})

# mix_args <mix> <space> -> ARM_ARGS (gram+weight+target triples)
mix_args() {
    local mix=$1 sp=$2 suf tgt_leg tgt_kadis
    case $sp in
        shaped) suf=shaped; tgt_leg=human_score; tgt_kadis=score_ssim2_gpu ;;
        raw)    suf=raw;    tgt_leg=human_score; tgt_kadis=score_ssim2_gpu ;;
        mm01)   suf=mm01;   tgt_leg=human_score__mm01; tgt_kadis=score_ssim2_gpu__mm01 ;;
        *) echo "bad space $sp" >&2; exit 2 ;;
    esac
    ARM_ARGS=(--gram "$G/safesyn_$suf.npz" --weight 1.0 --gram-target "$tgt_leg"
              --gram "$G/cid22t201_$suf.npz" --weight 1.0 --gram-target "$tgt_leg"
              --gram "$G/kadid_$suf.npz" --weight 0.5 --gram-target "$tgt_leg"
              --gram "$G/tid_$suf.npz" --weight 0.5 --gram-target "$tgt_leg")
    case $mix in
        AM1) ;;
        AM2) ARM_ARGS+=(--gram "$G/kadis700k_$suf.npz" --weight 0.1 --gram-target "$tgt_kadis") ;;
        AM5) ARM_ARGS+=(--gram "$G/kadis700k_$suf.npz" --weight 0.1 --gram-target "$tgt_kadis"
                        --gram "$G/negrich_$suf.npz" --weight 0.1 --gram-target "$tgt_kadis") ;;
        *) echo "bad mix $mix" >&2; exit 2 ;;
    esac
}

ANCHOR_ARGS=(--anchor-parquet "$E/ext_safesyn_full.parquet" --anchor-stride 139
             --anchor-parquet "$E/ext_cid22_train201.parquet" --anchor-stride 44
             --anchor-parquet "$E/ext_kadid.parquet" --anchor-stride 25
             --anchor-parquet "$E/ext_tid.parquet" --anchor-stride 7
             --anchor-target human_score --anchor-scale 100 --anchor-clip-min -100)

verdict() {
    local ship=$1 stem=$2
    local vjson="$VD/${stem}.full.json" vmd="$VD/${stem}.verdict.md"
    [[ -f "$vjson" ]] && return 0
    echo "== verdict $stem =="
    "${NI[@]}" "$BV" --bake "$ship" --regime 720 \
        --features-root "$E" \
        --dial-grid "$V/dial_grid_944col_2026-08-01.parquet" \
        --corruption-grid "$V/corruption_grid_944col_2026-08-01.parquet" \
        --corpora cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25,imazen26,nonphoto,hfnlproxy \
        --perpair-metrics /nonexistent-skip-kadis-block \
        --name "$stem" --full-json "$vjson" --output "$vmd" >/dev/null
}

spec_sidecar() {
    local bake=$1 desc=$2
    local commit; commit=$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
    jq -n --arg d "$desc" --arg c "$commit" \
        '{campaign: "SOTA-944 arm A (benchmarks/sota944_campaign_2026-08-03.md §3)",
          cell: $d, code_commit: $c,
          note: "full argv + gram shas embedded in the bake (zentrain.repro)"}' \
        > "$bake.spec.json"
}

run_lasso_cell() {
    local slice=$1 mix=$2 lam=$3 sp=$4
    local stem="A_${sp}_${slice}_${mix}_lam${lam}"
    local raw="$B/$stem.bin" ship="$B/${stem}_w.bin"
    if [[ ! -f "$ship" ]]; then
        echo "== fit $stem =="
        mix_args "$mix" "$sp"
        local sp_args=(--space raw) win_args=()
        if [[ "$sp" == "shaped" ]]; then
            sp_args=(--space shaped --transforms-tsv "$SCR")
            win_args=(--compose)
        fi
        "${NI[@]}" "$BDR" fit-lasso "${ARM_ARGS[@]}" "${sp_args[@]}" \
            --lam "$lam" --tau 0 --slice-file "$SL/slice_${slice}.txt" \
            "${ANCHOR_ARGS[@]}" --embed-repro --out "$raw"
        "${NI[@]}" "$BDR" add-winsor "${win_args[@]}" --in "$raw" --out "$ship" \
            --fit-corpus "$E/ext_safesyn_full.parquet"
        spec_sidecar "$ship" "$stem"
    fi
    verdict "$ship" "${stem}_w"
}

run_bvls_cell() {
    local slice=$1 mix=$2
    local stem="A_bvls_${slice}_${mix}"
    local raw="$B/$stem.bin" ship="$B/${stem}_w.bin"
    if [[ ! -f "$ship" ]]; then
        echo "== fit $stem =="
        mix_args "$mix" mm01
        "${NI[@]}" "$BDR" fit-lasso "${ARM_ARGS[@]}" --space raw \
            --solver bvls --bounds-tsv "$SIGNMASK" \
            --lam 0 --tau 0.005 --slice-file "$SL/slice_${slice}.txt" \
            "${ANCHOR_ARGS[@]}" --embed-repro --out "$raw"
        "${NI[@]}" "$BDR" add-winsor --in "$raw" --out "$ship" \
            --fit-corpus "$E/ext_safesyn_full.parquet"
        spec_sidecar "$ship" "$stem"
    fi
    verdict "$ship" "${stem}_w"
}

emit_tsv() {
    local tsv=$OUT/sota944_gridA.tsv
    {
        echo -e "cell\tcid22\tkonjnd\tsdr25\tnonphoto\timazen26\thfnl_perref\tcsiq\tlive\taic3\taic4\tkadid_g\ttid_g\tdial_mono\tdial_tied\tcorr_q20\tbake_sha256"
        for j in "$VD"/A_*.full.json; do
            [[ -f "$j" ]] || continue
            jq -r --arg cell "$(basename "$j" .full.json)" '
                [$cell,
                 (.rank.cid22.srocc // "NA"), (.rank.konjnd.srocc // "NA"),
                 (.rank.sdr25.srocc // "NA"), (.rank.nonphoto.srocc // "NA"),
                 (.rank.imazen26.srocc // "NA"), (.rank.hfnlproxy.per_ref_mean // "NA"),
                 (.rank.csiq.srocc // "NA"), (.rank.live.srocc // "NA"),
                 (.rank.aic3.srocc // "NA"), (.rank.aic4.srocc // "NA"),
                 (.rank.kadid.srocc // "NA"), (.rank.tid.srocc // "NA"),
                 (.dial.mono_pct // "NA"), (.dial.tied_pct // "NA"),
                 (.corruption.pass_q20 // "NA"), (.bake_sha256 // "NA")]
                | map(tostring) | join("\t")' "$j"
        done
    } > "$tsv"
    echo "grid -> $tsv"
}

if [[ "${1:-run}" == "tsv" ]]; then emit_tsv; exit 0; fi

for slice in P X Bplus; do
    for mix in AM1 AM2 AM5; do
        for lam in "${LAMS[@]}"; do
            run_lasso_cell "$slice" "$mix" "$lam" shaped
        done
    done
done
for lam in "${LAMS[@]}"; do
    run_lasso_cell X AM5 "$lam" raw
done
run_bvls_cell P AM1
run_bvls_cell X AM1
run_bvls_cell P AM5
run_bvls_cell X AM5
emit_tsv
