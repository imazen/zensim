#!/usr/bin/env bash
#
# linear924_grid.sh — the pre-registered E-LIN phase-1 grid runner
# (benchmarks/linear924_phase1_2026-08-01.md; commit the protocol BEFORE
# running this). 5 data arms x 7 lambdas -> fit-lasso bake -> add-winsor ->
# bake_verdict (924 invocation, --full-json) -> grid TSV.
#
# Idempotent: existing bakes/verdicts are skipped, so a crashed run resumes.
#
#   scripts/linear924_grid.sh            # run everything missing
#   scripts/linear924_grid.sh tsv        # just re-extract the grid TSV
#
# Env overrides: ZL_BIN (bake_dial_refit), ZL_BV (bake_verdict) — default to
# this repo's target/release (repo-relative, NEVER a worktree path).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BDR=${ZL_BIN:-$REPO_ROOT/target/release/bake_dial_refit}
BV=${ZL_BV:-$REPO_ROOT/target/release/bake_verdict}
E=/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27
V=/mnt/v/output/zensim/v2-eval-924-2026-07-27
OUT=/mnt/v/output/zensim/bakes/linear924
G=$OUT/grams
B=$OUT/bakes
VD=$OUT/verdicts
mkdir -p "$B" "$VD"
NI=(nice -n 19 ionice -c 3)

command -v jq >/dev/null || { echo "jq required" >&2; exit 2; }
[[ -x "$BDR" ]] || { echo "missing $BDR (cargo build --release -p zensim-validate --bin bake_dial_refit)" >&2; exit 2; }
[[ -x "$BV" ]] || { echo "missing $BV" >&2; exit 2; }

# Registered lambda grid (7 values spanning 3e-4 x10 both ways).
LAMS=(3e-5 1e-4 3e-4 5e-4 1e-3 2e-3 3e-3)

# The shared M1 gram block: safesyn 1.0 + cid22t201 1.0 + kadid 0.5 + tid 0.5
# (targets: every leg gram stores q under human_score).
M1_ARGS=(--gram "$G/safesyn_full.npz" --weight 1.0 --gram "$G/cid22t201.npz" --weight 1.0
         --gram "$G/kadid.npz" --weight 0.5 --gram "$G/tid.npz" --weight 0.5)
M1_TGT=(human_score human_score human_score human_score)

BC_VIEWS=(zenavif_lossy zenjpeg_lossy zenjxl_lossless zenjxl_lossy zenpng_lossless zenwebp_lossless zenwebp_lossy)

# arm_args <arm> -> sets ARM_ARGS + ARM_TGTS
arm_args() {
    local arm=$1
    ARM_ARGS=("${M1_ARGS[@]}")
    ARM_TGTS=("${M1_TGT[@]}")
    case $arm in
        M1) ;;
        M2) ARM_ARGS+=(--gram "$G/kadis700k.npz" --weight 0.1); ARM_TGTS+=(score_ssim2_gpu) ;;
        M3) ARM_ARGS+=(--gram "$G/negrich.npz" --weight 0.1); ARM_TGTS+=(score_ssim2_gpu) ;;
        M4) for ds in "${BC_VIEWS[@]}"; do ARM_ARGS+=(--gram "$G/bc_$ds.npz" --weight 0.05); ARM_TGTS+=(score_ssim2); done ;;
        M5) ARM_ARGS+=(--gram "$G/kadis700k.npz" --weight 0.1 --gram "$G/negrich.npz" --weight 0.1); ARM_TGTS+=(score_ssim2_gpu score_ssim2_gpu) ;;
        *) echo "unknown arm $arm" >&2; exit 2 ;;
    esac
}

# Registered anchor (same for every cell): stride-sampled M1 legs, y = human_score x100 clip -100.
ANCHOR_ARGS=(--anchor-parquet "$E/ext_safesyn_full.parquet" --anchor-stride 139
             --anchor-parquet "$E/ext_cid22_train201.parquet" --anchor-stride 44
             --anchor-parquet "$E/ext_kadid.parquet" --anchor-stride 25
             --anchor-parquet "$E/ext_tid.parquet" --anchor-stride 7
             --anchor-target human_score --anchor-scale 100 --anchor-clip-min -100)

run_cell() {
    local arm=$1 lam=$2
    local stem="${arm}_lam${lam}"
    local raw="$B/$stem.bin" ship="$B/${stem}_w.bin"
    local vjson="$VD/${stem}_w.full.json" vmd="$VD/${stem}_w.verdict.md"
    arm_args "$arm"
    local gt_args=()
    for t in "${ARM_TGTS[@]}"; do gt_args+=(--gram-target "$t"); done
    if [[ ! -f "$ship" ]]; then
        echo "== fit $stem =="
        "${NI[@]}" "$BDR" fit-lasso "${ARM_ARGS[@]}" "${gt_args[@]}" \
            --space raw --lam "$lam" --tau 0 \
            "${ANCHOR_ARGS[@]}" --out "$raw"
        echo "== winsor $stem =="
        "${NI[@]}" "$BDR" add-winsor --in "$raw" --out "$ship" \
            --fit-corpus "$E/ext_safesyn_full.parquet"
        # Provenance sidecar (bake_verdict reads .spec.json when no embedded
        # zentrain.repro): arm composition + gram shas + argv essentials.
        local commit; commit=$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
        for target_bake in "$raw" "$ship"; do
            {
                echo "{"
                echo "  \"campaign\": \"E-LIN linear924 phase1 (benchmarks/linear924_phase1_2026-08-01.md)\","
                echo "  \"arm\": \"$arm\", \"lam\": \"$lam\", \"space\": \"raw\", \"tau\": 0,"
                echo "  \"winsor\": $([[ $target_bake == "$ship" ]] && echo '"safesyn_full p0.1/p99.9"' || echo null),"
                echo "  \"anchor\": \"M1 legs stride 139/44/25/7, human_score x100 clip -100\","
                echo "  \"code_commit\": \"$commit\","
                echo "  \"grams\": ["
                local first=1
                local i=0
                while [[ $i -lt ${#ARM_ARGS[@]} ]]; do
                    if [[ "${ARM_ARGS[$i]}" == "--gram" ]]; then
                        local gp="${ARM_ARGS[$((i+1))]}"
                        local gw="${ARM_ARGS[$((i+3))]}"
                        [[ $first -eq 0 ]] && echo ","
                        first=0
                        printf '    {"gram": "%s", "weight": %s, "sha256": "%s"}' \
                            "$gp" "$gw" "$(sha256sum "$gp" | cut -d' ' -f1)"
                    fi
                    i=$((i+1))
                done
                echo ""
                echo "  ]"
                echo "}"
            } > "$target_bake.spec.json"
        done
    fi
    if [[ ! -f "$vjson" ]]; then
        echo "== verdict $stem =="
        "${NI[@]}" "$BV" --bake "$ship" --regime 720 \
            --features-root "$E" \
            --dial-grid "$V/dial_grid_924col_2026-07-28.parquet" \
            --corruption-grid "$V/corruption_grid_924col_2026-07-27.parquet" \
            --corpora cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25 \
            --perpair-metrics /nonexistent-skip-kadis-block \
            --name "$stem" --full-json "$vjson" --output "$vmd" >/dev/null
    fi
}

emit_tsv() {
    local tsv=$OUT/linear924_grid.tsv
    {
        echo -e "arm\tlam\tcid22\tkonjnd\tsdr25\tcsiq\tlive\taic3\taic4\tkadid_guard\ttid_guard\tdial_mono\tdial_tied\tcorr_q20\tbake_sha256"
        for arm in M1 M2 M3 M4 M5; do
            for lam in "${LAMS[@]}"; do
                local j="$VD/${arm}_lam${lam}_w.full.json"
                [[ -f "$j" ]] || continue
                jq -r --arg arm "$arm" --arg lam "$lam" '
                    [$arm, $lam,
                     (.rank.cid22.srocc // "NA"), (.rank.konjnd.srocc // "NA"),
                     (.rank.sdr25.srocc // "NA"), (.rank.csiq.srocc // "NA"),
                     (.rank.live.srocc // "NA"), (.rank.aic3.srocc // "NA"),
                     (.rank.aic4.srocc // "NA"), (.rank.kadid.srocc // "NA"),
                     (.rank.tid.srocc // "NA"),
                     (.dial.mono_pct // "NA"), (.dial.tied_pct // "NA"),
                     (.corruption.pass_q20 // "NA"), (.bake_sha256 // "NA")]
                    | map(tostring) | join("\t")' "$j"
            done
        done
    } > "$tsv"
    echo "grid -> $tsv"
    column -t "$tsv" | sed -n '1,40p'
}

if [[ "${1:-run}" == "tsv" ]]; then
    emit_tsv
    exit 0
fi

for arm in M1 M2 M3 M4 M5; do
    for lam in "${LAMS[@]}"; do
        run_cell "$arm" "$lam"
    done
done
emit_tsv
