#!/usr/bin/env bash
#
# sota944_gridB.sh — the pre-registered SOTA-944 ARM-B runner (B-recipe replay
# at 944; benchmarks/sota944_campaign_2026-08-03.md §4). Two heads + raw-space
# z-normed convex blends:
#   kon head: BVLS (sign-mask bounds, tau 0.005), mm01 per-corpus targets,
#             safesyn 1.0 + cid22t201 1.5 + kadid 0.5 + tid 0.5 (canonhdr15
#             weights minus its hdr_v3mix leg — registered deviation).
#   cid head: lasso tau 0 on AM2 (legs + kadis 0.1), lambda in {1e-3, 2e-3, 3e-3}.
#   blends:   alpha in {0.7, 0.8, 0.9} x cid-head lambda, z-normed on the
#             registered anchor, collapsed to ONE identity-scaler layer,
#             f16 pack -> shared spline -> add-winsor.
# Cells: 9 blends + 2 standalone heads = 11 verdicts. Idempotent.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BDR=${ZL_BIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_dial_refit}
E=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
OUT=/mnt/v/output/zensim/bakes/sota944
G=$OUT/grams
B=$OUT/bakes
H=$OUT/heads
SIGNMASK=$REPO_ROOT/benchmarks/feature_sign_mask_2026-05-26.tsv
mkdir -p "$B" "$H"
NI=(nice -n 19 ionice -c 3)
[[ -x "$BDR" ]] || { echo "missing $BDR" >&2; exit 2; }

ANCHOR_ARGS=(--anchor-parquet "$E/ext_safesyn_full.parquet" --anchor-stride 139
             --anchor-parquet "$E/ext_cid22_train201.parquet" --anchor-stride 44
             --anchor-parquet "$E/ext_kadid.parquet" --anchor-stride 25
             --anchor-parquet "$E/ext_tid.parquet" --anchor-stride 7
             --anchor-target human_score --anchor-scale 100 --anchor-clip-min -100)

# ── kon head (BVLS, mm01 targets, canonhdr15-minus-hdr weights) ────────────
KON_RAW="$B/B_konhead.bin"
KON_NPZ="$H/konhead.npz"
if [[ ! -f "$KON_NPZ" ]]; then
    echo "== fit kon head =="
    "${NI[@]}" "$BDR" fit-lasso \
        --gram "$G/safesyn_mm01.npz" --weight 1.0 --gram-target human_score__mm01 \
        --gram "$G/cid22t201_mm01.npz" --weight 1.5 --gram-target human_score__mm01 \
        --gram "$G/kadid_mm01.npz" --weight 0.5 --gram-target human_score__mm01 \
        --gram "$G/tid_mm01.npz" --weight 0.5 --gram-target human_score__mm01 \
        --space raw --solver bvls --bounds-tsv "$SIGNMASK" --lam 0 --tau 0.005 \
        "${ANCHOR_ARGS[@]}" --emit-fit-npz "$KON_NPZ" --embed-repro --out "$KON_RAW"
fi

# ── cid heads (lasso on AM2) ───────────────────────────────────────────────
cid_head() {
    local lam=$1
    local npz="$H/cidhead_lam${lam}.npz" raw="$B/B_cidhead_lam${lam}.bin"
    [[ -f "$npz" ]] && return 0
    echo "== fit cid head lam $lam =="
    "${NI[@]}" "$BDR" fit-lasso \
        --gram "$G/safesyn_raw.npz" --weight 1.0 --gram-target human_score \
        --gram "$G/cid22t201_raw.npz" --weight 1.0 --gram-target human_score \
        --gram "$G/kadid_raw.npz" --weight 0.5 --gram-target human_score \
        --gram "$G/tid_raw.npz" --weight 0.5 --gram-target human_score \
        --gram "$G/kadis700k_raw.npz" --weight 0.1 --gram-target score_ssim2_gpu \
        --space raw --lam "$lam" --tau 0 \
        "${ANCHOR_ARGS[@]}" --emit-fit-npz "$npz" --embed-repro --out "$raw"
}

# ── blend + ship form + verdict ────────────────────────────────────────────
blend_cell() {
    local lam=$1 alpha=$2
    local stem="B_blend_lam${lam}_a${alpha}"
    local raw="$B/$stem.bin" ship="$B/${stem}_w.bin"
    if [[ ! -f "$ship" ]]; then
        echo "== blend $stem =="
        "${NI[@]}" "$BDR" blend-heads \
            --head "$H/cidhead_lam${lam}.npz" --head "$H/konhead.npz" \
            --alpha "$alpha" \
            "${ANCHOR_ARGS[@]}" --embed-repro --out "$raw"
        "${NI[@]}" "$BDR" add-winsor --in "$raw" --out "$ship" \
            --fit-corpus "$E/ext_safesyn_full.parquet"
    fi
    [[ -f "/mnt/v/output/zensim/bakes/sota944/verdicts/${stem}_w.full.json" ]] || \
        bash "$REPO_ROOT/scripts/sota944_verdict.sh" "$ship" "${stem}_w"
}

head_ship_verdict() {
    local raw=$1 stem=$2
    local ship="$B/${stem}_w.bin"
    [[ -f "$ship" ]] || "${NI[@]}" "$BDR" add-winsor --in "$raw" --out "$ship" \
        --fit-corpus "$E/ext_safesyn_full.parquet"
    [[ -f "/mnt/v/output/zensim/bakes/sota944/verdicts/${stem}_w.full.json" ]] || \
        bash "$REPO_ROOT/scripts/sota944_verdict.sh" "$ship" "${stem}_w"
}

for lam in 1e-3 2e-3 3e-3; do cid_head "$lam"; done
for lam in 1e-3 2e-3 3e-3; do
    for alpha in 0.7 0.8 0.9; do blend_cell "$lam" "$alpha"; done
done
head_ship_verdict "$KON_RAW" B_konhead
head_ship_verdict "$B/B_cidhead_lam2e-3.bin" B_cidhead_lam2e-3
echo "arm B complete"
