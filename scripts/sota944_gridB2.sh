#!/usr/bin/env bash
#
# sota944_gridB2.sh — the "B-gap resolution" rerun (SOTA-944 amendment,
# benchmarks/sota944_campaign_2026-08-03.md): arm B WITH the hdr_v3mix-944
# leg, canonhdr15-faithful mix (safesyn 1.0 + cid22t201 1.5 + kadid 0.5 +
# tid 0.5 + hdr_v3mix 15.0, per-corpus minmax01 anchors) through the
# blend-heads path; same λ/α grid as arm B; plus ONE additive+hdr cell
# (the arm-A sdr25-best additive config, slice X AM5 + the hdr leg).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BDR=${ZL_BIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_dial_refit}
E=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
HDRLEG=${SOTA944_HDRLEG:-/mnt/v/output/zensim/hdr944-leg/hdr_v3mix944_traindigits_2026-08-03.parquet}
OUT=/mnt/v/output/zensim/bakes/sota944
G=$OUT/grams
B=$OUT/bakes
H=$OUT/heads
SIGNMASK=$REPO_ROOT/benchmarks/feature_sign_mask_2026-05-26.tsv
SCREEN=$REPO_ROOT/scripts/sota944/screen944_monotone.tsv
mkdir -p "$B" "$H"
NI=(nice -n 19 ionice -c 3)
[[ -x "$BDR" && -f "$HDRLEG" ]] || { echo "missing $BDR or hdr leg $HDRLEG" >&2; exit 2; }

# ── hdr leg grams (raw + mm01 + shaped) ────────────────────────────────────
[[ -f $G/hdrv3mix944_raw.npz ]] || "${NI[@]}" "$BDR" gram --parquet "$HDRLEG" \
    --target human_score --target-scale 100 --target-clip-min -100 \
    --expect-n-feat 944 --space raw --out $G/hdrv3mix944_raw.npz
[[ -f $G/hdrv3mix944_mm01.npz ]] || "${NI[@]}" "$BDR" gram --parquet "$HDRLEG" \
    --target human_score --target-scale 100 --target-minmax01 \
    --expect-n-feat 944 --space raw --out $G/hdrv3mix944_mm01.npz
[[ -f $G/hdrv3mix944_shaped.npz ]] || "${NI[@]}" "$BDR" gram --parquet "$HDRLEG" \
    --target human_score --target-scale 100 --target-clip-min -100 \
    --expect-n-feat 944 --space shaped --transforms-tsv "$SCREEN" \
    --out $G/hdrv3mix944_shaped.npz

ANCHOR_ARGS=(--anchor-parquet "$E/ext_safesyn_full.parquet" --anchor-stride 139
             --anchor-parquet "$E/ext_cid22_train201.parquet" --anchor-stride 44
             --anchor-parquet "$E/ext_kadid.parquet" --anchor-stride 25
             --anchor-parquet "$E/ext_tid.parquet" --anchor-stride 7
             --anchor-target human_score --anchor-scale 100 --anchor-clip-min -100)

# ── kon head, canonhdr15-FAITHFUL (hdr_v3mix at 15.0, minmax01) ────────────
KON=$H/konhead_hdr.npz
if [[ ! -f "$KON" ]]; then
    echo "== fit kon head (canonhdr15-faithful, +hdr 15.0) =="
    "${NI[@]}" "$BDR" fit-lasso \
        --gram "$G/safesyn_mm01.npz" --weight 1.0 --gram-target human_score__mm01 \
        --gram "$G/cid22t201_mm01.npz" --weight 1.5 --gram-target human_score__mm01 \
        --gram "$G/kadid_mm01.npz" --weight 0.5 --gram-target human_score__mm01 \
        --gram "$G/tid_mm01.npz" --weight 0.5 --gram-target human_score__mm01 \
        --gram "$G/hdrv3mix944_mm01.npz" --weight 15.0 --gram-target human_score__mm01 \
        --space raw --solver bvls --bounds-tsv "$SIGNMASK" --lam 0 --tau 0.005 \
        "${ANCHOR_ARGS[@]}" --emit-fit-npz "$KON" --embed-repro --out "$B/B2_konhead_hdr.bin"
fi

# ── cid heads WITH the hdr leg (B's cid side trained ON hdr_v3mix) ─────────
cid_head() {
    local lam=$1
    local npz="$H/cidhead_hdr_lam${lam}.npz"
    [[ -f "$npz" ]] && return 0
    echo "== fit cid head (+hdr) lam $lam =="
    "${NI[@]}" "$BDR" fit-lasso \
        --gram "$G/safesyn_raw.npz" --weight 1.0 --gram-target human_score \
        --gram "$G/cid22t201_raw.npz" --weight 1.0 --gram-target human_score \
        --gram "$G/kadid_raw.npz" --weight 0.5 --gram-target human_score \
        --gram "$G/tid_raw.npz" --weight 0.5 --gram-target human_score \
        --gram "$G/kadis700k_raw.npz" --weight 0.1 --gram-target score_ssim2_gpu \
        --gram "$G/hdrv3mix944_raw.npz" --weight 15.0 --gram-target human_score \
        --space raw --lam "$lam" --tau 0 \
        "${ANCHOR_ARGS[@]}" --emit-fit-npz "$npz" --embed-repro --out "$B/B2_cidhead_hdr_lam${lam}.bin"
}

blend_cell() {
    local lam=$1 alpha=$2
    local stem="B2_blend_hdr_lam${lam}_a${alpha}"
    local raw="$B/$stem.bin" ship="$B/${stem}_w.bin"
    if [[ ! -f "$ship" ]]; then
        echo "== blend $stem =="
        "${NI[@]}" "$BDR" blend-heads \
            --head "$H/cidhead_hdr_lam${lam}.npz" --head "$KON" --alpha "$alpha" \
            "${ANCHOR_ARGS[@]}" --embed-repro --out "$raw"
        "${NI[@]}" "$BDR" add-winsor --in "$raw" --out "$ship" \
            --fit-corpus "$E/ext_safesyn_full.parquet"
    fi
    [[ -f "$OUT/verdicts/${stem}_w.full.json" ]] || \
        bash "$REPO_ROOT/scripts/sota944_verdict.sh" "$ship" "${stem}_w"
}

head_ship_verdict() {
    local raw=$1 stem=$2
    local ship="$B/${stem}_w.bin"
    [[ -f "$ship" ]] || "${NI[@]}" "$BDR" add-winsor --in "$raw" --out "$ship" \
        --fit-corpus "$E/ext_safesyn_full.parquet"
    [[ -f "$OUT/verdicts/${stem}_w.full.json" ]] || \
        bash "$REPO_ROOT/scripts/sota944_verdict.sh" "$ship" "${stem}_w"
}

# ── the ONE additive+hdr cell: arm-A sdr25-best additive config (shaped X
# AM5 λ3e-3 was the shaped-P sdr25 zone; registered as X AM5 + hdr shaped) ──
additive_hdr_cell() {
    local stem="B2_addX_AM5hdr_lam3e-3"
    local raw="$B/$stem.bin" ship="$B/${stem}_w.bin"
    if [[ ! -f "$ship" ]]; then
        echo "== fit $stem =="
        "${NI[@]}" "$BDR" fit-lasso \
            --gram "$G/safesyn_shaped.npz" --weight 1.0 --gram-target human_score \
            --gram "$G/cid22t201_shaped.npz" --weight 1.0 --gram-target human_score \
            --gram "$G/kadid_shaped.npz" --weight 0.5 --gram-target human_score \
            --gram "$G/tid_shaped.npz" --weight 0.5 --gram-target human_score \
            --gram "$G/kadis700k_shaped.npz" --weight 0.1 --gram-target score_ssim2_gpu \
            --gram "$G/negrich_shaped.npz" --weight 0.1 --gram-target score_ssim2_gpu \
            --gram "$G/hdrv3mix944_shaped.npz" --weight 15.0 --gram-target human_score \
            --space shaped --transforms-tsv "$SCREEN" --lam 3e-3 --tau 0 \
            --slice-file "$REPO_ROOT/scripts/sota944/slice_X.txt" \
            "${ANCHOR_ARGS[@]}" --embed-repro --out "$raw"
        "${NI[@]}" "$BDR" add-winsor --compose --in "$raw" --out "$ship" \
            --fit-corpus "$E/ext_safesyn_full.parquet"
    fi
    [[ -f "$OUT/verdicts/${stem}_w.full.json" ]] || \
        bash "$REPO_ROOT/scripts/sota944_verdict.sh" "$ship" "${stem}_w"
}

for lam in 1e-3 2e-3 3e-3; do cid_head "$lam"; done
for lam in 1e-3 2e-3 3e-3; do
    for alpha in 0.7 0.8 0.9; do blend_cell "$lam" "$alpha"; done
done
head_ship_verdict "$B/B2_konhead_hdr.bin" B2_konhead_hdr
head_ship_verdict "$B/B2_cidhead_hdr_lam2e-3.bin" B2_cidhead_hdr_lam2e-3
additive_hdr_cell
echo "B-gap resolution grid complete"
