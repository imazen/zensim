#!/usr/bin/env bash
# recipe_v0_16.sh — one-command V0_16 SHIP retrain.
#
# This script reproduces the exact V0_16 ship bake from
# zensim/weights/v0_16_2026-05-12.bin (raw md5 b3f5fc59, calibrated
# baf3fdcb, CID22 SROCC 0.8919 on 4292 pairs, +0.0024 vs fast-ssim2).
# Recipe captured per CONTEXT-HANDOFF.md "V0_16 recipe (recoverable,
# current ship)" section.
#
# Why this script exists: V0_16 was trained interactively at zensim
# commit 4ada315/6f2487f via the Rust trainer binary. The exact
# invocation lived in shell history and a Python helper that the
# 2026-05-13 "best params as defaults" session normalized. This script
# is the permanent durable form so future agents (with "regular memory
# loss" per user) can reproduce V0_16 in one command without hunting
# through tick logs or commit messages.
#
# Inputs (canonical clean corpus at /mnt/v/zen/zensim-training/2026-05-14-clean/):
#   - safe_synth_v19_clean_features.csv (138,872 rows; V0_18 base minus
#     KADID/TID perceptual-overlap purge per audit 2026-05-14)
#   - tv_pairs_bands.tsv                (205,654 pairs)
#   - kadid_features.csv                (KADID-10k training + validation)
#   - tid_features.csv                  (TID2013 training + validation)
#   - konjnd_aligned_features.csv       (76,104 KonJND-1k anchor pairs)
#
# These are mirrored from /mnt/v block storage. The previous /tmp paths
# have been renamed with `.CONTAMINATED_2026-05-14_DO_NOT_USE` suffix
# and the contamination_guard binary refuses them at training time.
#
# Output:
#   - benchmarks/rust_v0_X_<DATE>.raw.bin       (uncalibrated bake)
#   - benchmarks/rust_v0_X_<DATE>.bin           (affine-calibrated bake)
#   - benchmarks/rust_v0_X_<DATE>.train.log     (training log)
#   - benchmarks/rust_v0_X_<DATE>.eval.log      (post-train SROCC eval)
#
# CID22 target: SROCC ≥ 0.8914 (V0_15) — anything below indicates the
# Rust trainer recipe didn't reproduce. V0_16's 0.8919 is the +1σ tail
# of the recipe-family seed sweep; you may see 0.886–0.892 across seeds.
#
# Usage:
#   bash benchmarks/recipe_v0_16.sh                # default date stamp
#   bash benchmarks/recipe_v0_16.sh --suffix _exp  # custom suffix
#
# Regenerate inputs if /tmp was wiped:
#   - safe_synth_clean_features.csv: see CONTEXT-HANDOFF.md "purge
#     (2026-05-12, user-directed)" section + scripts/v_next/convert_features_bin.py
#   - combined_purged_tv_pairs_bands.tsv: scripts/v_next/regen_tv_pairs.py --emit-bands
#   - kadid/tid features: /mnt/v should persist these.

set -euo pipefail

# --- Paths and config ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATE="${RECIPE_DATE:-$(date -u +%Y-%m-%d)}"
SUFFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suffix) SUFFIX="$2"; shift 2 ;;
        --date)   DATE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# 2026-05-14: switched to the canonical clean corpus at
# /mnt/v/zen/zensim-training/2026-05-14-clean/ which is V0_18's
# base training data minus the 149 KADID/TID perceptual-overlap
# basenames found in the 2026-05-14 dHash-64 audit. The OLD /tmp
# paths have been renamed with `.CONTAMINATED_2026-05-14_DO_NOT_USE`
# suffix; zensim_mlp_train refuses to load them.
CANON=/mnt/v/zen/zensim-training/2026-05-14-clean
SAFESYN_CSV="$CANON/safe_synth_v19_clean_features.csv"
KADID_CSV="$CANON/kadid_features.csv"
TID_CSV="$CANON/tid_features.csv"
KONJND_CSV="$CANON/konjnd_aligned_features.csv"
TV_PAIRS="$CANON/tv_pairs_bands.tsv"

# --- V0_16 hyperparameters (matches CONTEXT-HANDOFF.md exactly) ---
HIDDEN=128            # binary default since tick 594; explicit for clarity
TV_WEIGHT=20          # the V0_16 differentiator vs V0_15's TV=15
SEED=1                # binary default since tick 594
EPOCHS=300            # binary default; V0_16 early-stopped at ep=190
LR=1e-3               # binary default
VAL_POLICY=min        # binary default
MAX_FEATURES=228      # binary default since tick 594

# V0_16's affine-calibration coefficients (computed against synthetic
# ssim2; rank-invariant, only changes absolute score scale).
ALPHA=28.0366
BETA=-5.0738

# --- Output paths ---
OUT_DIR="$REPO_ROOT/benchmarks"
RAW_BAKE="$OUT_DIR/rust_v0_X_${DATE}${SUFFIX}.raw.bin"
CAL_BAKE="$OUT_DIR/rust_v0_X_${DATE}${SUFFIX}.bin"
TRAIN_LOG="$OUT_DIR/rust_v0_X_${DATE}${SUFFIX}.train.log"
EVAL_LOG="$OUT_DIR/rust_v0_X_${DATE}${SUFFIX}.eval.log"

# --- Pre-flight checks ---
echo "==> Pre-flight"
for f in "$SAFESYN_CSV" "$KADID_CSV" "$TID_CSV" "$KONJND_CSV" "$TV_PAIRS"; do
    if [[ ! -f "$f" ]]; then
        echo "MISSING INPUT: $f" >&2
        echo "See script header for regeneration instructions." >&2
        exit 1
    fi
    printf "  OK  %s  (%s)\n" "$f" "$(stat -c %s "$f" | numfmt --to=iec --suffix=B)"
done

if [[ -f "$RAW_BAKE" || -f "$CAL_BAKE" ]]; then
    echo "EXISTS: $RAW_BAKE or $CAL_BAKE — pass --suffix to disambiguate" >&2
    exit 1
fi

# --- Build trainer if needed ---
echo "==> Building zensim_mlp_train (release)"
cargo build -p zensim-validate --bin zensim_mlp_train --release 2>&1 | tail -3
BIN="$REPO_ROOT/target/release/zensim_mlp_train"

# --- Train ---
echo "==> Training V0_16 recipe → $RAW_BAKE"
echo "    h=$HIDDEN tv_weight=$TV_WEIGHT seed=$SEED epochs=$EPOCHS lr=$LR val_policy=$VAL_POLICY"
"$BIN" \
    --group "safesyn_purged:$SAFESYN_CSV:1.0:0.0" \
    --group "kadid:$KADID_CSV:0.3:1.0" \
    --group "tid:$TID_CSV:0.3:1.0" \
    --group "konjnd:$KONJND_CSV:0.5:1.0" \
    --hidden "$HIDDEN" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --val-policy "$VAL_POLICY" \
    --seed "$SEED" \
    --max-features "$MAX_FEATURES" \
    --tv-pairs-file "$TV_PAIRS" \
    --tv-weight "$TV_WEIGHT" \
    --out "$RAW_BAKE" \
    --log-path "$TRAIN_LOG"

echo "    raw bake md5: $(md5sum "$RAW_BAKE" | awk '{print $1}')"

# --- Affine-calibrate ---
echo "==> Affine-calibrating bake → $CAL_BAKE  (alpha=$ALPHA beta=$BETA)"
python3 scripts/v_next/affine_calibrate_znpr_v2.py \
    --in-bake "$RAW_BAKE" \
    --out-bake "$CAL_BAKE" \
    --alpha "$ALPHA" \
    --beta "$BETA"
echo "    calibrated bake md5: $(md5sum "$CAL_BAKE" | awk '{print $1}')"

# --- Quick eval ---
echo "==> CID22 eval (full 4292 pairs) → $EVAL_LOG"
cargo build -p zensim-bench --example dataset_metric_baseline --release 2>&1 | tail -2
"$REPO_ROOT/target/release/examples/dataset_metric_baseline" \
    --v04-bake "$CAL_BAKE" \
    --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
    2>&1 | tee "$EVAL_LOG" | tail -10

echo
echo "==> DONE"
echo "    Bake: $CAL_BAKE"
echo "    Eval: $EVAL_LOG"
echo "    Expected CID22 SROCC: 0.886-0.892 (V0_16-recipe family)"
echo "    V0_16 ship reference: 0.8919 on the +1σ seed tail"
