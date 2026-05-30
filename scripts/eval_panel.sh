#!/usr/bin/env bash
# Full zensim eval panel — MANDATORY for every ship-grade bake comparison.
#
# Runs BOTH halves of the panel against STORED feature sets (no re-encoding):
#   1. RANK panel   — bake_verdict: full Mohammadi 2025 stats (SROCC/PLCC/KROCC/
#                     OR/PWRC/Z-RMSE) per corpus on the 6 canonical val parquets.
#   2. DIAL panel   — qsweep_eval on the densified multi-codec q-sweep grid:
#                     monotonicity + tied-rate + per-q dial span across codec
#                     configs (G1 dynamic range, G3 monotonicity, G4 reach).
#
# The dial grid is the densified feature set (q0 + step-1 near-lossless +
# JND-zone + jxl-in-butter), stored on R2 and downloaded on demand so any model
# rescores against the IDENTICAL stored features. See
# `docs/EVAL_PANEL_REQUIREMENT.md`.
#
# Usage:
#   scripts/eval_panel.sh <bake.bin> [label] [post_mode]
#   post_mode: clamp (default, raw IS score) | mapped (distance bakes) | raw
#
# Env:
#   DIAL_GRID   override local dial-grid parquet path
#   SKIP_RANK=1 / SKIP_DIAL=1 to run only one half

set -euo pipefail
BAKE="${1:?usage: eval_panel.sh <bake.bin> [label] [post_mode]}"
LABEL="${2:-$(basename "$BAKE" .bin)}"
POST="${3:-clamp}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/eval_panel_$LABEL}"
mkdir -p "$OUT_DIR"

DIAL_GRID="${DIAL_GRID:-/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29.parquet}"
R2_EP="https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com"
R2_DIAL="s3://zentrain/eval-grids/dial_grid_372col_2026-05-29.parquet"

BAKE_VERDICT="$REPO/target/release/bake_verdict"
QSWEEP="$REPO/target/release/qsweep_eval"

echo "== zensim eval panel: $LABEL ($BAKE) ==" >&2

# ---- 1. RANK panel ----
if [[ "${SKIP_RANK:-0}" != "1" ]]; then
  echo "[rank] bake_verdict (6 canonical val corpora, full Mohammadi panel)..." >&2
  "$BAKE_VERDICT" --bake "$BAKE" --output "$OUT_DIR/rank_panel.md"
  echo "  -> $OUT_DIR/rank_panel.md" >&2
fi

# ---- 2. DIAL panel ----
if [[ "${SKIP_DIAL:-0}" != "1" ]]; then
  # Download the dial grid from R2 on demand if not present locally.
  if [[ ! -f "$DIAL_GRID" ]]; then
    echo "[dial] dial grid not local; downloading from R2..." >&2
    mkdir -p "$(dirname "$DIAL_GRID")"
    if [[ -f "$HOME/.config/cloudflare/r2-credentials" ]]; then
      # shellcheck disable=SC1090
      source "$HOME/.config/cloudflare/r2-credentials"
      AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
        aws s3 cp "$R2_DIAL" "$DIAL_GRID" --endpoint-url "$R2_EP"
    else
      echo "  R2 credentials missing at ~/.config/cloudflare/r2-credentials — cannot fetch dial grid" >&2
      exit 2
    fi
  fi
  echo "[dial] qsweep_eval on densified multi-codec grid (dial range + reach)..." >&2
  python3 "$REPO/scripts/dial_grid_to_qsweep.py" "$DIAL_GRID" \
    "$OUT_DIR/dial_features.csv" "$OUT_DIR/dial_manifest.tsv"
  "$QSWEEP" --features "$OUT_DIR/dial_features.csv" --manifest "$OUT_DIR/dial_manifest.tsv" \
    --bake "${LABEL}=${BAKE}:${POST}" --out "$OUT_DIR/dial_panel.md"
  echo "  -> $OUT_DIR/dial_panel.md" >&2
fi

echo "== panel complete: $OUT_DIR/{rank_panel,dial_panel}.md ==" >&2
# Surface the headline numbers
if [[ -f "$OUT_DIR/rank_panel.md" ]]; then
  echo "--- RANK (per-corpus SROCC) ---"
  grep -E "^\| (CID22|KADIK|TID|KonJND|AIC-3|AIC-4)" "$OUT_DIR/rank_panel.md" | awk -F'|' '{printf "  %-12s SROCC %s\n",$2,$4}'
fi
if [[ -f "$OUT_DIR/dial_panel.md" ]]; then
  echo "--- DIAL (monotonicity / tied) ---"
  grep -E "^\| $LABEL " "$OUT_DIR/dial_panel.md" | awk -F'|' '{printf "  monotonicity %s  tied %s\n",$7,$8}'
fi
