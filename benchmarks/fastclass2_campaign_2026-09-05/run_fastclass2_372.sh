#!/bin/bash
# fastclass2 — the SERVABLE (372-layout) lane runner. Train + score inline.
# Cells are SET:HIDDEN:HEAD, same grammar as the 944 runner; SET in
# {S156, S228, SFULL372}. Scored with bake_verdict --regime 372 on the current
# default 372 root, and with the 372 ladder instrument for A7r.
set -euo pipefail
REPO="${FC2_REPO:-/home/lilith/work/zen/zensim--fastclass2}"
cd "$REPO"
BIN="${FC2_BIN:-/mnt/v/zen/cargo-targets/fastclass2/release}"
export ZL_TRAIN="$BIN/zensim_mlp_train"
O="${FC2_OUT372:-/mnt/v/output/zensim/fastclass2-2026-09-05/serv372}"
mkdir -p "$O/bakes" "$O/gaddr"
TRAIN_SH="$REPO/benchmarks/fastclass2_campaign_2026-09-05/train_372_student.sh"
RH="$HOME/work/zen/scripts/run-heavy"
HB="${FC2_HB372:-$HOME/tmp/fc2/serv372/run}"
mkdir -p "$(dirname "$HB")"
S="$REPO/scripts/sota944"
R372=/mnt/v/zen/zensim-training/2026-08-30-full-features-372
# 372 dial + probes: the ladder instrument (A7r under --floor-rule resolvable)
LG=/mnt/v/output/zensim/ladder-2026-09-05/instruments/dial_grid_372col_ladder.parquet
LT=/mnt/v/output/zensim/ladder-2026-09-05/instruments/dialcells_ssim2_ladder.tsv
ID=/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments/identity_probe_372_postC_2026-09-05.parquet
NT=/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments/negtail_probe_372_postC_2026-09-05.parquet
ANCHOR=/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet

slice_of() {
  case "$1" in
    S156)     echo "$S/slice_basic156.txt" ;;
    S228)     echo "$S/slice_basic156_peaks.txt" ;;
    SFULL372) echo "full" ;;
    *) echo "ABORT: unknown set $1" >&2; exit 2 ;;
  esac
}
CELLS="${FC2_CELLS372:?FC2_CELLS372 required}"
SEEDS="${FC2_SEEDS:-4004 4005 4006}"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$HB.log"; }
say "START372 cells=[$CELLS] seeds=[$SEEDS]"
for CELL in $CELLS; do
  SET="${CELL%%:*}"; REST="${CELL#*:}"; W="${REST%%:*}"; HEAD="${REST#*:}"
  KEEP=$(slice_of "$SET")
  for SEED in $SEEDS; do
    NAME="S372_${SET}_H${W}_${HEAD}_s${SEED}"
    RAW="$O/bakes/${NAME}.bin"; PACKED="$O/bakes/${NAME}_packed.bin"
    if [ -f "$O/${NAME}.fulleval.json" ]; then say "SKIP (scored): $NAME"; continue; fi
    if [ ! -f "$RAW" ]; then
      say "TRAIN $NAME"
      (
        unset WR4_HIDDEN WR4_ALPHA_HEAD WR4_N_HIDDEN_LAYERS WR4_SKIP WR4_NO_COARSE_DECAY || true
        [ "$W" = 128 ] || export WR4_HIDDEN="$W"
        case "$HEAD" in
          p)  : ;;
          a1) export WR4_ALPHA_HEAD=1 ;;
          a2) export WR4_ALPHA_HEAD=1 WR4_N_HIDDEN_LAYERS=2 ;;
          sk) export WR4_SKIP=1 ;;
          nd) export WR4_NO_COARSE_DECAY=1 ;;
          *)  echo "ABORT: unknown head $HEAD"; exit 2 ;;
        esac
        "$RH" --mem 16G --jobs 8 -- "$TRAIN_SH" "$KEEP" "$SEED" "$RAW"
      ) >>"$HB.train.log" 2>&1 || { say "TRAIN FAILED $NAME"; touch "$O/${NAME}.TRAIN_FAILED"; continue; }
    fi
    say "PACK+SCORE $NAME"
    {
      [ -f "$PACKED" ] || "$BIN/bake_dial_refit" pack --in "$RAW" --out "$PACKED" --neg-tail \
        --anchor "$ANCHOR" --target-col target_score \
        --verify "$R372/cid22_features_372col_2026-05-15.parquet" --verify-col human_score
      "$BIN/bake_verdict" --bake "$PACKED" --regime 372 --features-root "$R372" \
        --name "$NAME" --full-json "$O/${NAME}.fulleval.json" --output "$O/${NAME}.verdict.md"
      "$BIN/bake_verdict" --bake "$PACKED" --regime 372 --features-root "$R372" \
        --dial-grid "$LG" --gaddr-grid-truth "$LT" --floor-rule resolvable \
        --negtail-probe "$NT" --identity-probe "$ID" --name "${NAME}@ladder" \
        --gaddr-json "$O/gaddr/gaddr_${NAME}_ladder.json" --output /dev/null
    } >>"$HB.score.log" 2>&1 || { say "SCORE FAILED $NAME"; touch "$O/${NAME}.HARVEST_FAILED"; continue; }
    say "DONE $NAME"
  done
done
say "ALL 372 CELLS COMPLETE"
touch "$O/PHASE372.done"
