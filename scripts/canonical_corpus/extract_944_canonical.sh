#!/bin/bash
# ext944-canonical extraction (PLAN_SOTA944 P1, task #9) — the 11 canonical
# local legs at ZENSIM_AB_MODE=foldapp2 (944 = 924 ++ append2-20), over the
# SAME pairs TSVs as the W1 ext924 run
# (/mnt/v/output/zensim/ext924-run-2026-07-27/run.log). Only the mode differs
# from that run; extends scripts/external_reads/asrun/bandvis_loo_944/
# extract_944.sh (the instrument-run record) with env-configurable BIN/OUT
# (no hardcoded worktree/scratch paths, per the lint-scripts rule).
#
# Gate after: gate_backfill944.py per leg vs ext924-canonical-2026-07-27,
# then promote_ext944_canonical.py.
#
# Env:
#   ZM944_BIN  v2_ab_extract binary (build: cargo build --release
#              --example v2_ab_extract -p zensim
#              --features feature-regime-v2,threads,training)
#   ZM944_OUT  output dir for the per-leg CSVs
#   ZM944_MODE ZENSIM_AB_MODE for every leg. Default "foldapp2" = the
#              canonical zero-block 944 regime. "foldapp2pools" =
#              R1b's all-live regime (f156..371 = v1's pool blocks LIVE,
#              V1PoolsMode::Full, regime tag folded720append2pools);
#              "foldapp2carriers" = the ten carrier slots only. Each is
#              its OWN regime -- never column-mix their rows.
#   ZM944_LEGS space-separated leg names to run (default: all 11). A leg
#              name is the ext_* label below, e.g. "ext_tid ext_kadid".
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${ZM944_BIN:-$REPO_ROOT/target/release/examples/v2_ab_extract}"
OUT="${ZM944_OUT:-/mnt/v/output/zensim/ext944-run-$(date -u +%F)}"
MODE="${ZM944_MODE:-foldapp2}"
LEGS="${ZM944_LEGS:-}"
[ -x "$BIN" ] || { echo "ABORT: extractor binary missing: $BIN"; exit 1; }
mkdir -p "$OUT"
ts() { date -u +%H:%M:%SZ; }
echo "== extract_944_canonical MODE=$MODE OUT=$OUT LEGS=${LEGS:-<all>}"

run_leg() {
  local name="$1" pairs="$2"
  if [ -n "$LEGS" ]; then
    case " $LEGS " in *" $name "*) ;; *) echo "== $name SKIPPED (not in ZM944_LEGS)"; return 0;; esac
  fi
  echo "== $name start $(ts)"
  ZENSIM_AB_MODE="$MODE" "$BIN" "$pairs" "$OUT/$name.csv"
  local rc=$?
  local rows=-1 cols=-1
  if [ -f "$OUT/$name.csv" ]; then
    rows=$(( $(wc -l < "$OUT/$name.csv") - 1 ))
    cols=$(head -1 "$OUT/$name.csv" | awk -F, '{print NF}')
  fi
  local want=$(( $(wc -l < "$pairs") - 1 ))
  echo "== $name done rc=$rc rows=$rows/$want cols=$cols $(ts)"
  if [ "$rc" -ne 0 ] || [ "$rows" -ne "$want" ] || [ "$cols" -ne 946 ]; then
    echo "ABORT: $name failed (rc=$rc rows=$rows want=$want cols=$cols want-cols=946)"
    exit 1
  fi
}

run_leg ext_sdr25            /mnt/v/output/zensim/v2-backfill-2026-07-20/sdr25_pairs.tsv
run_leg ext_aic4             /mnt/v/output/zensim/v2-backfill-2026-07-20/aic4_pairs.tsv
run_leg ext_konjnd_jpeg_val  /mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv
run_leg ext_aic3             /mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv
run_leg ext_live             /mnt/v/datasets/LIVE/live_r2_pairs.tsv
run_leg ext_csiq             /mnt/v/dataset/csiq/csiq_pairs.tsv
run_leg ext_tid              /mnt/v/dataset/tid2013/tid_pairs_ab.tsv
run_leg ext_cid22val         /mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv
run_leg ext_kadid            /mnt/v/dataset/kadid10k/kadid_pairs_ab.tsv
run_leg ext_cid22_train201   /mnt/v/output/zensim/v2-backfill-2026-07-20/cid22_train201_pairs.tsv
run_leg ext_safesyn_full     /mnt/v/output/zensim/v2-ab-2026-07-19/safesyn_jpeg_FULL_pairs_ab.tsv
echo "EXT944-ALL-LEGS-DONE $(ts)"
