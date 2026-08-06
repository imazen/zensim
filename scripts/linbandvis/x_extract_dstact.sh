#!/bin/bash
# x_extract_dstact.sh — APPENDIX X (Thread 2, X-E1): the BANDVIS-ON extraction.
#
# Re-extracts the 11 canonical local legs + the two konjnd_bpg legs with
# ZENSIM_APPEND2_DSTACT=1 (the P1.5 shipped GAIN-only combine) at
# ZENSIM_AB_MODE=foldapp2 — same pairs TSVs as extract_944_canonical.sh /
# build_konjnd_bpg_944.py, same extractor example, ONLY the toggle env added.
#
# Output CSVs -> $XDST_OUT ; promote + the X-G1 lanes-only gate are
# scripts/linbandvis/x_promote_dstact.py (run AFTER this).
#
# Env:
#   XDST_BIN  v2_ab_extract binary (build: cargo build --release
#             --example v2_ab_extract -p zensim
#             --features feature-regime-v2,threads,training)
#   XDST_OUT  output dir for the per-leg CSVs
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${XDST_BIN:-$REPO_ROOT/target/release/examples/v2_ab_extract}"
OUT="${XDST_OUT:-/mnt/v/output/zensim/ext944-dstact-run-$(date -u +%F)}"
[ -x "$BIN" ] || { echo "ABORT: extractor binary missing: $BIN"; exit 1; }
mkdir -p "$OUT"
ts() { date -u +%H:%M:%SZ; }

run_leg() {
  local name="$1" pairs="$2"
  if [ -f "$OUT/$name.csv" ]; then echo "== $name cached"; return; fi
  echo "== $name start $(ts)"
  ZENSIM_APPEND2_DSTACT=1 ZENSIM_AB_MODE=foldapp2 "$BIN" "$pairs" "$OUT/$name.csv"
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

# Small legs first (fast signal that the chain works), safesyn last.
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
run_leg konjnd_bpg_train_944 /mnt/v/output/zensim/wave7/konjnd_bpg_train_pairs.tsv
run_leg konjnd_bpg_val_944   /mnt/v/output/zensim/wave7/konjnd_bpg_val_pairs.tsv
run_leg ext_safesyn_full     /mnt/v/output/zensim/v2-ab-2026-07-19/safesyn_jpeg_FULL_pairs_ab.tsv
echo "XDST-ALL-LEGS-DONE $(ts)"
