#!/bin/bash
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/bandvis-loo-2026-07-28/harness/extract_944.sh
# sha256(source): 1fe375363fabe6e00ae016b2fc6367c26b0ba27c4612185697210eefd3a4a1d6
# build_commit:  b1d4bc257e57f7c3215ec8a237e9f87cdad8e35f
# Protocol doc:  benchmarks/bandvis_loo_944_2026-07-28.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
# ext944-instrument extraction — ZENSIM_AB_MODE=foldapp2 (944 streaming append2)
# over the SAME 11 canonical local legs (same pairs TSVs) as the W1 ext924 run
# (/mnt/v/output/zensim/ext924-run-2026-07-27/run.log). Only the mode differs.
# Driver binary: built from git archive of zensim origin/main
#   b1d4bc257e57f7c3215ec8a237e9f87cdad8e35f (sha256-identical to the main
#   checkout's prebuilt example).
set -u
BIN="$HOME/tmp/loo944/build-tree/target/release/examples/v2_ab_extract"
OUT=/mnt/v/output/zensim/ext944-run-2026-07-28
mkdir -p "$OUT"
ts() { date -u +%H:%M:%SZ; }

run_leg() {
  local name="$1" pairs="$2"
  echo "== $name start $(ts)"
  ZENSIM_AB_MODE=foldapp2 "$BIN" "$pairs" "$OUT/$name.csv"
  local rc=$?
  local rows=-1 cols=-1
  if [ -f "$OUT/$name.csv" ]; then
    rows=$(( $(wc -l < "$OUT/$name.csv") - 1 ))
    cols=$(head -1 "$OUT/$name.csv" | awk -F, '{print NF}')
  fi
  local want=$(( $(wc -l < "$pairs") - 1 ))
  echo "== $name done rc=$rc rows=$rows/$want cols=$cols $(ts)"
  if [ "$rc" -ne 0 ] || [ "$rows" -ne "$want" ]; then
    echo "ABORT: $name failed (rc=$rc rows=$rows want=$want)"
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
echo "LOO944-ALL-LEGS-DONE $(ts)"
