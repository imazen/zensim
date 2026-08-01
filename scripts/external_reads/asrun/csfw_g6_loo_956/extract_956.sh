#!/bin/bash
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/csfw-g6-loo-2026-07-29/harness/extract_956.sh
# sha256(source): 2a8f291b0a8e3701d8ffb808df3e0e5d1477bc22c817c90105de6a9244eefcf5
# build_commit:  7bfd511de78f85e8fcd618df15716ca56575bb60
# Protocol doc:  benchmarks/csfw_g6_loo_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
# ext956-instrument extraction — ZENSIM_AB_MODE=foldcsfw (956 streaming csfw tier-1)
# over the SAME 11 canonical local legs (same pairs TSVs) as the ext944-instrument
# run (/mnt/v/zen/zensim-training/ext944-instrument-2026-07-28/_MANIFEST.json,
# itself the W1 ext924 pairs — /mnt/v/output/zensim/ext924-run-2026-07-27/run.log).
# Only the mode differs. Driver binary: built in the zensim main checkout at
# origin/main 7bfd511de78f85e8fcd618df15716ca56575bb60 (copied out; sha256
# f7343158f925ab9c0148a9250cc0fae8f03f4012387dbeed7ba577e40a1042a8).
set -u
BIN="$HOME/tmp/g6loo/v2_ab_extract"
OUT=/mnt/v/output/zensim/ext956-run-2026-07-29
mkdir -p "$OUT"
ts() { date -u +%H:%M:%SZ; }

run_leg() {
  local name="$1" pairs="$2"
  echo "== $name start $(ts)"
  ZENSIM_AB_MODE=foldcsfw "$BIN" "$pairs" "$OUT/$name.csv"
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
echo "LOO956-ALL-LEGS-DONE $(ts)"
