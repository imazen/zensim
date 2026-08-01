#!/bin/bash
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/bandvis-loo-2026-07-28/harness/verdict_parallel.sh
# sha256(source): 3f20d7ccfdde4a1ead69c9a24b831209ca7e65afd705893c8ecbeb50a304e685
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
# 4-lane parallel variant of verdict_loo944.sh (idempotent: skips cached JSONs).
set -u
BV="$HOME/tmp/loo944/build-tree/target/release/bake_verdict"
ROOT=/mnt/v/zen/zensim-training/ext944-instrument-2026-07-28
BAKES=/mnt/v/output/zensim/bandvis-loo-2026-07-28/bakes
OUT=/mnt/v/output/zensim/bandvis-loo-2026-07-28/verdicts944
mkdir -p "$OUT"
CORP=cid22,kadid,tid,csiq,live,konjnd,aic3,aic4

one() {
  local name="$1" bin="$2"
  if [ -f "$OUT/$name.json" ]; then echo "skip $name (cached)"; return 0; fi
  nice -n19 ionice -c3 "$BV" --bake "$bin" --regime 720 \
    --features-root "$ROOT" --corpora "$CORP" --dial-grid /nonexistent \
    --json "$OUT/$name.json" > "$OUT/$name.stdout.log" 2>&1
  echo "done $name rc=$?"
}
export -f one
export BV ROOT OUT CORP

{
  echo "lin944 $BAKES/lin944.bin"
  for b in "$BAKES"/lin944_loo/drop_*.bin; do
    echo "$(basename "$b" .bin) $b"
  done
} | xargs -P 4 -n 2 bash -c 'one "$0" "$1"'
echo VERDICTS944-DONE
