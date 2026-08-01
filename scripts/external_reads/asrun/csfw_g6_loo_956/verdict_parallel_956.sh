#!/bin/bash
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/csfw-g6-loo-2026-07-29/harness/verdict_parallel_956.sh
# sha256(source): 09e67bc2d171bf4a4d956af946a4c962728fc9595ecd2ff9d32d6e7b0a38cebe
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
# Score lin956 + every LOO drop bake on the 8 canonical 956 legs via
# bake_verdict (tip build 7bfd511d, copied out of the checkout), features-root
# = the ext956 instrument dir. 4-lane parallel, idempotent (skips cached
# JSONs). Corpora = the 8 of e2's suite that exist as canonical local legs
# (the imazen26/nonphoto NN-joined legs have no 956 extraction — reported as
# such, same caveat as the 944 wave).
set -u
BV="$HOME/tmp/g6loo/bake_verdict"
ROOT=/mnt/v/zen/zensim-training/ext956-instrument-2026-07-29
BAKES=/mnt/v/output/zensim/csfw-g6-loo-2026-07-29/bakes
OUT=/mnt/v/output/zensim/csfw-g6-loo-2026-07-29/verdicts956
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
  echo "lin956 $BAKES/lin956.bin"
  for b in "$BAKES"/lin956_loo/drop_*.bin; do
    echo "$(basename "$b" .bin) $b"
  done
} | xargs -P 4 -n 2 bash -c 'one "$0" "$1"'
echo VERDICTS956-DONE
