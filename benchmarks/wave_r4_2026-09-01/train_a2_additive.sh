#!/usr/bin/env bash
# wave-r4 ARM A2 (W-FAST-1a) / A6 (196k safesyn leg) — the 156-set additive
# student, ADD156's OWN recipe VERBATIM on the wave-r4 root.
#
# ADD156's registered recipe (benchmarks/sota944_campaign_2026-08-03.md
# "Lambda. PRIMARY lam = 2e-3 (ADD156's own)"; `safesyn_only` = safesyn-only,
# raw space, lasso, target-clip-min -100 per the hybrid lane's own
# reproduction (benchmarks/hybrid_candidate_2026-09-01.md SS12.1: the missing
# clip cost +0.007 to close): gram --space raw --max-feat 372
# --target-clip-min -100, fit-lasso --lam 0.002 --slice-file
# slice_basic156.txt --anchor-stride 37 --anchor-prefix --embed-repro. Same
# owner tooling as scripts/hybrid_distill.sh (SS3.0.1: reuse, don't rewrite).
#
# A2 = this recipe on the wave-r4 111k leg (ext_safesyn_full / safesyn_pure,
#      available without the decode step).
# A6 = the SAME recipe on the 196k leg (once decoded + extracted) — the
#      hybrid lane's own measured +0.057 CID22 lever (SS12.1/SS12.3).
#
# Usage: train_a2_additive.sh <ARM:A2|A6> <safesyn.parquet>
set -euo pipefail
ARM="${1:?ARM required: A2|A6}"
SAFE="${2:?safesyn parquet required}"
[ -f "$SAFE" ] || { echo "ABORT: missing $SAFE"; exit 1; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BDR="${ZL_BDR:-/mnt/v/zen/cargo-targets/waver4/release/bake_dial_refit}"
SLICE="$REPO/scripts/sota944/slice_basic156.txt"
OUT="${WR4_OUT:-/mnt/v/output/zensim/wave-r4-2026-09-01/bakes}"
mkdir -p "$OUT"
LAM=0.002

echo "== $ARM gram (raw, max-feat 372, target-clip-min -100) on $SAFE $(date -u +%H:%M:%SZ)"
"$BDR" gram --parquet "$SAFE" --target human_score --target-scale 100 \
    --target-clip-min -100 --space raw --max-feat 372 --out "$OUT/${ARM}_gram_human_c100.npz"

echo "== $ARM fit-lasso lam=$LAM (ADD156's own) $(date -u +%H:%M:%SZ)"
"$BDR" fit-lasso --gram "$OUT/${ARM}_gram_human_c100.npz" --space raw --target human_score \
    --lam "$LAM" --slice-file "$SLICE" \
    --anchor-parquet "$SAFE" --anchor-target human_score --anchor-scale 100 \
    --anchor-stride 37 --anchor-prefix \
    --embed-repro --out "$OUT/${ARM}_r4.bin"

echo "DONE $ARM -> $OUT/${ARM}_r4.bin"
