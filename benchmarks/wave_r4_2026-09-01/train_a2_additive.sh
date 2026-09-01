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
# A2b = the free-features lane's 2026-09-01 hand-off (benchmarks/
#       free_features_2026-09-01.md SS4): 156 + 109 free slots (72 peaks,
#       already live in any pools-regime wave-r4 table since a full 944 walk
#       computes them via the append kernel regardless of the RawMoments
#       toggle -- VERIFIED empirically on ext_safesyn_full.parquet, all 37
#       raw-moment indices substantially nonzero, no re-extraction needed).
#       Needs WR4_MAXFEAT=944 (the free slots sit past f371) and
#       WR4_SLICE=slice_basic156_free.txt -- both env-overridable below,
#       defaulting to A2's exact prior behavior (372 / slice_basic156.txt)
#       so this change is backward compatible.
#
# Usage: train_a2_additive.sh <ARM:A2|A6|A2b|...> <safesyn.parquet> [lam]
set -euo pipefail
ARM="${1:?ARM required: A2|A6|A2b|...}"
SAFE="${2:?safesyn parquet required}"
[ -f "$SAFE" ] || { echo "ABORT: missing $SAFE"; exit 1; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BDR="${ZL_BDR:-/mnt/v/zen/cargo-targets/waver4/release/bake_dial_refit}"
SLICE="${WR4_SLICE:-$REPO/scripts/sota944/slice_basic156.txt}"
MAXFEAT="${WR4_MAXFEAT:-372}"
OUT="${WR4_OUT:-/mnt/v/output/zensim/wave-r4-2026-09-01/bakes}"
mkdir -p "$OUT"
LAM="${3:-0.002}"

GRAM="$OUT/${ARM}_gram_human_c100_mf${MAXFEAT}.npz"
if [ ! -f "$GRAM" ]; then
  echo "== $ARM gram (raw, max-feat $MAXFEAT, target-clip-min -100) on $SAFE $(date -u +%H:%M:%SZ)"
  "$BDR" gram --parquet "$SAFE" --target human_score --target-scale 100 \
      --target-clip-min -100 --space raw --max-feat "$MAXFEAT" --out "$GRAM"
else
  echo "== $ARM gram already built at $GRAM (max-feat $MAXFEAT), reusing"
fi

echo "== $ARM fit-lasso lam=$LAM slice=$(basename "$SLICE") $(date -u +%H:%M:%SZ)"
"$BDR" fit-lasso --gram "$GRAM" --space raw --target human_score \
    --lam "$LAM" --slice-file "$SLICE" \
    --anchor-parquet "$SAFE" --anchor-target human_score --anchor-scale 100 \
    --anchor-stride 37 --anchor-prefix \
    --embed-repro --out "$OUT/${ARM}_r4_l${LAM}.bin"

echo "DONE $ARM -> $OUT/${ARM}_r4_l${LAM}.bin"
