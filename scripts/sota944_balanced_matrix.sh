#!/usr/bin/env bash
# sota944 balanced-selection pool matrix (AMENDMENT 8, campaign doc §8.1).
# Loops fulleval/verdict JSONs through the OWNER — `freeze_check --profile
# balanced-2026-08-04 --tsv` — and collates rows. This driver computes
# NOTHING itself (no-duplication rule); even the TSV header comes from the
# owner (`--tsv-header`).
#
# usage: scripts/sota944_balanced_matrix.sh [FULLEVAL_DIR] [OUT_TSV] [EXTRA_JSON...]
#   FULLEVAL_DIR  default /mnt/v/output/zensim/reports/fulleval (read-only)
#   OUT_TSV       default /mnt/v/output/zensim/reports/balanced/balanced_matrix_<UTC>.tsv
#   EXTRA_JSON    additional *.full.json cells (e.g. wave-7 arm-H verdicts)
# env: FREEZE_CHECK=/path/to/freeze_check overrides the binary.
set -euo pipefail

FE_DIR="${1:-/mnt/v/output/zensim/reports/fulleval}"
OUT="${2:-/mnt/v/output/zensim/reports/balanced/balanced_matrix_$(date -u +%Y-%m-%d).tsv}"
shift $(( $# > 2 ? 2 : $# )) || true
BIN="${FREEZE_CHECK:-$(dirname "$0")/../target/release/freeze_check}"
# The committed invalidation/annotation registry (board-integrity pass
# 2026-08-04): absent-not-failed axes are distinct from measured fails.
ANN="${ANNOTATIONS:-$(dirname "$0")/../benchmarks/eval_annotations.json}"

if [ ! -x "$BIN" ]; then
  echo "freeze_check not built at $BIN — cargo build --release -p zensim-validate --bin freeze_check" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"

"$BIN" --tsv-header > "$OUT"
n=0
score() {
  local f="$1" rc=0
  "$BIN" --fulleval "$f" --profile balanced-2026-08-04 --tsv \
      --annotations "$ANN" >> "$OUT" || rc=$?
  # rc=1 is a normal floor FAIL; rc>=2 is a parse/usage error — fail loud.
  if [ "$rc" -ge 2 ]; then
    echo "ERROR rc=$rc on $f" >&2
    exit "$rc"
  fi
  n=$((n + 1))
  echo "$(date -u +%H:%M:%SZ) [$n] scored $(basename "$f")" >&2
}

shopt -s nullglob
for f in "$FE_DIR"/*.fulleval.json; do score "$f"; done
for f in "$@"; do score "$f"; done

echo "matrix: $OUT ($(( $(wc -l < "$OUT") - 1 )) rows)" >&2
