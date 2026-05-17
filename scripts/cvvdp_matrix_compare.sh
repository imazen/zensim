#!/usr/bin/env bash
# Compare V_22-CVVDP matrix bakes side-by-side via bake_verdict.
# Usage: bash scripts/cvvdp_matrix_compare.sh <bakes_dir> [output.md]

set -euo pipefail

DIR=${1:-benchmarks/cvvdp_matrix_2026-05-17}
OUT=${2:-${DIR}/matrix_summary.md}

cd "$(dirname "$0")/.."

VERDICT=./target/release/bake_verdict
[ -x "$VERDICT" ] || { echo "bake_verdict not built"; exit 1; }

# Run verdict on each bake, capture aggregate SROCC/Z-RMSE per corpus
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

declare -a BAKES
for f in "$DIR"/*.bin; do
  BAKES+=("$f")
done

# Extract per-corpus aggregate row from each verdict
{
  echo "# CVVDP matrix verdict summary ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo ""
  echo "Bakes evaluated:"
  for b in "${BAKES[@]}"; do
    echo "- $(basename "$b" .bin) ($(stat -c%s "$b") bytes)"
  done
  echo ""
  echo "## Aggregate SROCC by corpus"
  echo ""
  echo "| variant | CID22 | KADID | TID | KonJND | AIC-3 |"
  echo "|---|---:|---:|---:|---:|---:|"

  for b in "${BAKES[@]}"; do
    name=$(basename "$b" .bin)
    "$VERDICT" --bake "$b" --output "$TMP/${name}.md" > /dev/null 2>&1
    # Aggregate panel row is `| V_X bake | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |`
    # One per corpus, in order CID22, KADID, TID, KonJND, AIC-3
    SROCC=$(grep -E '^\| V_X bake \|' "$TMP/${name}.md" | awk -F'|' '{print $3}' | tr -d ' ')
    echo -n "| $name "
    for v in $SROCC; do echo -n "| $v "; done
    echo "|"
  done

  echo ""
  echo "## Aggregate Z-RMSE by corpus"
  echo ""
  echo "| variant | CID22 | KADID | TID | KonJND | AIC-3 |"
  echo "|---|---:|---:|---:|---:|---:|"

  for b in "${BAKES[@]}"; do
    name=$(basename "$b" .bin)
    ZRMSE=$(grep -E '^\| V_X bake \|' "$TMP/${name}.md" | awk -F'|' '{print $8}' | tr -d ' ')
    echo -n "| $name "
    for v in $ZRMSE; do echo -n "| $v "; done
    echo "|"
  done

  echo ""
  echo "## Aggregate PWRC by corpus"
  echo ""
  echo "| variant | CID22 | KADID | TID | KonJND | AIC-3 |"
  echo "|---|---:|---:|---:|---:|---:|"

  for b in "${BAKES[@]}"; do
    name=$(basename "$b" .bin)
    PWRC=$(grep -E '^\| V_X bake \|' "$TMP/${name}.md" | awk -F'|' '{print $7}' | tr -d ' ')
    echo -n "| $name "
    for v in $PWRC; do echo -n "| $v "; done
    echo "|"
  done

  echo ""
  echo "_Corpora are in bake_verdict default order: CID22, KADID, TID, KonJND, AIC-3._"
  echo "_Each cell is the aggregate Mohammadi-panel statistic for that variant on that corpus._"
} > "$OUT"

echo "wrote $OUT"
head -40 "$OUT"
