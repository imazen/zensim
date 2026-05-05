#!/usr/bin/env bash
# Build the rebalanced extended training CSV by:
#   1. Concordance-filter the gen-* encoding output (training_v06_rebalance.csv)
#      to a safe-extended sub-CSV.
#   2. Concatenate the existing safe-synthetic + e1-fill extended CSV with the
#      new gen-* safe rows.
#   3. Output: training_safe_synthetic_rebalanced.csv (header + all rows).
#
# The resulting CSV is the input to train_v06_baseline.sh / train_v06_cclass.sh
# (set EXTENDED_CSV=... to override).

set -euo pipefail

ROOT=${ROOT:-/home/lilith/work/zen/zensim--v06-rebalance}
SAFE_BASE=${SAFE_BASE:-/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv}
E1_FILL=${E1_FILL:-/mnt/v/output/zensim/training.csv}
GEN_CSV=${GEN_CSV:-/mnt/v/output/zensim/v06-rebalance/training.csv}
OUT_DIR=${OUT_DIR:-/mnt/v/output/zensim/v06-rebalance}
OUT=${OUT:-$OUT_DIR/training_safe_synthetic_rebalanced.csv}
TS=$(date -u +%Y%m%dT%H%M%S)

mkdir -p "$OUT_DIR"

[ -f "$SAFE_BASE" ] || { echo "missing $SAFE_BASE"; exit 1; }
[ -f "$E1_FILL" ] || { echo "missing $E1_FILL"; exit 1; }
[ -f "$GEN_CSV" ] || { echo "missing $GEN_CSV (gen-* encoding output)"; exit 1; }

# Step 1: filter the gen-* output through concordance + ban.
GEN_FILTERED=$OUT_DIR/training_gen_filtered_${TS}.csv
echo "[1/3] concordance-filter gen-* encoding output"
python3 "$ROOT/benchmarks/build_extended_safe_csv.py" "$GEN_CSV" "$GEN_FILTERED"

# Step 2: filter the e1 fill (already done in safe-synthetic-extended; redo here
# to keep the pipeline self-contained and reproducible against the current
# training.csv state).
E1_FILTERED=$OUT_DIR/training_e1_filtered_${TS}.csv
echo "[2/3] concordance-filter e1 fill"
python3 "$ROOT/benchmarks/build_extended_safe_csv.py" "$E1_FILL" "$E1_FILTERED"

# Step 3: concatenate base safe + e1 filtered + gen filtered (header dedup).
echo "[3/3] concatenating safe-base + e1-filtered + gen-filtered"
{
  cat "$SAFE_BASE"
  tail -n +2 "$E1_FILTERED"
  tail -n +2 "$GEN_FILTERED"
} > "$OUT"

n_safe=$(($(wc -l < "$SAFE_BASE") - 1))
n_e1=$(($(wc -l < "$E1_FILTERED") - 1))
n_gen=$(($(wc -l < "$GEN_FILTERED") - 1))
n_total=$(($(wc -l < "$OUT") - 1))
echo "rows: safe=$n_safe + e1=$n_e1 + gen=$n_gen = total=$n_total"
echo "wrote $OUT"

# Quick class breakdown via filename heuristic
echo ""
echo "content-class breakdown of source paths (filename heuristic):"
awk -F',' 'NR>1 {print $1}' "$OUT" | awk -F'/' '{print $NF}' \
  | sed -E 's/_[0-9]+sq.*$//; s/_[0-9]+x[0-9]+.*$//' | sort -u \
  | awk '{
      l = tolower($0);
      if (l ~ /^gen-screen__/) c="screen";
      else if (l ~ /^gen-doc__/) c="document";
      else if (l ~ /^gen-chart__/) c="lineart";
      else if (l ~ /^gen-line__/) c="lineart";
      else if (l ~ /^gen-mixed__/) c="photo";
      else if (l ~ /^(terminal|windows|gui|imac|imessage|gmessages)/) c="screen";
      else if (l ~ /chart|graph|piechart/) c="lineart";
      else c="photo";
      cnt[c]++
    } END {
      for (k in cnt) print "  " k ": " cnt[k]
    }' | sort
