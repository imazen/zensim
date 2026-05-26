#!/usr/bin/env bash
# Generate mock-iwssim VALIDATION-ONLY CSVs for V_22-IW training.
#
# ⚠ DATA-INTEGRITY (2026-05-25, task #215): the column this script emits is a
# verbatim copy of `human_score`. An earlier version named it `iwssim`, and the
# "mock" qualifier lived ONLY in the output filename. When a downstream join
# renamed the file, the mock column became indistinguishable from a real metric
# and leaked into the canonical TRAINING parquets — the kadid/tid iwssim
# corruption root-caused in benchmarks/DATA_INTEGRITY_root_cause_2026-05-25.md.
#
# FIX: the mock column is now named `iwssim_MOCK_VAL_ONLY`. The canonical
# builders (scripts/canonical_corpus/join_safety.py) and the trainer REJECT any
# column whose name matches `*mock*`, so this signal can no longer enter a
# training/canonical corpus by accident. Use it ONLY as a per-epoch validation
# SROCC monitor (--group ...:0.0:1.0, i.e. val_weight only, train_weight 0).
#
# The V_22-IW trainer runs with --target-column iwssim. The safesyn
# training group carries the real Wang & Li 2011 IW-SSIM target.
# KADID / TID / KonJND validation groups do NOT have iwssim columns
# (those corpora aren't covered by the safesyn IW-SSIM computation);
# they only carry `human_score`.
#
# This script creates side-by-side mock CSVs where `iwssim_MOCK_VAL_ONLY`
# is a verbatim copy of `human_score`. The trainer's RankNet loss is
# rank-invariant within each group, so the absolute scale doesn't
# matter — what matters is that each (KADID|TID|KonJND) row's mock
# column places it in its native rank order vs other rows in
# the same group. That's exactly what `human_score` provides for the
# validation corpora.
#
# Effect on training: each epoch's held-out validation SROCC against
# `human_score` (under the name `iwssim`) tells the trainer how well
# the safesyn-trained MLP generalizes to real human MOS. This is
# strictly the right signal for V_22-IW's V_18 ssim2-bias-escape
# hypothesis — see benchmarks/v0_22_iw_methodology_2026-05-16.md.
#
# Usage:
#   ./scripts/v_next/v0_22_iw_make_mock_val_csvs.sh
#
# Inputs:
#   /mnt/v/zen/zensim-training/2026-05-15-full-features/{kadid,tid,konjnd}_features_372col_2026-05-15.csv
#
# Outputs:
#   /mnt/v/zen/zensim-training/2026-05-16/{kadid,tid,konjnd}_features_372col_2026-05-15_iwssim_mock.csv

set -euo pipefail

SRC_DIR=/mnt/v/zen/zensim-training/2026-05-15-full-features
OUT_DIR=/mnt/v/zen/zensim-training/2026-05-16

mkdir -p "$OUT_DIR"

for name in kadid tid konjnd; do
  src="$SRC_DIR/${name}_features_372col_2026-05-15.csv"
  out="$OUT_DIR/${name}_features_372col_2026-05-15_iwssim_mock.csv"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: source CSV missing: $src" >&2
    exit 2
  fi
  echo "writing $out (iwssim_MOCK_VAL_ONLY := human_score copy for validation only)"
  # Header row: insert "iwssim_MOCK_VAL_ONLY" between human_score and f0.
  # Data rows: duplicate the human_score (field 2) into a new field 3.
  # The _MOCK_VAL_ONLY suffix is load-bearing: canonical builders + the trainer
  # reject any column matching *mock*, so this val-only signal can NEVER leak
  # into a training/canonical corpus (the 2026-05-25 corruption fix).
  awk -F, '
    NR==1 {
      print "ref_basename,human_score,iwssim_MOCK_VAL_ONLY," substr($0, length($1) + length($2) + 3)
      next
    }
    {
      print $1 "," $2 "," $2 "," substr($0, length($1) + length($2) + 3)
    }
  ' "$src" > "$out"
  echo "  rows: $(wc -l < "$out")"
done

echo
echo "Mock validation CSVs ready at $OUT_DIR/"
echo "The mock column is named iwssim_MOCK_VAL_ONLY — VALIDATION ONLY."
echo "Pass to zensim_mlp_train with train_weight=0.0 (val_weight only):"
echo "  --group kadid:$OUT_DIR/kadid_features_372col_2026-05-15_iwssim_mock.csv:0.0:1.0"
echo "  --group tid:$OUT_DIR/tid_features_372col_2026-05-15_iwssim_mock.csv:0.0:1.0"
echo "  --group konjnd:$OUT_DIR/konjnd_features_372col_2026-05-15_iwssim_mock.csv:0.0:1.0"
echo
echo "Do NOT pass iwssim_MOCK_VAL_ONLY as --target-column with train_weight>0:"
echo "the trainer rejects *mock* columns to prevent the 2026-05-25 leak."
