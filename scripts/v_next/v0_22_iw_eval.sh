#!/usr/bin/env bash
# V_22-IW seed=1 evaluation harness — full Mohammadi panel on every
# held-out corpus per CLAUDE.md "SROCC-only verdicts BANNED + ssim2-target
# training bias" (2026-05-15).
#
# Usage:
#   ./scripts/v_next/v0_22_iw_eval.sh [<bake_path>] [<output_log>]
#
# Defaults to the V_22-IW seed=1 bake from 2026-05-16, writing to
# `benchmarks/v0_22_iw_seed1_2026-05-16_eval.log`. Pass alternate args
# for sweep runs (seed=2, seed=3) or alternate dates.
#
# The eval harness `dataset_metric_baseline` was upgraded with T3.1
# (commit 76360ae, 2026-05-16) to emit the full Mohammadi panel per
# (band, metric): SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE + MAE.
# Two band tables emit: 10-band PRIMARY (B0..B9 width-10) and the
# legacy 4-band CID22 cuts (B0..B3 + Near-PJND). Aggregate panel +
# significance tests (MRR + Wilcoxon) emit per dataset.
#
# Per CLAUDE.md "Multi-stat agreement ship gate": V_22-IW ships when at
# least 3 of 5 stats (SROCC, PLCC, KROCC, PWRC, Z-RMSE) agree on
# improvement vs V_18 ship on the held-out corpus.

set -euo pipefail

DEFAULT_BAKE="benchmarks/v0_22_iw_seed1_2026-05-16.bin"
DEFAULT_LOG="benchmarks/v0_22_iw_seed1_2026-05-16_eval.log"

BAKE="${1:-$DEFAULT_BAKE}"
LOG="${2:-$DEFAULT_LOG}"
PER_PAIR="${LOG%.log}_per_pair.csv"

if [[ ! -f "$BAKE" ]]; then
  echo "ERROR: bake not found at $BAKE" >&2
  echo "Did training finish? Check /tmp/v0_22_iw_seed1_train.log" >&2
  exit 2
fi

EVAL_BIN="$(dirname "$0")/../../target/release/examples/dataset_metric_baseline"
if [[ ! -x "$EVAL_BIN" ]]; then
  echo "ERROR: dataset_metric_baseline not built at $EVAL_BIN" >&2
  echo "Run: cargo build --release -p zensim-bench --example dataset_metric_baseline --features training" >&2
  exit 2
fi

echo "Running eval against $BAKE"
echo "Output: $LOG (per-pair: $PER_PAIR)"
echo

"$EVAL_BIN" \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
  --konjnd /mnt/v/datasets/KonJND-1k/KonJND-1k \
  --aic3 /mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv \
  --v04-bake "$BAKE" \
  --max-pairs 99999 \
  --per-pair-output "$PER_PAIR" \
  2>&1 | tee "$LOG"

echo
echo "Eval complete: $LOG"
echo
echo "Next steps:"
echo "  1. Inspect the 10-band full Mohammadi panel section per dataset"
echo "  2. Compare to V_18 ship results at benchmarks/v0_18_methodology_2026-05-13.md"
echo "  3. Apply the ship-gate rule from CLAUDE.md: >=3 of 5 stats"
echo "     (SROCC, PLCC, KROCC, PWRC, Z-RMSE) must agree on the win"
echo "  4. If hypothesis confirms at seed=1: sweep seeds 2 and 3"
echo "  5. If hypothesis falsifies at seed=1: stop, document the negative result"
echo "     in benchmarks/v0_22_iw_methodology_2026-05-16.md and update CLAUDE.md"
