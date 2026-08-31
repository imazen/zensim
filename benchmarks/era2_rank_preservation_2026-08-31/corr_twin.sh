#!/bin/bash
set -u
ARM="$1"
WT="${ZR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ZM944_BIN="$WT/target/release/examples/v2_ab_extract"
export CORR944_WORK="$HOME/tmp/era2rank/corr944-$ARM"
export CORR944_OUT="/mnt/v/output/zensim/era2-rank-2026-08-31/grids/corruption_grid_944col_$ARM.parquet"
cd "$WT" && python3 scripts/v_next/build_corr944.py
