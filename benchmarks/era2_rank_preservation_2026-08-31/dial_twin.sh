#!/bin/bash
# Build a dial_grid_944col twin at one ZENSIM_H_TILE setting (owner: build_dial944.py).
set -u
ARM="$1"; TILE="${2:-}"
WT="${ZR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ZM944_BIN="$WT/target/release/examples/v2_ab_extract"
export DIAL944_WORK="$HOME/tmp/era2rank/dial944-$ARM"
export DIAL944_OUT="/mnt/v/output/zensim/era2-rank-2026-08-31/grids/dial_grid_944col_$ARM.parquet"
[ -n "$TILE" ] && export ZENSIM_H_TILE="$TILE"
cd "$WT" && python3 scripts/v_next/build_dial944.py
