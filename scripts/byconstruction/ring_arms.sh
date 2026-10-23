#!/bin/bash
# Train the ringing-corrected (hpm) arms: B's signed recipe, certified-monotone M, and FR (ringing sign-free), 12 E21
# cells each, on the derived hpm root. Exploration record: benchmarks/byconstruction_2026-10-22.md.
# Usage: ring_arms.sh ROOT BIN_DIR JOBS_PER_ARM
set -euo pipefail
ROOT=$1; BIN=$2; J=${3:-8}
H=$(cd "$(dirname "$0")" && pwd)/../rev4_featpot
export BYC_ROOT=$ROOT REV4_V2_BIN_DIR=$BIN RAYON_NUM_THREADS=1
BASE=sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1
COLS=$HOME/tmp/byc/cols_c.txt
FOLDS=kadid,tid2013,konfig,cid22_a25
python3 "$H/byc_cells.py" run --arm "" --base-spec $BASE --cols $COLS --label Bhpm --folds $FOLDS --seeds 0,1,2 --jobs $J > "$HOME/tmp/byc/run_Bhpm.log" 2>&1 &
python3 "$H/byc_cells.py" run --arm mono:thr:mp32 --base-spec $BASE --cols $COLS --label Mhpm --folds $FOLDS --seeds 0,1,2 --jobs $J > "$HOME/tmp/byc/run_Mhpm.log" 2>&1 &
python3 "$H/byc_cells.py" run --arm mono:thr:freering --base-spec $BASE --cols $COLS --label FRhpm --folds $FOLDS --seeds 0,1,2 --jobs $J > "$HOME/tmp/byc/run_FRhpm.log" 2>&1 &
wait
echo ALL-ARMS-DONE
