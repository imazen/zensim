#!/usr/bin/env bash
# Interleaved marginal-cost run of the restored families (zenbench, paired rounds), single thread
# and 8 threads, sizes 256/1024/2048/4096. The arms are the C8-on baseline plus the nested restored
# chain (see extract_paths_bench.rs `toggles_restore`). NOT wrapped in run-heavy (its cgroup makes
# zenbench's gate refuse); serialised with the other lanes by the shared flock only.
# Usage: cost.sh <tag>      Outputs: $ROOT/cost_<tag>_mt{1,8}.zenbench
set -euo pipefail
tag=${1:?tag}
ROOT=/var/tmp/restore-cuts
exe=$(cat "$ROOT/bin/bench_path")
for threads in 1 8; do
    out=$ROOT/cost_${tag}_mt${threads}.zenbench
    rm -f "$out"
    flock /home/lilith/tmp/devin/heavy.lock env \
        ZEN_XP_ARMS=fold1502_gmsbank,fold1562_mapdev,fold1790_z1max,fold1820_gmsnative,fold1825_dvifmgate \
        ZEN_XP_SIZES=256,1024,2048,4096 ZEN_XP_ROUNDS=8 ZEN_XP_MIN_ROUNDS=8 ZEN_XP_WALL_S=180 \
        ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt RAYON_NUM_THREADS=$threads ZENBENCH_RESULT_PATH="$out" \
        "$exe" 2>&1 | tail -80
    test -s "$out"
    sha256sum "$out"
done
