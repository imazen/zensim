#!/usr/bin/env bash
# gmsd lane: GMSD as an arm of the zensim speed owner (ssim2_speed_bar),
# same protocol as scripts/demos/speed_matrix_run.sh: zenbench interleaved,
# ZEN_S2_SINGLE_CALL=1, 32 rounds, taskset-pinned, nice/ionice idle. Rev1
# process = named profiles + peers; Rev3 process = the frozen Rev3 rich
# ensemble; fast_ssim2 (and gmsd, revision-independent) in both as bridges.
# 1 thread (cpu 2) and 8 threads (cpus 0-7, one CCD / one L3).
# Not wrapped in run-heavy (its cgroup makes zenbench's gating refuse); the
# caller holds the lanes' shared lock with a bare flock instead.
#   speed_run.sh <bench-binary> <out-dir>
set -euo pipefail
BIN=${1:?bench binary}; OUT=${2:?out dir}
CAL=/var/tmp/zensim-validation-2026-09-15/recovery/calibrated
SIZES=${SIZES:-64,256,1024,2048,4096}; ROUNDS=${ROUNDS:-32}; WALL_S=${WALL_S:-3600}
mkdir -p "$OUT"
E2=$(for s in 17101 17103 17107 17111 17113; do printf '%s,' "$CAL/R915_basic228_h128_s$s.bin"; done); E2=${E2%,}
W=0.2,0.2,0.2,0.2,0.2
python3 - "$CAL" <<'PY'
import hashlib, json, sys
d = json.load(open(f"{sys.argv[1]}/FROZEN.json"))
for name, m in d["models"].items():
    for mem in m["members"]:
        h = hashlib.sha256(open(mem["path"], "rb").read()).hexdigest()
        assert h == mem["sha256"], f"{name}: {mem['path']} is not the frozen bake"
print("frozen bakes verified")
PY
run_one() { # tag threads cpus rev arms
  local tag=$1 threads=$2 cpus=$3 rev=$4 arms=$5; local -a ens=()
  [ "$rev" = 3 ] && ens=(ZEN_S2_ENSEMBLE="$E2" ZEN_S2_ENSEMBLE_WEIGHTS="$W" ZEN_S2_ENSEMBLE_NAME=rev3_rich_basic228_ens5)
  rm -f "$OUT/$tag.json"
  echo "=== $tag rev=$rev threads=$threads cpus=$cpus arms=$arms $(date -u +%T)"
  uptime | tee "$OUT/$tag.load-before.txt"
  env ZENSIM_FORMULA_REV="$rev" "${ens[@]}" ZEN_S2_ARMS="$arms" ZEN_S2_SIZES="$SIZES" \
      ZEN_S2_SINGLE_CALL=1 ZEN_S2_ROUNDS="$ROUNDS" ZEN_S2_MIN_ROUNDS="$ROUNDS" ZEN_S2_WALL_S="$WALL_S" \
      RAYON_NUM_THREADS="$threads" ZENBENCH_RESULT_PATH="$OUT/$tag.json" \
      taskset -c "$cpus" nice -n19 ionice -c3 "$BIN" > "$OUT/$tag.log" 2>&1
  uptime | tee "$OUT/$tag.load-after.txt"
}
R1=fast_ssim2,butteraugli,gmsd,zensim_B,zensim_D
R3=fast_ssim2,gmsd,rev3_rich_basic228_ens5
run_one 1t-rev1 1 2 1 "$R1"
run_one 8t-rev1 8 0-7 1 "$R1"
run_one 1t-rev3 1 2 3 "$R3"
run_one 8t-rev3 8 0-7 3 "$R3"
echo "speed_run done $(date -u +%T)"
