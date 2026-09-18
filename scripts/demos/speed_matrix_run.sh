#!/usr/bin/env bash
# The cross-generation speed matrix: every zensim generation and the peer
# metrics, interleaved, across five geometries, at one thread and at sixteen.
#
# TWO PROCESSES, not one, and the reason is arithmetic rather than tidiness.
# `ssim_form::active_revision` is a process-global `OnceLock`, so
# `ZENSIM_FORMULA_REV=3` changes the pixel kernels for EVERYTHING in the
# process. The frozen Rev3 ensembles need it; the named profiles (`B`, `C`,
# `D`, `PreviewV0_2`) carry revision-1 coefficients and under Rev3 zensim
# itself prints
#
#   "ZENSIM_FORMULA_REV pins Rev3 pixels, but a built-in profile's bake is
#    revision-1 coefficients ... not a served score"
#
# and the timing moves with the arithmetic (measured: zensim_B at 64^2 ran
# 177.8us at Rev1 and 165.8us at Rev3). So a single-process matrix would
# publish the named-profile arms under a revision they were not fit for.
# Instead: one Rev1 process for the profiles and peers, one Rev3 process for
# the two frozen ensembles, and `fast_ssim2` in BOTH as the cross-process
# anchor — it is revision-independent, so if it moves between the two
# processes the box moved and the two halves are not comparable.
#
# Do NOT wrap this in `run-heavy`: its cgroup scope + job caps make zenbench's
# own resource gating refuse the run (same reason `just rev3-cost` says so).
# The workload is already pinned and at idle CPU/IO priority below, which is
# what the machine-safety rule asks for. The BUILD above it is a separate,
# genuinely heavy step and does go through run-heavy.
#
# Usage: scripts/demos/speed_matrix_run.sh <out-dir> [calibrated-bake-dir]
set -euo pipefail

OUT="${1:?usage: speed_matrix_run.sh <out-dir> [calibrated-bake-dir]}"
CAL="${2:-/var/tmp/zensim-validation-2026-09-15/recovery/calibrated}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SIZES="${SIZES:-64,256,1024,2048,4096}"
ROUNDS="${ROUNDS:-32}"
WALL_S="${WALL_S:-3600}"
# The 1/4/8-thread rows all sit on CCD0 (cpus 0-7, the 96 MiB-L3 die), so the
# scaling column across them is within ONE L3 and means what it looks like.
# The 16-thread row necessarily spans both dies: it is more cores AND a
# different cache regime, so its factor is not comparable to the other three.
# Stated wherever the numbers are published.
CPU_1T="${CPU_1T:-2}"
CPU_4T="${CPU_4T:-0-3}"
CPU_8T="${CPU_8T:-0-7}"
CPU_MT="${CPU_MT:-0-15}"
THREADS_MT="${THREADS_MT:-16}"

mkdir -p "$OUT"

bakes() { local p="$1" s; local -a out=(); for s in 17101 17103 17107 17111 17113; do
  out+=("$CAL/${p}_s${s}.bin"); done; (IFS=,; echo "${out[*]}"); }
E1="$(bakes R915_y60_h32)"
E2="$(bakes R915_basic228_h128)"
W=0.2,0.2,0.2,0.2,0.2
# Explicit arm lists, never "" (which means ALL). The Rev3 ensembles must not
# run inside the Rev1 process: `BakeScorer`'s narrow-plan fast path does NOT
# refuse a revision mismatch, so they would quietly score at revision-1 pixels
# and be published as revision-3 numbers. Naming the arms is what keeps the two
# processes from bleeding into each other.
REV1_ARMS=fast_ssim2,butteraugli,ssimulacra2_rs,zensim_V0_2,zensim_B,zensim_C,zensim_D
ENS_ARMS=fast_ssim2,rev3_fast_y60_ens5,rev3_rich_basic228_ens5
# The intermediate thread counts drop the two non-threading peers. Neither
# butteraugli nor the rust-av ssimulacra2 has a thread pool at all, so a 4T and
# an 8T column for them would be three more copies of their 1T number at the
# cost of most of the run's wall time. They stay in the 1T and 16T rows, where
# that flatness is the point being shown.
REV1_ARMS_MT=fast_ssim2,zensim_V0_2,zensim_B,zensim_C,zensim_D

# The frozen bakes are evidence: refuse to benchmark bytes that do not match
# the digests their own FROZEN.json recorded.
python3 - "$CAL" <<'PY'
import hashlib, json, sys
d = json.load(open(f"{sys.argv[1]}/FROZEN.json"))
for name, m in d["models"].items():
    for mem in m["members"]:
        h = hashlib.sha256(open(mem["path"], "rb").read()).hexdigest()
        assert h == mem["sha256"], f"{name}: {mem['path']} is not the frozen bake"
        print(f"frozen-ok {name} {mem['path'].rsplit('/', 1)[1]} {h}")
PY

# `BIN_PLAIN` / `BIN_RAYON` come from the caller (the `just` recipe reads them
# out of cargo's JSON messages), because `ls -t` cannot tell the two feature
# builds apart and picking the wrong one would silently publish a threaded
# fast-ssim2 as the single-threaded arm. Falling back to newest-wins is only
# for the one-build case.
BIN_PLAIN="${BIN_PLAIN:-$(ls -t "$REPO"/zensim-bench/target/release/deps/ssim2_speed_bar-* 2>/dev/null | grep -v '\.d$' | head -1)}"
BIN_RAYON="${BIN_RAYON:-}"
: "${BIN_PLAIN:?build first: cargo bench --no-run --manifest-path zensim-bench/Cargo.toml --bench ssim2_speed_bar}"

run_one() {  # <tag> <binary> <threads> <cpuset> <rev> <arms>
  local tag=$1 bin=$2 threads=$3 cpus=$4 rev=$5 arms=$6
  local json="$OUT/$tag.json" log="$OUT/$tag.log"
  local -a ens=()
  # `RUN_ONLY=4t-rev1,8t-rev1` re-runs (or adds) named configurations without
  # discarding the ones already measured. Adding a thread column to a finished
  # matrix should not mean re-measuring the columns that are already good.
  if [ -n "${RUN_ONLY:-}" ] && ! printf '%s' ",$RUN_ONLY," | grep -q ",$tag,"; then
    echo "=== $tag: skipped (RUN_ONLY) ==="
    return 0
  fi
  rm -f "$json"
  # The ensemble bakes are declared ONLY for the revision-3 process; the
  # revision-1 process does not even load them, so it cannot accidentally
  # price them and does not pay for their per-size validation computes.
  if [ "$rev" = 3 ]; then
    ens=(ZEN_S2_ENSEMBLE="$E1" ZEN_S2_ENSEMBLE_WEIGHTS="$W"
         ZEN_S2_ENSEMBLE_NAME=rev3_fast_y60_ens5
         ZEN_S2_ENSEMBLE_2="$E2" ZEN_S2_ENSEMBLE_2_WEIGHTS="$W"
         ZEN_S2_ENSEMBLE_2_NAME=rev3_rich_basic228_ens5)
  fi
  : "${arms:?every run names its arms explicitly; \"\" would mean ALL}"
  echo "=== $tag: rev=$rev threads=$threads cpus=$cpus arms=$arms ==="
  uptime | tee "$OUT/$tag.load-before.txt"
  env ZENSIM_FORMULA_REV="$rev" "${ens[@]}" \
      ZEN_S2_ARMS="$arms" ZEN_S2_SIZES="$SIZES" ZEN_S2_SINGLE_CALL=1 \
      ZEN_S2_ROUNDS="$ROUNDS" ZEN_S2_MIN_ROUNDS="$ROUNDS" ZEN_S2_WALL_S="$WALL_S" \
      RAYON_NUM_THREADS="$threads" ZENBENCH_RESULT_PATH="$json" \
      taskset -c "$cpus" nice -n19 ionice -c3 "$bin" >"$log" 2>&1
  uptime | tee "$OUT/$tag.load-after.txt"
}

# A RUN_ONLY top-up must not rewrite the provenance of the sweep it is adding
# to: `started_utc` would then describe the top-up rather than the matrix.
if [ -n "${RUN_ONLY:-}" ] && [ -f "$OUT/run.meta.json" ]; then
  echo "run.meta.json kept (RUN_ONLY top-up)"
else
{
  echo "{"
  echo " \"started_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo " \"commit\": \"$(jj log -r @ --no-graph -T 'commit_id' 2>/dev/null || git -C "$REPO" rev-parse HEAD)\","
  echo " \"rustc\": \"$(rustc -V)\","
  echo " \"sizes\": \"$SIZES\", \"rounds\": $ROUNDS,"
  echo " \"cpu_1t\": \"$CPU_1T\", \"cpu_mt\": \"$CPU_MT\", \"threads_mt\": $THREADS_MT,"
  echo " \"bin_plain\": \"$(sha256sum "$BIN_PLAIN" | cut -d' ' -f1)\","
  echo " \"bin_rayon\": \"${BIN_RAYON:+$(sha256sum "$BIN_RAYON" | cut -d' ' -f1)}\""
  echo "}"
} > "$OUT/run.meta.json"
fi

# Revision-1 process: the named-profile generations and the two peer metrics.
run_one 1t-rev1 "$BIN_PLAIN" 1 "$CPU_1T" 1 "$REV1_ARMS"
# Revision-3 process: the two frozen ensembles, plus the anchor.
run_one 1t-rev3 "$BIN_PLAIN" 1 "$CPU_1T" 3 "$ENS_ARMS"

# Every threaded row uses the `ssim2-rayon` build, so fast-ssim2's Gaussian
# blur is threaded too and the anchor is a threaded opponent rather than a
# single-threaded one wearing an MT label.
if [ -n "$BIN_RAYON" ]; then
  run_one mt16-rev1 "$BIN_RAYON" "$THREADS_MT" "$CPU_MT" 1 "$REV1_ARMS"
  run_one mt16-rev3 "$BIN_RAYON" "$THREADS_MT" "$CPU_MT" 3 "$ENS_ARMS"
  run_one 4t-rev1 "$BIN_RAYON" 4 "$CPU_4T" 1 "$REV1_ARMS_MT"
  run_one 4t-rev3 "$BIN_RAYON" 4 "$CPU_4T" 3 "$ENS_ARMS"
  run_one 8t-rev1 "$BIN_RAYON" 8 "$CPU_8T" 1 "$REV1_ARMS_MT"
  run_one 8t-rev3 "$BIN_RAYON" 8 "$CPU_8T" 3 "$ENS_ARMS"
fi

echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
