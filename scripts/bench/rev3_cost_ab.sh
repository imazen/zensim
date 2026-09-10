#!/usr/bin/env bash
# Paired cost A/B for the arithmetic-revision axis (issue #61).
#
# `ssim_form::active_revision` is a process-global `OnceLock`, so revision 1
# and revision 3 cannot be interleaved as two arms inside one zenbench group
# the way `extract_paths_bench` interleaves its extraction families. This
# driver does the next best thing and the one this repo already sanctions:
#
#   * ONE binary, built once, used for every block — a rebuild alone has moved
#     a 2304^2 timing ~10% here (benchmarks/era2_perf_break_2026-08-31.md
#     §22.5), so an A/B across two builds is not a measurement.
#   * `ZENSIM_FORMULA_REV=1` and `=3` are the SAME BYTE LENGTH, which is why
#     `active_revision` accepts single digits: an environment block's size
#     shifts binary layout.
#   * Blocks ALTERNATE A B A B ..., so thermal/turbo/neighbour drift is shared
#     rather than accumulated onto whichever revision ran second.
#   * `fast_ssim2` is in every block and is revision-independent, so it is the
#     cross-block anchor: if it moves between blocks, the box moved and the
#     zensim deltas in those blocks are not comparable.
#
# One pinned worker, plain release, no target-cpu=native.
#
# Do NOT wrap this in `run-heavy`: its cgroup scope + job caps make zenbench's
# own resource gating refuse the run (MEASURED 2026-09-09 — the group aborted
# right after the calibration header). The workload is already one core at
# idle CPU/IO priority via `taskset` + `nice -n19 ionice -c3`, which is what
# the machine-safety rule asks for.
#
# Usage: scripts/bench/rev3_cost_ab.sh <out-dir> [blocks] [sizes] [cpu]
#
# TWO-BINARY MODE (`BIN_A=<path> BIN_B=<path>`): the arms are two builds at
# the DEFAULT revision instead of two revisions of one build — for "what did
# the default path pay between commit X and commit Y". Everything above about
# one-binary purity still applies inside each arm; between arms the only thing
# that may differ is the code, so: give both binaries paths of the SAME LENGTH
# (argv[0] sits on the stack like the environment does — copy them to
# `~/tmp/ab/bin_A` / `bin_B`), and read the anchor first — `fast_ssim2` is
# compiled into both binaries and must NOT move between them, or the
# difference is the build, not the change.
set -euo pipefail

OUT="${1:?usage: rev3_cost_ab.sh <out-dir> [blocks] [sizes] [cpu]}"
BLOCKS="${2:-2}"
SIZES="${3:-1024,2048}"
CPU="${4:-8}"
ROUNDS="${ZEN_XP_ROUNDS:-30}"
# Threads per arm. 1 = the single pinned worker (streaming path). For the
# threaded path set THREADS=8 and give taskset a matching CPU range via CPU=8-15.
THREADS="${THREADS:-1}"

mkdir -p "$OUT"
if [ -n "${BIN_A:-}" ] || [ -n "${BIN_B:-}" ]; then
  : "${BIN_A:?two-binary mode needs BOTH BIN_A and BIN_B}"
  : "${BIN_B:?two-binary mode needs BOTH BIN_A and BIN_B}"
  if [ "${#BIN_A}" -ne "${#BIN_B}" ]; then
    echo "BIN_A and BIN_B paths differ in length (${#BIN_A} vs ${#BIN_B}); copy them to equal-length names first" >&2
    exit 2
  fi
  ARMS="A B"
  echo "binary A: $BIN_A"; echo "binary B: $BIN_B"
  { sha256sum "$BIN_A"; sha256sum "$BIN_B"; } | tee "$OUT/binary.sha256"
else
  ARMS="1 3"
  BIN=$(ls -t target/release/deps/extract_paths_bench-* 2>/dev/null | grep -v '\.d$' | head -1)
  if [ -z "${BIN:-}" ]; then
    echo "build it first: cargo bench --no-run --bench extract_paths_bench -p zensim \\" >&2
    echo "  --features custom-profiles,feature-regime-v2,threads,training" >&2
    exit 2
  fi
  echo "binary: $BIN"
  sha256sum "$BIN" | tee "$OUT/binary.sha256"
fi
{
  echo "host: $(uname -srm) $(hostname)"
  echo "cpu_pinned: $CPU"
  echo "threads: $THREADS"
  echo "sizes: $SIZES"
  echo "rounds_min: $ROUNDS"
  echo "blocks_per_revision: $BLOCKS"
  echo "arms: $ARMS"
  echo "formula_rev_env: ${ZENSIM_FORMULA_REV:-default}"  # two-binary mode inherits it; one-binary mode overrides per arm
  echo "commit: $(jj log -r @- --no-graph -T 'commit_id' 2>/dev/null || git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$OUT/run.meta"

for b in $(seq 1 "$BLOCKS"); do
  for arm in $ARMS; do
    log="$OUT/block${b}_rev${arm}.txt"
    echo "=== block $b arm $arm -> $log ==="
    case "$arm" in
      A) ZEN_XP_SIZES="$SIZES" ZEN_XP_ROUNDS="$ROUNDS" RAYON_NUM_THREADS="$THREADS" \
           taskset -c "$CPU" nice -n19 ionice -c3 "$BIN_A" >"$log" 2>&1 ;;
      B) ZEN_XP_SIZES="$SIZES" ZEN_XP_ROUNDS="$ROUNDS" RAYON_NUM_THREADS="$THREADS" \
           taskset -c "$CPU" nice -n19 ionice -c3 "$BIN_B" >"$log" 2>&1 ;;
      *) ZEN_XP_SIZES="$SIZES" ZEN_XP_ROUNDS="$ROUNDS" \
         RAYON_NUM_THREADS="$THREADS" ZENSIM_FORMULA_REV="$arm" \
           taskset -c "$CPU" nice -n19 ionice -c3 "$BIN" >"$log" 2>&1 ;;
    esac
    echo "  done: $(grep -c . "$log") lines"
  done
done
echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/run.meta"
