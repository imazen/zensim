#!/usr/bin/env bash
# The FEATURE-EXTRACTION half of the speed matrix: the fold/buffered extraction
# families across thread counts and both arithmetic revisions.
#
# `zensim/benches/extract_paths_bench.rs` is the existing owner of this
# question and this only drives it; the arm set is whatever that bench defines
# (`buf_v1_228`, `buf_v1_372`, `fold156_basic`, `fold228_peaks`,
# `fold228_moments`, `fold228_classc`, `fold372_full`, `fold944_off`,
# `fold944_full`, and the `fast_ssim2` anchor). There is NO arm for the y60 /
# coarse-Y plan the revision-3 fast ensemble reads — that regime is planned
# from the bake's declared feature IDs at serve time and is not one of this
# bench's enumerated walks. The closest measurement of it is the end-to-end
# `rev3_fast_y60_ens5` arm in the ssim2_speed_bar half of this matrix. Do not
# read any fold* arm here as "the y60 extraction".
#
# Pinning follows `benchmarks/k4_st_mt_2026-09-10.md` so the numbers are
# comparable to that record: 1T = cpu 8, 4T = cpus 8-11, 8T = cpus 8-15 (all
# one CCD, so the 1/4/8 scaling column is within one L3), 16T = cpus 0-15
# (both CCDs — a different cache regime, not just more cores).
#
# Both revisions are run because `ssim_form::active_revision` is process-global
# and revision 3 took fused-kernel work after the September 10 record, which
# was revision 1 only.
#
# Not wrapped in run-heavy, for the same reason as the rest of this matrix.
#
# Usage: scripts/demos/speed_matrix_extract.sh <out-dir>
set -euo pipefail

OUT="${1:?usage: speed_matrix_extract.sh <out-dir>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIZES="${SIZES:-256,1024,2048,4096}"
ROUNDS="${ROUNDS:-16}"
MIN_ROUNDS="${MIN_ROUNDS:-12}"
WALL_S="${WALL_S:-240}"

mkdir -p "$OUT"
BIN_XP="${BIN_XP:-$(ls -t "$REPO"/target/release/deps/extract_paths_bench-* 2>/dev/null | grep -v '\.d$' | head -1)}"
: "${BIN_XP:?build first: cargo bench --no-run --locked --bench extract_paths_bench -p zensim --features custom-profiles,feature-regime-v2,threads,training}"
echo "extract binary: $BIN_XP"
sha256sum "$BIN_XP" | tee "$OUT/extract-binary.sha256"

for rev in 1 3; do
  for cfg in "1:8" "4:8-11" "8:8-15" "16:0-15"; do
    threads=${cfg%%:*}; cpus=${cfg##*:}
    tag="extract-${threads}t-rev${rev}"
    # `RUN_ONLY=extract-4t-rev1,...` re-measures named configurations without
    # discarding the rest. This exists because the anchor check earns it: when
    # `fast_ssim2`'s median in a thread column disagrees with its 1-thread
    # median, that column's process was contended and has to be redone, and
    # redoing the whole matrix to fix one column wastes an hour.
    if [ -n "${RUN_ONLY:-}" ] && ! printf '%s' ",$RUN_ONLY," | grep -q ",$tag,"; then
      echo "=== $tag: skipped (RUN_ONLY) ==="
      continue
    fi
    rm -f "$OUT/$tag.json"
    echo "=== $tag: rev=$rev threads=$threads cpus=$cpus ==="
    uptime | tee "$OUT/$tag.load-before.txt"
    env ZENSIM_FORMULA_REV="$rev" \
        ZEN_XP_SIZES="$SIZES" ZEN_XP_ROUNDS="$ROUNDS" \
        ZEN_XP_MIN_ROUNDS="$MIN_ROUNDS" ZEN_XP_WALL_S="$WALL_S" \
        RAYON_NUM_THREADS="$threads" ZENBENCH_RESULT_PATH="$OUT/$tag.json" \
        taskset -c "$cpus" nice -n19 ionice -c3 "$BIN_XP" >"$OUT/$tag.log" 2>&1
    uptime | tee "$OUT/$tag.load-after.txt"
  done
done
echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
