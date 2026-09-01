#!/usr/bin/env bash
# hybrid_speed.sh — the AMENDED-W4 speed matrix
# (benchmarks/hybrid_candidate_2026-09-01.md §1, APPENDIX B of the ssim2 exam).
#
# W4 now binds at BOTH 1 and 8 threads, and each candidate is priced on ITS OWN
# extraction regime plus ITS OWN forwards (an ensemble = one extraction + every
# member's forward). The arms live in `zensim-bench/benches/ssim2_speed_bar.rs`
# so the opponent and every candidate are interleaved by zenbench inside ONE
# process — era-2 §22.5's protocol item 1, the strongest form.
#
# Estimator, on top of that: N process starts with ASLR ON, CCD0-pinned
# (`taskset -c 0` at 1T, `-c 0-7,16-23` at 8T — the 9950X3D's 96 MiB-L3 die),
# `min` over starts per arm. The opponent's `rayon` feature is a per-BUILD
# cargo feature, so the matrix runs BOTH builds; the amended clause reads the
# 8 T column off the rayon build, and `zensim_B` is the unchanged cross-build
# anchor that says whether the box moved between them.
#
#   $0 build          build both binaries
#   $0 run [starts]   run the matrix (default 7 starts per cell)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${HY_OUT:-/mnt/v/output/zensim/hybrid-2026-09-01}/speed
TGT=${HY_BENCH_TARGET:-$REPO/target-bench}
export ZEN_HY_MLP=${ZEN_HY_MLP:-/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin}
export ZEN_HY_LIN=${ZEN_HY_LIN:-/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.2_a0.2_b0.97.bin}
mkdir -p "$OUT"

bin_for() {   # bin_for plain|rayon
    local d=$TGT; [ "$1" = rayon ] && d=$TGT-rayon
    ls -t "$d"/release/deps/ssim2_speed_bar-* 2>/dev/null | grep -v '\.d$' | head -1
}

case "${1:-run}" in
build)
    ( cd "$REPO/zensim-bench" && CARGO_TARGET_DIR=$TGT nice -n19 ionice -c3 \
        cargo build --release --bench ssim2_speed_bar )
    ( cd "$REPO/zensim-bench" && CARGO_TARGET_DIR=$TGT-rayon nice -n19 ionice -c3 \
        cargo build --release --bench ssim2_speed_bar --features ssim2-rayon )
    for v in plain rayon; do printf '%s %s\n' "$v" "$(sha256sum "$(bin_for $v)" | cut -c1-16)"; done ;;
run)
    starts=${2:-7}
    export ZEN_S2_SIZES=${ZEN_S2_SIZES:-576,1152,2304}
    export ZEN_S2_ROUNDS=${ZEN_S2_ROUNDS:-10} ZEN_S2_MIN_ROUNDS=${ZEN_S2_MIN_ROUNDS:-5}
    export ZEN_S2_WALL_S=${ZEN_S2_WALL_S:-15}
    for v in plain rayon; do
      b=$(bin_for $v); [ -n "$b" ] || { echo "no $v binary — run '$0 build'" >&2; exit 2; }
      for t in 1 8; do
        pin=0; [ "$t" = 8 ] && pin=0-7,16-23
        for i in $(seq 1 "$starts"); do
          f="$OUT/s2_${v}_${t}t_start${i}.txt"
          RAYON_NUM_THREADS=$t nice -n19 ionice -c3 taskset -c "$pin" "$b" \
              --format=csv >"$f" 2>"$f.err" || { echo "FAILED $f" >&2; tail -3 "$f.err" >&2; exit 3; }
          echo "  $v ${t}T start $i -> $f"
        done
      done
    done ;;
*) echo "usage: $0 {build|run [starts]}" >&2; exit 2 ;;
esac
