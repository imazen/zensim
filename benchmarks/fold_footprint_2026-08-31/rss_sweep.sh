#!/usr/bin/env bash
# Peak-RSS sweep for the fold-footprint model
# (`benchmarks/fold_footprint_2026-08-31.md`).
#
# One process per (arm, size, threads) cell — `/usr/bin/time -v` maximum
# resident set size, which is the only memory evidence this repo accepts
# alongside heaptrack. The bench binary's `ZEN_FE_RSS` single-arm loop is the
# owner of the workload; this script only drives it and shapes the TSV.
#
#   FE_BIN=<path to fold_engine_bench binary> ./rss_sweep.sh <out.tsv> [sizes] [threads] [arms]
#
# Columns: arm, size, threads, iters, peak_kib, input_kib, workingset_kib, load1
set -uo pipefail
BIN="${FE_BIN:?set FE_BIN to the fold_engine_bench binary}"
OUT="${1:?out.tsv}"
SIZES="${2:-512 768 1152 1536 2048 2304 3072}"
THREADS="${3:-1 8 16}"
ARMS="${4:-score_buffered score_fold}"
ITERS="${ZEN_FE_ITERS:-5}"

printf 'arm\tsize\tthreads\titers\tpeak_kib\tinput_kib\tworkingset_kib\tload1\n' > "$OUT"
for size in $SIZES; do
  # Both `test_pair` images: w*h*3 bytes each, two of them.
  input_kib=$(( size * size * 6 / 1024 ))
  for th in $THREADS; do
    for arm in $ARMS; do
      log=$(ZEN_FE_RSS="$arm" ZEN_FE_SIZE="$size" ZEN_FE_ITERS="$ITERS" \
            RAYON_NUM_THREADS="$th" /usr/bin/time -v "$BIN" 2>&1)
      peak=$(printf '%s\n' "$log" | awk '/Maximum resident set size/ {print $NF}')
      load=$(awk '{print $1}' /proc/loadavg)
      if [ -z "$peak" ]; then
        printf '%s\t%s\t%s\t%s\tFAIL\t%s\t\t%s\n' "$arm" "$size" "$th" "$ITERS" "$input_kib" "$load" >> "$OUT"
        printf '%s\n' "$log" | tail -5 >&2
        continue
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$arm" "$size" "$th" "$ITERS" "$peak" "$input_kib" "$(( peak - input_kib ))" "$load" >> "$OUT"
      printf '%-16s %5s %2sT peak=%8s KiB wset=%8s KiB load=%s\n' \
        "$arm" "$size" "$th" "$peak" "$(( peak - input_kib ))" "$load"
    done
  done
done
