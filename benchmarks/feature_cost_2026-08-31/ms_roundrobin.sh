#!/usr/bin/env bash
# 1T wall-clock for the model-class arms, using the bench binary's own
# ZEN_XP_RSS single-arm loop (which takes NO zenbench lock, so it does not
# jump another lane's queue).
#
# Fixed overhead (process start, test_pair construction, first-touch page
# faults) is removed by a TWO-POINT subtraction: t(N) - t(1) over N-1 extra
# iterations. Arms are visited ROUND-ROBIN so any drift in the box's
# background load spreads across all of them equally instead of landing on
# whichever one ran last.
set -uo pipefail
BIN="${1:?bench binary}"
OUT="${2:?out tsv}"
ROUNDS="${ROUNDS:-5}"
ARMS="buf_v1_228 buf_v1_372 fold156_basic fold228_peaks fold372_full fold944_off fold944_full"
printf 'round\tsize\tarm\titers\tt1_s\ttN_s\tms_per_iter\n' > "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for sz in 576 1152 2304; do
    case $sz in 576) N=40;; 1152) N=16;; *) N=6;; esac
      N=$(( N * ${ITER_MULT:-1} ))
    for a in $ARMS; do
      t1=$( { /usr/bin/time -f %e env ZEN_XP_RSS=$a ZEN_XP_SIZE=$sz ZEN_XP_ITERS=1 "$BIN" >/dev/null; } 2>&1 | tail -1 )
      tN=$( { /usr/bin/time -f %e env ZEN_XP_RSS=$a ZEN_XP_SIZE=$sz ZEN_XP_ITERS=$N "$BIN" >/dev/null; } 2>&1 | tail -1 )
      ms=$(awk -v a="$t1" -v b="$tN" -v n="$N" 'BEGIN{printf "%.3f",(b-a)*1000.0/(n-1)}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$sz" "$a" "$N" "$t1" "$tN" "$ms" >> "$OUT"
    done
  done
  echo "round $r done $(date -u +%H:%M:%S) load=$(awk '{print $1}' /proc/loadavg)"
done
printf 'done\n' > "${OUT}.COMPLETE"
