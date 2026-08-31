#!/bin/bash
# Branch-behaviour matrix: perf stat over the fold/buffered walks at three
# sizes x four width classes x {1,16}T. Emits TSV.
set -u
B="${ZB_BIN:?}"
OUT="${ZB_OUT:?}"
PERF=/usr/bin/perf
EV="cycles,instructions,branches,branch-misses"

printf 'arm\tw\th\tthreads\titers\tcycles\tinstructions\tbranches\tbranch_misses\tsec\tload1\n' > "$OUT"

run() {
  local arm="$1" w="$2" h="$3" t="$4" it="$5"
  local load; load=$(cut -d' ' -f1 /proc/loadavg)
  local tmp; tmp=$(mktemp ~/tmp/blurstudy/perfXXXX.txt)
  RAYON_NUM_THREADS="$t" ZEN_XP_RSS="$arm" ZEN_XP_W="$w" ZEN_XP_H="$h" ZEN_XP_ITERS="$it" \
    "$PERF" stat -x, -e "$EV" -o "$tmp" "$B" >/dev/null 2>&1
  local cy ins br bm sec
  cy=$(awk -F, '$3=="cycles"{print $1}' "$tmp")
  ins=$(awk -F, '$3=="instructions"{print $1}' "$tmp")
  br=$(awk -F, '$3=="branches"{print $1}' "$tmp")
  bm=$(awk -F, '$3=="branch-misses"{print $1}' "$tmp")
  sec=$(awk '/seconds time elapsed/{print $1}' "$tmp")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$w" "$h" "$t" "$it" "$cy" "$ins" "$br" "$bm" "$sec" "$load" >> "$OUT"
  rm -f "$tmp"
}

for arm in fold944_full fold372_full buf_v1_372; do
  for t in 1 16; do
    # square sizes, ITERS sized for ~3-4 s of work
    run "$arm" 576  576  "$t" 250
    run "$arm" 1152 1152 "$t" 60
    run "$arm" 2304 2304 "$t" 15
    # width classes at height 2304 (mod-8 / mod-16 coverage)
    run "$arm" 2296 2304 "$t" 15   # 8| not 16|
    run "$arm" 2303 2304 "$t" 15   # 7 mod 8
    run "$arm" 2297 2304 "$t" 15   # 1 mod 8
    # process baseline (0 iters) at the biggest size, to bound setup cost
    run "$arm" 2304 2304 "$t" 0
  done
done
echo "MATRIX-DONE"
