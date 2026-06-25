#!/usr/bin/env bash
# Heaptrack PROCESS-PEAK for the zensim CPU feature path that commit 4bb5febf
# called "the per-image memory hog that OOMs large frames".
#
# Measures three public entries at 1/4/12/30 MP:
#   score  — Zensim::compute                        (228 streaming score; cpu_profile's baseline)
#   ext372 — Zensim::compute_extended_features      (372/WithIw; the sweep --feature-output path, NEVER heaptracked before)
#   strip  — Zensim::compute_streaming_strips_default (228 via strip aggregation; the >16MP path)
#
# Driver: zensim-bench/examples/peak_entry.rs (replicates the sweep's exact
# StridedBytes + latest_preview + compute_extended_features call).
#
# Parse logic + bytes(1024) conversion lifted verbatim from
# zenmetrics/benchmarks/heaptrack/refresh_2026-05-28/run_heaptrack_sweep.sh
# so numbers are comparable to that TSV.
set -uo pipefail
ROOT=/home/lilith/work/zen/zensim
BIN="$ROOT/zensim-bench/target/release/examples/peak_entry"
OUT="$ROOT/benchmarks/heaptrack/feature_path_2026-06-25"
TSV="$OUT/peaks.tsv"
TRACE=/tmp/zsfeat_traces
mkdir -p "$OUT" "$TRACE"
[[ -x "$BIN" ]] || { echo "MISSING BIN: $BIN"; exit 1; }

to_bytes(){ local h="$1" num unit; num=$(echo "$h"|grep -oE '^[0-9.]+'); unit=$(echo "$h"|grep -oE '[KMGTB]$')
  case "$unit" in
    K) awk -v n="$num" 'BEGIN{printf "%d",n*1024}';;
    M) awk -v n="$num" 'BEGIN{printf "%d",n*1024*1024}';;
    G) awk -v n="$num" 'BEGIN{printf "%d",n*1024*1024*1024}';;
    T) awk -v n="$num" 'BEGIN{printf "%d",n*1024*1024*1024*1024}';;
    B|"") awk -v n="$num" 'BEGIN{printf "%d",n}';;
    *) echo 0;;
  esac; }

git -C "$ROOT" rev-parse --short HEAD > "$OUT/commit.txt" 2>/dev/null
printf 'entry\tsize_label\tw\th\tpixels\tpeak_heap_bytes\tpeak_heap_human\tbytes_per_px\tpeak_rss_human\tscore\tnfeat\n' > "$TSV"
for entry in score ext372 strip; do
  for sz in "1MP 1024 1024" "4MP 2048 2048" "12MP 4000 3000" "30MP 6000 5000"; do
    read -r label w h <<< "$sz"
    pfx="$TRACE/ht_${entry}_${label}"
    rm -f "$pfx".*
    echo "=== $entry $label ${w}x${h} ==="
    RUN=$(heaptrack -o "$pfx" "$BIN" "$entry" "$w" "$h" 2>&1); rc=$?
    score=$(echo "$RUN"|grep -oE 'score=[0-9.eE+-]+'|head -1|sed 's/score=//')
    nfeat=$(echo "$RUN"|grep -oE 'nfeat=[0-9]+'|head -1|sed 's/nfeat=//')
    tr=$(ls -t "$pfx".* 2>/dev/null|head -1)
    PR=$(heaptrack_print "$tr" 2>/dev/null)
    peak_h=$(echo "$PR"|grep 'peak heap memory consumption'|head -1|sed -E 's/.*consumption: *//')
    rss_h=$(echo "$PR"|grep 'peak RSS'|head -1|sed -E 's/.*: *//')
    pb=$(to_bytes "$peak_h"); px=$((w*h))
    bpp=$(awk -v b="$pb" -v p="$px" 'BEGIN{if(p>0)printf "%.1f",b/p; else print "NA"}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$entry" "$label" "$w" "$h" "$px" "$pb" "$peak_h" "$bpp" "$rss_h" "${score:--}" "${nfeat:--}" >> "$TSV"
    echo "  rc=$rc peak=$peak_h (${pb}B) -> ${bpp} B/px  score=${score:--} nfeat=${nfeat:--}"
  done
done
echo "DONE -> $TSV"
