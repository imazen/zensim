#!/usr/bin/env bash
# Peak heap (heaptrack) of the extractor on the synthetic pairs, baseline (--full-gmsbank) vs
# candidate with the four families + prefix, 1 and 8 threads, 256/1024/2048/4096. Serialise through
# the shared lock: `flock ~/tmp/devin/heavy.lock nice -n 5 bash scripts/restore_cuts/memory.sh`.
set -euo pipefail
ROOT=/var/tmp/restore-cuts
export ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt
out=$ROOT/logs/memory.tsv
printf 'size\tthreads\tarm\tpeak_heap\tpeak_rss_heaptrack\twall_s\n' > "$out"
for n in 256 1024 2048 4096; do
  for t in 1 8; do
    for arm in base cand; do
      if [ $arm = base ]; then bin=$ROOT/bin/extract_base; extra=(--full-gmsbank); else
        bin=$ROOT/bin/extract_cand; extra=(--restore-cuts prefix,mapdev,z1max,gmsnative,dvifmgate); fi
      csv=$ROOT/synth/mem_${arm}_${n}_$t.csv; rm -f "$csv" "$csv".*
      ht=$ROOT/synth/ht_${arm}_${n}_$t
      start=$(date +%s.%N)
      RAYON_NUM_THREADS=$t heaptrack -o "$ht" "$bin" --corpus pairs-tsv --path "$ROOT/synth/pairs_$n.tsv" \
          --out "$csv" --input-contract legacy-rgb8 "${extra[@]}" > "$ROOT/logs/mem_${arm}_${n}_$t.log" 2>&1
      wall=$(echo "$(date +%s.%N) - $start" | bc)
      summ=$(heaptrack_print "$ht".zst 2>/dev/null || heaptrack_print "$ht".gz)
      peak=$(grep -m1 'peak heap memory consumption' <<<"$summ" | sed 's/.*: //')
      rss=$(grep -m1 'peak RSS' <<<"$summ" | sed 's/.*: //')
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$t" "$arm" "$peak" "$rss" "$wall" | tee -a "$out"
    done
  done
done
