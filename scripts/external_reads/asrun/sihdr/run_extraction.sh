#!/bin/bash
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/sihdr-transfer-2026-07-29/run_extraction.sh
# sha256(source): 690c36491db45d6805563580ecf21cfc4e5f467d8c47a5b1a3177c2eebaceb5f
# build_commit:  34cbd9cf03673c48d69127b7c648bc2fd7d95adc
# Protocol doc:  benchmarks/sihdr_transfer_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
# SI-HDR full extraction — 12 method x clip batches, disk-constrained.
# Progress streams to stdout (tee'd by caller into ~/tmp/sihdr-extract.log).
set -u
W=~/tmp/sihdr-work
Z=/mnt/tower/input/datasets/si-hdr
OUT=/mnt/v/output/zensim/sihdr-transfer-2026-07-29
EX=$W/sihdr_features_extract
cd "$W" || exit 1

echo "[$(date -u +%H:%M:%S)] extracting reference.zip (181 EXR)"
mkdir -p ref
nice -n19 ionice -c3 unzip -o -q -j "$Z/reference.zip" 'sihdr/reference/*.exr' -d ref/ || exit 1
echo "[$(date -u +%H:%M:%S)] refs ready: $(ls ref | wc -l) files, $(du -sh ref | cut -f1)"

mkdir -p feats
for m in drtmo expandnet hdrcnn hdrgan maskhdr singlehdr; do
  for c in 95 97; do
    tag="${m}_${c}"
    if [ -s "feats/feats_${tag}.csv" ]; then
      echo "[$(date -u +%H:%M:%S)] SKIP $tag (already done)"
      continue
    fi
    echo "[$(date -u +%H:%M:%S)] batch $tag: unzip"
    rm -rf batch; mkdir -p batch
    nice -n19 ionice -c3 unzip -o -q -j "$Z/reconstructions.zip" \
      "sihdr/reconstructions/$m/clip_$c/*.exr" -d batch/ || { echo "UNZIP FAIL $tag"; exit 1; }
    n=$(ls batch | wc -l)
    : > batch.tsv
    for f in batch/*.exr; do
      id=$(basename "$f" .exr)
      if [ ! -f "ref/$id.exr" ]; then echo "NO_REF $tag $id"; continue; fi
      printf 'i%s-%s-%s\t%s\t%s\t%s\n' "$id" "$c" "$m" "$c" "$W/ref/$id.exr" "$W/$f" >> batch.tsv
    done
    echo "[$(date -u +%H:%M:%S)] batch $tag: $n files, $(wc -l < batch.tsv) pairs, extracting"
    RAYON_NUM_THREADS=20 nice -n19 ionice -c3 "$EX" \
      --manifest batch.tsv --mode 944 --out "feats/feats_${tag}.csv" \
      2> "feats/stderr_${tag}.log"
    rows=$(($(wc -l < "feats/feats_${tag}.csv") - 1))
    echo "[$(date -u +%H:%M:%S)] batch $tag: $rows rows done; stderr: $(grep -c -E 'FAIL|SKIP' "feats/stderr_${tag}.log" || true) fail/skip, $(grep -c CROP "feats/stderr_${tag}.log" || true) crops"
    rm -rf batch batch.tsv
  done
done

echo "[$(date -u +%H:%M:%S)] concatenating"
head -1 feats/feats_drtmo_95.csv > "$OUT/sihdr_feats_944.csv"
for f in feats/feats_*.csv; do tail -n +2 "$f" >> "$OUT/sihdr_feats_944.csv"; done
echo "[$(date -u +%H:%M:%S)] total rows: $(($(wc -l < "$OUT/sihdr_feats_944.csv") - 1))"
sha256sum "$OUT/sihdr_feats_944.csv"
echo "[$(date -u +%H:%M:%S)] DONE"
