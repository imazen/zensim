#!/bin/bash
set -u
ZENSIM_TARGET="/home/lilith/work/zen/zensim--productionize-v6/target/release/zensim-target"
ZEN_METRICS="/home/lilith/work/zen/zenmetrics/target/release/zen-metrics"

IMAGES=(
  "kadid_I05:/home/lilith/work/codec-eval/codec-corpus/kadid10k/I05.png"
  "kadid_I12:/home/lilith/work/codec-eval/codec-corpus/kadid10k/I12.png"
  "kadid_I25:/home/lilith/work/codec-eval/codec-corpus/kadid10k/I25.png"
  "kadid_I40:/home/lilith/work/codec-eval/codec-corpus/kadid10k/I40.png"
  "kadid_I55:/home/lilith/work/codec-eval/codec-corpus/kadid10k/I55.png"
  "kadid_I70:/home/lilith/work/codec-eval/codec-corpus/kadid10k/I70.png"
  "gb82_gui:/home/lilith/work/codec-eval/codec-corpus/gb82-sc/gui.png"
  "gb82_codec_wiki:/home/lilith/work/codec-eval/codec-corpus/gb82-sc/codec_wiki.png"
  "gb82_terminal:/home/lilith/work/codec-eval/codec-corpus/gb82-sc/terminal.png"
  "gb82_imac_dark:/home/lilith/work/codec-eval/codec-corpus/gb82-sc/imac_dark.png"
)

declare -A EXT=([zenjpeg]=jpg [zenwebp]=webp [zenavif]=avif [zenjxl]=jxl)
CODECS=(zenjpeg zenwebp zenavif zenjxl)
TARGET=63
TMP=/tmp/zensim_v6_demo
mkdir -p "$TMP/encoded"

printf "image\tcodec\ttarget\tachieved\tknob\tbytes\titers\tconverged\tbutter_pnorm3\n" > "$TMP/results_full.tsv"
for entry in "${IMAGES[@]}"; do
  label=${entry%%:*}
  path=${entry##*:}
  for codec in "${CODECS[@]}"; do
    enc_out="$TMP/encoded/${label}_${codec}.${EXT[$codec]}"
    out=$("$ZENSIM_TARGET" "$path" --codec "$codec" --target "$TARGET" --quiet -o "$enc_out" 2>&1 | grep "^codec=" | head -1)
    achieved=$(echo "$out" | sed -E 's/.*achieved=([0-9.]+).*/\1/')
    knob=$(echo "$out" | sed -E 's/.*knob=([0-9.]+).*/\1/')
    bytes=$(echo "$out" | sed -E 's/.*bytes=([0-9]+).*/\1/')
    iters=$(echo "$out" | sed -E 's/.*iters=([0-9]+).*/\1/')
    converged=$(echo "$out" | sed -E 's/.*converged=(true|false).*/\1/')

    score_out=$("$ZEN_METRICS" score --metric butteraugli --reference "$path" --distorted "$enc_out" 2>&1 | tr ' ' '\n' | grep "butteraugli_pnorm3=" | head -1)
    pnorm3=${score_out#butteraugli_pnorm3=}
    printf "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" "$label" "$codec" "$TARGET" "$achieved" "$knob" "$bytes" "$iters" "$converged" "$pnorm3" >> "$TMP/results_full.tsv"
    echo "$label $codec -> ach=$achieved knob=$knob pnorm3=$pnorm3 conv=$converged"
  done
done
echo DONE
