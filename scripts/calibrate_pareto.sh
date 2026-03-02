#!/bin/bash
# Encode corpus images with zenjpeg AND mozjpeg-rs at q0-q100 step 5,
# plus libjpeg-turbo at standard quality levels.
# Captures file sizes + all quality metrics for Pareto analysis.

set -euo pipefail

ENCODER_BENCH="/tmp/encoder-bench/target/release/encoder-bench"
ZENSIM="/home/lilith/work/zensim/target/release/zensim-cli"
SSIM2="/home/lilith/work/jxl-efforts/libjxl/build/tools/ssimulacra2"
BA="/home/lilith/work/jxl-efforts/libjxl/build/tools/butteraugli_main"
DSSIM="dssim"
OUTDIR="/mnt/v/output/zensim/pareto"
TMPDIR_BASE="/tmp/pareto-cal"

rm -rf "$TMPDIR_BASE"
mkdir -p "$OUTDIR" "$TMPDIR_BASE"

measure_corpus() {
    local corpus_name="$1"
    local corpus_dir="$2"
    local csv="$OUTDIR/${corpus_name}.csv"
    local decoded_dir="$TMPDIR_BASE/$corpus_name"
    local flatdir="$TMPDIR_BASE/${corpus_name}_flat"
    local sizes_csv="$TMPDIR_BASE/${corpus_name}_sizes.csv"

    mkdir -p "$decoded_dir" "$flatdir"

    # Flatten source PNGs to RGB
    echo "Flattening $corpus_name sources to RGB..."
    for f in "$corpus_dir"/*.png; do
        local base=$(basename "$f" .png)
        convert "$f" -background white -flatten -type TrueColor -depth 8 "$flatdir/${base}.png" 2>/dev/null
    done

    # Encode with zenjpeg + mozjpeg-rs (Rust tool, captures sizes)
    echo "Encoding $corpus_name with zenjpeg + mozjpeg-rs..."
    $ENCODER_BENCH "$flatdir" "$decoded_dir" > "$sizes_csv" 2>&1

    # Also encode with libjpeg-turbo
    echo "Encoding $corpus_name with libjpeg-turbo..."
    for f in "$flatdir"/*.png; do
        local base=$(basename "$f" .png)
        local ppm="$TMPDIR_BASE/${base}.ppm"
        convert "$f" "$ppm" 2>/dev/null

        for q in 5 10 15 20 25 30 40 50 60 70 75 80 85 90 95 98; do
            local jpg="$TMPDIR_BASE/${base}_q${q}.jpg"
            cjpeg -quality "$q" -outfile "$jpg" "$ppm" 2>/dev/null
            local sz=$(wc -c < "$jpg")
            local decoded="$decoded_dir/libjpeg_${base}__q$(printf '%03d' $q).png"
            djpeg -bmp "$jpg" 2>/dev/null | convert bmp:- "$decoded" 2>/dev/null
            # Get dimensions
            local dims=$(identify -format "%w,%h" "$decoded" 2>/dev/null)
            echo "libjpeg,$base,$q,$sz,$dims" >> "$sizes_csv"
            rm -f "$jpg"
        done
        rm -f "$ppm"
    done

    # Now measure metrics on all decoded PNGs
    echo "encoder,image,quality,jpeg_size,width,height,zensim_score,zensim_raw,ssim2,butteraugli,dssim" > "$csv"

    local count=0
    while IFS=, read -r encoder img_name q sz iw ih; do
        # Skip header
        [[ "$encoder" == "encoder" ]] && continue

        local decoded="$decoded_dir/${encoder}_${img_name}__q$(printf '%03d' $q).png"
        local src="$flatdir/${img_name}.png"

        [[ ! -f "$decoded" ]] && continue
        [[ ! -f "$src" ]] && continue

        # zensim
        local zout zs zr
        zout=$($ZENSIM "$src" "$decoded" 2>/dev/null) || zout="score: NaN raw_distance: NaN"
        zs=$(echo "$zout" | awk '{print $2}')
        zr=$(echo "$zout" | awk '{print $4}')

        # ssimulacra2
        local s2
        s2=$($SSIM2 "$src" "$decoded" 2>/dev/null) || s2="NaN"

        # butteraugli
        local bav
        bav=$($BA "$src" "$decoded" 2>/dev/null | head -1) || bav="NaN"

        # dssim
        local ds
        ds=$($DSSIM "$src" "$decoded" 2>/dev/null | awk '{print $1}') || ds="NaN"

        echo "$encoder,$img_name,$q,$sz,$iw,$ih,$zs,$zr,$s2,$bav,$ds" >> "$csv"
        count=$((count + 1))
    done < "$sizes_csv"

    echo "  $corpus_name: $count measurements → $csv"

    # Clean up
    rm -rf "$decoded_dir" "$flatdir"
}

measure_corpus "cid22" "/mnt/v/GitHub/codec-corpus/CID22/CID22-512/validation"

echo ""
echo "=== Done ==="
for f in "$OUTDIR"/*.csv; do
    echo "  $(wc -l < "$f") rows in $(basename "$f")"
done

rm -rf "$TMPDIR_BASE"
