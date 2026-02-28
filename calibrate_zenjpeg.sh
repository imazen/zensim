#!/bin/bash
# Encode corpus images with zenjpeg at q0-q100 step 5,
# measure zensim, ssimulacra2, butteraugli, dssim.
# Output: one CSV per corpus with zenjpeg_q as the quality index.

set -euo pipefail

ZENJPEG_BENCH="/tmp/zenjpeg-bench/target/release/zenjpeg-bench"
ZENSIM="/home/lilith/work/zensim/target/release/zensim-cli"
SSIM2="/home/lilith/work/jxl-efforts/libjxl/build/tools/ssimulacra2"
BA="/home/lilith/work/jxl-efforts/libjxl/build/tools/butteraugli_main"
DSSIM="dssim"
OUTDIR="/mnt/v/output/zensim/calibration"
TMPDIR_BASE="/tmp/zenjpeg-cal"

rm -rf "$TMPDIR_BASE"
mkdir -p "$OUTDIR" "$TMPDIR_BASE"

measure_corpus() {
    local corpus_name="$1"
    local corpus_dir="$2"
    local csv="$OUTDIR/zenjpeg_${corpus_name}.csv"
    local decoded_dir="$TMPDIR_BASE/$corpus_name"
    local flatdir="$TMPDIR_BASE/${corpus_name}_flat"

    mkdir -p "$decoded_dir" "$flatdir"

    # Flatten source PNGs to RGB (handles RGBA, palette, etc)
    echo "Flattening $corpus_name sources to RGB..."
    for f in "$corpus_dir"/*.png; do
        local base=$(basename "$f" .png)
        convert "$f" -background white -flatten -type TrueColor -depth 8 "$flatdir/${base}.png" 2>/dev/null
    done

    # Encode all with zenjpeg (writes decoded PNGs to decoded_dir)
    echo "Encoding $corpus_name with zenjpeg..."
    $ZENJPEG_BENCH "$flatdir" "$decoded_dir" > "$TMPDIR_BASE/${corpus_name}_sizes.csv" 2>&1

    # Now measure metrics
    echo "corpus,image,zenjpeg_q,zensim_score,zensim_raw,ssim2,butteraugli,dssim" > "$csv"

    local count=0
    for decoded in "$decoded_dir"/*.png; do
        local fname=$(basename "$decoded" .png)
        # Parse: {image}__q{NNN}
        local img_name="${fname%__q*}"
        local q_str="${fname##*__q}"
        local q=$((10#$q_str))  # strip leading zeros

        local src="$flatdir/${img_name}.png"

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

        echo "$corpus_name,$img_name,$q,$zs,$zr,$s2,$bav,$ds" >> "$csv"
        count=$((count + 1))
    done

    echo "  $corpus_name: $count measurements → $csv"

    # Clean up decoded PNGs (large)
    rm -rf "$decoded_dir" "$flatdir"
}

measure_corpus "gb82-sc" "/mnt/v/GitHub/codec-corpus/gb82-sc"
measure_corpus "cid22" "/mnt/v/GitHub/codec-corpus/CID22/CID22-512/validation"
measure_corpus "clic2025" "/mnt/v/GitHub/codec-corpus/clic2025/final-test"

echo ""
echo "=== Done ==="
for f in "$OUTDIR"/zenjpeg_*.csv; do
    echo "  $(wc -l < "$f") rows in $(basename "$f")"
done

rm -rf "$TMPDIR_BASE"
