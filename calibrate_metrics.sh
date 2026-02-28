#!/bin/bash
# Encode corpus images at various JPEG qualities with libjpeg-turbo,
# measure zensim, ssimulacra2, butteraugli, dssim on each.
# Output: one CSV per corpus.

set -euo pipefail

ZENSIM="/home/lilith/work/zensim/target/release/zensim-cli"
SSIM2="/home/lilith/work/jxl-efforts/libjxl/build/tools/ssimulacra2"
BA="/home/lilith/work/jxl-efforts/libjxl/build/tools/butteraugli_main"
DSSIM="dssim"
OUTDIR="/mnt/v/output/zensim/calibration"
TMPDIR_BASE="/tmp/zensim-cal"

QUALITIES=(5 10 15 20 25 30 40 50 60 70 75 80 85 90 95 98)

rm -rf "$TMPDIR_BASE"
mkdir -p "$OUTDIR" "$TMPDIR_BASE"

measure_one() {
    local img="$1"
    local corpus_name="$2"
    local csv="$3"
    local base=$(basename "$img" .png)
    local tmpdir="$TMPDIR_BASE/$corpus_name/$base"
    mkdir -p "$tmpdir"

    # Flatten to 8-bit RGB PNG (handles RGBA, palette, grayscale)
    local rgb="$tmpdir/rgb.png"
    convert "$img" -background white -flatten -type TrueColor -depth 8 "$rgb" 2>/dev/null

    # Convert to PPM for cjpeg (libjpeg-turbo can't read PNG)
    local ppm="$tmpdir/input.ppm"
    convert "$rgb" "$ppm" 2>/dev/null

    for q in "${QUALITIES[@]}"; do
        local jpg="$tmpdir/q${q}.jpg"
        local decoded="$tmpdir/q${q}.png"

        # Encode with libjpeg-turbo
        cjpeg -quality "$q" -outfile "$jpg" "$ppm" 2>/dev/null

        # Decode back to PNG via djpeg → PPM → PNG
        djpeg -pnm "$jpg" 2>/dev/null | convert ppm:- "$decoded" 2>/dev/null

        # zensim: "score: 85.1234  raw_distance: 1.234567  time: ..."
        local zout zs zr
        zout=$($ZENSIM "$rgb" "$decoded" 2>/dev/null) || zout="score: NaN raw_distance: NaN"
        zs=$(echo "$zout" | awk '{print $2}')
        zr=$(echo "$zout" | awk '{print $4}')

        # ssimulacra2: prints a float
        local s2
        s2=$($SSIM2 "$rgb" "$decoded" 2>/dev/null) || s2="NaN"

        # butteraugli: first line is the max distance
        local bav
        bav=$($BA "$rgb" "$decoded" 2>/dev/null | head -1) || bav="NaN"

        # dssim: "0.001234\tpath"
        local ds
        ds=$($DSSIM "$rgb" "$decoded" 2>/dev/null | awk '{print $1}') || ds="NaN"

        echo "$corpus_name,$base,$q,$zs,$zr,$s2,$bav,$ds" >> "$csv"
    done

    rm -rf "$tmpdir"
    echo "  $base"
}

measure_corpus() {
    local corpus_name="$1"
    local corpus_dir="$2"
    local csv="$OUTDIR/${corpus_name}.csv"

    echo "corpus,image,quality,zensim_score,zensim_raw,ssim2,butteraugli,dssim" > "$csv"

    local images=("$corpus_dir"/*.png)
    local n=${#images[@]}
    echo ""
    echo "=== $corpus_name: $n images x ${#QUALITIES[@]} qualities ==="

    for img in "${images[@]}"; do
        measure_one "$img" "$corpus_name" "$csv"
    done

    echo "Wrote $csv ($((n * ${#QUALITIES[@]})) rows)"
}

echo "Building zensim-cli..."
(cd /home/lilith/work/zensim && cargo build --release -p zensim-cli 2>&1 | tail -1)

# Quick sanity check
echo "Sanity check..."
TDIR="$TMPDIR_BASE/sanity"
mkdir -p "$TDIR"
convert /mnt/v/GitHub/codec-corpus/CID22/CID22-512/validation/1025469.png -type TrueColor "$TDIR/src.png" 2>/dev/null
convert "$TDIR/src.png" "$TDIR/src.ppm" 2>/dev/null
cjpeg -quality 50 -outfile "$TDIR/q50.jpg" "$TDIR/src.ppm" 2>/dev/null
djpeg -pnm "$TDIR/q50.jpg" 2>/dev/null | convert ppm:- "$TDIR/q50.png" 2>/dev/null
echo "  zensim: $($ZENSIM "$TDIR/src.png" "$TDIR/q50.png" 2>/dev/null)"
echo "  ssim2:  $($SSIM2 "$TDIR/src.png" "$TDIR/q50.png" 2>/dev/null)"
echo "  ba:     $($BA "$TDIR/src.png" "$TDIR/q50.png" 2>/dev/null | head -1)"
echo "  dssim:  $(dssim "$TDIR/src.png" "$TDIR/q50.png" 2>/dev/null | awk '{print $1}')"
rm -rf "$TDIR"

measure_corpus "gb82-sc" "/mnt/v/GitHub/codec-corpus/gb82-sc"
measure_corpus "cid22" "/mnt/v/GitHub/codec-corpus/CID22/CID22-512/validation"
measure_corpus "clic2025" "/mnt/v/GitHub/codec-corpus/clic2025/final-test"

echo ""
echo "=== Done ==="
for f in "$OUTDIR"/*.csv; do
    echo "  $(wc -l < "$f") rows in $(basename "$f")"
done

rm -rf "$TMPDIR_BASE"
