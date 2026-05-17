#!/bin/bash
# zensim-sweep.sh — Parameter sweep framework for zensim encoder loop tuning.
#
# Tests parameter combinations against a set of images and measures quality
# (SSIM2) and size. Outputs TSV for analysis.
#
# Usage:
#   ./tools/zensim-sweep.sh                    # full sweep
#   ./tools/zensim-sweep.sh --quick            # 2 images, 1 distance
#   ./tools/zensim-sweep.sh --config sweep.tsv # custom parameter grid
#
# Environment:
#   SWEEP_IMAGES   - space-separated image paths (overrides defaults)
#   SWEEP_DISTS    - space-separated distances (default: "1.0 2.0 4.0")
#   SWEEP_MODES    - space-separated modes (default: "e7 e7-zen4 e8-zen4")
#   SWEEP_ITERS    - zensim iterations (default: 4)
#   SWEEP_OUTDIR   - output directory (default: /tmp/zensim_sweep)
#
set -euo pipefail

CJXL_RS="${CJXL_RS:-$HOME/work/zen/jxl-encoder-rs/target/release/cjxl-rs}"
DJXL="${DJXL:-$HOME/work/jxl-efforts/libjxl/build/tools/djxl}"
SS2="${SS2:-$HOME/work/jxl-efforts/libjxl/build/tools/ssimulacra2}"
CLIC_DIR="${CLIC_DIR:-$HOME/work/codec-corpus/clic2025-1024}"
CID22_DIR="${CID22_DIR:-/mnt/v/dataset/cid22/CID22/original}"
OUTDIR="${SWEEP_OUTDIR:-/tmp/zensim_sweep}"
ITERS="${SWEEP_ITERS:-4}"
LOGDIR="$OUTDIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$OUTDIR/sweep_${TIMESTAMP}.tsv"

QUICK=0
CONFIG_FILE=""

for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --config) shift; CONFIG_FILE="$1" ;;
        --config=*) CONFIG_FILE="${arg#--config=}" ;;
    esac
done

mkdir -p "$OUTDIR" "$LOGDIR"

# Select images
if [ -n "${SWEEP_IMAGES:-}" ]; then
    IFS=' ' read -ra IMAGES <<< "$SWEEP_IMAGES"
elif [ "$QUICK" -eq 1 ]; then
    readarray -t CLIC < <(ls "$CLIC_DIR"/*.png | head -2)
    readarray -t CID22 < <(ls "$CID22_DIR"/*.png | head -1)
    IMAGES=("${CLIC[@]}" "${CID22[@]}")
else
    readarray -t CLIC < <(ls "$CLIC_DIR"/*.png | head -5)
    readarray -t CID22 < <(ls "$CID22_DIR"/*.png | head -3)
    IMAGES=("${CLIC[@]}" "${CID22[@]}")
fi

# Select distances
if [ -n "${SWEEP_DISTS:-}" ]; then
    IFS=' ' read -ra DISTS <<< "$SWEEP_DISTS"
elif [ "$QUICK" -eq 1 ]; then
    DISTS=(2.0)
else
    DISTS=(1.0 2.0 4.0)
fi

# Select modes
if [ -n "${SWEEP_MODES:-}" ]; then
    IFS=' ' read -ra MODES <<< "$SWEEP_MODES"
else
    MODES=(e7 e7-zen4 e8-zen4)
fi

# Prepare stripped originals
declare -A STRIPPED
for img in "${IMAGES[@]}"; do
    name=$(basename "$img" .png)
    name="${name:0:12}"
    stripped="$OUTDIR/${name}_stripped.png"
    if [ ! -f "$stripped" ]; then
        convert "$img" -strip "$stripped" 2>/dev/null || cp "$img" "$stripped"
    fi
    STRIPPED["$img"]="$stripped"
done

# Encode and measure a single image with given parameters
encode_measure() {
    local img="$1" d="$2" mode="$3" config_id="$4"
    local name
    name=$(basename "$img" .png)
    name="${name:0:12}"
    local tag="${name}_d${d}_${mode}_${config_id}"
    local outjxl="$OUTDIR/${tag}.jxl"
    local outpng="$OUTDIR/${tag}.png"
    local stripped="${STRIPPED[$img]}"

    case "$mode" in
        e7)      "$CJXL_RS" "$img" "$outjxl" -d "$d" -e 7 > /dev/null 2>&1 ;;
        e8-bfly) "$CJXL_RS" "$img" "$outjxl" -d "$d" -e 8 > /dev/null 2>&1 ;;
        e7-zen*)
            "$CJXL_RS" "$img" "$outjxl" -d "$d" -e 7 --zensim-iters "$ITERS" > /dev/null 2>&1 ;;
        e8-zen*)
            "$CJXL_RS" "$img" "$outjxl" -d "$d" -e 8 --zensim-iters "$ITERS" > /dev/null 2>&1 ;;
    esac

    local size
    size=$(wc -c < "$outjxl")
    "$DJXL" "$outjxl" "$outpng" > /dev/null 2>&1
    local dec="$OUTDIR/${tag}_dec.png"
    convert "$outpng" -strip "$dec" 2>/dev/null || cp "$outpng" "$dec"
    local ss2
    ss2=$("$SS2" "$stripped" "$dec" 2>/dev/null | tail -1) || ss2="NA"

    # Clean up intermediate files to save space
    rm -f "$outjxl" "$outpng" "$dec"

    echo -e "${config_id}\t${name}\t${d}\t${mode}\t${size}\t${ss2}"
}

# Generate parameter grid
# Format: config_id \t MASKING \t SQRT \t HF \t EDGE_MSE \t NORM \t SPATIAL_W \t RATIO_MAX \t ALPHA \t FACTOR_MAX
generate_default_grid() {
    local id=0

    # Phase 1: Diffmap boolean toggles (keep redistribution params at defaults)
    # Baseline: masking=4, sqrt=1, hf=1, edge_mse=1
    for masking in none 2 4 6 8; do
        for sqrt in 0 1; do
            for hf in 0 1; do
                printf "dm%03d\t%s\t%s\t%s\t1\t2\t0.6\t3.0\t0.20\t1.15\n" "$id" "$masking" "$sqrt" "$hf"
                ((id++))
            done
        done
    done

    # Phase 2: Redistribution params (keep best diffmap config = defaults for now)
    for alpha in 0.10 0.15 0.20 0.25 0.30 0.40; do
        for factor_max in 1.05 1.10 1.15 1.20 1.30; do
            printf "rd%03d\t4\t1\t1\t1\t2\t0.6\t3.0\t%s\t%s\n" "$id" "$alpha" "$factor_max"
            ((id++))
        done
    done

    # Phase 3: Tile aggregation params
    for norm in 1 2 3 4; do
        for spatial_w in 0.3 0.5 0.6 0.8 1.0; do
            for ratio_max in 2.0 3.0 5.0; do
                printf "ta%03d\t4\t1\t1\t1\t%s\t%s\t%s\t0.20\t1.15\n" "$id" "$norm" "$spatial_w" "$ratio_max"
                ((id++))
            done
        done
    done
}

# Load or generate config grid
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    GRID_FILE="$CONFIG_FILE"
else
    GRID_FILE="$OUTDIR/grid_${TIMESTAMP}.tsv"
    generate_default_grid > "$GRID_FILE"
fi

TOTAL_CONFIGS=$(wc -l < "$GRID_FILE")
echo "Sweep: $TOTAL_CONFIGS configs × ${#IMAGES[@]} images × ${#DISTS[@]} distances × ${#MODES[@]} modes"
echo "Results → $RESULTS"
echo ""

# Header
echo -e "config_id\tmasking\tsqrt\thf\tedge_mse\tnorm\tspatial_w\tratio_max\talpha\tfactor_max\timage\tdistance\tmode\tsize\tss2" > "$RESULTS"

# First pass: encode baselines (e7 mode, no env vars needed)
echo "=== Encoding baselines (e7 mode, no zensim params) ==="
for img in "${IMAGES[@]}"; do
    for d in "${DISTS[@]}"; do
        for mode in "${MODES[@]}"; do
            if [[ "$mode" == "e7" || "$mode" == "e8-bfly" ]]; then
                result=$(encode_measure "$img" "$d" "$mode" "baseline")
                echo -e "baseline\t-\t-\t-\t-\t-\t-\t-\t-\t-\t$(echo "$result" | cut -f2-)" >> "$RESULTS"
                echo "$result"
            fi
        done
    done
done

# Main sweep
CONFIG_NUM=0
while IFS=$'\t' read -r config_id masking sqrt hf edge_mse norm spatial_w ratio_max alpha factor_max; do
    CONFIG_NUM=$((CONFIG_NUM + 1))
    echo ""
    echo "=== Config $CONFIG_NUM/$TOTAL_CONFIGS: $config_id (masking=$masking sqrt=$sqrt hf=$hf norm=$norm alpha=$alpha factor=$factor_max spatial=$spatial_w ratio=$ratio_max) ==="

    # Set environment variables
    export ZENSIM_MASKING="$masking"
    export ZENSIM_SQRT="$sqrt"
    export ZENSIM_HF="$hf"
    export ZENSIM_EDGE_MSE="$edge_mse"
    export ZENSIM_NORM="$norm"
    export ZENSIM_SPATIAL_W="$spatial_w"
    export ZENSIM_RATIO_MAX="$ratio_max"
    export ZENSIM_ALPHA="$alpha"
    export ZENSIM_FACTOR_MAX="$factor_max"

    for img in "${IMAGES[@]}"; do
        for d in "${DISTS[@]}"; do
            for mode in "${MODES[@]}"; do
                # Skip baselines — already encoded
                if [[ "$mode" == "e7" || "$mode" == "e8-bfly" ]]; then
                    continue
                fi
                result=$(encode_measure "$img" "$d" "$mode" "$config_id")
                echo -e "${config_id}\t${masking}\t${sqrt}\t${hf}\t${edge_mse}\t${norm}\t${spatial_w}\t${ratio_max}\t${alpha}\t${factor_max}\t$(echo "$result" | cut -f2-)" >> "$RESULTS"
                echo "  $result"
            done
        done
    done

    # Unset env vars
    unset ZENSIM_MASKING ZENSIM_SQRT ZENSIM_HF ZENSIM_EDGE_MSE
    unset ZENSIM_NORM ZENSIM_SPATIAL_W ZENSIM_RATIO_MAX
    unset ZENSIM_ALPHA ZENSIM_FACTOR_MAX

done < "$GRID_FILE"

echo ""
echo "=== Sweep complete: $RESULTS ==="
echo "Total configs tested: $CONFIG_NUM"
echo ""

# Quick analysis: compute average SSIM2 and size per config, relative to baseline
echo "=== Top 20 configs by average SSIM2 gain (e7-zen4 mode) ==="
echo ""
# Get baseline averages per image×distance
awk -F'\t' '
NR == 1 { next }
$1 == "baseline" && $13 == "e7" {
    key = $11 "\t" $12
    base_ss2[key] = $15
    base_size[key] = $14
}
$1 != "baseline" && $13 ~ /zen/ {
    key = $11 "\t" $12
    if (key in base_ss2) {
        configs[$1]["ss2_sum"] += ($15 - base_ss2[key])
        configs[$1]["size_sum"] += (($14 - base_size[key]) / base_size[key] * 100)
        configs[$1]["n"]++
    }
}
END {
    printf "%-10s %8s %8s %6s\n", "config", "avg_Δss2", "avg_Δsz%", "n"
    printf "%-10s %8s %8s %6s\n", "------", "--------", "--------", "---"
    for (c in configs) {
        n = configs[c]["n"]
        if (n > 0) {
            printf "%-10s %+8.3f %+8.2f %6d\n", c, configs[c]["ss2_sum"]/n, configs[c]["size_sum"]/n, n
        }
    }
}
' "$RESULTS" | sort -t$'\t' -k2 -rn | head -20

echo ""
echo "Full results: $RESULTS"
