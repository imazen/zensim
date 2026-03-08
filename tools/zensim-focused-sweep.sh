#!/bin/bash
# zensim-focused-sweep.sh — Focused parameter sweep in three phases.
#
# Phase 1: Diffmap toggles (20 configs, ~2 min each = ~40 min)
#   - masking_strength × sqrt × hf (most impactful)
#
# Phase 2: Redistribution params with best diffmap (30 configs, ~1 min each = ~30 min)
#   - alpha × factor_max
#
# Phase 3: Tile aggregation with best overall (15 configs, ~1 min each = ~15 min)
#   - norm × spatial_weight
#
# Total: ~65 configs, ~85 minutes on 3 images × 2 distances
#
set -euo pipefail

CJXL_RS="${CJXL_RS:-$HOME/work/zen/jxl-encoder-rs/target/release/cjxl-rs}"
DJXL="${DJXL:-$HOME/work/jxl-efforts/libjxl/build/tools/djxl}"
SS2="${SS2:-$HOME/work/jxl-efforts/libjxl/build/tools/ssimulacra2}"
CLIC_DIR="${CLIC_DIR:-$HOME/work/codec-corpus/clic2025-1024}"
CID22_DIR="${CID22_DIR:-/mnt/v/dataset/cid22/CID22/original}"
OUTDIR="/tmp/zensim_focused_sweep"
ITERS="${SWEEP_ITERS:-4}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$OUTDIR/results_${TIMESTAMP}.tsv"

mkdir -p "$OUTDIR"

# 3 images (2 CLIC + 1 CID22), 2 distances — fast enough for 65 configs
readarray -t CLIC < <(ls "$CLIC_DIR"/*.png | head -2)
readarray -t CID22 < <(ls "$CID22_DIR"/*.png | head -1)
IMAGES=("${CLIC[@]}" "${CID22[@]}")
DISTS=(1.0 2.0)

# Prepare stripped
declare -A STRIPPED
for img in "${IMAGES[@]}"; do
    name=$(basename "$img" .png)
    name="${name:0:12}"
    stripped="$OUTDIR/${name}_stripped.png"
    [ -f "$stripped" ] || convert "$img" -strip "$stripped" 2>/dev/null || cp "$img" "$stripped"
    STRIPPED["$img"]="$stripped"
done

encode_one() {
    local img="$1" d="$2" mode="$3" tag="$4"
    local name
    name=$(basename "$img" .png)
    name="${name:0:12}"
    local outjxl="$OUTDIR/${name}_${tag}.jxl"
    local outpng="$OUTDIR/${name}_${tag}.png"
    local stripped="${STRIPPED[$img]}"

    case "$mode" in
        e7)      "$CJXL_RS" "$img" "$outjxl" -d "$d" -e 7 > /dev/null 2>&1 ;;
        e7-zen)  "$CJXL_RS" "$img" "$outjxl" -d "$d" -e 7 --zensim-iters "$ITERS" > /dev/null 2>&1 ;;
        e8-bfly) "$CJXL_RS" "$img" "$outjxl" -d "$d" -e 8 > /dev/null 2>&1 ;;
        e8-zen)  "$CJXL_RS" "$img" "$outjxl" -d "$d" -e 8 --zensim-iters "$ITERS" > /dev/null 2>&1 ;;
    esac

    if [ ! -f "$outjxl" ]; then
        echo "0 0"
        return
    fi
    local size
    size=$(wc -c < "$outjxl")
    "$DJXL" "$outjxl" "$outpng" > /dev/null 2>&1 || true
    if [ ! -f "$outpng" ]; then
        rm -f "$outjxl"
        echo "$size 0"
        return
    fi
    local dec="$OUTDIR/${name}_${tag}_dec.png"
    convert "$outpng" -strip "$dec" 2>/dev/null || cp "$outpng" "$dec"
    local ss2
    ss2=$("$SS2" "$stripped" "$dec" 2>/dev/null | tail -1) || ss2="0"
    rm -f "$outjxl" "$outpng" "$dec"
    echo "$size $ss2"
}

# Header
echo -e "phase\tconfig\tmasking\tsqrt\thf\tnorm\tspatial_w\talpha\tfactor_max\tmode\tavg_size\tavg_ss2\tsize_pct\tss2_delta" > "$RESULTS"

# Compute baselines (e7 and e8-bfly)
echo "=== Computing baselines ==="
e7_size_sum=0; e7_ss2_sum=0; e7_n=0
e8_size_sum=0; e8_ss2_sum=0; e8_n=0
for img in "${IMAGES[@]}"; do
    for d in "${DISTS[@]}"; do
        name=$(basename "$img" .png); name="${name:0:12}"
        result=$(encode_one "$img" "$d" "e7" "base_e7_d${d}")
        read -r sz s2 <<< "$result"
        e7_size_sum=$(echo "$e7_size_sum + $sz" | bc)
        e7_ss2_sum=$(echo "$e7_ss2_sum + $s2" | bc)
        e7_n=$((e7_n + 1))
        echo "  e7 $name d=$d: size=$sz ss2=$s2"

        result=$(encode_one "$img" "$d" "e8-bfly" "base_e8_d${d}")
        read -r sz s2 <<< "$result"
        e8_size_sum=$(echo "$e8_size_sum + $sz" | bc)
        e8_ss2_sum=$(echo "$e8_ss2_sum + $s2" | bc)
        e8_n=$((e8_n + 1))
        echo "  e8-bfly $name d=$d: size=$sz ss2=$s2"
    done
done
e7_avg_size=$(echo "scale=1; $e7_size_sum / $e7_n" | bc)
e7_avg_ss2=$(echo "scale=4; $e7_ss2_sum / $e7_n" | bc)
e8_avg_size=$(echo "scale=1; $e8_size_sum / $e8_n" | bc)
e8_avg_ss2=$(echo "scale=4; $e8_ss2_sum / $e8_n" | bc)
echo ""
echo "Baselines: e7 avg_size=$e7_avg_size avg_ss2=$e7_avg_ss2"
echo "           e8-bfly avg_size=$e8_avg_size avg_ss2=$e8_avg_ss2"
echo ""

run_config() {
    local phase="$1" config="$2" mode="$3"
    local total_size=0 total_ss2=0 n=0
    for img in "${IMAGES[@]}"; do
        for d in "${DISTS[@]}"; do
            name=$(basename "$img" .png); name="${name:0:12}"
            result=$(encode_one "$img" "$d" "$mode" "${config}_d${d}")
            read -r sz s2 <<< "$result"
            total_size=$((total_size + sz))
            total_ss2=$(echo "$total_ss2 + $s2" | bc)
            n=$((n + 1))
        done
    done
    local avg_size avg_ss2 size_pct ss2_delta
    avg_size=$(echo "scale=1; $total_size / $n" | bc)
    avg_ss2=$(echo "scale=4; $total_ss2 / $n" | bc)
    if [ "$mode" = "e7-zen" ]; then
        size_pct=$(echo "scale=2; ($avg_size - $e7_avg_size) * 100 / $e7_avg_size" | bc)
        ss2_delta=$(echo "scale=3; $avg_ss2 - $e7_avg_ss2" | bc)
    else
        size_pct=$(echo "scale=2; ($avg_size - $e8_avg_size) * 100 / $e8_avg_size" | bc)
        ss2_delta=$(echo "scale=3; $avg_ss2 - $e8_avg_ss2" | bc)
    fi
    echo "$avg_size $avg_ss2 $size_pct $ss2_delta"
}

# ===== PHASE 1: Diffmap options =====
echo "===== PHASE 1: Diffmap toggles (masking × sqrt × hf) ====="
echo ""

BEST_P1_SCORE=-999
BEST_P1_CONFIG=""
BEST_P1_MASKING=4
BEST_P1_SQRT=1
BEST_P1_HF=1

P1_ID=0
for masking in none 1 2 4 6 8; do
    for sqrt in 0 1; do
        for hf in 0 1; do
            config="p1_${P1_ID}"
            P1_ID=$((P1_ID + 1))

            export ZENSIM_MASKING="$masking"
            export ZENSIM_SQRT="$sqrt"
            export ZENSIM_HF="$hf"

            echo -n "  $config (masking=$masking sqrt=$sqrt hf=$hf): "

            # Test e7-zen mode (most sensitive to diffmap quality)
            result=$(run_config "p1" "$config" "e7-zen")
            read -r avg_size avg_ss2 size_pct ss2_delta <<< "$result"

            # Score: SSIM2 gain minus size penalty (λ=0.5 per 1% inflation)
            score=$(echo "scale=4; $ss2_delta - 0.5 * $size_pct" | bc 2>/dev/null || echo "-999")
            # Clamp negative size_pct contribution (size savings are free)
            if (( $(echo "$size_pct < 0" | bc -l) )); then
                score="$ss2_delta"
            fi

            echo "Δss2=${ss2_delta} Δsz=${size_pct}% score=${score}"

            echo -e "p1\t${config}\t${masking}\t${sqrt}\t${hf}\t2\t0.6\t0.20\t1.15\te7-zen\t${avg_size}\t${avg_ss2}\t${size_pct}\t${ss2_delta}" >> "$RESULTS"

            if (( $(echo "$score > $BEST_P1_SCORE" | bc -l 2>/dev/null || echo 0) )); then
                BEST_P1_SCORE="$score"
                BEST_P1_CONFIG="$config"
                BEST_P1_MASKING="$masking"
                BEST_P1_SQRT="$sqrt"
                BEST_P1_HF="$hf"
            fi

            unset ZENSIM_MASKING ZENSIM_SQRT ZENSIM_HF
        done
    done
done

echo ""
echo "Phase 1 best: $BEST_P1_CONFIG (masking=$BEST_P1_MASKING sqrt=$BEST_P1_SQRT hf=$BEST_P1_HF) score=$BEST_P1_SCORE"
echo ""

# ===== PHASE 2: Redistribution params (with best diffmap) =====
echo "===== PHASE 2: Redistribution params (alpha × factor_max) ====="
echo ""

BEST_P2_SCORE=-999
BEST_P2_ALPHA=0.20
BEST_P2_FACTOR=1.15

export ZENSIM_MASKING="$BEST_P1_MASKING"
export ZENSIM_SQRT="$BEST_P1_SQRT"
export ZENSIM_HF="$BEST_P1_HF"

P2_ID=0
for alpha in 0.05 0.10 0.15 0.20 0.25 0.30 0.40; do
    for factor_max in 1.05 1.10 1.15 1.20 1.30; do
        config="p2_${P2_ID}"
        P2_ID=$((P2_ID + 1))

        export ZENSIM_ALPHA="$alpha"
        export ZENSIM_FACTOR_MAX="$factor_max"

        echo -n "  $config (alpha=$alpha factor=$factor_max): "

        # Test both modes
        result_e7=$(run_config "p2" "${config}_e7z" "e7-zen")
        read -r _ _ size_pct_e7 ss2_delta_e7 <<< "$result_e7"

        result_e8=$(run_config "p2" "${config}_e8z" "e8-zen")
        read -r avg_size_e8 avg_ss2_e8 size_pct_e8 ss2_delta_e8 <<< "$result_e8"

        # Combined score: average of e7-zen and e8-zen scores
        score_e7=$(echo "scale=4; $ss2_delta_e7 - 0.5 * $size_pct_e7" | bc 2>/dev/null || echo "-999")
        if (( $(echo "$size_pct_e7 < 0" | bc -l) )); then score_e7="$ss2_delta_e7"; fi
        score_e8=$(echo "scale=4; $ss2_delta_e8 - 0.5 * $size_pct_e8" | bc 2>/dev/null || echo "-999")
        if (( $(echo "$size_pct_e8 < 0" | bc -l) )); then score_e8="$ss2_delta_e8"; fi
        score=$(echo "scale=4; ($score_e7 + $score_e8) / 2" | bc 2>/dev/null || echo "-999")

        echo "e7z: Δss2=${ss2_delta_e7} Δsz=${size_pct_e7}%  e8z: Δss2=${ss2_delta_e8} Δsz=${size_pct_e8}%  score=${score}"

        echo -e "p2\t${config}\t${BEST_P1_MASKING}\t${BEST_P1_SQRT}\t${BEST_P1_HF}\t2\t0.6\t${alpha}\t${factor_max}\te7-zen\t-\t-\t${size_pct_e7}\t${ss2_delta_e7}" >> "$RESULTS"
        echo -e "p2\t${config}\t${BEST_P1_MASKING}\t${BEST_P1_SQRT}\t${BEST_P1_HF}\t2\t0.6\t${alpha}\t${factor_max}\te8-zen\t${avg_size_e8}\t${avg_ss2_e8}\t${size_pct_e8}\t${ss2_delta_e8}" >> "$RESULTS"

        if (( $(echo "$score > $BEST_P2_SCORE" | bc -l 2>/dev/null || echo 0) )); then
            BEST_P2_SCORE="$score"
            BEST_P2_ALPHA="$alpha"
            BEST_P2_FACTOR="$factor_max"
        fi

        unset ZENSIM_ALPHA ZENSIM_FACTOR_MAX
    done
done

echo ""
echo "Phase 2 best: alpha=$BEST_P2_ALPHA factor=$BEST_P2_FACTOR score=$BEST_P2_SCORE"
echo ""

# ===== PHASE 3: Tile aggregation =====
echo "===== PHASE 3: Tile aggregation (norm × spatial_weight) ====="
echo ""

BEST_P3_SCORE=-999
BEST_P3_NORM=2
BEST_P3_SPATIAL=0.6

export ZENSIM_ALPHA="$BEST_P2_ALPHA"
export ZENSIM_FACTOR_MAX="$BEST_P2_FACTOR"

P3_ID=0
for norm in 1 2 3 4 6; do
    for spatial_w in 0.3 0.5 0.6 0.8 1.0; do
        config="p3_${P3_ID}"
        P3_ID=$((P3_ID + 1))

        export ZENSIM_NORM="$norm"
        export ZENSIM_SPATIAL_W="$spatial_w"

        echo -n "  $config (norm=L$norm spatial=$spatial_w): "

        result_e7=$(run_config "p3" "${config}_e7z" "e7-zen")
        read -r _ _ size_pct_e7 ss2_delta_e7 <<< "$result_e7"

        result_e8=$(run_config "p3" "${config}_e8z" "e8-zen")
        read -r _ _ size_pct_e8 ss2_delta_e8 <<< "$result_e8"

        score_e7=$(echo "scale=4; $ss2_delta_e7 - 0.5 * $size_pct_e7" | bc 2>/dev/null || echo "-999")
        if (( $(echo "$size_pct_e7 < 0" | bc -l) )); then score_e7="$ss2_delta_e7"; fi
        score_e8=$(echo "scale=4; $ss2_delta_e8 - 0.5 * $size_pct_e8" | bc 2>/dev/null || echo "-999")
        if (( $(echo "$size_pct_e8 < 0" | bc -l) )); then score_e8="$ss2_delta_e8"; fi
        score=$(echo "scale=4; ($score_e7 + $score_e8) / 2" | bc 2>/dev/null || echo "-999")

        echo "e7z: Δss2=${ss2_delta_e7} Δsz=${size_pct_e7}%  e8z: Δss2=${ss2_delta_e8} Δsz=${size_pct_e8}%  score=${score}"

        echo -e "p3\t${config}\t${BEST_P1_MASKING}\t${BEST_P1_SQRT}\t${BEST_P1_HF}\t${norm}\t${spatial_w}\t${BEST_P2_ALPHA}\t${BEST_P2_FACTOR}\te7-zen\t-\t-\t${size_pct_e7}\t${ss2_delta_e7}" >> "$RESULTS"
        echo -e "p3\t${config}\t${BEST_P1_MASKING}\t${BEST_P1_SQRT}\t${BEST_P1_HF}\t${norm}\t${spatial_w}\t${BEST_P2_ALPHA}\t${BEST_P2_FACTOR}\te8-zen\t-\t-\t${size_pct_e8}\t${ss2_delta_e8}" >> "$RESULTS"

        if (( $(echo "$score > $BEST_P3_SCORE" | bc -l 2>/dev/null || echo 0) )); then
            BEST_P3_SCORE="$score"
            BEST_P3_NORM="$norm"
            BEST_P3_SPATIAL="$spatial_w"
        fi

        unset ZENSIM_NORM ZENSIM_SPATIAL_W
    done
done

unset ZENSIM_MASKING ZENSIM_SQRT ZENSIM_HF ZENSIM_ALPHA ZENSIM_FACTOR_MAX

echo ""
echo "Phase 3 best: norm=L$BEST_P3_NORM spatial=$BEST_P3_SPATIAL score=$BEST_P3_SCORE"
echo ""

# ===== Summary =====
echo "=============================================="
echo "  OPTIMAL CONFIGURATION"
echo "=============================================="
echo ""
echo "  Diffmap:        masking=$BEST_P1_MASKING sqrt=$BEST_P1_SQRT hf=$BEST_P1_HF"
echo "  Redistribution: alpha=$BEST_P2_ALPHA factor_max=$BEST_P2_FACTOR"
echo "  Aggregation:    norm=L$BEST_P3_NORM spatial_weight=$BEST_P3_SPATIAL"
echo ""
echo "  Environment variables:"
echo "    export ZENSIM_MASKING=$BEST_P1_MASKING"
echo "    export ZENSIM_SQRT=$BEST_P1_SQRT"
echo "    export ZENSIM_HF=$BEST_P1_HF"
echo "    export ZENSIM_ALPHA=$BEST_P2_ALPHA"
echo "    export ZENSIM_FACTOR_MAX=$BEST_P2_FACTOR"
echo "    export ZENSIM_NORM=$BEST_P3_NORM"
echo "    export ZENSIM_SPATIAL_W=$BEST_P3_SPATIAL"
echo ""
echo "Full results: $RESULTS"
