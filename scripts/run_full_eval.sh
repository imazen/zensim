#!/usr/bin/env bash
#
# run_full_eval.sh — ONE comprehensive Rust "full-eval" per bake → unified JSON.
#
#   scripts/run_full_eval.sh <bake.bin> <name> [regime=720]
#
# Chains the canonical Rust owners (NO Python for any statistic):
#   1. bake_verdict --full-json  → rank (per-corpus Mohammadi) + dial
#      (mono/tied/reach/dynamic_range) + corruption gate + a sampled multi-metric
#      per_pair block (pred vs mos/jnd for the rank corpora; pred vs
#      ssim2/butter/cvvdp from the KADIS-720 metric parquet).
#   2. diffmap_block_coherence --bake  → the G-STEER M3 (deployable diffmap↔ΔS
#      coherence) averaged over the 3 fixture image pairs.
#   3. jq injects the averaged M3 as the top-level `m3_coherence`.
#
# Output: /mnt/v/output/zensim/reports/fulleval/<name>.fulleval.json
#         (+ <name>.verdict.md — the human bake_verdict report, for reference)
#
# Schema + rationale: docs/FULL_EVAL.md.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: run_full_eval.sh <bake.bin> <name> [regime=720]" >&2
    exit 2
fi
BAKE=$1
NAME=$2
REGIME=${3:-720}

# Repo-relative — NEVER a hardcoded worktree path (CLAUDE.md). Works from the
# main checkout or any jj workspace: binaries build into that tree's target/.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FIX=${ZENSIM_M3_FIXTURES:-/mnt/v/output/zensim/diffmap-coherence-2026-07-18}
DIST_Q=${ZENSIM_M3_DIST_Q:-q50}       # fixture distortion level for M3 pairs
OUTDIR=${ZENSIM_FULLEVAL_OUT:-/mnt/v/output/zensim/reports/fulleval}
mkdir -p "$OUTDIR"

# nice/ionice so a build never starves a co-tenant (CLAUDE.md machine-safety).
HEAVY=(nice -n 19)
command -v ionice >/dev/null 2>&1 && HEAVY=(nice -n 19 ionice -c 3)

command -v jq >/dev/null 2>&1 || { echo "run_full_eval: jq is required" >&2; exit 3; }
[[ -f "$BAKE" ]] || { echo "run_full_eval: bake not found: $BAKE" >&2; exit 3; }

echo "== build (release): bake_verdict + diffmap_block_coherence ==" >&2
"${HEAVY[@]}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" \
    -p zensim-validate --bin bake_verdict >&2
# feature-regime-v2 so a >372 (720) bake's v2 block folds into the M3 map; the
# path is inert for a <=372 bake, so this one binary serves both regimes.
"${HEAVY[@]}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" \
    -p zensim --features custom-profiles,feature-regime-v2 \
    --example diffmap_block_coherence >&2

BV="$REPO_ROOT/target/release/bake_verdict"
DM="$REPO_ROOT/target/release/examples/diffmap_block_coherence"
JSON="$OUTDIR/$NAME.fulleval.json"
MD="$OUTDIR/$NAME.verdict.md"

echo "== bake_verdict --regime $REGIME --full-json ==" >&2
"${HEAVY[@]}" "$BV" --bake "$BAKE" --name "$NAME" --regime "$REGIME" \
    --full-json "$JSON" --output "$MD" >&2

# ── M3 diffmap-coherence: content × size × quality sweep ──────────────────
# WIDENED 2026-07-26 (stats review §Rec-8) from 3 fixtures × q50 to a
# content × size × quality grid — the size axis matters because M3 spatializes
# a per-block map and block count scales with resolution. Sizes are Mitchell
# DOWNSCALES of the 576px refs (never upscaling, per CLAUDE.md). Also captures
# the dropped-f156-371 |s_k| mass so a LOW M3 is read against pooled-feature
# reliance ("incoherent map" != "model uses non-spatializable pooled features").
M3_CONTENT=(city dog girl)
M3_SIZES=(576 384 256) # 576=orig; 384/256 = Mitchell downscales (no upscaling)
M3_QS=(20 50 75)
RESIZE=""
command -v magick >/dev/null 2>&1 && RESIZE="magick"
[[ -z "$RESIZE" ]] && command -v convert >/dev/null 2>&1 && RESIZE="convert"
# Generate the size×q fixtures once (idempotent; persisted under $FIX).
if [[ -n "$RESIZE" ]]; then
    for ref in "${M3_CONTENT[@]}"; do
        for sz in "${M3_SIZES[@]}"; do
            if [[ "$sz" == "576" ]]; then rp="$FIX/${ref}.png"; else
                rp="$FIX/${ref}_${sz}.png"
                [[ -f "$rp" ]] || "$RESIZE" "$FIX/${ref}.png" -filter Mitchell -resize "${sz}x${sz}" "$rp" 2>/dev/null
            fi
            for q in "${M3_QS[@]}"; do
                dp="$FIX/${ref}_${sz}_q${q}.jpg"
                [[ -f "$dp" ]] || "$RESIZE" "$rp" -quality "$q" "$dp" 2>/dev/null
            done
        done
    done
else
    echo "   M3: no magick/convert — size axis skipped, orig-size q-sweep only" >&2
    M3_SIZES=(576)
fi

echo "== M3 coherence: ${#M3_CONTENT[@]} content × ${#M3_SIZES[@]} size × ${#M3_QS[@]} q ==" >&2
M3_SUM=0; MASS_SUM=0; M3_N=0; MASS_N=0
for ref in "${M3_CONTENT[@]}"; do
    for sz in "${M3_SIZES[@]}"; do
        if [[ "$sz" == "576" ]]; then R="$FIX/${ref}.png"; else R="$FIX/${ref}_${sz}.png"; fi
        for q in "${M3_QS[@]}"; do
            D="$FIX/${ref}_${sz}_q${q}.jpg"
            [[ -f "$R" && -f "$D" ]] || { echo "   skip ${ref}/${sz}/q${q}: missing pair" >&2; continue; }
            log="$OUTDIR/$NAME.m3.${ref}_${sz}_q${q}.log"
            if ! "${HEAVY[@]}" "$DM" "$R" "$D" --bake "$BAKE" >"$log" 2>&1; then
                echo "   skip ${ref}/${sz}/q${q}: diffmap_block_coherence failed" >&2
                continue
            fi
            # READ + average the Rust-computed M3 SROCC + dropped-mass (never re-derive).
            m3=$(awk -F'=' '/^  M3 /{split($2,a," "); print a[1]; exit}' "$log")
            mass=$(grep -oE "mass: [0-9.]+" "$log" | head -1 | awk '{print $2}')
            [[ -z "$m3" ]] && { echo "   skip ${ref}/${sz}/q${q}: no M3 line" >&2; continue; }
            M3_SUM=$(awk -v s="$M3_SUM" -v v="$m3" 'BEGIN{printf "%.10f", s + v}')
            M3_N=$((M3_N + 1))
            if [[ -n "$mass" ]]; then
                MASS_SUM=$(awk -v s="$MASS_SUM" -v v="$mass" 'BEGIN{printf "%.6f", s + v}')
                MASS_N=$((MASS_N + 1))
            fi
        done
    done
done

if [[ "$M3_N" -gt 0 ]]; then
    M3_AVG=$(awk -v s="$M3_SUM" -v n="$M3_N" 'BEGIN{printf "%.6f", s / n}')
    echo "== M3 mean over $M3_N pair(s) = $M3_AVG ==" >&2
    jq --argjson m3 "$M3_AVG" --argjson n "$M3_N" '.m3_coherence = $m3 | .m3_n = $n' \
        "$JSON" >"$JSON.tmp" && mv "$JSON.tmp" "$JSON"
    if [[ "$MASS_N" -gt 0 ]]; then
        MASS_AVG=$(awk -v s="$MASS_SUM" -v n="$MASS_N" 'BEGIN{printf "%.4f", s / n}')
        echo "== M3 dropped-f156-371 mass mean = ${MASS_AVG}% (read a low M3 against this) ==" >&2
        jq --argjson dm "$MASS_AVG" '.m3_dropped_mass_pct = $dm' "$JSON" >"$JSON.tmp" \
            && mv "$JSON.tmp" "$JSON"
    fi
else
    echo "== M3: no successful pairs — leaving m3_coherence null ==" >&2
fi

echo "wrote $JSON" >&2
echo "$JSON"
