#!/usr/bin/env bash
#
# 924-era NOTE (2026-07-28): imazen26/nonphoto eval slices for 924-regime models come from the
# canonical bigcodec 924 TEST views, NOT the NN-matched ext_*_720 tables — see docs/FULL_EVAL.md
# "924-era eval slices" (user directive; fingerprint matching cannot cross regimes).
# run_full_eval.sh — ONE comprehensive Rust "full-eval" per bake → unified JSON.
#
#   scripts/run_full_eval.sh <bake.bin> <name> [regime=720]
#
# Chains the canonical Rust owners (NO Python for any statistic):
#   1. bake_verdict --full-json  → rank (per-corpus Mohammadi) + dial
#      (mono/tied/reach/dynamic_range) + corruption gate + a sampled multi-metric
#      per_pair block (pred vs mos/jnd for the rank corpora; pred vs
#      ssim2/butter/cvvdp from the KADIS-720 metric parquet).
#   2. diffmap_block_coherence --bake  → G-STEER coherence: M3 (legacy signal
#      fold) AND M3a (the DEPLOYABLE attribution-density map, task #67 —
#      exact integrands + SAT), averaged over the fixture sweep.
#   3. jq injects the averages as top-level `m3_coherence` + `m3a_coherence`.
#
# Output: /mnt/v/output/zensim/reports/fulleval/<name>.fulleval.json
#         (+ <name>.verdict.md — the human bake_verdict report, for reference)
#
# Schema + rationale: docs/FULL_EVAL.md.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: run_full_eval.sh <bake.bin> <name> [regime=720|372|924|944]" >&2
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

# Honor CARGO_TARGET_DIR (campaign workspaces build out-of-tree).
TGT="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BV="$TGT/release/bake_verdict"
DM="$TGT/release/examples/diffmap_block_coherence"
JSON="$OUTDIR/$NAME.fulleval.json"
MD="$OUTDIR/$NAME.verdict.md"

echo "== bake_verdict --regime $REGIME --full-json ==" >&2
# regime "924" = the folded+append campaign invocation: bake_verdict's
# feature-regime flag stays 720 (slot_720 filename map), but the three data
# roots swap to the canonical 924 extractions + the kadis-924 perpair source
# (docs/FULL_EVAL.md "924-era eval slices"; E-LIN linear924_phase1).
BV_EXTRA=()
BV_REGIME=$REGIME
if [[ "$REGIME" == "924" ]]; then
    BV_REGIME=720
    # Corpora = the slots that EXIST as canonical 924 extractions. imazen26 /
    # nonphoto are deliberately absent: their 720 NN tables cannot cross
    # regimes (docs/FULL_EVAL.md "924-era eval slices") and the bigcodec
    # 924-test-view slices are not wired as bake_verdict slot files yet.
    BV_EXTRA=(--features-root /mnt/v/zen/zensim-training/ext924-canonical-2026-07-27
              --dial-grid /mnt/v/output/zensim/v2-eval-924-2026-07-27/dial_grid_924col_2026-07-28.parquet
              --corruption-grid /mnt/v/output/zensim/v2-eval-924-2026-07-27/corruption_grid_924col_2026-07-27.parquet
              --perpair-metrics /mnt/v/zen/zensim-training/kadis-924-2026-07-27/kadis700k_924.parquet
              --corpora cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25)
fi
# regime "944" = the SOTA-944 campaign invocation (folded+append+append2;
# benchmarks/sota944_campaign_2026-08-03.md). Same slot map (--regime 720);
# roots swap to the ext944 canonical set — which ALSO carries the
# imazen26/nonphoto TEST-view slices (build_eval_slices_944.py), so those
# corpora are back on the list (bake_verdict's slot_720_file resolves
# ext_imazen26/ext_nonphoto under this root).
if [[ "$REGIME" == "944" ]]; then
    BV_REGIME=720
    BV_EXTRA=(--features-root /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
              --dial-grid /mnt/v/output/zensim/v2-eval-944-2026-08-01/dial_grid_944col_2026-08-01.parquet
              --corruption-grid /mnt/v/output/zensim/v2-eval-944-2026-08-01/corruption_grid_944col_2026-08-01.parquet
              --perpair-metrics /mnt/v/zen/zensim-training/kadis-944-2026-08-01/kadis700k_944.parquet
              --corpora cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25,imazen26,nonphoto,hfnlproxy)
fi
# Stash the previous JSON so ZENSIM_M3_REUSE=1 can carry its M3 fields after
# bake_verdict overwrites the file (bake_verdict always emits m3=null).
[[ "${ZENSIM_M3_REUSE:-0}" == "1" && -f "$JSON" ]] && cp "$JSON" "$JSON.pre"
"${HEAVY[@]}" "$BV" --bake "$BAKE" --name "$NAME" --regime "$BV_REGIME" \
    "${BV_EXTRA[@]}" \
    --full-json "$JSON" --output "$MD" >&2

# ── M3 diffmap-coherence: content × size × quality sweep ──────────────────
# ZENSIM_M3_REUSE=1: carry m3_coherence/m3_n/m3_dropped_mass_pct +
# m3a_coherence/m3a_n from the PREVIOUS fulleval JSON instead of re-measuring. Use for schema re-emits —
# the rank/dial/corruption portion is a cheap rescore over stored feature
# parquets, but the M3 sweep is 27 diffmap runs (~minutes/bake) measuring a
# value that cannot change unless the bake or fixtures changed. (2026-07-27:
# a 17-bake schema re-emit redid ~45 min of unchanged M3 before this existed.)
if [[ "${ZENSIM_M3_REUSE:-0}" == "1" && -f "$JSON.pre" ]]; then
    jq --slurpfile o "$JSON.pre" \
        '.m3_coherence=$o[0].m3_coherence | .m3_n=$o[0].m3_n | .m3_dropped_mass_pct=$o[0].m3_dropped_mass_pct | .m3a_coherence=$o[0].m3a_coherence | .m3a_n=$o[0].m3a_n' \
        "$JSON" >"$JSON.tmp" && mv "$JSON.tmp" "$JSON"
    rm -f "$JSON.pre"
    echo "== M3 carried from previous JSON (ZENSIM_M3_REUSE=1) ==" >&2
    echo "wrote $JSON" >&2
    echo "$JSON"
    exit 0
fi
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
M3_SUM=0; MASS_SUM=0; M3_N=0; MASS_N=0; M3A_SUM=0; M3A_N=0
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
            # READ + average the Rust-computed M3/M3a SROCCs + dropped-mass
            # (never re-derive). M3a = the DEPLOYABLE attribution-density map
            # (task #67, exact integrands + SAT); M3 = the legacy signal fold,
            # kept for the before/after story.
            m3=$(awk -F'=' '/^  M3 /{split($2,a," "); print a[1]; exit}' "$log")
            m3a=$(awk -F'=' '/^  M3a /{split($2,a," "); print a[1]; exit}' "$log")
            mass=$(grep -oE "mass: [0-9.]+" "$log" | head -1 | awk '{print $2}')
            [[ -z "$m3" ]] && { echo "   skip ${ref}/${sz}/q${q}: no M3 line" >&2; continue; }
            M3_SUM=$(awk -v s="$M3_SUM" -v v="$m3" 'BEGIN{printf "%.10f", s + v}')
            M3_N=$((M3_N + 1))
            if [[ -n "$m3a" ]]; then
                M3A_SUM=$(awk -v s="$M3A_SUM" -v v="$m3a" 'BEGIN{printf "%.10f", s + v}')
                M3A_N=$((M3A_N + 1))
            fi
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
    if [[ "$M3A_N" -gt 0 ]]; then
        M3A_AVG=$(awk -v s="$M3A_SUM" -v n="$M3A_N" 'BEGIN{printf "%.6f", s / n}')
        echo "== M3a (attribution density) mean over $M3A_N pair(s) = $M3A_AVG ==" >&2
        jq --argjson m3a "$M3A_AVG" --argjson n "$M3A_N" \
            '.m3a_coherence = $m3a | .m3a_n = $n' "$JSON" >"$JSON.tmp" \
            && mv "$JSON.tmp" "$JSON"
    fi
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
