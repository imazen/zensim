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
#   1. bake_verdict --fulleval  → the schema-complete fulleval JSON: rank
#      (per-corpus Mohammadi) + dial (mono/tied/reach/dynamic_range) +
#      corruption gate + a sampled multi-metric per_pair block (pred vs
#      mos/jnd for the rank corpora; pred vs ssim2/butter/cvvdp from the
#      KADIS metric parquet), with all five M3 slots emitted as nulls.
#   2. diffmap_block_coherence --bake  → G-STEER coherence: M3 (legacy signal
#      fold) AND M3a (the DEPLOYABLE attribution-density map, task #67 —
#      exact integrands + SAT), averaged over the fixture sweep.
#   3. jq injects the averages INTO the pre-nulled `m3_*`/`m3a_*` keys —
#      this script's remaining role is the M3/M3a measurement, not assembly.
#
# Output: /mnt/v/output/zensim/reports/fulleval/<name>.fulleval.json
#         (+ <name>.verdict.md — the human bake_verdict report, for reference)
#
# ENV MODES
#   ZENSIM_M3_REUSE=1  carry m3_*/m3a_* from the previous JSON instead of
#                      re-measuring (schema re-emits — the rank/dial part is a
#                      cheap rescore, the M3 sweep is not).
#   ZENSIM_M3_ONLY=1   the INVERSE: keep the existing JSON's rank/dial/
#                      corruption blocks untouched (bake_verdict is NOT run)
#                      and re-measure ONLY M3/M3a, injecting into the existing
#                      keys. Use when the coherence INSTRUMENT changed but the
#                      bake and corpora did not — e.g. the 2026-08-04 append2
#                      coverage fix (299ccc8c), which moved every 944-width
#                      M3a and nothing else. Requires the JSON to exist.
#   ZENSIM_M3_GRID     full (default, and the only accepted value — the
#                      registered 9-cell cheap grid was MEASURED and REJECTED,
#                      campaign appendix E.5; m3a_sweep.sh refuses it).
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
# regime "944" = bake_verdict's own `--regime 944` preset (the SOTA-944
# campaign invocation: ext944 roots, 944 grids, kadis-944 perpair, frozen
# 12-corpus list — benchmarks/sota944_campaign_2026-08-03.md §0). The paths
# and corpus list live IN the binary now (test-pinned), so this script can
# no longer drift from them — the wrapper-drift class that produced the
# published wrong EM4 HF-NL number (campaign doc, Corrections section).
if [[ "$REGIME" == "944" ]]; then
    BV_REGIME=944
    BV_EXTRA=()
fi
if [[ "${ZENSIM_M3_ONLY:-0}" == "1" ]]; then
    # Instrument-changed re-measure: leave every rank/dial/corruption number
    # exactly as the owning tool produced it, and refresh only M3/M3a.
    [[ -f "$JSON" ]] || { echo "run_full_eval: ZENSIM_M3_ONLY=1 needs an existing $JSON" >&2; exit 3; }
    [[ "${ZENSIM_M3_REUSE:-0}" == "1" ]] && { echo "run_full_eval: ZENSIM_M3_ONLY and ZENSIM_M3_REUSE are mutually exclusive" >&2; exit 2; }
    echo "== ZENSIM_M3_ONLY=1 — skipping bake_verdict; re-measuring M3/M3a only ==" >&2
else
# Stash the previous JSON so ZENSIM_M3_REUSE=1 can carry its M3 fields after
# bake_verdict overwrites the file (bake_verdict always emits m3=null).
[[ "${ZENSIM_M3_REUSE:-0}" == "1" && -f "$JSON" ]] && cp "$JSON" "$JSON.pre"
"${HEAVY[@]}" "$BV" --bake "$BAKE" --name "$NAME" --regime "$BV_REGIME" \
    "${BV_EXTRA[@]}" \
    --fulleval "$JSON" --output "$MD" >&2
fi

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

# Delegate the grid loop to its OWNER (scripts/m3a_sweep.sh, extracted
# 2026-08-04 per the no-duplication rule: one implementation, two callers —
# this script and any impact-accounting / selection run that needs M3a
# without the rank+dial panels). It emits key=value lines; we READ them.
echo "== M3 coherence: delegating to scripts/m3a_sweep.sh --grid ${ZENSIM_M3_GRID:-full} ==" >&2
"$REPO_ROOT/scripts/m3a_sweep.sh" --bake "$BAKE" --bin "$DM" \
    --grid "${ZENSIM_M3_GRID:-full}" --label "$NAME" --logdir "$OUTDIR" \
    --tsv "$OUTDIR/$NAME.m3a_cells.tsv" >"$OUTDIR/$NAME.m3a.kv" || true
kv() { awk -F= -v k="$1" '$1==k{print $2; exit}' "$OUTDIR/$NAME.m3a.kv"; }
M3_AVG=$(kv M3_MEAN);  M3_N=$(kv M3_N)
M3A_AVG=$(kv M3A_MEAN); M3A_N=$(kv M3A_N)
MASS_AVG=$(kv MASS_MEAN); MASS_N=$(kv MASS_N)
M3_N=${M3_N:-0}; M3A_N=${M3A_N:-0}; MASS_N=${MASS_N:-0}

if [[ "$M3_N" -gt 0 ]]; then
    echo "== M3 mean over $M3_N pair(s) = $M3_AVG ==" >&2
    jq --argjson m3 "$M3_AVG" --argjson n "$M3_N" '.m3_coherence = $m3 | .m3_n = $n' \
        "$JSON" >"$JSON.tmp" && mv "$JSON.tmp" "$JSON"
    if [[ "$M3A_N" -gt 0 ]]; then
        echo "== M3a (attribution density) mean over $M3A_N pair(s) = $M3A_AVG ==" >&2
        jq --argjson m3a "$M3A_AVG" --argjson n "$M3A_N" \
            '.m3a_coherence = $m3a | .m3a_n = $n' "$JSON" >"$JSON.tmp" \
            && mv "$JSON.tmp" "$JSON"
    else
        # M3a is a first-class SELECTION input (campaign appendix E.4:
        # freeze_check --select treats a missing M3a as UNMEASURED and
        # therefore NOT SELECTABLE). A silent absence would quietly make a
        # bake unselectable at the end of a wave, so say it out loud here.
        echo "== WARNING: M3a NOT MEASURED for $NAME — this bake will be" >&2
        echo "   UNMEASURED (and NOT SELECTABLE) under freeze_check --select ==" >&2
    fi
    if [[ "$MASS_N" -gt 0 ]]; then
        echo "== M3 dropped-f156-371 mass mean = ${MASS_AVG}% (read a low M3 against this) ==" >&2
        jq --argjson dm "$MASS_AVG" '.m3_dropped_mass_pct = $dm' "$JSON" >"$JSON.tmp" \
            && mv "$JSON.tmp" "$JSON"
    fi
else
    echo "== M3: no successful pairs — leaving m3_coherence null ==" >&2
    echo "== WARNING: M3a NOT MEASURED for $NAME — NOT SELECTABLE under --select ==" >&2
fi

echo "wrote $JSON" >&2
echo "$JSON"
