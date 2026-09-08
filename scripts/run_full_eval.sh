#!/usr/bin/env bash
# One evaluation pipeline: independently reusable verdict and coherence stages.
# Rust owns scores/statistics; bake_verdict and m3a_sweep own input identities.
# Usage: run_full_eval.sh [--stage all|verdict|coherence|qualify] bake name [regime] [root]
# Existing ZENSIM_M3_ONLY maps to coherence; M3_REUSE requests only VALID reuse.
# Historical results lacking identities are never a cache hit. Re-run their
# stage to establish provenance. Every stage is saved atomically, so an
# interrupted coherence sweep retains the completed verdict.
# See docs/FULL_EVAL.md for the schema, fixtures and qualification distinction.
set -euo pipefail
STAGE=${ZENSIM_EVAL_STAGE:-all}
[[ "${ZENSIM_M3_ONLY:-0}" == 1 ]] && STAGE=coherence
if [[ "${1:-}" == --stage ]]; then STAGE=${2:?}; shift 2; fi
case "$STAGE" in all|verdict|coherence|qualify) ;; *) echo "unknown stage: $STAGE" >&2; exit 2 ;; esac
if [[ $# -lt 2 ]]; then echo "usage: run_full_eval.sh [--stage all|verdict|coherence|qualify] bake name [regime] [root]" >&2; exit 2; fi
BAKE=$1; NAME=$2; REGIME=${3:-720}
FEATURES_ROOT_OVERRIDE=${4:-${ZENSIM_FEATURES_ROOT:-}}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTDIR=${ZENSIM_FULLEVAL_OUT:-/mnt/v/output/zensim/reports/fulleval}
TGT=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
BV=${ZENSIM_BAKE_VERDICT:-${ZL_BV:-$TGT/release/bake_verdict}}
DM=${ZENSIM_DIFFMAP_BIN:-$TGT/release/examples/diffmap_block_coherence}
HEAVY=("${ZENSIM_RUN_HEAVY:-$HOME/work/zen/scripts/run-heavy}" --mem 16G --jobs 8)
command -v jq >/dev/null
[[ -s "$BAKE" ]] || { echo "bake missing: $BAKE" >&2; exit 3; }
[[ "$NAME" != */* && -n "$NAME" ]] || { echo "name must be a filename stem" >&2; exit 2; }
mkdir -p "$OUTDIR"
JSON="$OUTDIR/$NAME.fulleval.json"; MD="$OUTDIR/$NAME.verdict.md"
VERDICT="$OUTDIR/$NAME.verdict-stage.json"; COHERENCE="$OUTDIR/$NAME.coherence-stage.json"
WORK=$(mktemp -d "$OUTDIR/.${NAME}.eval.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
# A stage owns this output stem while it runs. Another process may use a
# different stem. Keep the lock file: unlinking it would create two locks.
exec 9>"$OUTDIR/.$NAME.eval.lock"
flock 9
if [[ -z "${ZENSIM_BAKE_VERDICT:-${ZL_BV:-}}" && "$STAGE" != coherence && "$STAGE" != qualify ]]; then
    "${HEAVY[@]}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" -p zensim-validate --bin bake_verdict >&2
fi
[[ -x "$BV" ]] || { echo "build bake_verdict or run the verdict stage first" >&2; exit 3; }
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

# ── THE FEATURES ROOT COMES FROM THE BAKE ─────────────────────────────────
# `bake_verdict --print-features-root` runs the ONE derivation (see the header
# note) with the regime's own root as the FALLBACK, so this script never has a
# second opinion about which root a bake belongs to. It is a read-only mode:
# it loads the bake, resolves, prints, exits — no corpus is touched.
#
# Coherence-only resolves the same inputs to validate its prerequisite verdict;
# it does not rescore the corpus.
{
    if [[ -n "$FEATURES_ROOT_OVERRIDE" ]]; then
        RESOLVED_ROOT=$FEATURES_ROOT_OVERRIDE
        echo "== features-root: $RESOLVED_ROOT (explicit caller override) ==" >&2
    elif ! RESOLVED_ROOT=$("$BV" --bake "$BAKE" --regime "$BV_REGIME" "${BV_EXTRA[@]}" \
            --print-features-root); then
        echo "run_full_eval: FATAL — could not determine the features root for $BAKE." >&2
        echo "  The message above says why. Pass it explicitly:" >&2
        echo "    scripts/run_full_eval.sh <bake> <name> $REGIME <features-root>" >&2
        echo "  (or ZENSIM_FEATURES_ROOT=...). Never let it silently default —" >&2
        echo "  a wrong-root read returns plausible-looking numbers with no error." >&2
        exit 3
    fi
    # Re-emit BV_EXTRA with exactly one --features-root, the resolved one.
    # When it equals what the regime would have used this is a no-op by
    # construction (same path, same behavior); when it differs it is the whole
    # point of this block.
    BV_REBUILT=(); skip_next=0
    for a in "${BV_EXTRA[@]}"; do
        if [[ "$skip_next" == 1 ]]; then skip_next=0; continue; fi
        if [[ "$a" == "--features-root" ]]; then skip_next=1; continue; fi
        BV_REBUILT+=("$a")
    done
    BV_EXTRA=("${BV_REBUILT[@]}" --features-root "$RESOLVED_ROOT")
}

BV_ARGS=(--bake "$BAKE" --name "$NAME" --regime "$BV_REGIME" "${BV_EXTRA[@]}")
"${HEAVY[@]}" "$BV" "${BV_ARGS[@]}" --print-inputs > "$WORK/verdict-inputs.json"
valid_verdict() {
    [[ -s "$1" ]] && jq -e --slurpfile i "$WORK/verdict-inputs.json" \
        '.input_identity == $i[0] and .scoring.surface == "zensim::BakeScorer" and (.rank | type == "object")' "$1" >/dev/null
}
if ! valid_verdict "$VERDICT"; then
    if valid_verdict "$JSON"; then
        cp "$JSON" "$WORK/verdict.json"; mv "$WORK/verdict.json" "$VERDICT"
    elif [[ "$STAGE" == coherence || "$STAGE" == qualify ]]; then
        echo "$STAGE needs a verdict with matching model, table and evaluator identities; run --stage verdict first" >&2; exit 3
    else
        "${HEAVY[@]}" "$BV" "${BV_ARGS[@]}" --fulleval "$WORK/verdict.json" --output "$WORK/verdict.md" >&2
        valid_verdict "$WORK/verdict.json" || { echo "verdict inputs changed during evaluation" >&2; exit 3; }
        mv "$WORK/verdict.json" "$VERDICT"; mv "$WORK/verdict.md" "$MD"
    fi
else
    echo "== reused verified verdict stage ==" >&2
fi
if [[ "$STAGE" == qualify ]]; then
    valid_verdict "$JSON" || { echo "qualification needs a current aggregate; run --stage all first" >&2; exit 3; }
    "${HEAVY[@]}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" -p zensim-validate --bin freeze_check >&2
    status=0
    "$TGT/release/freeze_check" --qualify --fulleval "$JSON" > "$WORK/qualification.json" || status=$?
    (( status <= 1 )) || exit "$status"
    jq --slurpfile q "$WORK/qualification.json" '.qualification=$q[0]' "$JSON" > "$WORK/aggregate.json"
    mv "$WORK/aggregate.json" "$JSON"
    cat "$WORK/qualification.json"
    exit "$status"
fi
# Preserve attached evidence only when this aggregate has the current verdict
# identity. Qualification rechecks its content-bound artifacts. Clear the old
# qualification decision and M3, which are independently admitted below.
BASE="$VERDICT"
if valid_verdict "$JSON"; then BASE="$JSON"; fi
jq '.m3_coherence=null | .m3_n=null | .m3_dropped_mass_pct=null | .m3a_coherence=null | .m3a_n=null |
    .evaluation_stages={verdict:"complete",coherence:"not_requested"} | del(.qualification)' "$BASE" > "$WORK/aggregate.json"
mv "$WORK/aggregate.json" "$JSON"
if [[ "$STAGE" == verdict ]]; then echo "$JSON"; exit 0; fi
if [[ -z "${ZENSIM_DIFFMAP_BIN:-}" ]]; then
    "${HEAVY[@]}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" -p zensim \
        --features custom-profiles,feature-regime-v2 --example diffmap_block_coherence >&2
fi
M3_ARGS=(--bake "$BAKE" --bin "$DM" --grid "${ZENSIM_M3_GRID:-full}" --label "$NAME" --logdir "$OUTDIR")
# A missing historical fixture is a refusal. Generating one with a newer
# codec would silently mix fixture eras; use m3_fixture_gen in a NEW directory.
"$REPO_ROOT/scripts/m3a_sweep.sh" "${M3_ARGS[@]}" --print-inputs > "$WORK/coherence-inputs.json"
valid_coherence() {
    [[ -s "$COHERENCE" ]] && jq -e --slurpfile i "$WORK/coherence-inputs.json" \
        '.identity == $i[0] and .m3_n == 27 and .m3a_n == 27' "$COHERENCE" >/dev/null
}
if ! valid_coherence; then
    if ! "${HEAVY[@]}" "$REPO_ROOT/scripts/m3a_sweep.sh" "${M3_ARGS[@]}" \
        --tsv "$OUTDIR/$NAME.m3a_cells.tsv" > "$WORK/coherence.kv"; then
        echo "coherence stage failed; completed verdict retained" >&2; exit 3
    fi
    "$REPO_ROOT/scripts/m3a_sweep.sh" "${M3_ARGS[@]}" --print-inputs > "$WORK/coherence-after.json"
    cmp -s "$WORK/coherence-inputs.json" "$WORK/coherence-after.json" || { echo "coherence inputs changed during evaluation" >&2; exit 3; }
    kv() { awk -F= -v k="$1" '$1==k{print $2; exit}' "$WORK/coherence.kv"; }
    [[ "$(kv M3_N)" == 27 && "$(kv M3A_N)" == 27 ]] || { echo "coherence incomplete; all 27 cells are required" >&2; exit 3; }
    jq -n --slurpfile i "$WORK/coherence-inputs.json" \
        --argjson m3 "$(kv M3_MEAN)" --argjson m3a "$(kv M3A_MEAN)" --arg mass "$(kv MASS_MEAN)" \
        '{identity:$i[0],m3_coherence:$m3,m3_n:27,m3a_coherence:$m3a,m3a_n:27,m3_dropped_mass_pct:($mass | if . == "" then null else tonumber end)}' \
        > "$WORK/coherence.json"
    mv "$WORK/coherence.json" "$COHERENCE"
    cp "$WORK/coherence.kv" "$OUTDIR/$NAME.m3a.kv"
else
    echo "== reused verified coherence stage ==" >&2
fi
jq --slurpfile c "$COHERENCE" '. * ($c[0] | del(.identity)) |
    .evaluation_stages.coherence="complete" | .coherence_identity=$c[0].identity' "$JSON" > "$WORK/aggregate.json"
mv "$WORK/aggregate.json" "$JSON"
echo "$JSON"
