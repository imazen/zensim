#!/usr/bin/env bash
# m3a_sweep.sh — THE owner of the M3/M3a diffmap-coherence grid sweep.
#
# Extracted from `run_full_eval.sh` (2026-08-04, campaign appendix E.5) so the
# grid has ONE implementation with two callers: `run_full_eval.sh` (which
# injects the means into the fulleval JSON) and any impact-accounting /
# selection run that needs M3a without the rank+dial panels. Per the
# no-duplication rule, do NOT re-inline this loop anywhere.
#
# It computes NO statistics: `diffmap_block_coherence` produces every M3/M3a
# per cell and this script READS and averages them.
#
# GRIDS
#   full   (default, 27 cells) — 3 content x 3 sizes x 3 quality. The
#          registered instrument; `run_full_eval.sh` uses this.
#   cheap  (9 cells) — REGISTERED, MEASURED, and REJECTED. Kept as a hard
#          ERROR rather than deleted, so the rejection is stated at the point
#          of temptation. The subset was the balanced Latin square
#          `q_index = (content_index + size_index) mod 3` over
#          content (city, dog, girl) x size (576, 384, 256) x q (20, 50, 75)
#          — every content/size/q exactly 3x, balanced on all three axes by
#          construction, frozen in campaign appendix E.5 BEFORE any agreement
#          number existed. It FAILED both halves of its registered gate on
#          the 32-bake 944 population (see that doc; measurement is
#          reproducible via scripts/v_next/m3a_cheap_grid_agreement.py, which
#          derives the subset from full-grid TSVs and needs no support here).
#
# USAGE
#   scripts/m3a_sweep.sh --bake <bake.bin> [--bin <diffmap_block_coherence>]
#       [--grid full|cheap] [--tsv <out.tsv>] [--label NAME] [--logdir DIR]
#
# STDOUT (machine-readable, one key=value per line):
#   M3_MEAN=... M3_N=... M3A_MEAN=... M3A_N=... MASS_MEAN=... MASS_N=...
# Missing means are emitted as the key with an empty value, never as 0.
#
# EXIT: 0 = at least one cell scored; 2 = usage; 3 = no cell scored.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FIX=${ZENSIM_M3_FIXTURES:-/mnt/v/output/zensim/diffmap-coherence-2026-07-18}
TGT="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

BAKE=""; BIN="$TGT/release/examples/diffmap_block_coherence"
GRID=full; TSV=""; LABEL=""; LOGDIR=""

die() { echo "m3a_sweep: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bake)   BAKE=${2:?};   shift 2 ;;
        --bin)    BIN=${2:?};    shift 2 ;;
        --grid)   GRID=${2:?};   shift 2 ;;
        --tsv)    TSV=${2:?};    shift 2 ;;
        --label)  LABEL=${2:?};  shift 2 ;;
        --logdir) LOGDIR=${2:?}; shift 2 ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done
[ -n "$BAKE" ] || die "need --bake <file>"
[ -s "$BAKE" ] || die "bake not found or empty: $BAKE"
[ -x "$BIN" ]  || die "diffmap binary not found or not executable: $BIN"
LABEL=${LABEL:-$(basename "$BAKE" .bin)}
LOGDIR=${LOGDIR:-$(mktemp -d "${TMPDIR:-$HOME/tmp}/m3a_sweep.XXXXXX")}
mkdir -p "$LOGDIR"

HEAVY=(nice -n 19)
command -v ionice >/dev/null 2>&1 && HEAVY=(nice -n 19 ionice -c 3)

CONTENT=(city dog girl)
SIZES=(576 384 256)
QS=(20 50 75)

# Build the cell list for the requested grid.
CELLS=()
case "$GRID" in
    full)
        for ci in 0 1 2; do for si in 0 1 2; do for qi in 0 1 2; do
            CELLS+=("${CONTENT[$ci]} ${SIZES[$si]} ${QS[$qi]}")
        done; done; done ;;
    cheap)
        cat >&2 <<'REJECTED'
m3a_sweep: --grid cheap is REGISTERED, MEASURED and REJECTED — refusing.

The 9-cell balanced Latin square failed BOTH halves of its pre-registered
agreement gate (campaign appendix E.5) on the full 32-bake 944 population,
derived from the same per-cell measurements as the full grid:

    SROCC(cheap, full) = 0.8871   gate: >= 0.90    FAIL
    max |cheap - full| = 0.1021   gate: <= 0.02    FAIL   (mean 0.0193)

0.1021 is more than TWICE the whole 944-class M3a sd (0.0471), so a cheap-grid
M3a cannot be used for selection: it moves a bake further than the entire
signal being selected on. The full 27-cell grid also costs only ~66 s/bake,
below the registered 120 s trigger, so there was never a cost case either.

Use --grid full. To re-examine the rejection, run
scripts/v_next/m3a_cheap_grid_agreement.py over full-grid TSVs — it derives
the subset itself and needs no support here.
REJECTED
        exit 2 ;;
    *) die "unknown --grid: $GRID (want full)" ;;
esac

[ -n "$TSV" ] && printf 'label\tgrid\tcontent\tsize\tq\tm3\tm3a\tdropped_mass_pct\n' > "$TSV"

M3_SUM=0; M3_N=0; M3A_SUM=0; M3A_N=0; MASS_SUM=0; MASS_N=0
for cell in "${CELLS[@]}"; do
    read -r ref sz q <<<"$cell"
    if [ "$sz" = 576 ]; then R="$FIX/${ref}.png"; else R="$FIX/${ref}_${sz}.png"; fi
    D="$FIX/${ref}_${sz}_q${q}.jpg"
    if [ ! -f "$R" ] || [ ! -f "$D" ]; then
        echo "   skip ${ref}/${sz}/q${q}: missing pair" >&2; continue
    fi
    log="$LOGDIR/$LABEL.m3.${ref}_${sz}_q${q}.log"
    if ! "${HEAVY[@]}" "$BIN" "$R" "$D" --bake "$BAKE" >"$log" 2>&1; then
        echo "   skip ${ref}/${sz}/q${q}: diffmap_block_coherence failed" >&2; continue
    fi
    m3=$(awk -F'=' '/^  M3 /{split($2,a," "); print a[1]; exit}' "$log")
    m3a=$(awk -F'=' '/^  M3a /{split($2,a," "); print a[1]; exit}' "$log")
    mass=$(grep -oE "mass: [0-9.]+" "$log" | head -1 | awk '{print $2}')
    [ -n "$TSV" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$LABEL" "$GRID" "$ref" "$sz" "$q" "${m3:-}" "${m3a:-}" "${mass:-}" >> "$TSV"
    [ -z "$m3" ] && { echo "   skip ${ref}/${sz}/q${q}: no M3 line" >&2; continue; }
    M3_SUM=$(awk -v s="$M3_SUM" -v v="$m3" 'BEGIN{printf "%.10f", s + v}'); M3_N=$((M3_N + 1))
    if [ -n "$m3a" ]; then
        M3A_SUM=$(awk -v s="$M3A_SUM" -v v="$m3a" 'BEGIN{printf "%.10f", s + v}'); M3A_N=$((M3A_N + 1))
    fi
    if [ -n "$mass" ]; then
        MASS_SUM=$(awk -v s="$MASS_SUM" -v v="$mass" 'BEGIN{printf "%.6f", s + v}'); MASS_N=$((MASS_N + 1))
    fi
done

[ "$M3_N" -gt 0 ] || { echo "m3a_sweep: no cell scored" >&2; exit 3; }
awkmean() { awk -v s="$1" -v n="$2" -v f="$3" 'BEGIN{printf f, s / n}'; }
echo "M3_MEAN=$(awkmean "$M3_SUM" "$M3_N" '%.6f')"
echo "M3_N=$M3_N"
if [ "$M3A_N" -gt 0 ]; then echo "M3A_MEAN=$(awkmean "$M3A_SUM" "$M3A_N" '%.6f')"; else echo "M3A_MEAN="; fi
echo "M3A_N=$M3A_N"
if [ "$MASS_N" -gt 0 ]; then echo "MASS_MEAN=$(awkmean "$MASS_SUM" "$MASS_N" '%.4f')"; else echo "MASS_MEAN="; fi
echo "MASS_N=$MASS_N"
echo "LOGDIR=$LOGDIR"
