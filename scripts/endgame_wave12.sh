#!/usr/bin/env bash
#
# endgame_wave12.sh — the WAVE-12 endgame (campaign appendix AC + AD; playbook
# step 5's --then target). Idempotent; safe to re-run. Assumes the k W12
# fullevals exist (invoked by await_artifacts.sh --then, or by hand).
#
# Produces, under benchmarks/wave12/:
#   1. wave12_select_<date>.{txt,tsv} — freeze_check --select over the W12
#      battery + the C references (E.4 rule; the registered selection).
#   2. wave12_avifdial_<date>.tsv — the G-AC2 AVIF dial-ladder instrument:
#      bake_verdict --dial-grid avif_dial8_944col (the SAME dial_panel owner)
#      per W12 bake + shipped C (raw + packed twins), mono/tied extracted
#      from the full-json `dial` block. NOTHING is recomputed here.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BV=${ZL_BV:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_verdict}
FC=${ZL_FC:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/freeze_check}
FE=${ZENSIM_FULLEVAL_OUT:-/mnt/v/output/zensim/reports/fulleval}
BAKES=${SOTA944_OUT:-/mnt/v/output/zensim/bakes/sota944/bakes}
GRID=${AVIF_DIAL8_GRID:-/mnt/v/output/zensim/v2-eval-944-2026-08-01/avif_dial8_944col_2026-08-21.parquet}
OUTD="$REPO_ROOT/benchmarks/wave12"
ADIAL=${WAVE12_ADIAL_DIR:-/mnt/v/output/zensim/bakes/sota944/wave12_avifdial}
DATE=2026-08-21
SEEDS=(4201 4203 4205 4207 4209 4211)
MIN_K=${WAVE12_MIN_K:-4}   # registered: battery tolerates seed loss down to k=4

mkdir -p "$OUTD" "$ADIAL"
[[ -x "$BV" ]] || { echo "endgame: missing $BV" >&2; exit 2; }
[[ -x "$FC" ]] || { echo "endgame: missing $FC" >&2; exit 2; }
[[ -f "$GRID" ]] || { echo "endgame: missing AVIF dial grid $GRID" >&2; exit 2; }

# ── collect the battery's fullevals ──────────────────────────────────────
FEVALS=()
for s in "${SEEDS[@]}"; do
  f="$FE/W12_s${s}.fulleval.json"
  [[ -f "$f" ]] && FEVALS+=("$f") || echo "endgame: NOTE seed $s fulleval absent" >&2
done
if [[ ${#FEVALS[@]} -lt $MIN_K ]]; then
  echo "endgame: only ${#FEVALS[@]} fullevals (< registered floor $MIN_K) — refusing" >&2
  exit 3
fi

# C references ride along in the select table (comparators, labeled by name).
for c in W10L9_s4003 W10L9_s4003_packed; do
  [[ -f "$FE/$c.fulleval.json" ]] && FEVALS+=("$FE/$c.fulleval.json")
done

# ── 1. the registered selection (E.4 via freeze_check --select) ──────────
"$FC" --select "${FEVALS[@]}" > "$OUTD/wave12_select_${DATE}.txt" 2>&1
rc1=$?
"$FC" --select "${FEVALS[@]}" --tsv > "$OUTD/wave12_select_${DATE}.tsv" 2>/dev/null
echo "endgame: select rc=$rc1 -> $OUTD/wave12_select_${DATE}.txt"

# ── 2. the AVIF dial-ladder instrument (G-AC2 axis) ──────────────────────
TSV="$OUTD/wave12_avifdial_${DATE}.tsv"
echo -e "bake\tavif_mono\tavif_tied\tavif_dynamic_range\tavif_p5\tavif_p95" > "$TSV.tmp"
FAILED=0
for name in W12_s4201 W12_s4203 W12_s4205 W12_s4207 W12_s4209 W12_s4211 \
            W10L9_s4003 W10L9_s4003_packed; do
  bake="$BAKES/$name.bin"
  [[ -f "$bake" ]] || { echo "endgame: no bake $bake (skip)" >&2; continue; }
  j="$ADIAL/$name.avifdial.json"
  if [[ ! -s "$j" ]]; then
    "$BV" --bake "$bake" --regime 944 --corpora tid --dial-grid "$GRID" \
          --full-json "$j" --output "$ADIAL/$name.avifdial.md" \
          > "$ADIAL/$name.avifdial.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 || ! -s "$j" ]]; then
      echo "endgame: AVIF-dial FAILED for $name (rc=$rc, log $ADIAL/$name.avifdial.log)" >&2
      FAILED=$((FAILED + 1)); continue
    fi
  fi
  python3 - "$name" "$j" >> "$TSV.tmp" <<'PY'
import json, sys
name, path = sys.argv[1], sys.argv[2]
d = json.load(open(path)).get("dial") or {}
def g(k):
    v = d.get(k)
    return "NA" if v is None else f"{v:.6f}" if isinstance(v, float) else str(v)
print(f"{name}\t{g('mono_pct')}\t{g('tied_pct')}\t{g('dynamic_range')}\t{g('p5')}\t{g('p95')}")
PY
done
mv "$TSV.tmp" "$TSV"
echo "endgame: AVIF dial table -> $TSV (failures: $FAILED)"
[[ $FAILED -eq 0 ]] || exit 6
exit 0
