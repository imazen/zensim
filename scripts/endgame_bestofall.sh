#!/usr/bin/env bash
# ENDGAME for the BEST-OF-ALL wave (docs/WAVE_PLAYBOOK.md step 5). Committed and
# IDEMPOTENT: re-running it re-reads artifacts and overwrites its own outputs,
# never the wave's.
#
#   scripts/endgame_bestofall.sh            # everything
#   scripts/endgame_bestofall.sh m3a        # just the M3a injection
#   scripts/endgame_bestofall.sh gates      # just select + bootstrap + gates
set -uo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ZL_OUT:-/mnt/v/output/zensim/best-of-all-2026-09-06}"
BIN="${ZL_BIN:-$OUT/bin}"
export ZEN_PANEL_BIN="${ZEN_PANEL_BIN:-$BIN/panel}"
mkdir -p "$OUT/gates"
say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$OUT/endgame.log"; }

arms() { printf '%s\n' A_plain B_nonneg C_lad05 D_lad20 E_plainlad F_nonneg32; }

do_m3a() {
  # M3a via its OWNER (`run_full_eval.sh` with ZENSIM_M3_ONLY=1, which calls
  # m3a_sweep.sh and injects the means into the existing JSON). Without it
  # `freeze_check --select` lists a cell as UNMEASURED and NOT SELECTABLE — a
  # missing measurement is never scored as zero, which is correct and is why it
  # has to actually be measured.
  #
  # ~66 s/bake and it must run EXCLUSIVELY — never concurrently with fits.
  local ex="$WS/target/release/examples/diffmap_block_coherence"
  if [[ ! -x "$ex" ]]; then
    say "building diffmap_block_coherence (M3a's owner needs it)"
    ( cd "$WS" && nice -n19 ionice -c3 cargo build --release -p zensim \
        --example diffmap_block_coherence --features training,feature-regime-v2,custom-profiles ) \
      || { say "M3a NOT MEASURED — could not build diffmap_block_coherence"; return 3; }
  fi
  for f in "$OUT"/verdicts/*.fulleval.json; do
    local name; name=$(basename "$f" .fulleval.json)
    [[ "$name" == _* ]] && continue
    local bake="$OUT/bakes/${name}_packed.bin"
    [[ -s "$bake" ]] || { say "M3a SKIP $name — no packed bake"; continue; }
    if python3 -c "import json,sys; d=json.load(open('$f')); sys.exit(0 if d.get('m3a_coherence') is not None else 1)"; then
      say "M3a SKIP $name (already measured)"; continue
    fi
    say "M3a $name"
    ZENSIM_M3_ONLY=1 ZENSIM_FULLEVAL_OUT="$OUT/verdicts" \
      nice -n19 ionice -c3 "$WS/scripts/run_full_eval.sh" "$bake" "$name" 372 \
      >> "$OUT/logs/${name}.m3a.log" 2>&1 || say "M3a FAILED $name (reported, not fatal)"
  done
}

do_report() {
  say "report"
  python3 "$WS/scripts/bestofall_report.py" --out "$OUT" --md "$OUT/gates/arm_table.md" \
    > "$OUT/gates/report.txt" 2>&1
  tail -60 "$OUT/gates/report.txt"
}

do_bootstrap() {
  local refs=("$OUT/shipped_D.fulleval.json")
  for L in CTL_A_LSTAR_s4021_legacy_packed LSTAR3__S__i4041_p5001_packed W11J__S__i4013_p5001_packed; do
    local p="/mnt/v/output/zensim/replication-2026-09-05/fulleval/${L}.fulleval.json"
    [[ -f "$p" ]] && refs+=("$p")
  done
  for cand in "$@"; do
    local cf="$OUT/verdicts/${cand}.fulleval.json"
    [[ -f "$cf" ]] || { say "bootstrap SKIP $cand — no fulleval"; continue; }
    for ref in "${refs[@]}"; do
      say "bootstrap $cand vs $(basename "$ref" .fulleval.json)"
      python3 "$WS/scripts/bestofall_bootstrap.py" --a "$cf" --b "$ref" -B 2000 \
        >> "$OUT/gates/bootstrap_${cand}.txt" 2>&1
    done
    cat "$OUT/gates/bootstrap_${cand}.txt"
  done
}

do_inversions() {
  # The TWO-REFERENCE inversion reading lives in `dial.inversion_truth`, which is
  # emitted only by a --full-json run that was ALSO given --reference-truth. The
  # wave's rank verdict has no reference table (it falls back to `single` and
  # SAYS SO — "NOT MEASURABLE … FALLING BACK to `single`"), and the wave's G-ADDR
  # run has the table but writes only --gaddr-json. So capture it here: same
  # ladder instrument, same pnorm3 table, ~2 s per bake.
  local L=/mnt/v/output/zensim/ladder-2026-09-05/instruments
  local P=/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments
  for f in "$OUT"/verdicts/*.fulleval.json; do
    local name; name=$(basename "$f" .fulleval.json)
    [[ "$name" == _* ]] && continue
    local bake="$OUT/bakes/${name}_packed.bin"
    [[ -s "$bake" ]] || continue
    [[ -s "$OUT/gates/${name}.inv.json" ]] && { say "inversions SKIP $name"; continue; }
    say "inversions $name"
    "$BIN/bake_verdict" --bake "$bake" \
      --features-root /mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC \
      --corpora cid22 \
      --dial-grid "$L/dial_grid_372col_ladder.parquet" \
      --gaddr-grid-truth "$L/dialcells_ssim2_ladder.tsv" \
      --floor-rule resolvable --floor-margin 0.5 \
      --reference-truth "$L/reference_truth_ladder_pnorm3.tsv:pnorm3" \
      --inversion-truth agree \
      --negtail-probe "$P/negtail_probe_372_postC_2026-09-05.parquet" \
      --identity-probe "$P/identity_probe_372_postC_2026-09-05.parquet" \
      --name "${name}@inv" --full-json "$OUT/gates/${name}.inv.json" \
      --output /dev/null >> "$OUT/logs/${name}.inv.log" 2>&1 \
      || say "inversions FAILED $name"
  done
  python3 - "$OUT" <<'PY2'
import json, glob, os, sys
out = sys.argv[1]
print(f"{'cell':<22} {'mono_agree':>11} {'mono_single':>12} {'enc_attr':>9} {'unknown':>8}")
for f in sorted(glob.glob(os.path.join(out, "gates", "*.inv.json"))):
    d = json.load(open(f))
    it = (d.get("dial") or {}).get("inversion_truth") or {}
    name = os.path.basename(f)[: -len(".inv.json")]
    print(f"{name:<22} {it.get('mono_dial', float('nan')):>11.5f} "
          f"{it.get('mono_single', float('nan')):>12.5f} "
          f"{it.get('n_encoder_attributed', '—'):>9} "
          f"{it.get('n_attribution_unknown', '—'):>8}   {it.get('effective')}")
PY2
}

do_binparity() {
  # The wave ran on binaries frozen BEFORE the 2026-09-06 review fixes. Every
  # fix is argued to be inert for this configuration (the arms already derive
  # SCORE from their `:both` legs; the pool/hybrid and alpha=1.0 and f32-pin
  # paths are unused; --identity-rows defaults to 0). ARGUED is not MEASURED —
  # retrain one cell with the CURRENT binary and compare bake sha256.
  local name="${1:-B_nonneg_s4004}"
  local arm="${name%_s*}" seed="${name##*_s}"
  local cur="$WS/target/release/zensim_mlp_train"
  [[ -x "$cur" ]] || { say "binparity SKIP — no current binary at $cur"; return 3; }
  say "binparity: retraining $name with the CURRENT binary"
  ZL_BIN="$WS/target/release" ZL_OUT="$OUT/binparity" \
    "$WS/scripts/bestofall_wave.sh" cell "$arm" "$seed" \
    > "$OUT/logs/binparity_${name}.log" 2>&1 || true
  local a="$OUT/bakes/${name}.bin" b="$OUT/binparity/bakes/${name}.bin"
  if [[ -s "$a" && -s "$b" ]]; then
    local sa sb; sa=$(sha256sum "$a" | cut -d' ' -f1); sb=$(sha256sum "$b" | cut -d' ' -f1)
    if [[ "$sa" == "$sb" ]]; then
      say "binparity PASS — $name is BYTE-IDENTICAL across the review fixes ($sa)"
    else
      say "binparity FAIL — $name MOVED: frozen $sa vs current $sb"
    fi
  else
    say "binparity INCONCLUSIVE — missing $a or $b"
  fi
}

do_board() {
  # Promote the wave's cells onto the summer gauntlet. `promote_fulleval.py`
  # RELABELS and annotates; it recomputes nothing, and every stat block is
  # asserted byte-identical to the source verdict.
  #
  # ⚠ The board name carries a `BOA_` prefix. The lane's arm names are
  # `A_plain` / `B_nonneg` / …, and `gauntlet.family_of` already claims `A_` for
  # "arm A" and `B_` for "arm B" — 944 campaign families. Promoting under the raw
  # names would silently file a 228-slot constrained-MLP cell into a 944
  # campaign's toggle group, with no error.
  local BOARD=${ZL_BOARD:-/mnt/v/output/zensim/reports/fulleval}
  for f in "$OUT"/verdicts/*.fulleval.json; do
    local name; name=$(basename "$f" .fulleval.json)
    [[ "$name" == _* ]] && continue
    local g="$OUT/gaddr/gaddr_${name}.json"
    # Two steps, because `--graft-gaddr` is its own MODE (it takes
    # `--graft-into`, not `--verdict`): promote the cell, then attach the G-ADDR
    # block to the promoted file so the board's dial columns come from the same
    # run that produced them.
    say "promote BOA_${name}"
    python3 "$WS/scripts/promote_fulleval.py" --verdict "$f" --name "BOA_${name}" \
      --out-dir "$BOARD" >> "$OUT/gates/board_promote.log" 2>&1 \
      || { say "promote FAILED BOA_${name}"; continue; }
    if [[ -s "$g" ]]; then
      python3 "$WS/scripts/promote_fulleval.py" --graft-gaddr "$g" \
        --graft-into "$BOARD/BOA_${name}.fulleval.json" \
        >> "$OUT/gates/board_promote.log" 2>&1 \
        || say "gaddr graft FAILED BOA_${name}"
    fi
  done
  say "board promotion done — regen with scripts/v_next/bandwise_dashboard.py --fulleval-dir $BOARD"
}

do_select() {
  say "freeze_check --select --seed-group --min-k 2 --floor-basis all"
  "$BIN/freeze_check" --select "$OUT"/verdicts/*.fulleval.json \
      --seed-group --min-k 2 --floor-basis all > "$OUT/gates/select.txt" 2>&1
  say "select exit=$?"
  tail -40 "$OUT/gates/select.txt"
}

case "${1:-all}" in
  m3a)   do_m3a ;;
  inversions) do_inversions ;;
  board) do_board ;;
  binparity) shift; do_binparity "$@" ;;
  report) do_report ;;
  select) do_select ;;
  bootstrap) shift; do_bootstrap "$@" ;;
  gates) do_report; do_inversions; do_select ;;
  all)
    do_report
    do_inversions
    do_m3a
    do_select
    ;;
  *) echo "usage: $0 {all|report|m3a|select|bootstrap <cand...>|gates}" >&2; exit 2 ;;
esac
say "ENDGAME DONE ($1)"
