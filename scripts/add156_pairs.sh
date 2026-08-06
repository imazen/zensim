#!/usr/bin/env bash
#
# add156_pairs.sh — the REGISTERED APPENDIX U grid
# (benchmarks/sota944_campaign_2026-08-03.md "APPENDIX U"; the protocol was
# committed BEFORE this ran).
#
# ONE constrained operator: base = ADD156's 28 live coordinates; a cell is the
# SAME lasso re-solved over `base ∪ C` where C is a singleton or a pair drawn
# from ABOVE f155 (inside f0..155 is a provable no-op — appendix U P3/G-U1).
#
#   arm A = root A (v1-372), candidates f156..371  (peak72/masked72/iw72)
#   arm B = root B mm01 (folded-944), candidates f372..943 (v2-348/append/append2)
#
# The solver is deterministic coordinate descent — no RNG, so no seeds.
# PRIMARY sweeps are CONVERGED (CD exits on tol, so the cap is free); the
# recipe-faithful 400-sweep setting is the registered cross-check.
#
#   scripts/add156_pairs.sh enum          # write the cell manifest
#   scripts/add156_pairs.sh fit  <lam> <ns>   # fit every cell
#   scripts/add156_pairs.sh eval <lam> <ns>   # full battery on LIVE cells only
#   scripts/add156_pairs.sh tsv  <lam> <ns>   # collate the grid TSV
#
# LIVE/ZERO is decided on the CANDIDATE COEFFICIENTS in the fit npz, never on
# the bake sha. Offering the solver one extra coordinate perturbs the CD path,
# so a cell whose candidate lands at exactly 0.0 still differs from base by
# ~5e-11 on the other weights (sub-`tol` drift) and therefore by a few bake
# bytes. A sha test would call that LIVE; it is not. The ZERO cells are in turn
# the appendix's empirical NULL: a genuinely null intervention whose measured
# delta distribution sizes the floors (appendix U §U.5).
#
# Env: ZL_BIN (bake_dial_refit), ZL_BV (bake_verdict) — default to
# CARGO_TARGET_DIR/release, else this repo's target/release. NEVER a hardcoded
# sibling-worktree path (CLAUDE.md).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TGT=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
BDR=${ZL_BIN:-$TGT/release/bake_dial_refit}
BV=${ZL_BV:-$TGT/release/bake_verdict}

LP=/mnt/v/output/zensim-multicodec-probe/linear-probe            # root A frozen artifacts
E944=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01       # root B anchor
GRAMB=/mnt/v/output/zensim/bakes/add156repro/grams/e944_safesyn_mm01.npz  # root B mm01 gram (T.A1)
OUT=/mnt/v/output/zensim/bakes/add156pairs
B=$OUT/bakes
VD=$OUT/verdicts
FITS=$OUT/fits
SL=$OUT/slices
LOG=${ZL_LOG:-$HOME/tmp/add156pairs}
MAN=$OUT/cells.tsv
mkdir -p "$B" "$VD" "$FITS" "$SL" "$LOG"
NI=(nice -n 19 ionice -c 3)
JOBS=${ZL_JOBS:-5}

command -v jq >/dev/null || { echo "jq required" >&2; exit 2; }
[[ -x "$BDR" ]] || { echo "missing $BDR" >&2; exit 2; }
[[ -x "$BV" ]] || { echo "missing $BV" >&2; exit 2; }

# ADD156's 28 live coordinates (appendix U U.2; from the era fit npz).
BASE28="6 8 11 14 17 19 22 24 26 34 37 89 91 93 94 116 120 121 122 124 128 136 137 138 140 146 150 155"

tag() { echo "${1//e-/e}"; }   # lam 2e-3 -> 2e3, for filenames

# ---------------------------------------------------------------- enum -------
cmd_enum() {
    python3 "$REPO_ROOT/scripts/add156_pairs_enum.py" --out "$MAN"
    echo "wrote $MAN"
    awk -F'\t' 'NR>1{n[$2"/"$3]++} END{for(k in n) print k, n[k]}' "$MAN" | sort
    echo "total cells: $(( $(wc -l < "$MAN") - 1 ))"
}

# ----------------------------------------------------------------- fit -------
# $1 cell_id  $2 arm  $3 indices(comma)  $4 lam  $5 n_sweeps
do_fit() {
    local cid=$1 arm=$2 idxs=$3 lam=$4 ns=$5
    local t; t=$(tag "$lam")
    local stem="U_${arm}${t}s${ns}_${cid}"
    local bin="$B/$stem.bin"
    [[ -f $bin && -f "$FITS/$stem.npz" ]] && return 0
    local sf="$SL/$stem.idx"
    # NB: an `[[ ]] && cmd` as the group's last statement makes the group exit 1
    # on the false branch, which `set -e -o pipefail` then treats as a failed
    # pipeline — hence the explicit `if`.
    { printf '%s\n' $BASE28
      if [[ $idxs != "-" ]]; then tr ',' '\n' <<<"$idxs"; fi
    } | sort -n -u > "$sf"
    local args=(fit-lasso --space raw --lam "$lam" --tau 0
                --n-sweeps "$ns" --tol 1e-10 --slice-file "$sf" --out "$bin"
                --emit-fit-npz "$FITS/$stem.npz")
    if [[ $arm == A ]]; then
        args+=(--gram "$LP/grams/safesyn.npz" --weight 1.0 --target human_score
               --anchor "$LP/val/anchor.npz")
    else
        args+=(--gram "$GRAMB" --weight 1.0 --target human_score__mm01
               --anchor-parquet "$E944/anchor944_dial.parquet" --anchor-target target_score)
    fi
    "${NI[@]}" "$BDR" "${args[@]}" >>"$LOG/fit_${arm}${t}s${ns}.log" 2>&1 \
        || { echo "FIT FAIL $stem" >&2; return 1; }
    rm -f "$sf"
}
export -f do_fit tag
export B SL FITS LOG BDR LP E944 GRAMB BASE28

cmd_fit() {
    local lam=$1 ns=$2 t; t=$(tag "$lam")
    # base controls first: cmd_live needs both arms' base fits to classify against.
    do_fit BASE A - "$lam" "$ns"
    do_fit BASE B - "$lam" "$ns"
    tail -n +2 "$MAN" | awk -F'\t' '{print $1"\t"$2"\t"$5}' \
      | xargs -P "$JOBS" -n 1 -d '\n' bash -c '
          IFS=$'"'"'\t'"'"' read -r cid arm idxs <<<"$0"
          do_fit "$cid" "$arm" "$idxs" '"$lam $ns"'' \
      || { echo "some fits failed — see $LOG/fit_*" >&2; exit 1; }
    echo "[fit] lam=$lam ns=$ns done: $(ls "$B"/U_?"${t}"s"${ns}"_*.bin 2>/dev/null | wc -l) bakes"
}

# ---------------------------------------------------------------- live -------
# LIVE = at least one CANDIDATE coefficient is nonzero in the fit npz.
# NOT a bake-sha test: see the header note on sub-`tol` CD drift.
cmd_live() {
    local lam=$1 ns=$2 t; t=$(tag "$lam")
    python3 "$REPO_ROOT/scripts/add156_pairs_live.py" \
        --manifest "$MAN" --fits "$FITS" --tag "${t}s${ns}" \
        --out "$OUT/live_${t}s${ns}.tsv"
}

# ---------------------------------------------------------------- eval -------
do_eval() {
    local stem=$1
    local vj="$VD/$stem.full.json"
    [[ -s $vj ]] && return 0
    "${NI[@]}" "$BV" --bake "$B/$stem.bin" --regime 944 \
        --perpair-metrics /nonexistent/skip-scatter.parquet --full-json "$vj" \
        > /dev/null 2>>"$LOG/eval.log" || { echo "EVAL FAIL $stem" >&2; return 1; }
}
export -f do_eval
export VD BV

# NULLS: every Nth ZERO cell is evaluated too. Its candidate coefficients are
# exactly 0.0, so it is the base model perturbed only by sub-`tol` CD drift —
# a genuinely null intervention that goes through the identical fit/pack/spline/
# score chain. The spread of its deltas IS the appendix's measured noise floor
# (U.5), rather than a floor assumed from another study's axis.
NULL_STRIDE=${ZL_NULL_STRIDE:-9}

cmd_eval() {
    local lam=$1 ns=$2 t; t=$(tag "$lam")
    local f=$OUT/live_${t}s${ns}.tsv
    { echo -e "BASE\tA"; echo -e "BASE\tB"
      awk -F'\t' -v s="$NULL_STRIDE" 'NR>1 && ($3=="LIVE" || (++z) % s == 0){print $1"\t"$2}' "$f"
    } | awk -F'\t' -v t="$t" -v ns="$ns" '{print "U_"$2 t "s" ns "_" $1}' \
      | xargs -P "$JOBS" -n 1 bash -c 'do_eval "$0"' \
      || { echo "some evals failed — see $LOG/eval.log" >&2; exit 1; }
    echo "[eval] lam=$lam ns=$ns done"
}

# ----------------------------------------------------------------- tsv -------
cmd_tsv() {
    local lam=$1 ns=$2 t; t=$(tag "$lam")
    python3 "$REPO_ROOT/scripts/add156_pairs_collate.py" \
        --manifest "$MAN" --verdicts "$VD" --live "$OUT/live_${t}s${ns}.tsv" \
        --tag "${t}s${ns}" --out "$OUT/grid_${t}s${ns}.tsv"
}

case "${1:?usage: $0 enum|fit|live|eval|tsv [lam ns]}" in
    enum) cmd_enum ;;
    fit)  cmd_fit  "${2:?lam}" "${3:?n_sweeps}" ;;
    live) cmd_live "${2:?lam}" "${3:?n_sweeps}" ;;
    eval) cmd_eval "${2:?lam}" "${3:?n_sweeps}" ;;
    tsv)  cmd_tsv  "${2:?lam}" "${3:?n_sweeps}" ;;
    all)  cmd_enum; cmd_fit "${2:?lam}" "${3:?n_sweeps}"; cmd_live "$2" "$3"
          cmd_eval "$2" "$3"; cmd_tsv "$2" "$3" ;;
    *) echo "usage: $0 enum|fit|live|eval|tsv|all [lam n_sweeps]" >&2; exit 2 ;;
esac
