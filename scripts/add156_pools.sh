#!/usr/bin/env bash
#
# add156_pools.sh — the REGISTERED APPENDIX T pool grid
# (benchmarks/sota944_campaign_2026-08-03.md "APPENDIX T"; the protocol was
# committed BEFORE this ran).
#
# ONE additive recipe (ADD156's: safesyn-only, raw space, lasso, tau 0, f16,
# spline on the packed forward), several COORDINATE POOLS, two roots:
#
#   root A = the ADD156-era v1-372 root   pools: a (f0-155), b (f0-371)
#   root B = the current ext944 root      pools: a944, b944, c944, d944
#
# The solver is deterministic coordinate descent — no RNG, so no seeds (T.2).
# Root-A cells are 372-input bakes and are evaluated at `--regime 372` (their
# NATIVE root); root-B cells are 944-input and evaluated at `--regime 944`.
# Scoring a root-A `b` cell at the 944 root would feed it the folded regime's
# STRUCTURAL ZEROS for exactly the f156-371 block it relies on — the trap this
# split exists to avoid.
#
#   scripts/add156_pools.sh fit      # 24 fits (+ fit npz + slice files)
#   scripts/add156_pools.sh eval     # bake_verdict --full-json per cell
#   scripts/add156_pools.sh tsv      # collate the grid TSV
#   scripts/add156_pools.sh          # all three, in order
#
# Env: ZL_BIN (bake_dial_refit), ZL_BV (bake_verdict) — default to
# CARGO_TARGET_DIR/release, else this repo's target/release. NEVER a hardcoded
# sibling-worktree path (CLAUDE.md).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TGT=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
BDR=${ZL_BIN:-$TGT/release/bake_dial_refit}
BV=${ZL_BV:-$TGT/release/bake_verdict}

LP=/mnt/v/output/zensim-multicodec-probe/linear-probe          # root A frozen artifacts
E944=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01     # root B
OUT=/mnt/v/output/zensim/bakes/add156repro
G=$OUT/grams
B=$OUT/bakes
VD=$OUT/verdicts
SL=$OUT/slices
LOG=${ZL_LOG:-$HOME/tmp/add156repro}
mkdir -p "$B" "$VD" "$SL" "$LOG"
NI=(nice -n 19 ionice -c 3)

command -v jq >/dev/null || { echo "jq required" >&2; exit 2; }
[[ -x "$BDR" ]] || { echo "missing $BDR" >&2; exit 2; }
[[ -x "$BV" ]] || { echo "missing $BV" >&2; exit 2; }

# Registered lambda grid (T.2): ADD156's 2e-3 plus 3 neighbours.
LAMS=(3e-4 1e-3 2e-3 5e-3)

# pool -> "root:width:hi"  (hi = last coordinate index in the pool, inclusive)
# root A pools are named a/b; root B pools carry the 944 suffix.
pool_spec() {
    case $1 in
        a)    echo "A 372 155" ;;
        b)    echo "A 372 371" ;;
        a944) echo "B 944 155" ;;
        b944) echo "B 944 371" ;;
        c944) echo "B 944 719" ;;
        d944) echo "B 944 943" ;;
        *) echo "unknown pool $1" >&2; exit 2 ;;
    esac
}
POOLS=(a b a944 b944 c944 d944)

slice_file() {   # $1 = pool ; emits path (built on demand, never for a full pool)
    local pool=$1; read -r _root width hi <<<"$(pool_spec "$pool")"
    local f="$SL/${pool}.idx"
    if [[ $((hi + 1)) -eq $width ]]; then echo ""; return; fi   # full pool: no slice
    [[ -f $f ]] || seq 0 "$hi" > "$f"
    echo "$f"
}

do_fit() {
    local pool=$1 lam=$2
    read -r root _width _hi <<<"$(pool_spec "$pool")"
    local stem="T_${pool}_lam${lam}"
    local bin="$B/$stem.bin" npz="$B/$stem.fit.npz"
    [[ -f $bin ]] && { echo "[fit] $stem cached"; return; }
    local args=(fit-lasso --space raw --target human_score --lam "$lam" --tau 0
                --n-sweeps 400 --tol 1e-10 --emit-fit-npz "$npz" --out "$bin")
    local sf; sf=$(slice_file "$pool")
    [[ -n $sf ]] && args+=(--slice-file "$sf")
    if [[ $root == A ]]; then
        args+=(--gram "$LP/grams/safesyn.npz" --weight 1.0 --anchor "$LP/val/anchor.npz")
    else
        args+=(--gram "$G/e944_safesyn.npz" --weight 1.0
               --anchor-parquet "$E944/anchor944_dial.parquet" --anchor-target target_score)
    fi
    echo "== fit $stem =="
    "${NI[@]}" "$BDR" "${args[@]}" 2>&1 | tee "$LOG/$stem.fit.log"
}

do_eval() {
    local pool=$1 lam=$2
    read -r root _width _hi <<<"$(pool_spec "$pool")"
    local stem="T_${pool}_lam${lam}"
    local bin="$B/$stem.bin" vj="$VD/$stem.full.json"
    [[ -f $vj ]] && { echo "[eval] $stem cached"; return; }
    local regime=372; [[ $root == B ]] && regime=944
    local extra=()
    # The 372 root predates the sdr25 + hfnlproxy corpora (they exist only as
    # ext720/ext944 extractions), and `--regime 372`'s default list now asks
    # for them. Name the corpora the 372 root actually HAS; the two missing
    # axes are reported ABSENT for root A, never as a failure (T.4 rule).
    [[ $regime == 372 ]] && extra=(--corpora
        cid22,kadid,tid,csiq,pipal,live,konjnd,aic3,aic4,nonphoto,imazen26,hf_nearlossless)
    echo "== eval $stem (regime $regime) =="
    "${NI[@]}" "$BV" --bake "$bin" --regime "$regime" "${extra[@]}" --full-json "$vj" \
        > "$VD/$stem.verdict.md" 2> "$LOG/$stem.eval.log" || {
            echo "FAIL eval $stem — see $LOG/$stem.eval.log" >&2; return 1; }
}

cmd_fit()  { for p in "${POOLS[@]}"; do for l in "${LAMS[@]}"; do do_fit  "$p" "$l"; done; done; }
cmd_eval() { for p in "${POOLS[@]}"; do for l in "${LAMS[@]}"; do do_eval "$p" "$l"; done; done; }

# --- per-pair dumps + the paired bootstrap (T.3 noise instrument) -------------
# One dump per (cell, corpus); `bake_verdict --per-pair-output` writes the LAST
# `--corpora` entry's rows, so each dump is a single-corpus invocation. The
# statistic is NEVER computed here: `scripts/wave6_paired_bootstrap.py` (the
# registered paired-bootstrap instrument) reduces the dumps through
# `panel --batch`.
DUMP=$OUT/perpair
BOOT_CORPORA_A=${BOOT_CORPORA_A:-cid22 konjnd nonphoto imazen26 csiq live}
BOOT_CORPORA_B=${BOOT_CORPORA_B:-cid22 konjnd nonphoto imazen26 csiq live}

do_dump() {   # $1 = cell stem (T_<pool>_lam<l>) ; $2 = pool ; $3 = corpus
    local stem=$1 pool=$2 corpus=$3
    read -r root _w _hi <<<"$(pool_spec "$pool")"
    local regime=372; [[ $root == B ]] && regime=944
    local f="$DUMP/${stem}_${corpus}.tsv"
    [[ -f $f ]] && return
    "${NI[@]}" "$BV" --bake "$B/$stem.bin" --regime "$regime" --corpora "$corpus" \
        --per-pair-output "$f" > /dev/null 2>> "$LOG/dump.log" || {
            echo "dump FAIL $stem/$corpus" >&2; return 1; }
}

cmd_dump() {
    mkdir -p "$DUMP"
    local cells=${ZL_BOOT_CELLS:?set ZL_BOOT_CELLS="pool:lam pool:lam ..."}
    for spec in $cells; do
        local pool=${spec%%:*} lam=${spec##*:}
        read -r root _w _hi <<<"$(pool_spec "$pool")"
        local list=("${BOOT_CORPORA_A[@]}"); [[ $root == B ]] && list=("${BOOT_CORPORA_B[@]}")
        for c in "${list[@]}"; do
            echo "== dump T_${pool}_lam${lam} / $c =="
            do_dump "T_${pool}_lam${lam}" "$pool" "$c"
        done
    done
}

cmd_tsv() {
    local tsv=$OUT/pool_grid.tsv
    { printf 'pool\troot\tregime\tlam\tn_active\tn_active_gt155\tcid22\tkonjnd\tnonphoto\timazen26\tcsiq\tlive\taic3\taic4\thfnl_perref\tkadid_true\ttid\tdial_mono\tdial_span\tbytes\tsha12\n'
      for p in "${POOLS[@]}"; do
        read -r root _w _hi <<<"$(pool_spec "$p")"
        local regime=372; [[ $root == B ]] && regime=944
        for l in "${LAMS[@]}"; do
          local stem="T_${p}_lam${l}" vj="$VD/T_${p}_lam${l}.full.json" bin="$B/T_${p}_lam${l}.bin"
          [[ -f $vj ]] || continue
          # n_active / >f155 survivors from the fit npz (the weights, not the bake)
          read -r na ng <<<"$(python3 - "$B/$stem.fit.npz" <<'PY'
import sys, numpy as np
w = np.load(sys.argv[1])["w"]
nz = np.nonzero(w)[0]
print(len(nz), int((nz > 155).sum()))
PY
)"
          jq -r --arg p "$p" --arg root "$root" --arg regime "$regime" --arg lam "$l" \
                --arg na "$na" --arg ng "$ng" \
                --arg sz "$(stat -c%s "$bin")" --arg sha "$(sha256sum "$bin" | cut -c1-12)" '
            def g(c;f): (.rank[c][f] // "") | tostring;
            [$p,$root,$regime,$lam,$na,$ng,
             g("cid22";"srocc"), g("konjnd";"srocc"), g("nonphoto";"srocc"),
             g("imazen26";"srocc"), g("csiq";"srocc"), g("live";"srocc"),
             g("aic3";"srocc"), g("aic4";"srocc"), g("hfnlproxy";"per_ref_mean"),
             ((.rank.kadid.srocc_signed // 0) * -1 | tostring), g("tid";"srocc"),
             (.dial.mono_pct // "" | tostring), (.dial.dynamic_range // "" | tostring),
             $sz, $sha] | @tsv' "$vj"
        done
      done
    } > "$tsv"
    echo "wrote $tsv"; column -t -s$'\t' "$tsv"
}

case "${1:-all}" in
    fit) cmd_fit ;;
    eval) cmd_eval ;;
    tsv) cmd_tsv ;;
    dump) cmd_dump ;;
    all) cmd_fit; cmd_eval; cmd_tsv ;;
    *) echo "usage: $0 [fit|eval|tsv|dump|all]" >&2; exit 2 ;;
esac
