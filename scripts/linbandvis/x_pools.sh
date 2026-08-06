#!/usr/bin/env bash
#
# x_pools.sh — APPENDIX X Thread 1: the best-informed additive/linear grid
# (benchmarks/sota944_campaign_2026-08-03.md "APPENDIX X"; registered BEFORE
# this ran). Modeled on the committed add156_pools.sh (appendix T's driver).
#
# ONE recipe held at ADD156's values (raw feature space, lasso CD, tau 0, f16,
# spline on the packed forward), varied on four registered axes:
#   pools:  a944 (f0-155) | c944 (f0-719) | cb944 (f0-719 + 8 BANDVIS lanes)
#           a (root A f0-155) | b (root A f0-371)
#   mixes:  Ms (safesyn 1.0) | M4 (safesyn 1.0 + cid22t201 1.0 + kadid 0.5 + tid 0.5)
#   conv:   mm (mm01 targets, primary) | rw (raw x100 clip -100; c944 x M4 only)
#   lambda: 3e-4 1e-3 2e-3 5e-3 1e-2   (T's band + ONE registered edge extension)
# Solver: lasso, converged sweeps (200000) PRIMARY; plus 2 BVLS cells
# (c944 x {Ms,M4}, sign-mask bounds). Deterministic CD — no RNG, no seeds.
#
# KADID grams are REBUILT here from the corrected ext_kadid.parquet
# (sha 286f1b23..., wave-10 fix); safesyn/cid22t201/tid root-B grams are
# rebuilt into this appendix's dir for one-commit provenance and gated
# byte-identical against the frozen T/sota944 grams (determinism check).
#
#   x_pools.sh grams | fit | eval | tsv | dump | all
#
# Env: ZL_BIN (bake_dial_refit), ZL_BV (bake_verdict), ZL_LOG.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TGT=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
BDR=${ZL_BIN:-$TGT/release/bake_dial_refit}
BV=${ZL_BV:-$TGT/release/bake_verdict}

LP=/mnt/v/output/zensim-multicodec-probe/linear-probe           # root A frozen artifacts
E944=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01      # root B canonical
OUT=/mnt/v/output/zensim/bakes/linbandvis
G=$OUT/grams
B=$OUT/bakes
VD=$OUT/verdicts
SL=$OUT/slices
DUMP=$OUT/perpair
LOG=${ZL_LOG:-$HOME/tmp/linbandvis}
mkdir -p "$G" "$B" "$VD" "$SL" "$DUMP" "$LOG"
NI=(nice -n 19 ionice -c 3)

command -v jq >/dev/null || { echo "jq required" >&2; exit 2; }
[[ -x "$BDR" ]] || { echo "missing $BDR" >&2; exit 2; }
[[ -x "$BV" ]] || { echo "missing $BV" >&2; exit 2; }

LAMS=(3e-4 1e-3 2e-3 5e-3 1e-2)
SWEEPS=${ZL_SWEEPS:-200000}          # converged primary (X.2/P3)
SIGNMASK=$REPO_ROOT/benchmarks/feature_sign_mask_2026-05-26.tsv

# ---------------------------------------------------------------- slices ----
build_slices() {
    [[ -f $SL/a944.idx ]] || seq 0 155 > "$SL/a944.idx"
    [[ -f $SL/c944.idx ]] || seq 0 719 > "$SL/c944.idx"
    if [[ ! -f $SL/cb944.idx ]]; then
        { seq 0 719; printf '%s\n' 924 925 929 930 934 935 939 940; } > "$SL/cb944.idx"
    fi
    [[ -f $SL/a.idx ]] || seq 0 155 > "$SL/a.idx"
    [[ -f $SL/b.idx ]] || seq 0 371 > "$SL/b.idx"
}

# ---------------------------------------------------------------- grams -----
# Root-B grams from the canonical tables (mm01 + raw conventions).
build_gram() {  # $1 name  $2 parquet  $3 conv(mm|rw)
    local name=$1 pq=$2 conv=$3 out="$G/xg_${name}_${conv}.npz"
    [[ -f $out ]] && { echo "[gram] $name/$conv cached"; return; }
    local args=(gram --parquet "$pq" --target human_score --space raw
                --expect-n-feat 944 --out "$out")
    case $conv in
        mm) args+=(--target-minmax01) ;;
        rw) args+=(--target-scale 100 --target-clip-min -100) ;;
    esac
    echo "== gram $name/$conv =="
    "${NI[@]}" "$BDR" "${args[@]}" 2>&1 | tee "$LOG/gram_${name}_${conv}.log"
}

cmd_grams() {
    build_slices
    for c in mm rw; do
        build_gram safesyn   "$E944/ext_safesyn_full.parquet"   $c
        build_gram cid22t201 "$E944/ext_cid22_train201.parquet" $c
        build_gram kadid     "$E944/ext_kadid.parquet"          $c
        build_gram tid       "$E944/ext_tid.parquet"            $c
    done
    # Determinism gate: my safesyn_mm01 must be byte-identical to the frozen
    # appendix-T gram (same table, same owner, 0 RNG). Report loudly either way.
    for ref in /mnt/v/output/zensim/bakes/add156repro/grams/e944_safesyn_mm01.npz \
               /mnt/v/output/zensim/bakes/sota944/grams/safesyn_mm01.npz; do
        if [[ -f $ref ]]; then
            if cmp -s "$G/xg_safesyn_mm.npz" "$ref"; then
                echo "GRAM-DETERMINISM PASS: xg_safesyn_mm == $(basename "$ref")"
            else
                echo "GRAM-DETERMINISM NOTE: xg_safesyn_mm != $(basename "$ref") (investigate before fits)"
            fi
        fi
    done
}

# ------------------------------------------------------------------ fits ----
# mix -> gram args for a given root/convention
mix_grams() {  # $1 root(A|B) $2 mix(Ms|M4) $3 conv(mm|rw)
    local root=$1 mix=$2 conv=$3
    if [[ $root == A ]]; then
        case $mix in
            Ms) echo "--gram $LP/grams/safesyn.npz --weight 1.0" ;;
            M4) echo "--gram $LP/grams/safesyn.npz --weight 1.0 --gram $LP/grams/cid22_train.npz --weight 1.0 --gram $LP/grams/kadid.npz --weight 0.5 --gram $LP/grams/tid.npz --weight 0.5" ;;
        esac
    else
        case $mix in
            Ms) echo "--gram $G/xg_safesyn_${conv}.npz --weight 1.0" ;;
            M4) echo "--gram $G/xg_safesyn_${conv}.npz --weight 1.0 --gram $G/xg_cid22t201_${conv}.npz --weight 1.0 --gram $G/xg_kadid_${conv}.npz --weight 0.5 --gram $G/xg_tid_${conv}.npz --weight 0.5" ;;
        esac
    fi
}

pool_root() { case $1 in a|b) echo A ;; *) echo B ;; esac; }

do_fit() {  # $1 pool  $2 mix  $3 conv  $4 lam  [$5 solver]
    local pool=$1 mix=$2 conv=$3 lam=$4 solver=${5:-lasso}
    local root; root=$(pool_root "$pool")
    local stem="X_${pool}_${mix}_${conv}_lam${lam}"
    [[ $solver == bvls ]] && stem="X_${pool}_${mix}_${conv}_bvls"
    local bin="$B/$stem.bin" npz="$B/$stem.fit.npz"
    [[ -f $bin ]] && { echo "[fit] $stem cached"; return; }
    local tgt=human_score
    [[ $root == B && $conv == mm ]] && tgt=human_score__mm01
    local args=(fit-lasso --space raw --target "$tgt" --tau 0
                --n-sweeps "$SWEEPS" --tol 1e-10
                --emit-fit-npz "$npz" --out "$bin" --slice-file "$SL/${pool}.idx")
    if [[ $solver == bvls ]]; then
        args+=(--solver bvls --bounds-tsv "$SIGNMASK" --lam 0)
    else
        args+=(--lam "$lam")
    fi
    # shellcheck disable=SC2046
    args+=($(mix_grams "$root" "$mix" "$conv"))
    if [[ $root == A ]]; then
        args+=(--anchor "$LP/val/anchor.npz")
    else
        args+=(--anchor-parquet "$E944/anchor944_dial.parquet" --anchor-target target_score)
    fi
    echo "== fit $stem =="
    "${NI[@]}" "$BDR" "${args[@]}" 2>&1 | tee "$LOG/$stem.fit.log"
}

cmd_fit() {
    build_slices
    # root B lasso: 3 pools x 2 mixes x 5 lams (mm01 primary)
    for p in a944 c944 cb944; do for m in Ms M4; do for l in "${LAMS[@]}"; do
        do_fit "$p" "$m" mm "$l"
    done; done; done
    # root A lasso: 2 pools x 2 mixes x 5 lams
    for p in a b; do for m in Ms M4; do for l in "${LAMS[@]}"; do
        do_fit "$p" "$m" mm "$l"
    done; done; done
    # raw-convention robustness family: c944 x M4 x 5 lams
    for l in "${LAMS[@]}"; do do_fit c944 M4 rw "$l"; done
    # BVLS cells
    do_fit c944 Ms mm 0 bvls
    do_fit c944 M4 mm 0 bvls
}

# ------------------------------------------------------------------ eval ----
CORPORA_A="cid22,kadid,tid,csiq,pipal,live,konjnd,aic3,aic4,nonphoto,imazen26,hf_nearlossless"

do_eval() {  # $1 stem  $2 root
    local stem=$1 root=$2
    local bin="$B/$stem.bin" vj="$VD/$stem.full.json"
    [[ -f $vj ]] && { echo "[eval] $stem cached"; return; }
    local regime=944 extra=()
    if [[ $root == A ]]; then regime=372; extra=(--corpora "$CORPORA_A"); fi
    echo "== eval $stem (regime $regime) =="
    "${NI[@]}" "$BV" --bake "$bin" --regime "$regime" "${extra[@]}" --full-json "$vj" \
        > "$VD/$stem.verdict.md" 2> "$LOG/$stem.eval.log" || {
            echo "FAIL eval $stem — see $LOG/$stem.eval.log" >&2; return 1; }
}

all_stems() {
    for p in a944 c944 cb944; do for m in Ms M4; do for l in "${LAMS[@]}"; do
        echo "X_${p}_${m}_mm_lam${l} B"; done; done; done
    for p in a b; do for m in Ms M4; do for l in "${LAMS[@]}"; do
        echo "X_${p}_${m}_mm_lam${l} A"; done; done; done
    for l in "${LAMS[@]}"; do echo "X_c944_M4_rw_lam${l} B"; done
    echo "X_c944_Ms_mm_bvls B"
    echo "X_c944_M4_mm_bvls B"
}

cmd_eval() {
    local jobs=${ZL_EVAL_JOBS:-4} n=0
    while read -r stem root; do
        do_eval "$stem" "$root" &
        n=$((n+1)); [[ $((n % jobs)) -eq 0 ]] && wait
    done < <(all_stems)
    wait
}

# ------------------------------------------------------------------ dump ----
read -r -a BOOT_CORPORA <<<"${ZL_BOOT_CORPORA:-cid22 konjnd nonphoto imazen26 csiq live}"
do_dump() {  # $1 stem  $2 root  $3 corpus
    local stem=$1 root=$2 corpus=$3 f="$DUMP/${stem}_${corpus}.tsv"
    [[ -f $f ]] && return
    local regime=944 extra=()
    [[ $root == A ]] && regime=372
    "${NI[@]}" "$BV" --bake "$B/$stem.bin" --regime "$regime" --corpora "$corpus" \
        --per-pair-output "$f" > /dev/null 2>> "$LOG/dump.log" || {
            echo "dump FAIL $stem/$corpus" >&2; return 1; }
}
cmd_dump() {
    local cells=${ZL_BOOT_CELLS:?set ZL_BOOT_CELLS="stem:root stem:root ..."}
    for spec in $cells; do
        local stem=${spec%%:*} root=${spec##*:}
        for c in "${BOOT_CORPORA[@]}"; do
            echo "== dump $stem / $c =="
            do_dump "$stem" "$root" "$c"
        done
    done
}

# ------------------------------------------------------------------- tsv ----
cmd_tsv() {
    local tsv=$OUT/x_grid.tsv
    { printf 'stem\troot\tregime\tpool\tmix\tconv\tlam\tn_active\tn_basic\tn_v1pool\tn_v2\tn_append\tn_append2\tcid22\tkonjnd\tnonphoto\timazen26\tcsiq\tlive\taic3\taic4\tsdr25\thfnl_perref\tkadid_signed\ttid\tdial_mono\tdial_span\tbytes\tsha12\n'
      while read -r stem root; do
        local vj="$VD/$stem.full.json" bin="$B/$stem.bin" npzf="$B/$stem.fit.npz"
        [[ -f $vj && -f $npzf ]] || continue
        local regime=944; [[ $root == A ]] && regime=372
        # census per block from the fit npz
        read -r na nb nv1 nv2 nap na2 <<<"$(python3 - "$npzf" <<'PY'
import sys, numpy as np
w = np.load(sys.argv[1])["w"]
nz = np.nonzero(w)[0]
blk = lambda lo, hi: int(((nz >= lo) & (nz <= hi)).sum())
print(len(nz), blk(0,155), blk(156,371), blk(372,719), blk(720,923), blk(924,943))
PY
)"
        local pool mix conv lam
        IFS=_ read -r _x pool mix conv lam <<<"$stem"
        jq -r --arg stem "$stem" --arg root "$root" --arg regime "$regime" \
              --arg pool "$pool" --arg mix "$mix" --arg conv "$conv" --arg lam "${lam#lam}" \
              --arg na "$na" --arg nb "$nb" --arg nv1 "$nv1" --arg nv2 "$nv2" \
              --arg nap "$nap" --arg na2 "$na2" \
              --arg sz "$(stat -c%s "$bin")" --arg sha "$(sha256sum "$bin" | cut -c1-12)" '
            def g(c;f): (.rank[c][f] // "") | tostring;
            [$stem,$root,$regime,$pool,$mix,$conv,$lam,$na,$nb,$nv1,$nv2,$nap,$na2,
             g("cid22";"srocc"), g("konjnd";"srocc"), g("nonphoto";"srocc"),
             g("imazen26";"srocc"), g("csiq";"srocc"), g("live";"srocc"),
             g("aic3";"srocc"), g("aic4";"srocc"), g("sdr25";"srocc"),
             g("hfnlproxy";"per_ref_mean"), g("kadid";"srocc_signed"), g("tid";"srocc"),
             (.dial.mono_pct // "" | tostring), (.dial.dynamic_range // "" | tostring),
             $sz, $sha] | @tsv' "$vj"
      done < <(all_stems)
    } > "$tsv"
    echo "wrote $tsv"
}

case "${1:-all}" in
    grams) cmd_grams ;;
    fit) cmd_fit ;;
    eval) cmd_eval ;;
    tsv) cmd_tsv ;;
    dump) cmd_dump ;;
    all) cmd_grams; cmd_fit; cmd_eval; cmd_tsv ;;
    *) echo "usage: $0 [grams|fit|eval|tsv|dump|all]" >&2; exit 2 ;;
esac
