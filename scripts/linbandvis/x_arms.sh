#!/usr/bin/env bash
#
# x_arms.sh — APPENDIX X Thread 2: the BANDVIS-ON paired arms.
#
#   ongrams   ON-definition mm01 grams (from the X-G1-gated dstact root)
#   addfit    additive ON cells: cb944 x {Ms,M4} x 5 lams (OFF twins live in
#             x_pools.sh; identical recipe, only the gram tables differ)
#   addeval   evaluate ON additive cells with --features-root <ON root>
#   onroot    assemble the ON features-root (parquets + canonical symlinks
#             for the bigcodec-derived slices, mismatch caveat registered)
#   mlpargv   derive + commit the paired-arm trainer argv (L9 echo, 6 local
#             legs kept, bigcodec/kadis/teacher legs dropped IDENTICALLY in
#             both arms; per-arm root remap; seeds {6101, 6103})
#   mlpfit    train the 4 cells (OFF/ON x 2 seeds) — run-heavy, <=2 concurrent
#   mlpeval   verdict + fulleval per cell against the matching-definition root
#
# Registered: appendix X (X.4/X.5). ON rows are a DIFFERENT GAIN definition —
# never column-mixed into canonical tables.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TGT=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
BDR=${ZL_BIN:-$TGT/release/bake_dial_refit}
BV=${ZL_BV:-$TGT/release/bake_verdict}
TRAIN=${ZL_TRAIN:-$TGT/release/zensim_mlp_train}

E944=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
ONROOT=${XDST_ROOT:-/mnt/v/zen/zensim-training/ext944-dstact-2026-08-06}
OUT=/mnt/v/output/zensim/bakes/linbandvis
G=$OUT/grams
B=$OUT/bakes
VD=$OUT/verdicts
SL=$OUT/slices
LOG=${ZL_LOG:-$HOME/tmp/linbandvis}
mkdir -p "$G" "$B" "$VD" "$SL" "$LOG" "$OUT/fulleval"
NI=(nice -n 19 ionice -c 3)

LAMS=(3e-4 1e-3 2e-3 5e-3 1e-2)
SWEEPS=${ZL_SWEEPS:-200000}
SEEDS=(6101 6103)

# ---------------------------------------------------------------- onroot ----
cmd_onroot() {
    # The extraction promote (x_promote_dstact.py) already wrote the 13 leg
    # parquets + manifest into $ONROOT. Symlink the bigcodec-derived eval
    # slices + anchor from canonical: those CANNOT be re-extracted at
    # experiment cost, carry the registered 4-column mismatch caveat for ON
    # cells, and are secondary axes for the toggle read (X.4).
    for f in ext_nonphoto.parquet ext_imazen26.parquet ext_hfnlproxy.parquet \
             anchor944_dial.parquet; do
        [[ -e $ONROOT/$f ]] || ln -s "$E944/$f" "$ONROOT/$f"
    done
    echo "ON root ready: $ONROOT"
    ls -la "$ONROOT" | sed -n '1,25p'
}

# --------------------------------------------------------------- ongrams ----
build_on_gram() {  # $1 name  $2 parquet
    local name=$1 pq=$2 out="$G/xon_${name}_mm.npz"
    [[ -f $out ]] && { echo "[ongram] $name cached"; return; }
    echo "== ongram $name =="
    "${NI[@]}" "$BDR" gram --parquet "$pq" --target human_score --target-minmax01 \
        --space raw --expect-n-feat 944 --out "$out" 2>&1 | tee "$LOG/ongram_${name}.log"
}
cmd_ongrams() {
    build_on_gram safesyn   "$ONROOT/ext_safesyn_full.parquet"
    build_on_gram cid22t201 "$ONROOT/ext_cid22_train201.parquet"
    build_on_gram kadid     "$ONROOT/ext_kadid.parquet"
    build_on_gram tid       "$ONROOT/ext_tid.parquet"
}

# ---------------------------------------------------------------- addfit ----
on_mix_grams() {  # $1 mix
    case $1 in
        Ms) echo "--gram $G/xon_safesyn_mm.npz --weight 1.0" ;;
        M4) echo "--gram $G/xon_safesyn_mm.npz --weight 1.0 --gram $G/xon_cid22t201_mm.npz --weight 1.0 --gram $G/xon_kadid_mm.npz --weight 0.5 --gram $G/xon_tid_mm.npz --weight 0.5" ;;
    esac
}
cmd_addfit() {
    [[ -f $SL/cb944.idx ]] || { seq 0 719; printf '%s\n' 924 925 929 930 934 935 939 940; } > "$SL/cb944.idx"
    for m in Ms M4; do for l in "${LAMS[@]}"; do
        local_stem="XON_cb944_${m}_mm_lam${l}"
        bin="$B/$local_stem.bin"; npz="$B/$local_stem.fit.npz"
        [[ -f $bin ]] && { echo "[fit] $local_stem cached"; continue; }
        echo "== fit $local_stem =="
        # shellcheck disable=SC2046
        "${NI[@]}" "$BDR" fit-lasso --space raw --target human_score__mm01 \
            --lam "$l" --tau 0 --n-sweeps "$SWEEPS" --tol 1e-10 \
            --slice-file "$SL/cb944.idx" --emit-fit-npz "$npz" --out "$bin" \
            $(on_mix_grams "$m") \
            --anchor-parquet "$E944/anchor944_dial.parquet" --anchor-target target_score \
            2>&1 | tee "$LOG/$local_stem.fit.log"
    done; done
}

cmd_addeval() {
    local jobs=${ZL_EVAL_JOBS:-4} n=0
    for m in Ms M4; do for l in "${LAMS[@]}"; do
        stem="XON_cb944_${m}_mm_lam${l}"
        vj="$VD/$stem.full.json"
        [[ -f $vj ]] && { echo "[eval] $stem cached"; continue; }
        echo "== eval $stem (ON root) =="
        "${NI[@]}" "$BV" --bake "$B/$stem.bin" --regime 944 \
            --features-root "$ONROOT" --full-json "$vj" \
            > "$VD/$stem.verdict.md" 2> "$LOG/$stem.eval.log" &
        n=$((n+1)); [[ $((n % jobs)) -eq 0 ]] && wait
    done; done
    wait
}

# ---------------------------------------------------------------- mlpargv ---
cmd_mlpargv() {
    # Base argv = the wave-10 L9 echo (token-authoritative), filtered:
    #   drop --group legs bigcodec / kadis / tsafesyn / ttbig (not
    #   ON-extractable at experiment cost; dropped IDENTICALLY in both arms)
    #   keep every other token verbatim; per-arm root remap; seed/out set here.
    WAVE10_ECHO=1 bash "$REPO_ROOT/scripts/wave10_seed.sh" L9 424242 > "$LOG/l9_echo_raw.txt"
    python3 - "$LOG/l9_echo_raw.txt" "$E944" "$ONROOT" "$B" <<'PY'
import sys, shlex
raw, e944, onroot, bdir = sys.argv[1:5]
toks = open(raw).read().split()
DROP = {"bigcodec", "kadis", "tsafesyn", "ttbig"}
out, i = [], 0
while i < len(toks):
    t = toks[i]
    if t == "--group":
        leg = toks[i + 1].split(":", 1)[0]
        if leg in DROP:
            i += 2
            continue
        out += [t, toks[i + 1]]
        i += 2
        continue
    if t in ("--seed", "--out"):
        i += 2
        continue
    out.append(t)
    i += 1
base = " ".join(out)
for arm, root in (("off", e944), ("on", onroot)):
    argv = base.replace(e944, root)
    for seed in (6101, 6103):
        stem = f"XMLP_{arm}_s{seed}"
        line = f"{argv} --seed {seed} --out {bdir}/{stem}.bin"
        open(f"{sys.argv[4]}/../{stem}.argv", "w").write(line + "\n")
        print(stem, "argv tokens:", len(line.split()))
PY
    # The .argv artifacts land next to bakes/ (in $OUT); diff-verify: the two
    # arms must differ ONLY in root paths + --out; seeds only across seeds.
    for s in "${SEEDS[@]}"; do
        if diff <(tr ' ' '\n' < "$OUT/XMLP_off_s${s}.argv" | sed "s|$E944|ROOT|g") \
                <(tr ' ' '\n' < "$OUT/XMLP_on_s${s}.argv" | sed "s|$ONROOT|ROOT|g" ) \
                > "$LOG/argv_diff_s${s}.txt"; then
            echo "ARGV-PAIR s${s}: arms differ only in root remap + --out? checking --out..."
        fi
        grep -c . "$LOG/argv_diff_s${s}.txt" || true
    done
    echo "argv artifacts written to $OUT/XMLP_*.argv"
}

# ---------------------------------------------------------------- mlpfit ----
launch_one() {  # $1 stem — foreground-launchable unit
    local stem=$1
    [[ -f $B/$stem.bin ]] && { echo "[train] $stem cached"; return; }
    echo "== train $stem (box census: $(pgrep -xc zensim_mlp_trai || true)) =="
    # shellcheck disable=SC2046
    "${NI[@]}" "$TRAIN" $(sed "s|--out .*||" "$OUT/$stem.argv") \
        --out "$B/$stem.bin" > "$LOG/$stem.train.log" 2>&1
}
cmd_mlpfit() {
    # <=2 concurrent local trainers (box-wide combined cap honored by the
    # census printed at each launch): the two ARMS of one seed run as a pair,
    # seeds sequential. Paired launch also equalizes box conditions per seed.
    for s in "${SEEDS[@]}"; do
        census=$(pgrep -xc zensim_mlp_trai || true)
        if [[ ${census:-0} -gt 2 ]]; then
            echo "ABORT: box trainer census $census > 2 before seed $s (combined cap 4)" >&2
            exit 3
        fi
        launch_one "XMLP_off_s${s}" &
        p1=$!
        launch_one "XMLP_on_s${s}" &
        p2=$!
        wait "$p1" "$p2"
    done
    ls -la "$B"/XMLP_*.bin
}

# ---------------------------------------------------------------- mlpeval ---
cmd_mlpeval() {
    for arm in off on; do
        root=$E944; [[ $arm == on ]] && root=$ONROOT
        for s in "${SEEDS[@]}"; do
            stem="XMLP_${arm}_s${s}"
            vj="$VD/$stem.full.json"
            [[ -f $vj ]] && { echo "[eval] $stem cached"; continue; }
            echo "== eval $stem (root $root) =="
            "${NI[@]}" "$BV" --bake "$B/$stem.bin" --regime 944 \
                --features-root "$root" --full-json "$vj" \
                > "$VD/$stem.verdict.md" 2> "$LOG/$stem.eval.log" || {
                    echo "FAIL eval $stem" >&2; return 1; }
        done
    done
}

case "${1:-help}" in
    onroot) cmd_onroot ;;
    ongrams) cmd_ongrams ;;
    addfit) cmd_addfit ;;
    addeval) cmd_addeval ;;
    mlpargv) cmd_mlpargv ;;
    mlpfit) cmd_mlpfit ;;
    mlpeval) cmd_mlpeval ;;
    *) echo "usage: $0 [onroot|ongrams|addfit|addeval|mlpargv|mlpfit|mlpeval]" >&2; exit 2 ;;
esac
