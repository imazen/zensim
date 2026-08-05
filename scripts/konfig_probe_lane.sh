#!/usr/bin/env bash
# konfig_probe_lane.sh — run the 4 Appendix-L KonFiG probe cells, 2 at a time,
# harvest each as it lands, and fail loud (campaign APPENDIX L §L.7/§L.10,
# pre-reg e93eba04).
#
# Lane discipline (registered): my lane is <=2 trainers concurrent; before each
# launch pair the box trainer census (pgrep -xc zensim_mlp_trai) must show <=3
# (the K.8 combined cap 5 minus my 2). Cells:
#   KFG25_s4101 KFG25_s4103   (train_w 0.25)
#   KFG75_s4101 KFG75_s4103   (train_w 0.75)
# Every cell = scripts/konfig_probe_seed.sh (echo-verified vs wave-10 L9).
# Harvest = scripts/harvest_bakes.sh into the STANDARD shared dirs.
#
# Heartbeat + terminal sentinel: $HB.log / $HB.status / $HB.done — the .done
# file appears exactly once on EVERY exit path.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HB=${KONFIG_LANE_HB:-$HOME/tmp/konfig/probe_lane}
RH=${RUN_HEAVY:-$HOME/work/zen/scripts/run-heavy}
mkdir -p "$(dirname "$HB")"
LOG="$HB.log"; DONE="$HB.done"; rm -f "$DONE"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { printf '%s %s\n' "$(now)" "$*" | tee -a "$LOG" >&2; }
STATE=RUNNING; RC=5; NOK=0; NFAIL=0
finish() {
    printf '%s %s rc=%s cells_ok=%s cells_failed=%s\n' \
        "$(now)" "$STATE" "$RC" "$NOK" "$NFAIL" > "$DONE"
    say "TERMINAL $STATE rc=$RC ok=$NOK failed=$NFAIL"
    exit "$RC"
}
trap 'STATE=SIGNAL; RC=5; finish' TERM INT HUP
trap 'finish' EXIT

census_gate() {
    # Wait until adding 2 trainers keeps the box at <=5 total (K.8 cap).
    local n t0=$SECONDS
    while :; do
        n=$(pgrep -xc zensim_mlp_trai || true); n=${n:-0}
        [ "$n" -le 3 ] && { say "census OK: $n trainers live"; return 0; }
        [ $((SECONDS - t0)) -gt 7200 ] && { say "census gate timeout (n=$n)"; return 1; }
        printf '%s census-wait n=%s\n' "$(now)" "$n" > "$HB.status"
        sleep 60
    done
}

run_pair() {
    local dose=$1 s1=$2 s2=$3 rc1 rc2
    census_gate || return 1
    say "launch $dose s$s1 + s$s2"
    "$RH" --mem 14G --jobs 8 -- "$REPO_ROOT/scripts/konfig_probe_seed.sh" "$dose" "$s1" \
        >> "$HB.$dose.s$s1.log" 2>&1 & local p1=$!
    "$RH" --mem 14G --jobs 8 -- "$REPO_ROOT/scripts/konfig_probe_seed.sh" "$dose" "$s2" \
        >> "$HB.$dose.s$s2.log" 2>&1 & local p2=$!
    wait "$p1"; rc1=$?
    wait "$p2"; rc2=$?
    say "$dose s$s1 rc=$rc1; s$s2 rc=$rc2"
    [ "$rc1" = 0 ] && [ "$rc2" = 0 ]
}

harvest_one() {
    local tag=$1 seed=$2
    local bake=/mnt/v/output/zensim/bakes/sota944/bakes/${tag}_s${seed}.bin
    if [ ! -s "$bake" ]; then say "MISSING BAKE ${tag}_s${seed}"; return 1; fi
    "$REPO_ROOT/scripts/harvest_bakes.sh" --bake "$bake" --stem "${tag}_s${seed}" \
        --regime 944 --heartbeat "$HB.harvest_${tag}_s${seed}" >> "$LOG" 2>&1
}

overall=0
for dose in w25 w75; do
    tag=$([ "$dose" = w25 ] && echo KFG25 || echo KFG75)
    if run_pair "$dose" 4101 4103; then
        for s in 4101 4103; do
            if harvest_one "$tag" "$s"; then NOK=$((NOK+1)); say "harvest OK ${tag}_s${s}"
            else NFAIL=$((NFAIL+1)); overall=1; say "HARVEST FAILED ${tag}_s${s}"; fi
        done
    else
        NFAIL=$((NFAIL+2)); overall=1; say "TRAIN PAIR FAILED $dose"
    fi
done
STATE=$([ "$overall" = 0 ] && echo COMPLETE || echo FAILED)
RC=$overall
exit "$RC"
