#!/usr/bin/env bash
# Run every bank set through the shared lock, one heavy call per set, appending each result
# to the worklog. Usage: nohup bash scripts/restore_cuts/run_all.sh > /var/tmp/restore-cuts/logs/run_all.log 2>&1 &
set -euo pipefail
REPO=/home/lilith/work/zen/zensim--restore-cuts
ROOT=/var/tmp/restore-cuts
log=$REPO/benchmarks/restore-cuts_WORKLOG.md
sets=(konjnd_jpeg_terminal aic4 konfig_train konfig_val aic3 konjnd_jpeg_select csiq cid22_b cid22_a25
      tid2013 kadid_terminal kadid_select kadid_train konjnd_bpg_val konjnd_bpg_train mcljci cid22_train safesyn)
(while :; do printf '\nHEARTBEAT %s restore-cuts extraction\n' "$(date -u +%FT%TZ)" >> "$log"; sleep 900; done) &
hb=$!
trap 'kill "$hb" 2>/dev/null || true' EXIT
for s in "${sets[@]}"; do
    avail=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    (( avail >= 20 )) || { echo "disk floor reached: ${avail}G"; exit 87; }
    out=$ROOT/logs/$s.log
    start=$(date -u +%FT%TZ)
    set +e
    /home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- bash "$REPO/scripts/restore_cuts/run_set.sh" "$s" > "$out" 2>&1
    rc=$?
    set -e
    {
        printf '\n### %s\n\nUTC %s-%s; cwd `%s`; command `heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh %s`; exit %s.\n\n' \
            "$s" "$start" "$(date -u +%FT%TZ)" "$REPO" "$s" "$rc"
        printf 'Exact result lines from `%s`:\n\n```text\n' "$out"
        grep -E '^(Loaded|scored |Wrote |WALL_SECONDS=|RESTORE_|[0-9a-f]{64}  )' "$out" || true
        printf '```\nLog sha256: `%s`.\n' "$(sha256sum "$out" | cut -d' ' -f1)"
    } >> "$log"
    (( rc == 0 )) || { echo "RESTORE_FAIL set=$s rc=$rc"; exit "$rc"; }
    echo "RESTORE_COMPLETE set=$s"
done
