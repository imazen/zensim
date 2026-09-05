#!/usr/bin/env bash
# Serial replication-wave runner. One fit at a time (machine safety).
set -uo pipefail
ROOT=/mnt/v/output/zensim/replication-2026-09-05
PROG=$ROOT/logs/PROGRESS.txt
: > "$PROG"
N=$(python3 -c "import json;print(len(json.load(open('/home/lilith/tmp/replicate/fits.json'))))")
for i in $(seq 0 $((N-1))); do
  TAG=$(python3 -c "import json;print(json.load(open('/home/lilith/tmp/replicate/fits.json'))[$i]['tag'])")
  CK=$(python3 -c "import json;print(json.load(open('/home/lilith/tmp/replicate/fits.json'))[$i]['ckpt'])")
  OUT=$(python3 -c "import json;print(json.load(open('/home/lilith/tmp/replicate/fits.json'))[$i]['out'])")
  if [ -f "$ROOT/logs/$TAG.done" ]; then echo "[$(date -u +%H:%M:%S)] SKIP $TAG (done)" | tee -a "$PROG"; continue; fi
  mkdir -p "$CK"
  python3 -c "
import json,shlex
f=json.load(open('/home/lilith/tmp/replicate/fits.json'))[$i]
open('/home/lilith/tmp/replicate/cur_argv.txt','w').write(' '.join(shlex.quote(a) for a in f['argv']))
"
  echo "[$(date -u +%H:%M:%S)] START $((i+1))/$N $TAG" | tee -a "$PROG"
  S=$(date +%s)
  ~/work/zen/scripts/run-heavy --jobs 12 -- bash -c "$(cat /home/lilith/tmp/replicate/cur_argv.txt)" > "$ROOT/logs/$TAG.log" 2>&1
  RC=$?
  E=$(date +%s)
  echo "rc=$RC elapsed_s=$((E-S)) out=$OUT bytes=$(stat -c%s "$OUT" 2>/dev/null || echo 0)" > "$ROOT/logs/$TAG.done"
  echo "[$(date -u +%H:%M:%S)] END   $((i+1))/$N $TAG rc=$RC elapsed_s=$((E-S))" | tee -a "$PROG"
done
echo "[$(date -u +%H:%M:%S)] WAVE COMPLETE" | tee -a "$PROG"
touch "$ROOT/logs/WAVE_COMPLETE"
