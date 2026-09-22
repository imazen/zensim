#!/usr/bin/env bash
# transplant lane: execute a slice of the runs.json grid. Supervisor-
# authorized out-of-lock at --jobs 4 --mem 8G (flock could not be acquired
# within the ~10min budget behind the loss lane). Idempotent: a run whose
# log exists is skipped.
#   drive_batch.sh <start_idx> <count>
#
# Transport is per-run script files (fits/runcmds/<id>.sh): the run commands
# embed quoted args (--group "name:path", --historical-replay "...(parens)"),
# which a here-string/eval chain mangles. Writing each command to a file and
# executing it with bash makes the quotes data, not shell syntax.
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-transplant-2026-09-20
export ZENSIM_FORMULA_REV=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       RAYON_NUM_THREADS=1
mkdir -p "$OUT/fits/logs" "$OUT/fits/runcmds"

# Emit one runnable script per run in the slice (idempotent overwrite).
python3 - "$OUT/fits/runs.json" "$1" "$2" "$OUT/fits/runcmds" <<'PY'
import json, sys, os
runs = json.load(open(sys.argv[1]))['runs']
d = sys.argv[4]
for r in runs[int(sys.argv[2]):int(sys.argv[2]) + int(sys.argv[3])]:
    p = os.path.join(d, r['id'] + '.sh')
    open(p, 'w').write('#!/bin/bash\n' + r['cmd'] + '\n')
    os.chmod(p, 0o755)
PY

# Run each slice script whose train log is absent, 4 at a time.
find "$OUT/fits/runcmds" -maxdepth 1 -name '*.sh' -print0 | sort -z |
xargs -0 -P4 -I{} bash -c '
  f="{}"; id=$(basename "$f" .sh)
  log="'"$OUT"'/fits/logs/$id.train.log"
  [ -s "$log" ] && { echo "SKIP $id"; exit 0; }
  echo "RUN $id $(date +%H:%M:%S)"
  bash "$f" > "$log" 2>&1
  echo "END $id rc=$?"'
echo "BATCH DONE"
