#!/usr/bin/env bash
# zgeom lane: sequential paired-seed fit batch (one run at a time).
#   drive_batch.sh <runs.json> <logdir> [jobs]
# Each run is materialized as a per-run script file — quoting inside a
# file is data, not syntax (the transplant-lane here-string bug).
set -uo pipefail
RUNS=${1:?runs.json}
LOGDIR=${2:?logdir}
JOBS=${3:-1}
SCRIPTS=$LOGDIR/scripts
mkdir -p "$LOGDIR" "$SCRIPTS"
command -v parallel >/dev/null || JOBS=1

# materialize run scripts once
python3 - "$RUNS" "$SCRIPTS" <<'EOF'
import json, os, sys
runs = json.load(open(sys.argv[1]))['runs']
os.makedirs(sys.argv[2], exist_ok=True)
for r in runs:
    p = os.path.join(sys.argv[2], r['id'] + '.sh')
    if not os.path.exists(p):
        open(p, 'w').write('#!/usr/bin/env bash\nset -uo pipefail\n' + r['cmd'] + '\n')
        os.chmod(p, 0o755)
print(len(runs), 'scripts')
EOF

ls "$SCRIPTS"/*.sh | while read -r s; do
  id=$(basename "$s" .sh)
  echo "$s|$LOGDIR/$id.log"
done | python3 -c "
import sys, subprocess, concurrent.futures as cf
todo = []
for line in sys.stdin:
    s, log = line.strip().split('|')
    # resume-safe: a run counts done iff its log has the trainer's
    # terminal best-validation line.
    if __import__('os').path.exists(log):
        with open(log) as f:
            if any('best validation mean SROCC' in ln or 'best-val' in ln
                   or 'best_val' in ln for ln in f):
                print('SKIP(done)', s); continue
    todo.append((s, log))
print(len(todo), 'to run')
def one(t):
    s, log = t
    with open(log, 'w') as lf:
        return subprocess.run(['bash', s], stdout=lf, stderr=subprocess.STDOUT).returncode
with cf.ThreadPoolExecutor(max_workers=$JOBS) as ex:
    for rc in ex.map(one, todo):
        pass
"
echo "batch done: $(ls $LOGDIR/*.log 2>/dev/null | wc -l) logs"
