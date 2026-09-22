#!/usr/bin/env bash
# THE single CID22-B read for the verdict lane.
# Preconditions (checked, then recorded): X2 results frozen, X1 winner
# artefact frozen, all X1 eval-leg score CSVs emitted. This script:
#   1) unseals B labels (once — unseal_cid22b_verdict.py refuses twice)
#   2) scores the frozen DVIFM winner on the cid22b caches
#   3) emits peer score CSVs for cid22b (B, D, R915×2, fast-ssim2)
# After this runs, cid22b metrics are computed once by build_verdict.py.
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-verdict-2026-09-20
WIN=${1:?usage: cid22b_read.sh <winner-form:gate|prior|curve>}
WINJSON=$OUT/x1/frozen_${WIN}.json
fails=0
# --- preconditions: everything frozen ---
for f in "$OUT/x2/x2_results.json" "$OUT/x2_decision.json" "$WINJSON"; do
  [ -s "$f" ] || { echo "PRECOND-FAIL missing $f"; exit 1; }
done
for leg in safesyn_dev cid22_dev codec_dev human_dev kadid135 konfig_val; do
  [ -s "$OUT/scores/dvifm_${WIN}__on__${leg}.csv" ] \
    || { echo "PRECOND-FAIL missing dvifm_${WIN} on $leg"; exit 1; }
done
# --- THE unseal (single-use) ---
python3 "$OUT/tools/unseal_cid22b_verdict.py" || exit 1
# --- rebuild the cid22b y meta from unsealed labels ---
python3 - << 'PYEOF'
import csv, json
from pathlib import Path
OUT = Path('/mnt/v/output/zensim/dvifm-verdict-2026-09-20')
rows = list(csv.DictReader(open(OUT/'pairs/cid22b_unsealed.tsv'),
                           delimiter='\t'))
y = [float(r['human_score'])*100.0 for r in rows]
ref = [r['ref_path'] for r in rows]
(OUT/'eval_meta/cid22b_y.json').write_text(
    json.dumps({'y': y, 'ref': ref}))
print('cid22b y meta rebuilt from unsealed labels:', len(y), 'rows')
PYEOF
# --- score the frozen winner on cid22b ---
python3 "$OUT/tools/score_dvifm.py" "$WINJSON" "dvifm_${WIN}" \
  "$OUT/score_tasks_cid22b.json" "$OUT/scores" || fails=$((fails+1))
# --- peers on cid22b (score_peers handles the unsealed tsv now) ---
bash "$OUT/tools/score_peers.sh" || fails=$((fails+1))
echo "CID22B READ DONE fails=$fails"
