#!/bin/bash
# Re-extract every training/assessment set of the v2e29 root under ZENSIM_RING_FORM=hpm, verify each against the
# stored Rev5 bank (all non-ringing slots must be bit-identical), then the E15 coverage pool, then the derived root.
# Exploration record: benchmarks/byconstruction_2026-10-22.md. Usage: ring_bank.sh EXTRACTOR BUILD_COMMIT OUT_DIR THREADS
set -euo pipefail
BIN=$1; COMMIT=$2; OUT=$3; THREADS=${4:-24}
HERE=$(cd "$(dirname "$0")" && pwd)
FEATPOT=$HERE/../rev4_featpot
mkdir -p "$OUT"
for s in konfig_train konfig_val tid2013 kadid_select kadid_train cid22_a25 cid22_train safesyn; do
  if [ -f "$OUT/bank/$s/_MANIFEST.json" ]; then echo "skip $s"; continue; fi
  echo "== $s start $(date -u +%T)"
  ZENSIM_RING_FORM=hpm nice -n 15 python3 "$FEATPOT/rev5_bank.py" extract "$s" --bin "$BIN" --build-commit "$COMMIT" \
    --revision 5 --era rev5_localwin_ringhpm --out "$OUT/bank" --threads "$THREADS" > "$OUT/bank_$s.log" 2>&1
  python3 "$FEATPOT/rev5_bank.py" verify-against "$s" --out "$OUT/bank" --reference /var/tmp/rev5-featbank \
    > "$OUT/verify_$s.json" || true
  echo "== $s done $(date -u +%T) $(cat "$OUT/verify_$s.json")"
done
if [ ! -f "$OUT/pool/_MANIFEST.json" ]; then
  echo "== pool start $(date -u +%T)"
  nice -n 15 python3 "$HERE/ring_root.py" pool --bin "$BIN" --out "$OUT/pool" --threads "$THREADS"
fi
echo "== root start $(date -u +%T)"
python3 "$HERE/ring_root.py" build --bank "$OUT/bank" --pool "$OUT/pool" --out "$OUT/root"
echo ALL-DONE
