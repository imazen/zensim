#!/bin/bash
# wave-r4 A6: everything after the 85,018-row decode, in one command.
#
#   1. verify the decode produced every PNG the pairs TSV names
#   2. extract 196,086 rows at foldapp2pools   (extract_safesyn_big_r4.sh)
#   3. promote through the OWNER                (promote_ext944_canonical.py)
#   4. derive the foldapp2 view                 (derive_foldapp2_views.py)
#   5. run the JPEG-subset agreement gate       (gate_safesyn_big_r4.py)
#
# Nothing here computes a statistic or writes a parquet itself; every step
# shells the owning tool. Env: BIGLEG_NAME (default ext_safesyn_big),
# BIGLEG_PAIRS (default the primary pairs TSV).
set -u
ROOT=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01
RUN=/mnt/v/output/zensim/waver4-run-2026-09-01
REPO=/home/lilith/work/zen/zensim
NAME="${BIGLEG_NAME:-ext_safesyn_big}"
PAIRS="${BIGLEG_PAIRS:-$ROOT/pairs/pairs_safesyn_big.tsv}"
ts() { date -u +%H:%M:%SZ; }

echo "== finalize $NAME start $(ts)"
echo "== pairs=$PAIRS"

# ---- 1. every distorted side must exist ------------------------------------
echo "== step 1: distorted-side existence $(ts)"
python3 - "$PAIRS" <<'PY' || exit 1
import csv, os, sys, collections
p = sys.argv[1]
miss = collections.Counter(); n = 0
for r in csv.DictReader(open(p, newline=''), delimiter='\t'):
    n += 1
    d = r['dist_path']
    if not os.path.isfile(d) or os.path.getsize(d) == 0:
        miss[d.split('/')[-2] if d.endswith('.jpg') else 'DECODED'] += 1
print(f"   rows={n} missing={sum(miss.values())} {dict(miss)}")
sys.exit(1 if miss else 0)
PY
[ $? -ne 0 ] && { echo "ABORT: distorted sides missing"; exit 1; }

# ---- 2. extract ------------------------------------------------------------
echo "== step 2: extract $(ts)"
ZM944_BIN=/mnt/v/zen/cargo-targets/waver4/release/examples/v2_ab_extract \
ZM944_OUT="$RUN" ZM944_MODE=foldapp2pools ZM944_PAIRS="$PAIRS" ZM944_NAME="$NAME" \
  "$REPO/benchmarks/wave_r4_2026-09-01/extract_safesyn_big_r4.sh" || exit 1

# ---- 3. promote via the owner ---------------------------------------------
echo "== step 3: promote $(ts)"
cp -a "$ROOT/_MANIFEST.json" "$RUN/_MANIFEST.before_$NAME.json" 2>/dev/null || true
EXT944_RUN="$RUN" EXT944_DEST="$ROOT" EXT944_MODE=folded720append2pools \
ZENSIM_COMMIT=75c09149e6c32fc84a07aa9bc144daa92fb3ac11 \
EXT944_EXTRA_LEGS="$NAME" EXT944_LEGS="$NAME" \
  python3 "$REPO/scripts/canonical_corpus/promote_ext944_canonical.py" || exit 1

# ---- 4. derive the foldapp2 (pool-zeroed) view -----------------------------
echo "== step 4: derive foldapp2 view $(ts)"
WR4_ROOT="$ROOT" WR4_ZERO_TABLES="$NAME.parquet" \
  python3 "$REPO/benchmarks/wave_r4_2026-09-01/derive_foldapp2_views.py" || exit 1

# ---- 5. the agreement gate -------------------------------------------------
echo "== step 5: JPEG-subset agreement gate $(ts)"
BIGLEG_CSV="$RUN/$NAME.csv" BIGLEG_PAIRS="$PAIRS" \
  python3 "$REPO/benchmarks/wave_r4_2026-09-01/gate_safesyn_big_r4.py"
echo "== gate rc=$? $(ts)"
echo "FINALIZE-DONE $NAME $(ts)"
