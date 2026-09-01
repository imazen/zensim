#!/bin/bash
# wave-r4: the KADIS leg — the ONE leg of the flagship recipe with no local
# distorted pixels.
#
# The recipe's kadis group is
#   --group kadis:/mnt/v/zen/zensim-training/kadis-944-2026-08-01/kadis_944_ssim2_50k.parquet:0.15:1.0:both
# (50,000 rows, target `human_score` = score_ssim2_gpu/100).  That table carries
# only  source_id / source_filename / score_ssim2_gpu / human_score /
# ref_basename  + f0..f943 — it does NOT carry a pointer to the distorted image.
#
# RESCORE-FROM-LINKS (the registered method — kadis-distort regeneration is
# REJECTED, its RNG diverges, mean|Δ| 9.8):
#   the sibling `kadis700k_944.parquet` in the same dir carries `distorted_url`
#   for all 699,999 cells.  MEASURED 2026-09-01: the key (source_id,
#   score_ssim2_gpu) resolves 50000/50000 of the recipe rows with 0 misses.
#   90 rows land on a key with >1 candidate; every such candidate set is a
#   dist_type=7 JPEG ladder whose q1..q5 collapsed to the SAME object (identical
#   ETag on the store, and bit-identical f0..f943 in the era-1 table), so a
#   deterministic pick is exact.  `resolve` re-asserts that rather than assuming
#   it.
#
# STORE: the distorted PNGs are on **R2 only**.  MEASURED 2026-09-01: the LAN
# store (tower MinIO) holds s3://zentrain/kadis-700k-gpu/canonical/ but has NO
# .../distorted/ prefix (HeadObject -> 404).  So this script forces ZEN_STORE=r2
# and verifies the prefix before fetching anything.
#
# Stages (run one, or `all`):
#   resolve  -> $WORK/kadis50k_urlmap.tsv        (50,000 rows; gated)
#   fetch    -> $STAGE/png/<object>.png          (49,988 distinct objects; gated)
#   pairs    -> $WORK/pairs_kadis_png.tsv        (50,000 rows, parquet row order)
#   extract  -> $OUT/ext_kadis.csv               (50,000 x 946; gated)
#
# Env:
#   ZM944_BIN   extractor (default the wave-r4 build)
#   ZM944_MODE  default foldapp2pools  (regime folded720append2pools, 944 wide)
#   WORK/STAGE/OUT  see below
#   S5_WORKERS  default 192 (measured 179.6 obj/s / 52.0 MiB/s at 192)
#   NROWS       smoke-test override (default 50000 — the real leg)
set -uo pipefail

REC=/mnt/v/zen/zensim-training/kadis-944-2026-08-01/kadis_944_ssim2_50k.parquet
SRC=/mnt/v/zen/zensim-training/kadis-944-2026-08-01/kadis700k_944.parquet
REFS=/mnt/v/datasets/kadis700k/refs
WORK=${WORK:-/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/kadis}
STAGE=${STAGE:-/mnt/v/output/zensim/waver4-kadis-stage-2026-09-01}
OUT=${OUT:-/mnt/v/output/zensim/waver4-run-2026-09-01}
BIN=${ZM944_BIN:-/mnt/v/zen/cargo-targets/waver4/release/examples/v2_ab_extract}
MODE=${ZM944_MODE:-foldapp2pools}
S5W=${S5_WORKERS:-192}
S3ENV=${S3ENV:-/home/lilith/work/zen/zenmetrics/scripts/lib/s3env.sh}
NROWS=${NROWS:-50000}      # override ONLY for a smoke run; the real leg is 50000
NCOLS=946
MIN_FREE_GB=25

URLMAP="$WORK/kadis50k_urlmap.tsv"
PNGDIR="$STAGE/png"
PAIRS="$WORK/pairs_kadis_png.tsv"
CSV="$OUT/ext_kadis.csv"

die() { echo "ABORT: $*" >&2; exit 1; }
ts()  { date -u +%H:%M:%SZ; }

# ---------------------------------------------------------------- preflight
preflight() {
  [ -f "$REC" ]   || die "recipe parquet missing: $REC"
  [ -f "$SRC" ]   || die "distorted_url parquet missing: $SRC"
  [ -d "$REFS" ]  || die "local refs mirror missing: $REFS"
  [ -x "$BIN" ]   || die "extractor missing/not executable: $BIN"
  [ -f "$S3ENV" ] || die "s3env resolver missing: $S3ENV"
  command -v s5cmd >/dev/null || die "s5cmd not on PATH"
  local nrefs; nrefs=$(find "$REFS" -maxdepth 1 -name '*.png' | wc -l)
  [ "$nrefs" -eq 140000 ] || die "refs mirror has $nrefs png, expected 140000"
  mkdir -p "$WORK" "$STAGE" "$OUT" || die "cannot create output dirs"
  echo "== preflight OK  refs=$nrefs  bin=$BIN  mode=$MODE"
}

# ---------------------------------------------------------------- resolve
do_resolve() {
  echo "== resolve start $(ts)"
  URLMAP="$URLMAP" REC="$REC" SRC="$SRC" REFS="$REFS" NROWS="$NROWS" python3 - <<'PY' || die "resolve failed"
import collections, hashlib, os, sys
import numpy as np, pyarrow.parquet as pq

REC, SRC, REFS = os.environ['REC'], os.environ['SRC'], os.environ['REFS']
URLMAP, NROWS = os.environ['URLMAP'], int(os.environ['NROWS'])
PROBE = ['f%d' % i for i in range(0, 944, 16)]   # 59-col spread, full width

a = pq.read_table(REC, columns=['source_id', 'source_filename', 'score_ssim2_gpu',
                                'human_score'] + PROBE)
if a.num_rows != NROWS:
    sys.exit('ABORT: recipe parquet has %d rows, expected %d' % (a.num_rows, NROWS))
b = pq.read_table(SRC, columns=['source_id', 'score_ssim2_gpu', 'distorted_url',
                                'severity_level'] + PROBE)

sa = np.asarray(a['source_id']); qa = np.asarray(a['score_ssim2_gpu'])
hs = np.asarray(a['human_score']); fn = a['source_filename'].to_pylist()
fa = np.stack([np.asarray(a[c], dtype=np.float32).view(np.uint32) for c in PROBE], 1)
sb = np.asarray(b['source_id']); qb = np.asarray(b['score_ssim2_gpu'])
url = b['distorted_url'].to_pylist(); sev = np.asarray(b['severity_level'])
fb = np.stack([np.asarray(b[c], dtype=np.float32).view(np.uint32) for c in PROBE], 1)

idx = collections.defaultdict(list)
for i in range(len(sb)):
    idx[(sb[i], qb[i])].append(i)

out, amb, miss = [], 0, 0
for j in range(len(sa)):
    c = idx.get((sa[j], qa[j]))
    if not c:
        miss += 1
        continue
    if len(c) > 1:
        # GATE: a multi-candidate key is only safe when every candidate carries
        # the SAME features as the recipe row (the collapsed-JPEG-ladder case).
        m = [i for i in c if (fb[i] == fa[j]).all()]
        if len(m) < 1:
            sys.exit('ABORT: row %d key %r has %d candidates, none feature-matching'
                     % (j, (sa[j], qa[j]), len(c)))
        if not all((fb[i] == fb[m[0]]).all() for i in m):
            sys.exit('ABORT: row %d key %r candidates are NOT feature-identical — '
                     'the pick would be arbitrary' % (j, (sa[j], qa[j])))
        amb += 1
        c = sorted(m, key=lambda i: (sev[i], url[i]))   # deterministic
    out.append((fn[j], url[c[0]], hs[j]))

if miss or len(out) != NROWS:
    sys.exit('ABORT: resolved %d/%d rows (miss %d) — refusing a partial map'
             % (len(out), NROWS, miss))

# GATE: every reference must exist locally (no download fallback).
rmiss = sorted({f for f, _, _ in out if not os.path.exists(os.path.join(REFS, f))})
if rmiss:
    sys.exit('ABORT: %d reference images missing from %s (e.g. %s)'
             % (len(rmiss), REFS, rmiss[:3]))

tmp = URLMAP + '.tmp'
with open(tmp, 'w') as f:
    f.write('source_filename\tdistorted_url\thuman_score\n')
    for r in out:
        f.write('%s\t%s\t%.17g\n' % r)
os.replace(tmp, URLMAP)
print('resolve OK: %d rows, %d ambiguous keys resolved (all feature-identical), '
      '%d distinct objects' % (len(out), amb, len({u for _, u, _ in out})))
print('urlmap sha256 %s' % hashlib.sha256(open(URLMAP, 'rb').read()).hexdigest())
PY
  echo "== resolve done $(ts)"
}

# ---------------------------------------------------------------- fetch
do_fetch() {
  [ -f "$URLMAP" ] || die "urlmap missing — run 'resolve' first: $URLMAP"
  local want; want=$(tail -n +2 "$URLMAP" | cut -f2 | sort -u | wc -l)
  [ "$want" -gt 0 ] || die "urlmap has no objects"

  local freegb; freegb=$(df -BG --output=avail "$STAGE" | tail -1 | tr -dc '0-9')
  [ "$freegb" -ge "$MIN_FREE_GB" ] || die "only ${freegb}G free at $STAGE, need >= ${MIN_FREE_GB}G (~15.2 GB of pixels)"

  mkdir -p "$PNGDIR" || die "cannot mkdir $PNGDIR"

  # The distorted PNGs live on R2 ONLY — force it, never inherit an ambient
  # ZEN_S3_ENDPOINT pointing at the LAN store (which 404s on this prefix).
  # shellcheck disable=SC1090
  set +u; unset ZEN_S3_ENDPOINT; ZEN_STORE=r2; export ZEN_STORE; . "$S3ENV" >/dev/null 2>&1; set -u
  [ "${ZEN_S3_STORE:-}" = "r2" ] || die "s3env did not resolve to r2 (got '${ZEN_S3_STORE:-unset}')"
  [ -n "${EP:-}" ] || die "s3env did not export EP"

  # GATE: prove the prefix exists on the selected store before fetching 50k objects.
  local probe; probe=$(tail -n +2 "$URLMAP" | head -1 | cut -f2)
  local pbucket pkey
  pbucket=${probe#s3://}; pkey=${pbucket#*/}; pbucket=${pbucket%%/*}
  aws --endpoint-url "$EP" s3api head-object --bucket "$pbucket" --key "$pkey" >/dev/null 2>&1 \
    || die "selected store has no object s3://$pbucket/$pkey — the distorted prefix is NOT on this store"
  echo "== fetch: store=$ZEN_S3_STORE  objects=$want  workers=$S5W  $(ts)"

  # RESUMABLE: skip objects already staged (a 5-min fetch that dies partway
  # must not restart from zero).  The completeness gate below is unchanged —
  # it counts files in $PNGDIR against $want, so resuming cannot weaken it.
  local runfile="$STAGE/kadis_fetch.s5cmd" have="$STAGE/kadis_have.txt"
  find "$PNGDIR" -maxdepth 1 -name '*.png' -printf '%f\n' | sort > "$have"
  tail -n +2 "$URLMAP" | cut -f2 | sort -u \
    | awk -v d="$PNGDIR" -v h="$have" '
        BEGIN { while ((getline l < h) > 0) seen[l]=1 }
        { n=$0; sub(/.*\//,"",n); if (!(n in seen)) print "cp " $0 " " d "/" }' \
    > "$runfile"
  local todo; todo=$(wc -l < "$runfile")
  echo "== fetch: $todo to download, $((want-todo)) already staged"

  local t0=$SECONDS rc=0
  if [ "$todo" -gt 0 ]; then
    s5cmd --endpoint-url "$EP" --numworkers "$S5W" run "$runfile" > "$STAGE/kadis_fetch.log" 2>&1
    rc=$?
  else
    : > "$STAGE/kadis_fetch.log"
  fi
  local got; got=$(find "$PNGDIR" -maxdepth 1 -name '*.png' | wc -l)
  echo "== fetch rc=$rc got=$got/$want in $((SECONDS-t0))s $(ts)"
  [ "$rc" -eq 0 ] || die "s5cmd rc=$rc — see $STAGE/kadis_fetch.log"
  [ "$got" -eq "$want" ] || die "staged $got objects, expected $want — see $STAGE/kadis_fetch.log"
  # GATE: no zero-byte / truncated object.
  local empty; empty=$(find "$PNGDIR" -maxdepth 1 -name '*.png' -size -1k | wc -l)
  [ "$empty" -eq 0 ] || die "$empty staged objects are < 1 KiB (truncated fetch)"
}

# ---------------------------------------------------------------- pairs
do_pairs() {
  [ -f "$URLMAP" ] || die "urlmap missing — run 'resolve' first"
  [ -d "$PNGDIR" ] || die "png stage missing — run 'fetch' first"
  echo "== pairs start $(ts)"
  URLMAP="$URLMAP" PNGDIR="$PNGDIR" REFS="$REFS" PAIRS="$PAIRS" NROWS="$NROWS" python3 - <<'PY' || die "pairs failed"
import os, sys
URLMAP, PNGDIR, REFS = os.environ['URLMAP'], os.environ['PNGDIR'], os.environ['REFS']
PAIRS, NROWS = os.environ['PAIRS'], int(os.environ['NROWS'])
rows = [l.rstrip('\n').split('\t') for l in open(URLMAP)][1:]
if len(rows) != NROWS:
    sys.exit('ABORT: urlmap has %d rows, expected %d' % (len(rows), NROWS))
missing = []
tmp = PAIRS + '.tmp'
with open(tmp, 'w') as f:
    f.write('ref_path\tdist_path\thuman_score\n')
    for fn, u, h in rows:
        rp = os.path.join(REFS, fn)
        dp = os.path.join(PNGDIR, u.rsplit('/', 1)[1])
        if not os.path.exists(rp): missing.append(rp)
        elif not os.path.exists(dp): missing.append(dp)
        else: f.write('%s\t%s\t%s\n' % (rp, dp, h))
if missing:
    os.unlink(tmp)
    sys.exit('ABORT: %d pair paths missing (e.g. %s) — refusing a partial pairs TSV'
             % (len(missing), missing[:3]))
os.replace(tmp, PAIRS)
print('pairs OK: %d rows -> %s' % (len(rows), PAIRS))
PY
  local n; n=$(( $(wc -l < "$PAIRS") - 1 ))
  [ "$n" -eq "$NROWS" ] || die "pairs TSV has $n rows, expected $NROWS"
  echo "== pairs done rows=$n $(ts)"
}

# ---------------------------------------------------------------- extract
do_extract() {
  [ -f "$PAIRS" ] || die "pairs TSV missing — run 'pairs' first"
  local want; want=$(( $(wc -l < "$PAIRS") - 1 ))
  [ "$want" -eq "$NROWS" ] || die "pairs TSV has $want rows, expected $NROWS"
  echo "== extract start mode=$MODE rows=$want $(ts)"
  local t0=$SECONDS
  ZENSIM_AB_MODE="$MODE" "$BIN" "$PAIRS" "$CSV"
  local rc=$?
  local rows=-1 cols=-1
  if [ -f "$CSV" ]; then
    rows=$(( $(wc -l < "$CSV") - 1 ))
    cols=$(head -1 "$CSV" | awk -F, '{print NF}')
  fi
  echo "== extract done rc=$rc rows=$rows/$want cols=$cols $((SECONDS-t0))s $(ts)"
  [ "$rc" -eq 0 ]        || die "extractor rc=$rc"
  [ "$rows" -eq "$want" ] || die "extractor wrote $rows rows, expected $want"
  [ "$cols" -eq "$NCOLS" ] || die "extractor wrote $cols cols, expected $NCOLS"

  # GATE: the extractor emits in input-TSV order (registered fact, wave doc
  # §2.1.1).  Re-assert it row-wise on ref stem + human_score, so a reordering
  # regression cannot silently mis-key the targets.
  PAIRS="$PAIRS" CSV="$CSV" python3 - <<'PY' || die "row-order gate failed"
import csv, os, sys
p = [l.rstrip('\n').split('\t') for l in open(os.environ['PAIRS'])][1:]
r = csv.reader(open(os.environ['CSV'])); hdr = next(r)
if hdr[0] != 'ref_basename' or hdr[1] != 'human_score':
    sys.exit('ABORT: unexpected CSV header %r' % hdr[:2])
bad = 0
for i, row in enumerate(r):
    want_stem = os.path.basename(p[i][0]).rsplit('.', 1)[0]
    if row[0].rsplit('.', 1)[0] != want_stem: bad += 1
    elif abs(float(row[1]) - float(p[i][2])) > 1e-12: bad += 1
if bad:
    sys.exit('ABORT: %d rows out of input order / target mismatch' % bad)
print('row-order gate OK: %d rows match input TSV order on stem + human_score' % len(p))
PY

  echo "== NOTE: staged pixels remain at $PNGDIR (~15.2 GB). This script NEVER deletes them"
  echo "         (they cost ~5 min of R2 fetch to recreate); remove by hand when the wave closes."
  echo "KADIS-R4-DONE $(ts)  -> $CSV"
}

# ---------------------------------------------------------------- main
stage=${1:-all}
preflight
case "$stage" in
  resolve) do_resolve ;;
  fetch)   do_fetch ;;
  pairs)   do_pairs ;;
  extract) do_extract ;;
  all)     do_resolve; do_fetch; do_pairs; do_extract ;;
  *) die "unknown stage '$stage' (resolve|fetch|pairs|extract|all)" ;;
esac
