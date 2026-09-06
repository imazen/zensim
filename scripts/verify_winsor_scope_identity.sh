#!/usr/bin/env bash
# REV2-D-GUARD (docs/PLAN_FEATURE_REV2_2026-09-05.md section 12.7) — the gate
# for the two `bake_dial_refit` owner extensions landed 2026-09-06:
#
#   1. `add-winsor --slots`      restrict the guard to a slot subset
#   2. `shared-anchor --anchor`  became REPEATABLE
#
# Three checks, each with the control that makes it mean something:
#
#   A. DEFAULT BYTE-IDENTITY, against a PRE-CHANGE artefact. Re-emit the two
#      `W-all-carried_s<slice>_wins.bin` bakes the REV2-D lane produced with
#      the pre-`--slots` binary and require the stored sha256. This is the
#      real regression bar: every B-lineage and `W-all-carried` artefact on
#      disk came out of that code path.
#   B. NEGATIVE CONTROL for `--slots`. A scoped emit must DIFFER from the
#      unscoped one — otherwise check A would pass on a flag that does
#      nothing — and must declare exactly the requested count.
#   C. `shared-anchor`'s concatenation is a PURE ROW-APPEND: two `--anchor`
#      files must give byte-identically what one physically concatenated
#      parquet of the same rows in the same order gives. (A single-anchor
#      "control" would be tautological — one element of a loop — so the
#      non-trivial property is the one gated.)
#
# Inputs are the REV2-D lane's own artefacts; nothing here writes into them.
set -euo pipefail

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BDR=${BDR:-$WS/target/tgt-dguard/release/bake_dial_refit}
SRC=${SRC:-/mnt/v/output/zensim/rev2-d-arms-2026-09-06}
R6B=${R6B:-/mnt/v/output/zensim/rev2-2026-09-05/r6b}
ID21=${ID21:-/mnt/v/output/zensim/rev2-refit-2026-09-06/work/identity_anchor_ratio_n21.parquet}
TMP=${TMP:-$HOME/tmp/winsor-scope-gate}

[ -x "$BDR" ] || { echo "GATE: no bake_dial_refit at $BDR (build it, or set BDR)"; exit 2; }
mkdir -p "$TMP"
fail=0
say() { printf '%s\n' "$*"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

# ---- A. default scope reproduces the stored pre-change bytes ----------------
declare -A WANT=(
  [156]=e6abdb1347946429e7b8c1b7fd461ed75a8281f463d2a5161ac4c7ea1a3cde2f
  [228]=b468ce97c2bc364e6251733be6a8594ce6f7ccd3a30948b953899b3b2ad389f1
)
for s in 156 228; do
  out=$TMP/A_s${s}.bin
  "$BDR" add-winsor --in "$SRC/bakes/D_ratio_s${s}_raw.bin" --out "$out" \
      --fit-corpus "$R6B/tables/ratio/safesyn.parquet" >/dev/null 2>&1
  got=$(sha "$out")
  if [ "$got" = "${WANT[$s]}" ]; then
    say "A s$s: PASS — default scope BYTE-IDENTICAL to the pre-change artefact ($got)"
  else
    say "A s$s: FAIL — got $got, stored ${WANT[$s]}"; fail=1
  fi
done

# ---- B. negative control: a scoped emit must differ, and by the right count -
F17=12,25,38,51,64,77,90,103,116,129,142,155
"$BDR" add-winsor --in "$SRC/bakes/D_ratio_s156_raw.bin" --out "$TMP/B_f17.bin" \
    --fit-corpus "$R6B/tables/ratio/safesyn.parquet" --slots "$F17" >/dev/null 2>&1
if [ "$(sha "$TMP/B_f17.bin")" = "${WANT[156]}" ]; then
  say "B: FAIL — --slots produced the UNSCOPED bytes; the flag is a no-op"; fail=1
else
  say "B: PASS — --slots changes the bytes (negative control holds)"
fi
ZP=${ZP:-/home/lilith/work/zen/zenanalyze/target/release/zenpredict}
if [ -x "$ZP" ]; then
  n_w=$("$ZP" inspect "$TMP/B_f17.bin" 2>/dev/null | python3 -c '
import sys, json
t = sys.stdin.read(); d = json.loads(t[t.index("{"):])
md = {e["key"]: e.get("value_text", "") for e in d["metadata"]}
toks = md["zentrain.feature_transforms"].split("\n")
pars = md["zentrain.feature_transform_params"].split("\n")
w = [i for i, x in enumerate(toks) if x.strip() == "winsor_p99"]
# every unguarded slot must be identity with EMPTY params, and the two
# entries must both be exactly n lines long (build_fw_ops refuses otherwise)
assert len(toks) == len(pars) == 372, f"line counts {len(toks)}/{len(pars)}"
assert all(toks[i] == "identity" and pars[i] == "" for i in range(372) if i not in w), "unguarded slot is not empty identity"
assert all(pars[i] != "" for i in w), "guarded slot has no window"
print(len(w))
')
  if [ "$n_w" = "12" ]; then
    say "B: PASS — the scoped bake declares exactly 12 winsor_p99 tokens over 372 lines, every other slot identity with empty params"
  else
    say "B: FAIL — scoped bake declares ${n_w:-?} winsor_p99 tokens, expected 12"; fail=1
  fi
else
  say "B: SKIPPED token count — no zenpredict at $ZP (set ZP); the byte-difference control above still ran"
fi

# ---- C. shared-anchor: two --anchor == one concatenated parquet -------------
CAT=$TMP/anchor_cat.parquet
python3 - "$R6B/tables/ratio/anchor.parquet" "$ID21" "$CAT" <<'PY'
import sys, pyarrow as pa, pyarrow.parquet as pq
a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
cols = [f"f{i}" for i in range(372)] + ["human_score"]
ta, tb = pq.read_table(a, columns=cols), pq.read_table(b, columns=cols)
tb = tb.cast(ta.schema)
pq.write_table(pa.concat_tables([ta, tb]), out, compression="zstd")
print(f"  concat: {ta.num_rows} + {tb.num_rows} = {ta.num_rows + tb.num_rows} rows -> {out}")
PY
"$BDR" shared-anchor --in "$SRC/bakes/D_ratio_s156_raw.bin" --out "$TMP/C_two.bin" \
    --anchor "$R6B/tables/ratio/anchor.parquet" --anchor "$ID21" \
    --target-col human_score >/dev/null 2>&1
"$BDR" shared-anchor --in "$SRC/bakes/D_ratio_s156_raw.bin" --out "$TMP/C_one.bin" \
    --anchor "$CAT" --target-col human_score >/dev/null 2>&1
if [ "$(sha "$TMP/C_two.bin")" = "$(sha "$TMP/C_one.bin")" ]; then
  say "C: PASS — two --anchor == one concatenated parquet ($(sha "$TMP/C_two.bin"))"
else
  say "C: FAIL — two-anchor $(sha "$TMP/C_two.bin") != concat $(sha "$TMP/C_one.bin")"; fail=1
fi
# and the anchor SET must matter at all (else C would be vacuous)
"$BDR" shared-anchor --in "$SRC/bakes/D_ratio_s156_raw.bin" --out "$TMP/C_solo.bin" \
    --anchor "$R6B/tables/ratio/anchor.parquet" --target-col human_score >/dev/null 2>&1
if [ "$(sha "$TMP/C_solo.bin")" = "$(sha "$TMP/C_two.bin")" ]; then
  say "C: FAIL — dropping the id100 anchor changed nothing; the extra rows are inert"; fail=1
else
  say "C: PASS — the second anchor moves the spline (negative control holds)"
fi

[ $fail -eq 0 ] && say "verify_winsor_scope_identity: ALL PASS" || say "verify_winsor_scope_identity: FAILURES"
exit $fail
