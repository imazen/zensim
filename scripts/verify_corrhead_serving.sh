#!/usr/bin/env bash
# The corruption-head serving gates, in one runnable place.
#
#   G1/G2  numeric parity: the Rust ZCTH evaluator vs the sklearn model it was
#          exported from, on the frozen test fold + the whole gate grid.
#   G3     composition identity: `bake_verdict` on the INCUMBENT logistic head,
#          before vs after adopting `zensim::corruption_head::gate_score`,
#          byte-identical in `--full-json`. Needs a baseline binary built from
#          the pre-change commit; pass it with ZL_BV_BASE, else G3 is reported
#          as NOT RUN (never as a pass).
#   G4     end-to-end: `bake_verdict --corruption-head <tree>` reproduces the
#          theory lane's published gate-grid numbers through the Rust path.
#
# Pre-registration: docs/PLAN_CORRHEAD_SERVING_2026-09-06.md
# Record:           benchmarks/corruption_head_serving_2026-09-06.md
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-/mnt/v/output/zensim/corruption-head-2026-09-05}
TH=$ROOT/theories
BV=${ZL_BV:-./target/release/bake_verdict}
PARITY=${ZL_PARITY:-./target/release/corrhead_parity}
GRID=$ROOT/corruption_grid_372col_postC_2026-09-05.parquet
DIAL=zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin
TREE=$TH/corrhead_hgb_theoryfit_w372.zcth
LOGI=$ROOT/d228/corruption_head_d228.bin
OUT=${OUT:-$HOME/tmp/corrserve/gates}
mkdir -p "$OUT"
fail=0
say() { printf '%s\n' "$*"; }

for f in "$BV" "$PARITY" "$GRID" "$DIAL" "$TREE" "$LOGI" "$TH/parity_hgb.npz"; do
  [ -e "$f" ] || { say "MISSING $f"; fail=1; }
done
[ $fail -eq 0 ] || { say "verify_corrhead_serving: FAIL (missing inputs)"; exit 1; }

say "== G1 + G2: numeric parity =="
"$PARITY" --head "$TREE" --parity "$TH/parity_hgb.npz" || fail=1

say ""
say "== G3: composition identity on the incumbent logistic head =="
if [ -n "${ZL_BV_BASE:-}" ] && [ -x "${ZL_BV_BASE}" ]; then
  for arm in base new; do
    bin=$BV; [ "$arm" = base ] && bin=$ZL_BV_BASE
    "$bin" --bake "$DIAL" --corpora cid22 --corruption-grid "$GRID" \
      --corruption-head "$LOGI" --full-json "$OUT/g3_$arm.json" \
      --output "$OUT/g3_$arm.md" >"$OUT/g3_$arm.log" 2>&1 || { say "G3 $arm rc=$?"; fail=1; }
  done
  if cmp -s "$OUT/g3_base.json" "$OUT/g3_new.json"; then
    say "  --full-json BYTE-IDENTICAL  ($(sha256sum "$OUT/g3_new.json" | cut -c1-16)...)  PASS"
  else
    say "  --full-json DIFFERS  FAIL"; fail=1
  fi
else
  say "  NOT RUN — set ZL_BV_BASE to a bake_verdict built from the pre-change commit."
  say "  (NOT RUN is not a pass; the gate is unmeasured without a baseline.)"
fi

say ""
say "== G4: end-to-end, tree head through bake_verdict =="
"$BV" --bake "$DIAL" --corpora cid22 --corruption-grid "$GRID" \
  --corruption-head "$TREE" --full-json "$OUT/g4.json" \
  --output "$OUT/g4.md" >"$OUT/g4.log" 2>&1 || { say "  bake_verdict rc=$?"; fail=1; }
python3 - "$OUT/g4.json" "$TH/t6_gate_pass.tsv" <<'PY' || fail=1
import json, sys
j = json.load(open(sys.argv[1]))
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[2])]
hdr, body = rows[0], rows[1:]
want = {r[0]: dict(zip(hdr, r)) for r in body}
ok = True
def chk(name, got, exp):
    global ok
    # The published TSV carries 6 significant figures; compare there and print
    # the full-precision Rust value beside it.
    good = f"{got:.6g}" == f"{float(exp):.6g}"
    ok &= good
    print(f"  {name:34s} rust {got!r:22s} python {exp:>10s}  {'PASS' if good else 'FAIL'}")
h, d, dial = j["corruption_head"], j["corruption_deploy"], j["corruption"]
chk("head pass_q20",      h["pass_q20"], want["hgb"]["head_pass_q20"])
chk("head pass_q10",      h["pass_q10"], want["hgb"]["head_pass_q10"])
chk("DEPLOY pass_q20",    d["pass_q20"], want["hgb"]["deploy_pass_q20"])
chk("DEPLOY pass_q10",    d["pass_q10"], want["hgb"]["deploy_pass_q10"])
chk("D dial alone q20", dial["pass_q20"], want["D dial alone"]["deploy_pass_q20"])
chk("D dial alone q10", dial["pass_q10"], want["D dial alone"]["deploy_pass_q10"])
print(f"  applied deadband (score units): {d['threshold']!r}")
sys.exit(0 if ok else 1)
PY

say ""
if [ $fail -eq 0 ]; then say "verify_corrhead_serving: PASS"; else say "verify_corrhead_serving: FAIL"; fi
exit $fail
