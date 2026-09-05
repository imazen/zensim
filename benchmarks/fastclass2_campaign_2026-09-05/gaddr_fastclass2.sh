#!/bin/bash
# fastclass2 — the FULL G-ADDR grading of a 944-POOLS fast-class bake.
#
# `score_arm.sh` (the wave-r4 owner this campaign reuses for every arm) passes
# no probes, so its verdicts read C3/C4/C5/C6 as NOT MEASURED and A7r as NOT
# MEASURED. That is correct for a sweep -- the probes cost nothing to skip and
# nothing to add later -- but the ship rule needs the contract COMPLETE. This
# script re-grades one bake with every instrument attached. It retrains nothing
# and re-packs nothing.
#
# Instruments, and why each one:
#   dial grid   the campaign's standing 944-POOLS grid, so a candidate's dial
#               numbers stay comparable with every FC_*/A3b cell.
#   ladder grid + its ssim2 truth TSV: REQUIRED by `--floor-rule resolvable`,
#               which computes both its window and its bar live from the
#               mentor's own per-cell scores (benchmarks/
#               ladder_floor_resolution_2026-09-05.md). A7r cannot be graded
#               without it.
#   probes      the D+free lane's 944-POOLS identity + negative-tail probes --
#               the only ones that exist at this regime. C5/C6 and C3/C4.
#
# Usage: gaddr_fastclass2.sh <LABEL> <bake.bin> [features-root]
set -euo pipefail
LABEL="${1:?label required}"; BAKE="${2:?bake required}"
ROOT="${3:-/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01}"
BIN="${FC2_BIN:-/mnt/v/zen/cargo-targets/fastclass2/release}"
W="${FC2_OUT:-/mnt/v/output/zensim/fastclass2-2026-09-05}"
GRID="${FC2_DIAL:-/mnt/v/output/zensim/wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet}"
LGRID=/mnt/v/output/zensim/ladder-2026-09-05/instruments/dial_grid_944col_ladder.parquet
LTRUTH=/mnt/v/output/zensim/ladder-2026-09-05/instruments/dialcells_ssim2_ladder.tsv
IDPROBE=/mnt/v/output/zensim/dfree-2026-09-05/probes/identity_probe_944pools_2026-09-05.parquet
NTPROBE=/mnt/v/output/zensim/dfree-2026-09-05/probes/negtail_probe_944pools_2026-09-05.parquet
mkdir -p "$W/gaddr"
for f in "$GRID" "$LGRID" "$LTRUTH" "$IDPROBE" "$NTPROBE" "$BAKE"; do
  [ -f "$f" ] || { echo "ABORT: missing instrument $f"; exit 2; }
done

# (a) standing grid — contract C1..C6 complete, comparable with every FC_* cell
nice -n19 ionice -c3 "$BIN/bake_verdict" --bake "$BAKE" --regime 944 \
  --features-root "$ROOT" --dial-grid "$GRID" \
  --negtail-probe "$NTPROBE" --identity-probe "$IDPROBE" \
  --name "${LABEL}@stdgrid" --gaddr-json "$W/gaddr/gaddr_${LABEL}_stdgrid.json" \
  --output "$W/gaddr/gaddr_${LABEL}_stdgrid.md" > "$W/gaddr/gaddr_${LABEL}_stdgrid.log" 2>&1

# (b) ladder grid + mentor truth — A7r floor representability under `resolvable`
nice -n19 ionice -c3 "$BIN/bake_verdict" --bake "$BAKE" --regime 944 \
  --features-root "$ROOT" --dial-grid "$LGRID" --gaddr-grid-truth "$LTRUTH" \
  --floor-rule resolvable \
  --negtail-probe "$NTPROBE" --identity-probe "$IDPROBE" \
  --name "${LABEL}@ladder" --gaddr-json "$W/gaddr/gaddr_${LABEL}_ladder.json" \
  --output "$W/gaddr/gaddr_${LABEL}_ladder.md" > "$W/gaddr/gaddr_${LABEL}_ladder.log" 2>&1

python3 - "$W/gaddr/gaddr_${LABEL}_stdgrid.json" "$W/gaddr/gaddr_${LABEL}_ladder.json" <<'PY'
import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p))
    a = d.get("addressability", d)
    print(f"== {p.split('/')[-1]}")
    print(f"   contract={a.get('contract')}  regression={a.get('regression')}  shippable={a.get('shippable')}")
    for c in a.get("checks", []):
        if c["tier"] == "contract" or c["id"].startswith("A7"):
            print(f"   {c['id']:4s} {c['state']:13s} bar={c.get('bar')} {c.get('cmp','')} measured={c.get('measured')}  :: {c['what'][:56]}")
PY
