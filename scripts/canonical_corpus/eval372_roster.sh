#!/usr/bin/env bash
# Re-verdict the 372-class B lineage on BOTH extractor eras, same instrument.
#
# OLD root = /mnt/v/zen/zensim-training/2026-05-15-full-features (pre-fix:
#            masked/IW were a function of RAYON_NUM_THREADS — §3.27)
# NEW root = the dated root built by build_eval372_root.sh + pack_eval372_root.py
#
# Every run is `bake_verdict --full-json` with the DEFAULT corpus list, so the
# two eras differ in exactly one input: the feature tables.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BV="$REPO/target/release/bake_verdict"
OLD="${OLD_ROOT:-/mnt/v/zen/zensim-training/2026-05-15-full-features}"
NEW="${NEW_ROOT:-/mnt/v/zen/zensim-training/2026-08-30-full-features-372}"
OUT="${OUT_DIR:-/mnt/v/output/zensim/eval372-roster-2026-08-30}"
mkdir -p "$OUT"/{old,new,json,kon504}

# label|bake  — 372-caller-width board cells + the two 156-input immune controls.
# EXCLUDED (named): wlin4_a0.5 + C_co3a (944-input; the ext944 root is untouched
# by this drift and a 372-root read of them is the wrong-root class).
ROSTER=(
  "B_shipped|$REPO/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin"
  "v02_bvls_shaped|/mnt/v/output/zensim/bakes/v02_bvls_shaped_2026-05-28.bin"
  "v02_bvls_NO_shaping|/mnt/v/output/zensim/bakes/v02_bvls_NO_shaping_2026-05-28.bin"
  "blend_2L_H128|/mnt/v/output/zensim/reports/b_negatives/mlp_2L_diverse_H128_2026-07-15.bin"
  "v47A_strict_QAT|$REPO/zensim/weights/v47_strict_qat_native_2026-05-27.bin"
  "cl_tfm_LQ_MLP|/mnt/v/output/zensim/corr-lq/cl_tfm.bin"
  "BHdr_sdr_route|$REPO/zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin"
  "T_b_lam1e-3|/mnt/v/output/zensim/bakes/add156repro/bakes/T_b_lam1e-3.bin"
  "ADD156|/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin"
  "winner_dial|/mnt/v/output/zensim/corr-lq/Ebothg_hfgain_winsor_dial.bin"
  "Ebothg_scr05|/mnt/v/output/zensim/screen-retrain-2026-07-18/Ebothg_scr0.5_dial.bin"
)

run() { # run <label> <bake> <era> <root> <extra...>
  local label="$1" bake="$2" era="$3" root="$4"; shift 4
  "$BV" --bake "$bake" --features-root "$root" \
        --full-json "$OUT/json/${label}_${era}.json" \
        --output "$OUT/${era}/${label}.md" "$@" \
    > "$OUT/${era}/${label}.log" 2>&1 \
    || { echo "FAIL $label $era (see $OUT/$era/$label.log)"; return 1; }
}

for entry in "${ROSTER[@]}"; do
  label="${entry%%|*}"; bake="${entry#*|}"
  [ -f "$bake" ] || { echo "SKIP $label — bake missing: $bake"; continue; }
  echo "=== $label ==="
  run "$label" "$bake" old "$OLD"
  run "$label" "$bake" new "$NEW"
  # kon504 ruler: the JPEG half only, one-file side roots, konjnd slot.
  "$BV" --bake "$bake" --corpora konjnd --features-root "$OUT/storedroot_kon504" \
        --full-json "$OUT/kon504/${label}_old.json" --output /dev/null \
        > "$OUT/kon504/${label}_old.log" 2>&1 || echo "  kon504/old failed"
  "$BV" --bake "$bake" --corpora konjnd --features-root "$NEW/kon504" \
        --full-json "$OUT/kon504/${label}_new.json" --output /dev/null \
        > "$OUT/kon504/${label}_new.log" 2>&1 || echo "  kon504/new failed"
done
echo "done -> $OUT"
