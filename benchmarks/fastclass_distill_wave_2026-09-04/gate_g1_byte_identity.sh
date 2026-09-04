#!/bin/bash
# GATE G1 (fastclass wave, run BEFORE any arm): the extended
# train_156_student.sh, with every new lever UNSET, must reproduce A4b's own
# bake. If it does not, the four new env vars are not the no-ops they claim to
# be and every Δ-vs-control in this wave would be measuring the script edit.
#
# ⚠ "REPRODUCE" IS NOT "sha256 OF THE WHOLE FILE" — measured 2026-09-04, the
# hard way. Every bake carries a MANDATORY embedded `zentrain.repro` section
# (argv + timestamp + input shas; CLAUDE.md "MANDATORY embedded repro"), so two
# runs that differ only in `--out` embed different argv and different
# timestamps and can NEVER be byte-identical. The first version of this gate
# compared raw sha256 and FAILED on exactly that: A4b 509,024 B vs C0 509,021 B,
# a 3-byte delta equal to the output-path length difference, with `best_val`
# identical to 16 significant digits. The gate was wrong, not the extension.
#
# The corrected gate compares the MODEL, two independent ways:
#   (a) sha256 with `zentrain.repro` stripped by the owner (`bake_dial_refit
#       strip`) — the provenance blob removed, every weight byte kept;
#   (b) bit-exact predictions over ext_cid22val through the production forward.
# Both must hold.
set -euo pipefail
REPO="${FCD_REPO:-/home/lilith/work/zen/zensim}"
cd "$REPO"
BIN=/mnt/v/zen/cargo-targets/waver4/release
export ZL_TRAIN="$BIN/zensim_mlp_train"
export WR4_KEEP="$REPO/scripts/sota944/slice_basic156_free.txt"
export WR4_DISTILL_SAFESYN=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views/safesyn_distill_hya_r4.parquet
R4=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01
O=/mnt/v/output/zensim/fastclass-2026-09-04
export WR4_OUT="$O/bakes"
W="${FCD_HB_DIR:-$HOME/tmp/fastclass}"
mkdir -p "$WR4_OUT" "$W"
REF=/mnt/v/output/zensim/wave-r4-2026-09-01/bakes/A4b_156_s4004.bin
OUT="$WR4_OUT/C0_s4004.bin"
if [ ! -f "$OUT" ]; then
  "$HOME/work/zen/scripts/run-heavy" --mem 16G --jobs 8 -- \
    "$REPO/benchmarks/wave_r4_2026-09-01/train_156_student.sh" distill 4004 "$OUT"
fi
rc=0
# (a) weights identical with the provenance section removed
"$BIN/bake_dial_refit" strip --in "$REF" --out "$W/g1_ref.norepro.bin" --key zentrain.repro >/dev/null
"$BIN/bake_dial_refit" strip --in "$OUT" --out "$W/g1_new.norepro.bin" --key zentrain.repro >/dev/null
SA=$(sha256sum "$W/g1_ref.norepro.bin" | cut -d' ' -f1)
SB=$(sha256sum "$W/g1_new.norepro.bin" | cut -d' ' -f1)
echo "(a) sha256 sans zentrain.repro:  A4b=$SA"
echo "                                  C0 =$SB"
if [ "$SA" = "$SB" ]; then echo "(a) PASS"; else echo "(a) FAIL"; rc=1; fi
# (b) bit-exact predictions through the production forward
"$BIN/bake_dial_refit" predict --bake "$REF" --corpus "$R4/ext_cid22val.parquet" --out "$W/g1_ref.pred.tsv" >/dev/null
"$BIN/bake_dial_refit" predict --bake "$OUT" --corpus "$R4/ext_cid22val.parquet" --out "$W/g1_new.pred.tsv" >/dev/null
if cmp -s "$W/g1_ref.pred.tsv" "$W/g1_new.pred.tsv"; then
  echo "(b) PASS — predictions bit-identical on $(( $(wc -l < "$W/g1_ref.pred.tsv") - 1 )) CID22 rows"
else echo "(b) FAIL — predictions differ"; rc=1; fi
# for the record: the raw-file delta the mis-specified gate tripped on
echo "(note) raw file sizes: $(stat -c%s "$REF") vs $(stat -c%s "$OUT") bytes — the embedded argv path-length delta"
if [ "$rc" = 0 ]; then echo "GATE G1: PASS (model identical; provenance section differs by construction)"; exit 0; fi
echo "GATE G1: FAIL — the script extension is NOT a no-op when unset"; exit 1
