#!/usr/bin/env bash
# paper-holdout lane: reference-clustered paired bootstrap (the ssim2-bar owner,
# benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py) on every corpus,
# deltas against fast-ssim2 and against B, same seed as the committed exam.
set -uo pipefail
A=${OUT:-/var/tmp/paper-holdout/a}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OWNER=$REPO/benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py
export ZEN_PANEL_BIN=$A/bin/panel TMPDIR=$A/tmp
mkdir -p $A/boot $A/tmp
fails=0
# Regression check of the 2026-09-23 owner edits WITHOUT touching CID22 (the
# coordinator ruling forbids new 49-reference statistics): the owner as
# committed at main (ffc14647) and as edited here, default arms, CSIQ,
# BOOT=2000, must print byte-identical output.
echo "== $(date -u +%FT%TZ) regression: original vs edited owner on CSIQ (BOOT=2000)"
# NOTE: lane workspaces are jj workspaces without a colocated .git; `git -C`
# fails here. `jj file show -r <rev>` resolves the committed owner instead.
(cd $REPO && jj file show -r ffc14647 benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py) > $A/tmp/owner_ffc14647.py
PYTHONPATH=$REPO/scripts/lib CORPUS=csiq BOOT=2000 python3 $A/tmp/owner_ffc14647.py > $A/boot/regression_csiq_orig.txt 2>&1 || fails=$((fails+1))
env -u ARMS -u REF_ARM -u ARM_CI CORPUS=csiq BOOT=2000 python3 $OWNER > $A/boot/regression_csiq_edited.txt 2>&1 || fails=$((fails+1))
if cmp -s $A/boot/regression_csiq_orig.txt $A/boot/regression_csiq_edited.txt; then
  echo "REGRESSION OK: byte-identical"; else echo "REGRESSION MISMATCH"; diff $A/boot/regression_csiq_orig.txt $A/boot/regression_csiq_edited.txt | head -20; fails=$((fails+1)); fi
for c in cid22a csiq aic3 aic4 aic4full sdr25 konjnd; do
  for ref in ssim2 B; do
    echo "== $(date -u +%FT%TZ) $c vs $ref"
    O=$A/pp CORPUS=$c REF_ARM=$ref ARM_CI=1 ARMS="B GMSD DVtalk DVours DVgate" \
      python3 $OWNER > $A/boot/${c}_vs_${ref}.txt 2>&1 || { echo "BOOT FAIL $c $ref"; tail -5 $A/boot/${c}_vs_${ref}.txt; fails=$((fails+1)); }
  done
done
echo "== $(date -u +%FT%TZ) boot done fails=$fails"
[ $fails -eq 0 ]
