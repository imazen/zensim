#!/usr/bin/env bash
# Within-image (per-reference) SROCC for the four reference metrics, from the
# stored per-pair tables — the axis a codec target loop consumes, which the
# board's peer rows leave empty. Every statistic is `panel --per-group`
# (zenstats::per_group_srocc); this script only renames columns.
set -uo pipefail
RM=/mnt/v/output/zensim/reports/refmetrics
O=${O:-/mnt/v/output/zensim/ssim2-bar-2026-08-31}
PANEL=${PANEL:?set PANEL to the panel binary}
printf 'metric\tcorpus\tn_groups\tmean\tmedian\tfrac_neg\tfrac_perfect\n'
for SPEC in \
  "ssim2:cid22:cid22_ssim2.tsv:MCOS:ssim2" \
  "ssim2:csiq:csiq_ssim2_gpu.tsv:human_score:ssim2_gpu" \
  "ssim2:live:live_ssim2_gpu.tsv:human_score:ssim2_gpu" \
  "ssim2:kadid:kadid_ssim2_gpu.tsv:DMOS:ssim2_gpu" \
  "ssim2:tid:tid_ssim2_gpu.tsv:MOS:ssim2_gpu" \
  "ssim2:aic3:aic3_ssim2_heldout.tsv:jnd:ssim2_gpu" \
  "butteraugli:cid22:cid22_butter.tsv:MCOS:butteraugli_max" \
  "cvvdp:cid22:cid22_cvvdp.tsv:MCOS:cvvdp" \
  "iwssim:cid22:cid22_iwssim.tsv:MCOS:iwssim" \
  ; do
  IFS=: read -r M C F HC MC <<<"$SPEC"
  [ -f "$RM/$F" ] || { printf '%s\t%s\tMISSING(%s)\t\t\t\t\n' "$M" "$C" "$F"; continue; }
  OUT="$O/perref_${M}_${C}.tsv"
  python3 - "$RM/$F" "$HC" "$MC" "$OUT" <<'PY' || { printf '%s\t%s\tCOLFAIL\t\t\t\t\n' "$M" "$C"; continue; }
import sys, csv, os
src, hc, mc, out = sys.argv[1:5]
with open(src, newline="") as fh:
    r = list(csv.DictReader(fh, delimiter="\t"))
if not r or hc not in r[0] or mc not in r[0]:
    sys.exit(f"missing {hc!r}/{mc!r} in {src}")
with open(out, "w") as w:
    w.write("predicted\ttarget\tband\n")
    for row in r:
        # band = the REFERENCE image; the group a codec ladder lives in.
        w.write(f"{row[mc]}\t{row[hc]}\t{os.path.basename(row['ref_path'])}\n")
PY
  LINE=$("$PANEL" --input "$OUT" --per-group 2>/dev/null | tail -1)
  set -- $LINE
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$M" "$C" "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}"
done
