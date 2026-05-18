#!/usr/bin/env bash
# Eval all V_22-372feat bakes via bake_verdict; emit per-bake markdown
# and aggregate the 5-seed CI panels into a CSV for analysis.
set -euo pipefail

OUT_DIR="/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18"
WORKSPACE="/home/lilith/work/zen/zensim--372feat"
VERDICT="${WORKSPACE}/target/release/bake_verdict"

PANELS_CSV="${OUT_DIR}/seed_summary_panels.tsv"
printf 'variant\tseed\tcorpus\tn\tSROCC\tPLCC\tKROCC\tOR\tPWRC\tZ-RMSE\n' > "${PANELS_CSV}"

for bake in "${OUT_DIR}"/*.bin; do
    name=$(basename "${bake}" .bin)
    md="${OUT_DIR}/verdict_${name}.md"
    if [ ! -f "${md}" ] || [ "${bake}" -nt "${md}" ]; then
        echo "Evaluating ${name}..."
        "${VERDICT}" --bake "${bake}" --output "${md}" 2>&1 | tail -1
    fi
    # Variant + seed parsing
    if [[ "${name}" =~ noLARGE_s([0-9]+) ]]; then
        variant="372feat_noLARGE"
        seed="${BASH_REMATCH[1]}"
    elif [[ "${name}" =~ v22_372feat_s([0-9]+) ]]; then
        variant="372feat_5grp"
        seed="${BASH_REMATCH[1]}"
    else
        continue
    fi
    # Extract per-corpus rows from the Summary table
    awk -v VARIANT="${variant}" -v SEED="${seed}" '
        /^\| Corpus \| n/ { in_table=1; next }
        /^\|---/ && in_table==1 { in_table=2; next }
        in_table==2 && /^$/ { in_table=0 }
        in_table==2 && /^\|/ {
            gsub(/\|/, "\t")
            gsub(/[[:space:]]+/, "")
            sub(/^\t/, "")
            sub(/\t$/, "")
            n=split($0, f, /\t/)
            if (n>=7) {
                printf "%s\t%s\t%s\n", VARIANT, SEED, $0
            }
        }
    ' "${md}" >> "${PANELS_CSV}"
done

echo "Per-bake markdowns under ${OUT_DIR}/"
echo "Panels CSV: ${PANELS_CSV}"
wc -l "${PANELS_CSV}"
