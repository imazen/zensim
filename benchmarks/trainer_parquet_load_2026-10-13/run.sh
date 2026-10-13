#!/usr/bin/env bash
# Existing trainer Parquet loader, complete 699,999-row Float32 table.
set -euo pipefail

if (( $# != 1 )); then
    echo "usage: $0 NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
out=$1
mkdir "$out"
input=/mnt/v/output/zensim/kadis-944-2026-08-01/kadis700k_944.parquet
binary=target/release/parquet_load_bench
expected=6ab12d50c41f94bef871d4b63257e05bbe481c95b7f1896d496ce5e817daf9a9
sha256sum "$input" "$binary" > "$out/inputs-before.sha256"
if [[ $(awk 'NR==1 {print $1}' "$out/inputs-before.sha256") != "$expected" ]]; then
    echo "Parquet input differs from the exact scientific storage study" >&2
    exit 1
fi
rustc --version > "$out/rustc-version.txt"
printf 'benchmark_source=' > "$out/source.txt"
jj log -r main --no-graph -T 'commit_id' >> "$out/source.txt"
printf '\n' >> "$out/source.txt"

selected=$(seq -s, 440 503)
for trial in 1 2 3; do
    if (( trial % 2 == 1 )); then
        order=(flat64 compact32)
    else
        order=(compact32 flat64)
    fi
    for mode in "${order[@]}"; do
        args=("$input" score_iwssim_gpu 1.0 "$mode")
        if [[ $mode == compact32 ]]; then
            args+=("$selected" 944)
        fi
        printf 'trial=%s mode=%s\n' "$trial" "$mode" >&2
        /usr/bin/time -v -o "$out/$mode-$trial.time" \
            "$binary" "${args[@]}" > "$out/$mode-$trial.json"
        jq -e --arg mode "$mode" '
            .mode == $mode and .rows == 699999 and .logical_features == 944 and
            .stored_values == (if $mode == "flat64" then 660799056 else 44799936 end) and
            .file_bytes == 2523688313 and .load_ns > 0
        ' "$out/$mode-$trial.json" > /dev/null
    done
done
sha256sum "$input" "$binary" > "$out/inputs-after.sha256"
if ! cmp -s "$out/inputs-before.sha256" "$out/inputs-after.sha256"; then
    echo "input or binary changed during measurement" >&2
    exit 1
fi
echo 'exact source and program identities unchanged; six loader runs completed' >&2
