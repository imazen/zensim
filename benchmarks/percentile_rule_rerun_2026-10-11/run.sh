#!/usr/bin/env bash
# Re-run recorded bake_compare decisions under both percentile rules.
# Same binary source (zensim main 8ef81d0f) built against zenstats 3e16cb45
# (order statistics) and 67b2ba74 (type 7); B=1000, seed 42 as recorded.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY' > jobs.tsv
import json
for i,p in enumerate(json.load(open('pairs.json'))):
    if p['a_run'] and p['b_run']:
        print(f"{i}\t{p['report'].split('/')[-1][:-3]}\t{p['a_run']}\t{p['b_run']}")
PY
while IFS=$'\t' read -r i name a b; do
  for rule in orderstat type7; do
    out="out/${name}__${rule}"
    [ -s "$out.json" ] && continue
    ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- bin/bake_compare_$rule --a "$a" --b "$b" \
      --corpora cid22,kadid,tid,konjnd,aic3 --features-root "$HOME/tmp/pctl/root" \
      --bands 10 --bootstrap-resamples 1000 --seed 42 --output "$out.md" --json "$out.json" > "$out.log" 2>&1
    echo "$(date -u +%FT%TZ) done $name $rule" >> progress.log
  done
done < jobs.tsv
echo "$(date -u +%FT%TZ) ALL DONE" >> progress.log
