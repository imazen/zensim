#!/bin/bash
set -euo pipefail
D=/mnt/v/output/zensim/dvifm-screen-2026-09-19/probe4/run
TRAIN=/home/lilith/work/zen/zensim/target/release/zensim_mlp_train
python3 - <<'PY' > /tmp/devin_probe4_arms.txt
import json
r=json.load(open('/mnt/v/output/zensim/dvifm-screen-2026-09-19/probe4/recipe.json'))
for arm,ids in r['arms'].items():
    print(arm + ' ' + ','.join(map(str,ids)))
PY
while read -r arm keep; do
  "$TRAIN" \
    --group "fit:$D/human_train.parquet:1:0:withinref,both" \
    --group "dev:$D/human_eval.parquet:0:1" \
    --target-column human_score --target-scale 1 --hidden 128 \
    --epochs 160 --pairs-per-epoch 8192 \
    --seed 9201 --init-seed 9201 --sample-seed 19201 \
    --pair-sampling stratified --max-features 986 \
    --keep-features "$keep" \
    --mse-weight 1 --early-stop-patience 0 --out-dtype f32 \
    --log-every 1 --no-auto-eval \
    --out "$D/probe-$arm.bin" > "$D/probe-$arm.log" 2>&1 &
done < /tmp/devin_probe4_arms.txt
wait
