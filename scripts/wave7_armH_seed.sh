#!/usr/bin/env bash
#
# wave7_armH_seed.sh <seed> — ONE arm-H training run (SOTA-944 WAVE 7,
# benchmarks/sota944_campaign_2026-08-03.md amendment 7, commit 62f0bcc3).
#
# Arm H = the C_co3a argv VERBATIM (recovered from C_co3a_s1301.bin.spec.json:
# the em944 base + the three distillation-twin teacher groups) + EXACTLY two
# group additions — the wave-7 reference-disjoint KonJND BPG leg at the v47
# weight precedent (train 1.2 on the train half, val 1.5 on the disjoint val
# half). Registered seeds: 2501 2503 2507. Tag: H_co3abpg_s<seed>.
#
# WAVE7_ECHO=1 prints the argv (one token per line) instead of training —
# the pre-flight equivalence gate against the stored co3a spec argv.
set -euo pipefail

SEED=${1:?usage: wave7_armH_seed.sh <seed>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
# Env-overridable data roots (fleet nodes stage locally; defaults = the
# local /mnt/v canonicals).
E=${SOTA944_E:-/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01}
T=${SOTA944_T:-/mnt/v/zen/zensim-training}
K=${SOTA944_K:-/mnt/v/zen/zensim-training/kadis-944-2026-08-01}
TE=${SOTA944_TEACHER:-/mnt/v/output/zensim/bakes/sota944/teacher}
OUT=${SOTA944_OUT:-/mnt/v/output/zensim/bakes/sota944/bakes}
mkdir -p "$OUT"
BAKE="$OUT/H_co3abpg_s${SEED}.bin"
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }

# WT40 + MASK2 = the 64 transform flags, verbatim from the co3a embedded repro
# (identical to sota944_armC_seed_w4.sh's list; kept inline so this driver is
# self-contained on fleet nodes).
FT=(
  winsor_p99:100:1.46128e-06,0.000106967 winsor_p99:139:4.06965e-07,4.32286e-05
  signed_cbrt:61: winsor_p99:152:2.44617e-16,4.53981e-05
  winsor_p99:126:5.45232e-16,5.2143e-05 winsor_p99:113:2.55688e-16,6.55178e-05
  signed_cbrt:22: winsor_p99:87:1.33566e-15,7.64692e-05
  signed_cbrt:132: signed_cbrt:130: signed_cbrt:131:
  winsor_p99:74:3.01356e-16,8.6061e-05 winsor_p99:134:0.000331953,0.00794472
  winsor_p99:48:4.53039e-15,0.000103628 winsor_p99:155:0,0.163769
  winsor_p99:35:4.23601e-16,0.000101019 winsor_p99:135:0.000199307,0.00491572
  winsor_p99:129:0,0.203516 winsor_p99:9:4.04794e-07,4.9929e-05
  winsor_p99:90:0,0.442516 winsor_p99:116:0,0.329577
  winsor_p99:133:3.24358e-05,0.00573569 winsor_p99:142:0,0.0740237
  winsor_p99:51:0,0.381636 signed_cbrt:92: winsor_p99:77:0,0.285164
  signed_cbrt:93: winsor_p99:12:0,0.318713 signed_cbrt:137: signed_cbrt:38:
  signed_cbrt:91: winsor_p99:95:0.000610845,0.0131861
  winsor_p99:64:0,0.000272277 winsor_p99:141:0,0.0431522
  winsor_p99:96:0.000119094,0.0137795 winsor_p99:118:1.4128e-05,0.0366709
  winsor_p99:144:3.92265e-05,0.0499891 winsor_p99:103:0,0.0600395
  winsor_p99:102:0,0.0443022 winsor_p99:119:5.98459e-06,0.0301378
  winsor_p99:731:0,0 winsor_p99:748:0,0 winsor_p99:765:0,0 winsor_p99:782:0,0
  winsor_p99:799:0,0 winsor_p99:816:0,0 winsor_p99:833:0,0 winsor_p99:850:0,0
  winsor_p99:867:0,0 winsor_p99:884:0,0 winsor_p99:901:0,0 winsor_p99:918:0,0
  winsor_p99:732:0,0 winsor_p99:749:0,0 winsor_p99:766:0,0 winsor_p99:783:0,0
  winsor_p99:800:0,0 winsor_p99:817:0,0 winsor_p99:834:0,0 winsor_p99:851:0,0
  winsor_p99:868:0,0 winsor_p99:885:0,0 winsor_p99:902:0,0 winsor_p99:919:0,0
)
FT_ARGS=()
for t in "${FT[@]}"; do FT_ARGS+=(--feature-transform "$t"); done

ARGS=(
  --group "safesyn:$E/ext_safesyn_full.parquet:1.0:0.5:both"
  --group "cid22_train:$E/ext_cid22_train201.parquet:1.0:2.0:both"
  --group "kadid:$E/ext_kadid.parquet:0.5:1.0:rank"
  --group "tid:$E/ext_tid.parquet:0.5:1.0:rank"
  --group "bigcodec:$T/tbig_944_200k.parquet:0.5:1.0:both"
  --group "kadis:$K/kadis_944_ssim2_50k.parquet:0.15:1.0:both"
  --group "tsafesyn:$TE/safesyn_teacher944.parquet:0.5:1.0:both"
  --group "ttbig:$TE/tbig_teacher944.parquet:0.5:1.0:both"
  --group "tkadis:$TE/kadis_teacher944.parquet:0.5:1.0:both"
  --group "konjnd_bpg:$E/konjnd_bpg_train_944.parquet:1.2:0.0:both"
  --group "konjnd_bpg_val:$E/konjnd_bpg_val_944.parquet:0.0:1.5:both"
  --n-hidden-layers 0 --target-column human_score --target-scale 100
  --epochs 120 --pairs-per-epoch 50000 --seed "$SEED"
  --max-features 944 --allow-narrow-features
  --coarse-decay 1e-5
  "${FT_ARGS[@]}"
  --out "$BAKE"
)

if [[ "${WAVE7_ECHO:-0}" = "1" ]]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi
[[ -x "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 2; }
exec "$TRAIN" "${ARGS[@]}"
