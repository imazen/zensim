# `benchmarks/featsub/` — appendix-J feature-subset ablation

Registered protocol: `benchmarks/sota944_campaign_2026-08-03.md`, **REGISTERED
APPENDIX J — IS 944 TOO MANY?** (committed before any fit in the pass).

| file | what it is |
|---|---|
| `idx/ranked.tsv` | the frozen contribution ranking of all 944 inputs (`mean_abs` desc, ties by idx asc) from `bake_contrib_H_co3abpg_s2507_2026-08-04.tsv` |
| `idx/top{64,128,256,512,667}.idx` | the `--keep-features` index files the K arms train on |
| `live_input_structure_2026-08-04.tsv` | live/dead/constant-column counts per model class + the cross-model ranking-agreement numbers (in the `.meta`) |
| `k128_stage_map_2026-08-05.tsv` | every K128 index decoded to (block, scale, channel, local, producing pass) — the extraction-stage map (`benchmarks/k128_stage_map_2026-08-05.md`; generator `scripts/featsub/k128_stage_map.py`) |

Tools (all owner extensions — no new trainer, no Python fit):

- `zensim_mlp_train --keep-features SPEC` — exact K-wide fit
- `zensim_mlp_train --group-l1 LAMBDA` — decoupled group-lasso proximal step
- `bake_contrib --live-mask TSV` — structural per-input live/dead dump
- `scripts/featsub/topk_from_contrib.py` — ranking → index files
- `scripts/featsub/featsub_seed.sh` — one cell, argv inherited from `wave7_armH_seed.sh`
- `scripts/featsub/featsub_queue2.sh` — N-worker cell runner (per-cell locks, RAM gate, start mutex)
- `scripts/featsub/featsub_verdicts.sh` — the campaign's ONE verdict invocation over these bakes
- `scripts/featsub/featsub_table.py` — the K sweep against the frozen ±2·sd band
- `scripts/featsub/featsub_stability.py` — stability selection over the λ×seed grid

Bakes live at `/mnt/v/output/zensim/bakes/featsub/`; verdicts share the campaign
store `/mnt/v/output/zensim/bakes/sota944/verdicts/` with an `FS_` stem prefix.

## A note on the dropped columns' scaler entries

`--keep-features` zeroes a dropped column's raw value *after* the recipe's
feature transforms run, so the scaler sees a constant-zero column: `mean = 0`
and `std` floored to `1e-8`. At inference the un-pruned bake therefore
standardizes a dropped input to `t(x)/1e-8` — a large but finite number — and
multiplies it by a weight row that is exactly zero. `x * 0.0 == 0.0` for every
finite `x`, so the contribution is exactly zero, which is what the pack
identity gate demonstrates empirically: 2,035 anchor scores came out
**bit-identical** between the un-pruned K64 net and its pruned twin. After
`bake_dial_refit pack` the columns carry `FeatureTransform::Drop` and are never
read at all, so the shipping artifact does not depend on that argument.
