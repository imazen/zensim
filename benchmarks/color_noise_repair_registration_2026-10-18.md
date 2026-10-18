# TRAIN-only color/noise repair, 2026-10-18

The owner authorized a matched 2×2 experiment after inspecting the LIVE
white-noise inversion documented in [the matrix trace](gauntlet_live_color_trace_2026-10-18.md).
LIVE has been exposed and is diagnostic only. It cannot choose this repair.

Four fixed arms, each at seed indices 0, 1, 2:

| Arm | Auxiliary coverage | Fine-scale X/B | Derived inputs |
|---|---|---|---|
| current | cf98: light, spatial, new | omitted | 410 direct + 410 products |
| coverage | cf9e: adds colour and noise | omitted | 410 + 410 |
| chroma | cf98 | restored | 492 + 492 |
| both | cf9e | restored | 492 + 492 |

All use the existing strict `v2_confirm_fit` owner, the retained original E33
native trainer/packer, H128 N head, coverage weight16, original loss/group
weights, 120 epochs ×50000 sampled pairs, and final epoch119. Initialization
seeds1101/1103/1107 pair with sample seeds101/100000101/200000101.
The additional declaration uses the existing same-cell difference×fragility
runtime representation. No new extractor or inference implementation is added.
Packing remains quantize-before-TRAIN-calibration with the original E33 output
stage; a refusal is retained as ineligible, never repaired by changing a gate.

Selection uses only the existing reference-disjoint TRAIN-development human
splits (four receipt-approved sources). For each repair versus its matched
current control, average canonical signed SROCC differences across the three
seeds. Require equal-source mean ≥−0.002 and each source mean ≥−0.005.
Require positive equal-source noise-subset SROCC improvement: KADID types11–14,
TID types1–7. Select the eligible arm with greatest noise improvement; ties use
human mean improvement, then registration order. Seed0 is the assessment model;
seeds1/2 are reported, never chosen. No eligible repair retains current.
This is exploratory screening, not the original ten-seed E33 qualification;
all original inferential, spatial, dial and product requirements still apply.

Before external reads, freeze the selected composition and model hashes.
Assess fixed candidates on previously exposed CSIQ/PIPAL and the exposed LIVE
regression examples through the native scorer and canonical panel statistics.
Report every arm, failed fits and refusals. No adaptive reruns are authorized
by this experiment. Protected sets and all exclusions in the gauntlet brief
remain untouched. Existing sparse ordinal ladders are reused; no newly dense
training sweep or shipping qualification is claimed.

Artifact root, checked on dev: `/mnt/v/output/zensim/color-noise-repair-2026-10-18/`.
`REGISTRATION.json` pins recipes, roles, original archive/binary hashes and seeds;
`ADMISSION.json` binds the seven TRAIN/development inputs and all504 read
features. Native logs and every fitted/packed model are retained. NAS mirror:
`/mnt/tower/output/zensim/color-noise-repair-2026-10-18/`.
