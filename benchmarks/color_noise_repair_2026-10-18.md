# Color/noise repair comparison, 2026-10-18

The original single-model screen retains C. None of the three repairs clears
all registered TRAIN guards. The separately registered equal-weight
three-seed screen selects fine chroma, with positive human and noise deltas
and every per-source floor met. This is exploratory screening, not shipping
qualification or a completed color-noise fix.

## TRAIN decision

Deltas use native packed f16 serving predictions and canonical signed SROCC.
Single rows average the three paired seeds; Mean3 rows compare the native
three-member ensemble against the corresponding current ensemble. Human means
weight four sources equally; noise means weight the registered KADID/TID
subsets equally. Full per-source values and all individual noise ladders are
retained in the two TRAIN_SELECTION.json files.

| Composition | Human mean delta | Noise delta | Guard outcome |
|---|---:|---:|---|
| Single coverage | +0.0009339 | -0.0051962 | noise |
| Single chroma | +0.0012040 | -0.0001354 | noise, kadid |
| Single both | +0.0011296 | +0.0020535 | cid22_a25 |
| Mean3 coverage | -0.0004469 | -0.0028158 | noise |
| Mean3 chroma | +0.0018544 | +0.0012988 | PASS (exploratory) |
| Mean3 both | +0.0017270 | +0.0034808 | cid22_a25 |

All twelve registered fits complete the original120-epoch budget and E33
packing. All three current controls reproduce the original f32 checkpoint
and f16 serving payload after native removal of reproduction metadata;
initialization, sample seeds, training weights and development curves match.
All four native ensembles exactly match independent member-score arithmetic
on2387 TRAIN-development rows each. No seed, weight or checkpoint search.

## The first LIVE severity inversion remains

For manfishing, severe wn/img41 has human quality−0.002387841445669281;
moderate wn/img73 has0.30972455693091216. Higher native scores mean better
quality. These exposed diagnostic examples never enter fitting or selection.

| Composition | Severe img41 | Moderate img73 |
|---|---:|---:|
| Frozen C | 49.71428716500554 | −13.914564690799029 |
| Coverage seed0 | 0.7228257889898053 | −31.283144514017266 |
| Fine-chroma seed0 | 62.609424336127546 | −5.9527465690130885 |
| Both seed0 | 0.25654446166895184 | −17.82591684087877 |
| Current Mean3 | 30.384362821126274 | −12.807286043517673 |
| Coverage Mean3 | 11.981959592851439 | −23.866178294287057 |
| Fine-chroma Mean3 | 50.01161719646959 | −9.350291956462492 |
| Both Mean3 | 31.451349702193628 | −12.690442076446336 |

Every row still reverses this pair. Restoring fine X/B inputs and adding
coverage do not suffice to remove this measured failure. The earlier
[signed matrix trace](gauntlet_live_color_trace_2026-10-18.md) remains relevant;
a monotone output calibration cannot repair an already inverted ordering.

## Reproduction and limits

The [prospective registration](color_noise_repair_registration_2026-10-18.md)
records recipes, source roles, seed/budget rules, guards and the separate
ensemble supplement. Artifacts, checked on dev:
`/mnt/v/output/zensim/color-noise-repair-2026-10-18/`.
Both choices were frozen before external scoring. All five assessment panels
were previously exposed; LIVE motivated the investigation. Protected sets
remain untouched. Original ten-seed, spatial, dial, runtime and product gates
are still required before promotion. The retained external extractor source
commit remains unresolved; the twelve-pair504-feature f32/f64 replay is
bounded arithmetic evidence. Runtime cost of the new compositions is unmeasured.

The original production prediction wire prints17 decimal places; the ensemble
CLI retains round-trip float64 TSVs. The compatibility adapter records the
original fixed17 protocol and requires exact original control strings on all
five panels. The initial failed transport and its native outputs remain
archived; no error tolerance, model, fitting or selection gate was changed.
