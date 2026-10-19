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

## Frozen external assessment

Every row uses the same five original gauntlet populations. The four fixed
seed0 arms and four fixed ensembles were specified before these reads; these
results do not revise either TRAIN decision. This is previously exposed TEST
assessment, not independent confirmation. KonJND is JPEG SELECT, not terminal.

| Model | CSIQ865 | PIPAL23200 | CID22-B2100 | JPEG JND404 | LIVE267 |
|---|---:|---:|---:|---:|---:|
| Frozen C | 0.953279 | 0.607488 | 0.909689 | 0.511471 | 0.907877 |
| Frozen W-C | 0.961758 | 0.617801 | 0.904991 | 0.589398 | 0.889548 |
| Frozen A | 0.944465 | 0.613416 | 0.905267 | 0.521301 | 0.932257 |
| Matched C seed0 | 0.953279 | 0.607488 | 0.909689 | 0.511471 | 0.907877 |
| Coverage seed0 | 0.954923 | 0.606528 | 0.905948 | 0.525611 | 0.940478 |
| Fine chroma seed0 | 0.956753 | 0.621236 | 0.906305 | 0.543026 | 0.877034 |
| Both seed0 | 0.945456 | 0.609574 | 0.906312 | 0.544550 | 0.944441 |
| CVVDP-FHD | 0.919116 | 0.588347 | 0.837048 | 0.427112 | 0.948960 |
| SSIMULACRA2 | 0.904710 | 0.529307 | 0.913071 | 0.554131 | 0.960526 |
| Butteraugli3 | 0.814693 | 0.497251 | 0.781063 | 0.128297 | 0.908635 |
| Current Mean3 | 0.953827 | 0.607317 | 0.907892 | 0.534560 | 0.912888 |
| Coverage Mean3 | 0.956288 | 0.608293 | 0.907282 | 0.545997 | 0.930138 |
| Fine chroma Mean3 (TRAIN choice) | 0.959121 | 0.613893 | 0.905715 | 0.539797 | 0.905241 |
| Both Mean3 | 0.952145 | 0.606724 | 0.905488 | 0.554351 | 0.923390 |

Coverage seed0 improves LIVE while giving up some PIPAL/CID22 rank. Fine
chroma seed0 improves PIPAL while reducing LIVE rank. The TRAIN-selected
fine-chroma ensemble improves CSIQ/PIPAL against the matched current ensemble
but reduces LIVE rank. None removes the first severity inversion. No default
profile or release recommendation is promoted by this experiment.

The HTML overview shows all14rows ×5panels and all10original severe-noise
image pairs, with native before/after scores and every actual severity for
each reference. Two canonical dashboards contain10rows ×5panels each; their
shared retained comparators are identical. All original/new fulleval JSONs
retain every population point, source-cluster SROCC intervals, fitted PLCC,
KROCC and scatter/tail diagnostics. PLCC/KROCC intervals remain unmeasured.

Validation: unchanged browser, URL-comparison and strict-JSON gates pass;
each dashboard attaches40MOS and10JND charts. HTTP200/full-GET SHA256 matches
cover the overview, both dashboards, all12packed models,4composition files
and20original images. Recomputed C SROCC/CI/PLCC/KROCC exactly reproduce the
original retained statistics on every panel. Native full assessment statistics
receipt: `run-heavy: done rc=0 604s | peak-RSS 0.18GiB`; browser gates4s,
peak-RSS0.18GiB. These are execution/resource receipts, not metric-speed claims.

Auxiliary source closure: the entire retained ordinal-pool union contains8794
reference PNGs, all decoded as RGB by an independent Pillow read. None matches
any admitted external reference RGB hash, and none matches any of the six
admitted D1 human banks. This conservatively includes unused families. Current
file/pixel hashes and every compared row are recorded in AUXILIARY_OVERLAP.json
and AUXILIARY_D1_OVERLAP.json. Crop/resize lineage and historical raw-source
immutability remain outside this exact-current-source check. Audit receipt:
`run-heavy: done rc=0 33s | peak-RSS 0.24GiB`; D1 metadata comparison1s,
peak-RSS0.14GiB. All27focused policy/provenance/archive checks pass.

[Verified evidence locations and hashes](color_noise_repair_2026-10-18.pointer.md).
