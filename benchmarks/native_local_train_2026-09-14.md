# Native local supervision: a measured tradeoff, not a qualified model

Later September 14 correction: the [explicit-pair study](native_robust_train_2026-09-14.md)
proves that this study's nearby sampling seeds overlap as raw-stream windows.
Its point scores and initialization variation remain valid; distinct digests
do not establish independent sampling replicas. The later study preserves
historical arithmetic and uses recorded disjoint windows for its primary fits.

September 14, 2026. **Neither treatment advances.** Native-local training reduces some scalar preference failures in basic156, but loses human-quality rank and introduces a native linearization failure. Y60 gains rank while introducing a screenshot preference error. Keep the existing release gates and all previous no-ship verdicts.

## Frozen TRAIN experiment

Twelve fits compare plain y60/H32 and basic156/H128, with three matched seeds per recipe. Controls use the established human/codec recipe. Treatments allocate 10% of accepted pairs to native local SSIMULACRA2 ordering, reducing human and global-codec shares equally to45% each. Architectures, epochs, draw count, initializer seeds and optimizer settings are unchanged. This compares complete data recipes: adding a group also changes the scaler population and sample stream, so it does not isolate a loss term. All fits use the existing Rust trainer, f32 packer and public BakeScorer. No new scorer, feature kernel or training implementation was added.

Eight fitting and four development families inherit the existing product TRAIN assignments. No calibration, EVAL, TEST or terminal data is accessed. Selection uses family hashes and existing256-long-edge renditions, excluding the previous three native-replay families. Broad classes cover photo/render, document, graphic and screen. The photo fitting bucket includes render2210; this is not twelve natural photographs. Development origins are1442,5044,7102,8286, one family per broad class. Four development families remain a small generalization screen.

The source-closed September8 native JXL generator is pinned to preserve the encoder intervention experiment; the current sibling checkout was also synchronized. Fixed effort8 Reference, distances1/3, coarse transform unions, quantizer factors.8/1.2, exact neutral repeats and no inner loop yield812 new bitstreams. One screen has15 nonempty unions rather than16; whole-transform coverage is verified, not forced to fit a rectangular-grid assumption. All812 outputs are freshly decoded with the pinned canonical decoder and extracted as actual full944 Rev3 features. Partial planned vectors are not used as full training rows. Deduplication retains516 fit and264 development rows; local grouping is by source/rendition/distance. Labels are algorithmic SSIM2 proxies, not human truth.

## Native development results

The following uses only the four development families: eight cells,272 encodes,79 robust same-buffer SSIM2/Butteraugli consensus interventions. Conflicts require SSIM2 and Butteraugli agreement beyond.1/.005 and an opposite model change greater than.1. The two peers are not human ground truth. Mass/response rank is a native mechanism diagnostic, not matched-RD improvement.

| Model | Peer conflicts /79 | Median mass/response rank | Native M2 failures below.99 |
|---|---:|---:|---:|
| NL914_y60_control_ens3 | 0 | 0.5574 | 0 /8 |
| NL914_y60_native10_ens3 | 1 | 0.5647 | 0 /8 |
| NL914_basic156_control_ens3 | 9 | 0.3426 | 0 /8 |
| NL914_basic156_native10_ens3 | 5 | 0.4397 | 1 /8 |
| D_frozen_revision1 | 0 | 0.6382 | 0 /8 |

All five measured ensembles/baselines have complete additive density here. No model achieves native allocation qualification. Frozen revision1 D is evaluated on exactly the same newly decoded pixels and peers. All control reruns match the original frozen ensembles exactly on1,624 native pixel scores;1,056 new ensemble pixel scores match separately extracted cached development scores.

Basic156 reduces graphic conflicts5→2 and screen conflicts2→1, while photo conflicts remain2. Its fitting-family conflicts stay10/214. On the previous diagnostic packet, conflicts only change15→14/160, with all13 graphic conflicts remaining. Y60 changes0→1/79 on new development (screen), while retaining0/160 on the previous packet. These observations do not establish that more of the same random within-cell sampling will solve the failure.

## Unchanged human/codec development assessment

Human development has1,000 rows from eight KADID TRAIN references. The unchanged codec panel has1,629 rows, including identities; the report retains separate distorted, near-lossless, source, codec and ladder views. These are TRAIN comparisons, not the full human evaluation gauntlet or recovered release composite. The native packet is separate.

| Model | Human SROCC | Codec SROCC, including identities | Human geometric out4 | Codec raw residual p99 |
|---|---:|---:|---:|---:|
| NL914_y60_control_ens3 | 0.83894 | 0.92555 | 0.300% | 50.563 |
| NL914_y60_native10_ens3 | 0.84609 | 0.92626 | 0.200% | 49.737 |
| NL914_basic156_control_ens3 | 0.88380 | 0.93869 | 0.400% | 45.273 |
| NL914_basic156_native10_ens3 | 0.87797 | 0.93928 | 0.900% | 47.293 |

Basic156 human SROCC falls.005830, exceeding the preregistered.005 noninferiority tolerance; paired seed changes are−.013278,−.009336,+.006332. Human geometric out4 increases.4%→.9%, raw coverage.75→.70, and codec raw residual p99 worsens45.27→47.29 despite higher pooled codec rank. Y60 improves human rank for all three seeds, but human raw residual p99 worsens64.04→66.21, raw coverage.85→.75, and codec geometric out4 increases6.94%→7.73%. Rank gains alone therefore do not justify advancement.

The canonical Rust panel/scatter owner produces21,584 panels across12 seeds and4 ensembles, including raw density, geometry, clumping, saturation, per-reference and tail views. Sparse ladders have19,680 undefined statistic fields (9,840 each for OR and z-RMSE); these are explicitly stored as null with paths and status, never fabricated passes. A first JSON serialization attempt failed on those undefined values; its partial output is retained, and the corrected report uses the same Rust results with explicit missing-value encoding. Panel count is not a count of independent content families.

## Reproducibility and limits

The complete fit/pack/cached-parity campaign took195 seconds; individual fits took26–38 seconds. This is not a serving-latency benchmark or a timed end-to-end five-minute guarantee. Sampler replay agrees with every training digest and measures9.964–10.016% native accepted-pair share. Raw and packed scores agree across all three development tables for every seed.

Source8012 was refused after204 encodes because of an ICC profile; subsequent metadata screening also excludes8014. Replacement8384 is selected by the next unused fitting-family hash, without quality inspection, color stripping or split changes. The initial204 bitstreams remain, and their repeated counterparts are checked for exact byte identity. Total new encoder work is1,016 encodes, including that failed attempt. A fixed16-region assumption in the orchestration script failed before decoding and was corrected to honor the existing native owner’s nonempty unions; no Rust geometry or gate was weakened.

Artifacts: `~/work/zensim-validation-2026-09-14/native-local-train/`. Frozen protocol, all script versions, commands, manifests, native bytes, models, complete scores, full panels and failure outputs are retained. The dedicated [development gallery](/zensim/reports/native-local-train-2026-09-14/index.html) shows five matched models and three exact A/B failures; its displayed development calls reproduce their parent packet case results exactly. [Compact results](native_local_train_2026-09-14.results.json) bind the principal evidence and model bytes.

Next investigate TRAIN-only explicit, robust local preference pairs versus indiscriminate within-cell draws using the existing trainer owner, after registering the comparison. Keep feature profiles and human/global-codec supervision fixed and preserve development families. Do not search weights against EVAL or promote this recipe. Native matched-RD allocation, target loops, corruption composition, broader human gates, HDR and qualified runtime/memory evidence remain open. The full goal remains active.

CI-exact Clippy, script lint, formatting and whitespace checks pass. Browser inspection caught blank initial lazy-image rows; the shared native gallery renderer now reserves image dimensions and eagerly loads its small A/B set. The corrected served page was inspected with all three image triples visible. Scientific inputs, scores and plots are unchanged.
