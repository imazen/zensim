# Robust native pairs: fewer conflicts, unresolved quality and map tradeoffs

September 14, 2026. **No model is promoted.** Basic156 fails the registered human-rank noninferiority bar. Y60 passes the numeric advancement screen, but its human residual tail worsens and native map guidance remains weak. These are TRAIN results, not frozen EVAL or shipping qualification.

## Registered comparison

Reuse the preceding native-local packet without new encodes, extraction, calibration, or source admission: eight TRAIN fitting families and four disjoint TRAIN development families. No EVAL, TEST, terminal, or historical protected segment is accessed. Development covers photo, document, graphic and screen, one family each at 256-long-edge and JXL distances 1/3. This is a small development screen, not broad independent generalization.

Each native training bucket contains the two endpoints of an explicit baseline/probe preference. `allpairs10` admits 500 distinct oriented pixel pairs with nonzero SSIM2 difference; `robust10` retains 214 pairs whose SSIM2 and Butteraugli directions agree beyond the registered .1/.005 margins. The resulting tables contain 1,000 and 428 rows. Both arms retain actual SSIM2 targets and canonical full944 Rev3 features. Pair bucket identifiers never redefine family splits.

Both arms use the same corrected group weights and fixed attempted-draw budget, giving about 45% human, 45% codec and 10% native accepted pairs. Native same-row rejection is exactly .5 in both designs. Group counts and skipped counts match exactly between arms for each sampling seed. Changing pair selection also changes scaler populations: this compares complete recipes, not an isolated loss term. The earlier multi-row native10 recipe has a different effective-update budget and is contextual evidence only.

Plain y60/H32 and basic156/H128 are fitted with the existing Rust trainer, 32 epochs of 8,192 attempted pairs, three initialization seeds, no early stopping or auto-eval, and f32 packing. All members and uniform three-member ensembles are scored through public Rust BakeScorer owners. Both SSIM2 and Butteraugli now select training examples: native consensus is mentor consistency, not an independent quality judge.

## Native development evidence

| Complete model | Conflicts /79 | Median map mass/response rank | Cells below .70 /8 | M2 failures /8 |
|---|---:|---:|---:|---:|
| NRS914_y60_allpairs10_ens3 | 0 | 0.5368 | 5 | 0 |
| NRS914_y60_robust10_ens3 | 0 | 0.5647 | 5 | 0 |
| NRS914_basic156_allpairs10_ens3 | 2 | 0.4485 | 7 | 0 |
| NRS914_basic156_robust10_ens3 | 1 | 0.4382 | 6 | 0 |
| D_frozen_revision1 | 0 | 0.6382 | 5 | 0 |

All measured models have complete additive density here. The .70 map line is a mechanism diagnostic, not native allocation RD qualification. The remaining basic156 conflicts are on screen content; filtering introduces no newly failing class. On fitting families its conflicts change 2→3/214; on the prior three-family diagnostic packet, 4→3/160. Y60 retains zero conflicts on both packets. Four-family development is too small to support a universal ordering claim.

## Human/codec panels and tails

Human development contains 1,000 rows from eight admitted KADID TRAIN references. Codec development contains 1,629 rows including identities, with separate codec, source, distorted and ladder panels retained.

| Model | Human SROCC | Codec SROCC | Human geometric out4 | Human raw residual p99 | Human raw coverage | Codec geometric out4 |
|---|---:|---:|---:|---:|---:|---:|
| NRS914_y60_allpairs10_ens3 | 0.83954 | 0.92603 | 0.10% | 61.81 | 0.85 | 8.23% |
| NRS914_y60_robust10_ens3 | 0.83645 | 0.92497 | 0.10% | 66.08 | 0.80 | 7.31% |
| NRS914_basic156_allpairs10_ens3 | 0.87274 | 0.93751 | 1.60% | 67.34 | 0.80 | 9.09% |
| NRS914_basic156_robust10_ens3 | 0.87182 | 0.93857 | 1.40% | 68.27 | 0.70 | 8.84% |

Basic156 robust human rank is .011979 below the frozen untreated control, beyond the registered .005 tolerance. Its matched-control difference is only −.000918, and native conflicts improve, but those do not override the original-control requirement. Human out4 improves 1.6%→1.4% against the matched recipe while remaining worse than the original control’s .4%; raw coverage falls .80→.70.

Y60 meets the registered rank/conflict/M2 conditions. Its robust ensemble human rank falls .003088 against matched allpairs10, although all three individual members improve (+.002946, +.002352, +.000357). Rank of the served ensemble is not the average of member ranks. Human residual p99 worsens 61.81→66.08, maximum residual 115.85→135.39, raw coverage .85→.80 and clumping .117→.127. Codec geometric out4 improves 8.23%→7.31%. This mixed outcome does not justify promotion; no new numeric tail gate is retroactively invented.

All 12 primary members and four ensembles receive the existing canonical Rust full panel/scatter assessment: 21,584 panels, with 19,680 undefined sparse OR/z-RMSE fields explicitly null and indexed. Full-population raw and geometric diagnostics include clumping, saturation, ranges and tails. No full release composite or independent native RD benefit is claimed.

## Sampling findings and replay correction

The first 12 fits used sampling seeds 17101/17103/17107. The legacy initializer is `s*gamma+C`, while SplitMix advances by that same gamma per raw word. Thus seeds separated by four start one four-word within-reference attempt apart. A regression using the real pair-draw owner verifies 262,143 consecutive attempted pairs match after this shift. Different digests had established different sequences, not independent sampling replicas. Historical scores and initialization variation remain valid measurements; the sampling-replication interpretation must be narrowed.

A registered amendment retained those initial fits and ran 12 corrected primary fits with sample seeds derived from the first eight big-endian SHA256 bytes of `zensim/native-robust/2026-09-14/sample/<initializer-seed>`. Exact seeds and commands are retained. An opt-in `subset_sim --require-disjoint-sampler-windows` preflight rejects overlapping conservative four-word-per-attempt windows, including wraparound, before reading tables. It refuses unsupported stratified schedules. It does not change historical RNG arithmetic or promise independence of auxiliary trainer randomness. Primary native accepted shares are 9.822%–10.014%; all training digests match canonical replay.

Separately, `subset_sim --fulleval` ignored the recorded sampling override and used the legacy seed. It now prefers structured/argv `sample_seed`, falls back to legacy `seed`, and refuses contradictory metadata. Explicit `--seeds` still takes precedence. An actual admitted TRAIN sidecar replay matches explicit-seed output byte-for-byte and reproduces the training digest. Explicit historical replay output is unchanged byte-for-byte. No protected fulleval corpus is opened.

## Reproduction and next decision

All 24 fits, amendments, exact model bytes, commands, scores, failed hypotheses and tool hashes are retained under `~/work/zensim-validation-2026-09-14/native-robust-train/`. The primary 12-fit campaign took about 195 seconds; this is not a controlled serving benchmark. Raw/packed public scores agree for every member on human, codec and native development. All 812 matched native decoded-pixel/mentor rows agree with the frozen packet; 1,056 complete ensemble pixel/cache comparisons agree. Dedicated gallery results exactly reproduce their parent cases.

The [development gallery](/zensim/reports/native-robust-train-2026-09-14/index.html) includes five models and two exact A/B failures, with [compact results](native_robust_train_2026-09-14.results.json) binding the evidence. The original packet and its source/encoder/decoder provenance remain available in the [preceding report](native_local_train_2026-09-14.md).

Do not expand this into an unconstrained pair-weight sweep. Before further fitting, reconcile the existing constrained best-of-all and two-reference training work with these scalar-tail and native-response failures, and register the smallest mechanistic comparison. Any advancement still requires native allocation, witnessed 1/2/3-shot targeting, broader frozen human gates, corruption composition, HDR and qualified runtime/memory evidence. The complete goal remains active.
