# JND Phase A: the declared side-output map (2026-10-11)

Phase A of [the JND dial plan](../docs/JND_DIAL_PLAN_2026-10-11.md), decided by the owner on 2026-10-11, run under
the registered protocol [`jndcal_protocol_2026-10-11.md`](jndcal_protocol_2026-10-11.md). The 0–100 dial is
unchanged. A profile may carry a declared map from its served score to JND, served by `ZensimProfile::jnd_map`
(`zensim/src/jnd.rs`). No AIC data of any kind is used here.

## Result in one paragraph

Profile B gets a map, fitted on B's served score over the KonFiG-IQA F-scale TRAIN sources:
- the map is `J(s) = 0.012653 · max(0, 100 − s)^1.3174`, with the knee fixed at 100;
- its out-of-fold RMSE is 1.427 F-scale JND (source-clustered percentile 95 % [1.074, 1.583], Student [0.828, 2.026],
  3 sources), and τ̂ by maximum likelihood is 1.323.

The registered join-point rule (protocol §7) finds **no supported bin** for B. Its lowest evaluable bin, (0.5, 1],
has an out-of-fold RMSE of 1.47 JND, above the 0.75 bar. So the map supports no range above identity:
- `in_range` is false and the output reads "beyond JND range" for every served score below 100;
- 100 and above map to 0 JND.

This is the registered rule's answer on these data. It is not a defect in the map or the API.

## Which served models get a map now, and why

**B only.** The reasons:
1. **The map belongs to a served score.** Protocol §2 says "One score is mapped in both phases: the frozen final
   candidate's served dial value". `ZensimProfile` owns the served scores. B is `codec_target()`, the shipped
   default and the dial every consumer reads.
2. **Candidate A (E33, `7cf4cfd9…`) and seed 0 (`f803b74c…`) are not served profiles.** They are external bakes run
   through `BakeScorer`, and no `ZensimProfile` variant carries them. QUAL-A also found A not release-qualified. A
   map for them has nowhere to attach and no served score to describe. Their raw-score KonFiG fits stay report-only
   in protocol §10, where J* is 1.5 for both. If either becomes a served profile, its map is fitted the same way on
   its served score.
3. **`BHdr` and `CHdr` are out of scope.** They score absolute-luminance input, while KonFiG is SDR at native pixels.
   The calibration's viewing conditions do not cover HDR.
4. **C and D are candidate profiles, not the dial.** The plan maps the dial's score. Fitting maps for candidates would
   declare a side output nobody serves as the target.

## The score: B's served value

- **Scorer:** the `zensim_score` example, which serves `Zensim::new(ZensimProfile::codec_target())`, built at zensim
  `8f388bda` (binary sha256 `9864ca5f756f5fff5d3ef82d833e3e2c9fef6bfe91c215fde53632f105442995`).
- **Driver:** `benchmarks/jndcal_2026-10-11/score_konfig_train.py --served B_served=<zensim_score>`, the new
  `--served` mode.
- **Output:** `/mnt/v/output/zensim/jnd-phase-a-2026-10-11/konfig_train_served_b_scores.tsv` (dev), 273 stimuli,
  sha256 `8b0c375d…`.
- **Served equals raw:** over all 273 stimuli, the served scores equal the raw bake scores of the protocol test
  (`score_pair_with_bake --bake-post raw`, `konfig_train_scores.tsv` `243e573d…`) within 5e-7, which is the raw
  file's print precision. B's disposition (no mapping, extrapolate, no head) is the raw path.
- **Same fit:** every number the driver prints for the served-score fit equals the raw-score B fit in protocol §10.

## Calibration data and viewing

- **Data:** KonFiG-IQA F-scale, TRAIN sources SRC06, SRC28 and SRC50 only, filtered at read time from
  `/mnt/v/dataset/konfig-iqa/derived/konfig_fscale_trainval_2026-10.parquet` (sha256 `9eb809fb…`).
- **Rows:** 273 stimuli. The 21 scale anchors (level 0, pixel-identical, JND = 0, SE = 0) are excluded, leaving 252
  rows in the headline fit.
- **Viewing:** native pixels, no display model, no resampling, SDR.
- **Unit:** **F-scale JND**, from flicker-boosted triplets. It is not the AIC unit (protocol §10 caveats).

## Fit

`jnd_calibrate` from zenmetrics PR #68 head `1a49beb6`, binary sha256 `a4469e6f…`, which reproduces the public
KonFiG revision-2 outputs byte for byte:

```sh
jnd_calibrate --input konfig_train_served_b_scores.tsv --model B_served --resamples 2000 --seed 20261011
jnd_calibrate --input konfig_train_served_b_scores.tsv --model B_served --range-max 4 --resamples 2000 --seed 20261011
```

- **Method:** leave-one-source-out; a source-clustered bootstrap with B = 2000 and every map refitted inside each
  draw; percentile and Student intervals.
- **Runtime:** 6 s and 5 s under `run-heavy`, `/usr/bin/time -v` max RSS 3,264 KB and 3,336 KB.
- **Outputs:** `/mnt/v/output/zensim/jnd-phase-a-2026-10-11/` (dev), with `SHA256SUMS` (sha256 `80c0a3af…`).

**No label cut (headline).**

| family | role | n | OOF RMSE (JND) | percentile 95 % | Student 95 % | τ̂ ML | τ̂ moment | rmse* | full-data a, b, c |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| KneePowerFixed, b = 100 | **served map** | 252 | 1.427 | [1.074, 1.583] | [0.828, 2.026] | 1.323 | 1.236 | 0.964 | 0.01265, 100, 1.317 |
| KneePower, b fitted | diagnostic | 252 | 1.421 | [1.030, 1.583] | [0.718, 2.124] | 1.318 | 1.229 | 0.960 | 2.544e-09, 217.5, 4.008 |
| Logistic4 | diagnostic | 252 | 1.388 | [0.960, 1.553] | [0.635, 2.140] | 1.267 | 1.191 | 0.913 | 4.614, 1.421, 32.36, -4.516 |
| Linear (first-order) | diagnostic | 252 | 1.438 | [1.076, 1.583] | [0.877, 2.000] | 1.334 | 1.249 | 0.967 | 5.138, -0.05564 |
| Isotonic | diagnostic | 252 | 1.402 | [0.862, 1.511] | [0.596, 2.209] | 1.296 | 1.208 | 0.946 | — |

- **Full-precision coefficients:** a = 0.012653289786345816, c = 1.3174113811942716. They are pinned bit for bit by
  `jnd::tests::profile_b_coefficients_match_the_recorded_fit`.
- **The family's share of the error is small.** The out-of-fold isotonic map, the best monotone shape, is 1.402. The
  first-order map is 1.438.

**Join-point support, served map, no cut** (stimuli/sources, out-of-fold RMSE; ✓ supported, ✗ evaluable but over the
0.75 bar, · not evaluable):

| (0, 0.5] | (0.5, 1] | (1, 1.5] | (1.5, 2] | (2, 2.5] | (2.5, 3] | (3, 3.5] | (3.5, 4] | J* |
|---|---|---|---|---|---|---|---|---:|
| 10/2 0.305 · | 32/3 1.474 ✗ | 18/3 1.772 · | 17/3 0.893 · | 15/3 1.146 · | 24/3 1.115 ✗ | 26/3 1.210 ✗ | 40/3 1.907 ✗ | none |

**0 < JND ≤ 4, selected on the label** (comparability view only; it is not the served map): primary fixed-knee RMSE
0.788 [0.694, 0.850], a = 0.02603, c = 1.084. The support row there is also J* none, because its (0.5, 1] bin is at
1.000.

## What the runtime serves

`ZensimProfile::B.jnd_map()` returns `JndMap { a, c, support: None }`.
- **`estimate(s)`:**
  - for s ≥ 100: `InRange { jnd: 0, uncertainty: 0 }`;
  - for any other finite s: `BeyondRange`, printed as "beyond JND range";
  - for non-finite s: `BeyondRange`.
- **Declared uncertainty, for a map that has a supported range:** the out-of-fold RMSE pooled over the calibration
  stimuli whose out-of-fold prediction lies in the supported bins:
  sqrt(Σ nᵢ·rmseᵢ² / Σ nᵢ) over the driver's support rows with upper edge ≤ J*.
  - This is fixed now, before any map with a supported range exists.
  - B has no supported range, so it has no in-range uncertainty.
- **`bake_verdict` caller:** it reports the map at dial 90, 80 and 70, per codec, at the grid setting whose median
  dial is nearest each point, giving J at that setting's p75, median and p25 dial. It attaches the map only when the
  scored surface is profile B itself:
  - one member whose bytes hash to B's bake;
  - no corruption head;
  - the dial panel scored that member, not a peer's stored scores.

  On B's bake today every cell reads "beyond JND range".

## Caveats

- **Unit:** F-scale JND, not AIC JND (boosted versus plain triplets).
- **Content:** KonFiG's distortions are mostly not compression (colour diffusion, sharpening, jitter, lens and motion
  blur, multiplicative noise, JPEG 2000). B was fitted for codec targeting, and its manifest lists no KonFiG leg.
- **Sources:** with 3 sources the intervals are descriptive. The Student half-width is 0.60 JND.
- **SE provenance:** the reconstruction SEs are an observed-Fisher delta-method add-on in zensim's port (protocol §10).

## Reproduce

```sh
cargo build --release -p zensim --example zensim_score
uv run --with pyarrow python3 benchmarks/jndcal_2026-10-11/score_konfig_train.py \
    --out konfig_train_served_b_scores.tsv --served B_served=target/release/examples/zensim_score
# jnd_calibrate from imazen/zenmetrics 1a49beb6 (crates/zenmetrics-stats, example jnd_calibrate):
jnd_calibrate --input konfig_train_served_b_scores.tsv --model B_served --resamples 2000 --seed 20261011
jnd_calibrate --input konfig_train_served_b_scores.tsv --model B_served --range-max 4 --resamples 2000 --seed 20261011
```
