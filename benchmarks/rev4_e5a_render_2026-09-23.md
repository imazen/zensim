# Rev4 E5A — does a testing-specialised metric beat general metrics on rendering regressions? (2026-09-23)

Preregistered: `benchmarks/e5a-render_prereg_2026-09-23.md`. Lane: `quarantine/devin/e5a-render`
(nothing pushed). Scripts: `scripts/e5a_pipeline.sh`, `scripts/e5a_pairs.py`,
`scripts/e5a_analyze.py`, `scripts/e5a_record.py`. Raw output: `/var/tmp/e5a-render/`
(score tables + 993-item `pairs.tsv` + diffmaps, hashed in
`benchmarks/rev4_e5a_render_2026-09-23.json`). Bootstrap: origin-clustered, B = 2000,
seed 20260923, paired across arms. Severity SROCC via `panel --batch --stats srocc`.

**Question:** at a false-alarm rate fixed by benign implementation drift, does a
testing-specialised error-map metric (`testlin`, best of 5 linear/encoded error-map
candidates selected on TRAIN families only) beat the best general-purpose metric at
detecting rendering-code regressions?

## Decision (prereg §7)

**NO-SHIP.** The frozen selection rule picked `t4_enc_max` — the max of the u8
encoded error map — which *is* the general `maxabs` arm (row-identical on all
993 scored pairs, verified). Detection on held-out TEST families is therefore
identical (Δ = 0.000, 95 % CI [0.000, 0.000], upper bound < 0.10 → NO-SHIP by
the preregistered rule). **The Δ = 0 result is structural**: the preregistered
testlin candidate set itself contained a general arm, so once T4 won the TRAIN
selection, Δ ≡ 0 regardless of the data. What the lane establishes is narrower:
no testing-specialised candidate *beat* maxabs on TRAIN, and under the
preregistered decision rule nothing justifies shipping a separate testing
metric. It does **not** establish that maxabs outranks the perceptual arms —
see "Benign pool composition" and "Sensitivity" below.

| arm | TEST detect @1 % FA | 95 % CI | TEST detect @0.1 % FA | benign realized FA @t99 |
|---|---|---|---|---|
| **testlin = t4_enc_max** | **0.836** | [0.823, 0.854] | 0.836 | 0.000 |
| maxabs | **0.836** | [0.823, 0.854] | 0.836 | 0.000 |
| butter_p3 | 0.830 | [0.807, 0.855] | 0.818 | 0.012 |
| gmsd | 0.824 | [0.802, 0.846] | 0.794 | 0.012 |
| r915_fast | 0.824 | [0.802, 0.845] | 0.800 | 0.012 |
| dssim | 0.800 | [0.792, 0.811] | 0.788 | 0.012 |
| zensim_b | 0.800 | [0.784, 0.817] | 0.794 | 0.012 |
| psnr | 0.794 | [0.773, 0.817] | 0.782 | 0.012 |
| ssim2 | 0.788 | [0.768, 0.809] | 0.739 | 0.012 |
| butter_max | 0.782 | [0.742, 0.821] | 0.745 | 0.012 |
| zensim_d | 0.770 | [0.742, 0.800] | 0.758 | 0.012 |
| r915_rich | 0.691 | [0.649, 0.732] | 0.673 | 0.012 |

n = 165 non-inert TEST pairs (5 families × 12 origins), 352 TRAIN pairs. Detection =
fraction flagged over every scored item (NaN scores count as unflagged). Realized FA
is pooled over 342 non-inert benign pairs; thresholds are pooled-benign 99th/99.9th
percentiles, so realized FA ≤ nominal by construction — `maxabs`/`testlin` sit at
0.000 because benign `maxabs` is bimodal (318 items = 1, 24 items = 9) and the 99th
percentile lands on the top of the distribution.
**Bootstrap CIs are approximate**: an independent recompute with the same
resampling scheme (B = 2000, seed 20260923) reproduces every point estimate but
shifts CI bounds by ≤ 0.009 (RNG consumption order differs).

## Benign pool composition (scored, non-inert; threshold basis)

| benign family | n scored | maxabs (constant) |
|---|---|---|
| route_u8_vs_u8f32 | 72 | 1 |
| resize_f32_vs_i16 | 48 | 1 |
| resize_u16_vs_f32_lin | 48 | 1 |
| route_u8_vs_u8u16_lin | 48 | 1 |
| quantize_dithered_vs_plain | 46 | 1 |
| composite_f32_vs_u16 | 36 | 1 |
| **dither_phase** | **24** | **9** |
| srgb_lut_vs_poly | 20 | 1 |
| (inert, 0 scored) | 126 | — |

Every benign family is constant in `maxabs`. **`dither_phase` alone holds the
top 8 benign scores of every arm, so it alone sets every arm's t99
threshold** — the pooled ranking in the decision table is contingent on this
one family being in the benign pool. The brief's "two correct resamplers
differing only in rounding" and "±1 LSB rounding" categories survive only
through `route_*`, `resize_*` and `quantize_dithered_vs_plain`:
`resize_streaming_vs_fullframe` and `quantize_round_half_even_vs_away` came out
100 % inert (0 scored items). `quantize_dithered_vs_plain`, `route_u8_vs_u8f32`
and `route_u8_vs_u8u16_lin` were added in `96941b3a`, after the prereg.

## Sensitivity check (NOT preregistered)

Reviewer-computed, `dither_phase` removed from the benign pool (n = 318),
thresholds recomputed at t99, TEST @1 % FA:

| arm | TEST@1 % |
|---|---|
| zensim_b | 0.964 |
| r915_fast | 0.964 |
| ssim2 | 0.952 |
| butter_p3 | 0.945 |
| r915_rich | 0.945 |
| dssim | 0.939 |
| psnr | 0.933 |
| gmsd | 0.927 |
| maxabs | 0.927 |
| butter_max | 0.921 |
| zensim_d | 0.921 |

The arm ranking reorders completely — the pooled-detection order in the
decision table is **not robust** to benign-pool composition. In this pool the
TRAIN selection would still pick T4 (TRAIN 0.980) and testlin − best general is
negative, so the **NO-SHIP decision is unchanged**.

## Labelling conflict in the bitdepth family

The design conflicts with the brief, which lists "a changed dither" as a
*regression*: benign `dither_phase` (5 bpc Bayer, phase p vs p+1) has maxabs 9,
while corruption `bitdepth/dither_removed` also has maxabs 9 — undetectable by
maxabs by construction. Corruption `bitdepth/trunc_vs_round` has maxabs 1,
identical to the benign ±1-LSB pairs. Benign `quantize_dithered_vs_plain` is
"dither vs no dither" labelled benign while `dither_removed` is the same
direction labelled corruption. **The bitdepth detection rates (0.31–0.49 on
every arm) measure this labelling conflict, not the metrics.**

## testlin selection (TRAIN families only, frozen before TEST)

| candidate | def. | TRAIN detect @t99 | thr99 |
|---|---|---|---|
| t1_lin_max | max linear-light error | 0.778 | 0.0719 |
| t2_lin_q999 | 99.9th-pct linear error | 0.756 | 0.0719 |
| t3_lin_q99 | 99th-pct linear error | 0.608 | 0.0699 |
| **t4_enc_max** | max u8 error = maxabs | **0.824** | 9 |
| t5_enc_q999 | 99.9th-pct u8 error | 0.770 | 9 |

The linear-domain candidates under-perform encoded maxabs on TRAIN, so the selection
collapsed to the trivial arm. `testlin`'s localisation map is the u8 error map.

## Per-family detection @t99 (TEST families in **bold**)

| family | maxabs | psnr | ssim2 | butter_max | butter_p3 | dssim | gmsd | zensim_b | zensim_d | r915_fast | r915_rich |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gamma_downsample (T) | **1.000** | 0.938 | 0.927 | **1.000** | **1.000** | 0.958 | **1.000** | **1.000** | **1.000** | 0.990 | 0.438 |
| alpha_nopremul (T) | **0.604** | 0.250 | 0.257 | 0.500 | 0.500 | 0.271 | 0.493 | 0.382 | 0.410 | 0.271 | 0.035 |
| geometry_shift (T) | **0.886** | 0.841 | 0.841 | 0.909 | 0.864 | 0.841 | 0.864 | 0.841 | 0.841 | 0.841 | 0.841 |
| channel_chroma (T) | **1.000** | 0.800 | 0.750 | 0.700 | 0.650 | 0.600 | 0.550 | 0.800 | 0.600 | 0.750 | 0.500 |
| exif_orientation (T) | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.979 |
| **alpha_premul_state** | **1.000** | 0.944 | 0.986 | 0.986 | **1.000** | 0.986 | 0.986 | 0.986 | 0.986 | 0.986 | 0.917 |
| **gamma_apply** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.917 |
| **wrong_kernel** | **1.000** | **1.000** | 0.875 | 0.583 | 0.875 | **1.000** | **1.000** | **1.000** | **1.000** | 0.917 | 0.542 |
| **primaries_dropped** | **0.700** | 0.300 | 0.000 | 0.600 | 0.600 | 0.100 | 0.000 | 0.000 | 0.000 | **0.700** | 0.000 |
| **bitdepth** | 0.314 | 0.343 | 0.400 | 0.400 | 0.400 | 0.343 | **0.486** | 0.371 | 0.229 | 0.343 | 0.371 |

## Severity ordering (SROCC, score vs per-item linear-mean-error anchor)

Values are **signed** Spearman ρ of raw score vs anchor: positive means score
rises with measured damage, negative means it falls. For higher-is-worse arms
(maxabs, butter_*, dssim, gmsd, testlin) the expected direction is positive;
for lower-is-worse arms (psnr, ssim2, zensim_*, r915_*) it is negative. A sign
opposite to expectation is an ordering failure, not a weak one.

- **maxabs/testlin order *against* severity on gamma_downsample (ρ = −0.125,
  n = 96) and gamma_apply (ρ = −0.053, n = 24)** — the single-pixel max
  saturates early, so heavier variants can score below lighter ones. Both are
  signed *wrong-direction* results for a higher-is-worse arm.
- Strong correct-direction ordering: alpha_nopremul, alpha_premul_state,
  channel_chroma, bitdepth, exif_orientation — most arms |ρ| ≈ 0.7–0.9 in their
  expected direction (e.g. alpha_nopremul butter_p3 +0.835, dssim +0.849,
  zensim_d −0.870, r915_fast −0.868).
- **wrong_kernel orders poorly on every arm** (|ρ| ≤ 0.72, mostly < 0.3,
  e.g. dssim +0.718 vs psnr −0.342): kernel swaps move few pixels far, so the
  severity anchor itself is nearly flat. primaries_dropped n = 10, noisy.
- gmsd loses ordering on gamma_apply (ρ = +0.056) and exif_orientation
  (+0.247) despite being higher-is-worse — it fires on edges rather than
  global tone shifts.

## Localisation (lift = mean map on changed cells / mean on unchanged; coverage =
fraction of changed cells in the map's own top decile)

| map arm | lift range over families | coverage range | note |
|---|---|---|---|
| butteraugli diffmap | 1.70 – 45.2 | 0.16 – 0.60 | consistently best localiser |
| zensim_b diffmap | 1.18 – 8.73 | 0.10 – 0.47 | moderate lift |
| gmsd map | 0.64 – 1.00 | 0.006 – 0.41 | no localisation (gradient map fires everywhere) |
| u8/lin error map | undefined | 0.20 – 1.00 | lift structurally undefined: unchanged-region mean = 0 (n_lift_defined = 0 for every family; reported, not a defect) |

Degenerate items (changed fraction ≈ 0 or ≥ 0.99 — whole-image shifts, exif flips)
are counted per cell (`n_degenerate`) and excluded from lift/coverage means.

## Inert accounting (prereg: dropped and counted)

993 scored pairs total = 525 corruption (8 inert: byte-identical a/b twins, e.g.
`streaming_vs_fullframe` route pairs) + 468 benign (126 inert: bit-exact benign
route pairs, mostly `round_half_even_vs_away`/`streaming` contexts where both
implementations are mathematically identical). Inert items are excluded from
threshold pools, detection denominators, and the bootstrap; counts are reported.

## Reading

- **What is established:** under the preregistered protocol — benign pool as
  built, pooled t99 thresholds, T1–T5 selection on TRAIN only — the
  testing-specialised arm does not beat the best general arm. The preregistered
  candidate set itself contained a general arm (T4 = maxabs), so the NO-SHIP
  verdict is real but its margin is structural, not measured.
- **What is NOT established (withdrawn from earlier drafts):** that maxabs
  "ties or beats" the perceptual arms in any robust sense, and that "learned
  quality metrics add nothing". Every arm's threshold is set by the 24
  `dither_phase` items; under the non-preregistered sensitivity check the
  ranking inverts (zensim_b/r915_fast 0.964 lead; maxabs 0.927). The pooled
  order is a property of the benign pool, not of the metrics.
- **Per-family facts that do reproduce** (reviewer-verified): on
  `primaries_dropped`, maxabs and r915_fast detect 0.700 while
  ssim2/gmsd/zensim_b/zensim_d/r915_rich detect 0.000 — the P3↔sRGB confusion
  is deliberately sub-JND and quality metrics are blind to it by design. On
  `wrong_kernel`, butter_max (0.583) and r915_rich (0.542) trail everything
  else (≥0.875). On `alpha_nopremul`, maxabs leads (0.604) with perceptual arms
  at 0.25–0.50.
- **`bitdepth` rates are not a metric comparison**: they measure the
  labelling conflict documented above — do not read "gmsd 0.486 beats maxabs
  0.314" as perceptual arms winning a family.
- **r915_rich is weak in the preregistered pool** (TRAIN 0.40, TEST 0.69) but
  recovers under the sensitivity check (0.945); both numbers are threshold-set
  by dither_phase, so neither proves much alone.
- **Severity ordering and detection trade off,** and this part stands: the
  arm that detects best under the preregistered pool (maxabs) orders severity
  *wrong-direction* on two families; dssim/butteraugli order correctly where
  the anchor is meaningful. A regression gate wants detection; a triage tool
  wants ordering + maps.
- **Benign-pool caveat:** benign `maxabs` is bimodal (±1 LSB vs ±9), so its
  realized FA is 0, not 1 %. Per-family-variant detail (44 groups) is in
  `results.json` (see `.pointer.md`).

## Provenance

- Lane commits on `quarantine/devin/e5a-render`: prereg `3d28b733`, generator +
  scorer `f5a1afa7`, fix-up `96941b3a`, unit-test fixes `cc335310`, analyzer
  `19fc9db2`/`939556a7`. Generator revision baked into fixture manifests:
  `cc335310`.
- Sources: 12 origins × {256, 512} px renditions from
  `imazen-26-variants/cleanpicker-ladder11@2026-08-23` (hashes in prereg §2).
- R915 bakes: `recovery/calibrated/` copies, hashes verified = prereg §3 pins.
- Scoring: `e5a_render_score` (993 rows, 0 failures), `peer_metric_pairs`
  (993 rows, 0 failures), `zenmetrics batch dssim` (CPU), `score_pairs_tuner`
  (r915_fast + r915_rich parquet).
- `cargo test --features e5a-render --example m3_fixture_gen`: 13/13 pass.
- Input/output file hashes: `benchmarks/rev4_e5a_render_2026-09-23.json`.
