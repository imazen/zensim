# V0_20d — JND-anchored output calibration (design + initial findings)

**Status**: design doc + initial empirical analysis, 2026-05-14 eve.
Task #41 / #45 (V0_20d). Goal: anchor V_X output range to two
independent JND references (KonJND-1k + AIC-3) and surface where
they disagree.

## Motivation

zensim is a **user-facing quality dial** — users type a target
zensim score and the codec picks an encode hitting it. For that to
work the score range must be *psychophysically grounded*: the
boundary between "perceptible artifact" and "below threshold" must
land at a documented, defensible value.

Per CLAUDE.md goal #3: "Anchor at perceptibility thresholds.
KonJND-1k (mean PJND scored against ssim2 ≈ 63 per CID22 paper
Table 4) is the anchor. A trained model must score at-PJND pairs ≈
63 ± 5; if it saturates to 100 there, 'visually lossless'
calibration is broken."

V0_18 ship is calibrated via `affine_calibrate` against KonJND-1k
+ truth-distribution percentile match. Anchor target: **63**.

JPEG AIC-3 (Mohammadi 2025) provides a *second* JND anchor with
different methodology (PTC test, JND-unit subjective scale,
high-fidelity codec corpus). This doc establishes whether the two
anchors agree.

## AIC-3 JND landscape (300 stimuli, anchor CSV)

The Anchor CSV reports per-stimulus subjective distortion in **JND
units** (1.0 = first noticeable difference). Distribution across
the 300 anchor stimuli:

| Cut | n | % |
|---|---:|---:|
| distortion < 0.5 (sub-threshold)  |  74 | 24.7 % |
| distortion < 1.0 (at-or-below JND) | 115 | 38.3 % |
| distortion < 1.5                  | 167 | 55.7 % |
| distortion < 2.0                  | 227 | 75.7 % |

Range: 0.000..3.784 JND. Median 1.374 JND — i.e. AIC-3 is
intentionally weighted into the sub-2-JND high-fidelity regime
where compression decisions matter most.

### Metric scores AT the AIC-3 JND threshold (distortion ∈ [0.9, 1.1], n=21)

| Metric | median | p25 | p75 |
|---|---:|---:|---:|
| SSIMULACRA2  | **76.71** | 74.95 | 78.71 |
| IW-SSIM      | 0.9935 | 0.9925 | 0.9950 |
| MS-SSIM      | 0.9943 | 0.9936 | 0.9959 |
| PSNR-Y       | 40.66 dB | 40.13 | 41.10 |
| CVVDP        | 9.68 | 9.66 | 9.71 |

## V_18 ship empirical landing (this session)

Ran V_18 ship on 250 AIC-3 Anchor stimuli (`/tmp/per_pair_v0_18_aic3_2026-05-14.csv`),
overall SROCC vs JND-distortion subjective = **0.9149**.

V_18 ship median output by AIC-3 JND band:

| AIC-3 band | JND range | n | V_18 median | V_18 p25/p75 | SSIMULACRA2 median |
|---|---|---:|---:|---|---:|
| sub-JND     | [0.0, 0.5] | 47 | **96.60** | 90.33 / 100.77 | 86.53 |
| AT-JND      | [0.9, 1.1] | 17 | **73.92** | 71.95 / 77.58  | 76.70 |
| 1.5–2 JND   | [1.5, 2.0] | 57 | 60.20 | 55.60 / 64.21 | 66.30 |
| visible     | [2.5, 4.0] | 37 | 51.32 | 44.47 / 55.59 | 60.97 |

**V_18 ship outputs 73.92 at AIC-3 1-JND** — only **2.78 below
SSIMULACRA2's 76.70 at the same band**. V_18 is **well-calibrated
to the AIC-3 JND anchor** without any explicit AIC-3 supervision.
That's an honest cross-corpus validation of the KonJND-fit
calibration.

But: V_18's 73.92 ≠ the KonJND-anchor target of 63 either. The
AIC-3 1-JND lands ~11 V_18 score units ABOVE the KonJND PJND
target.

## The discrepancy with KonJND-1k

| Anchor | JND-score (V_18 / ssim2 proxy) | Source |
|---|---:|---|
| KonJND-1k PJND mean | **63** (target) | CID22 paper Table 4 |
| V_18 at KonJND PJND | ~63 (calibrated to) | `affine_calibrate` ship |
| AIC-3 1-JND median  | **76.71** (ssim2) / **73.92** (V_18) | this analysis |

**Δ = 13.71 score units.** These are not measurement noise — they
reflect two different operational definitions of "just noticeable":

1. **KonJND-1k**: PJND = "probability that 50 % of observers detect
   the artifact in 2AFC." Per-pair value; mean across 1,008 sources.
   Corpus: JPEG + BPG, mixed quality range. Subjects: crowd-sourced,
   non-experts.

2. **AIC-3 PTC**: JND = "1 JND above reference, calibrated via
   triplet comparison ladder." Anchor stimuli intentionally span
   sub-1-JND to ~3.8-JND. Corpus: 5 codecs (AVIF, HEIC, JXL,
   JPEG XS, VVC). Subjects: trained per-codec, controlled lab
   conditions.

Both are valid; they measure different things. **Neither is wrong;
the gap is real and informative.**

## V0_20d shipping decision

Two non-mutually-exclusive options:

### Option A: keep KonJND anchor, ADD AIC-3 as sanity check

- Continue calibrating against KonJND-1k PJND (zensim 63 at PJND mean).
- Run an **AIC-3 sanity check** in CI: V0_18 ship output at AIC-3
  JND threshold should land in some target window (e.g. 70..82).
- Document both anchors in the README / methodology so users know
  what "zensim 63" and "zensim 77" mean in JND terms.
- **Cost**: ~2 hours (sanity script + README update). No code change.
- **Risk**: low.

### Option B: re-calibrate to a midpoint anchor

- New anchor: zensim 70 at the GEOMETRIC MEAN of (KonJND PJND
  median, AIC-3 1-JND median) on ssim2 = √(63 · 76.71) ≈ **69.5**.
- Refit affine α/β to land V0_18's raw output at 69.5 when input
  pairs have ssim2 in [62..78] (the range where both anchors agree
  "near JND").
- Update KonJND validation gate: at-PJND zensim ≈ 70 ± 5 (was ± 5
  around 63).
- **Cost**: ~4 hours (recalibration + 10-band re-validation + ship
  swap + doc).
- **Risk**: medium — every downstream picker / RD-curve tool that
  hard-codes "zensim 63 = JND" needs to update to "zensim 70 = JND".

### Recommended: Option A first, B later

A is the cheap win — adds the AIC-3 sanity check without changing
the dial users already know. B is the principled long-term move
but requires a coordinated dial-semantic flip across the entire
stack (imageflow / cavif / zensquoosh all consume zensim scores).

## Concrete next steps (status)

1. ✓ **DONE**: empirical AIC-3 JND analysis — 76.71 vs KonJND's 63
   documented.
2. ✓ **DONE**: per-pair V_18 ship run on 250 AIC-3 Anchor stimuli
   committed at `benchmarks/per_pair_v0_18_aic3_anchor_2026-05-14.csv`.
   Overall SROCC 0.9149.
3. ✓ **DONE**: sanity script
   `scripts/v_next/aic3_jnd_sanity.py` reads per-pair CSV, prints
   per-JND-band V_X median table, asserts AT-1-JND median ∈ target
   ± tolerance. Default target is V_18's measured 73.92 ± 5; use
   `--target-at-jnd 70` if/when Option B (midpoint) ships.
4. ✓ **DONE**: pairs-TSV builder
   `scripts/v_next/aic3_anchor_pairs_tsv.py` re-generates the
   `/tmp/aic3_anchor_pairs.tsv` input from the Anchor CSV +
   data-root for any future bake.
5. **NEXT (queued)**: V0_20d Option B recalibration — needs
   dial-semantic-flip coordination across imageflow / cavif /
   zensquoosh consumers (every site that hard-codes "zensim 63 =
   JND" must update). Sized ~4 hr but blocked on coordination.
6. **PARALLEL**: per-codec breakdown — does V_18 land near 73.92
   at 1-JND for every AIC-3 codec (AVIF / HEIC / JXL / VVC / JPEG
   XS), or only on average? Quick run-on-existing-data analysis,
   ~30 min. Add to sanity script as a sub-mode.

## V_18 ship JND-band landing (from the sanity run)

| JND band  | JND range  | n  | V_18 median | p25   | p75    |
|---|---|---:|---:|---:|---:|
| sub-JND   | [0.0, 0.5] | 47 | **96.60**   | 90.33 | 100.77 |
| near-JND  | [0.5, 0.9] | 24 | 80.42       | 77.63 | 84.46  |
| **AT-1-JND** | [0.9, 1.1] | 17 | **73.92** | 71.95 | 77.58  |
| 1–1.5 JND | [1.1, 1.5] | 37 | 68.88       | 65.36 | 72.08  |
| 1.5–2 JND | [1.5, 2.0] | 57 | 60.20       | 55.60 | 64.21  |
| visible   | [2.0, 4.0] | 68 | 54.23       | 46.97 | 58.48  |

Monotonic decline 96.6 → 54.2 across JND bands. The dial behaves
as users expect — higher V_X = lower perceptual distortion.

## References

- CID22 paper Table 4 (KonJND-1k PJND mean = ssim2 ≈ 63):
  `/mnt/v/zen/zensim-training/2026-05-07/papers/CID22_wg1m99012.pdf`
- Mohammadi 2025 (AIC-3 methodology + Anchor CSV provenance):
  `arXiv:2509.13150`, repo `github.com/shimamohammadi/EvaluationMetrics`
  cloned to `/mnt/v/input/datasets/aic3/EvaluationMetrics`
- Anchor CSV:
  `/mnt/v/input/datasets/aic3/EvaluationMetrics/Anchor_assessment_on_PTC_full_resolution_Aug_3_2025.csv`
- Path eval ranking (V0_20d at #4):
  `docs/v0_20_path_evaluation_2026-05-14.md`
