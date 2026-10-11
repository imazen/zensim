# JNDCAL: score → JND calibration protocol (registration, 2026-10-11)

**Status:** registered protocol, plus an interim test on KonFiG TRAIN sources. No map is adopted. The calibration
that can adopt a map runs **once**, when AIC2026's human labels are available to us and admitted
(`docs/DATA_SPLITS.md` §3d). This implements Phase A of [the JND dial plan](../docs/JND_DIAL_PLAN_2026-10-11.md) and
fixes the method for its Phase B.

Statistics come only from `zenmetrics-stats` (`jnd`, `resample`), through its `examples/jnd_calibrate.rs` driver.
Nothing statistical is computed in zensim scripts.

## 1. Data roles

| Set | Role in this protocol | Why |
|---|---|---|
| KonFiG-IQA F-scale, **TRAIN sources** SRC06, SRC28, SRC50 (origin split) | Interim calibration and protocol test | Same Thurstone construction and ÷0.6745 as the AIC unit; TRAIN data may be used for calibration. |
| KonFiG VAL / TEST sources | Not read by this protocol | VAL stays for later validation; TEST was never parsed. |
| **AIC2026**, labelled sources | The one adopting calibration run, once, when its labels are available to us and admitted | Owner decision 2026-10-11; images different from the AIC-3 family, so no holdout contamination. `aic2026` stays in zensim's protected-population guards until that admission. |
| AIC-3 CTC, AIC-4 sample, SDR25 | Holdout: one registered zero-shot read of the frozen final map | Same ten images; never fitted (`jpeg-aic-family-holdout-2026-09-01`). |
| KonJND | Separate check, never fitted or merged with the JND anchor | PJND is a per-observer threshold distribution from a different protocol. |

## 2. Models and build

| Name | Model (sha256) | Formula revision |
|---|---|---|
| A (E33 candidate) | `7cf4cfd92bb8824405db3ed27287bbec973a01eaeadf09b7ad037db92ab9a275` | 5 |
| seed 0 (production) | `f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400` | 5 |
| B (served `codec_target()`) | `a96a5a66cd16282ded84580c37207b403334f1af069c924d9883346d23941276` | process default |

- **Scorer:** zensim at main `c107f88f`, `zensim-validate` `score_pair_with_bake --bake-post raw`. That is the
  `BakeScorer` model score, unclipped, so negative values survive as the dial contract allows. No corruption head
  and no composition step: this calibrates the model score. `ZENSIM_FORMULA_REV=5` is set for the Rev5 models; the
  tool refuses a revision mismatch.
- **The adopting run** calibrates the frozen final candidate's served composition at its frozen build, plus these
  three for reference. It records each model hash and the build.

## 3. Map family (declared before any fit)

- **Primary: `KneePower(Shape::Quality)`, J(s) = a·max(0, b − s)^c**, with a ≥ 0 and c > 0, and the knee b fitted.
  Reasons:
  - It is the family the JPEG AIC common test conditions use to put metrics on the JND scale, so zensim's map is
    comparable with published peer maps.
  - It is monotone by construction.
  - The knee models the visually lossless band above it.
  - Fixing b at 100 (identity) needs a fixed-knee variant in `zenmetrics-stats`. That is deferred; until then b is
    fitted and reported.
- **Report-only:** `Logistic4`, `Linear` and `Isotonic` (the best monotone shape, out of fold). They show how much
  of the error is the family's.

## 4. Rows, fit and error measures

- **Scale anchors are excluded.** These are rows with JND = 0 and SE = 0. In KonFiG the level-0 image of every
  sequence is pixel-identical to its reference (all 63 level-0 stimuli score exactly 100). It is 0 JND by
  definition and carries no calibration information beyond the identity anchor.
- **Primary range: 0 < JND ≤ 4.** That covers the dial plan's join (about 3 JND) and the range AIC2026's levels were
  designed to span (about 0.2–4 JND, per its public release notes). **The full range is report-only.**
- **Fit:** out of fold, leaving one source out (`jnd::out_of_fold`).
- **Error measures** on the pooled out-of-fold predictions:
  - out-of-fold RMSE in JND (headline);
  - **τ̂ by maximum likelihood** (headline), with σ = the reconstruction's per-stimulus SE. The moment τ̂ is
    report-only; §6 says why;
  - `rmse_star` with each stimulus' own 95 % interval, d = 0.

## 5. Intervals and reconstructions

- **Intervals** come from a source-clustered bootstrap (`Design::OneWay(source)`), B = 2000, seed 20261011.
  - The maps are **refitted inside every draw**, with copy-aware folds from `Draw::expand`.
  - The percentile interval is reported with the Student interval estimate ± t(G−1) · se · √(G/(G−1)) beside it,
    whenever there are fewer than about 30 sources.
  - With KonFiG's 3 TRAIN sources both are descriptive only.
- **Reconstructions.** KonFiG has one reconstruction: per-sequence pointwise Thurstone MLE, F condition. If the
  labelled AIC2026 release provides more than one (for example per-stimulus pointwise and a parametric curve),
  every headline is reported under each, and a flip between them means "not decidable by these data".

## 6. KonFiG TRAIN test (interim, report-only)

Inputs:
- `/mnt/v/output/zensim/jndcal-2026-10-11/konfig_train_scores.tsv`: 273 stimuli × 3 models. It is produced by
  `benchmarks/jndcal_2026-10-11/score_konfig_train.py` from F-scale table sha256 `9eb809fb…`.
- One JSON file per model and range beside it (`konfig_train_fit_*.jsonl`); SHA256SUMS in the same directory.
- Driver: zenmetrics `204f5062` (local, not yet pushed), `jnd_calibrate` binary sha256 `52e19992…`.

All six result files reproduce byte for byte from that commit.

#### Primary range (0 < JND ≤ 4)

| model | family | n | OOF RMSE (JND) | percentile 95 % | Student 95 % | τ̂ ML | τ̂ moment | rmse* | full-data parameters |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| A | KneePower (primary) | 184 | 0.626 | [0.444, 0.732] | [0.151, 1.102] | 0.532 | 0.000 | 0.297 | 0.05069, 81.53, 0.9911 |
| A | Logistic4 | 184 | 0.633 | [0.445, 0.732] | [0.142, 1.124] | 0.540 | 0.000 | 0.307 | 4.952, -1.069, 41.95, -28.88 |
| A | Linear | 184 | 0.614 | [0.470, 0.760] | [0.184, 1.044] | 0.529 | 0.000 | 0.292 | 3.952, -0.04771 |
| A | Isotonic | 184 | 0.631 | [0.367, 0.675] | [0.131, 1.131] | 0.536 | 0.000 | 0.296 | — |
| seed0 | KneePower (primary) | 184 | 0.646 | [0.401, 0.758] | [0.082, 1.211] | 0.555 | 0.000 | 0.312 | 0.08212, 80.13, 0.8712 |
| seed0 | Logistic4 | 184 | 0.645 | [0.393, 0.760] | [0.053, 1.237] | 0.556 | 0.000 | 0.322 | 4.278, -0.7275, 45.6, -24.65 |
| seed0 | Linear | 184 | 0.637 | [0.423, 0.732] | [0.138, 1.135] | 0.556 | 0.000 | 0.308 | 3.787, -0.04413 |
| seed0 | Isotonic | 184 | 0.648 | [0.340, 0.736] | [0.017, 1.279] | 0.562 | 0.000 | 0.331 | — |
| B | KneePower (primary) | 184 | 0.792 | [0.675, 0.850] | [0.586, 0.998] | 0.757 | 0.339 | 0.509 | 0.001309, 126.4, 1.66 |
| B | Logistic4 | 184 | 0.794 | [0.639, 0.849] | [0.520, 1.068] | 0.754 | 0.344 | 0.508 | 3.308, 0.9459, 39.77, -8.143 |
| B | Linear | 184 | 0.787 | [0.702, 0.850] | [0.591, 0.983] | 0.756 | 0.328 | 0.505 | 3.753, -0.03831 |
| B | Isotonic | 184 | 0.777 | [0.568, 0.812] | [0.458, 1.097] | 0.740 | 0.303 | 0.499 | — |

#### Full range (report-only)

| model | family | n | OOF RMSE (JND) | percentile 95 % | Student 95 % | τ̂ ML | τ̂ moment | rmse* | full-data parameters |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| A | KneePower (primary) | 252 | 1.040 | [0.603, 1.436] | [-0.101, 2.181] | 0.848 | 0.757 | 0.513 | 0.01197, 87.31, 1.359 |
| A | Logistic4 | 252 | 1.077 | [0.602, 1.425] | [-0.054, 2.208] | 0.880 | 0.808 | 0.543 | 241.8, -5.53, -375.8, -121.4 |
| A | Linear | 252 | 1.040 | [0.700, 1.436] | [-0.001, 2.080] | 0.859 | 0.757 | 0.517 | 5.071, -0.06632 |
| A | Isotonic | 252 | 1.078 | [0.511, 1.342] | [-0.064, 2.220] | 0.870 | 0.808 | 0.542 | — |
| seed0 | KneePower (primary) | 252 | 1.063 | [0.587, 1.365] | [0.014, 2.112] | 0.925 | 0.788 | 0.598 | 0.006596, 94.44, 1.452 |
| seed0 | Logistic4 | 252 | 1.063 | [0.582, 1.364] | [0.030, 2.095] | 0.924 | 0.788 | 0.596 | 428.6, -4.401, -444.4, -116.2 |
| seed0 | Linear | 252 | 1.037 | [0.715, 1.371] | [0.094, 1.981] | 0.891 | 0.753 | 0.560 | 4.782, -0.06071 |
| seed0 | Isotonic | 252 | 1.060 | [0.479, 1.260] | [0.002, 2.118] | 0.883 | 0.784 | 0.566 | — |
| B | KneePower (primary) | 252 | 1.421 | [1.030, 1.583] | [0.718, 2.124] | 1.318 | 1.229 | 0.960 | 2.544e-09, 217.5, 4.008 |
| B | Logistic4 | 252 | 1.388 | [0.960, 1.553] | [0.635, 2.140] | 1.267 | 1.191 | 0.913 | 4.614, 1.421, 32.36, -4.516 |
| B | Linear | 252 | 1.438 | [1.076, 1.583] | [0.877, 2.000] | 1.334 | 1.249 | 0.967 | 5.138, -0.05564 |
| B | Isotonic | 252 | 1.402 | [0.862, 1.511] | [0.596, 2.209] | 1.296 | 1.208 | 0.946 | — |

**What the test shows:**
- **Error level.** On the primary range, candidate A's out-of-fold error is 0.63 JND (τ̂ 0.53) and seed 0's is
  0.65 (τ̂ 0.56). Served B's is 0.79 (τ̂ 0.76).
- **Family choice doesn't matter much here.** The four families are within about 0.02 JND of each other.
- **Range matters.** Over the full range (up to 8.2 F-scale JND) errors roughly double. That is consistent with
  declaring an "in range" limit.
- **The moment τ̂ is 0 for A and seed 0** on the primary range (B's is 0.34). The reconstruction SEs are strongly
  heteroscedastic: 25 of 184 stimuli have SE > 1 JND, constant within some sequences, so the mean σ² (0.51) exceeds
  A's and seed 0's mean squared out-of-fold error (0.39 and 0.42) and the moment estimator floors at 0. The ML τ̂
  weights stimuli by 1/(σ² + τ²) and stays informative, which is why ML is the headline.

**Caveats:**
- **Unit.** The KonFiG F scale comes from flicker-boosted triplets. The AIC JND unit is defined for unboosted
  comparisons, so this interim map is in **F-scale JND**, not AIC JND. KonFiG Experiment I also holds 87,854 plain
  (unboosted) responses; reconstructing that condition is listed as follow-up.
- **Training overlap.** Candidate A was trained on the D1 population, which includes KonFiG TRAIN+VAL, so its
  out-of-fold map error here is optimistic. B's manifest lists no KonFiG leg. Seed 0's training population was not
  checked in this lane.
- **SE provenance.** The SEs are an observed-Fisher delta-method add-on in zensim's reconstruction port, not the
  authors' code. The adopting run uses the dataset's own intervals.
- **Content.** KonFiG's distortions are mostly not compression: colour diffusion, high-boost sharpening, jitter,
  lens blur, motion blur, multiplicative noise, plus JPEG 2000.
- **Sources.** 3 sources: the intervals are descriptive, and the Student interval spans nearly ±0.5 JND.

## 7. What can adopt a map

- **Only the run on AIC2026's human labels, once, after this protocol is frozen**, with:
  - the acceptance gates of the dial plan's Phase B step 5;
  - one registered zero-shot read of the AIC-3 / AIC-4 / SDR25 holdout family.
- **Everything else is report-only**, including this KonFiG test and any re-run of it. No interim result changes a
  served value, a gate or a dial.

## 8. Reproduce

```sh
cargo build --release -p zensim-validate --bin score_pair_with_bake          # zensim c107f88f
python3 benchmarks/jndcal_2026-10-11/score_konfig_train.py --bin <score_pair_with_bake> \
    --out konfig_train_scores.tsv --model B=default:<b.bin> --model seed0=5:<seed0.bin> --model A=5:<A.bin>
cargo run --release -p zenmetrics-stats --example jnd_calibrate -- \
    --input konfig_train_scores.tsv --model A --range-max 4 --resamples 2000 --seed 20261011   # zenmetrics 204f5062
```
