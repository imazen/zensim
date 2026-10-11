# JNDCAL: score → JND calibration protocol (registration, 2026-10-11, revision 2)

**Status.**
- This doc registers the protocol and records an interim test on KonFiG TRAIN sources. **No map is adopted.**
- The calibration that can adopt a map runs **once**, when AIC2026's human labels are available to us and admitted
  (`docs/DATA_SPLITS.md` §3d).
- The doc implements Phase A of [the JND dial plan](../docs/JND_DIAL_PLAN_2026-10-11.md) and fixes the method for
  its Phase B.
- Revision 2 settles every choice the adopting run needs before any AIC2026 label is read: the map family and
  knee, the stimuli, the score, the reconstruction, the join point and the adoption gates. §12 lists what changed
  from revision 1.

Statistics come only from `zenmetrics-stats` (`jnd`, `resample`), through its `examples/jnd_calibrate.rs` driver.
Nothing statistical is computed in zensim scripts.

## 1. Data roles

| Set | Role in this protocol | Why |
|---|---|---|
| KonFiG-IQA F-scale, **TRAIN sources** SRC06, SRC28, SRC50 (origin split) | Interim calibration and protocol test | Same Thurstone construction and ÷0.6745 as the AIC unit; TRAIN data may be used for calibration. |
| KonFiG VAL / TEST sources | Not read by this protocol | The scoring script filters TRAIN sources when it reads the table, so VAL rows are never loaded. VAL stays for later validation; TEST was never parsed. |
| **AIC2026**, labelled sources | The one adopting calibration run, once, when its labels are available to us and admitted | Owner decision 2026-10-11; its images differ from the AIC-3 family, so there is no holdout contamination. `aic2026` stays in zensim's protected-population guards until that admission. |
| AIC-3 CTC, AIC-4 sample, SDR25 | Holdout: one registered zero-shot read of the frozen final map (§9) | The same ten images; never fitted (`jpeg-aic-family-holdout-2026-09-01`). |
| KonJND | Separate check, never fitted or merged with the JND anchor | PJND is a per-observer threshold distribution from a different protocol. |

## 2. Models, build and the score being mapped

**The score.** One score is mapped in both phases: **the frozen final candidate's served dial value `s`**. That is
the 0–100 value the zensim API returns for the candidate's served profile, composition included.
- The dial plan builds `D(s)` from the current dial `B(s)`. With `s` the served dial, `B(s) = s`, so the join slope
  is `k = 10 · |J′(s*)|`.
- The Phase A side-output map is fitted on that same score. So is the Phase A KonFiG map used in the Phase B
  zero-shot comparison, which makes that comparison like for like.
- **Gate before the label read:** the frozen build must ship a pair scorer that emits the served dial value. Today's
  `score_pair_with_bake` emits only the bake's model score (`raw`, `clamp` or `mapped`).

**The protocol test in §10** maps the raw model score (`score_pair_with_bake --bake-post raw`: the `BakeScorer`
model score, unclipped, so negative values survive as the dial contract allows; no corruption head, no
composition) of three models:

| Name | Model (sha256) | Formula revision |
|---|---|---|
| A (E33 candidate) | `7cf4cfd92bb8824405db3ed27287bbec973a01eaeadf09b7ad037db92ab9a275` | 5 |
| seed 0 (production) | `f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400` | 5 |
| B (served `codec_target()`) | `a96a5a66cd16282ded84580c37207b403334f1af069c924d9883346d23941276` | process default |

- **Scorer:** zensim main `c107f88f`, binary sha256 `74f3a937…`. `ZENSIM_FORMULA_REV=5` is set for the Rev5 models;
  the tool refuses a revision mismatch.
- **In the adopting run** these three raw-score fits are reported beside the primary fit, as report-only.
- **Every run** records each model hash and the build.

## 3. Map family (declared before any fit)

- **Primary, both phases: `KneePowerFixed(Shape::Quality, Knee(100))`, J(s) = a·max(0, 100 − s)^c**, with a ≥ 0 and
  c > 0 free, and the knee fixed at 100. This follows the dial plan. Reasons:
  - A score of 100 is pixel identity, which is 0 JND by definition. A fixed knee puts identity at exactly 0 and every
    score short of it above 0.
  - A fitted knee breaks that. On KonFiG TRAIN it lands at 81.5 for A, 80.1 for seed 0 and 126 for B (§10), so
    every score from about 81 to 100 would map to 0 JND, or identity would sit at about 0.3 JND. Under
    `D(s) = 100 − 10·J(s)`, the first would send all those scores to dial 100, breaking the identity, tie and
    near-identity contracts.
  - It is the JPEG AIC common-test-condition family with one parameter pinned by definition, so it stays comparable
    with published peer maps.
  - It is monotone by construction.
- **Report-only:**
  - `KneePower` with the knee fitted; its knee shows how far the data's own zero sits from identity;
  - `Logistic4`, `Linear` and `Isotonic` (the best monotone shape, out of fold), which show how much of the error
    is the family's.

No family or knee choice remains open after labels.

## 4. Stimuli, rows and viewing

**KonFiG test (§10).**
- All rows of the TRAIN sources, except scale anchors.
- **Anchors** are rows with JND = 0 and SE = 0. Every level-0 image is pixel-identical to its reference (all 63
  score exactly 100). It is 0 JND by definition, and the fixed knee already encodes it.
- **The headline uses every remaining row, with no cut on the label.**
- The `0 < JND ≤ 4` view is reported beside it, for comparability with AIC2026's designed span (approximately
  0.2–4.0 JND, per its public readme). That view selects on the label, so its error near the top of the range is
  biased low.

**The adopting AIC2026 run.**
- **Rows:** every (source, codec configuration, level) that carries a label in the primary reconstruction (§6):
  - all 17 configurations, both the 5 base codecs and the 2 extended codecs per source (490 source–codec
    assignments, each nested in its source);
  - every labelled level, including levels that only the BTC experiment presented (its readme lists BTC levels 01,
    03, 05, 08, 12, 16 and 20 and PTC study levels 03, 05, 08, 10, 13, 16 and 20);
  - no cut on the label.
- **The score is always computed on the plain image:**
  - the 840×944 `PTC_S{NN}_{CODEC}_{LL}.png` crop of that level, against `PTC_S{NN}_Ref_00.png`. The readme provides
    PTC crops for all levels;
  - BTC images are boosted renderings (amplified artifacts) and are never scored;
  - full-resolution images are not used, because the labels were collected on the crops.
- **Folds:** the 70 sources.

**Viewing conditions.** Native pixels, with no display model and no resampling. KonFiG images and AIC2026 crops are
scored at the resolution they were shown.

## 5. Fit and error measures

- **Fit:** out of fold, leaving one source out (`jnd::out_of_fold`).
- **Error measures** on the pooled out-of-fold predictions:
  - out-of-fold RMSE in JND (headline);
  - **τ̂ by maximum likelihood** (headline), with σ = the reconstruction's per-stimulus SE. The moment τ̂ is
    report-only; §10 says why;
  - `rmse_star` with each stimulus' own 95 % interval, d = 0.

## 6. Intervals and reconstructions

**Intervals.** A source-clustered bootstrap (`Design::OneWay(source)`), B = 2000, seed 20261011.
- The maps are **refitted inside every draw**, with copy-aware folds from `Draw::expand`.
- The percentile interval is always reported with the Student interval beside it (`resample::Interval::student`:
  estimate ± t(G−1) · se · √(G/(G−1))).
- With KonFiG's 3 TRAIN sources both are descriptive only.

**Reconstructions.** KonFiG has one: a per-sequence pointwise Thurstone MLE, F condition. For the adopting run:
- **Primary reconstruction:** the release's own reconstruction that combines its PTC and BTC responses, if it
  provides one (a curve-based joint reconstruction of the kind in arXiv:2509.13150 §III-D). Otherwise, the release's
  own per-stimulus pointwise reconstruction (Testolina et al. 2024, arXiv:2410.09501).
- **Every other reconstruction in the release** is report-only and gets every headline and gate.
- **A flip** is either of:
  - a gate in §9 that passes under the primary reconstruction and fails under another, or the reverse;
  - J* (§7) that differs between them by more than one bin (0.5 JND).
- **A flip means "not decidable by these data":** no adoption. The owner decides what follows, and the decision is
  recorded.

## 7. Join point (J*): the registered bar

The rule runs on the primary map's out-of-fold predictions from the no-cut run. It is implemented in
`jnd_calibrate`, which prints the `support` object.

1. **Bins** are 0.5 JND wide, covering (0, 0.5], (0.5, 1], … up to 4 JND. Binning uses the out-of-fold
   **prediction**, never the label, so the held-out label never selects its own rows.
2. **A bin is evaluable** when it holds at least **20 stimuli** from at least **min(10, G) sources**. That is 10 of
   AIC2026's 70, or all 3 of KonFiG's.
3. **A bin is supported** when it is evaluable and its out-of-fold RMSE is at most **0.75 JND**, which is 7.5 dial
   points, less than one JND step on the new dial.
4. **Bins below the lowest evaluable bin are skipped**, because the fixed knee anchors the zero there. From the
   lowest evaluable bin upward, **J\*** is the upper edge of the last bin in the unbroken run of supported bins.
5. **J\* is "none"** when the lowest evaluable bin is not supported or no bin is evaluable. It never exceeds 4.

Phase A's `in_range` flag holds up to the J* of the side-output map's KonFiG fit.

## 8. What can adopt a map

- **Only the run on AIC2026's human labels, once, after this protocol is frozen.** It adopts only if every gate in
  §9 passes under the primary reconstruction with no flip (§6), and the owner signs it off.
- **Everything else is report-only:** this KonFiG test, any re-run of it, and any read of AIC2026 data that is not
  the admitted labelled release. No interim result changes a served value, a gate or a dial.

## 9. Adoption gates (all must pass)

1. **Join point.** J* is not none, J* ≥ 1.0, and the bin (0.5, 1] is supported, so the dial-90 point lies inside
   the supported range.
2. **Dial structure** (dial plan step 5), checked numerically:
   - `D` is monotone in `s` over the scored range and at 1,000 evenly spaced points;
   - `D(100) = 100`;
   - at `s*`, the value and the slope on the two sides agree to within 1e-9 and 1e-6 relative.
3. **Holdout read.** One registered zero-shot read of the AIC-3 / AIC-4 sample / SDR25 family for the frozen final
   map:
   - RMSE in JND, τ̂, and agreement by pair class, with source-clustered percentile and Student intervals;
   - **gate:** the holdout RMSE exceeds the AIC2026 out-of-fold RMSE by no more than 0.5 JND.
4. **Dial 90 ≈ 1 JND per codec** on the holdout:
   - for each codec with at least 10 held-out stimuli whose new dial value lies in [87.5, 92.5], the mean held-out
     label must lie in [0.5, 1.5] JND;
   - its source-clustered percentile and Student intervals are reported beside it;
   - codecs with fewer stimuli in the window are listed as not testable;
   - the gate needs at least 3 testable codecs, and every testable codec must pass.
5. **Existing dial gates** (range, ties, near-identity, negative tails, G-STEER, targeting tails) pass at the new
   scale, or the owner accepts each regression on the record.
6. **KonFiG ↔ AIC2026 consistency** (dial plan step 2). Both zero-shot errors are reported. If the KonFiG map's
   zero-shot RMSE on AIC2026 exceeds the AIC2026 out-of-fold RMSE by more than 0.5 JND, that is a recorded finding,
   and the owner resolves it before the dial switches. Some gap is expected, because KonFiG's F scale is boosted.

## 10. KonFiG TRAIN test (interim, report-only)

Inputs:
- `/mnt/v/output/zensim/jndcal-2026-10-11/konfig_train_scores.tsv`: 273 stimuli × 3 models, sha256 `243e573d…`.
  - It is produced by `benchmarks/jndcal_2026-10-11/score_konfig_train.py` from F-scale table sha256 `9eb809fb…`.
  - Re-scored with the read-time TRAIN filter, the file is byte-identical.
- **Revision 2 fits:** `/mnt/v/output/zensim/jndcal-2026-10-11/r2-fixed-knee/konfig_train_fit_{A,seed0,B}_{full,le4}.jsonl`,
  with SHA256SUMS beside them.
  - Driver: zenmetrics `e4a8c5a5`, in imazen/zenmetrics PR #68 (head `1a49beb6`, to merge into `master` as a fast-forward so the hashes stay), `jnd_calibrate` binary sha256 `a4469e6f…`.
  - The six runs take 31 s on dev in total (`/usr/bin/time -v` max RSS 3,900 KB).
- **Revision 1's fits** (fitted-knee primary, driver `204f5062`) stay in the parent directory as a record.

Each bin cell below reads stimuli/sources, out-of-fold RMSE, then a mark: ✓ supported, ✗ evaluable but over the
bar, · not evaluable.

#### Full range, no label cut (headline)

| model | family | n | OOF RMSE (JND) | percentile 95 % | Student 95 % | τ̂ ML | τ̂ moment | rmse* | full-data parameters a, b, c |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| A | KneePowerFixed, b = 100 (primary) | 252 | 1.030 | [0.604, 1.441] | [-0.139, 2.199] | 0.839 | 0.744 | 0.511 | 0.002657, 100, 1.646 |
| A | KneePower, b fitted | 252 | 1.040 | [0.603, 1.436] | [-0.101, 2.181] | 0.848 | 0.757 | 0.513 | 0.01197, 87.31, 1.359 |
| A | Logistic4 | 252 | 1.077 | [0.602, 1.425] | [-0.054, 2.208] | 0.880 | 0.808 | 0.543 | 241.8, -5.53, -375.8, -121.4 |
| A | Linear | 252 | 1.040 | [0.700, 1.436] | [-0.001, 2.080] | 0.859 | 0.757 | 0.517 | 5.071, -0.06632 |
| A | Isotonic | 252 | 1.078 | [0.511, 1.342] | [-0.064, 2.220] | 0.870 | 0.808 | 0.542 | — |
| seed0 | KneePowerFixed, b = 100 (primary) | 252 | 1.056 | [0.588, 1.365] | [-0.017, 2.130] | 0.920 | 0.779 | 0.592 | 0.00347, 100, 1.573 |
| seed0 | KneePower, b fitted | 252 | 1.063 | [0.587, 1.365] | [0.014, 2.112] | 0.925 | 0.788 | 0.598 | 0.006596, 94.44, 1.452 |
| seed0 | Logistic4 | 252 | 1.063 | [0.582, 1.364] | [0.030, 2.095] | 0.924 | 0.788 | 0.596 | 428.6, -4.401, -444.4, -116.2 |
| seed0 | Linear | 252 | 1.037 | [0.715, 1.371] | [0.094, 1.981] | 0.891 | 0.753 | 0.560 | 4.782, -0.06071 |
| seed0 | Isotonic | 252 | 1.060 | [0.479, 1.260] | [0.002, 2.118] | 0.883 | 0.784 | 0.566 | — |
| B | KneePowerFixed, b = 100 (primary) | 252 | 1.427 | [1.074, 1.583] | [0.828, 2.026] | 1.323 | 1.236 | 0.964 | 0.01265, 100, 1.317 |
| B | KneePower, b fitted | 252 | 1.421 | [1.030, 1.583] | [0.718, 2.124] | 1.318 | 1.229 | 0.960 | 2.544e-09, 217.5, 4.008 |
| B | Logistic4 | 252 | 1.388 | [0.960, 1.553] | [0.635, 2.140] | 1.267 | 1.191 | 0.913 | 4.614, 1.421, 32.36, -4.516 |
| B | Linear | 252 | 1.438 | [1.076, 1.583] | [0.877, 2.000] | 1.334 | 1.249 | 0.967 | 5.138, -0.05564 |
| B | Isotonic | 252 | 1.402 | [0.862, 1.511] | [0.596, 2.209] | 1.296 | 1.208 | 0.946 | — |

#### Join-point support, primary map, full range

| model | (0, 0.5] | (0.5, 1] | (1, 1.5] | (1.5, 2] | (2, 2.5] | (2.5, 3] | (3, 3.5] | (3.5, 4] | J* |
|---|---|---|---|---|---|---|---|---|---:|
| A | 11/3 0.223 · | 19/3 0.527 · | 23/3 0.748 ✓ | 24/3 0.881 ✗ | 28/3 0.761 ✗ | 27/3 1.286 ✗ | 29/3 1.264 ✗ | 28/3 1.120 ✗ | 1.5 |
| seed0 | 14/3 0.227 · | 20/3 0.528 ✓ | 22/3 0.718 ✓ | 28/3 0.832 ✗ | 28/3 1.190 ✗ | 23/3 0.836 ✗ | 23/3 1.141 ✗ | 23/3 1.005 ✗ | 1.5 |
| B | 10/2 0.305 · | 32/3 1.474 ✗ | 18/3 1.772 · | 17/3 0.893 · | 15/3 1.146 · | 24/3 1.115 ✗ | 26/3 1.210 ✗ | 40/3 1.907 ✗ | none |

#### 0 < JND ≤ 4, selected on the label (comparability view)

| model | family | n | OOF RMSE (JND) | percentile 95 % | Student 95 % | τ̂ ML | τ̂ moment | rmse* | full-data parameters a, b, c |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| A | KneePowerFixed, b = 100 (primary) | 184 | 0.631 | [0.445, 0.745] | [0.215, 1.047] | 0.545 | 0.000 | 0.301 | 0.004991, 100, 1.46 |
| A | KneePower, b fitted | 184 | 0.626 | [0.444, 0.732] | [0.151, 1.102] | 0.532 | 0.000 | 0.297 | 0.05069, 81.53, 0.9911 |
| A | Logistic4 | 184 | 0.633 | [0.445, 0.732] | [0.142, 1.124] | 0.540 | 0.000 | 0.307 | 4.952, -1.069, 41.95, -28.88 |
| A | Linear | 184 | 0.614 | [0.470, 0.760] | [0.184, 1.044] | 0.529 | 0.000 | 0.292 | 3.952, -0.04771 |
| A | Isotonic | 184 | 0.631 | [0.367, 0.675] | [0.132, 1.131] | 0.536 | 0.000 | 0.296 | — |
| seed0 | KneePowerFixed, b = 100 (primary) | 184 | 0.661 | [0.405, 0.724] | [0.148, 1.173] | 0.581 | 0.000 | 0.322 | 0.008707, 100, 1.324 |
| seed0 | KneePower, b fitted | 184 | 0.646 | [0.401, 0.758] | [0.082, 1.211] | 0.555 | 0.000 | 0.312 | 0.08212, 80.13, 0.8712 |
| seed0 | Logistic4 | 184 | 0.645 | [0.393, 0.760] | [0.053, 1.237] | 0.556 | 0.000 | 0.322 | 4.278, -0.7275, 45.6, -24.65 |
| seed0 | Linear | 184 | 0.637 | [0.423, 0.732] | [0.138, 1.135] | 0.556 | 0.000 | 0.308 | 3.787, -0.04413 |
| seed0 | Isotonic | 184 | 0.648 | [0.340, 0.736] | [0.017, 1.279] | 0.562 | 0.000 | 0.331 | — |
| B | KneePowerFixed, b = 100 (primary) | 184 | 0.788 | [0.694, 0.850] | [0.593, 0.983] | 0.754 | 0.329 | 0.508 | 0.02603, 100, 1.084 |
| B | KneePower, b fitted | 184 | 0.792 | [0.675, 0.850] | [0.586, 0.998] | 0.757 | 0.339 | 0.509 | 0.001309, 126.4, 1.66 |
| B | Logistic4 | 184 | 0.794 | [0.639, 0.849] | [0.520, 1.068] | 0.754 | 0.344 | 0.508 | 3.308, 0.9459, 39.77, -8.143 |
| B | Linear | 184 | 0.787 | [0.702, 0.850] | [0.591, 0.983] | 0.756 | 0.328 | 0.505 | 3.753, -0.03831 |
| B | Isotonic | 184 | 0.777 | [0.568, 0.812] | [0.458, 1.097] | 0.740 | 0.303 | 0.499 | — |

**What the test shows:**
- **Error level, no label cut.**
  - The primary map's out-of-fold error is 1.03 JND for A (τ̂ 0.84), 1.06 for seed 0 (τ̂ 0.92) and 1.43 for B
    (τ̂ 1.32).
  - On the label-selected 0 < JND ≤ 4 view it is 0.63, 0.66 and 0.79.
  - The full range runs to 8.2 F-scale JND, beyond where the unit is measurable, and errors grow there.
- **Fixing the knee costs little.** Against the fitted knee, out-of-fold RMSE moves by −0.010 (A), −0.007 (seed 0)
  and +0.006 (B) with no label cut, and by +0.005, +0.015 and −0.004 on the label-selected view. The five families
  sit within about 0.05 JND of each other.
- **Join point.** J* is 1.5 F-scale JND for A and seed 0. For B it is none: B's lowest evaluable bin, (0.5, 1], has
  an out-of-fold RMSE of 1.47.
- **The moment τ̂ is 0 for A and seed 0 on the label-selected view** (B's is 0.33).
  - The reconstruction SEs are strongly heteroscedastic: 25 of 184 stimuli there have SE > 1 JND, constant within
    some sequences.
  - So the mean σ² (0.51) exceeds A's and seed 0's mean squared out-of-fold error, and the moment estimator floors
    at 0.
  - The ML τ̂ weights stimuli by 1/(σ² + τ²) and stays informative, which is why ML is the headline.

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
- **Sources.** With 3 sources the intervals are descriptive; the Student half-widths run from 0.2 JND (B, label-selected)
  to 1.2 JND (A, no cut).
- **The score.** These are raw model scores (§2), not the served dial of a frozen candidate. The Phase A side-output
  map is refitted on that served score once the candidate is frozen.

## 11. Reproduce

```sh
cargo build --release -p zensim-validate --bin score_pair_with_bake          # zensim c107f88f
uv run --with pyarrow python3 benchmarks/jndcal_2026-10-11/score_konfig_train.py --bin <score_pair_with_bake> \
    --out konfig_train_scores.tsv --model B=default:<b.bin> --model seed0=5:<seed0.bin> --model A=5:<A.bin>
cargo run --release -p zenmetrics-stats --example jnd_calibrate -- \
    --input konfig_train_scores.tsv --model A --resamples 2000 --seed 20261011              # zenmetrics e4a8c5a5
cargo run --release -p zenmetrics-stats --example jnd_calibrate -- \
    --input konfig_train_scores.tsv --model A --range-max 4 --resamples 2000 --seed 20261011
```

## 12. Changes from revision 1 (same day, before any AIC2026 label)

- **The knee is fixed at 100** for both phases, using `KneePowerFixed`, which was added to zenmetrics-stats and
  tested against scipy. The fitted knee is report-only.
- **New registrations:**
  - the join-point rule and its bar (§7);
  - the adoption gates with numbers (§9);
  - the primary reconstruction and what a flip means (§6);
  - the adopting run's stimuli and the score being mapped (§2, §4).
- **The join-point rule was revised once, after its first KonFiG run.**
  - As first written, it required every bin from 0 to be supported.
  - KonFiG's sparse (0, 0.5] bin (11 stimuli) then gave "none" for every model, a veto from the band the fixed knee
    already anchors.
  - The revision skips bins below the lowest evaluable bin. The bar itself (20 stimuli, min(10, G) sources,
    0.75 JND) is unchanged.
- **The KonFiG headline is now the no-cut run.** Selecting on the label (`0 < JND ≤ 4`) biases the top of the range,
  so that view is kept only for comparability.
- **Smaller changes:**
  - the Student interval now comes from the crate;
  - the KonFiG scorer filters TRAIN sources when it reads the table;
  - the reconstruction types now cite arXiv:2410.09501 and arXiv:2509.13150;
  - the viewing conditions are stated.
