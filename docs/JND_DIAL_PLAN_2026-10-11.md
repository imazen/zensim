# JND and the zensim dial: plan (owner decision 2026-10-11)

**Decision.**
- **Now, Phase A:** the 0–100 dial is unchanged, and zensim adds a declared
  **JND side output**.
- **Phase B, when AIC2026's human labels arrive:** the dial itself becomes JND
  near the top at **10 dial points per JND** (`dial = 100 − 10 · JND`), joined
  **smoothly** to the lower range. No seam.

Terms: JND is the JPEG AIC unit. 0 is the source image; 1 JND is a distortion an
observer picks out in 75 % of forced-choice comparisons. Scale values come from
Thurstone Case V divided by Φ⁻¹(0.75) (Testolina et al. 2024, arXiv:2410.09501).
Above about 3–4 JND the unit is not measurable: observers answer correctly
almost every time, so comparisons carry no information.

Related records:
- `DATA_SPLITS.md` §3d: AIC2026 becomes the calibration set once its labels are
  published.
- `CODEC_TARGET_GOALS.md` G2 and G8.
- Statistics live in zenmetrics-stats (`zenmetrics/docs/EVALUATION_METHODS.md`).

## Phase A — JND side output (start now)

**Output.** For a score `s`, report three things:
- the JND estimate `J(s)`;
- its uncertainty;
- an `in_range` flag that is false above the fit's supported maximum. Report
  "beyond JND range" there, never a number.

The dial value is unchanged. Adding the output is a zensim API change and goes
through the public-API process in `CLAUDE.md`: preregister the caller and the
exact delta.

**Map family, declared before fitting.**
- Family: knee power, `J(s) = a · max(0, b − s)^c` with `a ≥ 0`, `c > 0`
  (zenmetrics-stats `jnd::MapFamily::KneePower(Shape::Quality)`).
- The knee `b` is **fixed at 100**: a score of 100 is pixel identity, which is
  0 JND by definition. zenmetrics-stats has the fixed-knee variant (`039eb247`),
  so `b` is not fitted.
- Diagnostics, not the served map: the out-of-fold isotonic map and a
  first-order map, so the family's share of the error is visible.

**Calibration data.**
- **KonFiG-IQA F-scale, TRAIN sources only**
  (`/mnt/v/dataset/konfig-iqa/derived/konfig_fscale_trainval_2026-10.parquet`,
  restricted to the TRAIN origins). It uses the same Thurstone construction and
  ÷0.6745 as the AIC scale.
- Map error comes from leave-one-source-out refits inside KonFiG TRAIN:
  out-of-fold RMSE in JND and τ̂, with source-clustered intervals.
- No AIC-family data is used in Phase A.

**Range.** `in_range` holds up to the largest JND the TRAIN data supports
(KonFiG spans about 3 JND).

**Viewing conditions.** Native pixels, no display model. Every report states
the viewing setup, because JND depends on it.

**Done when:**
- the fitted map and its out-of-fold error are recorded under `benchmarks/`;
- the API addition is reviewed and merged;
- `bake_verdict` reports `J` at the targeting-relevant dial points (90, 80, 70
  in today's units) per codec;
- nothing about the dial, its gates or its consumers has changed.

## Phase B — the dial in JND, smoothly (when AIC2026 labels arrive)

**Trigger.**
1. AIC2026's human labels are published.
2. Their admission is registered as `DATA_SPLITS.md` §3d requires. `aic2026`
   leaves the protected-population guards only through that admission.

**Steps, in order:**

1. **Register the calibration protocol before the first look at any label.** It
   fixes the map family (as in Phase A, knee at 100), the folds (leave one
   AIC2026 source out), the join point rule (step 3) and the acceptance gates
   (step 5).
2. **Fit `J(s)` on AIC2026's labelled sources**, with leave-one-source-out error.
   Report it beside the Phase A KonFiG map: the KonFiG map's zero-shot error on
   AIC2026, and the reverse. A large disagreement is a finding to resolve before
   switching the dial.
3. **Build the new dial `D(s)` from the current dial `B(s)`:**
   - **JND part:** `D(s) = 100 − 10 · J(s)` while `J(s) ≤ J*`.
   - **Join point:** `J*` is the largest JND at which the AIC2026 fit is
     supported, meaning its out-of-fold error and data density meet the
     registered bar. Expect `J*` ≈ 3, which puts the join near dial 70.
     `s*` is the score where `J(s*) = J*`.
   - **Lower part (`J(s) > J*`):** an affine continuation of today's dial,
     `D(s) = D(s*) + k · (B(s) − B(s*))`. It keeps the current dial's ordering
     and relative spacing below the join.
   - **Smoothness:** `k` is set so the slope matches at `s*` (continuous value
     and first derivative: no kink, no step), `k = 10 · |J′(s*)| / B′(s*)`.
     `k > 0` keeps `D` monotone.
   - **Contract unchanged:** 100 = identity, negative scores stay valid, each
     codec's lowest distinct outputs stay representable.
4. **Version it.** The new dial is a new profile identity. B stays available
   unchanged for comparison during the transition.
5. **Acceptance gates.** All must pass:
   - monotone in `s`, identity = 100, continuous with matching slope at the join
     (checked numerically);
   - **one registered read of the AIC-3 / AIC-4 sample / SDR25 holdout family**,
     zero-shot, for the frozen final map: out-of-fold-style RMSE in JND, τ̂,
     agreement by pair class with source-clustered intervals, and the Student
     interval with few sources;
   - per codec, the score that lands at dial 90 is close to 1 JND on the held-out
     data. Consistency across codecs is the point of the change;
   - the existing dial gates (range, ties, near-identity, negative tails,
     G-STEER, targeting tails) pass at the new scale, or each regression is
     recorded and accepted by the owner.
6. **Migrate**, in one change set:
   - rescore stored data under the new profile;
   - refit the starting-quality predictors;
   - move picker targets;
   - update `CODEC_TARGET_METRIC.md` (dial contract: "90 = 1 JND, 70 ≈ 3 JND,
     10 points per JND above the join"), the `CHANGELOG` and the docs that quote
     dial values.

   No consumer is calibrated to B, C or D (owner ruling 2026-09-07), so there is
   no external migration.

**Later: the lower range from merged human data.** The owner's direction is to
map quality below the JND range by merging all available human data. That needs
a joint scale with a monotone link per dataset, overlap between datasets, and
the Squintly study to create it. When that scale exists, it replaces the affine
continuation below `J*` under the same rule: continuous value and slope at the
join, monotone, contract unchanged. It is a further versioned profile.

## What does not change

- Data roles: AIC-3, the AIC-4 sample and SDR25 are holdout only, never fitted.
  KonJND PJND remains a separate check from a different protocol, never merged
  with the JND anchor.
- The qualification track (candidate A, E34, open owner decisions) is
  independent of this plan.
