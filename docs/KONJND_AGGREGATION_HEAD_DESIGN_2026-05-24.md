# Konjnd-dense per-source aggregation head — design (2026-05-24)

**Task #4.** Architectural work in the Rust trainer to unblock the
KonJND training budget. Currently `konjnd-dense` can only carry weight
≤ 0.02 because higher weights cause per-pair MSE to push the network
toward per-ref-constant predictions (recovery_phase3b root cause).
Per-source aggregation breaks this by computing the loss on the
*pooled* per-ref prediction instead of the per-pair predictions.

## The structural problem (from `recovery_phase3b_falsification_2026-05-21.md`)

`konjnd-dense.pjnd_target` is per-source constant — all 20 distortion
levels for one ref share the SAME pjnd_target value. The trainer's
per-pair MSE asks the network to predict identical output for 20
different per-pair feature vectors, which is anti-correlated with the
safesyn/kadid/tid signal (where target tracks distortion). Net effect:

- **Within-ref RankNet** → zero gradient (target diffs are zero).
- **Within-ref MSE** → pushes prediction toward per-ref mean,
  conflicting with feature-driven gradient from other corpora.
- **At weight > 0.02** → CID22 SROCC collapses (0.85 → 0.48 in #203
  test) because the network smooths its feature → distortion mapping
  toward per-source constants.

## The fix: pool first, regress second

Compute the loss on the *aggregate* of predictions per ref-group, not
on individual predictions:

```
For each gradient step:
  Sample K refs from konjnd-dense uniformly.
  For each ref r:
    Sample S of r's 20 rows (S configurable, default 5).
    Forward S × MLP passes → S per-pair predictions y_{r,1}..y_{r,S}.
    Aggregate: agg_r = (1/S) · Σ y_{r,i}
  Loss = w · Σ_r (agg_r − pjnd_target_r)²
  Backprop: ∂L/∂y_{r,i} = 2w · (agg_r − pjnd_target_r) · (1/S)
            ↑ same gradient flows to each of the S row forwards
            ↑ each row then backprops through its own feature path
```

Critical properties:

1. **No zero-gradient pathology.** The within-ref residual `(agg − t)`
   is non-zero in general; gradient flows uniformly to all S rows.
2. **Feature-discriminating preserved.** Per-pair forwards still see
   distinct features, so the network learns content-conditional
   predictions. The aggregation only constrains the *mean* across
   distortion levels, not the per-level value.
3. **Training-time only.** The bake's network is unchanged — it's
   still a per-sample MLP. The runtime forward pass is identical.
   No new bake metadata key, no `forward_one_bake` dispatch change.

## Sampling

Two strategies tested in literature; we use option (a):

(a) **Uniform-K-refs-then-S-per-ref**: sample K refs (with replacement
ok), sample S rows from each. Total: K·S forwards per step.
- Pros: simple, easy to reason about gradient scale, K independent
  of trainer's existing pair-step batching.
- Cons: S small (e.g. 5) means the aggregate is a noisy estimate of
  the per-ref mean. The aggregate's variance ≈ σ²(y_{r,*})/S.

(b) **Full-ref-K-per-step**: sample K refs, forward ALL 20 rows of
each. K·20 forwards per step.
- Pros: aggregate is the true per-ref mean (zero noise).
- Cons: 4× the compute for K=1 step, weight tuning more sensitive.

Default S=5 is the trade-off — 5 forwards per ref gives a reasonable
mean estimate, K·5 forwards is similar cost to a normal mini-batch of
size K·5 per step.

## CLI surface

New flags to `zensim_mlp_train`:

```
--konjnd-aggregation-parquet PATH      [REQUIRED if --konjnd-aggregation-weight > 0]
  Path to konjnd-dense parquet with pjnd_target column populated.
  (canonical: canonical-2026-05-21/train/konjnd-dense.parquet)

--konjnd-aggregation-weight FLOAT      [default 0.0 = off]
  Loss weight applied to the per-ref aggregation MSE term.

--konjnd-aggregation-step-p FLOAT      [default 0.30]
  Probability per pair-step that the aggregation step fires
  (alongside the primary RankNet/MSE step).

--konjnd-aggregation-samples-per-ref U [default 5]
  S — number of rows sampled from each picked ref's 20.

--konjnd-aggregation-refs-per-step U   [default 8]
  K — number of refs picked per aggregation step.
```

## Compatibility with existing pjnd_passthrough

Both can run simultaneously — they compute different gradients on the
same konjnd-dense data:

- `pjnd_passthrough` regresses each row's prediction against the
  per-row target (or constant). Tests whether the row alone tracks
  pjnd_target — falsified at V11-D as structural.
- `konjnd_aggregation` regresses the *mean of per-ref rows* against
  the per-ref target. This is the un-tested alternative.

Recommended for Tuner v11 retrain: turn pjnd_passthrough OFF
(weight=0) and konjnd_aggregation ON (weight=0.3). If both are
needed, run pjnd_passthrough at low weight (0.01) for per-row
calibration and aggregation at higher weight (0.3) for per-ref MSE.

## Data structures

New struct in `mlp_train.rs`:

```rust
pub struct KonjndAggregationPool {
    /// Per-ref grouped features. Outer Vec indexed by ref-id (0..1008),
    /// inner Vec is the 20 standardized feature rows for that ref.
    pub per_ref_features: Vec<Vec<Vec<f32>>>,
    /// pjnd_target per ref-id (constant across the 20 rows).
    pub per_ref_pjnd_target: Vec<f32>,
    /// Per-ref training weight (defaults to 1.0; can be overridden).
    pub per_ref_weight: Vec<f64>,
}
```

Loader: read konjnd-dense parquet, group by `ref_basename`, standardize
features using the same `(mean, scale)` as the primary safesyn group
(this is the critical invariant — the aggregation head's predictions
must be on the same scale as the primary feature stream).

## Backprop plumbing

The existing per-sample-α head has the most complex gradient flow.
Per-row aggregation backprop is straightforward:

```
agg = (1/S) · Σ y_i                  (forward, scalar per ref)
res = agg - t                         (residual, scalar per ref)
∂L/∂agg = 2w · res                    (loss gradient)
∂L/∂y_i = ∂L/∂agg · (1/S) = (2w/S) · res
                                      (same scalar for every i in this ref)
```

For each of K refs:
  Compute `dl_dy = (2 · w / S) · (agg_r - t_r)` once.
  For each of S forwards from this ref:
    Apply tanh-output-head Jacobian (`dya_dpre`) per row.
    Call existing `backprop_step_per_sample_alpha_head(..., dl_dy * dya_dpre, ...)`.
  Adam step on the K·S accumulated gradients.

The backprop function signature stays the same — we just compute the
upstream gradient differently before calling it. This is the minimal
intrusion into the existing 8.4k LOC trainer.

## Implementation plan (file-by-file)

### Phase 1 — Data structures & loader (~150 LOC)

1. `mlp_train.rs:~700`: Add `MlpHyperparams::konjnd_aggregation_*`
   fields with defaults.
2. `mlp_train.rs:~3500` (near pjnd_passthrough loader): Add
   `load_konjnd_aggregation_pool()` function that reads
   konjnd-dense parquet, groups by ref, applies same standardization
   transform as primary stream.
3. `bin/zensim_mlp_train.rs:~780`: Add CLI flags (mirror existing
   pjnd-passthrough flags).
4. `bin/zensim_mlp_train.rs:~1800`: Wire flags into hyperparams.

### Phase 2 — Training-loop integration (~200 LOC)

5. `mlp_train.rs:~5950` (near pjnd_passthrough step): Add
   `konjnd_aggregation` step. Sample K refs, S rows each, forward,
   aggregate, compute loss, backprop with per-row gradient
   `(2w/S)·res·dya_dpre`.
6. `mlp_train.rs:~4960`: Gate setup logging.

### Phase 3 — Test fixture (~80 LOC)

7. `zensim-validate/tests/konjnd_aggregation_training.rs`:
   Synthetic 10-ref × 20-row test data where each ref's pjnd_target
   is a known function of a single feature. Trainer should reduce
   per-ref MSE to < 0.1 within 50 steps. Validates that the per-ref
   aggregation gradient backprops correctly.

### Phase 4 — Methodology + retrain (separate task #6)

Once Phase 1-3 land + tests pass, kick off task #6 (Tuner v11
retrain) using the new flag at weight=0.3.

## What this does NOT solve

- **The score-floor pathology** below 55 (task #1 finding #2) —
  aggregation only fixes the konjnd-budget gate; the dial deadzone
  needs LOW-Q TRAINING SAMPLES (which come from CID22 train-subset,
  task #5 ✓ done).
- **CID22 ↔ konjnd KonJND collapse** — aggregation lets us raise
  konjnd weight without collapse, but it doesn't guarantee CID22
  improves. The empirical question is settled by task #6.

## Falsification criteria for task #6 success

A Tuner v11 retrain with `--konjnd-aggregation-weight 0.3 --konjnd-aggregation-samples-per-ref 5`
SUCCEEDS iff (per Tuner trail gate in SOTA_TRAILS.md, AND beats v10
on the codec-target axis):

1. **KonJND val SROCC ≥ 0.85** (v10 baseline 0.2317 is the floor we
   beat; even matching v_balanced_v3's 0.8927 is a huge win).
2. **CID22 SROCC ≥ 0.864** (Compression-ship parity, currently
   v_tuner_v10 sits at 0.8540).
3. **Monotonicity ≥ 92.78%** strict on the 50-image × 19-q JPEG
   sweep (v10 floor).
4. **Cross-codec p50 |Δ| ≤ 1.0 in score 60–90** (v10 floor 0.6–1.5
   per task #1).
5. **Score 0–55 dial recovers**: at butter ≥ 6.8, per-anchor stddev
   no longer flat at ~55 (the v10 floor pathology disappears).

If 4/5 pass: ship as `PreviewV0_5TunerV5`. If only #1+#2 pass but #3
breaks: revert and try smaller aggregation weight. If #5 doesn't
recover: aggregation isn't enough; task #5 data alone wasn't enough
either; need fundamentally new low-q anchors (e.g., AIC-3 train-fold).

## See also

- `benchmarks/recovery_phase3b_falsification_2026-05-21.md` —
  root-cause analysis of the per-pair-target failure mode.
- `benchmarks/v11_d_pjnd_falsification_2026-05-20.md` — V11-D
  pjnd_passthrough mechanism (per-row, not aggregated) — falsified.
- `benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md` — task #1
  baseline numbers; goal #5 quantified here.
- `zensim-validate/src/mlp_train.rs:5950..6200` — existing
  pjnd_passthrough training step; aggregation step is mirrored from
  this.
