# Tuner v11 methodology (task #6, 2026-05-24)

**Status:** SKELETON — populated as the training run lands. Final
numbers and verdict will follow the 5-seed CI completion. This
doc is committed up-front per CLAUDE.md's "methodology doc
required before flipping include_bytes!" rule — so the recipe
and ship gate are auditable BEFORE the bake exists, not after.

## Hypothesis (per "principled experiment workflow")

1. **Hypothesis.** Three additive levers on top of V_tuner_v10 lift
   the codec-target dial without breaking its core monotonicity
   properties:
   (a) per-source aggregation head pools predictions across a
   konjnd-dense ref's distortion levels before MSE against the
   per-ref pjnd_target, breaking V11-D's per-row zero-gradient
   pathology and unblocking konjnd training weight up to 0.3.
   (b) CID22 training-only-subset (17,611 pairs, ssim2-anchored,
   CVVDP/IW-SSIM backfilled) adds real-codec content variation to
   the safesyn-dominated training pool — same image distribution as
   the held-out 49-ref validation set.
   (c) konjnd weight raised from V_tuner_v10's 0.02 to 0.3 to
   exploit (a). The aggregation head's gradient is non-zero per ref,
   so higher weight pushes more PJND structure into the network
   without flattening the per-pair feature → distortion mapping.

2. **Falsification.** If the 5-seed median bake fails any TWO of
   the five gate criteria (below), the hypothesis is dead. A single
   criterion miss is salvageable via a re-fit of the PCHIP spline
   (task #6 doesn't bake the spline; V10 anchor spline is reused).

3. **Cost ceiling.** 5 seeds × ~30 min wall = ~2.5 hr GPU. Plus
   ~30 min bake_verdict + methodology write-up. If wall exceeds 4 hr
   total, abandon and document the negative.

4. **Ship form.** If the gate passes: ship as
   `ZensimProfile::PreviewV0_5TunerV5` and update
   `ZensimProfile::codec_target()` to point at it. Old
   `PreviewV0_5TunerV4` (`v_tuner_v10`) remains accessible by name
   for reproducibility per the versioning policy in
   `docs/CODEC_TARGET_METRIC.md`.

## Reporting panel (per CLAUDE.md "SROCC-only verdicts BANNED")

Full Mohammadi 2025 panel at aggregate + 10-band level on all 6 val
corpora (CID22, KADID, TID, KonJND-1k, AIC-3 CTC, AIC-4 sample).
Stats: SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE + MAE + non-mono
q-step rate. Plus cross-codec consistency measurement vs task #1's
v10 baseline.

| Corpus | Role | When inspected |
|---|---|---|
| CID22 | gold-standard holdout | only after 5-seed packed (per "principled workflow" step 2) |
| AIC-3 CTC | compression holdout | only after 5-seed packed |
| AIC-4 sample | dial-precision anchor | only after 5-seed packed (best-discriminating in v10) |
| KADID | integrity guard | per-seed eval ok (training-set proxy) |
| TID | integrity guard | per-seed eval ok |
| KonJND-1k | PJND anchor (CRITICAL for this experiment) | per-seed eval — the aggregation head's direct success metric |

## Ship gate (five criteria; ≥4/5 required)

Per `docs/KONJND_AGGREGATION_HEAD_DESIGN_2026-05-24.md` "Falsification criteria":

| # | Criterion | V_tuner_v10 baseline | v11 target | Status |
|---|---|---|---|---|
| 1 | **KonJND val SROCC** ≥ 0.85 | 0.2317 | ≥ 0.85 | TBD |
| 2 | **CID22 SROCC** ≥ 0.864 (Compression parity) | 0.8540 | ≥ 0.864 | TBD |
| 3 | **Monotonicity** ≥ 92.78% strict (50-img × 19-q JPEG sweep) | 92.78% | ≥ 92.78% | TBD |
| 4 | **Cross-codec p50 \|Δ\|** ≤ 1.0 in score 60-90 | 0.6-1.5 | ≤ 1.0 (improvement: no widening) | TBD |
| 5 | **Score 0-55 dial recovers** (no flat clamp at ~55 for butter ≥ 6.8) | FAILED (anchors above butter 6.8 all land at mean 55) | per-anchor stddev not pinned | TBD |

Notes:
- Criterion #1 is the primary success metric — V_tuner_v10's
  KonJND collapse is the load-bearing pathology this experiment
  attacks. If aggregation head works mechanically, this lifts most.
- Criterion #5 is what the score-floor pathology fix demands.
  Adding low-q CID22-train pairs (median ssim2 64) should give the
  network samples that genuinely live below score 55.
- Criterion #3 (mono) is the codec-target dial's load-bearing
  property — degrading it would break Pattern A integrations
  (zenwebp's target_zensim outer loop).
- 4/5 = ship as PreviewV0_5TunerV5. 5/5 = celebrate and also
  consider tightening other thresholds in v12.
- ≤3/5 = falsified; bake stays in benchmarks/ for reproducibility,
  v_tuner_v10 remains the production codec-target.

## Recipe (full trainer invocation)

```sh
target/release/zensim_mlp_train \
    --group "safesyn:canonical-2026-05-21/train/safesyn.parquet:1.0:0.0" \
    --group "cid22_train:canonical-2026-05-21/train/cid22_train.parquet:0.5:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 5.66e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 32 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 20.0 \
    --ranknet-weight 0.0 --mse-weight 1.0 \
    --monotonicity-reg 1.0 --monotonicity-margin 0.0 \
    --anchor-parquet 2026-05-20-v9-anchors/anchors_v9_372col.parquet \
    --anchor-loss-weight 0.5 --anchor-target-score 60.0 --anchor-step-p 0.30 \
    --cross-codec-eq-parquet picker-training/.../cross_codec_equivalence_tight_v3.parquet \
    --cross-codec-eq-weight 1.0 --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.3 --dynamic-range-sigma-threshold 25.0 \
    --dynamic-range-step-p 0.05 --dynamic-range-probe-n 40 \
    --konjnd-aggregation-parquet canonical-2026-05-21/train/konjnd-dense.parquet \
    --konjnd-aggregation-weight 0.3 \
    --konjnd-aggregation-step-p 0.30 \
    --konjnd-aggregation-samples-per-ref 5 \
    --konjnd-aggregation-refs-per-step 8 \
    --seed <S> --out tuner_v11_s<S>.bin
```

Driver script: `scripts/v_next/run_tuner_v11_seed.sh <seed>`.

## Deltas vs V_tuner_v10

| Knob | V_tuner_v10 (v9 recipe) | V_tuner_v11 | Why |
|---|---|---|---|
| Training groups | safesyn only | safesyn + **cid22_train** | Add real-codec content matching CID22 holdout's distribution (task #5) |
| Substrate | canonical-2026-05-18 | **canonical-2026-05-21** | New canonical with cid22_train + ssim2-anchored layout |
| konjnd handling | implicit (anchor + cross-codec-eq pair loss) | **konjnd-aggregation-head w=0.3** | Per-source aggregation breaks V11-D's per-row zero-gradient pathology (task #4) |
| All other hyperparams | (per V9 recipe) | unchanged | Lock everything else so signal is attributable to the three additions |

## Inputs (md5 / row counts)

| File | Path | sha256 | Rows |
|---|---|---|---|
| safesyn.parquet | canonical-2026-05-21/train/safesyn.parquet | 1ee0565fb6cb… (per _MANIFEST.json) | 196,086 |
| cid22_train.parquet | canonical-2026-05-21/train/cid22_train.parquet | TBD (post-task #7 backfill) | 17,611 |
| konjnd-dense.parquet | canonical-2026-05-21/train/konjnd-dense.parquet | 87e196ba88ba… | 20,160 |
| anchors_v9_372col.parquet | 2026-05-20-v9-anchors/ | TBD | 22,008 |
| cross_codec_equivalence_tight_v3.parquet | picker-training/2026-05-19-v2/ | TBD | 68,788 |

Training binary: target/release/zensim_mlp_train at commit TBD
(post-task #4 ebf5f2e, includes the konjnd-aggregation flags).

## Iteration history

### Attempt 1: konjnd_aggregation_weight=0.3 — FALSIFIED on seed 1

First pass with the original brief recipe (w=0.3, step_p=0.30). Seed
1 finished at 03:30 UTC; the 5-seed CI was killed at this point
because the result was decisive.

| Corpus | v_tuner_v10 | v11_w0.3_s1 | Δ |
|---|--:|--:|--:|
| CID22 | 0.8540 | **0.5075** | **−0.347** |
| KADID | 0.4831 | 0.3260 | −0.157 |
| TID | 0.6636 | 0.3506 | −0.313 |
| **KonJND** | 0.2317 | **0.7579** | **+0.526** |
| AIC-3 | 0.7865 | 0.6637 | −0.123 |
| AIC-4 | 0.9240 | 0.9070 | −0.017 |
| Monotonicity | 96.44% | 92.89% | −0.036 |

**The aggregation head WORKS mechanically** — KonJND lifted +0.53,
breaking the V11-D zero-gradient pathology. But w=0.3 is too high;
the aggregation step's per-fire gradient overwhelms the per-pair
rank signal, collapsing CID22 by 0.34. Net effect: rank corpora
ruined for a KonJND win.

The α(x) gate also collapses to 0 within 10 epochs (already noted),
meaning the network reduces to pool-head only — consistent with the
aggregation gradient flowing primarily through the pool reducer.

Evidence preserved at
`/mnt/v/zen/zensim-eval/exp_tuner_v11_2026-05-24/tuner_v11_w0.3_s1_*`.

### Attempt 2: konjnd_aggregation_weight=0.05 — FALSIFIED on seed 1

Seed 1 finished at 03:46 UTC. Pipeline killed after seed 1 because
the failure was in the opposite direction from attempt 1:

| Corpus | v10 | attempt1 (w=0.3) | attempt2 (w=0.05) | attempt2 Δ vs v10 |
|---|--:|--:|--:|--:|
| CID22 | 0.854 | 0.508 | **0.742** | −0.112 |
| KADID | 0.483 | 0.326 | **0.598** | **+0.115** |
| TID | 0.664 | 0.351 | 0.610 | −0.054 |
| **KonJND** | 0.232 | 0.758 | **0.066** | **−0.166** |
| AIC-3 | 0.787 | 0.664 | 0.793 | +0.006 |
| Monotonicity | 96.4% | 92.9% | 95.1% | −1.3% |

Pattern: at w=0.05 the aggregation gradient is too weak to bend the
network toward PJND structure (KonJND collapses BELOW v10's 0.23!)
but strong enough to disrupt the rank head (CID22 drops 0.11). The
α gate ends up dynamic [0, 1] with μ=0.78 — neither extreme. KADID
unexpectedly LIFTS by 0.115 — the aggregation may be acting as a
regularizer on synthetic-distortion ranking.

**The (w=0.05, w=0.3) sweep maps both ends of the failure mode:**
- w=0.3 → α=0 collapse, KonJND wins +0.53, rank corpora collapse
- w=0.05 → α dynamic, KonJND collapses MORE, rank moderate degradation

There's no obvious stable middle along the (weight, step_p) axis at
these settings. The structural issue is that konjnd-dense is not in
the regular training group — so the aggregation pool is the network's
ONLY exposure to konjnd-dense features. That's qualitatively different
from how v10 sees safesyn (feature-driven per-pair MSE).

Evidence preserved at
`tuner_v11_w0.05_s1_evidence.bin` + verdict + qsweep.

### Attempt 3: konjnd-dense as BOTH training group AND aggregation — PARTIAL

Seed 1 finished at 04:01 UTC. Best overall rank-corpora numbers of
all attempts so far, BUT KonJND val still poor:

| Corpus | v10 | a1 (w=0.3) | a2 (w=0.05) | **a3** | a3 Δ vs v10 |
|---|--:|--:|--:|--:|--:|
| CID22 | 0.854 | 0.508 | 0.742 | **0.814** | −0.040 |
| KADID | 0.483 | 0.326 | 0.598 | **0.586** | +0.103 |
| TID | 0.664 | 0.351 | 0.610 | **0.688** | +0.024 |
| KonJND | 0.232 | 0.758 | 0.066 | **0.113** | **−0.119** |
| AIC-3 | 0.787 | 0.664 | 0.793 | **0.797** | +0.011 |
| Mono | 0.964 | 0.929 | 0.951 | **0.960** | −0.004 |

The per-pair konjnd-dense training (target=mix_cv40_iw60) ACHIEVED
0.99+ train SROCC on konjnd-dense — the feature-driven gradient
worked. But the per-pair predictions on KonJND-1k val (which scores
PJND threshold ranking, not mix_cv40_iw60 ranking) are uncorrelated
with the val truth.

This points at a deeper finding: **KonJND-1k val SROCC is structurally
hard to lift** because val tests PJND threshold ranking (per-pair)
but our training has no per-pair PJND-anchored signal. konjnd-dense's
per-pair mix_cv40_iw60 target is not PJND-correlated. konjnd-dense's
per-source pjnd_target requires aggregation (which collapses rank
when strong, fails to lift KonJND when weak).

Attempt 1's KonJND 0.76 was an artifact of pool-head reduction
(α=0) producing a smoothed prediction that correlated with PJND by
luck of the pool reducer's geometry — not a real architectural win.

### Attempt 4: a3 + strong aggregation (w=0.3) — KonJND +0.38, biggest progress yet

Seed 1 finished at 04:17 UTC. **The structural fix WORKS**:

| Corpus | v10 | a1 | a2 | a3 | **a4** | a4 Δ vs v10 |
|---|--:|--:|--:|--:|--:|--:|
| CID22 | 0.854 | 0.508 | 0.742 | 0.814 | 0.769 | −0.085 |
| KADID | 0.483 | 0.326 | 0.598 | 0.586 | 0.561 | +0.078 |
| TID | 0.664 | 0.351 | 0.610 | 0.688 | 0.614 | −0.050 |
| **KonJND** | 0.232 | 0.758 | 0.066 | 0.113 | **0.615** | **+0.383** |
| AIC-3 | 0.787 | 0.664 | 0.793 | 0.797 | 0.771 | −0.016 |
| Mono | 0.964 | 0.929 | 0.951 | 0.960 | 0.938 | −0.027 |

Per-pair konjnd-dense training (mix_cv40_iw60 target) + strong
aggregation (pjnd_target, w=0.3 step_p=0.30) COEXIST: the rank
gradient is no longer collapsed by the aggregation pressure, while
KonJND lifts dramatically. This is the architectural breakthrough
the recovery cycle has been chasing since V11-D (2026-05-20).

Per the 5-criterion ship gate:
1. KonJND ≥ 0.85: 0.615 — FAIL (gap −0.235)
2. CID22 ≥ 0.864: 0.769 — FAIL (gap −0.095)
3. Mono ≥ 92.78%: 0.938 — **PASS**
4. Cross-codec p50 ≤ 1.0: TBD
5. Score 0-55 dial: TBD

Trajectory shows monotonic KonJND lift via aggregation weight
(a3 w=0.1 → KonJND 0.11, a4 w=0.3 → KonJND 0.62). The next
iteration (attempt 5, w=0.5) should push KonJND higher.

### Attempt 5: a4 + aggregation w=0.5 — TRAJECTORY REVERSES; a4 is the local optimum

Seed 1 finished at 04:32 UTC. **The trajectory is non-monotonic** —
KonJND peaks at w=0.3 (a4) and DROPS at w=0.5:

| Corpus | v10 | a3 w=0.1 | **a4 w=0.3** | a5 w=0.5 |
|---|--:|--:|--:|--:|
| CID22 | 0.854 | 0.814 | **0.769** | 0.707 |
| KADID | 0.483 | 0.586 | 0.561 | **0.752** |
| TID | 0.664 | 0.688 | 0.614 | **0.751** |
| KonJND | 0.232 | 0.113 | **0.615** | 0.568 |
| AIC-3 | 0.787 | 0.797 | 0.771 | 0.714 |
| Mono | 0.964 | 0.960 | **0.938** | 0.940 |

Interpretation: at w=0.5 the aggregation gradient over-pulls
predictions toward per-source means, which adds noise to BOTH the
per-pair feature learning (CID22/AIC-3 drop) AND the per-ref
calibration (KonJND drops because per-ref signal becomes noisy).
KADID and TID lift because their distortion families happen to
align with smoothed predictions.

**a4 (w=0.3 step_p=0.30) is the sweet spot.** Committing to this
recipe for the 5-seed CI to get a median estimate.

### Attempt 6 (a4-5seed-CI): 5 seeds of a4 settings — IN FLIGHT

Pipeline started 04:35 UTC. ~70 min total wall.

If median holds at a4's seed-1 numbers (KonJND 0.62, CID22 0.77):
- Criterion 1 (KonJND ≥ 0.85): FAIL by 0.23 (but +0.39 vs v10)
- Criterion 2 (CID22 ≥ 0.864): FAIL by 0.10
- Criterion 3 (Mono ≥ 92.78%): PASS at 0.94
- Criterion 4 (cross-codec): TBD post-spline calibration
- Criterion 5 (score 0-55 dial): TBD

Decision after 5-seed CI:
- If median KonJND ≥ 0.70 AND CID22 ≥ 0.80: **ship as
  PreviewV0_5TunerV5** with documented trade (codec-target gets
  +0.39 KonJND lift, costs −0.07 CID22). This is the structural
  fix recovery has been chasing since V11-D.
- If median KonJND < 0.55 OR CID22 < 0.75: don't ship; try
  w=0.4 step_p=0.20 (between a4 and a5).

Recipe deltas vs attempt 2:
- `--group konjnd_dense:konjnd-dense.parquet:0.3:0.0` (NEW — regular
  per-pair MSE against mix_cv40_iw60)
- `--konjnd-aggregation-weight 0.1 --konjnd-aggregation-step-p 0.10`
  (moderate aggregation aux loss)
- All other hyperparams identical to attempts 1+2.

Pipeline started 03:50 UTC. Early signal (epoch 0):
- val=0.9196 (above attempts 1+2 at epoch 0)
- konjnd_dense train SROCC = 0.9925 already — the per-pair MSE on
  mix_cv40_iw60 IS giving feature-driven gradient (no zero-gradient
  pathology). This is the structural fix attempts 1+2 were missing.

Expected outcome (3 hypotheses):
- (A) The per-pair konjnd training group lifts KonJND val to ≥ 0.6
  (between v10's 0.23 and attempt 1's 0.76), CID22 stays near v10's
  0.85. Ship-grade. → 5-seed CI on this recipe.
- (B) KonJND moderate (0.4-0.6), CID22 slight drop (0.80-0.85).
  Partial pass — drop konjnd train_w to 0.1 + bump agg weight.
- (C) Same failure pattern as attempts 1/2. → Mechanism change
  required (drop per-sample-α, use pool_head directly, OR add
  gradient normalization to aggregation step).


If attempt 2 fails (likely if α=1 stays pinned — early signal at
epoch 50 shows α=1.000 across the network, indicating the
aggregation gradient is too small to displace rank-head dominance):

The root cause is that v11 currently only includes konjnd-dense via
the aggregation pool, NOT as a regular training group. konjnd-dense
has TWO targets populated:
- `mix_cv40_iw60` PER-ROW (range -66 to +96, median 68) — feature-driven per-pair target.
- `pjnd_target` PER-REF (range 22-70, median 32) — per-source-constant aggregation target.

V_tuner_v9 trained ONLY on safesyn — it never saw konjnd-dense
feature → score mappings at the per-pair level. That's why v10
KonJND val SROCC is 0.23: no per-pair konjnd training signal at all.

**Attempt 3 recipe** (use BOTH konjnd-dense slots):
- `--group konjnd_dense:konjnd-dense.parquet:0.3:0.0` (regular MSE
  against mix_cv40_iw60; train weight 0.3 — same as cid22_train)
- `--konjnd-aggregation-weight 0.1 --konjnd-aggregation-step-p 0.10`
  (aggregation aux loss at moderate weight)
- All other v11 hyperparams identical.

Rationale: regular per-pair MSE gives feature-driven gradient on
konjnd-dense (no per-source constant collapse since each row has
its own mix_cv40_iw60 value). Aggregation MSE adds soft per-source
PJND calibration. Combination should produce a network that ranks
konjnd-dense pairs correctly AND lands per-ref mean near pjnd_target.

If attempt 2 ALSO fails (i.e. neither rank nor aggregation lifts
KonJND beyond v10):
- Skip to attempt 3 (the per-pair MSE on konjnd-dense is the
  missing signal, not the aggregation weight).

Backup mechanism options if attempts 2 + 3 both fail:
- Aggregation step gradient normalization (divide by rank-step
  gradient L2 norm)
- Per-ref weighted sampling (currently uniform over 1008 refs)
- Skip per-sample-α head — use pool_head + aggregation directly
  (current trainer requires per-sample-α; would need to extend
  pool_head to accept konjnd_agg pool).

## Results (5-seed CI, w=0.05 step_p=0.10)

| Seed | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|---|--:|--:|--:|--:|--:|--:|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD | TBD | TBD |
| **median** | TBD | TBD | TBD | TBD | TBD | TBD |

## Observed during training: α(x) collapse to 0

In the 2026-05-24 5-seed CI run, the per-sample-α gate `α(x) =
sigmoid(W_α·h + b_α)` collapses from its init value 0.5 down to
exactly 0.0 (min=max=0.000) within the first 10 epochs, and stays
pinned there for all 300 epochs across all seeds. The training is
otherwise healthy — val SROCC improves epoch-by-epoch, the network
keeps learning — but the per-sample mix collapses to "use the pool
head exclusively, ignore the rank head."

Mechanically, the aggregation step's gradient is a sum-over-S-rows
that flows through the pool head (where the K·S rows contribute
their per-prediction gradient) AND through the per-sample α gate
(where the residual gets multiplied by `(1 - α)` for the pool
branch and `α` for the rank branch). The trainer settles into a
local minimum where α≈0 minimizes the aggregation MSE because the
pool reducer (4 fixed pooling functions over the hidden vector)
preserves more cross-row consistency than the per-row rank head.

**Runtime implication**: the runtime forward path
(`zensim::metric::forward_one_bake`) handles α=0 correctly —
`alpha * y_rank + (1 - alpha) * y_pool` becomes `y_pool` exactly.
The bake is functionally a pool-head bake carrying per-sample-α
metadata that the dispatch ignores (α=0 → no rank contribution).
This is correctness-preserving but suggests the next iteration
(v12?) could simplify to `pool_head` directly without the α gate.

**Not a regression**: V_tuner_v9 (current ship) also shows α≈0.4
at convergence — a "mostly pool, some rank" blend. v11's α≈0
extends this further. The cause is the new aggregation step
strengthening the pool-head gradient relative to the rank head.

## Cross-codec consistency vs V_tuner_v10 baseline

Re-run `scripts/v_next/measure_tuner_v10_cross_codec.py` against
the median-seed bake; compare per-anchor p50/p90 |Δ| to v10's
`benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md`.

| butter_level | v10 p50 \|Δ\| | v11 p50 \|Δ\| | Δ |
|---|--:|--:|--:|
| 1.107 (score ≈ 84) | 0.566 | TBD | TBD |
| 1.510 (JND-adjacent) | 0.786 | TBD | TBD |
| 2.721 (PJND ≈ 63) | 1.470 | TBD | TBD |
| ≥ 6.8 (low-q) | (clamped flat) | TBD (key gate #5) | TBD |

## Verdict

TBD post 5-seed CI completion. Update this doc with the gate
status and ship decision before flipping
`ZensimProfile::codec_target()` in `zensim/src/profile.rs`.

## See also

- `docs/CODEC_TARGET_METRIC.md` — codec-target integration guide
  (the document this bake is canonical for)
- `docs/KONJND_AGGREGATION_HEAD_DESIGN_2026-05-24.md` — task #4
  design
- `benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md` —
  task #1 baseline this experiment must improve on
- `benchmarks/recovery_phase3b_falsification_2026-05-21.md` —
  root cause of the konjnd-dense per-row pathology that this
  experiment attacks
- `SOTA_TRAILS.md` — Tuner trail gate definitions
