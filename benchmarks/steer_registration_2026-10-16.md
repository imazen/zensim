# STEER — smooth activations and exact block re-scoring

Registration revision 5, 2026-10-16. Owner Q1–Q3 resolved 05:48–05:49 MDT; **Owner Q12 option A approved, saved 2026-10-16 at 14:41 MDT.** **Staged freeze; scientific launch requires a separate coordinator go. This document alone authorizes no implementation, fit, feasibility probe or assessment.** The coordinator lands and freezes the owner-approved version. The approved settings are design choices, not measurements. The narrow deterministic-alias state treatment is owner-approved option A, saved 2026-10-16 at 14:41 MDT.

## 0. Scope and controlling precedent

Can a change to the score network or the mechanism that ranks repairs preserve accuracy and pass the existing steering and serving gates? Run separately alongside E35, using the existing E34 training recipe. Do not combine E35's data changes with the mechanism comparison. A winning data recipe is refitted on the fixed architecture in a later registration. **Output isolation until both experiments close:** neither experiment may consume the other's model bytes, predictions, metrics, gate outputs, checkpoints or intermediate tables, including pilot and failed-attempt outputs. This prohibits reading or using those outputs, not merely selecting from them. Shared scheduler/resource telemetry is allowed solely for allocation, contention control and worker health, without scientific outcomes or metrics. The coordinator records each experiment's complete terminal status and closure receipt; the boundary ends only when both closure receipts exist. Any later scientific transfer requires its own registration.

Controlling precedent is [E34 frozen revision 5, bc9be24a](https://github.com/imazen/zensim/blob/bc9be24a/benchmarks/e34_registration_freeze_draft_2026-10-15.md), including D12 exact gradients and D13 margin **0.0065 in both families**, read with [E34's evaluated interpretation and results](https://github.com/imazen/zensim/blob/3af52903/benchmarks/e34_results_2026-10-15.md). Formal §4.2 governs: mean Δ ≥ −0.002 is **report-only**, not a decisive guard. No numerical gate is weakened here. Owner Q1 fixes legacy linear M2 as decisive; **only S can qualify**. R and S+R are diagnostics regardless of measured gate outcomes. Approved Q12 option A (saved 2026-10-16 at 14:41 MDT) concerns only the prospective handling of predeclared deterministic statistical aliases, not M2, W2 or any numerical threshold (§4.3/§9).

Sources were read from local `origin/main` at `81a545a3ff3457c9fabae93bc9d262b19b50952e`, and E34's frozen revision, without changing a checkout. Revision-2 source checks used read-only `git show origin/main:<path>` at `90b2a45a5b323713e2dbd0beaf4de703f911954b`; the seed/zero-SE/verdict contracts cited below were checked against source. Revision 3 corrects the earlier seed-error reading: ordinary seed-only diagnostic refusals are nested nonfatal reports, and numeric dataset/W2 evidence is required for improvement comparisons too. This was rechecked against the unchanged E34 owner at source `8ce79ac6`; [focused re-review STEER-05/06](REG_REVIEW2.md) withdraws the earlier propagation claim. These corrections restore inheritance; Q12 option A is approved, saved 2026-10-16 at 14:41 MDT. Relevant history: [status §§30–47](https://github.com/imazen/zensim/blob/81a545a3/benchmarks/rev5_research_status_2026-10-05.md#30-release-gates-on-the-frozen-seed-0-model-2026-10-08-0125-utc). Existing roles and qualification requirements continue to apply. No label payload was opened for this draft; sources were documents, code and existing execution receipts.

## 1. Existing evidence and hypotheses

E34's exact-gradient controls scored A **131/130/128** and C **130/130/133** joint passes at seeds 0/1/2. W/K kept registered accuracy but missed G-STEER; only L-A passed **135/135/135**, while losing **0.0226126** on the E21 statistic and failing accuracy guards. These are [E34 results](https://github.com/imazen/zensim/blob/3af52903/benchmarks/e34_results_2026-10-15.md#fixed-steering-and-development-panels), not new measurements.

The coordinator's supplied E34 overlap summary (2026-10-16; SHA-256 `378919c2b7b2f9c822c32a330d7718fb7ba645976f8fed7224c163c5498c01ec`) reports 77 failing rows over 41 cases in 18 models; 25 cases fail in one model only. Of the failures, 75 miss only M2 and two miss both M2 and M3f; 47 use b64; four pictures carry 50. These descriptive counts motivated the design. No per-case data was newly analyzed for this registration.

**H-S:** spreading a ReLU slope transition over a finite interval reduces finite-repair linearization error enough to pass M2 without L's accuracy loss. A continuous derivative is not a guarantee: a smooth concave function can still misrank finite changes, and training may sharpen effective transitions by rescaling weights.

**H-R:** evaluating the loaded score on repaired feature rows eliminates the network-linearization error in the served repair ranking. It does not fix linear M2, extraction approximations, a constant output floor, or excessive serving cost. Exactness requires the complete native repair feature delta, not merely the subset currently replayed locally.

**H-SR:** the combination can separate residual feature-map error from score curvature on the same smooth weights. It costs no additional fit, but its evaluation cost is unmeasured.

The design is adaptive to historical results on the 135 cases. That panel is a fixed qualification check, not an untouched generalization sample or a tuning population.

## 2. Fixed recipe, families and controls

Use the immutable E34 rank-data archive SHA-256 `9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008` and its admitted row keys, exclusions, feature declarations, weights, split roles and order. No additional training dataset, repair penalty, feature, teacher, anchor or E35 recipe enters this experiment.

- **A / diff-410:** incumbent identity `7cf4cfd9`; spec `sel:3b7bd5ebe929@h32:H128:cv16:cf98`, the same 410 differences.
- **C / diff+frag-820:** incumbent identity `89cb4861`; same spec plus `:fx1`, the same 410 differences and 410 same-cell difference × reference-fragility products. No additive reference-only input.
- Head N, pin 100, hidden width 128, human weight 32, coverage cv16:cf98; 120 epochs × 50,000 pairs; select final epoch **119**, never the best development checkpoint. Rev5, fit tier v3, no feature transforms; unchanged optimizer and pack/output-stage recipe.
- LODO: the fixed zero-based fold order is **0=`kadid`, 1=`tid2013`, 2=`konfig`, 3=`cid22_a25`**, each at seed indices 0–9. No other fold is admitted. Full-data seeds 0–2 are the qualification models; seeds are never selected by results.
- V40 initialization stream: `[1101,1103,1107,1109,1117,1123,1129,1151,1153,1163]`; sampling stream `[101,100000101,200000101,300000101,400000101,500000101,600000101,700000101,800000101,900000101]`, as recorded in E34's archived fit contract. For LODO fold f and seed index i, the exact mapping is **init=INIT_SEEDS[i], sample=SAMPLE_SEEDS[(i+f)%10]**. For full-data fits there is **no rotation**: init=INIT_SEEDS[i], sample=SAMPLE_SEEDS[i], i∈{0,1,2}. Pair S and its fresh Control by the same i and f; never rotate the initialization stream. Bind both the fold order and the resolved numerical argv, not only the stream lists. These are the existing [seeds/confirm_seeds owners](https://github.com/imazen/zensim/blob/81a545a3/scripts/rev4_featpot/v2_common.py#L415).

The frozen argv table must expand the following exact routing table for both families and both fitted weight types (Control/S). `I` and `P` denote the initialization and sampling arrays above; each emitted trainer argv contains the literal resolved numbers. Symbolic formulas in this draft are not a replacement for the frozen expanded argv manifest.

| Route | Entry-point selection argv | Trainer `--init-seed` | Trainer `--sample-seed` | i=0 pair; i=9 pair |
|---|---|---|---|---|
| LODO f=0, `kadid` | `v2_lodo_mlp.py --seed-index i --heldout kadid` | `I[i]` | `P[(i+0)%10]` | 1101/101; 1163/900000101 |
| LODO f=1, `tid2013` | `v2_lodo_mlp.py --seed-index i --heldout tid2013` | `I[i]` | `P[(i+1)%10]` | 1101/100000101; 1163/101 |
| LODO f=2, `konfig` | `v2_lodo_mlp.py --seed-index i --heldout konfig` | `I[i]` | `P[(i+2)%10]` | 1101/200000101; 1163/100000101 |
| LODO f=3, `cid22_a25` | `v2_lodo_mlp.py --seed-index i --heldout cid22_a25` | `I[i]` | `P[(i+3)%10]` | 1101/300000101; 1163/200000101 |
| Full-data, i=0/1/2 | `v2_confirm_fit.py --seed-index i --train-only --pack-production --e33-output-stage` | `I[i]` | `P[i]` | 1101/101; 1103/100000101; 1107/200000101 |

Every route also binds the exact family `--spec`, `--head N`, archive locator/digest, `--strict-admission`, data-role receipt and destination through the existing executor contract. The emitted trainer retains `--epochs 120 --pairs-per-epoch 50000 --early-stop-patience 0` and the inherited remaining argv. Control selects existing ReLU; S selects the new serialized hinge with τ=1/8. The exact new activation selector/parameter syntax is an implementation binding still to be frozen, not a fabricated existing CLI flag. R and S+R have no training argv: they refer to freshly executed Control/S cells and separate serving-mode manifests.

Fresh **Control-A and Control-C** are the inferential comparators, reproducing those incumbent recipes under the new pinned program. Preserve the original incumbent bytes and E34 control receipts as historical parity anchors. Fresh controls are the owner's chosen contract: execute all86 new control fits. **No E34-control import or reuse is authorized**, and concurrent E35 controls are prohibited by §0. Historical receipts are parity anchors only, never new execution receipts. A full-budget control replay must reproduce the corresponding archived normalized weights and scored bits before scale-out; a mismatch stops admission for diagnosis.

## 3. Mechanisms and fit packet

### 3.1 S: a fixed C¹ convex hinge

Replace only the hidden ReLU by the following activation. Owner-approved width **τ = 1/8 = 0.125**, fixed for every unit, fold, seed and family in hidden preactivation units. It is not a pixel threshold, learned parameter or pilot-selected value.

```text
hτ(z) = 0                         if z ≤ 0
        z² / (2τ)                 if 0 < z < τ
        z − τ/2                   if z ≥ τ

hτ′(z) = 0                        if z ≤ 0
         z/τ                      if 0 < z < τ
         1                        if z ≥ τ
```

At z=0 the derivative is 0; at z=τ it is 1, agreeing from both sides. This is C¹, not C². The activation is nonnegative, nondecreasing and convex. Its asymptotic positive slope is 1. The value of τ is a preregistered hypothesis; its adequacy is **unmeasured**.

Keep the existing [nonneg_project owner](https://github.com/imazen/zensim/blob/81a545a3/zensim-validate/src/mlp_train/mod.rs): zero scaler means, b1=0, w2≤0, b2=100 at initialization and after every optimizer step. For standardized network input x,

```text
raw(x) = 100 + Σj w2j hτ(W1j·x).
```

Consequences, by construction:

- Each negative-weighted convex hinge is concave; their sum is concave in x. For C, products are affine in differences at a fixed reference, so the same statement applies there. This does not assert global concavity after nonlinear feature extraction, clamps or the output spline.
- hτ(0)=0 gives raw(0)=100; hτ≥0 and w2≤0 give raw(x)≤100 for every x. Negative raw scores remain permitted. A zero-difference identity remains zero in both A and C's input layouts. Identity must also pass through final packed f16 weights, native pixel scoring and all inherited identity checks.
- The activation is monotone in its own argument. The current unconstrained W1 does **not** prove coordinatewise score monotonicity; S introduces no such claim or additional sign projection. C1 and all observed monotonicity gates remain necessary.
- Score scale stays the existing raw pin and 0–100 serving dial with its negative tail. Quantize before fitting the existing identity-pinned output spline; no compensating scale factor, new calibration population or floor policy. Matching τ's positive slope does not guarantee unchanged score distributions or accuracy.

D12 must extend to this activation in the trainer, serialized zenpredict execution, bake/refit owner, independent f64 reference and [loaded-score reverse derivative](https://github.com/imazen/zensim/blob/81a545a3/zensim/src/metric/bake_gradient.rs). It differentiates the **loaded** weights, scalers, derived products, spline and existing clamp/floor handling. The current gradient owner explicitly refuses unknown activations; this is required implementation work, not an already-supported switch. Pin τ in model serialization; no inference-only approximation or silent ReLU fallback. Existing recipes retain their activation and bits.

Before freeze require endpoint/interior analytic derivative checks and independent references, packed round-trip parity, exact identity, native score/feature parity across required tiers and strided input, gradient wiring parity, and explicit unsupported-mode refusal. Extend canonical owners; do not create a second production scorer. Smooth curvature is not numerically differentiated. Record raw-unit curvature and effective transition widths τ/||W1j|| descriptively; zero-norm units are explicitly marked constant, not divided by zero. No rescaling constraint or new regularizer is added after observing those diagnostics.

### 3.2 R: exact re-scoring of every candidate block

**Owner Q2: K = B, all blocks on the 8/16/32/64 grids; no shortlist.** B is the number of valid edge-clipped, nonoverlapping grid rectangles for a requested block size on the current image. Reference repair means paste the reference pixels into that rectangle, with every other pixel unchanged, just as the existing instrument does. This experiment does not claim exact predictions for an arbitrary encoder edit.

For every rectangle b, obtain the complete native feature row fᵇ of that repaired image and evaluate the same packed scorer:

```text
Δfᵇ = fᵇ − f
qᵇ_R = F(f + Δfᵇ) − F(f).
```

F includes every declared input, derived product, loaded weight, output spline and clamp. In floating arithmetic, forward the extracted fᵇ row directly; do not reconstruct it by subtracting and re-adding f, which could lose bits. The equation denotes the mathematical intervention. Use the native score at identity too; do not bypass its exact-100 behavior. Preserve the ordinary comparison score and feature bits. No gradient replaces this finite forward; the D12 gradient remains available and unchanged for M2 and diagnostics.

The [served prepared path](https://github.com/imazen/zensim/blob/81a545a3/zensim/src/metric/bake.rs) is `prepare_steering` → `SteeringSession::compute` → [ScoredAttribution::refinement_gain](https://github.com/imazen/zensim/blob/81a545a3/zensim/src/attribution.rs). Extend that owner with an explicit research mode returning exact finite gains for the registered rectangles. Density remains an explanatory approximation; the encoder's block ranking and instrument M3f must consume the exact-gain result through the same path. Do not compute an oracle only in the harness while leaving serving linear.

For the approved serving contract, the diagnostic mode eagerly builds gains for **all rectangles in each of the 8,16,32,64 grids** on every comparison; each G-STEER case uses its matching grid. Rectangle order is y-major then x-major, edges clipped exactly as the instrument. Duplicate rectangles may share an exact result, with the cache and aliasing recorded. A rectangle outside the declared inventory is an explicit unsupported query, not a silently approximate result. Before freeze verify that every registered case rectangle is covered; missing coverage is INCOMPLETE, never a dropped case. Any broader caller requires a separately specified cost and contract.

Charge this entire eager calculation to the complete cached **bin-1** score/map arm, even when the timing harness does not subsequently ask each rectangle. Charge allocation, feature extraction, product construction, model forwards and gain-table materialization; no oracle feature cache or deferred work hides outside timing/RSS. The uncached score denominator remains the same candidate's ordinary score. This preserves the spatial threshold and prevents a cheap map-construction number from describing an expensive serving path.

**Exactness boundary:** current `LocalRefineSnapshot::deltas` provides a subset, notably coarse v2 replay; `refinement_gain` also has frozen-signal, moment/root and model-curvature limits. Evaluating F on those approximate/subset deltas is not this arm. Initial exact implementation may sequentially extract each full repaired image through the native owner, reusing the prepared reference where supported. An optimized complete-delta implementation must first match the full native repair feature row and loaded score under the unchanged parity contracts. No zero fill for missing deltas.

Reuse Control-A/C weights for R, including their LODO predictions. R is a serving mechanism, not a refit. Native independently extracted repair forwards provide the truth; cached results cannot verify themselves. Publish parity receipts, including identity, boundary rectangles, blur neighborhoods, maxima, products, repeated calls and refusal cases. No shortlist is part of the approved packet.

### 3.3 S+R and counts

S+R applies exactly §3.2 to the S models. R and S+R are **diagnostic-only by owner Q1**; neither may enter full qualification or replace an incumbent, even if all its measured gates pass. Include S+R: **zero extra fits**, useful as a paired mechanism diagnostic. Serving/evaluation cost remains unmeasured and is not assumed cheap.

| Fit jobsets | LODO fits | Full-data fits | Total |
|---|---:|---:|---:|
| Control-A | 4 × 10 = 40 | 3 | 43 |
| Control-C | 40 | 3 | 43 |
| S-A | 40 | 3 | 43 |
| S-C | 40 | 3 | 43 |
| R-A/C; S+R-A/C | reuse the above | reuse the above | 0 additional |
| **Approved total** | **160** | **12** | **172** |

Eight fit jobsets if each family/weight type has separate LODO and full-data sets. Six arm/family serving variants × three full-data seeds = **18 arm serving evaluations (six S qualification evaluations and12 R/S+R diagnostic evaluations)**, plus **six matched-control evaluations**, total **24** serving evaluations. Only S-A/S-C have a qualification route. R and S+R still execute distinct native serving evaluations; reuse of model bytes never substitutes one mechanism's gates for another's. Full MLP fits: 172; extra bounded admission smokes are separately identified and never entered as scientific cells. No lambda pilot, checkpoint search or activation-width search.

## 4. Evaluation and decision rule, fixed before any fit

### 4.1 Gate semantics: current definitions remain decisive

For each of the same **135 fixed cases**, over its complete rectangle inventory:

```text
t_b = F(fᵇ) − F(f)                              native complete-score true gain
l_b = s·(fᵇ − f)                               D12 exact loaded-score sensitivity
q_b = served refinement gain from this arm      encoder-facing quantity
M2  = Spearman(l_b, t_b)
M3f = Spearman(q_b, t_b).
```

Use the [existing engineering packet](https://github.com/imazen/zensim/blob/81a545a3/zensim/src/metric/steerfix_packet.rs), pinned case order, rank/tie rules and lower-floor-recovery policy. Qualification uses its served objective; pre-floor/raw results are labelled diagnostics. The complete-score truth retains serving floors even when the registered prepared gradient recovers pre-floor units. No change to this exception is implied by smoothing or re-scoring.

A case passes iff **M2 ≥ 0.99 and M3f ≥ 0.70**. Seed 0 must pass **135/135**. Each of seeds 1 and 2 must pass at least **max(its paired Control seed's count,133)**. Fixed counts decide, not intervals. Missing/nonfinite evidence never passes. No reference, size, tied case or historical failure is excluded.

For S, M2 measures the smooth network's exact local linearization and M3f measures the native approximate map. For R, **M2 still measures l**, and M3f measures exact q_R against independent t. For S+R, M2 is S's linearization and M3f is S+R's exact served gain. R does not inherit a pass merely because q_R=t: unchanged weights imply unchanged M2. Likewise S+R cannot rescue S's M2 failure. Exact self-ranking can be 1 only on a nonconstant finite gain vector; constant vectors retain the existing harness outcome. Do not redefine sensitivities to make a finite gain look like a derivative.

A separate `served_gain_rank = Spearman(q,t)` is already M3f, not a replacement M2. Owner Q1 resolved the semantics: legacy linear M2 remains decisive, **R and S+R stay diagnostic-only and only S can qualify**. No alternate gate semantics or diagnostic-to-qualification fallback is available in this registration.

Read the fixed panel once per frozen served variant after all fits, including all arms even if another gate fails. Retain per-case M2, M3f, relative-L2 linearization error, strict raw true>linear counts, gain vectors and the regret below. Report picture-clustered pass-rate intervals (percentile and Student, B=10,000, G=15), descriptively; those intervals cannot replace fixed counts. No use of its results to choose τ, K, a seed, grid, implementation approximation or stopping rule.

### 4.2 Accuracy and point guards

Inherit E34 formal §4.2 with D13 and its evaluated interpretation. For each dataset's withheld-fold rows and each paired seed, recompute signed SROCC. Δ is the equal-four-dataset mean of arm minus Control SROCC, averaged over the draw's seed copies. H0: **Δ ≤ −0.0065**, both families. Decisive point guards: every dataset Δ_d ≥ **−0.005**, and W2 Δ **> −2 × its two-way SE**. Despite its historical name, W2 averages the three lowest signed distortion-type SROCCs within each of KADID and TID2013, then averages their paired deltas across those two datasets and seeds, exactly as the E34 owner does. It is not the two worst types.

Mean Δ ≥ −0.002 is **NON-decisive**: report it and which arm/family verdicts would change if it were decisive. This interpretation is fixed in this draft before any STEER assessment p-value is read. Preserve dataset intervals, seed-unit/reference-only uncertainty, pair classes and high-quality-slice reports as in E34; none silently becomes a gate.

Statistics remain in zenmetrics-stats through canonical Rust panel owners, pinned at freeze (E34 used `48665c8f`). Paired `Design::TwoWay(reference,seed)` with globally dataset-qualified references; same reference and seed draws for both predictions. Preserve E34's unstratified/copy-aware design and its nonfinite-draw rules; no switch to a stratified estimator here. B=10,000, 95% type-7 percentile intervals and Student intervals; pooled Student df=9 with SE correction √(10/9). The one-sided bootstrap p is `(1 + count(theta* ≤ 0))/(B_finite + 1)`; Student uses theta_hat divided by corrected SE; **max(bootstrap p, Student p)** decides. theta=Δ+0.0065 for eligibility and Δ for improvement. Record finite/dropped draws, seeds, both p-values, both intervals and point guards. Missing/invalid required evidence is INCOMPLETE and contributes p=1, blocking the entire packet verdict as in E34. The only approved exception is the pre-fit, construction-declared alias state in §4.3, **owner Q12 option A, saved 2026-10-16 at 14:41 MDT**. It is never an automatic accuracy pass, guard pass or zero-SE inferential rejection.

### 4.3 Twelve-test state machine — owner-approved Q12 option A

**Owner approved Q12-A on 2026-10-16 at 14:41 MDT (coordinator relay in STEER_LAND_brief.md).** The following is the operative pre-fit rule. It is a narrow prospective departure from E34's packet-wide completeness rule; this decision does not itself authorize freeze, scientific fits or assessment. All numerical bars, the strict W2 comparison and legacy linear M2 remain unchanged.

One owner-approved Holm family at **α=0.05** keeps exactly **12 ordered slots**. All nonidentical comparisons execute unconditionally, including diagnostic-only comparisons; no slot is removed because of a gate outcome or because it cannot select an arm.

| Test index | Paired comparison | Role | Pre-fit comparison state |
|---|---|---|---|
| 1 | S-A − Control-A | S eligibility | ordinary |
| 2 | R-A − Control-A | diagnostic eligibility-form comparison | identical by construction |
| 3 | S+R-A − Control-A | diagnostic eligibility-form comparison | ordinary; same predictions as test1, not an identical pair |
| 4 | S-C − Control-C | S eligibility | ordinary |
| 5 | R-C − Control-C | diagnostic eligibility-form comparison | identical by construction |
| 6 | S+R-C − Control-C | diagnostic eligibility-form comparison | ordinary; same predictions as test4, not an identical pair |
| 7 | R-A − S-A | diagnostic improvement | ordinary |
| 8 | S+R-A − S-A | diagnostic improvement | identical by construction |
| 9 | S+R-A − R-A | diagnostic improvement | ordinary |
| 10 | R-C − S-C | diagnostic improvement | ordinary |
| 11 | S+R-C − S-C | diagnostic improvement | identical by construction |
| 12 | S+R-C − R-C | diagnostic improvement | ordinary |

Use E34 RNG base20261011 and offsets100×test_index for primary, +d=1..4 in KADID/TID2013/KonFiG/CID22-A order for dataset guards, +5 for W2, +6 for reference-only. Pair-class offsets remain1000+10×eligibility_index+c, c=1 same-ladder,2 cross-codec,3 cross-source. Descriptive series keep the §4.4 order and the frozen family A/C, variant Control/S/R/S+R, seed0/1/2, subset all/b8/b16/b32/b64 ordering, with RNG offset2000+j. Every result stores its exact seed and finite/dropped draws.

#### A1. Admission and construction declarations

Before **any fit**, freeze an explicit alias manifest: R-A→fresh Control-A, R-C→fresh Control-C, S+R-A→S-A, S+R-C→S-C for ordinary accuracy predictions, all four LODO folds and ten seeds. Only **test slots2,5,8,11** have pairwise identical arms by this declaration. Full-data ordinary scores also require corresponding aliases at seeds0–2, although finite-repair gains/maps differ and are measured independently.

After fitting, validate every required cell, provenance binding, dataset/row-key order, seed/fold mapping, orientation, finite target/prediction value and source/type coverage. Every constituent full-population SROCC and W2 type correlation must be defined; an alias cannot make invalid constituent evidence valid. Verify exact ordinary prediction-bit equality of each declared alias across every required row and seed, plus required native serving parity. A missing row, nonfinite value, broken alias, hash mismatch, missing cell or insufficient metric coverage is **INVALID**, leading to packet INCOMPLETE. Two vectors that happen to be equal after fitting but were not declared as a pair before fitting receive no exception. No zero-SE inspection can create an alias.

#### A2. Primary comparison and reference-only component

For an admitted **ordinary** slot, execute the unchanged E34 paired unstratified TwoWay primary and reference-only calculation. Apply its finite-draw rules, margin, corrected Student calculation and max-p rule. A missing/invalid interval, zero finite draws or undefined primary decision (including zero primary SE) produces hypothesis **INCOMPLETE**, decision p=1, and packet INCOMPLETE. Preserve all available diagnostics and reasons; do not return a substitute S verdict.

For admitted slots2,5,8,11, record hypothesis state **IDENTICAL_BY_CONSTRUCTION**, displayed exactly as **“identical by construction, p = 1”**. The mathematically constant paired Δ=0 is reported. Eligibility-form theta=.0065 in slots2/5; improvement theta=0 in slots8/11. Run the same fixed bootstrap draws and copy-aware draw admission/counting with the frozen seeds; retain finite/dropped counts. Every admitted paired draw has constant delta, hence variance0. Zero finite draws or invalid constituent calculations remain fatal, even for an alias. Record exact point/percentile delta bounds[0,0], eligibility theta bounds[.0065,.0065] or improvement theta bounds[0,0], explicitly analytic/degenerate; dataset/W2 alias delta bounds are[0,0]. A Student interval/tail at zero SE is **null, reason=construction-declared zero variance**, never a numeric interval or p invented by division. Primary/reference-only components carry the construction-declared state, not an ordinary COMPLETE inference. Set **decision p=1 by owner-approved pre-fit policy**, with bootstrap/Student inferential p fields null and the override reason stored. Do not reinterpret the analytic positive eligibility theta as an inferential rejection. If required constituents or draw admission are invalid, A1's fatal rule wins.

#### A3. Seed component and seed_stat's zero-SE refusal

The existing [seed_stat and seed_component](https://github.com/imazen/zensim/blob/81a545a3/zensim-validate/src/bin/panel_e34/mod.rs#L43) require ten finite four-source units and **SE>0** for the seed diagnostic. However, `seed_component` catches a `seed_stat` refusal into a nested **INCOMPLETE** diagnostic within `Ok`; the E34 W2 caller separately catches its seed-diagnostic refusal. Neither nested state changes the hypothesis's primary decision or packet completeness. The earlier statement that these refusals propagate into a packet veto was incorrect and is withdrawn. Seed-only diagnostics remain report-only for **every ordinary comparison**, including improvement slots7/9/10/12.

The STEER adapter branches on the validated **pre-fit alias declaration before calling seed_stat** for slots2/5/8/11. Verify the ten-by-four matrix of paired deltas is exactly zero, then emit a structured construction-declared seed record: n=10, delta=0, SE=0, ten zero seed deltas, Student/t fields null with the construction reason. Do the same for the W2 seed diagnostic on **all four alias slots**, including8/11. This is an analytic description of validated aliases, not a generic zero-SE waiver or inferential pass.

Every ordinary slot calls the unchanged seed owners and retains their output or structured nested refusal reason for both primary-seed and W2-seed reports. A constant per-seed mean or a seed-diagnostic shape/SE refusal, with otherwise valid admitted inputs, primary inference and required guard evidence, **does not change top-level COMPLETE, max-p, Holm or S eligibility**. Do not drop the refusal record or manufacture its statistics. This nonfatal reporting rule does not excuse invalid input admission, undefined/zero/nonfinite primary TwoWay inference or missing/nonfinite required guard estimates/SEs; those remain fatal. The narrow owner-approved Q12-A alias amendment concerns the primary zero-variance state, not a new seed-diagnostic veto.

#### A4. Dataset and W2 guards, separate from statistical completeness

For **every slot1–12**, retain the four dataset intervals and W2 interval with raw point estimates, SEs and states. Ordinary slots, including improvement slots7/9/10/12, compute them through the unchanged E34 two-way owner at their own fixed RNG offsets. Missing/nonfinite required estimates/SEs, invalid intervals or coverage make the hypothesis and packet **INCOMPLETE** regardless of whether its point guard affects adoption. There is no blanket NOT_REQUIRED branch for improvement evidence.

Retain the booleans for every dataset Δ_d≥−.005 and **strict W2 Δ>−2SE**. A finite measured **false** point guard is a completed result, not missing evidence. Only the eligibility guards on S slots1/4 are decisive for their families' S eligibility; finite false improvement point guards, including those on slots7/9/10/12, are nondecisive for S. Numeric completeness and the decision use of a boolean are separate.

For validated alias slots2/5/8/11 under approved Q12-A, require the same constituent/draw admission as A1/A2 and retain explicit analytic/degenerate dataset and W2 records: delta=0, SE=0, bounds[0,0], dataset point guards true, strict W2 point guard false, and construction-declared seed diagnostics. Invalid alias guard inputs still make the packet INCOMPLETE. Improvement aliases8/11 retain these records as diagnostic evidence, not S adoption gates.

For score-identical R−Control (slots2/5), each dataset Δ=0 and SE=0, so the dataset point guards are true. **W2 Δ=0 and two-way SE=0, hence `0 > −2×0` is false.** Retain a complete, deterministic **W2 FAIL** record. Do not change `>` to `≥`, add an epsilon, grant bit-identity retention, or mark this false guard undefined. Its degenerate Student display may be null under the declared alias state; the strict point comparison remains well-defined and false. Construction status does not turn it into a pass. R remains diagnostic-only irrespective of M2 or Holm.

S+R−Control (slots3/6) has S's ordinary predictions, so it has the corresponding S comparison's estimands and point values. Compute and retain its intervals/p-values/guards under its own fixed RNG slot; do not alias its Monte Carlo draws to test1/4. It is **not** an identical-pair exception and still has no qualification route. Identical diagnostic improvement slots8/11 have Δ=0, so their practical point improvement `Δ>.002` is false.

#### A5. Holm and packet-level completeness

Build the ordered **12-element p-vector**: ordinary COMPLETE slots use the unchanged max-p; validated IDENTICAL_BY_CONSTRUCTION slots2/5/8/11 use1; any INCOMPLETE/INVALID slot also uses1 **but sets a fatal packet flag**. Run canonical Holm over all12 at α=.05, retain every adjusted p/rejection and state, and never remove or renumber a slot. A reserved identical slot at p=1 is not equivalent to missing evidence at p=1: the explicit pre-fit declaration and validated state distinguish them. There are no omitted arms, absent-family placeholders or post-fit reserved slots in this proposed packet.

Packet statistical state is COMPLETE_FOR_S **only if all12 records are present in the frozen order, and every record is either ordinary COMPLETE or one of the four validated IDENTICAL_BY_CONSTRUCTION records**, with all required components admitted. Any other required missing/invalid evidence, ordinary primary zero/nonfinite-SE inference, nonfinite/missing numeric guard evidence on any slot, absent slot or alias validation failure gives packet **INCOMPLETE**. Nested ordinary primary-seed and W2-seed diagnostic refusals alone do not veto completeness; finite false improvement point guards do not veto S eligibility. Even a packet-admission refusal emits all12 manifest slots as explicit INCOMPLETE/p=1 records when numerical evaluation cannot proceed, retaining the failed admission receipt; this is not manufactured test evidence. No failed input can erase a slot. The owner-approved pre-fit alias exception is the **only** difference from E34's hypothesis-completeness predicate. Reporting the alias state never grants eligibility or waiver of a failed guard.

Require complete measured gate records for all declared serving variants at their registered seed scope, including diagnostics. A completed false diagnostic gate does not block a complete S decision; a missing/invalid required diagnostic measurement still makes the packet INCOMPLETE. Explicitly defined undefined *descriptive* intervals in §4.4 remain labelled report-only outputs, not fabricated decisive evidence; that is not permission to omit required inputs or measurements.

#### A6. Verdict and selection

Q12-A is approved. After separately authorized execution, when the packet is COMPLETE_FOR_S with complete gates, **only S-A and S-C can qualify**, each against its own fresh Control. An S arm is eligible iff its eligibility Holm slot (1 or4) rejects, every dataset/W2 point guard holds, G-STEER passes and all19 engineering gates pass. Otherwise retain that family's incumbent and list measured failures. No A-versus-C winner is chosen. R and S+R are always **DIAGNOSTIC_ONLY**, never incumbents, replacements or fallbacks, even if S fails and a diagnostic variant passes every measured engineering gate. All12 statistical outputs remain reported; the diagnostic improvement tests have no adoption effect.

With any fatal packet condition, report **INCOMPLETE; no substitute verdict for either S family**. Preserve previous production state; do not relabel this “none eligible” or draw a scientific falsification conclusion. A complete packet with no eligible S retains both incumbents; a complete packet with an eligible S permits entry into full qualification only. Production adoption remains the owner's decision.

#### A7. Invented-data contract tests required before freeze

[Invented-data test specification and executable](STEER_Q12_invented_tests.py) and [revision-3 fixture results](STEER_Q12_invented_TESTS.json) exercise this state machine using explicitly invented arrays and injected primary statistics/guards/gates. They are not fits, label reads, performance measurements or validation of the Rust adapter. Fixed fixture values below are invented boundary inputs, not observed scientific results. The contract model passed41 checks; script SHA-256 `f8bf8dcb221b1a6680690a6d865e774b73e8636fe952ef306416fb84764a05fc`, receipt SHA-256 `82af95f8090b7d291f95b6e4b85dbb70a9a8f5619c48a80217413f8b790802d8`. It catches ordinary seed-only refusals as nested nonfatal reports and requires numeric guards on all12 slots. The original revision-2 files/receipt are preserved with `.rev2.bak` suffixes.

Before freeze require the separate replay through canonical statistics/verdict owners, including actual seed_stat refusal, primary/guard interval admission, canonical twelve-slot Holm and both S verdicts. [FIX2 canonical boundary receipt](steer-fix2-canonical-cases-r3.json) and [execution log](steer-fix2-check-r3.log) bind that engineering replay separately; the injected-statistic contract model must never become the scientific evaluator. The boundary receipt explicitly identifies mutations made after real bootstrap/guard computation, and real invented-prediction tests separately exercise diagnostic refusals through the complete Rust input path. These fixtures exercise the now owner-approved Q12-A rule. Their approval decision comes from the owner, not from fixture success; counterfactual unapproved-policy refusals remain negative controls. Repeat the invented-data checks before freeze against the frozen owners.

| Invented-data test | Required state and consequence |
|---|---|
| Eligible S with exact aliases | Two family fixtures, ten seeds/four sources; admit exact R=Control and S+R=S bits. Eligible-S primary p=1e−6, S guards/gates pass; negative diagnostic improvement fixtures use p=.2. These are injected statistics, not p-values calculated from the invented arrays. Slots2/5/8/11 show the exact construction label/p=1, dataset guards true and R W2 false. All12 remain. Packet COMPLETE_FOR_S; S-A/S-C eligible; R/S+R diagnostic-only. |
| Alias seed branch | Ten-by-four zero differences; the source-matching seed_stat contract double refuses (real-owner replay required before freeze). Alias adapter emits expected delta/SE0 without calling the refusing seed owner; null Student fields with reason. All four aliases retain analytic dataset/W2 evidence, including8/11; no INCOMPLETE from these declarations. |
| Ordinary seed diagnostic refusal | With valid admitted input, primary and guards, replace test1 primary-seed matrix by ten zero four-source rows; repeat for W2-seed matrix and a malformed diagnostic matrix. Each nested diagnostic exposes INCOMPLETE/reason, while hypothesis COMPLETE, primary max-p, all12 Holm slots and eligible S verdicts remain unchanged. Real-owner replay also constructs valid predictions with zero seed-only SE but nonzero primary reference uncertainty. |
| Alias integrity failures | Missing/nonfinite row, broken alias bit, source/type coverage failure or missing declaration; fatal INCOMPLETE, no exemption based on observed equality. |
| Ordinary primary/guard failures | Undefined/zero-SE primary or nonfinite/missing numeric guard evidence on any slot ⇒ INCOMPLETE. In particular test7 with invalid guard inputs, NaN W2 SE and empty dataset deltas blocks both S verdicts; each missing/nonfinite mutation is also checked alone. Finite false improvement point guards are nondecisive for S. A valid S W2 value exactly at−2SE ⇒ strict guard false, completed ineligibility; no `≥` substitution. |
| All slots and diagnostic evidence | Delete any slot, change its declared pair, or omit a required diagnostic gate ⇒ packet INCOMPLETE. A measured diagnostic gate FAIL alone leaves S decidable and R/S+R diagnostic-only. |
| No fallback and no automatic rejection | Make S fail G-STEER with diagnostic engineering gates passing: complete packet retains family Control. Set ordinary p=.2 and all guards/gates true: no eligible S. Diagnostic identity p1 never rejects. |
| Approved A and negative policy controls | The approved-A fixture keeps all twelve slots and permits eligible S only with complete required evidence. Counterfactual pending approval refuses; the rejected alternative in §9 remains a counterfactual negative control that cannot issue an S qualification verdict. Neither negative fixture is an execution-time policy choice. |
| Seed/argv mapping | Verify all40 LODO seed pairs per fitted family/weight type, fold order, i=9 modulo wrap and the three unrotated full-data pairs against §2; no labels. |

### 4.4 Misranking regret — report-only, fixed now

For **every case, arm and seed**, including controls and passing cases, record every rectangle's independent native gain t_b, l_b, q_b and ranks. Measure selections on the same frozen base; never sum them as the gain of a joint multi-block edit.

Let B be the number of case blocks; define budgets **k∈{1,min(4,B),max(1,ceil(B/4))}**, deduplicated in that order. These proposed budgets describe one edit, a small batch and a quarter of the inventory. No outcome selects the budget. Rank descending, breaking exact ties by the instrument's y-major/x-major block index, consistently for every vector. Spearman itself retains canonical average-rank ties; this tie-break applies only to deterministic selections.

For each k let T_k be the k largest **native true gains**, and Q_k the k largest **served predicted gains**. Define:

```text
A_k = Σ[b in T_k] t_b − Σ[b in Q_k] t_b
D_k = Σ[b in T_k] abs(t_b) + Σ[b in Q_k] abs(t_b)
R_k = A_k / D_k                              when D_k > 0
R_k = 0                                     when D_k = 0 and A_k = 0.
```

The oracle maximizes the sum, so A_k≥0 mathematically even with negative gains. Both sets have the same k, so this reports **forced-k singleton opportunity cost**, not a claim that an encoder should take negative-gain actions. Record counts of selected nonpositive gains. D_k=0 implies every selected gain is zero, hence A_k=0; any contradictory state is an error. For finite data, 0≤R_k≤1 analytically. Use deterministic f64 reductions; retain the unrounded signed numerator and flag floating-roundoff violations rather than silently clamping them. Nonfinite inputs yield an explicit undefined record, never an epsilon denominator or a dropped case.

Also compute the same A_k,D_k,R_k for selection by **linear l_b**, to distinguish a local-gradient misranking from the delivered map ranking. Primary report-only metric order: served R_1,R_4,R_quarter, then corresponding A values in score units, then linear R/A in the same budget order, followed by top-k overlap fraction and selected-nonpositive count. Names carry actual k after deduplication. Record B, k, selections, D and all-zero flags; provide served and raw/pre-floor versions separately, with served primary. Never mix their score units.

Aggregate each case equally; report means, type-7 medians/p95, maxima and every per-case row. Paired arm-minus-Control mean-regret contrasts use picture-clustered OneWay intervals, resampling complete source clusters with all their cases and paired variants/seeds; seeds 0–2 are averaged within each resampled case and also shown separately. Use B=10,000 and 95% percentile plus Student intervals with the inherited small-cluster correction; G=15 on the full fixed panel, smaller subset G reported exactly. Undefined intervals stay undefined with counts; do not replace them with zero. Size/content breakdowns are descriptive, without significance or adoption claims. The scalar means, percentile summaries and maxima are distinct endpoints; no post-result preferred summary.

All-block R's served regret should be zero against the same-model oracle as an **analytic checksum** when extraction/score parity holds. That is not independent evidence of perceptual quality or encoder rate-distortion improvement. Low regret cannot excuse an M2 miss. Regret has **no eligibility threshold** in this draft; adding one needs an owner decision and freeze before any fit.

### 4.5 Every other gate, unchanged

Carry the exact existing harness populations, evaluator versions, tolerances and refusal rules into the freeze manifest. Inherit [E34's engineering-gate definitions](https://github.com/imazen/zensim/blob/3af52903/benchmarks/e34_results_2026-10-15.md#registered-engineering-gates) and the [E33 output-stage contract](https://github.com/imazen/zensim/blob/81a545a3/benchmarks/e33_registration_2026-10-09.md). All are required, including after another gate fails.

| Gate | Registered rule retained |
|---|---|
| C1–C6 | C1 mono≥.93; C2 ties≤.05 on both standard/ladder grids; C3 fraction of registered negative-tail probe scores below0 must be>0; C4 probe minimum<0; C5 exactly100 on all620 identity/tier rows; C6 zero grid rows above the identity-probe maximum plus the existing0.01 slack. No replacement thresholds. |
| G-DIAL | p5≤25, p95≥85, mono≥.93 on both grids. |
| N1,N2,N3 | N1/N2≥99.0 on all24 references; N3≥122/144 ladders. |
| K1–K3 | K1 stored derivatives>0 except the identity knot (report its derivative); K2 strict dense monotonicity on the existing million-point grid, with non-increase at ULP-adjacent probes as the canonical owner requires; K3 spline(pin)=100 and inherited identity proof. |
| K4 | Zero raw forwards at/below the exact stored lower floor on every inherited population, including calibration, grids, negative/identity probes, NearID and all G-STEER base/repair forwards. |
| K5 | Repeat C1/C3/C4/C6/G-DIAL through the packed serving owner. |
| Runtime | Six v4x/v3 ×64²/256²/1024², one thread; same family-Control ordinary-score comparison. Fail when paired-delta CI is wholly above +2% of control mean, with only the existing resolution-limited exemption. |
| Spatial | v4x one-thread1024²/2048², complete cached bin-1 score/map p95≤3× the same candidate's uncached p95. R's complete exact inventory is included (§3.2). |
| Memory | Fresh-process map incremental max RSS≤128B/pixel+64MiB, at the same two geometries; existing baseline subtraction and `/usr/bin/time -v` owner. |

The existing [dial-addressability owner](https://github.com/imazen/zensim/blob/81a545a3/zensim-validate/src/dial_addressability.rs) and [packed-spline checker](https://github.com/imazen/zensim/blob/81a545a3/zensim-validate/src/dial_spline.rs) own these comparisons; freeze their hashes and complete rule payloads, not rewritten thresholds. Full-data seed0 carries the inherited engineering verdict; G-STEER also uses seeds1/2 as above. Preserve the all-population K4 accounting and native score parity for every serving variant.

Runtime and RSS run on **r7900x**, as the owner directed for E34; this host choice is fixed here before any timing. Use the canonical COSTCMP/SPEEDQ collectors, portable build, paired analyzer, load1<2 and no foreign build/training, first32 clean rounds out of at most64, shared clean-round indices and original warmup rules. No competing E35 work during those measurements. Contaminated attempts are retained and retried under the same rules. No timing is claimed for another CPU or host.

## 5. Data exposure and feasibility probes

The §0 isolation rule also applies to data preparation, pilot inspection, assessment and archiving: neither lane reads the other's scientific outputs until both closure receipts exist; telemetry alone may coordinate resources. This draft admits no new payload read. The later fit keeps E34's existing D1 human/teacher/coverage populations and their historical exposure limits; each assessment source is withheld from its LODO fit. These are already-exposed design populations, not untouched external validation. No sealed, TERMINAL, secret holdout, new external labels or unregistered development read is authorized. The coordinator writes the exact exposure-ledger entry and hashes before any later assessment payload read.

The 135 G-STEER cases retain their pinned images, source roles, repairs, order and captures; original packet SHA-256 `b2af8465520139da6ebeede71c0c73c688d0d579e52e512cbf53f8181aec5ed9`. Label-free measurement does not permit tuning on that panel.

Optional **pre-freeze, label-free engineering probes** may use E34's existing disjoint eight-origin/96-case TRAIN development panel, whose frozen binding is `ae4effb602e6e5d79a7249e73b06fb7c8b4f0040e8f172fccec92789934acf82`. Keep its source families disjoint from all135 cases. No new human labels or fits are needed: existing control weights test R's complete-delta parity and cost; analytic toy smooth models test S's derivative/identity wiring. Report that dev96 has prior development exposure and is not qualification. Do not change τ or choose a shortlist from its pass counts under this draft.

If additional pictures are necessary for geometry/cost coverage, use existing TRAIN sources only after both origin/family assignments, the existing near-duplicate audit and E34 exclusions establish disjointness. Pin the inventory and generator before probing. Synthetic geometry fixtures may test arithmetic and runtime only; they cannot establish steering quality. No case from the135 panel is a feasibility probe. Store immutable receipts: pixels/source hashes, source-family eligibility, dimensions/grid/repair inventory, model and tool hashes, parity/refusal results, per-query work, timing, peak RSS, command, date and contention. No probes were run for this draft.

## 6. Compute and serving-cost basis

All later scientific fits go through **zenfleet / fit-cell-exec**, canonical archive-bound harvest verification and native inspection. Share the fleet with E35 through the coordinator's current scheduler, with the §0 no-output-consumption boundary lasting until both close; no remembered host list or former slot count grants capacity. Keep separate experiment manifests, queues and output namespaces; no new fleet jobs are launched by this drafting task. Heavy preparation/probes must be serialized with the other local lane and use run-heavy with24GiB; workers retain measured per-cell memory/no-swap caps.

The cost basis here is **E34 receipts only**. Its r3500 first-epoch receipt is bound by [E34 launch](https://github.com/imazen/zensim/blob/bc9be24a/benchmarks/e34_frozen_launch_2026-10-15.json), SHA-256 `a4c5a7c729392812aa10a38ee58f19db38a996a19d355c1e1c09eef987d15e3f`. These matching H128 entries were read:

| E34 H128 route | Measured elapsed to epoch0 (s) | Printed epoch bound including0.1s (s) | Receipt's 120-epoch projection (s) |
|---|---:|---:|---:|
| A LODO, later smoke |18.615880603|8.4|1018.215880603|
| A full, later smoke |19.019940287|8.7|1054.319940287|
| C LODO, later smoke |24.586245761|13.6|1642.986245761|
| C full, later smoke |24.543119308|13.4|1619.143119308|

These are the larger of the matching original/later H128 route projections, not full-run measured durations or averages across folds. Formula: elapsed_to_first_epoch+119×(printed_epoch_seconds+0.1). E34's common **4929s** cap and **6GiB/no-swap** worker limit came from all14 routes; observed cgroup memory peak across that receipt was **2,832,580,608bytes**. These are historical E34 observations, not measured S requirements or `/usr/bin/time` claims for this experiment.

Illustrative arithmetic only: pricing both S and Control at those ReLU route projections gives
`2×[40×1018.215880603+3×1054.319940287+40×1642.986245761+3×1619.143119308]/3600 = 63.5935968 cell-hours`.
**This is an unmeasured planning proxy**, extrapolated from E34 first-epoch receipts, not a forecast, bound or measurement of172 full STEER fits. S's overhead, fold effects, setup, shared-fleet contention, qualification and R's repair extraction are unmeasured. No wall-time estimate follows without measured capacity. Replace this proxy with new bounded smokes before freeze; do not reuse E34's caps as proof that S will finish.

| Mechanism | Analytical serving work; no measured multiplier | Gate threatened |
|---|---|---|
| S | Same matrix dimensions; replace128 ReLU evaluations by128 piecewise polynomial activations per hidden forward, plus corresponding exact derivative work. Extra branches/multiplies; no exponentials. | Ordinary runtime especially64²; spatial/RSS if derivative wiring allocates; accuracy/near-identity because curvature changes. |
| R | For total inventory B_total=Σ[b∈{8,16,32,64}]ceil(w/b)ceil(h/b), sequentially obtain a complete repaired row and one score forward per candidate, plus the base map. Cost expression `C_map + Σ(C_complete_repair_features + C_loaded_forward)`; reference reuse and duplicate sharing must be measured. | Spatial≤3× and map RSS most directly; ordinary-score runtime if the opt-in branch leaks overhead. No claimed ms/query. |
| S+R | R's inventory and feature work with S's forward; zero new fits. | Same spatial/RSS threats, plus S's ordinary-score runtime. |

Do not infer milliseconds for S or R from E34's ordinary-score timing: repaired-feature work is a different operation. [E34's measured r7900x spatial ratios](https://github.com/imazen/zensim/blob/3af52903/benchmarks/e34_results_2026-10-15.md#registered-engineering-gates) already failed four arms at3.03–3.13×; W-C passed2.83×/2.72×. This motivates a cost risk, not a prediction for either incumbent or new mechanism. Its exact [resource receipts](https://github.com/imazen/zensim/blob/3af52903/benchmarks/e34_report_only_2026-10-15.md#execution-resources) include steering6170s/0.30GiB peak-RSS, development3240s/0.30GiB, statistics6775s/12.89GiB; these are whole heterogeneous E34 tasks under their recorded loads, not per-model tariffs or r7900x quiet timing.

## 7. Falsification and adoption limits

- **S:** after option A approval and a complete packet, any failed accuracy, steering or required engineering gate makes S ineligible. If S keeps accuracy but misses M2/counts, the fixed τ/H128 intervention did not solve steering. If it passes steering but loses accuracy, the joint hypothesis fails. Neither result excludes every smooth activation.
- **R:** diagnostic-only by owner Q1. Ordinary scores identical to Control give W2 Δ=SE=0 and the strict guard **false**, independently of legacy M2. Q12-A gives its declared primary comparison p=1, never an inferential pass; Delivered gains that disagree with independently extracted complete native gains falsify exactness. Exactness with a failed cost gate shows this implementation misses the registered serving budget. Perfect finite ranking is not a pass of linear M2 or a route to qualification.
- **S+R:** diagnostic-only by owner Q1, even if every measured gate passes and S fails. Report exactness, regret, cost and legacy M2 on S's weights; it cannot replace S or an incumbent. The S+R−S identity comparison is p=1 under approved Q12-A (saved 2026-10-16 at 14:41 MDT) and never exceeds the practical improvement bar.
- **Joint qualification conclusion under A:** if the complete packet has neither S family eligible, neither S enters qualification; retain both incumbents. Report failure components, not a universal claim about smooth networks or re-scoring. Missing/invalid required evidence means INCOMPLETE, not empirical falsification.

Misranking regret remains descriptive. No regret result changes a gate. There is no result-selected shortlist, activation width, seed, gate meaning or case exclusion. This experiment does not establish human perception, encoder bitrate gains or full product qualification. Adoption and any later combination with E35 remain the owner's decision, after both experiments close and a separate transfer registration.

## 8. Preparation and freeze checklist for the coordinator

1. Q1–Q3 are fixed as recorded in §9. Q12 option A is approved in writing, saved by the owner on 2026-10-16 at 14:41 MDT. Bind the approved alias/completeness policy and all inherited gate contracts.
2. Extend the canonical trainer/pack/zenpredict/gradient/reference owners for S and canonical prepared-steering owner for R. The current `panel_e34` explicitly validates nine named tests and its pack reporter expects21 models: add a separately versioned STEER inventory/schema within those existing owners for the declared12 tests and12 distinct full-data models, preserving E34's frozen mode and fixtures. Implement the full Q12-approved state machine, including the pre-fit declaration/admission branch before seed_stat, structured nonfatal ordinary primary-seed/W2-seed diagnostic refusals, fatal primary inference failures and numeric dataset/W2 completeness on all12 slots, finite false improvement guards nondecisive for S, all12 Holm slots and the packet-wide verdict predicate. Port and pass §4.3-A7's invented-data cases through the real owners; contract-model test success alone is insufficient. Do not merely pass this packet to the unmodified E34 CLI or relax its inventory validation. Version the activation and serving mode. New public interfaces, if needed, require their normal owner review; no API is added by this draft.
3. Bind data/archive and route declarations, exact feature IDs and derived arithmetic, split/role records, seed streams and §2 fold order/rotation plus the fully expanded literal trainer argv table, final epoch, model format, tool/source hashes and all135 pixel/case bindings. Preserve the historical archived-encoder source-closure limitation; this design does not require new augmentation or pretend that source was recovered.
4. Complete label-free disjoint parity/feasibility probes if used. Save all receipts. If cost or implementation is infeasible, report the blocker and obtain an amended registration before proceeding; do not silently switch to approximate deltas or fewer candidates.
5. Run bounded fit-cell-exec smokes for each distinct fit path on the slowest admitted class, including exact final-image harvest/inspection, and one full-budget matched-control replay. Measure fresh memory/time caps and scheduler placement jointly with E35. Smokes follow the preparation authorization; they are not part of this read-only draft task.
6. Freeze sources, program archive, container registry digest, data and gate packets, hypothesis/series manifests, worker argv, caps, qualification binaries/analyzers and launch-control records. Hash every binding; unresolved values are **pending**, not invented. Then the coordinator may authorize launch. No sealed/holdout access is part of this packet.
7. After every required fit is installed and archive-bound verified, write the assessment exposure ledger before payload access, compute every registered comparison/gate/report once, archive raw evidence and public-safe results with hashes, and apply the fixed verdict. Infrastructure failure reruns the identical cell; no scientific early stopping or seed replacement.

## 9. Owner decisions, including approved question12

Owner answers relayed by the coordinator, **2026-10-16,05:48–05:49 MDT**:

| Question | Approved decision | Consequence |
|---|---|---|
| Q1 | Keep legacy linear M2 decisive. | Only S may qualify; R and S+R are diagnostics, with no fallback or replacement route. R's strict W2 guard deterministically fails at Δ=SE=0 as well. |
| Q2 | All blocks on the8/16/32/64 grids. | Exact complete-feature re-scoring inventory and all costs as §3.2; no shortlist alternative. |
| Q3 | Both families; τ=1/8; fresh matched controls;172 fits; one12-test Holm family. | All86 fresh controls and86 S fits execute. No E34-control reuse; all12 slots remain, including diagnostic comparisons. |

**Question12 — option A approved, saved by the owner on 2026-10-16 at 14:41 MDT: treatment of comparisons declared identical by construction before any fit.** This explicitly approves the prospective exception to E34's packet-wide completeness rule.

- **Option A (approved; operative text in §4.3, owner answer saved 2026-10-16 at 14:41 MDT).** Only the four frozen pairwise alias slots2/5/8/11 may have state/display **“identical by construction, p = 1”**, after complete input/alias validation. Keep all12 Holm slots. Handle the expected seed_stat zero-SE refusal by the validated declaration branch, not a generic catch or zero-SE waiver. S may qualify if its own comparison/guards/gates pass and every other required record is valid. R remains diagnostic and its strict W2 failure remains false; no identity-based accuracy pass is introduced. Any other missing/invalid evidence still makes the entire packet INCOMPLETE as in E34. This is the explicitly approved prospective completeness amendment.
- **Option B — rejected alternative; E34 rule unchanged.** The known zero-SE aliases stay INCOMPLETE/p=1; the existing packet-wide completeness rule prevents a complete S verdict too. **The whole172-fit packet is diagnostic-only**, even if S's independent numbers meet every numerical bar. Keep all12 slots and fresh controls; changing scope to qualification later would require another prospective registration, not reinterpretation of these results.

Q12 option A is approved (owner answer saved 2026-10-16 at 14:41 MDT); freeze and fits still require the coordinator's separate authorization. Regret is report-only; no numerical gate threshold is changed.

## 10. Revision5 execution bindings (2026-10-16)

This appendix resolves preparation bindings only. Sections0–9 retain revision4's scientific text, numerical rules, Q12-A state machine, S-only eligibility, strict W2, mean report-only interpretation, twelve Holm slots and fresh controls. The accepted predecessor SHA-256 is `b249d1672de1d6b5fe672b9c8297e6bab07dfaa019c112340a3b217efdd3e964`. The coordinator reviews and publishes this local commit before authorizing scientific execution.

The immutable preparation root on dev is `/mnt/v/output/zensim/steer-freeze2-2026-10-16/`; NAS mirror is `/mnt/tower/output/zensim/steer-freeze2-2026-10-16/`. The [packet pointer](steer_freeze_packet_2026-10-16.pointer.md) and [launch plan](../docs/STEER_LAUNCH_PLAN_2026-10-16.md) locate full inventories and checksums. `PACKET.json` hashes every executable scientific input and control binding. Program/data digests are also carried by each canonical zenfleet manifest.

| Binding | SHA-256 |
|---|---|
| program.tar.gz | `9931ddb61eee776beb0a73c35a9e6f782c728b7701895a5da72d5321272a3e9f` |
| scientific-contract.json | `eef8e0577cb87cdf47389e46654b26e137dcd324d58c8ad52ec387d4e637bb69` |
| source-pins.json | `35779f55b4575d7bf9e652854256b55e814d8436321c360f68629d124dccf838` |
| GATE_BINDING.json | `1bec1063a098df6b0dd5cee8460d4355e094bf17ee6b5890d5de96ffd39ea9cd` |
| QUALIFICATION_BUNDLE.json | `4a168048c211409a78e95eaf575e8d8f80729c937e29d1a6909b92a97d8e154e` |
| qualification-tools.tar.gz | `95d810ceffd78d8fbc9895ca0a12b081a67a62b5e79237bad4a821dd180c0741` |
| fit-series.json | `a3e7cc6f5f14812df07c1bdd0bc5af46477bafd72b75a633acd0e62b379e77fb` |
| WORKER_BINDING.json | `c3ad24b50d2d2bb425bddc77ec49004cb96d3335cd86d5e62cbc47ecb5c9ad97` |
| STEER_JOB_PLAN.json | `d57afc55a60c720685f6c48586d567fbe684f2627506da7532805427eb458e58` |
| FINAL_CAPS.json | `de4bb3146106f8dd0ee04d7fea45300642d684a0d92a227f4ca151ff5a1a53c6` |

Scientific image: `ghcr.io/imazen/zenfleet-worker@sha256:36e86e70334698f06971d01b67383a0185430bbc8c0b8f6ed068081dbc21c0fa`. Data archive SHA-256: `9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008`. No bounded-smoke policy is present in this scientific archive. The archive uses native ReLU for Control and serialized SmoothHinge with fixed tau1/8 for S. Native inspection refuses mismatched activation. The exact mode remains `ZENSIM_STEER_EXACT=all-blocks-v1` for R/S+R and off for Control/S. Historical archived-encoder source-closure limitations remain unchanged; no new augmentation or source recovery is claimed.

`fit-series.json` contains172 complete literal trainer argv arrays plus driver argv digests, destination, exact feature IDs, initialized/sampling seeds, epoch119 selection, and the fixed fold order/rotation. The replay's actual trainer command matched its literal array exactly; other rows are prospective bindings, not executed-cell claims. Full-data model hashes are bound prospectively to canonical verified result/packed-model receipt fields; no model hash is fabricated before its fit. All86 controls are fresh. R/S+R refer to those fresh Control/S fits and add zero fits.

Full-budget admission replay: one r3500 Control-A/kadid/seed0,120 epochs×50,000 pairs, selected119, through fit-cell-exec. GNU-time fit-driver wall 1050.57s; peak RSS 1141136KiB. The separate store scope persisted its ledger/blob before the pinned-image archive/native harvest verifier admitted it. Canonical normalized weights match the historical E34 parity anchor;349 native score-bit comparisons across two procedural geometries and347 independent repairs match. This is engineering parity only; no scientific control reuse or135-case qualification is inferred. `attempt2/replay/REPLAY_RECEIPT.json` and `attempt2/replay/PARITY_PASS.json` preserve the evidence.

Final resource envelope:7663s per cell; one trainer thread;6GiB memory and6GiB memory+swap (no swap). The wall cap is the inherited E34 factor3 applied to the larger measured first-epoch full-budget cap input or full replay wall. The retained6GiB E34/smoke-tested memory floor exceeds the separately stated conservative2× observed peak rule. No numerical qualification bar changes. `FINAL_CAPS.json` records scope and measurements. A cap/environment failure stops the affected cell for diagnosis and identical replay; it does not shorten or exclude it.

`GATE_BINDING.json` binds every inherited gate population and owner, including calibration, standard/ladder grids, negative/identity probes, NearID and every G-STEER base/repair forward. `QUALIFICATION_BUNDLE.json` and `qualification-tools.tar.gz` bind all current native/statistical/runtime tools and analyzer hashes. Assessment/RSS timing remains on r7900x with the original quiet, warmup and clean-round rules. The future descriptive-report adapter must implement unchanged §4.4 before assessment payload access; fixed E34's21-model `reports` command is not a STEER adapter, and preparation does not claim that adapter implemented or executed.

`STEER_JOB_PLAN.json` fixes eight jobsets in order, with counts40,40,40,40,3,3,3,3. First release is exactly one registered cell; its worker must drain, ledger/blob/native-image verification and NAS persistence must pass before expanding the same first jobset and releasing the rest. The admission replay is separate from this inventory. `WORKER_BINDING.json` fixes fresh run/store prefixes, lease TTL14400s, stale-claim28800s, cap/pinning and claim-deadline transport. All eight scientific prefixes were observed empty, and preparation wrote none of them.

The shared E35 pool permits at most one STEER worker per active host across all jobsets. r3500 shares one6GiB slot at cell boundaries; tower's planned new pool is12GiB shared with E35, retaining12GiB and at least8 CPU threads. The final23:54Z observation found21GiB of existing configured reservations on i270; its STEER slot is held until at least one3GiB reservation and CPU capacity are released while retaining its4GiB system reserve. No running worker changes. `POOL_ALLOCATION.json` binds observations, maxima and live admission inequalities. Conditional i134/r5600g loan ends2026-10-17T22:00:00Z; neither is currently admitted. Only a trusted reachable Ubuntu host may enter, without trust changes or reboots. A deadline-aware worker refuses new claims after2026-10-17T19:47:17Z (full cap+300s reserve); shorter cells and60s external snapshots apply. r3800x stays excluded and q1705/q1706 stay untouched.

`LAUNCH_PLAN.md` binds first-artifact admission, the canonical independent-jobset supervised harvest/restart loop,60s temporary-host persistence, NAS backups, counts and a readable status path. No experiment consumes the other's models, predictions, metrics, gate outputs, checkpoints or intermediate tables until both close; scheduler telemetry is allowed. The registered assessment exposure ledger is future work and must precede any payload access.

Preparation deviation: an earlier cap-receipt projection inadvertently emitted historical E34 loss/development metrics. The event is preserved in `UNINTENDED_EXPOSURE.json`; no target value, STEER/E35 scientific output or p-value was read, and no scientific setting was selected from those metrics. Subsequent reads projected timing/budget/structural fields only.
