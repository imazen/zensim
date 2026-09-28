# What zensim Rev4 should be — design report, September 23, 2026

> **Superseded 2026-09-23.** The user judged that this brief fails its mission. Its conclusions are replaced
> by the experiment program in [REV4_EXPERIMENTS_2026-09-23.md](REV4_EXPERIMENTS_2026-09-23.md). §1, the
> evidence, remains valid background. Corrections from the user: zensim is a platform, so fast and rich variants
> are fine; a separate testing metric is fine if it earns its place; speed below 512² does not matter;
> multimetric agreement belongs in a gate, not the training signal; and CVVDP's teacher display is
> unexamined.

Written by the coordinating Opus session at the user's request, after the evidence audit for the zensim-2026
companion paper. The paper is **held until Rev4 qualifies** (user, 2026-09-23). This report is the design
brief for Rev4. It sits under the [controlling production plan](PRODUCTION_PRIORITIES_2026-09-15.md) and
extends the [unified-model proposal](PLAN_ZENSIM_UNIFIED_2026-09-23.md). All the production plan's split,
owner and gate rules still bind.

Every number below comes from a dated record, named in brackets. Records marked *lab* are unmerged work from
sibling workspaces. Records marked *prelim.* are awaiting audit.

## 0. Bottom line

1. **Rev4 is one model**: one feature contract, one head, one calibration, one score. SDR and HDR are trained
   together. The spatial map is the scalar's own attribution, and integrity is a separate flag. Every other
   named profile becomes a deprecated alias.
2. **The contract is built in, not bolted on.** The head is a *non-negative distance* model, so identity scores
   exactly 100 and nothing scores above a perfect copy. That construction has already passed the full G-ADDR
   contract (6/6) on real data [best_of_all_2026-09-06 §5.3].
3. **The binding problems are data, not architecture.** Codec-floor ordering moved only when training ladders
   reached the encoders' true floors [§5.18]. Local and cross-codec ordering need supervision that
   SSIMULACRA2 labels do not carry. Near-threshold accuracy needs human data.
4. **Prefer the simplest head that passes.** Start from a monotone *additive* non-negative form. Use a
   non-negative MLP only if a registered TRAIN screen shows it wins without losing the contract, floors or
   KonJND.
5. **Honest target.** If Rev4 ties SSIMULACRA2 on human rank while being contract-clean, deterministic, several
   times faster and much lighter on memory, that is a legitimate release. It is not "better than SSIMULACRA2",
   and the paper will not say it is.

## 1. What the evidence says

**A. Accuracy is capped by the teacher.** Most training targets are SSIMULACRA2 scores. In the latest
registered TRAIN view only 9.6% of rows come from human-labelled legs, plus 0.6% KonFiG
[DATA_SPLITS addendum 2026-09-20]. The results fit that ceiling:
- the best 944-class model ties SSIMULACRA2 on CID22 (0.8927 vs 0.8894) [ssim2_replacement_bar_2026-08-31];
- the served default trails it within-image (−0.0079, CI excluding zero) [same];
- it also trails on 379,048 JPEG-AIC forced choices: SSIMULACRA2 scores 0.7302 against an observer ceiling
  of 0.7346, and B is −0.0025 [−0.0037, −0.0014] relative to it. Only cross-codec questions separate the
  metrics at all [hfhuman_2026-09-01].

Nine separate mechanisms failed to move KonJND for the fast class. The near-threshold quantity is absent from
the training signal [fastclass_distill_wave_2026-09-04].

**B. Rank correlation is blind to the defects that break codec control.** On AIC2026 ladders the fraction of
correct steps is:

| metric | fraction of correct steps |
|---|---|
| DSSIM | 0.9959 |
| D | 0.9953 |
| PSNR-Y | 0.9951 |
| SSIMULACRA2 | 0.9926 |
| B | 0.9723 |

[aic2026_agreement_2026-09-19]. An unconstrained 228-input MLP ranked with the 944 leaders but scored identity
at 90.94, put 17.1% of grid cells above identity, and failed every codec floor [best_of_all §0]. No output
spline can repair that downstream [dial_addressability_gate_2026-09-04 §10.3].

**C. Constraining the head works, but costs a little rank.** The non-negative-distance arm passes the contract
6/6 on every seed, with the grid maximum at exactly 100.000. At k=3, CID22 goes 0.8891 → 0.8800. A
within-ladder hinge recovers 67% of the monotonicity the constraint costs, and cuts seed spread 6.3×
[best_of_all §5.3, §5.7].

**D. Codec floors are a data problem.** Loss, hinge weight and architecture leave A7r failing on 5/5 codecs.
Adding floor-reaching ladders improves four of five floors; avif-svt moves +0.188. The one codec that does not
improve, avif-rav1e, is the one codec with no anchor ladders [best_of_all §5.18].

**E. Capacity trades against the product rows.** The best constrained arm, compared with shipped D:

| axis | Δ vs D | 95% CI | reading |
|---|---|---|---|
| CID22 | +0.020 | [+0.017, +0.024] | win |
| CSIQ | +0.054 | — | win |
| KonJND | −0.062 | [−0.094, −0.030] | loss |
| A7r | — | — | still fails 5/5 |

The CIs are i.i.d. pair CIs; reference-clustered CIs run about 2× wider [best_of_all §5.21;
stats notes in the paper harvest]. D, a linear model, is the only lineage that clears all five codec floors
[board_ladder_ruler_2026-09-06]. It also has the best native map association [spatial records]. Simpler
additive forms generalize the product rows better.

**F. Targeting is limited by score smoothness and by the controller.**
- The richer Rev3 model's three-shot p95 errors (JPEG 3.815, JXL 3.078, WebP 7.502) miss a bar of 1
  [PRODUCTION_PRIORITIES P1].
- With the same controller steering on SSIMULACRA2, no loop target meets the bar either. SSIMULACRA2 lands
  closer than the zensim models in 11 of 12 matched comparisons, though each rests on only 3–11 requests
  [peer-target run, prelim.].

**G. Peers show cheap wins and clear limits.**
- **GMSD** runs in about 2 ms per megapixel and beats fast-ssim2 on KADID SELECT (0.8476 vs 0.8079) and KonFiG
  (0.7590 vs 0.7351). It fails on colour-only distortions: KADID colour shift 0.58, saturation 0.53
  [gmsd_2026-09-22, lab].
- Adding GMSD's deviation statistic to zensim as extra columns lost net −0.0039, because the width cost
  dominated [same].
- **butteraugli-max** is the strongest off-the-shelf corruption tripwire [corruption records; paper §10].
- **CVVDP's AIC-4 gap** was display geometry, not a porting error: 0.8906 at 4K vs 0.9609 at the study's
  geometry [cvvdp_aic_discrepancy_2026-09-22, lab].

**H. The runtime is a strength, with two holes.**
- At 1024², one thread: fast-ssim2 66.7 ms, B 42.4 ms, D 28.4 ms, Rev3 fast 12.7 ms median
  [speed_matrix_2026-09-18].
- Peak heap at 40 MP: zensim 1.06 GB vs fast-ssim2 6.74 GB, butteraugli 8.67 GB, IW-SSIM 5.66 GB
  [GMSD lane heaptrack, lab].
- **Hole:** B is slower than fast-ssim2 at 64² (ratio 0.87) [speed_accuracy_2026-09-18.json].
- **Hole:** FMA and unfused scores differ by up to 0.028 [paper §12].

**I. HDR needs joint training.** SDR-trained weights regress on UPIQ: BHdr 0.753, Rev3 fast 0.696, Rev3 rich
0.704 [PRODUCTION_PRIORITIES P4]. The display-matched CVVDP reads 0.8245 [paper §7].

**J. Maps are not yet a product.** No current map policy saves bytes under independent judges. Maps fail on
text and screens, and the two judges agree only about 0.37 [paper §9].

## 2. Rev4 in one paragraph

Rev3's deterministic arithmetic computes one registered set of non-negative per-scale distances on an XYB
pyramid. Each distance is exactly zero at identity. The distances are:
- SSIM-family terms with the error variance formed directly;
- gradient-similarity terms with both mean *and deviation* pooling;
- a small masking/peak set;
- the colour planes.

A monotone non-negative head maps them to one raw distance. One monotone spline, fitted after quantization,
maps that distance to the public 0–100 score. Training uses a mixture of agreement-filtered teachers, a
within-ladder hinge, floor-reaching ladders for every codec, cross-codec matched pairs, and every human leg the
data roles allow, with SDR and HDR in one mix. The spatial map is the per-location sum of the same additive
terms. A separate integrity flag reads a worst-local statistic. Anything built from the same features is
identical across thread counts, and within a stated tolerance across SIMD tiers.

## 3. Specification

### 3.1 The contract — what Rev4 guarantees, by construction where possible

| Guarantee | How |
|---|---|
| Identity scores exactly 100 at every entry point, including prepared references | Every distance is 0 at identity (Rev3 forms (a−b)² directly); the prepared reference keeps a source hash (needs D2) |
| No input scores above identity | Non-negative distances × non-negative weights; `raw ≤ pin`. `raw(identity) = pin` is verified by test, not assumed; see best_of_all §2.4 |
| Monotone: more distortion never raises the score, per feature | Monotone non-negative head |
| One public scale, one number | One spline; SDR/HDR in one mix |
| Deterministic | Bit-identical across thread counts; one measured tolerance per SIMD tier (the FMA vs unfused gap is today's baseline) |
| Integrity is not quality | Separate thresholded flag, never folded into the score |

### 3.2 Features

- **Arithmetic: Rev3 only.** The Rev1 defects (variance by subtraction, a non-local blur) stay retired.
- **Choose one registered set** by a TRAIN screen (U0): basic228-class vs Y60-class. Speed does not decide
  it; both are several times faster than fast-ssim2 at 1024². Choose by the full gate set, and prefer the
  smaller set on ties.
- **Pooling, not width.** Add deviation (standard-deviation) pooling of the similarity maps the extractor
  already computes. GMSD's lesson is that the pooling statistic carries signal cheaply; zensim's lesson is that
  extra columns cost more than they return.
- **Keep colour.** GMSD shows the failure mode of a luma-only metric.
- **Every feature is a non-negative distance** that is exactly 0 at identity. Features that cannot meet this
  are either transformed into such a distance or dropped.

### 3.3 Head

Two candidates, in order of preference. Both are in the non-negative-distance class:
1. **Monotone additive (GAM-like).** Each distance passes through its own monotone non-negative shape function
   (a small spline), and the results are summed. It is interpretable, it gives the spatial map as an exact
   attribution, it is the closest relative of D (which clears every floor and reads best on KonJND), and it
   is cheap.
2. **Non-negative MLP.** The `F_nonneg32` form, the best constrained arm on rank [best_of_all §5.12]. Use it
   only if U1 shows a rank gain with no loss on contract, floors, KonJND or maps.

Selection uses k ≥ 3 seed means and reference-clustered CIs, and never best-of-k. The registered selection rule
must see the contract. On Sept 6 it picked the control because it could not [best_of_all §5.14].

### 3.4 Training signal (the part that decides whether Rev4 can do better than a tie)

1. **Agreement-filtered teacher mixture** instead of SSIMULACRA2 alone. Candidate teachers: SSIMULACRA2,
   butteraugli (3-norm), display-matched CVVDP, and GMSD for structural distortions. Where they disagree, keep
   the pair as a rank constraint only when a registered agreement rule holds, as the two-reference inversion
   rule already does. Report the teacher share of every bake.
2. **Within-ladder hinge, weight about 0.5.** This setting buys reproducibility (seed spread 6× smaller)
   [§5.9].
3. **Floor-reaching anchor ladders for every codec**, AVIF rav1e and AVIF SVT included. This is the only lever
   that moved A7r. It is subject to the AVIF hold on new AVIF encodes (2026-09-04), if it is still in force: until it
   lifts, use only the ladder data that already exists.
4. **Cross-codec matched pairs**: the same source, different codecs, labelled by teacher agreement. These
   target the one question type where metrics separate on human forced choice.
5. **Human legs** wherever the roles allow (D3). **Squintly** near-threshold data when the study is ready.
   It is the only path to KonJND.
6. **SDR+HDR joint mix** with explicit target-scale alignment and source-family separation (P4).

### 3.5 Calibration and selection

One monotone spline, quantize-then-calibrate, TRAIN calibration families only. Selection reads **global
cross-source** correlation first (the AIC practice), then per-codec, per-source and within-image. A candidate
that wins only within-image, or only pooled, is not selected.

### 3.6 HDR

One model. The input declares transfer, primaries and peak, and the existing CMS owner converts it; a
perceptually uniform encoding handles HDR luminance. P4 comes first: shared weights. An internal head selected
by metadata is allowed only if TRAIN evidence forces it, and the user still sets one number.

### 3.7 Integrity flag

A separate small head over a worst-local statistic of the same distances, in the spirit of butteraugli-max.
Honest negatives (valid low-quality encodes) are in its TRAIN packet. The P5 activation contract applies
unchanged. It never lowers the quality score silently.

### 3.8 Maps

The map is the per-location sum of the additive terms, so the map and the scalar cannot disagree. Rev4 ships it
as a diagnostic (the heatmap). Encoder steering stays out of the release claims until the P3 gates pass: finite
response coherence, then independent-judge RD.

### 3.9 Runtime

These are proposed release bars, not measured facts:
- no slower than fast-ssim2 at any tested size, **64² included**;
- p95 at 1024², one thread, at or below today's Rev3 fast figure;
- peak heap per megapixel no worse than today;
- cold and warm latency both measured;
- no thread-scaling regression;
- the determinism gate of §3.1.

The small-image fixed cost and the serial share at 4K are the named optimization targets.

### 3.10 API and naming

- One default profile: the Rev4 model (proposed name: `ZensimProfile::Rev4`, serving as the default).
- The older profiles become deprecated aliases for one release (D1).
- `PreviewV0_2` behaviour stays available through its alias for crates.io 0.2.7 users.
- A prepared reference certifies identity (D2).
- The arithmetic revision is declared by the bake, never by an environment variable.

## 4. What Rev4 will not do

- **No 944-width features.** They tie SSIMULACRA2 and fail the speed bar.
- **No metric-derived near-threshold proxy.** It saturated on TRAIN and failed on the human ruler.
- **No CVVDP score as the training target.** That was a dead end, recorded with its display caveat.
- **No fast/rich model pair.** Speed does not require one.
- **No encoder-steering claims** until P3 passes.
- **No fitting against CID22-49, AIC or KonJND.** The CID22-B seal holds.
- **No best-of-k headlines, and no mixing eras** inside one table.

## 5. How Rev4 is judged

The paper's evaluation protocol is the release gate. Every row is run for Rev4 and, wherever it applies, for
the peers at their best configurations: display-matched CVVDP, butteraugli 3-norm and max, GMSD, IW-SSIM,
DSSIM, and our DVIFM-ish reimplementation.

- Human rank: global cross-source SROCC first, then per-codec, per-source and within-image, with
  reference-clustered CIs and k-seed means. Crop and full-resolution renderings are both reported. Forced
  choice is split by question type.
- G-ADDR C1–C6 and A7r, codec floors, two-reference inversion attribution.
- Targeting at 1/2/3 shots against the registered bar, including a peer as the loop target with the same
  controller.
- Integrity under the P5 contract.
- HDR under the P4 contract.
- Speed at every size and thread count; memory at 1, 16 and 40 MP; cold and warm latency; the determinism
  gate.
- Exposure ledger for every held-out read.

A FAIL or INCOMPLETE on any row blocks the word "qualified".

## 6. Execution

Delegated execution goes to Devin swe-2 lanes on quarantine branches that show their work. An Opus audit is
required before any value leaves quarantine. Opus spends tokens only on rulings and audits. Heavy jobs go
through the lab lock. Every step gets a preregistration, a fresh result directory, and a DONE report with the
MISSING list first.

| Step | Question | Output | Depends on |
|---|---|---|---|
| R0 | Anchor-ladder coverage | A census of floor-reaching ladders per codec, and a list of what the AVIF hold blocks | — |
| R1 | U0: basic228 vs Y60, and deviation pooling | One registered feature set | Existing Rev3 TRAIN caches |
| R2 | U1: additive vs non-negative MLP × {hinge 0.5, floor anchors, cross-codec pairs, teacher mix} | One head recipe; k=3 means; full gate set on TRAIN development families | R0, R1 |
| R3 | P4 joint SDR/HDR on the R2 recipe | One frozen composition | R2 |
| R4 | P1 controller on the frozen composition | Controller change supported by TRAIN traces | R3 |
| R5 | P5 integrity flag | Separate head, activation contract | R3 |
| R6 | P6 runtime + full qualification | The paper's evaluation, end to end | R3–R5 |

R2 is the decision point that matters. If no arm keeps the contract and the floors without losing KonJND
against D, the report's fallback is to **ship the additive arm with the KonJND limitation stated** while
Squintly data is collected. It is not to add capacity.

## 7. Risks

- **The ceiling may be the teacher's.** Without new human data Rev4 may tie SSIMULACRA2 at best. That is
  acceptable if the product rows pass, and the paper should say it plainly.
- **KonJND vs contract.** Constrained arms lost KonJND against D. The additive form may recover it; if not,
  it is a stated limitation.
- **The AVIF hold** limits floor-anchor coverage for AVIF until it lifts.
- **Shared HDR weights may fail P4.** The fallback is the internal head.
- **Delegated-work quality.** Devin output has already inverted a statistic's sign once. Nothing is used
  without an audit.

## 8. Decisions for the user

- **D1.** Deprecate the older profiles to aliases for one release. Recommended: yes.
- **D2.** Public-API change so prepared references can certify identity. Recommended: yes.
- **D3.** Which human legs may enter TRAIN beyond today's roles.
- **D5.** The AGPL headers on five files in the MIT/Apache crate.
- **D6.** Head preference on ties: additive (recommended) or non-negative MLP.
- **D7.** Whether R0 may request new AVIF anchor ladders when the AVIF hold lifts.
- **D4 is decided:** the whole paper is held for Rev4.
