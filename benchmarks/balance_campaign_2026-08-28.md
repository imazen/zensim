# 24h BALANCE CAMPAIGN — registered 2026-08-28T02:4xZ (user directive)

Directive verbatim: "work independently as a scientist for the next 24
hours, seeking truly balanced models that will work for hidden held out
sets, trying new hypothesis and thinking critically." Window: to
~2026-08-29T02:40Z. Overnight rule active (no questions 00:00-08:00 Denver;
decisions logged here, presented after).

## The critical framing (stated before any result)

1. **Selection-induced overfitting is now OUR biggest validity threat.**
   This session alone evaluated 9+ SDR seeds and 10+ HDR artifacts against
   the same axes (cid22, hfnlproxy, imazen26, nonphoto, bands, G-OUT).
   Every axis used for eligibility is EXPOSED: the winner's score on it is
   biased upward by selection. "Works on hidden held-out sets" therefore
   requires (a) a terminal panel the selection never touched, used ONCE,
   and (b) quantifying the selection bias itself.
2. **"Truly balanced" is operationalized as maximin**, not composite: the
   candidate's WORST normalized axis, not its weighted mean. Composites
   hide sacrificed axes (measured: the E.4 winner s4005P carries a
   CI-significant hfnl sacrifice its composite absorbs).

## Hypotheses (each falsifiable; frozen gates per-arm at execution)

- **H-DIAL-FLOOR**: the incumbent-family +5..+8 emission floor (LF-extreme
  collapse) is a PACK-ANCHOR property (anchor negative coverage), not a
  model property. Test: repack incumbent + s4006P + s4010P with a
  negrich-including anchor; if reach appears with rank axes unchanged
  (±0.002), the LF-extreme axis is a pack-recipe fix, not a selection axis.
- **H-ENS**: a 2-member ensemble of a cid22-best and an hfnl-best purity
  seed dominates both parents on maximin, and averaging halves the lone
  G-OUT B outlier. Test: averaged-prediction fulleval + G-OUT + paired CIs;
  ensemble class rules apply (M3a NOT COMPUTABLE, never penalized).
- **H-BAL (epoch-level balance)**: the cid22↔hfnl anti-separation across
  seeds arises because best_val epoch selection sees no HF signal; adding
  an HF val leg to the val geomean lets a SINGLE run select a balanced
  checkpoint. Test: one arm, purity recipe + tbig_hf val-weighted (0:1.0),
  vs SPH1 (train-weighted) — mechanism comparison.
- **H-MAXIMIN**: maximin-selected candidates generalize better to the
  hidden panel than composite-selected ones. Test at terminal read only.
- **H-KON**: the konjnd 3× peer outlier rate traces to the konjnd_bpg
  train leg's val-only twin (train_eq_val mismatch on JND content).
  Diagnose from per-pair structure before any arm.
- **H-HID (the hidden panel, built now, read ONCE at campaign end)**:
  KADIS-700k GPU-canonical cells NEVER in any training slice (stratified
  ~20k, disjoint source_ids from the 50k train slice), with its 7
  independent metric targets; plus the never-selected axes (sdr25, aic3,
  aic4, csiq, tid) as semi-hidden secondaries; plus external reads for HDR
  finalists. No candidate decision may touch H-HID before the terminal
  read; this registration is the enforcement.

## Standing constraints
NO ship-default flips; freezes remain proposals to the user; every arm
registered in its owning wave md before fit; every result committed; docs +
memory updated in-pass. SPH1 + GH2 (already registered, user-called)
continue and fold into the balance picture.

## H-DIAL-FLOOR: FALSIFIED (2026-08-28, no repack needed — decisive from existing data)

The premise was wrong on inspection: `anchor944_dial.parquet` ALREADY holds
negatives (min −100, 2.46% of 2,035 rows < 0, p1 −34.1). The decisive
evidence against the pack-property hypothesis: every flooring candidate's
emission floor EQUALS its bottom-knot dial value exactly (incumbent 5.4 =
knot 5.42; s4006P/s4010P 7.8 = knot 7.78) — meaning their raw outputs never
go below the bottom knot's raw x on any SDR corpus, so the below-knot linear
extrapolation is never exercised. s4005P (−5.8 < knot 5.11) does emit
below-knot raw. **The floor is a MODEL raw-range property (output-range
compression on OOD-severe content), not anchor coverage.** Refined
mechanism, recorded: the floor's VALUE sits at the pack's knot-placement
percentile (a deeper bottom knot would lower every floor), but WHETHER a
model extrapolates below is its own raw range — so below-zero reach remains
a legitimate model discriminator, and the LF-extreme axis stays in the
two-zone scorecard. Critical-thinking note: this is the campaign's own
first registered hypothesis, killed by its first measurement — as it
should be.

## H-ENS: FALSIFIED in strong form (2026-08-28)

`bake_verdict --ensemble` fullevals of three parent pairings: E1
(s4010+s4006) PASSES G-OUT v2 (B tamed: 26.8) with cid22 0.8904 — but its
HF paired-Δ vs incumbent is −0.0325 [−0.0390,−0.0260]: averaging moves a
candidate ALONG the recipe's cid22↔hfnl front, it does not escape it. E2/E3
(with s4004) inherit its hfnl:B outlier at ~36-37 (halved from 45.9, still
over the 35 bar) and FAIL. Ensembling this recipe's seeds cannot produce
the balanced candidate.

## H-CONTAM (the sharpest critical question, asked and answered): FALSIFIED

Hypothesis: the incumbent's hfnl edge is train-on-eval-family leakage (its
pre-purge tbig table holds 5 test-bucket + 12 validate-bucket ids; 2
families overlap the hfnl eval = 352/9,167 rows, 3.8%). Measured by
splitting the paired Δ: **incumbent−s4005P on family-CLEAN rows +0.0430
[+0.0363,+0.0500] — the edge is fully present where no leakage is possible**
— and smaller (+0.0197, CI incl 0) on the overlapped rows. The incumbent's
HF advantage is genuine skill, not contamination. (Also decisive a priori:
3.8% exposure cannot mechanically produce a +0.04 whole-set edge.)

Campaign state after three falsifications: the cid22↔hfnl anti-separation
is a property of the RECIPE'S function class + data mix, not of packing
(H-DIAL-FLOOR), not escapable by averaging (H-ENS), not an artifact of
incumbent leakage (H-CONTAM). Live mechanisms: SPH1 (HF signal in the
GRADIENT — training now), H-BAL (HF signal in the SELECTION only), GH2
(HDR micro-dose, queued), H-MAXIMIN + H-HID (terminal).

## H-KON: RESOLVED — the "3× peer rate" was an unaligned-instrument artifact (2026-08-28)

The konjnd OR comparison behind the recorded finding put candidates (n=504
pairs) against peers (n=1,008 pairs) — different pair sets, so the rate
comparison is invalid. CORRECTION to the G-OUT registration's konjnd
paragraph: that finding is NOT COMPARABLE as stated. What IS measured on
each model's own set: severe outliers (|z|>4) — s4005P 1, incumbent 0,
cvvdp 8, butter 26; candidates' konjnd train_eq_val=False (no memorization
flag); all severe misses across every model are UNDER-predictions in the
40-58 PJND band. No candidate-side konjnd pathology stands. Folding konjnd
into any gated scope would first require an aligned pair set for peers.

## H-MAXIMIN definition + H-SEL shrinkage instrument (registered before computing)

**Maximin score** := min over the six G-OUT axes (cid22, hfnlproxy,
imazen26, nonphoto, kadid, live) of the candidate's within-pool percentile
rank of `srocc_signed`. Pool = the 9 purity seeds + incumbent + SPH1 +
H-BAL seeds + gate-passing ensembles. konjnd excluded (H-KON: unaligned),
sdr25 excluded (comparator by standing rule). The maximin pick is compared
against the E.4 composite pick on the H-HID hidden panel at terminal read —
that comparison IS the H-MAXIMIN test.

**H-SEL selection shrinkage** := with K seeds, select the best on a random
half of cid22 pairs, measure on the other half; the mean (select-half −
measure-half) gap over 200 splits estimates the upward bias our K-seed
selection puts on the winner's reported score. Reported alongside any
winner so the board number carries its own bias estimate.

## H-SEL RESULT: selection bias ≈ ZERO for our K=9 seed selections (2026-08-28)

200 random split-halves of the aligned cid22 matrix (9 seeds × 4,292 pairs):
mean(select-half − measure-half) for the per-split winner = **−0.0002
(sd 0.0062)**. Mechanism: seed-to-seed differences (spread ~0.018) exceed
half-set sampling noise, so the same seeds win across splits — the ordering
is signal. Consequence: wave-level K≤9 selections on cid22 carry negligible
optimism, and paired-CI eligibility (±0.002 resolution) is measuring real
differences. Caveats recorded: one axis (cid22), one recipe family;
board-scale selection (300 rows) or noisier axes (per-ref hfnl) would show
more — the H-HID terminal read remains the generalization check of record.

## H-MAXIMIN lenses FROZEN + the forking-paths guard (2026-08-28, pre-terminal)

Computing the registered percentile-maximin exposed its scale-blindness
(BAL_E1 tops it while carrying a CI-significant −0.33 hfnl deficit, because
tightly-packed axes rank-count the same as chasms); z-maximin has the
mirrored defect (pool-spread-inverse weighting amplifies milli-differences
on packed axes). Rather than iterate lenses until one flatters a favorite,
the lens SET is frozen here with each lens's pick, and the sealed hidden
panel adjudicates BETWEEN lenses at terminal (which pick generalizes).
No lens edits after this commit.

| lens | pick | note |
|---|---|---|
| freeze_check E.4 (composite + M3a tie-break) | W10L9P_s4005_packed | sel_comp 0.9876 |
| wave eligibility-first (frozen two-zone + G-OUT) | **W10L9PH_s4004_packed** | sole eligible |
| percentile-maximin (as first registered) | BAL_E1 / PH_s4003 (tied 0.462) | scale-blind, kept for the record |
| z-maximin | W10L9PH_s4003_packed | spread-inverse, kept for the record |
| CI-unit maximin (gap-to-best / paired-CI halfwidth; hw: cid22 .0017, hfnl .0169, others .003 approx) | **W10L9PH_s4004_packed** | worst gap 5.1 units (kadid); SPH1 seeds sweep top-3 (5.1/5.5/8.2 vs 18.7 next) |

Whatever the lens, the SPH1 family IS the balance result; lenses only
permute within it. Distinct terminal picks to score on H-HID:
{W10L9P_s4005_packed, W10L9PH_s4004_packed, W10L9PH_s4003_packed, BAL_E1}.

Semi-hidden preview (axes never used in eligibility; |srocc| read down
columns, aic4/sdr25 stored orientations negative): PH_s4004 beats the
INCUMBENT on all six (aic3 .800, aic4 .914, csiq .944, tid .939, sdr25
.977, konjnd .501) — no hidden-axis collapse.

## MODEL-SEARCH PHASE CLOSED (2026-08-28 ~04:1xZ)

No further arms will be trained in this campaign window without the user.
Final SDR pool: 20 candidates (incumbent + 9 purity + 6 SPH1 + 3 H-BAL +
E1). Sole fully-eligible: `W10L9PH_s4004_packed`. HDR lane: incumbent
`HDR944_L1T1_s4005_hfpack` case complete (2 challengers falsified across 5
doses, teacher exonerated). Next: board regen, then the ONE-TIME H-HID
terminal read over the frozen lens picks {W10L9P_s4005_packed,
W10L9PH_s4004_packed, W10L9PH_s4003_packed, BAL_E1} + incumbent + shipped-B
reference, then the overnight decision brief.

## H-HID TERMINAL READ (touch-once, 2026-08-28T04:08:45Z) + H-MAXIMIN VERDICT

Sealed panel (20k rows, 4k never-used KADIS sources, 24 synthetic
distortion types, 6 independent metric targets; sha 5ec40732…). srocc,
quality-oriented (butter/dssim negated); MIN = worst target:

| candidate | ssim2 | butter_max | cvvdp | iwssim | dssim | zensim | **MIN** |
|---|---|---|---|---|---|---|---|
| incumbent W10L9_s4003 | **.8573** | **.6331** | .8207 | **.8696** | **.8944** | .7984 | **.6331** |
| PH_s4003 | .8514 | .6075 | .8305 | .8504 | .8894 | **.8046** | .6075 |
| BAL_E1 (true avg) | .8525 | .6072 | **.8380** | .8355 | .8882 | .7981 | .6072 |
| PH_s4004 (eligibility pick) | .8292 | .5843 | .8064 | .8224 | .8598 | .7991 | .5843 |
| s4005P (E.4 pick) | .8346 | .5754 | .8210 | .8245 | .8753 | .7811 | .5754 |

**H-MAXIMIN: CONFIRMED in the lens comparison.** The percentile-maximin
picks (E1 + PH_s4003) are exactly the two best off-distribution NEW models;
the composite lens's pick (s4005P) is LAST on hidden MIN; the
eligibility-first pick (PH_s4004) second-to-last. Maximin-style selection
out-generalized composite selection.

**The deeper finding (the campaign's headline): in-domain balance and
off-distribution robustness are DIFFERENT AXES.** The incumbent — trained
on 15k more (pre-family) rows with no HF leg — wins 5/6 hidden targets and
the MIN; the codec-domain balance champion (PH_s4004) gives up 0.049 hidden
MIN for its in-domain domination; PH_s4003/E1 sit between, keeping ~90% of
the HF gain at a 0.026 hidden-MIN cost. On the most perceptual hidden
target (cvvdp) the ordering INVERTS: E1 > PH_s4003 > … > incumbent >
PH_s4004 — the incumbent's MIN edge rides on butteraugli-max, the
outlier-sensitive fidelity norm.

**Stated limitations (before anyone over-reads):** hidden targets are
METRIC scores (no human labels) and KADIS distortions are synthetic — this
panel measures distribution-shift robustness, NOT product-domain (codec)
skill, which the board's imazen26/nonphoto/hfnl axes measure and where the
SPH1 family dominates. Both readings are true simultaneously; they answer
different questions.

## HYPOTHESIS LEDGER (one-glance; details in sections + wave mds)

| hypothesis | verdict | one-line evidence |
|---|---|---|
| H-DIAL-FLOOR (floor = pack anchor) | FALSIFIED | anchor already −100; floors == bottom-knot dial exactly; model raw-range property |
| H-ENS (ensembles escape the front) | FALSIFIED | E1 slides along it (hfnl −0.0325 CI-sig); s4004-parented fail B |
| H-CONTAM (incumbent HF edge = leakage) | FALSIFIED | edge +0.0430 CI-sig on family-CLEAN rows |
| H-KON (candidate konjnd pathology) | ARTIFACT | 504-vs-1008 unaligned sets; own-set severe: candidates cleaner |
| H-SEL (selection bias inflates winners) | ≈ZERO | −0.0002 ± 0.0062 over 200 split-halves, K=9 |
| SPH1 (HF signal in the GRADIENT) | **CONFIRMED, step change** | hfnl 0.73-0.76 seed-consistent (3×), cid22 +0.006 on the eligible seed; PH_s4004 sole full-eligibility pass of 20 |
| H-BAL (HF in SELECTION only) | REFUTED as remedy | recovers ~25% of HF while cratering im26 to 0.87; M3a 0.93 side-finding |
| GH1/GH2 (HDR era-B admixture, 4 doses) | NO DISPLACEMENT | HF +0.012..+0.028 real at every dose; jxl mono broken at every dose; teachers exonerated (0.998) |
| H-MAXIMIN (maximin generalizes better) | **CONFIRMED (lens form)** | maximin picks = best hidden new models; composite pick LAST on hidden MIN |
| H-HID (hidden panel adjudicates) | READ ONCE | incumbent MIN .6331 > PH_s4003 .6075 ≈ E1 .6072 > PH_s4004 .5843 > s4005P .5754; cvvdp ordering inverts |

## CAMPAIGN SYNTHESIS — the three-way tension, measured

1. **Codec-domain balance**: `W10L9PH_s4004_packed` — sole candidate
   passing every frozen clause; dominates the incumbent on BOTH zones
   in-domain; beats it on all six never-selected semi-hidden axes.
2. **Distribution-shift robustness**: the incumbent `W10L9_s4003_packed` —
   hidden MIN 0.6331; its extra (pre-family) training breadth is the
   likely mechanism; its board reads on family-overlapped axes remain
   leakage-questionable in principle (H-CONTAM cleared hfnl specifically).
3. **The compromise**: `W10L9PH_s4003_packed` (single) or `BAL_E1`
   (ensemble) — ~90% of the HF gain, best-new hidden robustness, best
   hidden cvvdp; PH_s4003's in-domain cid22 −0.0033 CI-sig vs incumbent is
   its recorded cost.

Registered follow-up PROPOSAL (not run — model-search closed): "SPH1-broad"
— the SPH1 recipe on the UN-purged 208k tbig + HF leg, testing whether
training breadth restores hidden robustness while the HF leg keeps the
balance; its board axes would need family-leakage discounting, which is the
trade the user weighs. All freeze decisions remain the user's; presented
after 08:00 Denver per the overnight rule.

## TERMINAL-READ ADDENDUM (post-hoc decomposition of the SEEN panel — labeled as such; no new selection event)

Per-distortion-type decomposition of the already-read hidden panel answers
WHERE the incumbent's MIN edge lives, and it reverses the product reading:

**Codec-like subset** (compress_jpeg, compress_jp2k, pixelate,
color_quantize, blur_gauss, denoise_dncnn; n=2,495 hidden rows —
the distortion families the steering product actually encounters):

| model | ssim2 | butter | cvvdp | MIN |
|---|---|---|---|---|
| **PH_s4004** | **0.9549** | **0.7571** | **0.9190** | **0.7571** |
| incumbent | 0.9346 | 0.7279 | 0.9166 | 0.7279 |
| PH_s4003 | 0.9324 | 0.7261 | 0.9170 | 0.7261 |

**PH_s4004 beats the incumbent on ALL THREE independent targets on
never-seen codec-like content.** Its whole-panel deficit is concentrated in
NON-codec synthetics: noneccentricity (patch-shuffle, Δbutter −0.120),
color_saturate_hsv (−0.089), color noise/shift classes — distortions no
codec emits and which the product design assigns to the corruption HEAD.
It is BETTER than the incumbent on mean_shift (+0.073), blur_lens (+0.069),
contrast (+0.051), color_block (+0.038).

**Synthesis, sharpened:** the "three-way tension" largely dissolves under
the product lens — PH_s4004 wins in-domain (board), wins hidden codec-like
(off-distribution, independent targets), and cedes only non-codec synthetic
territory that is the corruption head's by design. The incumbent's hidden
MIN edge was a non-product-distortion artifact. This decomposition is
post-hoc on a seen panel — stated openly; its claims are subgroup reads of
one sealed sample, not a new sealed test.

## PAIRED-SYSTEM CLOSURE (2026-08-28 ~05:0xZ) — the corruption axis neutralized

`bake_verdict --corruption-head corrhead944_s13.bin` joint reports, same
head, both candidates: **identical — joint pass_q10 0.9256 / pass_q20
0.7932 for BOTH** PH_s4004 and the incumbent (dial-alone 0.077-0.192; the
corruption instrument's pass decisions are head-dominated by design). The
paired-system corruption axis therefore does not discriminate the SDR
candidates, and the incumbent's hidden non-codec-synthetic edge — its last
standing advantage — is exactly the territory the (identical) head covers.

**Final SDR case, complete:** (1) paired corruption equal; (2) PH_s4004
dominates in-domain (sole eligible, both zones CI-sig, G-OUT PASS);
(3) PH_s4004 beats the incumbent on the hidden CODEC-LIKE subset on all
three independent targets; (4) the incumbent leads only on unpaired
non-codec synthetics, neutralized by (1). Recorded costs of PH_s4004,
stated plainly: M3a 0.7628 (steering-map coherence, vs incumbent 0.8626)
and whole-panel unpaired hidden MIN. All freeze decisions remain the
user's, presented after overnight.

## MORNING BRIEF (prepared 04:2xZ; fires as AskUserQuestion after 08:00 Denver)

**Q1 — SDR freeze.** Recommend `W10L9PH_s4004_packed` (sha 61ebc456…):
sole full-eligibility pass of 20; dominates the incumbent in-domain (cid22
+0.006, hfnl +0.332, both CI-sig; LF bands + corruption + reach); beats it
on the hidden CODEC-LIKE subset on all three independent targets; paired
corruption identical (head-dominated). Stated costs: M3a 0.7628 (steering
coherence, vs 0.8626) and unpaired non-codec synthetics. Alternatives:
incumbent (status quo; shift-robust unpaired), PH_s4003/E1 (compromise),
hold.

**Q2 — HDR freeze.** Recommend `HDR944_L1T1_s4005_hfpack` (sha 0a437d99…):
case complete — E.4 selection, G-HF, amended route panel (only faithful
dial), 6/6 author panel, t2 falsified (CI reversal + G-OUT + route), GH
family falsified at 4 doses with teachers exonerated. Alternative: hold for
the Krasula human study (multi-day).

**Q3 — SPH1-broad arm (optional).** The un-purged 208k tbig + HF leg:
tests whether training breadth restores the (non-codec) hidden robustness
while keeping the HF gain; its board axes would carry a family-leakage
discount. Run / skip.

Both freezes are candidate-of-record designations only — the shipped
default stays B; any default flip remains a separate proposal per the
standing rule.

## H-TRAJ (epoch-M3a trajectory) — registered 2026-08-28 ~04:3xZ BEFORE any code/fit

Motivation: the recommended SDR candidate's sole weakness is M3a 0.7628;
H-BAL showed HF-val-selected (earlier-class) checkpoints carry pool-best
M3a (0.93); SPH1 systematically lowered M3a vs purity twins. **Hypothesis
(frozen): M3a declines across training epochs (coarse-mass drift, E-M
mechanism) while rank rises; an epoch window exists with M3a ≥ 0.83 AND
≥95% of final cid22+hfnl.** If such a checkpoint exists AND passes the
frozen eligibility, it becomes a MORNING OPTION (registered as such — not a
freeze, and not a new arm: same recipe, same seed, same trajectory,
different stopping point).

Method: (1) owner feature `zensim_mlp_train --dump-checkpoints-every N`
(bakes current weights at each N-epoch boundary via the existing v3 bake
path; test included); (2) re-run the EXACT SPH1 s4004 argv+seed with dumps
every 10 epochs; the re-trained FINAL bake must byte-match the original
(a free reproducibility validation of the embedded-repro chain — a mismatch
is itself a reportable finding); (3) per checkpoint: parity pack + M3a +
cid22/hfnl rank; plot the trajectory.

## H-TRAJ RESULTS — hypothesis CONFIRMED; the M3a↔granularity trade measured (2026-08-28 ~05:1xZ)

Retrace of the exact PH_s4004 argv+seed with the new checkpoint dumps
(trainer feature + plain-lane fix, both tested):

**Reproducibility validated bit-exactly:** after stripping the
`zentrain.repro` metadata (whose argv legitimately differs by the dump
flags), the retrace and original bakes are **byte-identical** — the
embedded-repro chain reproduces weights exactly; metrics matched to 4dp on
three instruments before the strip check.

Trajectory (M3a oscillates — NOT monotone decline; best_val epochs 10/30 =
the cid22 peaks — land on M3a-LOW checkpoints, an anti-correlation at val
peaks):

| epoch | m3a | cid22 | hfnl |
|---|---|---|---|
| 0 | 0.9162 | 0.8721 | 0.6029 |
| 10 | 0.7673 | 0.8958 | 0.7167 |
| 20 | 0.8314 | 0.8755 | 0.7157 |
| **30 = best_val = final** | **0.7628** | **0.8927** | **0.7524** |
| 40 | 0.7551 | 0.8850 | 0.7336 |
| 50 | 0.8349 | 0.8785 | 0.6943 |
| **60** | **0.8333** | 0.8869 | 0.7273 |
| 70 | 0.8541 | 0.8735 | 0.7208 |
| 80 | 0.8412 | 0.8781 | 0.7222 |

**The registered window EXISTS (hypothesis CONFIRMED)**: e060 and e070
clear M3a ≥0.83 at ≥95% of final rank. Full frozen eligibility on both
(harvested, G-OUT, paired CIs): **e060 passes every clause EXCEPT (iv)
HF dial tied-rate 0.37 vs bar 0.335** (the final: 0.228) — with hfnl
+0.3069 BETTER, cid22 +0.0002 NOT-WORSE, G-OUT PASS, konjnd 0.540 (best
measured), M3a 0.8333. e070: cid22 −0.0132 SACRIFICED → out. **The M3a
recovery is real and costs top-zone dial granularity** — the two are the
same coarse-structure coin. PH_s4004 (final) remains the sole
full-eligibility candidate; e060 is the quantified M3a-recovery option
whose clause-(iv) relaxation only the user can grant.

## GRANULARITY GOALS RATIFIED + CLAUSE-(iv) AMENDMENT (user: "yes, amend the gate to span+reach+mono", 2026-08-28 ~05:3xZ)

The user's framing — "granularity matters only up to a useful point" — is
now the recorded design position, with these ratified goals:

- **Dial**: useful quantum = 1 point (users type integers; every censused
  loop succeeds at ±2, best land ±0.3-1.5). Goals: HF-zone (q≥88) span ≥ 8
  per codec; reach to the codec's true ceiling; median step ≈ 0.5 per
  step-1 rung; ladder mono ≥ 0.93; the BINDING test is closed-loop (k3
  census median |err| ≤ 1). Tied-rate is a DIAGNOSTIC capped ~0.5 — honest
  ties exist (adjacent rungs that are perceptually identical SHOULD tie; a
  metric that always separates adjacent encoder rungs is hallucinating).
- **Score**: quantum 0.5; report to 0.1; cross-model differences < 0.25 are
  noise; bounded within the declared range (G-OUT clause D unchanged).
- **Diffmap**: consumed at codec partition sizes — smallest actionable
  8px. Goals: block-Δ rank/magnitude correct at 8-128px (M3a ≥ 0.85 gold
  at 16-128), sum-consistent with the scalar, O(1) rect query. Finer than
  8px = visualization only.

**AMENDED two-zone eligibility clause (iv) — G-GRAN**, replacing the
HF-tied bar (operationalization registered before recomputation):
per codec on the HF zone (q ≥ 88): **span** p50(top)−p50(bottom) ≥ 8;
**reach** p50(top) ≥ incumbent's p50(top) − 1 (non-inferiority, avoids
codec-ceiling absolutes); **mono** ladder-mono ≥ 0.93. HF tied-rate stays
REPORTED with a 0.5 diagnostic cap. Clauses (i)-(iii) and G-OUT v2
unchanged. Measured basis for the amendment: e060's spans/reach are equal
or better than the final's (jpeg 14.9 vs 10.6, webp 9.7 vs 8.4, jxl 27.2
vs 25.5) — its higher tied-rate was clustered honest ties inside a WIDER
range, the exact failure mode of tied-rate as a gate.

## FINE TRAJECTORY + SPH1-BROAD RESULTS — the selection is now unambiguous (2026-08-28 ~08:2xZ, logged overnight)

**SPH1-broad: FALSIFIED as displacement** — on its registered deciding axis
(cid22, human, fully held-out) both seeds are CI-WORSE than the incumbent
AND than PH_s4004 (br4003 −0.0071/−0.0131; br4005 −0.0024/−0.0085); br4003's
hfnl edge over PH_s4004 (+0.0184) cannot compensate under its own
registration. Training breadth does not pay on clean axes; the family
purge stands vindicated.

**Fine trajectory (every-2, 44-72):** two new near-miss candidates, each ONE
clause short — a conservation pattern across the whole family:

| candidate | cid22 Δinc | hfnl Δinc | m3a | fails |
|---|---|---|---|---|
| **PH_s4004 final** | **+0.0060 BETTER** | **+0.3320** | 0.7628 | — (ELIGIBLE, 8/8 floors) |
| **e060** | +0.0002 NOT-WORSE | +0.3069 | **0.8333** | — (ELIGIBLE, 7/8 floors; no embedded repro) |
| f054 | **+0.0068 BETTER (family best)** | +0.3115 | 0.8238 | tid bot-band 0.828 < 0.847 |
| f058 | +0.0057 BETTER | +0.3250 | 0.8256 | G-GRAN webp (span 7.7, reach 88.9) |
| e070 | — | — | 0.8541 | cid22 −0.0132 SACRIFICED |

Every M3a-recovering checkpoint surrenders a different clause — coarse-map
coherence trades against SOME fine-structure axis every time, but WHICH
axis varies by epoch. Under the user-ratified gates (incl. G-GRAN) the
eligible set is **{PH_s4004 final, e060}**, and the frozen selection rule
(E.4 floors PRIMARY) picks **W10L9PH_s4004_packed**. e060 stays the
documented M3a-alternative: choosing it requires the user to accept 7/8
floors (and its missing below-zero reach), which only they can do.

**Packaging gap found by freeze_check:** checkpoint dumps carry no embedded
`zentrain.repro` (the trainer embeds only the final). Fix queued at the
owner (trainer post-train append to dumps) so checkpoint candidates are
freeze-packagable. Also flagged, not edited: freeze_check's bar table still
carries the old dial-tied ≤5% row, inconsistent with the ratified
granularity goals — amending those bars (zenpapers plan §5 lineage) is a
user call.

## CHECKPOINT REPRO PACKAGING — gap closed at the owner (2026-08-28, overnight)

Two owner extensions, both tested: (1) the trainer now stamps every
checkpoint dump with the run's `zentrain.repro` (+`checkpoint_epoch`)
post-training — idempotent, FATAL on failure like the final embed;
(2) `bake_dial_refit append-meta` (strip's inverse, zenpredict-bake
section splice) for post-hoc stamping. The four live checkpoint candidates
(e060/e070/f054/f058) are stamped as `*_stamped.bin` siblings — round-trip
BYTE-IDENTICAL (append→strip == original) and SCORE-IDENTITY EXACT
(forwards byte-equal on the 3,036-row HF wire). The harvested fullevals
keep their original (unstamped) bake shas; a freeze of a checkpoint uses
the stamped sibling and re-records its sha at packaging.

## USER-REQUESTED DECISION INSTRUMENTS (2026-08-28 ~08:5xZ): bands, translation, in-loop, timestamps

**B0-B6 low bands** (merged-decile scheme, signed, read down columns):
cid22 [..0.7] n=1775: A +0.714 / B +0.698 / inc +0.707; tid [..0.5] n=1418:
A +0.857 / B +0.851 / inc +0.857; kadid [..0.2] n=1697: A +0.492 / **B
+0.539** / inc old-scheme-excluded. A leads or ties everywhere except
kadid-severe (B +0.047).

**Translation utility** (bake units → metric units; monotone quantile map
fit on even hidden-panel rows, residuals on odd rows, in TARGET units):
all three candidates translate comparably — residual p50 ≈ 18-24% of each
target's IQR (ssim2 ~15 of IQR 67; cvvdp ~0.27-0.31 of IQR 1.45; iwssim
~0.014 of IQR 0.07); incumbent marginally best on this (synthetics) panel,
A ≥ B on 3 of 4 targets. Codec-domain translation favors A per the
codec-like subgroup ordering.

**In-loop diffmap utility (the decisive new datum)** — jxl 2/3-shot 27-cell
instrument, SAME-SUBSTRATE trio (fresh v47A control + A + B; the substrate
probe FAILED against the 2026-08-05 carried controls — the Aug-27 encoder
commits changed encode outputs, so carried rows are invalid on the current
binary and the trio is its own same-substrate block):

| arm | k2 med\|err\| (±2 hits) | k3-best med\|err\| (±2 hits) |
|---|---|---|
| v47A control | 0.832 (21/27) | 0.355 (24/27) |
| **A: PH_s4004** | 0.971 (21/27) | **0.315 (25/27) — best in trio** |
| B: e060 | 1.088 (16/27) | 0.570 (23/27) |

**A steers better than B everywhere in the loop, and beats the v47 control
at k3** — M3a coherence (B 0.833 vs A 0.763) does NOT translate into loop
utility; A's finer top-zone dial structure does. This is the product test
the granularity goals point at, and it separates the pair cleanly.
Cells: `~/tmp/jxlloop/fresh_{v47ctl,candA,candB}/fresh/target_ab_*.tsv`
(medians derived from the TSVs; abs_err column agrees).

**Board**: baked date+TIME now rendered (repro timestamp_epoch, file-mtime
fallback), regen 339 bakes gates PASS. SUBSTRATE NOTE registered below for
the 2026-08-05 loop panel.

## ★ CAMPAIGN CLOSE-OUT (2026-08-28T08:53Z): BOTH FREEZES EXECUTED BY THE USER

SDR: **W10L9PH_s4004_packed** (61ebc456…). HDR: **HDR944_L1T1_s4005_hfpack**
(0a437d99…). The 24h directive's deliverable — truly balanced models argued
to hidden held-out sets with critical instruments — is complete: 10
hypotheses resolved by measurement, 2 gates ratified by the user (G-OUT v2,
G-GRAN + granularity goals), 26 SDR candidates evaluated, every counter-
argument tested, and the freezes made on the full instrument set.

## ADVERSARIAL AUDIT (user: "be adversarial", 2026-08-28 ~09:3xZ) — why prior bakes out-score the frozen candidate on CID22, tested not narrated

The date-sorted board shows prior bakes above the frozen `W10L9PH_s4004_packed`
(0.8927) on CID22: `FS_GL2_s2503`/`R1_GL2_s2503_packed` **0.9010**,
`W10L3_s4001` 0.8998, `ens_E1_k2/k3` 0.894, `winner_dial` 0.894. Honest
admissions first: (a) these CID22 edges are REAL — CID22 is validation-only
forever, immune to every contamination class here; (b) the balance
campaign's "sole eligible of 26" was a claim about ITS registered pool, not
the 339-bake board — the CID22 leaders had never been run through the
user-ratified gates. So they were run through them tonight:

| challenger | cid22 Δinc | verdict under today's gates |
|---|---|---|
| **R1_GL2_s2503_packed** | **+0.0143 [+0.0098,+0.0186] BETTER** | G-OUT PASS, hfnl NOT-WORSE, M3a 0.849 — but **G-GRAN FAILS ALL 4 CODECS** (tops 84.8-89.2 vs bars 90.9-95.6, spans 4.7-6.1 on 3) |
| W10L3_s4001 | +0.0131 | NO output spline (raw head, emissions [−13,13] — not a dial); G-OUT FAIL kadid:R+S; hfnl 0.118 |
| ens_E1_k2/k3 | +0.007 | hfnl ≈ 0.00 (dead HF rank) — two-zone instant fail |
| winner_dial (era) | +0.001 | 372-era; im26 .824/nonph .858 (≪ .92+), composite .846 |

**The GL2 kill was earned, not assumed:** an HF-anchored re-pack (the exact
move that fixed the HDR incumbent) was tried — it reproduced the SAME
narrow spline (y-range 38.3→82.4) because **the compression is in the
MODEL's raw output**: on the near-lossless slice GL2's predictions top out
at **89.9 (p95 88.9)** vs PH_s4004's 100.0. The top dial decade does not
exist in GL2's range; near-lossless is structurally unaddressable —
failing the user's founding requirement ("the hf dial zone must be
addressible so jxl low distances can be reached"). Process honesty: my
first repack check printed a FALSE "restores addressability" verdict from
a vacuous loop (empty dial block in the light --json); caught and
corrected in-transcript before it informed anything.

**The answer to the user's question, measured:** last month's campaign
optimized CID22-max within its 5-row bar and produced rank-brilliant
models with compressed or absent dials and dead HF; this week's campaign
optimized the balanced product dial. The ~0.008 CID22 gap is the measured
price of addressability + both zones + bounded emissions + loop utility.
Additionally, the old population's im26/nonphoto/hfnl reads carry
pre-family TRAINING exposure (their prediction rescores were re-sliced,
their training was not) — inflated-suspect on exactly the axes where they
already lose; registry entry added below.

## LOOP-UTILITY PROXY (user directive) — design REGISTERED before computing (2026-08-28 ~09:5xZ)

Goal: a per-bake k2/k3 loop-utility measure for jxl AND avif that needs no
encodes and no census (substrate-consistent, cheap), measured (a) in NATIVE
bake units and (b) in WELL-TRANSLATED butter and ssim2 units, BOTH
directions. Substrate: the stored 944 dial grid (4,817 cells; jxl 33
images × 49 rungs, avif 35 × 40) joined to the peer scores measured on the
SAME cells (refmetrics dialgrid ssim2/butter TSVs).

**Frozen definition** (`scripts/v_next/loop_proxy.py` = the owner):
- Search emulation: bracketed q-bisection over the stored ladder (the
  family loop's shape), k ∈ {2,3} evaluations, emit-best on the bake's own
  score; landing rung's TRUE peer scores are then read off the same cell.
- Readings per (codec, k, t∈{70,80,88}):
  1. NATIVE: med |bake(landed) − t|, ±2 hits — comparable to the census.
  2. FWD (bake-steered, metric-judged): med |M(landed) − map_b→M(t)| in M
     units, M ∈ {ssim2, butter_max}; maps = monotone quantile maps fit on
     the joined codec-domain cells (20 anchors, enforced monotone).
  3. REV (metric target, bake-steered): t_M → map_M→b → steer → judge
     |M(landed) − t_M| — the round-trip composition, the "both directions"
     reading.
- FAIRNESS FLOORS reported beside every number: the M-steered oracle (same
  k-bisection steering on M itself) and the k∞ ladder-quantization floor —
  a bake is judged against what the ladder makes achievable, never against
  zero. A proxy table without its floors is the deceptive form; the floors
  ARE the fairness device.
- VALIDATION GATE: the proxy's k3 native ordering on jxl must reproduce
  tonight's measured census ordering (A better than e060) or the proxy is
  rejected.

### LOOP-PROXY RESULTS — validation PASSED on v2 (seeded-secant); the native-units deception MEASURED

v1 (blind bisection) FAILED the registered validation gate (B ≤ A at jxl
k3, contradicting the census) — it measured search geometry, not the bake;
recorded, rejected. v2 emulates the family loop (fixed seed: jxl d2.5,
avif q78; secant steps; emit-best): **gate PASSES** — jxl k3 native A
0.526 (93/99) < B 0.817 (75/99), matching the measured census ordering,
magnitudes in the census range.

Key readings (med |err|; full table `benchmarks/loop_proxy_2026-08-28.json`):

| bake | jxl k3 native (±2) | jxl k3 ssim2-judged (oracle) | avif k3 native | avif k3 ssim2-judged |
|---|---|---|---|---|
| A PH_s4004 | 0.526 (93/99) | 1.05 (0.41) | 0.639 | 1.06 |
| B e060 | 0.817 (75/99) | 1.50 (0.41) | 0.511 | 1.03 |
| incumbent | 0.400 (92/99) | 0.74 (0.47) | 0.537 | 1.02 |
| GL2 | **0.434 (97/99)** | **1.36** | 0.444 | **1.60** |

**The deception, measured:** GL2's native numbers are the table's best while
its translated (true-quality) errors are the worst — a compressed dial
(38→82 span) makes each native unit worth more quality, so native ±2
flatters exactly the models with the least addressable dials. **Native
loop error is NOT cross-comparable between bakes with different dial
spans; the ssim2/butter-judged columns (with their oracle floors) are the
fair reading.** Registry entry + board caption added. Rev (metric-unit
targets, round-trip translated) ≈ fwd throughout — the translation maps
compose cleanly both directions.

## PROXY SCOPE HONESTY + MAPPING OPTIMALITY (user challenge, 2026-08-28)

**What the proxy can and cannot estimate.** The loop proxy estimates
SCALAR-STEERING utility only: it selects among fixed-q ladder rungs by the
bake's scalar. Diffmap utility is ALLOCATION — bit movement within the
image producing operating points OFF the fixed-q ladder — structurally
invisible to any on-ladder simulation. The truthful instrument hierarchy:
(1) M3a = self-coherence (measured this week to NOT predict loop value);
(2) map-vs-external-judge block agreement on stored pixels (registerable
future instrument); (3) **paired redistribution-on/off encodes judged
independently — the only true measure** — run tonight as the `h3own` arm
(candidate's OWN attribution map steering real encodes vs its scalar base,
same 27-cell matrix, fresh substrate) for both freeze-decision candidates.
Board columns relabeled "(scalar)" and the caption states the boundary.

**Mapping optimality, corrected.** The v1 maps (20-bin quantile medians +
flat clamp) were the right CLASS (monotone) but not the optimal estimator.
Upgraded in `loop_proxy.py` to: **weighted PAVA isotonic regression** on
the paired cells (the optimal monotone conditional-mean estimator;
per-image 1/n weights so oversampled ladders don't dominate) with **linear
tail extrapolation** (the old flat clamp made above-anchor near-lossless
targets untranslatable — a product defect), plus a **round-trip drift
diagnostic** since the two directions are independent fits: measured
ssim2 0.27-0.45 bake-units (tight), butter up to 1.3 on avif (the
heavy-tailed scale is the loose direction — quote butter translations with
that uncertainty). Validation gate re-run under the new maps: STILL PASSES
(A 0.526 < B 0.817 jxl k3; GL2 still best-native/worst-translated).
Remaining non-optimality, stated: the maps are unconditional (content-
conditional translation would cut residuals but requires features at
target-setting time — a registered future lever, not silently added).

## DIFFMAP-UTILITY A/B (h3own, paired encodes, fresh substrate) — THE MAP REVERSES THE LOOP STORY (2026-08-28)

Own-attribution-map H3 steering vs scalar-only base, same 27 cells, both
candidates, engagement gates exact:

| candidate | arm | k2_best med\|err\| | k3_best med\|err\| | k3 dBytes |
|---|---|---|---|---|
| A PH_s4004 | base (scalar) | 0.971 | **0.343** | — |
| A PH_s4004 | h3own (map) | 0.817 | 0.404 (W/L 9/17) | −0.83% |
| B e060 | base (scalar) | 1.088 | 0.570 | — |
| B e060 | **h3own (map)** | 0.908 | **0.205 (W/L 18/8) — best measured tonight** | **−1.45%** |

**With the map IN the loop, the ordering flips: e060's map carries real
allocation value** (0.570 → 0.205, beating every same-substrate number
including A's base 0.343 and the v47 control 0.355, at −1.45% bytes),
while **A's own map makes A worse at k3** (0.343 → 0.404). The earlier
"M3a does not translate to loop utility" conclusion was true only of
SCALAR-ONLY loops — the loop trio ran base arms where the map is unused.
With map steering on, the M3a-strong checkpoint wins: **M3a's product
meaning is rehabilitated as a map-on loop-value predictor** (one candidate
pair, one codec, H3-mag arm, n=27 paired cells — scope stated).

**Freeze implication, stated plainly:** the SDR freeze of A rested partly
on "A best in-loop", which was the scalar-only reading. The product
contract steers WITH the diffmap. Under the map-on reading, B (e060) is
the better steering metric on this instrument. A remains
candidate-of-record (rank eligibility + 8/8 floors + scalar loop +
translation all still favor it); the map-on loop evidence now favors B —
this goes back to the user rather than being absorbed silently.

## SPLIT-ROLE TEST (user call) — CONFIRMED, the companion pattern works (2026-08-28)

Owner change: jxl-encoder `JXL_ZENSIM_MAP_BAKE=<path>` (fd2f4351) mounts a
second bake whose FD gradient drives the map/H3 steering while
`JXL_ZENSIM_RD_PROFILE` keeps scoring; unset = structurally identical path;
loud width/probe asserts. Cross arm run on the SAME substrate binary as the
trio (the concurrent jxl rebase was NOT rebuilt into it — substrate
integrity preserved):

| arm (jxl, k3_best, 27 cells) | med\|err\| | ±2 | dBytes |
|---|---|---|---|
| A base (scalar) | 0.343 | 25/27 | — |
| A + own map | 0.404 | 23/27 | −0.83% |
| **A + B-map (SPLIT)** | **0.300** | **25/27** | **−1.01%** |
| B + own map | 0.205 | 23/27 | −2.27% |

**Scoring with A + steering with e060's map beats both A-alone forms at
equal hit-rate with byte savings** — A's rank quality + B's map value
compose. (B+own keeps the lowest raw error at B's weaker scoring and fewer
±2.) k2 cross 0.652 also beats A-base k2 0.971. Product pattern implied:
A = candidate-of-record scorer, e060's stamped bake = the registered
STEERING-MAP COMPANION (the corruption-head pattern) — adoption is the
user's call.

## BOARD: shaping-aware scatters, hfnl scaling, knob-end default (user asks, all live)

- **Shape-normalized scatter (default ON, toggle)**: predictions mapped BY
  RANK onto the reference's own quantiles — spline/range shaping removed,
  cells visually comparable, dashed diagonal = ideal, tooltip keeps the raw
  value; ρ unchanged (rank-invariant). Raw-units mode remains for
  calibration reading.
- **hfnl scaling**: hfnl cells use tight band-quantile axis limits — the
  0.90-0.985 near-lossless band fills the plot.
- **Knob-end default exclusion**: G-GRAN semantics (q≥88 span ≥ 8 AND top
  ≥ reach−1 per codec) computed per bake at build, scoped to SDR dial
  models (peers are not dials; HDR bakes are judged on their route panel).
  Default compare set: **13** (frozen candidates + HDR family + peers +
  2 era survivors); **27 curated knob-end failers** — including the SDR
  incumbent itself (avif span) — move behind the 'curated+knobfail'
  preset. Honest note: 309 of 322 board rows fail the ratified top-zone
  goals; the goals are far stricter than the board's history.

## ERA-LESSONS REVISIT (user: "we may have forgotten lessons from previous model eras" — they were right; 2026-08-28)

Sweep of the cookbook's validated-science lessons against this campaign,
each verified, not assumed:

| era lesson | status this campaign |
|---|---|
| trainer has NO linear mode — `--n-hidden-layers 0` ignored, every bake = 128-hidden LeakyReLU MLP | **VIOLATED IN LABELS, caught tonight**: all "L0/0-hidden" descriptions of the W10L9/PH family were the additive-vs-MLP mislabel recurring; verified from bytes (667→128→1); wave mds corrected in place. No numeric result changes — only architecture descriptions. |
| MLP gradients: SIGNED fold; M2=1.0 (piecewise-linear, locally exact) | consistent — the split-role FD-gradient map rests on exactly this; h3-mag magnitude semantics per #69 |
| never fit a spline on a spline | OK — `pack` strips before refitting (re-packs of packed bakes are strip→refit, not stacking) |
| CID22 human MOS = validation-only forever | held throughout ✓ |
| `:both` loss = the dial+rank recipe; MSE-only = collapse | recipe uses :both/:rank per the embedded repro ✓ |
| hf_gain unbounded ratio + IW divergence → surgical winsor guards | present in the recipe's transform list ✓ |
| **shipped jxl loop pinned to ZensimProfile::A** (deprecated; "the pin stays until the calibration table is re-seeded") | **STILL TRUE — flagged**: every experiment mounted bakes via RD override; the DEFAULT loop still scores with old Profile A. Follow-up: re-seed ZENSIM_DISTANCE_TARGETS against the candidate-of-record, then re-pin. |
| distance/starting-q tables legacy-seeded — "re-seed before trusting convergence-pass counts" | partially addressed (jxl S4 elasticity prior, svt S1, zenjpeg zq are FRESH fits); the jxl census seed_d=2.5 constant is legacy-lineage — noted on the census records |
| QAT-era identity requirement (identity ≈ dial top, ZERO above-identity pairs) | **NOT re-verified this era — open follow-up** (needs an identity-pair scoring run per finalist; no instrument ran it this campaign) |
| two-panel (rank+dial) verdicts mandatory | held ✓ (every candidate fulleval carries both) |

Also answered here: "A" in this campaign = `W10L9PH_s4004_packed` (new, frozen
yesterday) — NOT ZensimProfile::A (deprecated v47-QAT, which indeed lost to B).

## IDENTITY CHECK (era requirement, user-ordered) — the frozen pair is PERFECT; the incumbent violates (2026-08-28)

Instrument built at the owner: `zensim-bench/examples/sdr944_extract`
(the local SDR twin of hdr944_extract — 944 Folded720Append2 features for
arbitrary PNG pairs; the fleet's zensim-foldapp2 was jobexec-only).
Identity = (ref,ref) features → each finalist's forward; above-identity =
any of the 4,817 dial-grid predictions exceeding the same image's identity.

| finalist | identity p5/p50/min | above-identity |
|---|---|---|
| **A W10L9PH_s4004_packed** | **100.00 / 100.00 / 100.00** | **0/4817** |
| **B e060 (companion)** | **100.00 / 100.00 / 100.00** | **0/4817** |
| incumbent W10L9_s4003 | 92.64 / 96.64 / 91.45 | **228/4817 VIOLATIONS** |
| HDR L1T1 hfpack (off-route grid) | 96.98 / 97.34 / 96.90 | 144/4817 (SDR grid; HDR-route identity = follow-up) |

The frozen candidate-of-record + companion PASS the era identity
requirement perfectly; the incumbent FAILS it (distorted encodes scoring
above pristine = dial-integrity violation) — retroactively strengthening
the freeze and vindicating the era-lessons revisit.

## NEXT-GEN PROGRAM (user: "proceed with everything needed to do a better job…") — registered 2026-08-28

**W11 JOINT-SELECTION WAVE (launching now, frozen):** the campaign proved
rank-best and map-best live at different checkpoints of the same run
(s4004-final vs e060) and that checkpoint dumps + per-checkpoint M3a + the
loop proxy make BOTH measurable per checkpoint. Arms: SPH1 recipe verbatim,
seeds {4012,4013,4014}, dumps every 10 (now repro-stamped by the trainer).
Selection (frozen): per seed, the eligibility-passing checkpoint (all
frozen clauses + G-OUT + G-GRAN) with the highest M3a; the top two picks
then take the h3own paired-encode A/B (own-map + split vs the frozen
pair). Goal: ONE artifact that is rank-eligible AND map-strong — or the
measured proof that the two are inherently different checkpoints (then the
split-role companion ships as the pattern).

**Loop hygiene (follow-ups from the revisit, owner = jxl-encoder):**
re-seed `ZENSIM_DISTANCE_TARGETS` against the candidate-of-record, then
re-pin the shipped loop off deprecated ZensimProfile::A; replace the
legacy census seed_d=2.5 with an S4-style fitted seed.

**PU21-SINGLE-MODEL EXPERIMENT (launching after W11, frozen):** the user's
hypothesis — if the PU-route front-end scores SDR content as well as the
SDR pipeline, one model can serve SDR+HDR. Test: stratified CID22-512
subset (~800 pairs across the MOS range) → `srgb_to_pq_png` (the
convention-matched nits mapping) → `hdr944_extract` (PQ 944 features) →
forward the HDR candidate-of-record → canonical panel vs MCOS, compared
against the SDR CoR's 0.8927 native read. Decision rule (frozen): PU-route
CID22 SROCC within 0.01 of native ⇒ register the joint single-model
training wave (SDR+HDR legs through the PU front-end); worse ⇒ two-model
architecture stands with the measurement recorded.

## METHODOLOGY AUDIT (user: eval-vs-test, bake-quantity overfitting, epoch/pairs optimality — 2026-08-28)

**1. Which split does selection actually use? — stated precisely, including
the uncomfortable part.** cid22 = the 4,292-pair validation set (validation-
only vs TRAINING by law, but REUSED across selection rounds — standard for
this project, monitored below). kadid/tid = whole corpora (train==val
memorization known; integrity guards only). imazen26 / nonphoto /
hfnlproxy = **TEST-family slices** (the D1 re-slice cut them from the
family TEST bucket) — meaning this campaign's repeated selection has been
consuming the TEST bucket while the VALIDATE bucket (661 families) sits
unused. That is a split-ladder inversion. **Fix registered and tooled:**
`build_eval_slices_944.py --split validate` (added) → build
validate-family selection slices; all FUTURE selection runs on them; the
current test-family slices RETIRE to touch-once terminal reads (the KADIS
hidden-panel discipline). Queued behind the running chain.

**2. Is bake-quantity overfitting a concern? — measured, two answers.**
Selection-noise optimism: full-pool split-half shrinkage over K=21 aligned
models (every candidate + checkpoint with per-pair data), 120 splits:
cid22 **−0.0014 ± 0.0061**, hfnl **+0.0006 ± 0.0102**, winner identity
120/120 stable on both axes — between-model differences dwarf half-set
noise, so the reported winners' scores carry negligible selection
optimism even at this pool size. BUT the second sense of overfitting —
axis EXPOSURE — is real and unmeasurable from inside: the test-family
slices and cid22 have absorbed dozens of selection looks this campaign.
The validate-ladder fix (above) + sealed terminal panels are the
structural answer; shrinkage monitoring rides every future wave.

**3. Epochs/pairs optimal? — NO, and here is the evidence.** epochs=120 /
pairs=50k are INHERITED from the W10 campaign recipe, never swept this
era. Direct evidence of epoch non-optimality: the frozen candidate's own
trajectory put best_val at **epoch 30 of 120** — three quarters of the
training budget added nothing to the selected checkpoint (harmless for
quality via best_val checkpointing; a 4× compute waste). The epoch axis is
now internally swept by W11's checkpoint dumps (every 10). The UNSWEPT
axis is pairs-per-epoch: **W12 probe registered** — 2 seeds ×
pairs ∈ {25k, 100k} with dumps, vs the 50k baseline; compare best-
checkpoint quality + wall time; queued behind PU21. Per the sweep
discipline, none of these constants may be called optimal until that grid
exists.

## VALIDATE-SLICE SELECTION RERUN (user directive) — THE FROZEN SELECTION REPLICATES (2026-08-28 ~22:45Z)

Validate-family slices built (`--split validate`, family-atomic bucket
filter + conservative sharing-family drop; imazen26 61 / nonphoto 61 /
hfnl 87 validate origins; staged root `valsel-2026-08-28/`). All six
eligibility-relevant candidates rescored on the NEVER-SELECTED data:

| candidate | v-hfnl | v-im26 | v-nonph | v-hfnl Δinc (CI) |
|---|---|---|---|---|
| **A PH_s4004** | **0.6993** | 0.9314 | 0.9280 | **+0.3654 [+0.3481,+0.3834] BETTER** |
| B e060 | 0.6624 | **0.9360** | **0.9328** | +0.3285 BETTER |
| f054 | 0.6563 | 0.9316 | 0.9281 | +0.3224 BETTER |
| s4005P | 0.2823 | 0.9316 | 0.9268 | −0.0515 SACRIFICED |
| s4010P | 0.2309 | 0.9337 | 0.9287 | −0.1028 SACRIFICED |
| incumbent | 0.3337 | 0.9238 | 0.9204 | — |

**Verdict: the frozen selection replicates on validate-family content with
a LARGER hfnl margin than on test (+0.365 vs +0.332)** — the test-slice
selection was not a fluke of exposure; every family-ordering fact
reproduces (SPH1 family dominant, HF-less seeds sacrificed, incumbent
last on all three axes). Going forward, selection runs on THESE slices;
the test-family slices are retired to touch-once terminal reads.
PU21 note: the converter's multiprocessing children die at import
(kadis-distort env guard) — being fixed; the PU21 verdict follows.

## PU21 SINGLE-MODEL EXPERIMENT — VERDICT under the frozen rule (2026-08-28 ~22:5xZ)

800 stratified CID22 pairs (human MOS), both routes on the SAME pairs
(converter fixed serial; extractor gained 16-bit PNG handling mid-run):

| route | model | CID22 SROCC (n=800) |
|---|---|---|
| native SDR (sRGB features) | SDR CoR PH_s4004 | **0.8939** |
| PU21 (sRGB→203-nit→PQ, HDR-route features) | HDR CoR L1T1_hfpack | 0.8343 |
| PU21 features → SDR CoR (cross-curiosity) | PH_s4004 | 0.8134 |

**Gap 0.0596 ≫ the frozen 0.01 rule ⇒ a single PU-front model is NOT
viable as-is; the two-model architecture stands.** Honest footnote,
recorded as a future-lever motivation (NOT pursued): 0.8343 from a model
trained exclusively on HDR-grid content — zero SDR images — through a
domain-transposed front-end is a strong transfer result; a JOINT-trained
PU-front model (SDR+HDR legs through one PU front) might close much of the
gap and would be the registered follow-up wave if the user wants the
single-model line pursued.

## W11 JOINT-SELECTION VERDICT (2026-08-28 ~23:0xZ) — no full pass; the frontier moved anyway

Full frozen battery on the shortlist (G-OUT + paired CIs on cid22 +
VALIDATE-hfnl + bands + G-GRAN):

| cell | cid22 vs A | v-hfnl vs A | m3a | fails |
|---|---|---|---|---|
| s4014_final | −0.0074 WORSE | +0.0043 NOT-WORSE | **0.8719** | tid-band 0.838; webp span 6.7/top 89.4 |
| **s4014_e050** | **+0.0041 BETTER (campaign cid22 champ, +0.0101 vs inc)** | −0.0360 WORSE | 0.8415 | tid-band 0.839; **webp top 90.57 vs 90.9 (−0.33)** |
| s4012_e080 | −0.0137 WORSE | **+0.0179 BETTER** | 0.8276 | c22-band 0.682; **webp top 90.67 (−0.23)**; has below-zero reach −12.7 |
| s4013_e060 | — | — | — | G-OUT kadid:D |

**Per the frozen W11 rule: zero cells pass full eligibility ⇒ A
(W10L9PH_s4004_packed) remains the sole full-eligibility candidate; the
h3own A/B step is not triggered** (it required eligibility-passing picks).
The conservation pattern extends to a fourth wave: every challenger beats
A on exactly one axis (e050: cid22; e080: v-hfnl; final: m3a) and pays
elsewhere. NOTED FOR THE USER, not acted on: e050/e080 miss webp reach by
0.23-0.33 dial points — under the ratified useful quantum (1 point) — so
one user-approved bar adjustment would admit them; the frozen bars stand
meanwhile. s4014_final remains the best map-strong artifact (m3a 0.872,
v-hfnl parity with A) for a possible companion-map role.

## GATE-PROVENANCE AUDIT (2026-08-28, user directive: "missing by hairs is sus")

**Provenance of every frozen bar** (audit trigger: three hairline decisions
all traced to incumbent-derived floors, and the incumbent is the certified
UNBALANCED model — LF specialist, 228/4817 identity violations):

| bar | level | provenance | ever decided by a hair? |
|---|---|---|---|
| (i)/(ii) hfnl+cid22 paired CI | non-inferiority CI | principled (paired, no point value) | no — CI form self-reports noise |
| (iii) LF floors c22 0.697 / tid 0.847 | incumbent LF band − 0.01 | incumbent-derived POINT bars | yes: e050 0.839, s4006P 0.682, f054 0.828 |
| (iv) G-GRAN span ≥8 / mono ≥0.93 | ratified granularity goals / convention | principled / arbitrary-small | no |
| (iv) G-GRAN reach = incumbent − 1 | incumbent dial curves | incumbent-derived, AND reach is a CALIBRATION property (more ≠ better) | yes: e050 webp −0.33, e080 −0.23 |
| G-OUT R (OR ≤ peer+0.005) | external peers | principled | no |
| G-OUT S cap 12 / B 35 / D span/3+105 | arbitrary tolerances | arbitrary — but only egregious cells ever hit them (s4007 −63 below floor) | no |

**The webp reach bar — empirically VINDICATED** (instrument:
`scripts/v_next/webp_ceiling_audit.py` (since superseded by `dial_range_gate.py`, which absorbs it) on the stored 944 dial grid + the
refmetrics ssim2/butteraugli sidecars). Peer truth at the top knob: webp
genuinely delivers the lowest ceiling of the four codecs (ssim2 p50 90.76
vs jpeg 93.79 / avif 95.13 / jxl 97.04; butteraugli p50 1.187 vs 0.547 /
0.259 / 0.066 — webp lossy is structurally 4:2:0). Inverting each bake's
OWN optimal-class translation map (loop_proxy.qmap) at the median top-cell
ssim2 gives the honest webp reach in model units: **91.2-92.4 across all
five maps** — the bar (91.9) sits AT the honest ceiling, not above it. The
incumbent's webp reach was CALIBRATED (+0.13 stretch, the smallest |stretch|
of the five bakes measured; its stretch problems are elsewhere — it
under-reports jxl top by 2.5). The W11 cells genuinely under-report webp's
top zone beyond their own honest values (e050 −0.92, e080 −1.33, s4014f
−1.74). **User's adjustment condition ("unless webp qualities not
addressable at top end") is NOT met — webp's top end IS addressable at the
bar; no adjustment.** Bonus finding: A under-reports webp top too (−0.91) —
top-zone webp is the whole family's soft spot, worth a targeted data leg in
W12.

**The tid-LF floor hairline — also a REAL regression, not gate noise.**
Converted the point comparison to the principled form: paired bootstrap
(B=5000, seed 11) on the tid LF band (human ≤ 0.5, n=1424 aligned pairs):
e050 − incumbent = **−0.0179 CI [−0.0303, −0.0056] — wholly below zero**;
A − incumbent = +0.0005 CI [−0.0135, +0.0151] (dead even). e050's miss is
outside noise; the floor happened to agree with the CI.

**Registered improvement (W12, user-gated):** convert the clause-(iii) LF
floors from incumbent-derived POINT bars to paired-CI non-inferiority vs
the incumbent (the same form clauses (i)/(ii) already use). This audit
shows the point bar and the CI agreed HERE, but only the CI form makes a
hairline miss self-interpreting. The G-GRAN reach bar stays point-form but
gains a documented meaning: it now has a measured honest-ceiling basis
(91.2-92.4), not just incumbent lineage.

## W11 MAP-COMPANION A/B (2026-08-28 ~23:1xZ) — e060 keeps the companion crown

s4014_final (M3a 0.8719, the strongest map-side artifact by coherence) took
the h3own instrument as A's map companion and as own-map, same substrate
binary as the trio (NOT rebuilt), 27 cells, h3-mag, k3 emit-best. All
dBytes below are vs A-base-k3-best from the trio (the product-relevant
denominator: marginal cost of adding the companion to A; the earlier
table's −1.01% used the carried incumbent-base rows as denominator — both
stated, neither wrong):

| arm | k3_best med\|err\| | ±2 | dBytes vs A base |
|---|---|---|---|
| A base (scalar) | 0.343 | 25/27 | — |
| **A + e060-map (companion-of-record candidate)** | **0.300** | **25/27** | +0.32% |
| A + s4014f-map | 0.326 | 24/27 | −0.42% |
| s4014f own | 0.477 | 26/27 | **+16.80%** |

**e060 remains the best companion** (lower error at equal hit-rate).
s4014f-as-companion is close but not better; s4014f-own over-encodes
(+16.8% bytes) — its measured top-zone under-reporting (ceiling audit:
−1.74 on webp, −2.21 on jpeg) makes the loop chase scores the model won't
emit. Science note: within the map-strong class, M3a DECOUPLES from loop
value (s4014f 0.872 > e060 0.833 in M3a, yet steers slightly worse) — M3a
predicts map-on loop value coarsely (map-strong ≫ map-weak), not finely.
Scope: one codec, one instrument, n=27. Cells:
`~/tmp/jxlloop/{cross_As4014f,own_s4014f}/` + preserved crossAB cells.
Driver fix landed with this: `TS_BD` override completes on all 7 cells-TSV
sites (the h3own/h3ownsp/gainsweep phases used to clobber the committed
2026-08-05 baseline TSV — the crossAB run had done so; restored to HEAD,
cross cells preserved).

## G-GRAN v2 BUILT — peer-anchored dial gate + committed addressability expectations (2026-08-28)

User directive executed ("do all of this + establish addressability
expectations at the ends + jxl on distance + support integer and float
knobs"). Full record: `benchmarks/dial_addressability_2026-08-28.md`
(+ `.json`); owner `scripts/v_next/dial_range_gate.py`. Headlines:
the jxl q=100−4d display map is MEASURED unfair (q88 = ssim2 75.2 for jxl
vs 84.7 jpeg — jxl gates natively on distance now); effective knob quanta
measured by encode-identity (jpeg q99..100 = ONE encode, ditto q0..10;
webp top collapses 4-way; jxl truly continuous); forced ties on duplicate
encodes no longer count against models; two-sided calibration at three
peer anchors with cap-aware top tol; gaps gated only on integer-quantum
codecs (bar 4 = 2× ratified loop tolerance); attainability proxy is the
unifying gate. Verdicts: **e050 is the best-calibrated dial** (one fail,
−1.24 avif entry); the incumbent's jxl dial is NON-MONOTONE on the native
distance ladder (0.89) — invisible to the old q-axis check; **every bake
under-reports jpeg's top zone** (in-scale, no cap excuse) = the top W12
data target with webp top; no dial gaps and no stretch anywhere on the
six finalists. Rank-side verdicts unchanged. G-GRAN v2 = registered
W12 replacement candidate, adoption user-gated.

## BOARD: gate pre-filter + gates column + 🌟 stars (2026-08-28, user directive)

The scoreboard (first table) now carries a **gates** column — glyphs in
order G/E/D/K = G-OUT v2 / two-zone eligibility (HDR bakes: HDR-lane
battery) / G-GRAN v2 dial gate (registered W12 candidate) / knob-end —
✓ pass, ✗ fail, · NOT MEASURED (absent ≠ failed, hover for
detail+source). The top control bar gains a **gate filter**: per-gate
chips + a `usable` preset (G+E+K; dial-v2 opt-in while unadopted) that
EXCLUDE measured-fail rows from the list itself before reading, and drop
them from the visible chart set; a caption states the excluded count and
that not-measured never hides a row. **🌟** marks frozen /
breakthrough / domain-star models (hover = reason): the two FROZEN
candidates (W10L9PH_s4004_packed SDR, HDR944_L1T1_s4005_hfpack HDR), the
steering-map champion (PH_s4004_e060), the campaign cid22 champ + best
G-GRAN-v2 dial (w11_s4014_e050), the map-coherence star (w11_s4014_final),
the era rank incumbent, shipped B, and the additive-class star (ADD156).
Verdict source of truth: `benchmarks/board_gates_2026-08-28.json`
(committed, append-only, measured-verdicts-only — same discipline as the
annotations registry). Regen gates PASS (the star glyph initially broke
the harness's name-prefix row matching — moved after the name).

## W12-U "LODESTAR" — unified ship candidate wave, REGISTERED BEFORE ANY FIT (2026-08-28)

User directive: "doc and further test this kind of pair, but prioritize a
unified ship candidate." Pre-launch evidence closing two open questions:
**gain sweep on the pair (anchor-lantern) is CLOSED** — k3-best med|err|:
g5 0.281 (24/27), g10 (default) 0.300 (25/27), g20 0.281 (23/27), g40
0.350 (25/27) → flat optimum 5-20, default retained; and **no existing
bake is near-unified**: clear-ember (e050, the best-calibrated dial + cid22
champ) own-map scores 0.406 (24/27, +4.1% bytes) — WORSE than plain A
scalar (0.343). M3a 0.84 again fails to translate above the ~0.83 line.
A unified candidate must be TRAINED for it.

**Recipe (frozen):** the W11J argv + `tbig_hf` (1.0) + NEW
`tbig_hf_jw.parquet` (1.0) — the jpeg+webp subset of the family-clean HF
leg (3,523 of 11,941 rows by encoded extension), doubling jpeg/webp
top-zone mass to attack the measured universal jpeg/webp top-end
under-report (dial addressability audit). Seeds {4021,4022,4023}, stems
`LSTAR_s*`, checkpoint dumps every 10, standard pack/harvest.

**Selection + gates (frozen):** per seed, checkpoints take the FULL
standing two-zone battery (clauses i-iv with G-GRAN v1 as frozen; G-GRAN
v2 REPORTED informationally) + G-OUT v2; among passers pick highest M3a;
top-2 take the own-map h3own A/B. **UNIFIED BAR: own-map k3-best
med|err| ≤ 0.300 AND ±2 ≥ 25/27** — i.e. a single artifact must MATCH the
pair it would replace. **Fallback, stated up front:** if no cell passes,
the pair (anchor-lantern: north-anchor scores + river-lantern steers)
remains the proposed configuration and the lodestar track continues in a
later wave with different levers (map-aware training loss is the next
registered lever, not this wave's).

Codenames now govern candidate references
(`benchmarks/candidate_names.json`, append-only): north-anchor = A,
river-lantern = e060, anchor-lantern = the pair, clear-ember = e050,
deep-loom = s4014_final, aurora-anchor = HDR freeze, gray-tower =
incumbent, harbor-line = shipped B, pocket-lens = ADD156, lodestar = this
track.

### W12-U wave LANDED (2026-08-29 ~02:13Z) — 3/3 seeds clean; selection sweep running

Finals (test-era fulleval reads): LSTAR_s4021 cid22 0.8930 / hfnl 0.7111 /
**m3a 0.8388**; s4022 0.8904 / 0.7291 / 0.8212; s4023 0.8759 / 0.7015 /
0.7491. Two of three finals sit in the map-strong M3a class straight from
training — consistent with the jw-leg hypothesis. Selection per the frozen
protocol via `scripts/w12u_select.sh` (checkpoint 40-80 window + finals:
pack → M3a → cid22 + VALIDATE-hfnl), then full battery on the m3a-picks
and the own-map h3own unified bar.

### W12-U LODESTAR VERDICT (2026-08-29 ~02:5xZ) — zero full passers; fallback fires; the jw lever WORKED on its axis

Frozen battery on the six M3a-ranked cells (full table:
`~/tmp/w12ubat/verdict_full.txt`, G-OUT owner copy `gout_full.txt`):

| cell | cid22 vs inc | cid22 vs A | vhfnl vs A | c22/tid bots | G-GRAN v1 | G-OUT |
|---|---|---|---|---|---|---|
| lstar4021_final | **+0.0063 ✓** | +0.0003 even | **+0.0119 CI-POSITIVE** | 0.693✗h / 0.817✗ | avif+webp span, jpeg/webp top | FAIL kadid:S 2.96 |
| lstar4021_e080 | −0.0023 ✗ | −0.0083 ✗ | **+0.0517 CI-POSITIVE** | 0.693✗ / 0.879✓ | webp span/top | FAIL kadid:S 3.69 |
| lstar4022_final | +0.0037 ✓ | −0.0023 even | **+0.0298 CI-POSITIVE** | 0.724✓ / 0.844✗h | webp span 7.8/top −1.23 | FAIL kadid:S 2.92 |
| lstar4022_e070 | −0.0079 ✗ | −0.0140 ✗ | +0.0423 | 0.690✗ / 0.841✗ | webp | FAIL kadid:S |
| lstar4022_e080 | −0.0049 ✗ | −0.0109 ✗ | −0.0559 ✗ | 0.686✗ / 0.866✓ | **PASS** | FAIL kadid:S |
| lstar4023_e070 | +0.0057 ✓ | −0.0003 even | +0.0287 | 0.708✓ / 0.761✗ | **PASS** | FAIL kadid:S |

**Zero full passers ⇒ the registered fallback fires: anchor-lantern (the
pair) remains the proposed configuration.** The h3own selection step is
void (passers only); one INFORMATIONAL own-map probe on the map-strongest
cell (lstar4021_e080, M3a 0.854): **0.477 (23/27, −4.6% bytes)** — far
from the 0.300 unified bar; M3a decoupling above ~0.83 re-confirmed a
fourth time. The unified gap is LOOP VALUE, not rank — **the next
registered lever is map-aware training loss** (as pre-registered).

**The science, honestly split:** (a) the jw data lever DELIVERED its
target — validate-hfnl is CI-BETTER than A on 5 of 6 cells (up to
+0.0517), and for the first time in the campaign a candidate holds cid22
parity with A while CI-beating it on v-hfnl (lstar4021_final +0.0119 /
+0.0003; also beats the incumbent on cid22 outright +0.0063). (b) The
recipe pays on exactly three fronts: a UNIVERSAL single-clause G-OUT fail
— kadid:S chart-z p99 2.92-3.69 vs peer bar 2.74 (every other clause on
every axis passes; doubling jpeg/webp HF mass inflates the synthetic-
distortion chart spread), LF-floor hairlines (c22bot 0.686-0.724 around
the 0.697 bar; tid mostly under 0.847), and the webp span/top shortfall
(the vindicated bars). **W12-U2 registration (next wave, frozen now): jw
leg at HALF weight (0.5) to pull the kadid spread back + a map-aware loss
arm; same battery, same unified bar.** Board: lodestar family added,
cells + G-OUT verdicts appended to the gates registry, discussion set
"2026-08-29 lodestar verdict" appended.

### W12-U2 REGISTRATION COMPLETED + LAUNCH (2026-08-29, pre-fit; user standing directive "keep launching as you learn")

Two arms, serialized, seeds {4031,4032,4033} each, checkpoint dumps every
10, standard pack/harvest, SAME battery + unified bar as W12-U:
- **Arm jw05** (stems `LSTAR2_s*`): W11J argv + tbig_hf(1.0) +
  tbig_hf_jw(**0.5**) — the registered half-weight lever against the
  kadid:S chart-spread inflation, keeping most of the proven v-hfnl gain.
- **Arm jw05+cd** (stems `LSTAR2C_s*`): jw05 + `--coarse-decay 1e-5`.
  TRANSPARENT AMENDMENT to the registered "map-aware loss arm": a true
  gradient-supervised map loss needs second-order trainer surgery (owner
  work, registered separately); this arm uses the E-M campaign's one
  measured map-ADJACENT regularizer (decoupled coarse decay — the
  "KonJND +0.15, CSIQ +0.07, ~free" keeper, mechanism-linked to the
  coarse-mass drift that degrades maps). Amendment made BEFORE any arm-2
  fit; the arm is labeled jw05+cd, never "map-aware loss", in results.

Standing board discipline (user, this directive): every wave generation
appends its review set to `benchmarks/board_discussion_sets.json` so
generations are comparable from the dropdown; a cross-wave "unified
track" set accumulates the lodestar-lineage candidates.

### W12-U2 arm-2 VACUOUS (discovered at launch, 2026-08-29 ~03:11Z)

The jw05+cd arm failed at argv construction: **the base W10L9 recipe
already carries `--coarse-decay 1e-5`** (the E-M keeper is baked into
every wave since W10L9, including all of W12-U). So "add the map-adjacent
regularizer" was a no-op duplicate — and the finding sharpens: the one
measured map-adjacent trainer knob is ALREADY ON while M3a keeps
decoupling from loop value. **The loop-value lever escalates to the true
gradient-supervised map loss (second-order trainer surgery) — registered
as owner work, not a wave arm.** Arm jw05 (LSTAR2_s4031-3) trained clean;
selection + battery proceed on it alone.

### W12-U2 VERDICT (2026-08-29 ~04:0xZ) — weight lever FALSIFIED; group-structure hypothesis registered as W12-U3

jw05 battery (6 M3a-picked cells; full tables `~/tmp/w12u2bat/`):
**kadid:S fails on all six at jw weight 0.5 exactly as at 1.0** (p99 3.02-4.08
vs 2.74; 4031_e060 adds kadid:D+live:D, 4032_e070 adds imazen26:S+nonphoto:S)
— while W11 (same recipe, NO jw group) passed G-OUT on 3 of 4 cells. So the
kadid chart-spread inflation tracks the jw group's PRESENCE, not its weight.
**v-hfnl is robust: all six CI-positive vs A (+0.0128..+0.0543)** — the
lever's benefit does not need weight 1.0. But **cid22 is CI-worse vs A on
all six** (the jw1.0 wave's three parity cells did not reproduce at 0.5),
c22bot 0.676-0.703 straddles the floor, and G-GRAN webp/jpeg tops persist.
Zero full passers → pair stands.

**W12-U3 REGISTERED (pre-fit): the group-structure hypothesis.** Mechanism:
the trainer samples pairs PER GROUP, so adding any group dilutes every
other group's draw share (kadid included) regardless of the new group's
weight — consistent with weight-invariant kadid:S. Arm (frozen):
`tbig_hf_jwfold.parquet` = tbig_hf_pure ∪ (jw rows duplicated once) as a
REPLACEMENT for the tbig_hf group (same 1.0 weight, same group count as
W11 = the G-OUT-passing structure, same jpeg/webp emphasis as jw05 by
row mass). Seeds {4041,4042,4043}, stems `LSTAR3_s*`, dumps every 10,
same battery, same unified bar, pair = fallback. Launches AFTER the avif
matrix (machine-safety serialization; the avif eval is the user's
explicit priority — "avif matters a little more than jxl").

### W12-U3 VERDICT (2026-08-29 ~04:0xZ) — group-structure hypothesis ALSO falsified; kadid:S diagnosis diffuse; lodestar track holds

jwfold finals (single group, W11's exact group count, same jpeg/webp row
emphasis): **kadid:S fails on all three at 3.69-3.89** — indistinguishable
from the separate-group runs. With weight (U2) and structure (U3) both
falsified, the inflation tracks the jw DATA CONTENT itself. Per-type
chart-z attribution (row-order type derivation, 25/25 level-blocks
verified; NORMALIZATION CAVEAT: quick res/std, not the gate owner's exact
z — relative reads only): the top-1% offenders are DIFFUSE (JPEG2000/JPEG/
blur/noise for lodestar AND for A AND the incumbent alike; no jpeg-type
concentration) — the corpus-values-conflict hypothesis is NOT confirmed;
the lodestar class fattens the whole kadid chart, mechanism open (fit-line
rotation from ssim2-teacher HF targets is the surviving candidate,
untested). **Three falsifications in one cycle → lever-guessing stops
here per the rigor discipline.** The lodestar track HOLDS pending the
registered owner work (gradient-supervised map loss); the pair
(anchor-lantern) remains the proposed configuration; W12-U3 cells carry
gates-registry entries. Finals: s4041 0.8812/0.6645/0.782, s4042
0.8839/0.7287/0.784, s4043 0.8859/0.6918/0.812.

## W13 REGISTRATION — trained maps, screen-aware (2026-08-29, pre-fit; user directive "do the right thing on trained maps and screen content, use zenanalyze as needed, prod code paths all-Rust and perf-optimal")

**Measured basis (all this session):** avif per-SB map steering fails
catastrophically on screen content and both application-rule fixes are
falsified; jxl's per-tile pair HELPS screen cells (nonphoto 1.438→1.335,
5/9 screen cells better) — the damage is the avif application layer ×
map quality, not steering per se; and **the entire map zoo was selected
on photo-only M3a fixtures** (city/dog/girl) — screen coherence was
never measured.

**Production decisions (recorded):**
- jxl ships the pair UNGATED — a screen gate would forfeit the measured
  nonphoto improvement. No change.
- avif ships mapless (settled) until a candidate passes the NEW gates.
- **The standard screen classifier for any future steering gate is
  zenanalyze `PatchFraction ≥ 0.27`** (feature id 23; AUC 0.880 on the
  219-image labeled corpus, photos p50 0.002 vs screens 0.726) —
  all-Rust, tier-1 cheap, classified ONCE per source before the loop
  (zero per-iteration cost; perf-optimal). Wired only where measurement
  says it pays.
- CLI injection (aomenc/aomdec) is census-harness-only; every shipping
  loop path is pure Rust (audited: jxl/avif/svt/jpeg loops + judges).

**New instrument: M3a-SCREEN** — `m3a_sweep.sh` gained
`ZENSIM_M3_CONTENT`/fixtures overrides (owner extension); screen fixture
set `diffmap-coherence-screen-2026-08-29` (sc_gui/sc_imessage/sc_wiki ×
{576,384,256} × q{20,50,75}, ImageMagick-jpeg distortions — recorded;
the photo instrument stays frozen). **Map selection henceforth reports
BOTH M3a-photo and M3a-screen; a companion candidate must not be
screen-incoherent.**

**W13 arms (frozen order, cheapest-discriminating-first):**
1. **Zoo mining** — score the existing checkpoint zoo on M3a-screen; any
   member coherent on BOTH instruments + rank-parity vs river-lantern is
   a companion-upgrade candidate (no training).
2. **Screen-mass training** — the W11J recipe + a screen-class HF leg
   (tbig screen-class rows; zenanalyze PatchFraction labels the corpus),
   checkpoint dumps, selection on the DUAL M3a instruments + the
   bytes-at-parity gate (the re-gated steering objective).
3. **Gradient-supervised map loss** — the registered owner surgery,
   only if 1-2 fail.

### M3a-SCREEN FIRST MEASUREMENTS (2026-08-29 ~06:4xZ) — the gap is real, universal, and text-shaped

| bake | M3a-photo (frozen instrument) | **M3a-SCREEN (new)** |
|---|---|---|
| north-anchor (Profile C) | 0.763 | **0.493** |
| river-lantern (map champion) | 0.833 | **0.518** |
| deep-loom (photo-M3a star) | 0.872 | **0.497** |

**Every map in the zoo class is ~half as coherent on screen content as on
photos, and the photo ordering FLATTENS on screen** (the 0.11 photo spread
collapses to 0.025) — photo-M3a selection carried zero screen signal.
Structure (river-lantern cells): dense TEXT is the failure mode —
sc_wiki median **0.265** (near-random attribution on text) vs
sc_imessage 0.791; small crops worst (256px 0.437 vs 576px 0.736);
worst cells all sc_wiki/imessage at q20/q75 (0.11-0.15). This measurably
explains the avif screen steering catastrophe (incoherent maps → wrong
per-SB allocation on exactly the content where allocation matters most)
and gives W13 its precise target: **text-dense content at small-to-mid
scales**. Zoo-mining (arm 1) baseline set; a companion candidate must
clear BOTH instruments per the registration.

### W13 ARM-1 (zoo mining) OUTCOME — dual-instrument selection pays IMMEDIATELY (2026-08-29 ~07:0xZ)

Zoo M3a-screen ranking (11 checkpoints, 0 failures): **gray-tower 0.632 —
the incumbent beats every purity/lodestar-era checkpoint on screen
coherence** (the purity-era data changes apparently LOST screen map
coherence — a data hint for arm 2); htraj_e070 0.601;
lstar_s4021_e080 0.566; … river-lantern sat at 0.518. **No cell clears a
photo-gold-class bar (0.85) on screen** — no ready screen-coherent map
exists; arm 2 stands. BUT the mining surfaced a companion upgrade:
**PH_s4004_e070 dominates river-lantern on BOTH instruments** (photo
0.854 vs 0.833, screen 0.601 vs 0.518; it was only ever excluded as a
SCORER — cid22 clause ii — which the companion role never gates).

**Loop A/B (same substrate, jxl 27-cell, h3-mag):**

| arm (k3_best) | med \|err\| | ±2 | nonphoto med | Δbytes vs A-scalar (all / at parity) |
|---|---|---|---|---|
| A scalar | 0.343 | 25/27 | 1.438 | — |
| A + river-lantern (e060) | 0.300 | 25/27 | 1.335 | +0.32% / −0.41% |
| **A + e070 ("amber-lantern")** | **0.256** | 24/27 | **0.635** | **−1.52% / −0.15%** |

**The best loop configuration ever measured** — median −15% vs the e060
pair, nonphoto error HALVED (the screen-coherence gain showing up exactly
where the instrument predicted), byte-negative overall and at parity;
k2 also better (0.635/21 vs 0.652/21). One ±2 cell traded (24 vs 25).
**Proposed companion-of-record: e070 = "amber-lantern"; the pair =
"anchor-amber"** — adoption remains the user's call, as does e060's
retirement. This is the dual-instrument selection thesis validated on its
first application: one zoo scan + one A/B replaced the champion.

## W-LIN REGISTRATION — linear-projection B replacement (2026-08-29, pre-fit; user directive "make a decent linear projection model given our new bars and stats to replace B" + the size directive)

**Why linear:** B (harbor-line) is a 7.3 KB linear model; its successor
should stay linear-class (size, determinism, no collapse, trivially
coherent constant-gradient maps — M3a-screen measurable). The C-class
MLPs are 149-180 KB (now behind `candidate-profiles`, default-on).

**Fit (frozen):** owner = `bake_dial_refit fit-lasso` (pure Rust; the
shipped-BHdr producer) over a COMBINED 944-regime gram: the W10L9 purity
regression legs with the recipe's train weights (safesyn 1.0,
cid22_train 1.0, tbig_200k 0.5, kadis50k 0.15, tsafesyn 0.5, ttbig 0.5,
konjnd_bpg 1.2, tbig_hf 1.0; kadid/tid excluded — rank-mode in the
recipe), target human_score, RAW feature space first (the folded regime
zeroes the winsor-prone IW pools; shaped space only if raw fails),
λ ∈ {3e-4, 1e-3, 3e-3}, anchor spline on `anchor944_dial.parquet`,
f16 pack. Gram = `linear_projections cmd_gram` at ZLIN_NFEAT=944 per leg
+ an additive weighted combine (grams are additive by construction).

**Bars (frozen; the B-replacement lane judges vs B/harbor-line, not A):**
two-zone paired CIs vs B (cid22 + validate-hfnl not-worse), the B-lane
north stars (nonphoto/imazen26 + KonJND not-worse vs B), G-OUT v2,
G-GRAN v1, packed size ≤ 12 KB, dial mono; M3a photo+screen REPORTED
(constant-gradient maps). Ship/default flip stays USER-GATED.

### W-LIN ROUND 1 (raw space) — north stars beaten at 7KB; cid22/HF short; two era-findings falsified (2026-08-29 ~14:4xZ)

Full-mix raw-944 lasso (6.6-7.1 KB, 48-194 active features):
**imazen26 0.904 / nonphoto 0.895 — BOTH ABOVE B (0.831/0.864)** at
cid22 0.755 (B 0.882), |KonJND| 0.23 (B 0.519), hfnl 0.02-0.16 (B 0.503).
Falsified in-round: (1) the 372-era "bigcodec poisons linear CID22"
finding does NOT transfer to 944 (dropping tbig/ttbig made EVERYTHING
worse — cid22 0.646-0.686, imazen26 collapsed to 0.62-0.73); (2) heavy HF
upweighting (hf 2.5) is catastrophic (cid22 0.449). The full recipe mix
is the right mix for linear too. Remaining registered lever: SHAPED space
with north-anchor's own trained 944 transforms (winsor/signed_cbrt fit on
this exact mix; the raw heavy-tail features are exactly what linear
capacity cannot absorb) — screen extracted from the frozen bake, shaped
grams rebuilding.

### W-LIN ROUND 2 (shaped) + interim verdict (2026-08-29 ~14:5xZ)

Shaped space (north-anchor's trained 944 transforms) ≈ raw: cid22 0.756
(vs raw 0.755), imazen26 0.899 — **shaping does not close the human-axis
gap**. The pattern is architectural, not mix or shaping: B itself is a
BVLS MULTI-HEAD linear (the v02_bvls lineage measured 0.824 cid22 +
0.594 KonJND at 372), not a single global projection; a single-head 944
lasso tops out ~0.75 cid22 on this mix at every λ and both spaces.

**Interim W-LIN verdict (honest):** the 7 KB single-head linear is
already a NORTH-STAR winner — imazen26 0.904 / nonphoto 0.895, both
ABOVE shipped B (0.831/0.864) at half B's size — but NOT a B replacement
on cid22 (0.755 vs 0.882), KonJND, or HF-NL. **Registered continuation
(W-LIN arm 2): the BVLS multi-head architecture at 944** — B's own class
(bounded-variable least squares, multiple heads + the dense-dial
calibration lineage), the measured 372-era path to 0.82+ cid22 with
KonJND above B. Candidates so far preserved in
`/mnt/v/output/zensim/bakes/wlin-2026-08-29/` (shas in-log); the
imazen26-leading cell (`wlin_raw_lam0.0003`, 7,051 B, 194 features) is a
potential ssim2-lane SPECIALIST even if the B-replacement needs arm 2.

### W-LIN ARM 2 RESULTS — the multi-head blend works; best linear of the era (2026-08-29 ~15:3xZ)

The Profile-B mechanism (fit-lasso `--emit-fit-npz` + `blend-heads`, built
by the SOTA-944 campaign §4) ran end-to-end at 944 after one usage bug
(shaped heads blended WITHOUT `--transforms-tsv` → raw features through
shaped weights → degenerate z-norm; fixed). Findings, in order:
- kadid/tid HELP linear cid22 at 944 (canonical-shape head 0.835 vs 0.797
  without them — another 372-era exclusion reversed);
- RD mass dilutes cid22 in a single head (full mix 0.797) but carries
  imazen26 (0.90) — the specialist-blend premise re-confirmed at 944;
- **blend(canonical-head, full-mix-head, α=0.6): cid22 0.8453 /
  imazen26 0.8734 / nonphoto-strong — imazen26 +0.043 ABOVE shipped B
  (0.831), cid22 −0.037 below (0.882), at 1.7 KB** (single collapsed
  linear layer, B's shipped scaler shape).

**Remaining gaps (honest):** KonJND 0.12 (B 0.519) and HF-NL 0.06-0.11
(B 0.503) — the near-threshold and HF axes need TARGET-ENGINEERED
specialist heads (B's 372 kon head used per-corpus ssim2-anchor targets +
sign-constrained BVLS, machinery not yet exercised at 944; grams would
need anchor-target columns). Registered as the arm-2 continuation, with a
3-head chain (or kon-target gram rebuild) as the concrete next lever.
Candidates + head npzs preserved in `bakes/wlin-2026-08-29/` (shas
in-log). No ship/default claim — B remains shipped; wlin2_a0.6 is the
era's best linear artifact and a live candidate once the kon/HF axes
close.

## COMPOSITE-SCORE AUDIT (2026-08-29, user directive: "gate composite score matters too, lmk its flaws — B and its competitors")

**The composite now joins the W-LIN gates** (composite-vs-B, full-coverage
product_composite). The audit of the metric itself, each flaw pinned to
evidence:

1. **TWO formulas share the name.** Rust `product_composite` = a
   weight-NORMALIZED mean over {cid22 1.0, imazen26 0.5, nonphoto 0.3,
   konjnd 0.2, aic3 0.1, aic4 0.05}; the board's fallback `_composite`
   (used for any row without the stored value) = an UN-normalized sum
   WITHOUT the imazen26 term. One sortable column silently mixes two
   quantities (the stats-eval review's "2 composites" finding, now pinned
   to the exact code sites).
2. **Coverage renormalization makes composites incomparable.** A bake
   verdicted on fewer corpora renormalizes the denominator — same name,
   different quantity. (This audit's own wlin rescores needed the full
   6-corpus set to be comparable at all.)
3. **Polarity-blind terms.** `term()` reads plain `r.srocc` — JND
   corpora ride the |SROCC| convention, and any future orientation drift
   on an included corpus ADDS positively (the KADID-inversion incident
   class; kadid/tid are excluded, so current exposure is latent, not
   live).
4. **Held-out and trainable-toward axes are summed together.** cid22
   (sacred human holdout) + imazen26/nonphoto (ssim2-anchored,
   training-adjacent): a model can buy composite on trainable axes; the
   scalar hides WHICH kind of skill it holds.
5. **No HF-NL term — the documented product weak zone is invisible.**
   Showcase: incumbent 0.8602 vs A 0.8645 — 0.004 apart on composite
   while A leads validate-hfnl by +0.365 and the incumbent carries 228
   identity violations and a non-monotone native-distance jxl dial.
6. **Rank-only.** No dial/calibration, mono, map-coherence, or screen
   term — every dial gate this campaign built is outside the scalar.
7. **Hand-set weights, no provenance, no CI weighting.** 1.0/0.5/0.3/
   0.2/0.1/0.05 have no registered derivation; a 300-pair aic4 term and
   the 4,292-pair cid22 term carry weight ratios unrelated to their
   noise.

**The ranking under the flaws (product_composite, full coverage):**

| model | composite | what the scalar hides |
|---|---|---|
| north-anchor (A) | **0.8645** | UNDER-reads it: dial/identity/hfnl superiority invisible |
| clear-ember (e050) | 0.8616 | ranks ≈A while failing tid-LF + eligibility — composite is eligibility-blind |
| gray-tower (incumbent) | 0.8602 | THE flaw showcase (see #5) |
| **shipped B (harbor-line)** | 0.8487 | under-reads B's dial + KonJND 0.519 + HF-NL 0.503 (0.2/0-weighted); over-rewards its challengers' trainable axes |
| wlin2_a0.6 (arm-2 blend) | 0.7830 | correctly shows the KonJND/aic gaps; hides that it BEATS B on imazen26 (+0.043) at 2.6 KB |

**Registered proposal (composite-v2, adoption user-gated):** fixed
corpus set (absent ⇒ NOT COMPARABLE, never renormalized), signed
orientation-audited terms, split into TWO reported scalars (held-out
human composite | trainable-anchor composite), + an HF-NL term and a
dial-calibration term; weights derived and provenance-tagged like the
G-GRAN v2 bars. Until adopted, every composite citation should name the
formula and coverage.

### W-LIN arm-2 round 4 — BVLS heads land; best linear of the era (2026-08-29 ~16:0xZ)

`--solver bvls --bounds-tsv feature_sign_mask_2026-05-26.tsv` (the B
kon-head class, f372+ free): the canonical-mix BVLS head alone =
**cid22 0.8453 + KonJND 0.1931** (2.5× the lasso head's KonJND) —
sign-constrained density is real skill, not lore. **blend(konbvls,
full-mix, α=0.6) = cid22 0.8491 / imazen26 0.8727 / composite 0.7910**
— the era's best linear (B: 0.882/0.831/0.8487). Gap decomposition:
cid22 −0.033, composite −0.058, driven almost entirely by KonJND (0.188
vs 0.519). B's own kon head reached 0.67 KonJND WITHOUT konjnd training
data (ssim2-anchor targets + hdr_v3mix + BVLS at 372) — the 944
reproduction of that emergence is the registered next lever (hdr_v3mix
gram at 944 exists from the E-LIN campaign; check + reuse). Artifacts:
`wlin3_a0.6.bin` (sha ad1da162…) + head npzs preserved.

### W-LIN round 5 (hdrmix leg) — WAVE LEADER: wlin4_a0.5 (2026-08-29 ~16:4xZ)

The hdr_v3mix-944 leg (the campaign's own front-end-answer table) grammed
+ added to B's exact kon recipe shape (safesyn/cid22t/kadid/tid/hdrmix,
BVLS): kon-emergence partially reproduces (head KonJND 0.267, hfnl 0.20).
Final blends:

| candidate | cid22 | nonphoto | imazen26 | KonJND | hfnl | composite |
|---|---|---|---|---|---|---|
| **wlin4_a0.5 (WAVE LEADER)** | **0.8502** | **0.8713** | **0.8821** | 0.206 | 0.084 | **0.7970** |
| wlin4_a0.6 | 0.8506 | 0.8637 | 0.8742 | 0.213 | 0.113 | 0.7951 |
| shipped B | 0.8821 | 0.8640 | 0.8306 | 0.519 | 0.503 | 0.8487 |

**The 2-3 KB linear now beats B on nonphoto AND imazen26** (the ssim2
north stars, both first-class gates) with cid22 within 0.032. The
remaining composite gap is KonJND+hfnl — and the campaign's own
hdr_v3mix-944 conclusion ("the B-gap is front-end/regime, not missing
supervision") now reads as the LINEAR KonJND ceiling explanation at 944:
the folded-944 front end may not expose the near-threshold signal a
372-front linear could reach (B's kon head hit 0.67 KonJND at 372 with
NO konjnd data). Registered next levers (not run): a 372-front kon head
blended cross-regime (needs width-bridging — real design work), or
accepting the pair-of-profiles shape (wlin for RD-lane + B for JND-lane).
Wave artifacts complete in `bakes/wlin-2026-08-29/` (7 heads, 9 blends,
shas in-log).

## PROFILE B UNDER THE MODERN BATTERY (2026-08-29, user directive: "how does profile b do on the same flaw checks and gates")

| gate / instrument | B's result | context |
|---|---|---|
| G-OUT v2 | **FAIL nonphoto:S** (p99 chart-z 5.33 vs bar 4.86, max 10.9) | passes kadid by 0.02 (2.72/2.74 — where every lodestar cell failed), cid22/live/imazen26 clean; hfnlproxy absent from its per_pair (absent ≠ failed) |
| G-GRAN v1 (its own era's dial gate) | **FAIL all three measured codecs**: avif span 7.0 + top −0.61, jpeg top −2.08, webp top −1.76 (mono 1.00 everywhere; no jxl curve in its fulleval — not measured) | the incumbent-derived reach bars B misses are the ones the audit VINDICATED as ≈ honest ceilings |
| M3a photo | **0.597** — far below the MLP zoo (0.76-0.87) | the constant-gradient linear map is weak on photos |
| M3a screen | **0.543** — mid-pack (incumbent 0.632 > amber 0.601 > B) | but nearly FLAT across content (0.597→0.543) vs the MLPs' collapse (0.83→0.52) — linear maps are content-robust, mediocre everywhere |
| composite | 0.8487 (2nd behind A) | the audit's note stands: composite under-reads B's real strengths |
| where B still leads | cid22 **0.8821**, KonJND **0.519**, HF-NL **0.503**, full [0,100] spline, D-clause clean | the axes the composite under-weights are exactly B's |

Curve-level G-GRAN-v2 calibration was attempted and is INCONCLUSIVE
(the endpoint-interpolation method is degenerate at the peer top —
recorded so nobody cites the +0.00 "stretch" it prints); a proper
map-inversion read needs 372-class forwards over peer-scored cells
(regime purity blocks reusing the 944 grid features).

**The structural finding: the gate system is ASYMMETRIC.** Every modern
gate postdates B and was applied only to challengers — the shipped
default was never re-qualified and fails 2 of the 3 hard gates it can
take, while remaining the leader on exactly the axes the composite
under-weights. Implications registered: (1) B-replacement comparisons
must run on EQUAL gates (the W-LIN table now carries B's fail rows, not
just its wins); (2) any future default flip decision should weigh that
the bar "pass what B passes" is much lower than the bar challengers have
been held to.

## REGISTRATION — B FULL REMEASURE + W-LIN ROUND 6 (kon/hfnl lane) + R6-M MLP probe (2026-08-29, user directive: "remeasure b in full and try to get our new candidates hfnl and konjnd better")

**Correction first:** the B-battery table above says "no jxl curve in its
fulleval — not measured". WRONG — `dial.curves` has all four codecs
(avif/jpeg/jxl/webp); the earlier read walked the wrong key. The remeasure
below re-derives every dial verdict from the fresh JSON.

**Candidate baselines (board, pre-round):** the kon/hfnl-weak candidates are
the LINEARS — wlin4_a0.5 kon 0.206 / hfnl 0.084 vs B 0.519/0.503. The MLP
candidates already beat B on hfnl (north-anchor 0.752, amber 0.721) and sit
near kon parity (0.501 / 0.456 vs 0.519).

**B remeasure protocol:** `scripts/run_full_eval.sh <B bake> b_sdr_… 372`
with `ZENSIM_M3_REUSE=1` (M3a photo re-measured this session = stored value
to 4 dp; screen M3a 0.543 recorded in doc+registry — the fulleval schema has
no screen slot yet). Old JSON preserved as `.pre-remeasure-2026-08-29.bak`.
Fresh JSON ⇒ hfnlproxy per_pair lands (G-OUT can gate that axis), signed
sroccs + merged-decile bands + current corruption block from today's owners.
Then G-OUT v2 + G-GRAN v1 re-run off the fresh JSON; board regen.

**W-LIN round 6 arms (fit AFTER this registration):**
- **R6-K** kon-head refit = round-5 kon recipe (safesyn/cid22t/kadid/tid/
  hdrmix, BVLS shaped) **+ konbpg leg** (8,060 rows, the JND-threshold BPG
  encodes) at weight w ∈ {1.0, 2.0}. Report head-alone kon/hfnl; blend with
  head_cid at α ∈ {0.4, 0.5, 0.6}.
- **R6-H** sparse hf head: fit-lasso on l944_hf (11,941 rows, human_score),
  lam ∈ {1e-3, 3e-3} (sparse-additive is the measured hfnl-axis winner class,
  appendix O: 0.70–0.85 vs mid-MLP ~0.09). 3-way blend γ ∈ {0.1, 0.2} via a
  second blend-heads pass over the best R6-K blend.
- **Frozen bars (PASS):** kon ≥ 0.40 AND hfnl ≥ 0.40 AND cid22 ≥ 0.845 AND
  nonphoto ≥ 0.865 AND imazen26 ≥ 0.875 (hold wlin4's north-star wins within
  0.005/0.006). STRETCH: kon ≥ 0.45 AND hfnl ≥ 0.45. FALSIFIER: if no cell
  reaches kon 0.40 at cid22 ≥ 0.845, the folded-944 near-threshold cap is
  CONFIRMED for linears ⇒ registered outcome = pair-of-profiles shape (wlin
  RD-lane + B JND-lane) or the 372-front cross-regime design.
- Eval = bake_verdict --regime 944, corpora cid22,nonphoto,imazen26,konjnd,
  aic3,aic4,hfnlproxy (the round-5 instrument, unchanged).

**R6-M (MLP kon probe, background, k=1):** north-anchor recipe + konbpg
training leg (the W7 lever: kon 0.456–0.459 on the EM4 family, cid22 cost
certified THERE — untested on the PH line). Bars: kon ≥ 0.52 AND cid22 ≥
0.885 AND hfnl ≥ 0.70 (dominate B on every rank axis at once). One seed;
scale only if the probe passes.

### R6 mid-round findings + AMENDMENT R6-H2 (target-frame fix; 2026-08-29)

- **R6-K FALSIFIED at head level:** konbpg in the kon head HURTS its own axis
  (kon 0.154/0.143 at w 1.0/2.0 vs 0.267 without). Not blended further.
- **The hf head is the discovery:** fit-lasso on l944_hf alone, lam 3e-3
  (74 coeffs): kon **0.445** + hfnl **0.726** + cid22 0.808. Sparsity curve
  peaks at 3e-3 (lam 1e-3: kon 0.044; 2e-3: 0.369; 5e-3: 0.414). The
  944-folded front end CARRIES the near-threshold signal — the round-5
  "front-end cap" reading is OVERTURNED; the cap was data-frame, not features.
- **Convex blending CANCELS it:** head_cid is hfnl-ANTI (−0.016), and even at
  70% hf weight the blend reads hfnl 0.068 / kon 0.216. wlin5_a* all fail.
- **Single-head naive mix is WORSE (and diagnostic):** hf-w 5/10 in the mix
  → hfnl NEGATIVE, cid22 0.28–0.54. ROOT CAUSE (measured from gram moments):
  **the hf leg's per-corpus min-maxed target lives in a LOCAL frame** — leg
  y_mean 0.324 vs 0.50–0.78 everywhere else, i.e. its "0.32" = top third of
  the near-lossless band while safesyn's 0.77 = mid encodes. Mixed frames
  poison the joint fit; SROCC-invariance hides it in single-leg fits. This
  RETRO-EXPLAINS round-2's "hf 2.5 catastrophic" — never an upweighting
  problem, a target-frame incompatibility.
- **AMENDMENT R6-H2 (registered before the refit):** remap the hf gram's
  target moments into the global top slice via exact affine moment algebra
  (q′=a·q+b·s, Y1′=a·Y1+b·n, Y2′=a²Y2+2abY1+b²n), mappings y′ ∈
  {0.85+0.15y, 0.80+0.20y} (the band ≈ ssim2 ≥ 91 ⇒ top ~15% of frame),
  hf-w ∈ {3, 6}, lam 2e-3/3e-3 as needed. Bars UNCHANGED from the round-6
  registration.

### B FULL REMEASURE — results (2026-08-29; fresh fulleval, regime 372, current owners)

`scripts/run_full_eval.sh … 372`, M3 carried (photo M3a re-measured = stored
to 4 dp). Old JSON preserved as `.pre-remeasure-2026-08-29.bak`. Deltas:

| axis | stale row | fresh | why it moved |
|---|---|---|---|
| cid22 | 0.8821 | **0.8764** | current corpus/eval state (peers were scored on it; B never was) |
| konjnd | 0.5186 | **0.5466** | ditto (B *improves*) |
| aic3 / tid | 0.7650 / 0.7785 | 0.7774 / 0.7868 | ditto |
| kadid | −0.8085 | **+0.8201** | the documented era rule: fresh verdict on rebuilt tables; matches the ledger's "+0.8201" exactly |
| composite | 0.8487 | **0.8291** | **the stored composite was STALE-HIGH by 0.02** — it never matched its own rank rows (recomputed-from-old-rows = 0.8285). Peers are clean (drift ≤ 0.0001). **Ladder reorders: A 0.8645 > gray-tower 0.8602 > amber 0.8508 > B 0.8291** — B falls 2nd → 4th |
| hfnlproxy | grafted, no per-pair | 0.5027 + native per-pair (n 11,356) | the axis is now G-OUT-gateable for B |

**G-OUT v2 (fresh): FAIL 7 clauses / 4 axes** — hfnlproxy:R+S (p99 chart-z
**14.60** vs bar 3.24, max **25.1** — the worst outlier axis measured on the
board; the linear head extrapolates wildly inside the near-lossless band),
imazen26:R+S (3.66/3.18), nonphoto:R+S (5.24/3.47), kadid:S (2.92/2.73).
The earlier stale-row run (FAIL nonphoto:S only) is superseded — its per-pair
block was the old sampling.

**G-GRAN v1 (fresh): jxl PASSES** (top 95.95, span 27.6 — the "no jxl curve"
claim in the first battery table is RETIRED; fresh top-shortfalls: jpeg
−3.43, webp −3.06, avif −1.61 + span 7.0; mono 0.95–1.00). M3a photo 0.597 /
screen 0.543 unchanged.

### W-LIN ROUND 6 VERDICT — falsifier FIRED for single-linear; the PAIR passes (2026-08-29)

Probes after the amendment: hf-dominant mixes (broad 20/40%) lose both ends
(kon ≤ 0.15, cid22 ≤ 0.76); BVLS-on-hf kon 0.251; lam 4e-3 kon 0.424. Best
remap cell (r85_hf6): hfnl 0.360, cid22 0.758 — under bar on both.

**Verdict: no single 944 linear reaches kon ≥ 0.40 ∧ hfnl ≥ 0.40 while
holding cid22 ≥ 0.845 — the registered falsifier fires, but for a NEW reason:
not a front-end cap (the hf head PROVES the folded-944 features carry kon
0.445 + hfnl 0.726) — composition itself fails (blend cancellation vs an
hfnl-anti generalist head; joint-fit frame incoherence across per-corpus
min-maxed legs).**

**Registered outcome — the PAIR (both tiny linears, ~11 KB together):**
- **copper-line** = wlin4_a0.5 (RD lane): cid22 0.8502 / nonphoto 0.8713 /
  imazen26 0.8821 — holds the three RD bars.
- **jeweler-loupe** = head_hf0.003 (JND/HF lane): kon **0.445** / hfnl
  **0.726** / cid22 0.808 — holds both JND bars; 74 coefficients, lasso 3e-3
  on the l944_hf leg alone (recipe + sha in wave artifacts).
- Jointly the pair passes ALL FIVE frozen round-6 bars on their assigned
  lanes. vs B: loupe leads hfnl +0.223; kon −0.074 vs B's fresh 0.547 is
  now the ONE axis B keeps (fresh remeasure widened it: −0.102).
- Ship-shape (which lane a caller gets, and when) is a product decision —
  user-gated, like every default flip.

### The pair on EQUAL GATES (2026-08-29) — rank bars pass, dial battery does NOT

Per the equal-gates rule the pair took the same battery as fresh-B:

| | copper-line | jeweler-loupe | fresh B |
|---|---|---|---|
| G-OUT v2 | FAIL 6 (hfnl R+S 5.61/7.0, im26:R, nonphoto:R, cid22:S, live:S) | FAIL 6 (cid22:R, hfnl:S 3.28/3.24 hair-over, im26:R, kadid:S, live:S, nonphoto:R; off-band konjnd chart-z 15.9 is un-gated) | FAIL 7 (hfnl p99 14.6) |
| G-GRAN v1 | **FAIL all 4 incl. MONO** (avif 0.82 / jpeg 0.87 / jxl 0.75; tops 86–89) | FAIL all 4 (tops 86–93, spans 6–7; mono 0.95–1.00) | jxl PASS; jpeg/webp/avif top-reach fails; mono 0.95–1.00 |
| M3a (photo) | **0.780** — linear-with-transforms lands in the MLP coherence zone | **0.801** | 0.597 |

**Tempered conclusion (recorded before any enthusiasm ships):** the pair wins
its rank lanes and BOTH members' maps are far more coherent than B's — but
copper-line's per-codec dials are NON-MONOTONE at the medians (0.75–0.87),
which no spline refit can repair (spline is monotone; the medians are the
model's raw ordering on codec ladders). As a product DIAL the pair is
currently WEAKER than B; jeweler-loupe's own q≥88 zone (its lane!) tops out
86–93 with spans 6–7 vs the 8-bar. Next levers, in order: (1) the
bake_dial_refit dial pass (extend-top / shared-anchor) that B itself received
— never applied to any wlin bake; fixes spline-shape reach, NOT mono; (2) a
mono-targeted refit of copper-line (kon/dial-ordering data in the blend); (3)
accept scorer-lane-only roles. No ship claim.

### Dial-repair lever (1) MEASURED-FALSIFIED for the pair (2026-08-29)

`bake_dial_refit` gained `signed_cbrt`/`log1p` in the f64 fit-forward (owner
extended, unit-tested — the shaped-944 class was previously refused). Then:
**extend-top on copper-line is a NO-OP on its dial curves** (top 87.82
unchanged, rank identical) — its raw scores on the best grid encodes never
reach the spline's top region: the top-zone compression is MODEL resolution,
not spline shape. **loupe's saturation fit cannot converge** (non-decaying k)
— same class. So the pair's reach shortfalls are not repairable by any
spline pass; the remaining paths are a mono/top-targeted RETRAIN of the RD
member, or scorer-lane-only roles. Recorded before enthusiasm shipped.

### Two-zone cid22 CIs for B (paired bootstrap, B=5000, seed 11, aligned n=4292; 2026-08-29)

Completes the two-zone zone B never took. ΔSROCC vs fresh-B, medians + 95% CI:
- **north-anchor +0.0163 [+0.0115, +0.0212] — LEADS B, CI excludes 0** (the
  first significance-tested cid22 claim for the C candidate vs the default).
- copper-line −0.0262 [−0.0333, −0.0193] — B leads.
- jeweler-loupe −0.0682 [−0.0787, −0.0590] — B leads (band specialist, expected).
- v-hfnl zone: **NOT COMPARABLE cross-regime** — B's hfnl slice is the
  372-root (n 11,356), candidates' is the 944 TEST views (n 7,717); different
  pair populations, no row alignment. Stated, not skipped.

### R6-M VERDICT — FALSIFIED, direction negative (2026-08-29, k=1 s4004)

Kon-leg weight 1.2 → 2.5 on the exact north-anchor recipe (only variable;
seed-matched): cid22 0.8769 (−0.016), kon **0.4635 (−0.037 — the target axis
went DOWN)**, hfnl 0.6926 (−0.060), nonphoto/im26 +0.002/+0.008, composite
0.8558. All three bars missed. Mechanism: the konjnd_bpg val leg is already
saturated in-recipe (val srocc 0.9967) — extra mass shifts the model toward
the BPG train distribution without buying PJND generalization, and costs the
other axes. **No scale-up; north-anchor's recipe is at its kon optimum in
this weight family.** Fresh-B's KonJND 0.547 vs north-anchor 0.501 stands as
the one rank axis the default keeps (cid22 itself is now CI-tested in the
CANDIDATE's favor). Probe bake kept: `wlin-2026-08-29/R6M_PH_kon25_s4004.bin`.

## SPLIT-LADDER AUDIT (2026-08-29, user question: "what is our train vs eval split on konjnd, hfnl, tid, kadid, kadis, etc")

Verified against `docs/DATA_SPLITS.md` + manifests + the parquets themselves
(origin-digit histograms). New findings, all recorded in the registry (§3c):

1. **Three slice generations are live on the board** for imazen26/nonphoto/
   hfnlproxy: era candidates read the pre-D1 TEST-family cuts (n 7,869/7,255/
   9,167), today's pair reads the 2026-08-28 validate-family cuts (6,953/
   6,142/7,717), and B's remeasure read the 372-root TEST-family cuts (7,844/
   8,241/11,356). Cross-generation numbers are NOT same-ruler.
2. **Same-ruler rescore (validate slices) of the era candidates:**
   north-anchor cid22 0.8927 / nonphoto 0.9280 / im26 0.9314 / kon 0.5006 /
   hfnl **0.6993**; amber 0.8735/0.9314/0.9329/0.4557/**0.6393**; gray-tower
   0.8867/0.9204/0.9238/0.4988/**0.3337**. Rankings hold; hfnl drops 0.05–
   0.09 vs the test-family reads. **On the same ruler, jeweler-loupe (0.7260)
   is the hfnl LEADER** — ahead of north-anchor (0.6993).
3. **jeweler-loupe's training is holdout-clean, now VERIFIED:** its train leg
   `tbig_hf_pure` = 195 origins, all train digits {0,2,4,6,8}; the hfnlproxy
   eval = 87 validate-digit origins {1,3,5}; **origin intersection 0**.
4. **Bookkeeping fixed:** the 3 hf-leg files were unmanifested in the
   sdr-pure root (added with shas); the hfnl family had no DATA_SPLITS.md
   section (added §3c); the ext944 `_MANIFEST_eval_slices.json` hfnlproxy
   entry is STALE (records the pre-D1 11,356-row cut) and bake_verdict's
   "944 TEST views" display string is wrong for the validate slice (both
   flagged in §3c; display-string fix queued, not load-bearing).
5. **B's remeasure consumed the retired test-family slice** (372 root) — a
   terminal-read-class touch, consistent with D1's touch-once rule, but its
   hfnl/im26/nonphoto rows are not same-ruler with any 944 candidate.

## SPLIT POLICY v2 + THE FAIR B-vs-CHALLENGERS TABLE (2026-08-29, user directives)

**Policy v2 registered** (DATA_SPLITS.md §8): universal invariant (eval/test
content never trains), per-dataset TRAIN/SELECT/TERMINAL views built for
kadid (40/25/16 refs), tid (12/9/4), konjnd JPEG (—/404/100), kadis
(%10<8 view, 40,040 rows); enforcement owner
`scripts/canonical_corpus/check_split_compliance.py` (audits any bake's
embedded repro; hard-errors; guards WARN).

**Audit findings:** (1) **kadis 50k leg violated its own registered rule —
19.92% val/test sources; every 944-era bake trained on KADIS val AND test
sources** (the safety grid is not a content holdout for them; fixed view
built, next-generation recipes switch, checker now catches it — verified
exit 1 on the old leg, exit 0 on the new). (2) Everything else in the
flagship recipe is clean (tbig/hf legs 0 overlap vs every slice; konbpg
BPG⊥JPEG; cid22 201⊥49). (3) kadid/tid full-set train==val remains as
REGISTERED GUARD rows only.

**Split reasonableness (6-model rescore on the new views):** KADID 3-way
REASONABLE (buckets agree 0.01–0.03, rankings identical); TID SELECT
reasonable, 4-ref terminal level-shifted (+0.03–0.05, rank-preserving) —
guard only; KonJND SELECT (404) tracks the full set, 100-ref terminal
VOLATILE (amber 0.502→0.244; jeweler-loupe is the most content-stable JND
model: 0.449/0.418).

**Board made generation-consistent:** the three era fullevals re-ranked on
the current validate slices (backups kept): north-anchor im26 0.9314 / hfnl
0.6993 / composite 0.8664; amber 0.9329 / 0.6393 / 0.8527; gray-tower
0.9238 / 0.3337 / 0.8601. Every 944 board row now reads the same slice
generation as the pair.

### The fair comparison — B vs the SDR challengers (same-ruler axes marked)

| axis (ruler) | B (fresh) | north-anchor (C) | amber | copper-line | jeweler-loupe |
|---|---|---|---|---|---|
| cid22, identical 4,292 rows + paired CI | 0.8764 | **0.8927 (+0.016, CI excl 0)** | 0.8735 | 0.8502 | 0.8080 |
| konjnd JPEG 504, identical rows | **0.547** | 0.501 | 0.456 | 0.206 | 0.445 |
| kadid SELECT view, identical rows (all trained on these refs — equal footing) | 0.820 | **0.908** | 0.914 | 0.839 | 0.649 |
| tid SELECT view, ditto | 0.770 | **0.929** | 0.924 | 0.802 | 0.671 |
| nonphoto (B: 372 test-family / others: validate-family — POPULATION CAVEAT) | 0.8640 | 0.9280 | **0.9314** | 0.8713 | 0.8190 |
| imazen26 (same caveat) | 0.8306 | 0.9314 | **0.9329** | 0.8821 | 0.8217 |
| hfnl (same caveat) | 0.503 | 0.699 | 0.639 | 0.084 | **0.726** |
| M3a photo / screen (same instrument) | 0.597 / 0.543 | 0.763 / — | 0.83? / 0.601 | 0.780 / — | 0.801 / — |
| G-OUT v2 (same gates) | FAIL 7 clauses (hfnl p99 14.6 = worst on board) | full-eligibility PASS (Q1: sole pass of 20) | — | FAIL 6 | FAIL 6 |
| G-GRAN v1 | jxl PASS, 3 codecs top-reach FAIL, mono clean | faithful dial (Q1 case) | — | FAIL 4 + MONO | FAIL 4 |
| composite (same formula) | 0.8291 | **0.8664** | 0.8527 | 0.7970 | — |
| size | 7.3 KB | 149 KB (behind default-ON `candidate-profiles`) | 152 KB | ~3 KB | ~7.7 KB |

**Stated against the swap (SDR):** kon −0.046 (B's one axis; roughly parity
on the kon-SELECT view: NA 0.529); the touch-once hidden panel's maximin
ranked s4004 LAST among the finalists (incumbent 0.6331 > … > s4004 0.5843
— shift-robustness on non-codec synthetics); 20× the bytes.

### HDR: BHdr vs aurora-anchor (CHdr) — UPIQ human anchor

Fresh seven-domain external reads for aurora (`--scorer bake:`) vs shipped
BHdr's recorded upiq_panel row (same 380 UPIQ rows + JOD labels; different
pipelines — instrument caveat stated):

| | UPIQ pooled | narwaria | korshunov |
|---|---|---|---|
| shipped BHdr (λ3e-4 cvmix, recorded 2026-07-12) | **0.7536** | **0.7834** | 0.9175 |
| aurora-anchor (fresh) | 0.6664 | 0.6434 | **0.9280** |

Aurora also reads: hdrvdc 0.699/0.695/0.792, avt pooled 0.778, chug 0.739,
rousselot 0.821/0.847, sihdr 0.358. **On the HDR human anchor the shipped
default LEADS by +0.09 pooled / +0.14 narwaria** — aurora's superiority is
in-domain (route dial, hdr944 instruments), not transfer.

### RECOMMENDATION (user-gated, as always)

- **SDR: swap B → north-anchor (Profile C) — the evidence supports it.**
  It leads every same-ruler rank axis except kon (CI-tested on cid22),
  passes the eligibility battery B fails, has a coherent map (M3a +0.17)
  and a faithful dial, at stated costs: kon ≈ −0.05, hidden-panel
  shift-robustness (gray-tower led there), 149 KB vs 7.3 KB. If byte-size
  or the hidden-panel result weighs heavier, the conservative alternative
  is HOLD with C as candidate-of-record (status quo).
- **HDR: do NOT swap — keep BHdr.** Aurora loses the UPIQ human anchor by
  a wide margin; CHdr stays candidate-of-record and the next HDR lever is
  closing the transfer gap (UPIQ-gated retrain on the 944 HDR route).

## THE SINGLE-RULER REPORT (from the top, 2026-08-29 — user directive: same eval methods, flag test=train, tid train-only, B vs C vs the MLP/linear leaders SDR+HDR, path to improvement)

**Doctrine:** an axis enters this table ONLY if every model reads IDENTICAL
pairs with RANK-IDENTICAL targets (asserted programmatically from per-pair
vectors; value-encoding may differ across roots, ordering may not). Paired
bootstrap B=5000 seed 11 vs shipped B on every axis. Flags: CLEAN =
content-disjoint from every model's training; FAMILY = same corpus family,
content-disjoint; **T = train==eval** (all models trained on these refs).
TID is RETIRED TO TRAIN-ONLY (user ruling, §8.1). LIVE is EXCLUDED (targets
not rank-identical across roots — registered defect §8.2). Architectures
verified from the bakes: B/copper/loupe = single-layer linears; C/amber/
gray-tower/aurora = 2-layer MLPs (944→667/697→128→1).

| axis (flag) | n | B 372-lin | C MLP | amber MLP | gray-tower MLP | copper lin | loupe lin |
|---|---|---|---|---|---|---|---|
| cid22 (FAMILY) | 4292 | 0.8764 | **0.8927** ▲ | 0.8735 = | 0.8867 ▲ | 0.8502 ▼ | 0.8080 ▼ |
| konjnd JPEG-504 (B CLEAN; MLPs FAMILY-bpg) | 504 | **0.5935** | 0.5006 ▼ | 0.4557 ▼ | 0.4988 ▼ | 0.2064 ▼ | 0.4453 ▼ |
| aic3 (CLEAN) | 600 | 0.7774 | **0.8000** ▲ | 0.7785 = | 0.7968 ▲ | 0.7534 ▼ | 0.7151 ▼ |
| aic4 (CLEAN) | 300 | 0.8906 | **0.9144** ▲ | 0.9006 ▲ | 0.9019 ▲ | 0.8860 = | 0.8071 ▼ |
| csiq (CLEAN) | 866 | 0.9342 | 0.9443 = | **0.9533** ▲ | 0.9331 = | 0.9108 ▼ | 0.7531 ▼ |
| kadid-select (**T for all**) | 3125 | 0.8198 | 0.9082 ▲ | **0.9139** ▲ | 0.9078 ▲ | 0.8394 ▲ | 0.6489 ▼ |

▲/▼ = paired 95% CI vs B excludes 0; = spans 0. Full CIs in
`~/tmp/samepair_report*.json` (mirrored below in-doc where cited).

**Corrections this table forced:** (1) **B's board kon 0.5466 was DILUTED**
— its 372 file scores all 1,008 refs (both halves); on the JPEG-504 same
pairs B reads **0.5935** and leads EVERY challenger with CI excluding 0
(C −0.0930 [−0.1367,−0.0487]). The kon cost of a B→C swap is real and
larger than previously stated. New view: `konjnd_jpeg504_372_2026-08-29.parquet`.
(2) live excluded (§8.2). (3) family axes (imazen26/nonphoto/hfnl) are NOT
in this table — no cross-regime pair key exists (no distorted-id column in
any of the tables); they remain direction-only evidence: C 0.93/0.93/0.70 vs
B 0.86/0.83/0.50 on different cuts of the same family. Closing this gap =
roadmap item R1.

**HDR head-to-head (same instrument, each model on its OWN regime features —
the tool built for exactly this):** `upiq_panel.py --compare`, paired
per-stratum bootstrap, 5000 resamples. BHdr reproduces its recorded 0.7536
exactly (ruler confirmed).

| | pooled | narwaria | korshunov |
|---|---|---|---|
| shipped BHdr (372-PU linear) | **0.7536** | **0.7834** | 0.9175 |
| aurora / CHdr (944 MLP) | 0.6664 | 0.6434 | 0.9280 |
| paired Δ (BHdr−aurora) | **+0.0872, p=0.0000** | **+0.1400, p=0.0116** | −0.0105, p=0.96 (ns) |

**Verdicts under the single ruler:**
- **SDR:** C beats B with CI on cid22/aic3/aic4 (+ the T-flagged kadid) and
  ties csiq; **B beats every model on konjnd by ≥0.09 (CI)** — and B's kon
  read is the only CLEAN one (no konjnd data in its recipe). The swap
  recommendation STANDS but its cost is now precisely priced: −0.093 kon,
  plus the hidden-panel shift-robustness note, plus 20× bytes.
- **SDR linear lane (the "replace B with a linear" question):** no current
  linear beats B under the single ruler — copper loses cid22/kon/aic3/csiq
  (wins only T-flagged kadid), loupe loses everything broad (it is a
  specialist). The linears' family-axis wins (nonphoto/imazen26) are
  direction-only pending R1. The honest linear story: B is still the best
  all-round linear we have; copper/loupe are lane specialists.
- **HDR:** keep BHdr; aurora's UPIQ loss is significant. We NEED a better
  HDR model — path below.

**PATH TO IMPROVEMENT (registered lanes, in priority order):**
- **R1 — same-pair family axes (unblocks every remaining comparison):**
  re-extract 372 features for the validate-family slice content (~20.8k
  encodes; rescore-from-links pattern, fleet-sized) so B and the 944 estate
  read identical family pairs. Until then family axes stay direction-only.
- **R2 — kon closure for C (the swap's one cost):** weight levers are
  exhausted (R6-K, R6-M falsified). Three registered mechanisms: (a) distill
  B's kon head as a teacher leg (tkon: B-kon predictions over the train
  estate — cross-regime knowledge without cross-regime features); (b)
  near-threshold FEATURE work (BANDVIS-GAIN combine, the registered P3/LOO
  candidate); (c) a kon-companion micro-head at the profile level (pair
  shape). Gate: kon-JPEG504 ≥ 0.55 holding cid22 ≥ 0.885.
- **R3 — better HDR (the user's call-out):** (a) freeze a UPIQ-TRANSFER
  gate into HDR selection (aurora's in-domain wins never checked transfer);
  (b) BHdr+ refresh — same PU-372 shaped recipe refit with corrected target
  frames + an HDR near-lossless sparse head (the loupe recipe on the HDR
  band); (c) the binding constraint is HDR TRAIN DATA: hdr_v3mix is the
  only leg (7,410 rows / 58 origins, cvvdp-mix teacher) — expand origins +
  codecs (avif-HDR datagen remains USER-HALTED pending zenavif confirm —
  that gate must lift first) and mine SI-HDR's weak zone (both models 0.36 —
  reproduction-error domain is the data gap); (d) a 944-HDR retrain with
  UPIQ-gated selection. Gate: UPIQ pooled ≥ 0.75 AND korshunov ≥ 0.92 AND
  the HDR-lane freeze battery.
- **R4 — linear lane:** (a) global target-frame rebuild for ALL legs (R6
  proved per-corpus min-max frames poison joint fits — ONE calibrated frame
  unblocks single-model hf+kon+broad composition); (b) dial-mono supervision
  for copper (q-ladder rank constraints from the dial grid in the fit); (c)
  the pair stays scorer-lane fallback.
- **R5 — policy hygiene:** next-generation recipes on the v2 train views
  (kadis_train mandatory) + checker green; kadid board rows migrate to the
  select view (T-flagged until post-v2 models exist); live cross-root audit.

## R-LANE EXECUTION + THE GATES CANON (2026-08-29, user directives: "R1 was done long ago… proceed through all the rest… see if we can make a C class HDR… decide on gates… zenjpeg is float quality")

### R1 — resolved without regeneration (the user was right)

The validate-digit 372 feature tables EXIST:
`/mnt/v/output/zensim-multicodec-probe/bigcodec_{hqdedup,hqfill}_valdigits_2026-07-02.parquet`
(114,871 + 148,657 rows). Nothing was regenerated. Built B-side validate-BUCKET
views from them (`{imazen26,nonphoto,hfnlproxy}_valbucket_372_2026-08-29.parquet`,
manifested) and scored B: **imazen26 0.8644 / nonphoto 0.8728 / hfnlproxy
0.8403** (per-ref ~0.96). BUT the read teaches the deeper lesson: these are
FULL-density tables (263k/128k/26k rows) vs the 944 slices' strided-deduped
cuts (7k rows) — **slice SAMPLING moves srocc as much as bucket choice**
(B hfnl: 0.503 test-cut → 0.840 full-val-bucket; neither comparable to the
944 slices' 0.699-class numbers). Cross-regime PAIR identity was attempted
via the (ref, score) fingerprint and measured at 0-20% match (score
generations differ) — the documented join-trap, not pursued. **R1b
registered:** keyed rebuild of BOTH sides from the encode-sha lineage
(sidecar + canonical views) = the only honest same-pair family instrument.
Until R1b, family axes stay direction-only.

### R3 executed — the HDR transfer sweep + the path to a C-class HDR

Swept ALL 39 existing 944-HDR bakes through `upiq_panel` (own-regime
features; full table `~/tmp/hdr_upiq_sweep.log`, headline rows):
- **No existing 944 bake reaches shipped BHdr (0.7536).** Ceiling:
  **HDR944_L1T1_s4004 = 0.7254** (nar 0.7419 / kor 0.9312), then L1T2_s4005
  0.7201, L1T2_s4003 0.6995.
- **The frozen aurora (s4005, 0.6664) was not even its own family's best on
  transfer** — s4004 (+0.059) sat unpromoted. Transfer is violently
  seed-variant (L0 family 0.33–0.65; L1T1 0.665–0.725): selection without a
  transfer axis was blind to it — the R3 gate thesis, now measured.
- Paired vs BHdr: s4004 Δ pooled +0.0283 p=0.042 (borderline), narwaria
  +0.042 p=0.21 (ns), korshunov −0.014 (s4004 ahead, ns) — a NEAR-tie,
  vs aurora's decisive loss (p=0.0000).
- **Launched now:** UPIQ-gated seed-mining wave, L1T1 recipe verbatim ×
  seeds {4006–4009}, each bake UPIQ-read on completion
  (`~/tmp/hdr_seedwave*.log`). If any seed ≥ BHdr-parity on the paired test
  AND passes the HDR-lane battery, it becomes the CHdr-swap proposal;
  s4004 is the standing contender regardless (its battery run is the next
  step after the wave).

### THE GATES CANON (decided, as delegated)

1. **G-OUT v2 — FROZEN** (peer-derived bars; provenance audit stands).
2. **Two-zone eligibility — FROZEN** (SDR lane).
3. **G-GRAN v2 — ADOPTED as the dial gate**, superseding v1 for gating (v1
   stays a build-time knob-end display). CONDITION: its knob-quanta map must
   be re-derived for jpeg first (see zenjpeg correction below) — until that
   lands, the board 'usable' trio stays G+E+K rather than gate on a stale
   quantum table.
4. **HDR: UPIQ-TRANSFER GATE — FROZEN NOW.** An HDR ship/freeze candidate
   must show (a) paired UPIQ pooled vs shipped BHdr NOT significantly worse
   (p ≥ 0.05 on the paired bootstrap), (b) korshunov ≥ 0.92, (c) the
   HDR-lane freeze battery. HDR SELECTION runs the UPIQ read per seed
   (seed-variance 0.33–0.73 makes transfer-blind selection invalid).
5. **Composite-v2 — ADOPTED IN PRINCIPLE**: two scalars (held-out human |
   trainable-anchor), fixed coverage, signed audited terms, HF-NL + dial
   terms, derived weights. Implementation lane registered (bake_verdict owns
   the formula); until it lands every citation names formula + coverage.
6. kadid board rows migrate to the SELECT view (T-flagged); tid retired
   (both already recorded).

### zenjpeg IS FLOAT QUALITY (user correction — confirmed and FIXED)

`zenjpeg::EncodeOptions`-side and internal quality is `f32` end to end
(`target_quality.rs: pub quality: f32`); the INTEGER assumption lived in OUR
tooling: `search_target` rounded every trial to the integer grid on a false
premise comment ("JPEG quality is integer-valued") and zenjpeg-bench-utils
carries `quality: u8` fields. **Fixed in zenjpeg (`fad6a0af`, pushed +
verified):** float-native secant with `TargetOptions::quality_step`
(default 0.25; 1.0 restores the old grid), f32-bit trial cache, 2 new grid
tests, 11/11 pass. CONSEQUENCES REGISTERED: (a) the "zenjpeg error floor
1.06 = integer-q rung" census conclusion is SUPERSEDED — the floor was our
search's artifact; re-census zenjpeg targeting with step 0.25 to re-measure
the true floor; (b) G-GRAN v2's jpeg knob-quantum entry (integer ladder) is
wrong for zenjpeg-owned loops and must be re-derived; (c) bench-utils' u8
quality fields are a separate cleanup (they quantize sweeps, acceptable for
grids, wrong for loops).

### OVERFITTING + SEED STABILITY (user challenge, 2026-08-29 — measured before the seed wave lands)

The challenge was correct on the protocol and answerable on the population:

1. **Seed differences are REAL, not eval noise.** Paired within-stratum
   bootstrap s4004-vs-s4005 (same recipe, seeds only): **pooled Δ +0.0588,
   p=0.0036** (narwaria +0.098 p=0.085; korshunov +0.003). Selecting among
   seeds selects genuine model quality — but the SELECTED value still carries
   max-of-k inflation, which the protocol below makes measurable.
2. **In-domain selection is recipe-valid, seed-blind.** best_val vs UPIQ
   across 23 raw bakes: SROCC **+0.78** (in-domain signal is NOT overfit
   garbage at recipe level) — but within a family best_val saturates
   (L1T1: 0.9918–0.9924, range 6e-4) and cannot see the real 0.059 seed
   difference. This is exactly how aurora got frozen over s4004.
3. **No valid external seed proxy exists** (measured on the 6 family bakes):
   AVT 0.657–0.685 and CHUG 0.642–0.660 are seed-FLAT and wrongly ordered;
   rousselot (n=96) separates only the s4003 outlier; SI-HDR is unstable
   (−0.40…+0.36 across the family) and cannot rank anything.
4. **PROTOCOL (registered, applies to the running s4006–4009 wave):** UPIQ
   is split content-deterministically (image-id parity; 10 scenes/190 rows
   per half; `upiq_hdr_944_{selecthalf,confirmhalf}_2026-08-29.csv`,
   manifested). Seed SELECTION uses the SELECT half only (+ the in-domain
   battery as sanity); the CONFIRM half is read ONCE for the single
   pre-declared pick, and the select-vs-confirm gap of the pick IS the
   measured winner's curse, reported alongside. The full-380 sweep numbers
   measured today are PRE-PROTOCOL reads and are so labeled — for the
   existing 39 bakes the confirm half is not virgin (disclosed, not hidden);
   the protocol is clean for the new seeds. The frozen HDR UPIQ-transfer
   SHIP gate (paired-vs-BHdr) is unchanged — it gates the final candidate,
   not selection.
5. **SDR analogs on record:** M3a carries 42.3% seed-noise variance at fixed
   recipe (the coherence study) and the kon 100-ref terminal reorders models
   — same lesson: single-axis max-of-k selection on a small surface is how
   overfitting enters. The scorecard/battery selection style (multi-gate,
   CI-tested) is the counter-design.

### IS B OVERFIT? (user question, 2026-08-29 — measured verdict: NO; B is UNDERFIT where it fails, and seed-exact by construction)

1. **Capacity forbids memorization:** B is 373 f16 parameters (372→1 linear +
   bias) against ~340k training rows — 1e-3 params/row.
2. **The guard INVERSION is the signature of a non-memorizing model:** on the
   corpora B trained on wholesale it ranks LAST (kadid-select 0.820 vs the
   MLPs' 0.908–0.914; tid 0.787 vs ~0.93), while on CLEAN axes it ranks
   mid-to-top (kon-504 **0.594 = best on board, training-clean**; csiq 0.934;
   aic4 0.891). A memorizer dominates its own training corpora; B cannot.
3. **Train-leg gap is uniform across classes** (measured: cid22t-201
   ssim2-leg vs cid22-49 human holdout — C +0.098, amber +0.113, gray-tower
   +0.105, copper +0.109, loupe +0.109): no model shows an outsized
   trained-on advantage; the ~0.10 gap is target-type (ssim2 vs human MOS) +
   ref shift, not differential memorization. (B's own 372 train-leg row table
   no longer exists outside grams — its gap is not directly measurable;
   the guard inversion covers the question.)
4. **Selection bias: real, bounded, visible.** B was selected FOR cid22 (the
   name says cid80; final blend round n=5, campaign surface larger). The
   observed cost: fresh remeasure cid22 0.8821 → 0.8764 (−0.006) on updated
   corpora. Post-selection-era instruments it never saw during its search
   (kon-JPEG-504, sdr25, G-OUT v2, the valbucket views) read strong on rank —
   its generalization is real.
5. **Where B fails is UNDERFIT, not overfit:** the near-lossless band had
   ~zero training mass in its era → hfnl 0.503 + the worst outlier behavior
   on the board (p99 chart-z 14.6 = wild extrapolation in an unseen region),
   and honest dial top-reach misses. Coverage hole, not memorization.
6. **Seed stability: exact by construction.** B is a deterministic convex fit
   (lasso/BVLS coordinate descent on frozen grams — no SGD, no seed), with a
   BYTE-identical reproduction chain (sha-gated, reproduced repeatedly in
   this repo). Zero seed variance — against the MLPs' measured 42% M3a
   seed-noise share and the HDR family's 0.33–0.73 transfer swing. This is a
   structural advantage of the linear class worth keeping on the ledger.

### SEED-MINING FALSIFIED — the split-half protocol worked on first use (2026-08-29 ~21:1xZ)

Wave s4006–4009 landed (pre-protocol full-380 reads, disclosed: 0.6853 /
0.6474 / 0.6472 / 0.7209). Protocol applied over all 7 L1T1 seeds:

- **Select-half REORDERS the population**: aurora/s4005 (full-380 0.6664,
  the "loser") is the select-half TOP (0.7249); s4004 (full-380 winner
  0.7254) reads mid-pack (0.7031). Seed-rank agreement select-half vs
  full-380: **SROCC 0.14 over 7 seeds — nothing.**
- Interpretation: the paired s4004-vs-s4005 difference (p=0.0036) is real
  PER-ITEM but **content-idiosyncratic — it does not transfer across scene
  halves**. Any seed chosen by any 380-item UPIQ read is scene luck. With
  10 scenes per half, per-content skill dominates per-seed skill.
- **The honest family value: UPIQ 0.680 ± 0.030 (7 seeds, range 0.647–
  0.725).** The gap to shipped BHdr (0.7536) is a RECIPE-level gap sitting
  above the entire seed distribution. **Seed mining is FALSIFIED as the
  C-class-HDR path.**
- **Correction to this morning's framing:** "aurora wasn't even family-best
  on transfer / s4004 sat unpromoted" is RETRACTED as a selection-error
  claim — on scene-robust terms no L1T1 seed is meaningfully better than
  another; aurora's freeze was not a transfer mistake. (The structural point
  stands: selection had no transfer axis; it just turns out a transfer axis
  at n=380 cannot rank seeds either.)
- **The protocol's first save:** a full-380 selection would have promoted
  s4004 with an inflated near-tie-with-BHdr claim. The split caught it
  before promotion — this is exactly the winner's-curse control working.
- **C-class HDR path, sharpened to recipe/data only:** (a) BHdr+ refresh
  (372-PU linear refit — cheap, deterministic, no seed lottery); (b)
  RECIPE-level transfer-gated search (best_val↔transfer +0.78 across
  recipes supports recipe-level movement; seeds do not); (c) HDR training
  data expansion (the binding constraint; avif-HDR datagen gate). The
  confirm half remains UNREAD (virgin for the eventual recipe-level pick).

### BHdr+ REFRESH — registered arms (2026-08-29, pre-fit; the surviving C-class-HDR path)

Deterministic 372-PU linear refits on the shipped-BHdr chain (gram
hdr_v3mix.npz shaped, anchor.npz, screen tsv — `reproduce_bhdr.sh` argv):
- Arm λ: lam ∈ {1e-4, 3e-4 (= shipped control, byte-identity expected),
  1e-3, 3e-3} — the sparsity axis (the loupe lesson).
- Arm BVLS: `--solver bvls --bounds-tsv feature_sign_mask` (B's kon-head
  class; f372+ absent at 372 so the mask fully binds).
- SELECTION per the frozen HDR protocol: select-half UPIQ (pulinear
  select-half views, positional split, built + manifested) + the shipped
  control read on the same half. The CONFIRM half is spent ONLY if an arm
  beats the BHdr control on the select half — then the paired-vs-BHdr
  confirm read IS the promotion gate. Bars: promote-interest iff
  select-half |SROCC| > BHdr's select-half read; else record + close.
- No seed axis exists (convex deterministic fits) — the search is over
  recipes, where transfer moves recipe-level (+0.78 best_val correlation).

### BHdr+ VERDICT — no arm beats the shipped control; confirm half preserved (2026-08-29 ~21:2xZ)

Control reproduced BYTE-IDENTICALLY (lam 3e-4 = shipped sha). Select-half
reads: **shipped BHdr 0.7856** > lam1e-4 0.7712 > bvls 0.7570 > lam3e-3
0.7278 > lam1e-3 0.7174. Registered bar not met by any arm → record + close;
the CONFIRM half remains unread. Conclusion: BHdr's λ was already
near-optimal on its own gram — **the C-class-HDR gap is not reachable by
re-solving the existing HDR data.** The whole day's HDR arc now reads:
seed mining FALSIFIED (scene-idiosyncratic), same-gram refit FALSIFIED
(shipped pick optimal), transfer gate FROZEN, split-half protocol proven.
What remains is exactly the registered R3 hard path: NEW HDR training data
(hdr_v3mix is the only leg; avif-HDR datagen is user-gated) + recipe-level
search under the transfer gate + the 944-route/PU-route question. BHdr
stays the HDR default on merit.

## DID WE ERR DROPPING THE v1 IW/MASKED/PEAKS BLOCK FROM SCORING? (user question, 2026-08-29 — measured: PARTLY YES, on exactly one axis, and the "two models anyway" point is architecturally correct)

**What the record shows the drop was for** (720 feature-gap audit): NOT
diffmap zealotry alone — (a) the iw subfamily is structurally redundant on
basic+peaks (median R² 0.998, 71/72 ≥ 0.99; 0.97–1.49% of L0 mass in every
bake); (b) the measured A/B: 504-config (drop f156–371, add v2-348) BEAT
full-372 v1 nearly everywhere incl. KonJND +0.182 at that era; (c) keeping
it AND adding v2 (full-720) OVERFIT the small FR corpora (CSIQ −0.131,
LIVE −0.160); (d) the fold can never express its non-additive poolings.

**What is nonetheless true today (new measurements, `bake_contrib
--ablate-range 156..372` — joint exact block ablation, owner extended):**
- **The block is HALF of shipped B**: ablating it costs B cid22 −0.487,
  kadid-select −0.306, kon-504 −0.221, hfnl-valbucket −0.182. B without the
  block (kon 0.373) falls BELOW the 944 models — so v2 did replace much of
  the signal for models trained with it.
- **But not all of it on the near-threshold axis**: every 944 model trails
  full-B on kon-504 by ≥0.09 (CI), and B's kon mass is 48.6% block-borne.
  The 720-era "+0.182 kon" A/B result did NOT survive to the 944 era. The
  carriers, named (mean|Δ| on kon-504): **masked f237/f231/f243 > peaks
  f190/f226/f178/f196 > iw f333/f303/f321** — the "iwssim" label is a
  misnomer for what matters; masked+peaks dominate.
- **The "two models anyway" point stands**: the foldability constraint only
  ever needed to bind the MAP model. Since the scorer/map split (anchor-
  amber; the profile pairs), a SCORING regime never needed to give up
  non-foldable features. The remaining honest constraint is STREAMING
  (non-additive poolings need the buffered path), which is an opt-in cost,
  not a law.

**Registered consequence — R2's kon mechanism (b) is now concrete:** revive
the named carriers as an APPEND block (f944+, append-only discipline;
scoring-only regime, buffered path, fold-exempt because the map model is
separate). Gate unchanged: kon-504 ≥ 0.55 holding cid22 ≥ 0.885. This is
extractor work (the v1 masked/peaks/iw kernels exist; wire an appendix into
the extended path) + a retrain — the highest-evidence kon-closure lever on
the table.

## R2 EXECUTION — THE KON-COMPANION EXPERIMENT (registered pre-fit, 2026-08-29; user: "do the work all the way through to the final end")

**Design (the decision experiment for the carrier hypothesis at 944-class):**
combined scorer = small head over [C_score, the 10 named carriers
(masked f237/231/243, peaks f190/226/178/196, iw f333/303/321 — 372-root
values)], trained by THE trainer (n-hidden 0, 11 inputs, RankNet) on:
- konjnd_bpg train half (8,060 rows; BPG refs ⊥ the JPEG eval refs — CLEAN)
- kadid + tid full (T1 guards; frames min-maxed as in the B recipe)
- val group = konjnd_bpg val (%10∈{8,9})
C_score comes from north-anchor's canonical forward (bake_contrib
--dump-scores, parity-gated vs Predictor); carriers from the 372 tables;
row alignment across regimes VERIFIED per-corpus by target rank-identity /
(ref, target) join before any fit — misalignment aborts.

**Frozen bars:** PASS = combined kon-504 ≥ 0.55 AND combined cid22 ≥ 0.885
(C alone: 0.5006 / 0.8927; B: 0.5935 / 0.8764). STRETCH: kon-504 ≥ 0.5935
(match B). Eval surfaces are CLEAN of the fit (JPEG 504 refs never fit;
cid22-49 never fit). k=3 seeds — the k-seed spread is REPORTED (the
seed-stability discipline), selection by val group only.
**Falsifier:** no seed reaches both bars ⇒ the carriers are insufficient at
944-class in companion form ⇒ the 954 single-model fleet retrain loses its
cheap justification; record and close.
**If PASS:** name the companion, run the wider battery on the combined
scorer, and register the production step (two-stage scorer wiring vs the
954-regime fleet extraction) as the follow-on.

**Arm-1 result + open amendment (arm-2).** Tables verified exact (f0-only
reproduces C: kon 0.5006 / cid22 0.8927; carriers individually +0.12…+0.45
zero-shot on kon-504). Arm-1 (konbpg-trained) FAILS both bars on all 3 seeds
AND degrades kon below C (0.376–0.413) — BPG-half training poisons JPEG
transfer (consistent with R6's konbpg falsification and with B's kon being
training-clean). ARM-2 registered before running: same head, TRAIN =
kadid+tid rank only (no kon data — B's actual condition), val unchanged;
same bars, k=3 seeds.

**Arm-2 result + FINAL arm-3 (pre-declared).** Arm-2 lifts kon into B's zone
(0.489/0.532/0.517 — above C in 2 of 3 seeds; broad-fit carriers DO transfer
where kon-fit ones poisoned) but cid22 craters (0.838–0.850): the 2-corpus
fit surface cannot hold C's cid22 ordering. ARM-3 (FINAL — the falsifier
fires after it): add a self-distillation anchor leg (konbpg_tr pixels with
target := C's own score /100 — no human/kon labels consumed) at weight 2.0 to
pin f0-identity, kadid+tid rank 0.5/0.5 steer the carriers. Same bars, k=3.

### R2 KON-COMPANION — FINAL VERDICT: FALSIFIED (2026-08-29, ran to the registered end)

Nine fits across three pre-declared arms; no seed reaches both bars:

| arm (train data) | kon-504 (bar 0.55) | cid22 (bar 0.885) |
|---|---|---|
| 1: konbpg-trained | 0.376–0.413 (BELOW C) | 0.880–0.883 |
| 2: kadid+tid only | **0.489–0.532** (above C, 2/3) | 0.837–0.850 (craters) |
| 3: self-distill anchor + guards | 0.415–0.457 | 0.884–0.886 (2/3 pass) |

**What the frontier teaches (all recorded):** (1) kon-FIT carriers poison
JPEG transfer (arm-1 — consistent with R6's konbpg falsification and B's
training-clean kon); (2) broad-fit carriers DO lift kon at 944-class
(arm-2's 0.532 — the mechanism is partially real) but the 11-dim companion
surface cannot simultaneously hold C's cid22; (3) pinning identity (arm-3)
recovers cid22 and forfeits the kon lift. **B achieves both because its
carriers co-train with the full 372 set on full-breadth data — the carrier
VALUE is interaction-borne.** A bolt-on head structurally cannot reproduce
it.

**The lane's end state:** every cheap path to C-with-B's-kon is now
measured-closed (weight levers R6-K/R6-M; the companion arms 1–3). The one
remaining route is the **full 954-regime retrain** — carriers + all 944
features co-trained on the full legs — which requires the fleet
re-extraction campaign (new regime slots f944..953, all training legs, days
of LAN fleet + storage; zenfleet job class exists). Evidence FOR: the block
is half of B and 48.6% of its kon mass, and arm-2 proved partial transfer;
evidence tempering: the 720-era drop-vs-keep A/B favored dropping AT THAT
ERA, and interaction-dependence means the retrain is the only honest test
left. **The campaign is a real spend — registered as the R2 endpoint,
launch user-gated.** Until then the kon axis stays B's, priced at −0.09 in
the swap decision, with the pair/companion product shapes as the standing
alternatives. Artifacts: `bakes/koncompanion-2026-08-29/` (6 tables + 9
bakes + manifest), logs `~/tmp/koncomp*`.

### R2 AMENDMENT — THE 954 RETRAIN NEEDS NO NEW DATA (user correction #3, 2026-08-29)

"We never deleted the 372 features" — correct, and stronger than that: the
**ext720 root carries the UNFOLDED v1 block for the same legs, row-order
IDENTICAL to the ext944 legs** (cid22t/safesyn/kadid/tid verified 1:1), and
`tbig_720_full.parquet` (5.74M rows, the ENTIRE canonical corpus at 720,
keyed encode_sha) is on disk. The fleet-extraction claim is RETRACTED — the
954 experiment is a column-fusion job. Carrier width note: v1 features
DIVERGE across extraction widths (f237 median rel 7.6e-2, f333 2.05e-1 —
the padded-width divergence; f178 exactly 0) so ALL carriers come from ONE
width (the 720 extraction), evals included (ext720 cid22val + kon504 are
row-identical to ext944's). Legs without a 720-width source: kadis
(dropped, both arms — 2.4% of pairs) and konjnd_bpg (arm-B only, 372-width
carriers, flagged). ARM-A = C's recipe minus kadis/konbpg, all-954 fused;
ARM-B = arm-A + konbpg with 372-width carriers. Bars unchanged
(kon-504 ≥ 0.55 ∧ cid22 ≥ 0.885), k=2 seeds per arm.

**Arm-0 control (registered before any 954 result is read).** Arms A/B drop
kadis (both) and konbpg (A) relative to C — their deltas conflate
{carriers added} with {legs dropped}. ARM-0 = the SAME recipe as arm-A at
944 width (no carriers, same dropped legs), seeds 4004/4006, trained after
the A/B wave (serialized per machine-safety). The carrier effect = arm-A −
arm-0 at matched seeds; C's published numbers remain the ship reference.

### THE 954 VERDICT — carrier co-training FALSIFIED; the kon mechanism fully mapped (2026-08-30 ~01:2xZ, ran to the final end)

| model (kon504 / cid22) | s4004 | s4006 |
|---|---|---|
| C published (full recipe, 944) | 0.5006 / 0.8927 | — |
| arm-0 control (944, −kadis −konbpg) | 0.4204 / 0.8696 | 0.4487 / 0.8818 |
| arm-A (954+carriers, same drops) | 0.4420 / 0.8740 | 0.4488 / 0.8855 |
| arm-B (arm-A + konbpg@372w) | 0.4402 / 0.8805 | 0.4171 / 0.8776 |

- **Isolated carrier effect (arm-A − arm-0, matched seeds): kon +0.022/+0.000,
  cid22 +0.004/+0.004** — consistent with the joint ablation on the trained
  bake (kon −0.012 when removed): the optimizer, given the full 944 stack,
  assigns the carriers almost nothing. **Co-training does not rescue them.**
- **The leg-drop effect is the big term:** −kadis−konbpg alone costs kon
  0.05–0.08 vs C — CONFIRMING the standing "KADIS cracks KonJND" finding:
  kadis is a real kon ingredient in C's recipe (dropped here only for
  carrier-width purity).
- **konbpg at 954 adds nothing** (arm-B ≤ arm-A), consistent with every prior
  konbpg reading.

**The complete kon mechanism, end to end:** B's kon edge is NOT recoverable
by feature revival — the 944 feature stack spans the carriers' information
(they get no weight when co-trained), the companion form cancels
(interaction ≠ addable), and C's own kon rides partly on kadis data. The
residual B-edge source is B's era-composition (its mix incl. hdr_v3mix, the
linear inductive bias, min-maxed frames, 372-width numerics) — not a
transplantable component. **Every revival path is now measured-closed: the
carrier program ends here.** The kon axis remains B's at −0.09 in the swap
ledger. Remaining honest levers are DATA, not features: new near-threshold
human corpora (B7 KonFiG — overlap-audit CLEAN, runnable — is the on-disk
candidate) or accepting the priced trade. Artifacts:
`fused954-2026-08-29/` (12 legs + 6 bakes + manifest), evals `~/tmp/c954_*`,
`~/tmp/c944ctrl_*`.

## FLOAT-Q RE-CENSUS + C STABILITY (2026-08-30, user directives)

**Float-quality re-census (zenjpeg, step 0.25, same corpus9/judge/seeds —
`instrument-census-floatq-2026-08-30/`):**

| arm | k | median \|err\| | ±2 hits | integer-era |
|---|---|---|---|---|
| A anchor | 2 | 3.612 | 9/27 | 3.657 / 8 |
| A | 3 | **2.276** | **13/27** | 2.556 / 11 |
| B zq_seed | 2 | 2.297 | 13/27 | **1.905 / 14** |
| B | 3 | **1.426** | **17/27** | (k3 base in census md) |

Nuance worth the record: **float wins at k=3 (A −11%, B reaches 1.426/17)
but REGRESSES B at k=2** — with only two encodes the fine step
under-explores where integer rounding acted as a wider probe. Registered
consequence: the production default stays step 0.25 for k≥3 loops; 2-shot
budgets should use a coarser first step (hybrid schedule — a zenjpeg
follow-up, registered not run). G-GRAN v2's jpeg knob quantum: 0.25
(zenjpeg-owned loops), integer for foreign jpeg encoders.

**C stability (k=3 on-disk band + refinement verdict):** cid22
0.8788/0.8834/0.8927 (sd 0.006 — tight), **kon 0.4482/0.4744/0.5006
(spread 0.052 — the swap ledger's 0.5006 is the BEST of three seeds;
band median ≈0.474 ⇒ honest kon gap to B ≈ −0.12 at the median draw)**;
hfnl 0.699–0.734; composite 0.858–0.866. The 3-seed mean-score ENSEMBLE
(computed from stored per-pairs): cid22 0.8862 / kon 0.4814 — variance-free
but below the selected seed and 3× inference. **Refinement verdict: do NOT
ensemble; re-price the swap at the band (cid22 0.883±0.006, kon
0.474±0.026) and treat s4004's kon as selection-favored.** k=5 firming
seeds (s4007/s4008, exact argv) training now. Family axes for s4003/s4005
are pre-D1 cuts (flagged; cid22/kon are same-ruler).

## CRITERION 8 — ZENPICKER META-PICKER LANE (opened 2026-08-30, user GO)

Design (from the settled memos + criterion text): meta-picker = SOURCE
features ⊕ zq_norm → per-FAMILY bytes_log, masked argmin routes among the
production Zq loops. Pipeline, all from existing assets:
1. **Era-correct judge**: all 7 canonical picker datasets × 3 splits
   RESCORED under Profile B (`rescore_parquet --profile b`; 5.7M rows;
   first-cell gate: lossless rescores constant-high ✓) →
   `/mnt/v/output/zensim/picker-rescore-B-2026-08-30/` (21 files).
2. **Source features**: the existing `clean_features.tsv` (4,497 renditions,
   zenanalyze 0.2.0, provenance-stamped) — 36 size-gated columns
   (aq_map/noise_floor) are empty on the same 636 small renditions →
   COLUMNS dropped, every rendition kept ⇒ 61 features. No re-extraction
   (check-disk rule, third confirmation today).
3. **Meta-input builder** (committed:
   `scripts/canonical_corpus/build_metapicker_input_2026-08-30.py`):
   families become the trainer's cells via the plan-cell knob schema
   (`{"cell": family}`); rows keep q/score/bytes; splits stay the canonical
   origin-digit views. Built: train 2,946,036 / validate 1,764,808 / test
   1,031,816 rows, 0 missing renditions
   (`/mnt/v/output/zensim/metapicker-2026-08-30/` + manifest).
4. **Trainer**: `zenpicker-train --mode mlp` (the within-cell-optimal
   formulation, no q-leakage) — v1 seed 0 training now; origin-level
   honest panel via the EXISTING `--eval-bake` mode (a prior session's
   addition, confirmed not duplicated) on the validate view.
Gates for v1 (pre-declared): report argmin accuracy + byte-overhead
mean/p50/p90 on the origin-validate view vs the trivial baselines
(always-best-single-family; oracle=0 overhead). Ship/wire remains
user-gated per standing rules.

**C stability — k=5 FINAL (s4007/s4008 harvested 2026-08-30):** s4007 cid22
0.8929 / kon 0.4839 / nonphoto 0.9288 / hfnl 0.7087; s4008 0.8798 / 0.4291 /
0.9382 / 0.7193. Band: **cid22 median 0.8834 sd 0.0061 [0.8788–0.8929]; kon
median 0.4744 sd 0.0255 [0.4291–0.5006]**. The k=3 read replicates exactly.
SWAP-LEDGER RE-PRICE (final): C's honest kon = 0.474 (median draw), best-seed
0.501; vs B's 0.5935 the kon gap is **−0.12 at the median, −0.09 at the
selected seed**. cid22 remains a CI-solid C win at every draw (worst seed
0.8788 > B's 0.8764 only marginally — the worst-draw cid22 margin is ~0.002,
i.e. the cid22 claim is seed-robust in DIRECTION but thin at the band floor;
the +0.016 CI applies to the SELECTED seed). Refinement stance unchanged: no
ensemble; selection-aware pricing + the freeze battery is the control.

### LINEAR-QUESTION FINAL ARM (registered pre-fit, 2026-08-30): matched-mix kon heads 944 vs 954

Same legs/weights (safesyn 1.0 + cid22t 1.5 + kadid 0.5 + tid 0.5), same
solver (BVLS + sign-mask, f372+ free), same shaped space (screen944 +
identity rows for the carriers): ONE fit from the existing l944 grams, ONE
from a fresh 954 gram over the fused legs. Δ(kon504/cid22/hfnl) = the
carrier effect ON A TRUE LINEAR — the arm the trainer cannot produce (its
"linear" still inserts a hidden layer, verified). Read: if the 954 linear
recovers kon toward B, shaping-on-944(+carriers) enables a linear and 372
is NOT required; if not, the 372 requirement stands with the frame rebuild
(R4) as the only remaining linear lever.

### LINEAR-QUESTION ANSWERED (2026-08-30): 372 NOT required — the carriers enable the 944-class linear

Matched-mix TRUE-linear pair (BVLS shaped, safesyn 1.0/cid22t 1.5/kadid 0.5/
tid 0.5, sign-mask, identical everything except the 10 carrier columns):

| | kon-504 | cid22 | hfnl |
|---|---|---|---|
| 944 shaped, no carriers | 0.1644 | 0.8249 | 0.143 |
| **954 (+10 carriers)** | **0.4887** | **0.8502** | (no fused hfnl eval yet) |
| carrier effect | **+0.3243** | +0.0253 | |
| refs | B 0.5935 | B 0.8764 / wlin4 0.8502 | |

**Answer to the user's question: a good linear does NOT require the 372
front. Shaping on 944 alone does not enable it (kon 0.16); 944 + the v1
carriers DOES — one 954 head reaches C's kon band (0.489 ∈ [0.43, 0.50])
and equals the entire wlin4 blend on cid22, no blend needed.** This also
closes the loop on the whole kon saga: the carriers ARE the linear class's
kon backbone (matching the B-ablation's 48.6%), and the 954 MLP ignoring
them was an ARCHITECTURE-conditional outcome, not feature redundancy —
gradient descent with a hidden layer routes around them; a convex
sign-constrained fit cannot, and uses them. Fused-eval surface is
kon504+cid22 today (nonphoto/imazen26/hfnl validate slices cannot be
carrier-fused — no cross-root keys on the D1 cuts; registered gap, R1b's
keyed rebuild covers it). Next (same machinery, registered): full-mix 954
cid head (tbig/hf/teacher legs at 954) + blend sweep = the W-LIN
resurrection candidate against the round-6 bars.

### W-LIN 954 RESURRECTION — first candidates (2026-08-30)

The carrier discovery immediately re-opens the linear B-replacement lane.
(One repeated mistake caught in-run: the first cid-head fit mixed the RAW
hf leg at 1.0 — the R6 local-frame poison — dropped and refit.)

| candidate | kon-504 | cid22 | notes |
|---|---|---|---|
| head954_kon (standalone BVLS, 4 legs) | **0.4887** | 0.8502 | = wlin4's cid22 from ONE head; kon 2.4× wlin4 |
| **wlin954b l5e-4 α0.4 (blend)** | 0.4499 | **0.8551** | beats wlin4 on cid22 (+0.005) at kon +0.244 |
| wlin4_a0.5 (round-5 leader) | 0.2064 | 0.8502 | superseded on this surface |
| B | 0.5935 | 0.8764 | still ahead on both |

Round-6 bars: kon ≥0.40 PASSES both candidates; cid22 ≥0.845 passes the
blend; **hfnl/nonphoto/im26 bars are UNMEASURABLE at 954 today** (the D1
validate slices are keyless — no carrier fusion; R1b's keyed rebuild is the
unblock). Missing legs vs B's world: hdrmix (no 954 fusion — ext720 root
lacks it; buildable from the hdr944-leg + hdr_v3mix-720? registered check),
kadis (width), konjnd (width). The linear lane's gap to B: kon −0.10,
cid22 −0.021 — the closest a non-372 linear has come, from a 4-6 leg mix.

### ZENPICKER v1 — HONEST PANELS + BASELINE GATE (2026-08-30 ~02:4xZ)

Grid winner [128,128] lr 2e-3 (6-candidate bounded search, ranked by held-out
argmin). **Origin-validate view (held-out odd origins, 38,668 (image,zq)
rows, 245,402 cell pairs): argmin accuracy 0.7499 · byte overhead mean
4.47% / p50 0.00% / p90 14.52% · bytes-SROCC 0.9869.** Train-side grouped
panel agrees (0.702 / 4.6% / 0 / 14.4%) — no split-shift pathology.

**Pre-declared baseline gate — PASSED decisively.** Always-best-single-family
on the same view (coarse 5-target grid; picker numbers are dense-grid —
stated): best fixed choice = always-avif at **20.4% mean overhead** (p90
55.1%); every other fixed family is far worse (webp 60%, jxl 81%, jpeg 94%
mean). The picker's 4.47% is **4.5× better than the best possible fixed
routing**; median pick costs ZERO extra bytes. Interesting corpus fact
banked: avif reaches every (image,zq) cell and is the single-family
runner-up by a wide margin — consistent with the avif-priority directive.

Remaining criterion-8 steps: wire the bake into `zenpicker::MetaPicker`
(inert, flip user-gated), test-view touch-once at ship proposal, k-seed
spread report (the stability discipline applies to the picker too).
Artifacts: `/mnt/v/output/zensim/metapicker-2026-08-30/` (bake + toml
manifest + inputs + _MANIFEST).

### THE 10 CARRIERS — NAMED + COSTED (2026-08-30, user question)

Decoded from the v1 layout (metric.rs feature tables; peaks base 156, masked
base 228, iw base 300; scale-major × {X,Y,B} × slot):

| f | family | name | scale · ch | what it measures |
|---|---|---|---|---|
| f178 | peaks | art_l8 | s1 · X | L8-pooled edge ARTIFACT (ringing/blocking heavy tail) |
| f190 | peaks | art_l8 | s1 · B | 〃 |
| f196 | peaks | art_l8 | s2 · X | 〃 |
| f226 | peaks | art_l8 | s3 · B | 〃 |
| f231 | masked | masked_art_4th | s0 · X | L4 edge artifact × FLATNESS mask (artifacts in flat regions) |
| f237 | masked | masked_art_4th | s0 · Y | 〃 |
| f243 | masked | masked_art_4th | s0 · B | 〃 |
| f303 | iw | iw_art_4th | s0 · X | L4 edge artifact × IW weight (artifacts in TEXTURED regions) |
| f321 | iw | iw_art_4th | s1 · X | 〃 |
| f333 | iw | iw_art_4th | s1 · B | 〃 |

**All ten are the same physical quantity — edge-artifact error — under
extreme poolings**: heavy-tail peaks (L8), flat-region weighting, and
texture weighting, at fine scales. The near-threshold JND signal is
"ringing/blocking in its worst hiding places", which is why mean-pooled
stacks miss it and why B (which carries these slots) owns kon.

**Compute cost — MEASURED** (`zensim-bench extended_iw_perf`, 576², 15
iters, buffered path, rayon): standard-228 2.23 ms → +masked 2.73 → +IW
2.76 → **both 2.75 ms (1.24×, +0.52 ms/compare)**. The 4 peaks carriers are
FREE (always-computed accumulators in the base pass); the masked+IW passes
share their extra sweep (+0.5 ms covers all 6). Structural caveat: these
poolings are non-streaming — the extended regime is BUFFERED-path only
(memory shape, not speed, is the streaming constraint). Dev-only
`[patch.crates-io] zenanalyze` added to zensim-bench (zenjpeg 0.9.0's
unpublished dep; flagged, revert at release).

**hdrmix-954 status (checked):** no 720-width hdr_v3mix extraction exists
(the leg is HDR-route 944-native) — hdrmix stays absent from the 954 linear
mix until an extraction lands (registered; not blocking the candidates).

## CORRECTIONS + MEASUREMENTS (2026-08-30, user questions: buffering, peak memory, feature ids, bench quality, skip-unused)

**1. "Buffered-only" RETRACTED — the streaming pass already computes every carrier.**
`streaming.rs` accumulates `ssim_d8/edge_art8/edge_det8` (L8), the max
slots, `masked_*` (incl. `masked_art4`) and `iw_*` per strip and finalizes
`art_l8`, `masked_art_4th`, `iw_art_4th` (lines 405–425, 557–621). The
production folded path emits `[156..372) = 0.0` by POLICY ("deprecated: no
current model reads them", metric.rs:1684), not by inability. v1 is fully
streamable; the 944 layout simply drops the block. (The measured
"padded-width divergence" is buffered-372 vs streaming-720 numerics; the
fused carriers came from the streaming-720 family, consistent with the 944
legs.)

**2. Peak memory by feature set — MEASURED (`/usr/bin/time -v` max RSS):**

| path | 576² | 2048² | 4096² |
|---|---|---|---|
| streaming 944 (`foldapp_stream_bigpair`) | 17 MB / 0.01 s | 71 MB / 0.68 s | 188 MB / 2.78 s |
| buffered v1-372 (`extended_iw_perf`, 4-config process = upper bound) | 35 MB / 2.8 ms | 235 MB / 26.8 ms | 664 MB / 135 ms |

Inside the streaming pass the carriers cost accumulators only (no extra
planes: flatness/IW maps are already built for v2's masked/iw slots) —
the memory-optimal structure is the streaming pass with the block emitted.
Per-config buffered RSS (228 vs 372) NOT separable in that harness — not
measured.

**3. "Why add 10 features that already have ids" — CONCEDED.** f178/190/
196/226/231/237/243/303/321/333 are native slots of the 944 layout
(structural zeros). The discipline-correct design is to UN-ZERO the native
slots under a regime flag (zeroed vs filled rows never column-mixed), not to
append f944+. Rebuilt at native slots (`fused944native-2026-08-30/`, width
944) and re-ran the matched BVLS pair — at native slots the sign-mask PINS
the carriers ≥0 (B's own class): **kon 0.4570 / cid22 0.8726** (appended-
free: 0.4887/0.8502; no-carrier: 0.1644/0.8249). The carrier effect stands
either way (+0.29 kon); the pinned form trades kon for cid22 — 0.8726 is
within 0.004 of B. The "10 vs whole block" question resolves the same way:
emit the WHOLE block (free) and let dead-column pruning drop what a bake
doesn't use.

**4. Bench honesty:** the "+0.52 ms" was the existing `extended_iw_perf`
wall-clock loop (15 iters, synthetic pair) on the OLD buffered v1 path —
a real measurement but NOT zenbench-grade (no interleaving / paired stats)
and NOT the optimal structure. The correct instrument is a zenbench paired
A/B of the streaming pass with the block emitted vs zeroed (expected
noise-level) — registered as the next perf measurement.

**5. Skip-unused-blocks (user design ask):** family gates already exist as
config booleans (`compute_all_features` / `extended_features` /
`compute_iw_features`); the optimal extractor generalizes them to a
bake-derived FAMILY MASK (from `bake_block_profile` / the live-mask) so
extraction skips any v1/v2/append family the loaded bake does not read —
registered as the extractor design item, zenbench-gated per family.

## AVIF-HDR DATAGEN — GATE LIFTED (user, 2026-08-30) — registered plan + ground truth

**Ground truth (read from the estate, not the plan):** `hdrgrid-2026-08-06`
is ALREADY multi-codec — arms zenjxl / zenav1-svt (= AV1-HDR) / jpeg-gainmap,
1,140 PQ sources × 30 q, 102,485 encodes done, **hdrfeat944 features for
ALL THREE arms** (34,200 / 34,085 / 34,200) and cvvdp for jxl 80% / svt 65%.
The hdrgrid944 leg used zenjxl ONLY because **ssim2 is 0% on svt and
gainmap** — the GPU score wave failed: ledger 1,824 `worker_lost` (marked by
the GPU-only audit) + 72 `encoder_panic` on r7900x (the documented
exec-gpu-without-`hdr-gainmap` decode gap). The zenavif arm is ABSENT by the
B5 record ("additive-later; rows independent"). The executor's zenavif HDR
encode path is REAL (`sweep::hdr::encode_avif_hdr`, knobs lossless/speed).

**Plan (first-cell-gated, zenfleet only):**
1. Build + push a GPU executor image with `hdr-gainmap` (+avif HDR decode)
   — the gating step for every HDR score wave (local build, canonical
   package, new tag).
2. Re-declare the ssim2/butter GPU score waves for the svt + gainmap cells
   (gap/reconcile on the existing blobs — zero new encodes) → build the
   **3-arm HDR leg** immediately (jxl+svt+gainmap; the "multi-codec leg vs
   BHdr" the plan queued).
3. Declare the **zenavif arm** via `hdrgrid_cells.py` (+ `("zenavif",
   {"speed": <quality tier>})`, 34,200 encodes, hdr:true) → score → hdrfeat944
   → the 4-arm leg → the HDR model wave under the frozen UPIQ-transfer gate.
Every step persists encodes/metrics/diffmaps; first cell before scale-up.

### FULL-944 SPEED — zenbench-grade (2026-08-30, user question "how fast can 944 be if all 944 feats are computed")

`fused944_bench` (zenbench paired, serial, reference precomputed, 576²):
**score-only folded-944 extraction 16.8 ±2.2 ms** (fused score+map 40.3;
loop-today two-call 29.5). `v2_speed_baseline` (zenbench, pixels/s):

| size | v1-372 1-thr | v1-372 N-thr | v2 folded 1-thr | v2 folded N-thr | v2 with-ref 1-thr |
|---|---|---|---|---|---|
| 256² | 3.0 ms | 1.3 | 3.5 | 1.7 | 3.2 |
| 576² | 16.5 ms | 3.6 | 13.0 | 5.7 | 11.8 |
| 1024² | 43.7 ms (24 Mpx/s) | 7.3 (145 Mpx/s) | **129 ms** (8 Mpx/s) | **49.8** (21 Mpx/s) | 137 |

**Answer:** emitting the whole v1 block on top of the folded path costs at
most the masked+IW accumulator sweep — ≤0.5 ms at 576² (buffered upper
bound; expected smaller in-stream) — i.e. ≤3–4% of the pass. Full-944 ≈
**~17 ms serial / ~6 ms multithreaded at 576²; ~130 / ~50 ms at 1024²**.
The v1 block is NOT where the time goes: **the v2-348 block dominates and
scales badly** (at 1024² it is 3× v1 single-thread and 7× v1 multithreaded
— memory-bound behavior, the "bounded" strips). Implications registered:
(1) the skip-unused family mask pays on the v2/append side, not v1; (2) the
1024²+ v2 scaling is the perf lane worth a proper profile (`v2_stage_profile`
/ `v2_feature_group_cost` are the instruments). Logs:
`~/tmp/fused944_bench.log`, `~/tmp/v2_speed_baseline.log`.

### AVIF-HDR DATAGEN — STEP 1+2 EXECUTED: first-cell GPU gate PASSED (2026-08-30 ~04:4xZ)

**Ledger truth (the "failed GPU wave" re-read from `hdrgrid-sf-gpu-20260807`
ledger_snapshot + 71 parts):** r7900x-med attempted ONLY the zenjxl jobs
(456 attempts: 384 done, 72 `encoder_panic`); the 760 svt + 760 gainmap +
304 jxl rows marked `worker_lost` carry **attempts=0** and worker
`audit-blobs-scan` — the audit's `--scan-errors` requeue vocabulary, NOT a
lost worker. Their blobs (from the 08-26 image era) hold
`"error":"hdr decode: unsupported HDR input extension: .bin"` on every
variant (verified on the svt blob `cd5dd909…` and gainmap `11b29758…`) —
the extension-sniff gap of the exec-gpu-without-`hdr-gainmap` image.
So step 1 of the plan was ALREADY satisfied: the current GPU pin
`exec-gpu-cuda13-6d4f9963` decodes both blob types; no new GPU image needed.
A CPU HDR executor `exec-zensim944hdr-1f92f30c` (musl; sweep,png,jpeg,webp,
avif,jxl,cpu-metrics,hdr-gainmap,hdr-svt; digest `421d278e…`) was built +
pushed locally for the encode/feature arms and pinned in `fleet.env`
(zenmetrics `bacf504c`). Side find: two sessions had landed the same
`zenanalyze` `[patch.crates-io]` key in zenmetrics' Cargo.toml → every build
failed at manifest parse ("duplicate key"); deduped + pushed (`e93814d6`).

**First-cell gate (registered "first cell before scale-up"):** a 2-cell
manifest (`s3://zentrain/jobs/hdrgrid-sfgpu-probe-20260830/`, kind
`score_file [ssim2-gpu, iwssim-gpu] hdr:true`, one svt blob `6fc90aa4…`
+ one gainmap blob `0e9030c9…`) run via `lan_gpu_sequence.sh i134 gpu` on
the RTX 3070 (cuda13 image, `ZM_VRAM_CAP=7.5e9`): **pass 1 done=2 failed=0
in ~3 s**, blobs carry real scores, no `error` keys:

| cell | ssim2-gpu | iwssim-gpu |
|---|---|---|
| zenav1-svt · 1541_nature_blue-wildflower… scale768x1024 | 81.000 | 0.99358 |
| jpeg-gainmap · 1230_interiors_empty-tiled-pool… scale1200x1600 | 27.258 | 0.94885 |

**Residual wave design (job system's own tool, zero new encodes):**
`zenfleet-ctl gap --manifest <sf-gpu manifest> --ledger <snapshot + 71
parts> --out residual.json` → declare as a new run → `lan_gpu_sequence.sh
i134 gpu <run>`. Expected residual = 1,824 never-run + 72 jxl panics (one
retry each; genuine panics re-poison with current-era evidence). Then the
3-arm HDR leg, then the zenavif arm declare.

### CORRECTION (2026-08-30 ~05:1xZ): the AVIF-HDR "ground truth" above was STALE — steps 1+2 were ALREADY COMPLETE on 2026-08-27

Re-read from the estate, not from the 08-26 snapshot the section above relied on:
- `hdrgrid-sf-gpu-20260807` DRAINED on 2026-08-27 14:21Z. After the audit's
  requeue rows (attempts=0 over the `.bin`-extension error blobs) the LAN
  nomad pool (wsl/i134/r5600g) re-ran every svt + gainmap + jxl job
  (13:5x–14:3x UTC ledger parts) with real scores. `zenfleet-ctl gap` over the
  snapshot + all 71 parts: **0 of 2,291 remain**. The first-cell probe above
  reproduced two of those stored scores **bit-identically** (svt
  81.00002466634244 / 0.9935848616225961; gainmap 27.257822109597257 /
  0.9488483317924166) — so its actual value is: the cuda13 image + LAN
  sequencer path is validated for HDR AVIF/gainmap blobs (needed for the
  zenavif-arm score wave), and the stored svt/gainmap scores are confirmed
  GPU-exact. No residual wave exists to launch.
- `harvest-2026-08-27/scores.parquet` (post-drain full writeback, manifested)
  carries ssim2_gpu at **99.9% jxl / 99.6% svt / 99.5% gainmap** — the
  "ssim2 is 0% on svt and gainmap" line above was read off the 08-26
  jxl-only leg manifest (`coverage_note: all-zenjxl by score-wave size-tier
  design`), i.e. a pre-drain artifact.
- The 3-arm HDR legs EXIST: `hdrgrid-mc944-t2-2026-08-27` (era-B zensim
  target; 37,991/20,742) and `hdrgrid-mc944-t1-2026-08-27` (cvvdp-mix;
  41,788/22,860), built by the owner `build_hdrgrid_mc944_t2_leg.py`. The
  HDR-944 L1 wave RAN on them (`benchmarks/hdr944_bake_wave_2026-08-27.md`):
  BOTH arms pass the pre-registered UPIQ bar on all seeds; SELECTED
  `HDR944_L1T1_s4005` (UPIQ pooled +0.656, narwaria +0.605, korshunov +0.925,
  M3a 0.764, sel_comp 0.8867; packed 180,139 B, real 12-knot dial) = the
  "C-class HDR" candidate (codename aurora / CHdr); ship/freeze USER-GATED.
- What is genuinely NOT done in the HDR estate: the **zenavif arm** (B5
  "additive-later"; executor path `encode_avif_hdr` via zenrav1e, knobs
  lossless/speed) and any **aom** arm (no AV1 encode via zenav1-aom exists in
  the executor; zenav1-aom's `aom-encode` is ALLINTRA and byte-matches
  aomenc at every speed — a real encoder to wire). The svt arm exists for HDR
  ONLY (`encode_svt_hdr`, 10-bit PQ 4:2:0 CQP); there is no SDR svt/aom AVIF
  arm anywhere. That is the lane the user's "fresh avif encode with zen svt
  and zen aom backends" opens: new executor codec arms (svt SDR + aom SDR/HDR
  through zenavif-serialize), then declare.

## FRESH AVIF WAVE — SVT BACKEND (registered pre-launch, 2026-08-30 ~05:4xZ; user: "fresh avif encode with zen svt and zen aom backends")

**Owner discovery (before any code):** zenavif ALREADY carries the svt backend —
`Av1Backend::SvtRs` (`encode-svt-rs` feature; `src/encoder_svt_rs.rs`: the
pure-Rust zenav1-svt port as a 4:2:0 still encoder, 8/10-bit, arbitrary dims,
muxed in-crate by zenavif-serialize; speed 1..=10 → SVT preset 0..=13 linear).
The executor's SDR AVIF arm only ever built zenravif. Change (zenmetrics, this
commit): feature `avif-svt = ["avif", "zenavif/encode-svt-rs"]` + a `backend`
knob on the zenavif sweep arm (`"zenravif"` default | `"svt-rs"`; unknown →
loud error; `svt-rs` on a build without the feature → loud error, NEVER a silent
zenravif fallback; svt-rs forces 4:2:0 — the first-cell gate caught the
missing `chroma_subsampling` refusal before any fleet minute was spent).
zenavif's own `rd_core` planner has no backend stratum, so this wave is a
knob-grid wave (cell identity = `{"backend":"svt-rs","speed":S}`), emitted by
`scripts/jobsys/avifsvt_cells.py` → `zenfleet-ctl declare-encodes`.

**aom:** there is NO aom encode backend anywhere yet — `zenav1-aom/aom-encode`
has no frame-level driver (its parity tests hand-assemble KEY frames), zenavif's
`aom-backend` is DECODE-only, and the concurrent zenav1-aom session's Zq census
shells out to the `aomenc` CLI. Registered follow-up (separate chunk, after this
wave's first cell): an `aomenc` CLI arm baked into the executor image (libaom
allintra — the exact oracle zenav1-aom byte-matches at every `--cpu-used`;
box has 3.13.1, the port's oracle is 3.14.1 — pin one and record it).

**Wave spec (frozen):**
- sources: `train_renditions_2026-06-14` (the avif944 corpus sources; local
  `/mnt/v/output/imazen-26-features/train_renditions_2026-06-14/`, store
  `s3://zentrain/refs/train-renditions-2026-06-14/`, 1,482 renditions), minus
  the >16 MP monster tier (parsed from `.scaleWxH.`; avifgen excluded 27) →
  expected 1,455 sources. Train-side origins only (0/2/4/6/8) by construction —
  same split rule as avif944 (`origin_split.py`), never per-rendition.
- q: 1 + 5..70 step 5 + 72..100 step 2 = 30 points (avif944 grid).
- speeds: {4, 6, 8} (SVT presets ≈ 4/7/10); additive-later for more.
- cells: 1,455 × 3 × 30 = 130,950 encodes, codec `zenavif`, hdr:false.
- persistence: content-addressed AVIF bytes in `jobs/avifsvt-enc-20260830/blobs/`
  + Parquet ledger (job system), then the standard score waves (ssim2/butter GPU
  on the LAN GPU boxes via the validated cuda13 sequencer; cvvdp + 944 features
  CPU) → harvest → training views under the avif944 rules (AC.R1).
- gates: (1) first cell locally: encode OK + decode-back OK + bytes within 3× of
  the zenravif cell at the same q (sanity, not a bar); (2) first fleet chunk:
  blobs present + `error` rate 0 before scale-up; (3) encode-fail rate < 0.5%
  over the wave (avif944 residue reference); (4) orientation ladders (q↑ →
  bytes↑ monotone per source×speed) ≥ 99% before any table is built.
- images: CPU executor rebuilt with `avif-svt` (new tag; fleet.env pin bump).

### CORRECTION (2026-08-30 ~05:2xZ, user challenge: "why are you saying there is no zen aom encoder port")

The "no aom encode backend exists anywhere" line above was WRONG. The precise
state of `zenav1-aom` (read from PARITY.md + `aom-bench/src/lib.rs`):
- The encoder port EXISTS and is BYTE-IDENTICAL to real `aomenc` for ALLINTRA
  at every `--cpu-used 0..9` — landed gates `encoder_gate_e2e_byte_match`,
  `encoder_gate_real_image_e2e_kb6_repro` (real content 30/30, partial-SB
  edges), `encoder_gate_speed{1..9}_textured_allintra`, chroma-subsampling e2e.
- Its end-to-end driver is `aom_bench::EncodeCell::port_encode(&self,
  bootstrap)`: strided copy + border extension + the full SB search + pack walk
  + LF-level search (+ restoration search) + OBU assembly, returning the frame
  OBU payload. What it still takes from a reference aomenc stream is the
  **sequence-header + uncompressed-frame-header FIELD PARSE** (bootstrap; the
  coding DECISIONS are never copied — rule 4 "no bootstrap leak"). So the gap
  is a standalone header derivation + a public frame-level API (y/u/v + cq +
  speed → bytes), not the encoder.
- What I had grepped for (a top-level `pub fn encode_*` in `aom-encode`) does
  not exist; the driver lives in `aom-bench`. Wrong inference from a narrow
  grep — the ledger said "byte-matches aomenc at every speed" and that was the
  fact to trust.
Consequence for the datagen lane: an `aomenc --allintra` arm produces the SAME
bytes the port produces (byte-identical by the landed gates at every speed), so
the "zen aom backend" data can be generated now through the CLI — either
directly, or via `port_encode` with the aomenc stream as its bootstrap (the
parity-preserving variant, which also exercises the port on every cell). The
executor arm + a baked libaom (the port's oracle is v3.14.1; the box has 3.13.1
— pin the oracle version in the image) is the registered chunk after the svt
wave clears its first fleet chunk.

### SVT WAVE — GATES 1+2 PASSED, SCALED (2026-08-30 ~05:1xZ)
- Gate 1 (local first cell, 1552.scale1024x745, speed 6, q 30/60/90): svt-rs
  3/3 encode+decode OK; bytes 2,350 / 6,185 / 22,193 vs zenravif 2,720 / 6,383 /
  31,680 (0.70-0.97×); encode 28-57 ms vs 1,392-1,620 ms (~40× faster at the
  same nominal speed); zensim 38.2/65.9/80.7 vs 49.7/73.8/86.4 — the two
  backends map `q` differently (documented in zenavif), so `q` is a knob, not a
  quality equivalence.
- Image trap caught by the first fleet chunk: the executor image overlaid the
  glibc-built `zenfleet-worker` (Ubuntu 26.04 box → GLIBC_2.38/2.39 symbols)
  onto the 24.04 base → every pass rc=1 before a cell ran. Fixed at the owner
  (`build_executor_image.sh` prefers the static musl worker, warns on a dynamic
  overlay; zenmetrics `5589dd4d`); `exec-zensim944hdr-c4b01933` re-pushed
  (digest `3017f9f2…`).
- Gate 2 (first fleet chunk, r5900xt 32c): ledger 10,004 done / 0 failed at the
  first read; blobs are `ftypavif`; compacted to a snapshot at 14,842 distinct
  done; scaled to r7900x (24c). Both boxes were idle before load (observed:
  i134 + r5600g carry other sessions' workers, i265 a fleetbench — left alone).
  Run: `s3://zentrain/jobs/avifsvt-enc-20260830/`.

### AOM ARM — REGISTERED (user directive 2026-08-30: "i want our rust code exercised")
Design (executor `backend: "aom-rs"`, feature `avif-aom`, zenmetrics): every
cell runs the zenav1-aom port end to end
(`aom_bench::EncodeCell::port_encode`) with the pinned C libaom v3.14.1 oracle
(`aom-sys-ref`, cmake-built by its build.rs) supplying ONLY the header-field
bootstrap (`c_encode_defaults` = plain `aomenc --allintra` defaults: cdef OFF,
restoration ON, qm OFF); the port's frame OBU payload MUST equal the oracle's
byte-for-byte or the cell fails loud (no unverified bytes ever emitted); the
emitted AVIF = the port's payload spliced into the OBU frame (TDs dropped),
muxed by zenavif-serialize with BT.601 full-range nclx. Colour path = `zenyuv`
(THE owner; same BT.601 full-range 4:2:0 convention as zenavif's svt-rs
backend). Mapping recorded: `cq_level = round((100−q)·63/100)` clamped 1..63;
`speed` = `--cpu-used` 0..9 (default 6). Wave grid (frozen, mirrors the svt
wave): 1,455 sources × speeds {4, 6, 8} × 30 q = 130,950 cells,
`s3://zentrain/jobs/avifaom-enc-20260830/`. Gates: first cell local (port ==
oracle, decode-back OK), first fleet chunk (blobs + 0 errors), encode-fail
< 0.5%, orientation ladders ≥ 99%. The oracle build makes this arm's executor a
glibc (not musl) build — image path to be settled at the first-cell gate.

### AOM-RS ARM — FIRST-CELL GATE RESULT: THE PORT IS BYTE-IDENTICAL EVERYWHERE EXCEPT THE 720p BAND (2026-08-30 ~05:5xZ)
Executor arm landed (zenmetrics `2690d66e`, feature `avif-aom`, `backend: "aom-rs"`):
every cell = zenav1-aom `port_encode` + the v3.14.1 oracle bootstrap, refuse on
any payload mismatch. The refusal FIRED on the first cell (1552.scale1024x745,
all q). Localisation (speed 6, restoration-ON defaults bootstrap; q 30/60/90):
192×256, 384×206, 384×512, 576×768, 1536×1152, 1536×2048, 1730×3072, 3000×4000
= **byte-identical (24/24 cells)**; **1024×745 and 1280×800 = DIVERGED (6/6)**.
Same 1024×745 image: centre crops 196²/352×288/512×384/640×480 identical in all
four arms (speed 0/6 × restoration on/off); restoration-OFF bootstrap narrows
the full frame's divergence to q90 (cq 6) only. Pattern = libaom's
`is_720p_or_larger && !is_1080p_or_larger` framesize-dependent ALLINTRA sf arm
(min-dim 576 pass, 745/800 fail, 1152+ pass). The port runs FAST enough for
datagen (3 cells + 3 oracle encodes at 1024×745 in 0.34 s wall).
**Filed: imazen/zenav1-aom#14** (repro cells + hypothesis; assigned).
**Wave envelope decision:** declare the aom-rs wave over the svt-wave sources
EXCLUDING renditions with 720 ≤ min(w,h) < 1080 (recorded as a deferred
residual, content-addressed, re-declared additively when #14 lands); every
declared cell is still byte-verified per cell, so any further divergence
surfaces as a loud failure, never as unverified data.

### AOM-RS ARM — EXECUTOR IMAGE + FIRST FLEET CHUNK (2026-08-30 ~06:0xZ)
- Wave declared: `s3://zentrain/jobs/avifaom-enc-20260830/` — 1,404 sources × {4,6,8}
  × 30 q = **126,360 cells** (`avifsvt_cells.py --backend aom-rs --exclude-min-dim
  720:1080`; the 51 band renditions in `cells_aom_declare.jsonl.excluded.txt`,
  `/mnt/v/output/avifaom-2026-08-30/`).
- Static musl executor with the arm: the glibc-built libaom oracle objects
  reference `fopen64` (glibc LFS symbol; musl exports none) → linked a 1-line
  `fopen64→fopen` shim as a bare OBJECT on `*-musl` targets only (an archive was
  never searched: rustc orders `-l static=` before the rlibs carrying the oracle
  objects — first attempt left every reference unresolved). Also stamps
  `ZEN_CODEC_ZENAV1_AOM_COMMIT` so every aom-rs cell carries the port commit it
  was byte-verified at. Smoke (musl binary, 384×512 s6 q60): 1/1 cell, port ==
  oracle, 30,054 B, 282 ms. zenmetrics `fca352dd`.
- Image `ghcr.io/imazen/zenfleet-worker:exec-zensim944hdr-fca352dd` (digest
  `44f4d55d…`; superset of c4b01933: avif-svt + avif-aom + HDR arms), pinned in
  `fleet.env` (`ba73a466`).
- First fleet chunk launched on r3500 (6c; the only idle LAN CPU box — r5900xt +
  r7900x carry the svt wave, i134/r5600g/i265 other sessions' workers).
  Gate: blobs present + 0 errors before scale-up.
- svt wave at launch time: 48,131 / 130,950 blobs, ledger 49,487 done / 0 failed.

### AOM-RS WAVE — FIRST FLEET CHUNK: 199 VERIFIED / 67 REFUSED; the port's envelope is patchier than the census showed (2026-08-30 ~06:1xZ)
r3500 first 266 rows: **199 done (byte-identical port encodes), 67 `encoder_panic`
= the arm's PORT-DIVERGED refusal** (verified by local repro of every failing
class: never a panic). Refusals cluster by (image, speed, q): tiny frames
(85×128, 128×80, 128×128) at specific cq at s6/s8 with byte deltas of ±1–25 B,
and **1920×1080 at s6 with the port emitting 2–4× the oracle's bytes** (s8
~1.3×) — a tool family missing/mis-gated at min-dim = 1080 while ≥1152 and
≤576 matched. The 10-point speed-6 census was too coarse to see either.
Posted to imazen/zenav1-aom#14 (comment with the cell table).
**Decision:** the wave runs on as a census — each cell is either a verified port
encode or a recorded divergence (ledger `failed/encoder_panic` rows carry
(w,h,speed,q)); no scale-up beyond idle boxes (r3500 now; the svt boxes roll
onto this run when svt drains). Expected: ~25% refusals × one retry. The
verified cells are exactly the "zen aom backend" data the user asked for;
the refusal map is the port session's next worklist.

## DIVERGENCE WORK (user: "work on divergences", 2026-08-30 ~06:0x–06:4xZ) — KB-41: the refusals were ALL screen-detected frames; harness mismatch closed at the arm, port residual pinned with tiny reproducers
- Claimed `zenav1-aom` (stale 24 h marker, clean tree; worked on `main@origin`), read
  its differential playbook (§10 "diagnose to the decision") and the framesize KBs
  (22/26/28/36/38) — none matched. The decisive read was the CONTENT: 8172.scale1920x1080
  is a mailing-list screenshot; 1280x800 a NASA page; 128x80 a tiny screenshot; 128x128 and
  85x128 product shots on flat white; 1024x745 a dark sunset photo.
- **Localizer `crates/aom-bench/tests/kb41_screen_detected_defaults.rs`** (on-demand,
  `ZENAV1_PLANES_DIR` = the exact planes the arm fed both encoders, dumped by the new
  `ZEN_AOMRS_DUMP_PLANES` knob): reads the oracle stream's own
  `allow_screen_content_tools` — **30/30 refused cells have it set** (libaom's
  anti-aliasing-aware detector flags the dark photo too). The arm drove the port with
  `ToggleKnobs::default()` (palette+IntraBC OFF) against `c_encode_defaults` (libaom
  ALLINTRA: both ON). **My "720p band" hypothesis in #14 was WRONG** — corrected on the
  issue (comment 2) and in the zenav1-aom KB-41 entry; the passing sizes were simply
  frames the detector left alone.
- **Arm fix (zenmetrics `a90eb727`):** mirror the header bit into the port's knobs. Effect
  (cpu 6/8, cq {6,32,57}): 128x128 6/6 and 128x80 5/6 byte-identical; 85x128 1/6 (five
  cells: equal length, different bytes; `ZENAV1_DECODE_BOTH=1` shows the first recon
  divergence at luma (0,87)/(17,61)/(48,32)); 1024x745 +6/+8/−32 B; 1280x800 ≈ −2 kB;
  1920x1080 −9..−20 kB (from +32..+231 kB). The residual = the port's palette/IntraBC
  search fidelity on real screen content (KB-15 / KB-P29 / PALETTE_MANY_COLORS_OPEN) — now
  with an 85x128 cq57 cpu6 reproducer (267 B both sides). Landed on zenav1-aom main
  (`4e9c2d22`: localizer + KB-41 entry + T4 row).
- Fleet: image `exec-zensim944hdr-a90eb727` (digest `98cc36ac…`); the 1,575
  harness-mismatch-era `encoder_panic` rows were pardoned (`zenfleet-ctl requeue`,
  documented broken-era window) and r3500 relaunched; **svt wave DRAINED** (gap 0/130,950;
  pairs table 130,590 DONE cells) → r5900xt + r7900x rolled onto the aom run;
  svt score waves declared: `avifsvt-sf-gpu-20260830` (ssim2-gpu+butteraugli-gpu, 11,608
  jobs, on i134 via the validated cuda13 sequencer) and `avifsvt-sf-cpu-20260830`
  (cvvdp+zensim-foldapp2 944 features, 11,608 jobs, CPU pool after the aom encodes drain).
- Perf note for the port lane: with IntraBC on, the port's 1920x1080 cells took ~80 s each
  in the localizer (the oracle: ~1 s) — the screen-tools search is far outside Gate 3.

### KB-41 ROOT CAUSE + FIX LANDED IN THE PORT (2026-08-30 ~07:1xZ) — two palette cost-table plumbing defects; four pinned divergence classes close
Method (playbook §10 to the decision): the localizer's new decode-side SYNTAX diff put the
first divergent block of `85x128 cq32 cpu8` at mi(22,6) — port took a 3-colour UV palette,
C kept SMOOTH; a sibling instrumented libaom (`~/tmp/libaom-instr`, fprintf in
`av1_rd_pick_intra_sbuv_mode` / `av1_rd_pick_palette_intra_sbuv` / `intra_mode_info_cost_uv`)
vs matching port dumps: every tokenonly rate/dist equal; only the palette HEADER rate differed.
Kernels excluded first: `palette_shim.c` exports the DISPATCHED `av1_calc_indices_dim{1,2}` /
`av1_k_means_dim{1,2}`; `palette_kmeans_diff.rs` = 1,050 cases bit-identical.
- Root 1: `palette_uv_mode_cost` NEVER FILLED (the fill fn had a unit differential and zero
  callers) → UV palette flag cost 0 where C has 23 (no palette) / 2592 (palette).
- Root 2: palette size/colour-index cost tables read from the FRAME-INIT tables for the whole
  frame while every other mode cost follows the per-SB/SB-row `INTERNAL_COST_UPD_*` refresh
  (979 vs 864 = default-CDF vs adapted cost).
Fix (zenav1-aom, one commit, pushed after the gate sweep): `real_costs.rs` fills the flag table;
`pack.rs` routes `sb_real.palette_costs` into the per-SB `PickFrameCfg`.
Closures (self-promoting pins fired and were re-pinned): `85x128 cq32 cpu8` byte-identical;
`rd_close_palette` `text_420_128_cq20` (a "genuine near-tie" since the palette landing) →
BYTE_EXACT; **`PALETTE_ON_SPEED8_OPEN` CLOSED** (9/9 rows); **the full-RD half of
`PALETTE_MANY_COLORS_OPEN` CLEAN** (9/9); **`SCREEN_ARRAY_OPEN_ROWS` CLOSED** (1/1);
`encoder_gate_e2e_byte_match` 32/32 unchanged. Remaining: `ui_420_128_cq32` (pinned near-tie),
the speed-9 `fc256 n40 cq40 −1 B` row, and whatever the re-running KB-41 census still shows.
Datagen note: with IntraBC mirrored ON for screen-detected frames the PORT's DV search is the
wave's bottleneck (1080p screenshots ~80 s/cell at cpu6 vs ~1 s for the oracle; cpu4 being
timed) — the aom wave's done-rate fell to ~0/min for a 10-min window while the boxes sat at
full load. Gate-3 territory for the port; a size/screen-tier scoping decision for the wave.

### AOM-RS WAVE — THROUGHPUT DECISION: screen-detected frames above 0.25 MP DEFERRED (2026-08-30 ~07:4xZ)
Measured: with palette+IntraBC mirrored ON, the port's IntraBC DV search is the wave's
bottleneck — 1920x1080 screenshot ≈ 80 s/cell at cpu6, and at **cpu4 a single cell had not
finished after 40 min** (killed; the oracle codes the same cell in ~1 s). After the mirror
image went live the three encode boxes sat at full load and wrote NO ledger rows for ~35 min
(in-flight cells), done-rate ~280/min → ~0. Decision: executor knob
`ZEN_AOMRS_MAX_SCREEN_MP` (default 0.25 MP): a screen-detected frame above the cap is
REFUSED with a distinct "DEFERRED … not a divergence" message (ledger `failed` row →
pardon + re-declare, content-addressed, when the port's search is fast); below the cap the
port is exercised + byte-verified as before. Non-screen frames are unaffected. This is Gate-3
territory for the port (KB-41 perf note in zenav1-aom's CLAUDE.md); the deferred set is
recoverable at zero data cost.

### AOM-RS WAVE RELAUNCHED ON THE FIXED PORT (2026-08-30 07:06Z) — image `exec-zensim944hdr-361864f9`
Chain: image build+push (first push hit a ghcr.io i/o timeout; retried), `zenfleet-ctl requeue
--classes encoder_panic` (the pre-fix refusals), compact (21,656 distinct done), relaunch
r3500 / r5900xt / r7900x. First 30 s: +900 blobs (the pardoned tiny screen cells verifying);
first chunk rows on the new image: 380 done / 30 refused (7%; deferred-vs-divergence split
is in the executor stderr, not the ledger — classified by local repro). Blob count 24,071 /
126,360. The svt GPU score run on i134 was relaunched with `ZEN_PASS_TIMEOUT=7200` + a ledger
snapshot after its first pass hit the 1,800 s pass limit on a big chunk (a slow chunk, not a
hang — the worker released the claim and moved on); the svt CPU score run (cvvdp + 944
features) stays queued until a CPU box frees up.

### KB-41 CLOSED — the aom-rs port is byte-exact on every datagen census cell (2026-08-30, zenav1-aom `735a0a6d`)
The six cpu-6 residuals (1920x1080 cq6/32/57, 1280x800 cq6/25/44) were FOUR unported speed
features on the IntraBC path, each localized to one block decision by the sibling-C + port
dumps and re-measured after each fix (1920x1080 cq57 s6: −8,887 B → −637 → −618 → +77 →
byte-exact):
1. the port ran libaom's speed-0 IntraBC DV search at every speed (`intrabc_search_level` 1
   at ≥6: 4x4/8x8/16x16 only, ABOVE-only, pixel search only on a hash miss; hash-8x8 cap at ≥4;
   64-candidate prune + doubled mesh thresh at ≥1; DIAMOND site config at ≥3 with NO mesh; the
   speed≤2 resolution×qindex search-method bands; `use_downsampled_sad=2` for ≥720p at EVERY
   speed) — the witness block's level-0 pixel search overrode a valid hash match with an invalid
   lower-cost point and dropped the IntraBC candidate C coded;
2. the intrabc coeff arm used pixel-domain distortion at every speed (C: DEFAULT_EVAL
   transform-domain type 1/2 from speed 1, `predict_dc_level` 1 at ≥6);
3. the intrabc var-tx knobs were speed-0 constants (inter init depth 1 at ≥1, ml split 4000,
   PRUNE_2/3 rows, `skip_tx_search`, framesize `prune_tx_type_using_stats`);
4. C re-derives an IntraBC coeff block's skip flag at ENCODE time (`av1_encode_sb`: all-zero
   after the encode-pass trellis → skip=1); the port wrote the search's skip=0 + coefficients.
Final census **30/30 byte-identical**; 88/88 palette/screen/HD gates; 311/311 aom-encode unit
tests; bonus: `rd_close_intrabc`'s pin promoted the KB-15 cell `scc_480x180_196_cq48`. The
`ZEN_AOMRS_MAX_SCREEN_MP` datagen cap stays (a throughput cap, not a correctness one — the
port's DV search is still ~80 s/1080p cell at cpu6). Not ported: BIGDIA (unreachable on
allintra), the speed-4/5 `prune_tx_type_est_rd` arm. Record: zenav1-aom `CLAUDE.md` KB-41
roots #3-#6, `PARITY.md`.

### KB-41 ROOTS #7-#13 — the census widened past the datagen cells and every residual closed (2026-08-30 ~09:0x–13:0xZ, zenav1-aom `38a92657`)
Widening the localizer to the two 85x128 repro dirs + the 1280x800/1024x745 band + the six
1920x1080 aomplanes cells exposed seven more C mechanisms, each localized by the sibling-C +
port dumps and closed in turn (census 30/30 → **57/57 byte-identical**, 0 DIV, 0 panics):
- **#7** the SEARCH-time `allow_intrabc` is the screen detector's decision
  (`estimate_screen_content_antialiasing_aware`, now ported as `screen_detect.rs`); C flips the
  header to 0 only after the tiles when no block used IntraBC (encodeframe.c:2443), so the
  port had been searching with the FINAL bit and paying the wrong `intrabc_cost`.
- **#8/#9/#12** three search-ctx CDF shadows (`TileCtxState::search_*`): C refills the search
  costs per SB from `xd->tile_ctx`, which `update_stats` adapts under different gates than the
  writer — `intrabc_cdf` under the search-time allow; palette-Y flag/size only for
  chroma-reference intra blocks (`av1_sum_intra_stats` early return; IntraBC winners are
  `is_inter_block` and skip it — the missing exclusion cost a 1280x800 s6 regression);
  `tx_size_cdf` for EVERY coded intra block at the search-time `TX_MODE_SELECT` even when the
  header ends `TX_MODE_LARGEST` via the `txb_split_count == 0` flip (encodeframe.c:2797) — the
  85x128 cq62 s8 non-screen cell (winner-mode tx-size cost 220 vs 42 at ctx 2). All three sit
  behind `allow_update_cdf` (the `cdf0` config-permutation cells caught the missing gate).
- **#10** allintra speed >= 8 never runs the DV search (`rd_pick_intrabc_mode_sb` returns under
  `rt_sf.use_nonrd_pick_mode`, rdopt.c:3432-3434) — the port used IntraBC in 6,857 blocks of
  1920x1080 cq32 s8 where the oracle used none. `use_nonrd_pick_mode` + `mv_sf.use_intrabc`
  modelled; the {7,9} speed-feature class split (re-pinned from the C source).
- **#11** frame-edge HORZ_4/VERT_4 code fewer than 4 strips (`rd_pick_4partition` breaks at the
  first out-of-frame strip, partition_search.c:3948); the port's all-4-strips envelope guard is
  gone and `SbTree::{Horz4,Vert4}` carry `Option` strips — 85x128 cq19 s4.
- **#13** `av1_set_screen_content_options` arm order: seq-forced → `--tune-content=screen`
  (new `ToggleKnobs::tune_content_screen`, kb37's reference) → detection OFF at speed 9
  (`use_nonrd_pick_mode && !hybrid_intra_pickmode`, kb35's control) → detector. NOT ported:
  the two-pass trial encode `av1_determine_sc_tools_with_encoding` (live on allintra < speed 8
  when the detector says off; the bench asserts the decision against the header and names it).
Gates after the closure: config_permutations 87/87 (speed classes re-pinned: 7/8/9 now distinct),
rd_close_palette 2/2 (the last pinned near-tie `ui_420_128_cq32` promoted to byte-exact — the
"near-ties" were wrong search costs), rd_close_intrabc, toggles 25/25, kb22 (+3 ignored-arm
cells), kb35 3/3, kb36 2/2, kb37 3/3, aom-encode lib 85/85. Record: zenav1-aom `PARITY.md` row
"KB-41 roots #7-#13", `CLAUDE.md` KB-41 entry. The datagen wave (below) still runs the
roots-#3-#6 image; the encoder_panic cells it refuses are re-queued after this lands.

### KB-30 CLOSED + THE 512² SCREEN CAP IS TOO LOW (2026-08-30 ~11:0xZ, zenav1-aom `07d207ba`)
`cid22_6292444` (the one CID22 photo that diverged at every quantizer at cpu 6 since 2026-08-01)
is a SCREEN-DETECTED frame: the oracle's ALLINTRA defaults turn palette + IntraBC on for it, and
the old xbench arm ran the port with both off. Replayed through the executor's own plane dump
(`ZEN_AOMRS_DUMP_PLANES`) + the KB-41 localizer on the roots-#7-#13 port: **11/11 byte-identical**
(cq 1..60, cpu 6); the tools-off arm reproduces the old −0.25..−3.7 % byte deltas exactly. No
per-block dump needed. MEASURED encode+verify time for that 512² screen-detected cell: **0.12 s /
0.38 s / 3.2 s at cpu 8 / 6 / 4** — so the `ZEN_AOMRS_MAX_SCREEN_MP` 0.25 MP cap (set for the
40-min 1080p cpu-4 case) is deferring the 512² renditions (0.262 MP) for nothing. Default raised to
0.30 MP in zenmetrics (`encode.rs`); the running wave keeps 0.25 (a relaunch would orphan its
in-flight claims); the DEFERRED class is re-queued with the raised cap at the drain.

### ROOTS #7-#13 FALLOUT: TWO MORE LONG-OPEN BANDS CLOSED WITHOUT NEW CODE (2026-08-30 ~11:3xZ, zenav1-aom `799de43f`)
Replaying the port's other self-promoting pins on the `38a92657` tree: **`RD_BAND_OPEN`** (1272x724
cq24, open since 2026-08-02 at cpu 2..5 with −14/−104/−167/−189 B) is byte-exact at every RD speed
0..7 and the cpu-9 control is exact while reaching 22 non-square leaves — the reach matches root
#11 (a partial superblock on both frame edges; the 4-way partition types are searched exactly at
2..5). **KB-13's last two real-content cells** (`00-quantizer-00` 128²@64,64 and `23-film_grain-50`
64²@96,64, both cpu3 cq63, open since 2026-08-03) are byte-exact too — 77 B and 60 B on both sides,
not screen-detected — consistent with root #12 (at cq63 nothing splits, so the header ends
TX_MODE_LARGEST while the search ran at SELECT). Neither attribution is bite-proved by per-root
revert. New self-promoting replay `kb13_cpu3_cq63.rs`; the remaining pins (mono s0, HBD, HD
format/speed, crop, tiles, nonrd size bands) are being swept the same way.

### KB-41 ROOTS #14-#17 — THE cpu-4 SCREEN PROBE FOUND A NON-CONFORMANT STREAM (2026-08-30 ~11:4xZ, zenav1-aom `4e0229e1`)
Timing the 1280x800 screen cell at cpu 4 for the cap policy (11 s for oracle + both port arms —
the "40 min" figure predates roots #3-#6's search gating) turned up the first non-conformant
stream the port ever produced: both libaom and the port decoder rejected it ("intrabc DV failed
validity"). The wave never wrote one (every cell is byte-verified against the oracle and refused on
mismatch), and cpu 4 screen cells above 0.25 MP were deferred anyway. Root #14: with
`allow_intrabc` C skips EVERY post-filter stage (`if (!allow_intrabc) loopfilter_frame()`,
encoder.c:3780) and the header codes no restoration params, but the port still ran its restoration
search at speed <= 5 and the re-pack wrote 54 LR units the header never announced. Then three RD
roots on the now-decodable cell: #15 the var-tx first child is bounded by `ref_best_rd - 0` (C never
recomputes rdcost after setting the partition-flag rate); #16 the speed-4/5 est-rd tx-type prune
(`prune_txk_type` / `_separ`) replaces the 2D-NN prune on inter/IntraBC blocks — ported; #17 the
pixel-domain-final downgrade reads the POST-prune mask. Cell byte-exact after the four. Method note:
the writer/reader block-sequence probe (`PACK`/`DEC` per block) localized the desync to the first
partition symbol in one run; the header field-diff (`read_uncompressed_header` on both TUs, 1060
lines equal) then proved it was tile data, and the pass-tag probe proved the spliced bytes came
from the LR re-pack. Widened census (the s6 planes replayed at s4/s5 against a fresh oracle): s4 6/9 exact (open: 1280x800 cq44 +10 B, 1920x1080 cq57 +10 B, 1920x1080 cq6 −33 B), s5 1/3 (open: 1280x800 cq25 −5 B, 1920x1080 cq32 −123 B) — RD-level residuals, decodable streams; all 68 prior census cells and every gate unchanged.

### AOM-RS WAVE RELAUNCHED ON THE BYTE-EXACT PORT (2026-08-30 08:39Z) — image `exec-zensim944hdr-47f5ab9c`
zenmetrics master `0bcede27` (fleet.env CPU pin bumped). Chain: musl rebuild (the new port's unique
strings confirmed in the binary) → image build + push (digest `5f4f9bd5…`) → `zenfleet-ctl requeue
--classes encoder_panic` (**5,312** pre-fix refusals + deferrals pardoned) → compact (96,881 distinct
done) → relaunch r3500 / r5900xt / r7900x. Ledger at relaunch: aom-rs 96,881/126,360 done;
svt encode 130,590/130,950; svt GPU score (i134, `zen-score-gpusf`) 5,136/11,608 done + 113 failed
(61 encoder_panic + 52 oom — to classify + requeue after the drain; the oom class wants a bigger
`ZEN_VRAM_CAP` pass); svt CPU score (cvvdp + 944) 0 — still queued for a free CPU box. zenav1-aom#14
closed with the KB-41 roots (the "720p band" was which renditions are screen-detected).

### SVT SCORE RUNS — GPU pass-kill artifacts requeued; CPU-score run launched on the tower (2026-08-30 08:5xZ)
- **GPU (i134, `zen-score-gpusf`):** the 113 `failed` rows (61 encoder_panic + 52 oom, all on
  sub-1 MP renditions) share ONE timestamp — 07:02:24Z on the first-pass worker `i134-med`, i.e.
  the 1,800 s pass-timeout kill's in-flight cells, not a tiny-frame defect (21 of the 34 renditions
  also have done cells). Requeued (`--classes oom,encoder_panic --before 07:03Z`, 113 pardon
  rows). The relaunched worker is healthy (GPU 56%, 3.8 GB VRAM, pass 1 under the 7,200 s
  budget — chunks flush at pass end).
- **CPU (`avifsvt-sf-cpu-20260830`, 11,608 score_file jobs: cvvdp + zensim-foldapp2/944):** no
  LAN CPU box is free (all three on the aom-rs wave), so it runs on the **tower** inside a capped
  container per the media-server rule — image `exec-zensim944hdr-47f5ab9c`, `--cpuset-cpus 0-23
  --cpu-shares 256 --memory 24g`, worker `Tower-cpusf`, container `zen-score-cpusf`,
  `ZEN_REQUIRE_SNAPSHOT=0` (fresh run). The LAN launcher cannot pass caps over its ssh line
  (only the listed `ZM_*` tokens cross), so this was a hand-mirrored `docker run` of the same
  env/entrypoint — extend `lan_score_launch.sh` to forward `ZM_CPUSET/ZM_CPU_SHARES/ZM_MEMORY`
  before the next tower launch.

### KB-41 ROOTS #18-#21 — THE cpu-4/5 SCREEN RESIDUALS ALL CLOSE; THE WAVE'S SCREEN CAP IS LIFTED (2026-08-30 ~13:2xZ, zenav1-aom, after `4e0229e1`)

The five cells left open after roots #14-#17 (s4: 1280x800 cq44 +10 B, 1920x1080
cq57 +10 B, 1920x1080 cq6 −33 B; s5: 1280x800 cq25 −5 B, 1920x1080 cq32 −123 B)
were four more C conventions, each found by the same instrument — first-syntax-diff
on the two decoded streams, then paired C/port probes around the first diverging
block (per-txb eobs on the OUTPUT run, `search_tx_type` quant/trellis state, the
intra edge availability + neighbour samples, the IntraBC per-direction RD, the
est-rd prune arrays):

- **#18** AB-partition split-ctx reuse (`is_split_ctx_is_ready`) needs a SPLIT child
  leaf with no palette and no CfL — the port reused any leaf.
- **#19** the SEARCH-side txfm-partition contexts are stamped on the OUTPUT run too
  (`tx_partition_count_update` → `update_txfm_count` → `txfm_partition_update`,
  partition_search.c:511-516; the dry run stamps via `tx_partition_set_contexts`).
  The port stamped only on dry runs, so after an SB's output run the search arrays
  held the value restored at SB start — the row-ABOVE SB's stamp — and the next SB
  row costed a tx split at ctx 18 where C had 19 (1080p cq57 s4, mi(32,90) under the
  8x8 IntraBC at mi(30,90), leaf TX_4X4 eobs [0,0,5,0] on both sides).
- **#20** the IntraBC candidate's predict-skip SSE is `pixel_diff_dist(x, 0, 0, 0,
  bsize, bsize, NULL)` — the VISIBLE block only. The port summed the whole block,
  so a frame-bottom 64x64 (56 visible rows) carried its 8 off-frame rows in the skip
  candidate (847872 vs C's 716800), lost to PAETH, and the next SB coded differently
  (same cell, mi(256,16), dv (−512,0)). Found via a DC-prediction mismatch (224 vs
  226) that turned out to be the LEFT neighbour's IntraBC+DC recon — a red herring
  the per-direction IBC-RD probe resolved.
- **#21** a pick-skip'd var-tx txb hands its siblings the SEARCHED entropy context
  (`no_split->txb_entropy_ctx = p->txb_entropy_ctx[block]`, tx_search.c:2447 —
  `pick_skip_txfm` zeroes only eob + tx type; the encode pass re-derives 0 via
  `is_blk_skip`). The port zeroed it, so the (1,1) child of an 8x8 IntraBC got
  txb_skip_ctx 3 where C had 5 once the (0,1) child searched eob 13 and was
  pick-skip'd (1080p cq6 s4, mi(20,154), dv (−368,−824)). This one closed BOTH the
  cq6 s4 and the cq32 s5 cells.

Census after the four: **s4 9/9, s5 3/3, the s6/s8 dir 24/24**, every gate suite
green, all temporary probes stripped before the commit (a `cargo fmt` that would
have reflowed 103 files was rolled back via `jj op restore`; the repo is not
rustfmt-clean and the commit stays semantic: 4 files, +46/−6).

**Cap lifted.** The s4 census — three 1920x1080 cells among nine — runs in 81.5 s
INCLUDING the oracle encode and both decodes, ~20 s per 1080p cell at cpu 4 (the
executor comment's ">40 min at cpu 4" predates roots #10/#16). The 2,976 remaining
`encoder_panic`-class cells of `avifaom-enc-20260830` (112 renditions above 0.30 MP,
all q, all three speeds, refused 11:52–12:11Z by the cap; 123,024 done) are real
encodes now: zenmetrics `ZEN_AOMRS_MAX_SCREEN_MP` default 0.30 → 16 MP (largest
rendition 3062x4096 = 12.5 MP, ~2-3 min at cpu 4), executor image rebuilt on the
roots-#18-#21 port, the cells re-queued, the three LAN boxes relaunched.

### SVT WAVE HARVESTED — 130,590 cells × (4 metrics + 944 features) (2026-08-30 13:1xZ)

Both svt score runs finished clean (`avifsvt-sf-gpu-20260830` 11,608/11,608 done, 0
failed; `avifsvt-sf-cpu-20260830` 11,608/11,608, 0 failed — the 113 GPU pass-kill rows
requeued at 08:5xZ all completed). Write-back (`writeback_scores.py avifsvt avif
<both runs>`, two-stage env: `ZEN_STORE=tower ZEN_JOBS_BUCKET=zentrain
ZEN_PAIRS_PARQUET=/mnt/v/output/avifsvt-2026-08-30/pairs_svt.parquet`, metrics
`butteraugli-gpu,cvvdp,ssim2-gpu,zensim-foldapp2`; 23,218 blobs; run-heavy peak RSS
6.85 GiB, 133 s) → `/mnt/v/output/avifsvt-2026-08-30/harvest-2026-08-30/`:

- `scores.parquet` — 130,590 rows × 10 cols: `butteraugli_max_gpu` (0–64.9, mean 4.84),
  `butteraugli_pnorm3_gpu` (0–21.0), `cvvdp_cpu_imazen_v0_1_0` (3.58–10, mean 9.55),
  `ssim2_gpu` (−141–99.99, mean 69.5; **921 nulls = 19 renditions at ≥ 12 MP
  (3000x4000 / 4000x3000 / 2945x4417 …) where ssim2-gpu errored** — the 969 "error
  rows skipped" — a CPU fast-ssim2 backfill for those 19 is registered, not run).
- `features_folded720append2.parquet` — 130,590 rows × 944 `feat_*` (the 944 regime,
  `miss_sha=0 miss_score=0`).
- `zensim_score` is **empty by construction** in both files: the spec declared the
  944 feature extractor (`zensim-foldapp2`) and no scalar zensim metric; the scalar comes
  from a bake over the features (the training views), not from this harvest.

Next for this lane: `avifgen_training_views.py`-style train_944 / eval8_944 views over
the harvest (origin even/odd rule — the corpus is train-side only), then the
`fused944native` comparison the carrier report registered.

### AOM-RS WAVE RELAUNCHED WITH THE CAP LIFTED (2026-08-30 13:20Z) — image `exec-zensim944hdr-e7a99c2d`; SVT TRAINING VIEWS EMITTED

- Port commit zenav1-aom `0c92ef1f` (roots #18-#21) is on origin/main; zenmetrics
  `e7a99c2d` (cap 0.30 → 16 MP) + `3004dd52` (fleet.env pin) on origin/master. The
  executor image was rebuilt on the sibling checkout (musl, `build_executor_image.sh`),
  the 2,976 cap-deferred `encoder_panic` cells pardoned (`requeue --classes
  encoder_panic`, pardon rows `pass-requeue-pardon-1788096000.parquet`, snapshot
  123,024 done + 2,976 newest-failed at the compact) and r3500 / r5900xt / r7900x
  relaunched at 13:20:22Z against the LAN store. Expected: ~50 box-hours (112
  renditions from 0.41 to 12.5 MP × 30 q × 3 speeds; ~20 s per 1080p cell at cpu 4).
- **svt training views** (`avifgen_training_views.py`, AC.R1 rule — the corpus is
  train-side only): `/mnt/v/output/avifsvt-2026-08-30/views-2026-08-30/train_944.parquet`
  **106,380 rows / 871 origins** (terminal digits 0/2/4/6) and `eval8_944.parquet`
  **24,210 rows / 209 origins** (digit 8; never trained on); ID columns asserted
  row-aligned between the score and feature sidecars (`features.parquet` is a symlink to
  the harvest's `features_folded720append2.parquet`). `view_counts.json` beside them.

### AOM-RS SCORE WAVES DECLARED ON THE 123,024 DONE CELLS (2026-08-30 13:24Z) — GPU on i134, CPU on the tower

Rather than idle the scorers until the 2,976 large-screen encodes finish, the pairs
bridge was cut on the run's DONE rows now (`zenfleet-ctl pairs --ledger
s3://zentrain/jobs/avifaom-enc-20260830/ledger --refs-prefix
s3://zentrain/refs/train-renditions-2026-06-14 --blobs-prefix
s3://zentrain/jobs/avifaom-enc-20260830/blobs` → `/mnt/v/output/avifaom-2026-08-30/
pairs_aom.parquet`, **123,024 DONE cells**, 2,976 non-done skipped; one in-flight ledger
chunk was unreadable and skipped — a live worker's partial file) and two score-file runs
declared with `--full-uri --cell-codec zenavif` (the svt shape; without `--full-uri` the
declare wants `dist_member` columns the bridge does not carry): `avifaom-sf-gpu-20260830`
(ssim2-gpu + butteraugli-gpu, **10,950 jobs**) and `avifaom-sf-cpu-20260830` (cvvdp +
zensim-foldapp2 944 features, **10,950 jobs**).

- **GPU:** the drained svt worker's restart loop on i134 (`zen-score-gpusf`, idle since
  07:04Z) was removed and the cuda13 sequencer launched (`lan_gpu_sequence.sh i134 gpu`,
  image `exec-gpu-cuda13-6d4f9963`, `ZM_VRAM_CAP=8 GiB`); claiming at 13:24:09Z, GPU 15%.
- **CPU:** the tower's drained `zen-score-cpusf` removed and relaunched capped
  (`ZEN_CPUSET=0-23 ZEN_CPU_SHARES=256 ZEN_MEMORY=24g`, verified on the container:
  cpuset 0-23 / shares 256 / 24 GiB — the launcher forwards the caps now) on image
  `exec-zensim944hdr-e7a99c2d`. First start died on `ZEN_REQUIRE_SNAPSHOT=1` with no
  snapshot for a fresh run; `zenfleet-ctl compact --run avifaom-sf-cpu-20260830 --upload`
  (0 rows) satisfied it and the worker claimed at 13:24:48Z.
- The 2,976 cells still encoding get a gap-fill declare (`pairs` again → a second pair of
  runs) at the encode drain; `writeback_scores.py` merges runs.

### THE CARRIERS' NATIVE SLOTS ARE LIVE IN THE STREAMING PASS — AND MEASURED NOT FREE (2026-08-30 ~14:0xZ, zensim)

The registered product step from the carrier report ("un-zero the native slots
under a regime flag; zenbench paired A/B emit-vs-zero") is implemented:
`V2NewFeatureToggles::v1_pools: V1PoolsMode { Off | Carriers | Full }`. The fold
hook (`fold_v1_basic_bands`) already replayed v1's 32-row band tiling bit-for-bit
for the basic block; it now also replays v1's extended strip section per band
(the fused kernel's `store_mu` means, the ref-side activity map over the same
band buffer, the V-blurred sigma planes, the fused masked+IW SSIM / edge / MSE
kernels at `k = k_iw = 4`), so:

- **Parity: BIT-IDENTICAL to v1's frozen 372 extraction** for all of `f156..372`
  at every exact-width fixture (96x64, 64x300, 208x144) — gate
  `folded720_v1_pools_match_v1_path`, which also asserts the toggle changes nothing
  outside the block and that `Off` still zeroes it. Padded-width class (127, 200)
  diverges the way the basic block does (v1's pad wart), max rel 0.17 / 0.82.
- **`Carriers`** = exactly the ten `fused944native` slots (f178/190/196/226 art_l8,
  f231/237/243 masked_art_4th, f303/321/333 iw_art_4th) live, everything else 0 —
  the regime those tables were built in; **`Full`** = all 216. The result carries
  the mode (`ZensimV2Result::v1_pools()`), extractor modes `foldapp2carriers` /
  `foldapp2pools`. Same 944 width either way — a distinct regime by the purity
  rule, never column-mixed with zeroed-block rows.
- **Cost, zenbench paired (`benches/fold_pools_bench.rs`, serial; the box was
  loaded — 4 clean rounds per arm, 240 noisy):**

  | pair | zeroed | carriers (10) | full (216) |
  |---|---|---|---|
  | 576² | 15.4 ±0.3 ms | 18.6 ±0.7 (+18.1–24.8%) | 19.9 ±0.3 (+27.9–32.0%) |
  | 1152² | 59.0 ±0.3 ms | 72.2 ±0.2 (+21.5–23.4%) | 77.0 ±0.5 (+29.0–31.6%) |

  So the carrier report's item 2 ("inside the streaming pass the carriers cost
  accumulators only … expected noise-level") is FALSIFIED as stated: the peaks are
  free (the kernel returns them), but the two art-L4 carriers need the ref-side
  activity map and a second fused edge pass at scales 0-1, and scale 0 is the whole
  image. +0.52 ms was the buffered-harness figure and is superseded by this table.
  First lever taken (same session): v1's activity is `|src − H_blur(src)|` and the fold
  already holds `H_blur(src)` as the shared `mu1_h` plane, so the re-blur became one
  abs-diff pass (bit-exactness kept — the two H kernels agree):

  | pair | zeroed | carriers (10) | full (216) |
  |---|---|---|---|
  | 576² | 15.0 ±0.3 ms | 17.2 ±0.1 (+11.9–16.5%) | 18.3 ±0.0 (+17.7–26.2%) |
  | 1152² | 59.2 ±1.1 ms | 67.8 ±0.3 (+14.5–15.3%) | 72.9 ±1.3 (+21.3–25.7%) |

  (v2's own activity plane is `|src − mu1(2D)|` blurred — a different quantity — so it
  cannot be reused.) Registered next lever (not claimed): the weighted art-L4 sums inside
  the fused V-blur kernel (no `store_mu`, no second edge pass) — that is where the
  "accumulators only" structure actually lives.

### PASS-TIMEOUT INCIDENT: every big-cell pass died at the 1800 s default — all five workers relaunched with a 4 h pass budget (2026-08-30 13:5xZ); svt ssim2 gap backfilled on CPU

- **What happened:** the three aom encode boxes' pass 1 (13:20Z) and the tower's score
  pass 1 all ended `rc=124` at +30 min — the worker's per-pass `timeout` (default
  `ZEN_PASS_TIMEOUT=1800`). A pass claims a CHUNK of cells; the cap-lifted chunks hold
  ≥ 12 MP screen renditions at cpu 4 (~2-3 min each), and a 12 MP score_file job covers 12
  variants, so no pass could finish. The worker's timeout handler releases the chunk claim
  ("spot preemption — released chunk claim … for fast requeue"), so nothing was lost — the
  cells simply re-entered the gap every 30 min and the boxes made no progress (drain watcher:
  `idle_all=0`, three `rc=124` lines). i134's sequencer showed the same shape
  (`consec_fails=1`).
- **Fix:** `lan_score_launch.sh` already forwards `ZEN_PASS_TIMEOUT`; the GPU sequencer did
  not — zenmetrics `0cc07589` adds `ZM_PASS_TIMEOUT` → `-e ZEN_PASS_TIMEOUT` to
  `_lan_gpu_seq_driver.sh` / `lan_gpu_sequence.sh`. All five relaunched with
  `ZEN_PASS_TIMEOUT=14400` (verified on the containers: r7900x, tower, i134): encode boxes
  13:57Z, tower cpusf (caps intact: cpuset 0-23 / shares 256 / 24 GiB), i134 sequencer
  (VRAM cap 8 GiB). Stops were `docker stop -t 90` (SIGTERM) before `rm`, so the in-flight
  chunk claims were released, not orphaned — `reassert` at the drain covers any that were
  not.
- **svt harvest ssim2 gap CLOSED on CPU:** the 921 cells (19 renditions ≥ 12 MP) whose
  ssim2-gpu was null were scored locally with `zenmetrics batch --metric ssim2` (fast-ssim2,
  run-heavy 1026 s, peak RSS 1.96 GiB) → keyed sidecar
  `harvest-2026-08-30/ssim2_cpu_backfill.parquet` (921 rows × ID + `ssim2_cpu`, sha
  `31dcb3df…`, manifest beside it). Kept as its OWN column: CPU-vs-GPU ssim2 equivalence on
  the overlap is not measured here, so a consumer coalesces only after that check.

### THE CARRIERS REGIME IS A FLEET METRIC NOW — `zensim-foldapp2carriers`; svt carriers feature run declared (2026-08-30 14:2xZ)

- zenmetrics `c0cdf095`: `ZensimFeatureRegime::Folded720Append2Carriers` (regime tag
  `folded720append2carriers`, 944 wide) = the streaming 944 extraction with
  `V1PoolsMode::Carriers` — the `fused944native` regime produced by the fleet directly
  instead of by table surgery. SDR-only (the HDR arm emits an explicit error row); gate
  `folded944carriers_matches_driver_args` (bit-identical to the direct zensim call; exactly
  the ten carrier slots differ from the plain 944 regime). Executor image rebuilt on it
  (`exec-zensim944hdr-c0cdf095`, musl, `strings` check on the metric name).
- Declared `avifsvt-sf-carriers-20260830` over the svt wave's 130,590 DONE cells (11,608
  score_file jobs, `--full-uri --cell-codec zenavif`, empty snapshot uploaded). Launch:
  on the tower as a second capped container (cpuset 0-23 / shares 256 / 24 GiB — inside
  the media-server cap, sharing it with the aom CPU scorer) once the image is up; the
  write-back then yields `features_folded720append2carriers.parquet` beside the svt harvest —
  the training view for a linear/BVLS carrier head on fresh svt-rs encodes, the
  registered "fused944native comparison" on data the trainer has never seen.

### AOM WAVE DRAINED (2026-08-30 15:32Z): 125,688 done; the 312 poisoned cells are TWO port classes, both reproduced locally; svt carriers features landed; a write-back join defect found and fixed

- **Drain:** `avifaom-enc-20260830` idle on all three boxes at 15:32Z — **125,688 done + 312
  poison** (attempts 2, `encoder_panic`). Gap-fill pairs cut (`pairs_aom_full.parquet`,
  125,688 DONE cells; 2,664 new since the 13:24Z bridge) and two gap-fill score runs
  declared (`avifaom-sf-{gpu,cpu}-gap-20260830`, 278 jobs each); the CPU gap-fill runs on
  r7900x (`zen-score-cpugap`, 15:42Z); the GPU gap-fill waits for i134's main bucket. The
  three encode workers were retired.
- **The 312 poison cells, reproduced through the executor's own arm** (`zenmetrics sweep`
  on the dev box, planes dumped): they are NOT timeouts.
  - **277 cells on 56 renditions > 1 MP (screen-detected), q ≤ 35 (cq ≥ ~40), cpu 4/8**: a
    genuine port divergence — e.g. `6012.scale2091x3072` q1 (cq62): **port 162,676 B vs
    oracle 162,231 B** at s4, 157,273 vs 157,073 at s8. First syntax diff (localizer,
    `~/tmp/aom_poison_repro/planes_6012.scale2091x3072.png`): s4 mi(124,328) — the port
    codes a 16x8 PALETTE block (4 colours) where the oracle splits to 8x4 V_PRED;
    s8 mi(80,280) — port 16x16 with a 2-colour palette + TX_8X8, oracle no palette + TX_16X16.
    So at qindex 249 the port's palette search accepts palettes the oracle rejects — the
    next KB-41 class ("palette at extreme cq on large screen content"); localization in
    progress (paired C/port palette probes).
  - **35 cells on 21 tiny renditions (59x128 … 99x128)**: the port's screen-content decision
    disagrees with the oracle header (`palette=0 intrabc=0 photo=6 fast=true`; detector 0,
    header 1) — the still-unported `av1_determine_sc_tools_with_encoding` trial-encode arm
    (two q ≥ 244 fixed-partition encodes), which the bench asserts by name. A port of that arm
    is the fix; registered, not started.
- **svt carriers features:** `avifsvt-sf-carriers-20260830` drained 11,608/11,608 (0 failed)
  in 91 min on the tower under the media cap; write-back →
  `harvest-2026-08-30-carriers/features_folded720append2carriers.parquet` (130,590 × 944):
  exactly the ten carrier slots live (f178/190/196/226/231/237/243/303/321/333, 91-99.7%
  non-zero), every other pool slot 0, the plain harvest's block all 0 — the regime is what it
  claims.
- **Write-back join defect (zenmetrics `writeback_scores.py`), found by that check:** 18
  rows disagreed between the two svt harvests on NON-carrier slots. Cause: different source
  images can encode to byte-identical bytes (15 shas shared across 30 svt cells, all q=1
  tiny renditions; 7 shas / 14 cells on the aom wave), and the write-back keyed scores +
  features by `encode_sha` ALONE — last cell wins, so those rows carried the OTHER cell's
  (ref, dist) values. Verified: refs 7064/7020.scale128x128 differ in 99.9% of pixels yet
  share blob `4b1e87e5…`; the current zensim reproduces the plain harvest's f0 (0.0618) for
  (7064, blob) and the carriers harvest's f0 (0.0478) for (7020, blob). Fixed: keys are
  `(ref basename, sha[, metric])`; both svt harvests and the views re-written from the cached
  blobs (old files kept as `shakey-bak/`). The earlier per-codec harvests (hdrgrid, avifgen)
  carry the same latent defect wherever a sha is shared across refs — a one-line count per
  pairs table tells whether any row is affected.

## 2026-08-30 16:05Z — orchestration hand-off (user directive) + aom score waves drained

**User directive (2026-08-30):** execution is delegated to Opus subagents; the Fable session
orchestrates only (watchers, launches, ledgers). First delegation: the KB-41 large-screen
partition divergence (root #22 candidate — the port evaluates HORZ_B at the 16x16 mi(124,328)
of `2091x3072_cq62_s4` and skips HORZ_4, where libaom evaluates HORZ_4 and never runs the AB
arm; best_rdc 852,710,591 vs 880,017,878) — the Opus agent owns the fix, regate, probe strip,
commit, PARITY/CLAUDE rows, and the 35-tiny-cell trial-encode arm if cheap.

**aom score waves — status at hand-off (all MEASURED from worker logs):**
- i134 GPU main (`avifaom-sf-gpu-20260830`, ssim2-gpu + butteraugli-gpu): sequencer dropped
  `lan_gpu_seq.COMPLETE` at 15:55:02Z ("ALL BUCKETS DRAINED").
- tower CPU main (`avifaom-sf-cpu-20260830`, cvvdp + zensim-foldapp2): idle passes
  (`done=0 … rows=0`) from 15:40Z on — drained.
- r7900x CPU gap-fill (`avifaom-sf-cpu-gap-20260830`, 2,664 cells): idle passes 7–8 at
  16:03Z — drained.
- **i134 GPU gap-fill (`avifaom-sf-gpu-gap-20260830`) LAUNCHED 16:04Z** (stale COMPLETE marker
  cleared first; `ZEN_PASS_TIMEOUT=14400`, snap=1376 rows seen on pass 1).
- Watcher `~/tmp/aom_gap_watch.sh` (log `aom_gap_watch.log`): on DRAINED it runs
  `aom_harvest.sh writeback` (4 runs, `(ref basename, sha, metric)` joins) then `views`, stops the
  three idle CPU score containers, and writes `aom_gap_watch.done` on every exit path (8 h cap).
  Pending after that: CPU backfill for the GPU-refused tiny/odd-dim cells (27 seen on the main
  run) and the requeue of the 277 refused large-screen cells once root #22 lands + the executor
  image is rebuilt.

## 2026-08-30 ~17:1xZ — OPUS LANES, ROUND 1 RESULTS (orchestrated; execution delegated per user directive)

**User directives this round:** "bake zenpicker, do R1b, explain why 720 instead of an all-feature
944 extraction — optimize a pass that extracts all 944 first, use Opus for all."

**Why the carriers came from a 720-width table (answered):** the 944 regime (`folded720append2`)
writes f156–371 as structural zeros; the only table with those slots live was the `ext720` root
(unfolded v1 block, row-identical to the ext944 legs); and v1 pool features diverge across
extraction widths (padded-width effect: f237 rel 7.6e-2, f333 2.05e-1), so all carriers had to
come from ONE width. The user's amendment replaces that fusion with a one-pass, one-width
all-live extraction (`V1PoolsMode::Full` → regime `folded720append2pools`, 944 live slots).

### KB-41 lane (zenav1-aom) — DONE: roots #22-#23, both >1 MP screen cells byte-exact
- **#22** `is_rect_ctx_is_ready` needs a palette-free (luma AND chroma) rect winner, not only
  non-CfL (partition_search.c:3613-3619) — the RECT twin of #18's SPLIT gate. The hand-off premise
  "C never evaluates HORZ_B" was a stale-binary dump artifact: C evaluates it and rejects it on
  sub-block 2 (494,626,860 cumulative vs the port's palette-leaf reuse 427,208,793).
- **#23** `mbmode_cost` is NOT dead on a KEY frame: the speed≥8 nonrd palette shell charges
  DC_PRED through `mbmode_cost[size_group][DC_PRED]` (intra_mode_search.c:1139-1152); the port had
  an all-zero placeholder (C 375 vs port 3) → every nonrd palette header 372 units too cheap.
  Closes `PALETTE_MANY_COLORS_OPEN` (both kb37 pins → zero).
- Cells: `2091x3072_cq62_s4` 162,231 B, `_s8` 157,073 B (+0). Census **104/104**. Commits
  `fb745179` + `d950eec1` (verified on origin). 35 tiny cells = unported
  `av1_determine_sc_tools_with_encoding` — registered NOT-cheap (6-item checklist, PARITY C3).
- **KB-42 OPENED (found while gating):** zenav1-aom `main` CI RED since `735a0a6d` (roots #3-#6);
  300/16 encode integration tests locally + aarch64 aom-bench gates; the landings' gate lists
  ran `--lib` only. Not caused by #22-#23 (revert-measured). Carrier narrowed to var_tx /
  partition_pick tx-policy plumbing → Opus KB-42 lane launched (fix + all CI legs green + gate
  list closed). The 277 refused wave cells requeue after the next executor image (zenmetrics-cli
  takes the port as a PATH dep, so any image build carries #22-#23).

### R1b lane (zensim) — DONE: `benchmarks/r1b_keyed_rebuild_2026-08-30.md` (commits `c0d5bacf`…`68f02f74`)
- Keyed: the 11 canonical local legs (149,195 rows; `(ref_path, dist_path)` IS the key — pairs
  TSVs on disk, row counts equal) + the 3 D1 validate slices (20,812 rows; keys recovered one
  level up in the bigcodec views, G-KEY 3/3 row-for-row). Keyable-not-built (fleet scale):
  tbig_200k / tbig_hf / teacher legs / kadis. **NOT KEYABLE at this regime: hdrmix** (no SDR-route
  extraction of hdr_v3mix exists).
- Extracted at pools-944: **170,007 rows** + a same-binary zero-block control (18,521).
  **G-P1 5/5: 728/728 non-pool columns BIT-IDENTICAL to the control** (the regime flag changes the
  pool block and nothing else); G-P2 216/216 live. G-E 11/11. Drift vs the 2026-08-01 root:
  22/728 append cols at ~1e-8 (extractor version; reported, not gated).
- **REPRO HAZARD caught (G-B):** re-extracting KADID from its pairs TSV reproduces the
  pre-2026-08-05 INVERTED target (−0.582360) — the 08-05 correction was applied to the parquets,
  not the TSV. Repaired via the owner (+0.582360); anyone re-extracting any ext leg inherits this.
- **First same-pair bars (B on native 372, arms on native 944):**

  | model | cid22 | \|kon504\| | nonphoto | imazen26 | hfnl |
  |---|---|---|---|---|---|
  | **B (shipped)** | **0.8763** | **0.5183** | **0.9093** | **0.9142** | **0.3553** |
  | A0 zero-block | 0.8311 | 0.2062 | 0.8773 | 0.8806 | 0.2398 |
  | A2 pools-live | 0.8332 | 0.1911 | 0.8784 | 0.8815 | 0.2474 |
  | R-1 A0 / A2 | 0.8646 / 0.8652 | 0.3994 / 0.4105 | 0.8709 / 0.8779 | 0.8834 / 0.8891 | 0.2342 / 0.2521 |

  B leads every axis; all arms FAIL the round-6 bars. Whole-block-live moves the head ~0.002
  cid22 / 0.01–0.02 elsewhere. **A3 (discriminating):** the same driver on the ledger's OWN
  fused944native tables gives the same near-null (kon −0.1914 vs A2 −0.1911) ⇒ the 720-width
  fusion is NOT what separates this read from the ledger's +0.3243 carrier head — the RECIPE is:
  those heads were fit ad hoc with no committed driver/argv. **R1b does not falsify the carrier
  finding; it makes it UNREPRODUCED until the recipe is recorded.** One-width all-live extraction
  and the fused tables are behaviourally equivalent for this head (0.0003 kon / 0.0009 cid22).
- Open: full-mix cid head + `wlin954b` blend (needs bigcodec legs at this regime — fleet job);
  hdrmix; B's ledger kon 0.5935 vs R1b 0.5183 (instrument difference, unadjudicated). Corpus
  defects found: `tid2013/reference_images_png/i25.png` lowercase vs TSV `I25.png` (silently loses
  120 rows); v1's feature-vector length is size-dependent in both extractors (~6.5% of slice rows
  emit 279 not 372) — the 944 fold is fixed-width. Follow-ups handed back to the lane.

### zenpicker lane (zenanalyze) — IN PROGRESS: 10 commits (`782ee43`→`d7eb26b`)
- Inert `zenpicker::cell` wiring, touch-once contract test, 13 CI-runnable refusal tests,
  `zenpicker-train --baselines` (baseline gate at the owner) + k-seed driver, slot→feature identity
  table, e2e demo. v1 panel reproduced 0.7500 / 4.47% / p50 0 / p90 14.53% / 0.9869; `--hidden
  128,128 --seed 0` reproduces `metapicker_v1.bin` BYTE-FOR-BYTE.
- Baseline gate final: picker 4.47% vs always-avif **19.75%** (avif covers only 94.44% of cells —
  the ledger's "20.4%" corrected). Band so far (2/5 seeds): s0 0.7500/4.47%/14.53%,
  s4 0.7551/4.27%/13.70% — s4 is WORSE on the trainer's internal held-out split but BETTER on the
  honest panel: the grid search's selection surface inverts against the honest one at this margin
  (registered). Remaining seeds finalize ~17:41Z.

### aom score harvest — on disk
i134 GPU gap-fill drained 16:37Z (33 min); write-back: 125,688 cells × 4 metrics
(`scores.parquet`), 125,687 × 944 (`features_folded720append2.parquet`), 1,540 error rows
skipped; views blocked on `avifgen_training_views.py`'s positional ID assert (one feature-less
cell) → ID-column join fix + rerun handed to the all-944 lane. Idle score containers removed.

## 2026-08-30 ~18:0xZ — OPUS LANES, ROUND 2 RESULTS

### All-944-live pass (lane 2) — DONE: zenmetrics `905ae73d`, zensim `d3092e98`, image `exec-zensim944hdr-03bdf64b`
- `zensim-foldapp2pools` metric = regime `folded720append2pools` (every one of the 944 slots live),
  `V1PoolsMode::Full`; **9/9 cells all 944 slots bit-identical to `v2_ab_extract --mode
  foldapp2pools`; f156–371 bit-identical to the v1 372 regime (0 of 1,944 slots differ)**; non-pool
  slots bit-identical to plain 944. NOTE (brief correction): `zenmetrics batch --metric` cannot reach
  ANY zensim feature metric — `jobexec` and the regime enum are the surfaces.
- Perf lever shipped: reuse the fused V-blur kernel's `sigma²/sigma12` register values (side
  outputs, activity-first ordering, inner-rows-only stores) instead of two `box_blur_v_from_copy`
  sweeps. Paired zenbench (20 rounds/arm): full216 **+18–25 % → +16–19 % @576², +22–29 % →
  +18–20 % @1152²** over zeroed; full-vs-carriers **+11.0 % → +4.5 % @1152²** (+5.4 → +5.0 @576²).
  The registered "art-L4 sums inside the fused kernel" half is **unshippable bit-exact**
  (column-group-major vs row-major f64 sums drift; gate fails) — remaining bit-exact levers ranked
  in `benchmarks/pools_full_extraction_2026-08-30.md`. Bench budget raised (`min_rounds(25)`,
  600 s): the stock 120 s ceiling gave 4 rounds and ±10-pt CIs; earlier ledger figures are on the
  4-round budget and NOT comparable. **GATE HAZARD:** `folded720_v1_pools_match_v1_path` is
  `#[cfg(feature = "training")]` — the feature list `custom-profiles,feature-regime-v2` silently
  compiles it out (206 lib tests pass, gate never runs); with `training`: 322/0.
- Image built + pushed locally, verified from INSIDE the container (1-cell jobexec →
  `regime=folded720append2pools`, 944 feats, bit-identical to local); pinned `ZEN_FLEET_IMAGE_CPU`
  (`affcee66`); predates `d3092e98` (output-identical; speed-only rebuild owed).
  `ZEN_WRITEBACK_METRICS` must LIST `zensim-foldapp2pools` (defaults carry neither live-pool metric).
- `avifgen_training_views.py`: positional assert → 4-column ID key join with a uniqueness assert.
  `(image_path, encode_sha)` alone has **4,242 duplicate rows per side** (tiny renditions saturate
  → byte-identical encodes across q). aom views: **train_944 102,180 rows / 871 origins; eval8_944
  23,507 / 209**; 1 scores-only drop (`5006.scale2896x4096` q94 s6, feature blob never landed).

### aom requeue round (fleet) — RUNNING (`~/tmp/aom_requeue6.sh`, launched 17:32Z)
312 `encoder_panic` cells pardoned (277 large-screen + 35 tiny trial-encode, which re-poison by
design) → `aomenc` workers r3500/r5900xt/r7900x on `03bdf64b` (roots #22-#23 in; KB-42 fixes not
yet — the executor byte-verifies every cell) → drain → gap-2 score declares (i134 GPU, tower CPU)
→ write-back over all six runs → views (old harvest/views kept as `*-pre-requeue.bak`).

### R1b follow-ups — DONE (`37961318`, `657100db`, `63aef3bc`, `251febdd`)
- **TID `i25.png` casing** (`build_fr_corpus_pairs.py::build_tid()` upper-cased the reference name
  while resolving the distorted side case-insensitively; the 120 rows of the one lowercase ref were
  dropped with a printed count and exit 0): both sides now case-insensitive; an unresolved label row
  is FATAL across the builders (`FRPAIRS_ALLOW_MISSING=1` opt-out); `v2_ab_extract` refuses to
  write a partial table when rows != pairs (exit 3; `ZENSIM_AB_ALLOW_MISSING=1` opt-out). Canonical
  TSV regenerated 3,000/3,000 (only the 120 paths changed); TID re-extracted at pools-944 + control:
  G-P1 728/728, G-E equal, orientation +1.000000; root manifest back to 14 corpora.
- Stale regroup trace logs under `s3://zentrain/canonical/2026-06-27/_regroup/` removed from R2
  (5 objects, 811 MiB; no copies on tower MinIO / local mirror / Tower NFS; the `.done` record kept).
- **v1-372 WIDTH DEFECT — "size-dependent" RETRACTED, it is BATCH-DEPENDENT and deterministic**
  (`docs/DATASET_HISTORY.md` §3.26): short (279-wide) rows occur at 168 distinct sizes (min side
  36…1024, same sizes in both sets), 259/957 refs carry both widths; threads 1/2/8 → 33/33/33 (not
  a race); full batch twice → identical 453-row set; **the same 453 pairs re-run alone → 33 short;
  5 alone → 0**; identical set from both v1 extractors, grouped and per-pair. A v1-372 vector is
  not a pure function of its pair. Canonical 372 parquets are unaffected in practice (full width,
  0 NaN, 0 zero-padded; fixed-size corpora complete: cid22 4,292 / kadid 10,125 / konjnd 1,008; a
  ragged CSV cannot become a parquet silently — the builder raises). The 944 fold is fixed-width by
  construction. Root cause NOT diagnosed → diagnosis lane launched (see round 3).
- Cross-reference: the carrier-recipe lane (`fdd13b0f`) recovered the missing kon-head argv — the
  shaping screen is the TRAINED-BAKE screen (R1b §2b used `screen944_monotone.tsv`) with no min-max
  framing — which explains R1b's kon-degenerate head; R1b's arm numbers stand as measured but are
  NOT evidence about carriers. The A2-vs-A3 root equivalence (0.0003 kon) is unaffected.

## 2026-08-30 ~19:2xZ — OPUS LANES, ROUND 3 RESULTS + CORRECTIONS

### CORRECTIONS to this ledger (from `benchmarks/carrier_head_recipe_2026-08-30.md`, verdict RECIPE-DEPENDENT)
- **WITHDRAWN:** "shaping-on-944-alone falsified (kon 0.16)" — that 0.16 was the per-corpus
  MIN-MAX-framed head; the raw-framed 944 head with NO carriers reads **kon 0.4403 / cid22 0.8726**.
- **RE-PRICED:** "the 10 carriers take kon 0.164 → 0.489 (+0.324)" — numbers exact, cause wrong:
  the matched pair was matched on features, NOT on the target frame (no-carrier arm mm01, carrier
  arms raw). 2×2 at matched frame: **+0.2759 target frame (85 %) + 0.0317 954-append-vs-944-native
  layout (10 %) + 0.0167 carriers (5 %)**; carrier term on cid22 = **0.0000**. "Carriers ARE the
  linear kon backbone" — NOT supported (+0.015…+0.046). "A good linear does not require the 372
  front" — STANDS, on firmer ground. Recipe recovered bit-exactly (legs safesyn 1.0 / cid22t 1.5 /
  kadid 0.5 / tid 0.5; trained-bake screen 914 id / 30 winsor_p99 / 10 signed_cbrt inside
  f0..f155; `--tau 0.005`; raw `human_score`); driver `scripts/carrier_head_fit.sh` (+`CHF_MM01`),
  `scripts/extract_bake_transform_screen.py`; reproduction residual 0.0000; commits `fdd13b0f`,
  `6d0a393a`. **Positive:** dropping the min-max frame = +0.276 kon / +0.048 cid22 / +0.074 hfnl /
  +0.014 imazen26 at zero cost (−0.015 kadid, −0.009 tid train-side). Keyed pools-944 raw-frame
  arms: K0zero 0.8726/0.4403/0.8296/0.8470/0.2195, K1carr 0.8769/0.4553/0.8187/0.8374/0.1972,
  K2pools 0.8440/0.4866/0.8127/0.8386/0.1957 (cid22/|kon|/nonphoto/imazen26/hfnl) — round-6 bars
  FAIL on all arms; B leads every family axis. **W-LIN round 7** (raw frame, full mix incl. the
  bigcodec/hf/teacher legs extracted at pools-944 via the fleet) launched as an Opus lane.
- **RETRACTED (round-2 entry above):** "v1-372 width is BATCH-dependent" — FALSIFIED by
  `f9fac41e`'s lane: the width is a pure function of (W, H). Mechanism: the pyramid walk starts at
  `simd_padded_width(W)` but plain `H` and stops at `< 8`, so 4 scales need padded-W ≥ 64 AND
  H ≥ 64; `combine_scores` sizes the output from the surviving scales (3 → 279, 2 → 186, 1 → 93).
  `compute_with_config_inner` reflect-pads; THREE entries did not: `compute_zensim_with_config`
  (silent short vector — both v1-372 extractors call it), `compute_zensim_with_ref_and_config`
  (panic) and the PRODUCT `Zensim::compute_with_ref_into` (panic; no caller found). The W-vs-H
  asymmetry (54x96 full, 96x54 short) is what made "too small" look falsified. Fix: one owner
  (`needs_pyramid_pad` / `reflect_pad_for_scales`, `num_scales`-aware) at all seven entries; gate
  `tests/v1_feature_width_pure_function.rs` (8 tests, 5 fail pre-fix); 19,444/19,444 previously-372
  R1b rows BYTE-IDENTICAL post-fix, 1,368 short rows now 372 with f0..f155 bit-identical to the
  944 fold. **Blast radius: no shipped table exposed** (25 canonical 372 parquets exact width; 0 of
  149,195 canonical pairs could truncate; 944 tables byte-identical pre/post). Commits `f9fac41e`,
  `f3097091`, `ca7e65cf`; `benchmarks/v1_width_defect_2026-08-30.md`; DATASET_HISTORY §3.26 RESOLVED.
  Open: `r1b-samepair372-2026-08-30` dropped the 1,368 short rows (size-correlated 6.5 % restriction)
  and `r1b-372root` has three dangling symlinks — full-width replacements at
  `/mnt/v/output/zensim/v1width-fix-recheck-2026-08-30/` (handed to round 7).
- **NEW, PRE-EXISTING (found by the width lane): v1-372 EXTRACTOR DRIFT vs the stored canonical
  tables** — a fresh v1-372 extraction differs from the STORED cid22val/kon504 tables on essentially
  every masked/IW slot (100 % of rows, max_rel 0.34 / 1.0); two binaries (pre/post width fix)
  agree byte-for-byte, so it is drift between the extractor era and the stored tables, not the
  fix. Shipped B consumes f156–371 at runtime from the CURRENT extractor while its verdicts read the
  STORED tables → B's runtime behaviour vs its evaluated behaviour is UNVERIFIED. Diagnosis lane
  launched (round 4).

### zenpicker (criterion 8) — DONE: 11 commits `782ee43`→`18971ef` on zenanalyze
# Campaign-ledger rows for `zensim/benchmarks/balance_campaign_2026-08-28.md`
# (criterion-8 section — I do not hold the zensim repo; paste these there.)

### ZENPICKER v1 — WIRING LANDED (INERT) + REPRODUCTION GATES (2026-08-30)

**Both v1 numbers reproduce, one of them BYTE-IDENTICALLY.** (a) The registered
honest panel re-runs to **argmin 0.7500 · overhead mean 4.47% / p50 0 /
p90 14.53% · bytes-SROCC 0.9869**, matching the ledger's 0.7499/4.47/0/14.52/
0.9869 on every reported statistic. The ledger's 38,668-row count is
`--val-frac 0.999` (MEASURED: 0.999 → 38,668 rows, 1.0 → 38,696;
`grouped_split_picker` rounds `n_images·val_frac`, so 0.999 withholds exactly
one image); no statistic moves between them, and the k-seed wave uses 1.0 = the
ENTIRE view. (b) `zenpicker-train --hidden 128,128 --seed 0` reproduces
`metapicker_v1.bin` **byte-for-byte** (sha256 `4479ef9c874ebf1c…`) with an
identical `[heldout]` block to every digit — so `--hidden` really is grid
candidate #2, training is bit-deterministic, and the k-seed wave varies ONLY
the seed.

**Wiring: landed, INERT, additive** (zenanalyze `782ee433` … `d7eb26b1`, 10
commits). The registered mis-map was real — v1's 7 cells are family×mode while
`CodecFamily` is a 6-enum, so `MetaPicker::pick` would read
`CodecFamily::ALL[cell_index]`. New `zenpicker::cell`: `CellContract`
(+`from_model` validation, `build_input`), `CellPicker`
(+`from_znpr_bytes[_with_schema]`, `predict_cells`, `meta_picker`),
`FamilyModeCell`, `CellMode`, `CellPrediction`, 4 consts,
`MetaPickerError::CellContract`. `default_route` / `route` / `pick` /
`default_routers` and the three shipped routers are untouched — a test asserts
each shipped router is REFUSED as a cell bake. `no_std+alloc` clean
(`aarch64-unknown-none`). Flip stays user-gated.

**Touch-once test**: the contract mapping is asserted a BIJECTION — each of the
61 declared source features requested exactly once, `zq_norm` placed exactly
once and never requested, nothing outside the contract read, every slot
carrying the value of the name declared there (probe source injective by
construction). Bake located via `ZENPICKER_METAPICKER_V1_BAKE`; unset ⇒ FAIL
LOUD; the skip decision lives in the caller (`just metapicker-v1-test`; CI
`-- --skip metapicker_v1_`). A second file (`cell_contract.rs`, 13 tests on
synthetic in-process bakes) covers every refusal path on every CI run.

**The identity gap (open, upstream, machine-checked)**: v1 declares its
features POSITIONALLY (`feat_0..feat_60`) and carries no
`zentrain.feature_columns`, so `Model::feature_columns()` is empty and
`MetaPicker::feature_request()` is `None` — **v1 cannot consume a shared
zenanalyze-api `Offer`**. Cause located: zensim
`scripts/canonical_corpus/build_metapicker_input_2026-08-30.py:59` renames the
qualified source columns to `feat_<j>` and records the originals nowhere.
Recovered (all 61 qualified `name@hex8`) into
zenanalyze `benchmarks/metapicker_v1_feature_slots_2026-08-30.tsv` by
re-running the builder's own rule against the sha-pinned source TSV; a test
keeps it in lockstep. **The fix belongs upstream** — not applied here because
changing the bake metadata would void the byte-identity gate.

**Baseline gate, re-measured on the picker's OWN dense grid** (new
`zenpicker-train --baselines`; the v1 20.4%/55% always-avif figure came off a
coarse 5-target side grid, as the ledger says). The fixed-policy table is bake-INDEPENDENT (it is a property of the dataset +
oracle) and came out byte-identical on every seed, so it is the gate at EVERY
seed including the worst:

| fixed policy (its own best reachable cell) | mean | p50 | p90 | argmin | coverage |
|---|---|---|---|---|---|
| always-avif  | **0.1975** | 0.0000 | 0.5213 | 0.5156 | **0.9444** |
| always-jxl   | 0.7795 | 0.3535 | 2.3123 | 0.2180 | 1.0000 |
| always-webp  | 0.8052 | 0.3795 | 2.1829 | 0.1705 | 1.0000 |
| always-jpeg  | 0.9421 | 0.4266 | 1.4689 | 0.0497 | 0.8878 |
| always-png   | 5.4751 | 1.8689 | 11.7862 | 0.0462 | 1.0000 |

**The ledger's 4.5x HOLDS on the dense grid**: best fixed choice = always-avif
at 19.75% mean / 52.13% p90 (coarse-grid figure was 20.4% / 55.1%). But a
CORRECTION: avif does **not** reach every (image,zq) cell — coverage **0.9444**
(jpeg 0.8878, jxl-lossy 0.7745, webp-lossy 0.7398); only the three
lossless-bearing families reach everywhere, and avif's 19.75% is measured on
the 94.44% it can reach. Against the best FULL-coverage fixed family (jxl,
77.95% mean) the picker is 17.4x better.

<<BAND SECTION>>

## 2026-08-30 ~20:0xZ — ROUND 4: THE v1 EXTRACTOR DRIFT, RESOLVED — RUNTIME B ≠ EVALUATED B (drift lane: `1246823c`,`1672cc26`,`35112cab`,`fe6428f2`; doc `benchmarks/v1_extractor_drift_2026-08-30.md`, DATASET_HISTORY §3.27)

- **Drift = masked f228–299 + IW f300–371 ONLY** (100 % of rows, max_abs 0.12, max_rel to 1.0);
  basic+peaks BIT-IDENTICAL stored↔HEAD ⇒ pixels identical (decode ruled out, measured).
- **Bisect: ONE commit, `2dab8f30` (2026-05-17)** — the activity map had read strip-OVERLAP rows
  the fused V-blur never writes (undefined, cross-strip stale state). `2dab8f30`→HEAD moves
  NOTHING (0 cells over tol across every SIMD/streaming/archmage change since).
- **The stored tables never reproduce at their own build commit**: the pre-fix masked/IW block was
  a function of RAYON thread count (1/2/8/28 → four different outputs, all 144 slots, |Δ| ≤ 0.086).
  Intended fix of an unintended NONDETERMINISM — extractor untouched, no golden re-baselined, no
  tolerance widened; the stored 372-era tables are declared STALE for runtime purposes. New gates:
  `v1_372_is_bit_identical_across_rayon_pool_sizes` + `v1_masked_and_iw_blocks_are_thread_invariant`.
- **Runtime B is NOT the evaluated B** (23 of B's 95 live inputs are in the drifted block, incl.
  its largest weight f353): SROCC stored → runtime = CID22 0.87638 → **0.88212** (+0.006), KonJND
  0.54665 → **0.64967** (+0.103), AIC-3 +0.017, TID +0.008, KADID −0.016 (train==val memorization
  corpus falling = the expected sign). **Per-pair dial shift mean −5.0 to −5.9 points, max 17.4,
  ~100 % of pairs > 0.5** — a real train/serve skew (B trained on pre-fix-era features).
  ⇒ Every B row in this ledger (incl. R1b's 0.8763/0.5183 instrument row) is STORED-ERA; the
  runtime bar is HIGHER, especially kon. Comparisons vs B must name which B they read.
- **2026-05-20 canonical-audit claim corrected in CLAUDE.md**: it sampled only f0..f99 — entirely
  inside the non-drifting block — at a commit already past the fix.
- Side findings: kon504 `SRC0437` pair-list defect (mean PJND exactly 58.50; loader rounds to
  `_059`, the committed TSV names `_058` — two different images; R1b's keyed 504 inherits it);
  `zensim-validate --extract-only --format tid2013` drops 120 TID pairs on decode failure (open);
  main's `clippy --tests -D warnings` broken by two excessive_precision literals from the in-flight
  extract-perf lane (`8a98a286`) — handed to that lane.
- **Registered follow-ups**: new DATED 372 root (never overwrite in place), B-lineage re-verdict on
  it, B training re-extraction (~227k pairs, fleet wave). Root+re-verdict lane launched (round 4b).

## 2026-08-30 ~21:0xZ — ROUNDS 5 + 6: fleet claims/wall-time, KB-42 CI green, the dated 372 root + era table

### Fleet lane (zenmetrics `7e4695b3`..`5438f8f5`) — the 312-cell round + wall-time fixes
- ORCHESTRATOR ERROR, owned: the 17:32Z requeue launched on lane 2's pools image (`03bdf64b`,
  built WITHOUT `avif-aom`) — all 312 cells claimed and poisoned in 28 s ("build lacks the
  avif-aom feature" → EncoderPanic, poison-on-first-deterministic-failure). Two zenfleet defects
  fixed with tests: `required_capabilities()` ignored the backend knob (capability gate inert;
  proven both directions — the old image is now excluded 126,360/126,360), and reconcile's Poison
  arm counted nothing (the `rows=0` ambiguity that read as "never claimed").
- Round rerun on a correct image: **253/312 done (222 large + 31 tiny — roots #22-#23 cured some
  tiny cells), 59 poison = 41 REAL divergences (31 screen residual + 10 PHOTO `screen_tools=false`,
  the first non-screen class) + 18 tiny (sc_tools arm)**. Planes staged
  `/mnt/v/output/avifaom-2026-08-30/poison-planes-2026-08-30/` → KB-43 lane launched (in flight).
  Harvest: **125,941 scores / 125,940×944 features; views train 102,373/871, eval8 23,567/209**.
- Wall time (user ask): wave = 12.09 h wall, **49.1 % of 37 box-h idle**; scheduler defect =
  per-box chunk boundaries → 19.6 % duplicate executions (2.939× exec/distinct) →
  `pack_chunks_lpt_uniform` + skipped≠idle shipped, A/B measured **1.000×**, disjoint exact
  splits. Remaining 12.5 idle box-h = operator loop → auto pardon-and-relaunch + completion
  beacon REGISTERED, not started. GPU `ZM_VRAM_CAP` usable = 5,500,000,000 (full 8 GiB OOM'd 23
  cells). Sidecar names overwrite across generations (snapshot = the durable done-set).

### KB-42 (zenav1-aom `c80b40d1`,`cb76cda9`,`5ac5fae6`) — CI GREEN all 7 legs ×2 runs
Three roots, single-carrier premise falsified: (A) census ceiling fired on roots #3-#6's
legitimate IntraBC reachability gain — re-pinned with BOTH floors rising; (B) root #12's
`search_tx_mode_is_select` placeholder `false` at ALL 19 call sites froze search tx-size costs at
frame-init on TX_MODE_SELECT frames (worst at low q; encode cell derived it correctly — why the
census stayed 104/104); fixed = derive `!coded_lossless` everywhere; (C) a test bootstrapped
screen tune without declaring the knob. 316/0 encode tests; census 104/104; process hole closed
(`just gate-encode` + mandatory gate list + PARITY rule 6 — `--lib` runs no byte gates).

### Round-4b (zensim `2d94890c`..`39ffc008`) — the dated 372 root + the era table + a drift-doc CORRECTION
- New root `/mnt/v/zen/zensim-training/2026-08-30-full-features-372/` (8 corpora re-extracted at
  HEAD `ea16c7ee`; csiq/live/pipal BIT-IDENTICAL to stored — pre-fix era touched only some
  corpora; 6 byte-copied with era flags, aic4 unrefreshable). Old root untouched (verified).
  Two more loader defects fixed en route (cid22 path resolution; tid i25 case → 3,000/3,000).
- **Era shift is MODEL-SPECIFIC** (0.00000 on the three basic-only bakes — the negative control —
  up to |Δ| 0.489): **41 ordering flips**; B 4th → **1st on CID22**; `cl_tfm_LQ_MLP` 1st → LAST on
  KonJND (0.761→0.272); composite leader cl_tfm → blend_2L; **the 2-layer blend's "+0.004 CID22
  over B" was an era artifact (current era: −0.0002, B ahead)**; its TID +0.062 / nonphoto +0.088
  survive, KonJND deficit deepens to −0.145.
- **CORRECTION to round 4's §3b**: the drift lane's KADID/TID/AIC-3 runtime rows were key-joined on
  a non-unique `(ref_basename, human_score)` (aic3 100/600 distinct). Corrected positionally:
  runtime B KADID **0.80847**, TID **0.77852**, AIC-3 **0.76501** — TID and AIC-3 now slightly
  BELOW stored, so "runtime B better on every genuine holdout" is FALSIFIED; what stands:
  **CID22 +0.006 and KonJND +0.103** (0 duplicate keys there), KADID down (memorization), TID/AIC-3
  down ~0.01. Drift doc + CLAUDE.md corrected in place.
- Registry: 3 appended annotation entries (stored-era invalidated; basic-only era-independent;
  the 372 dial/corruption grids are themselves pre-fix era and unrebuildable without a decode
  pass). kon504 = two files: R1b keyed (post-fix) vs the 372 root's `_2026-08-29` (pre-fix subset).
- **OPEN — governance (user):** flipping `bake_verdict`'s default `--features-root` to the new
  root moves every future 372 number mid-campaign — needs an explicit decision; until then
  current-era reads use the explicit flag. Board promotion of the 11 `<label>_new.json` verdicts +
  gauntlet regen → lane launched (round 6b). B training re-extraction (~227k, fleet) still
  registered, user-gated.

## 2026-08-30 ~22:2xZ — ROUND 7 (W-LIN PASS) + ROUND 6b (board era rows + the features-root flip)

### W-LIN ROUND 7 — VERDICT **PASS** (`e3cdd752`..`7ac3a8a3`; doc `benchmarks/wlin_round7_rawframe_2026-08-30.md`, 133-arm TSV)
**17/133 arms clear ALL FIVE round-6 bars — the round-6 falsifier is REVERSED by one variable, the
raw target frame** (priced +0.154 kon / +0.105 hfnl vs the mm01 control; flips the generalist head
hfnl −0.083 → +0.288). Rule-selected `PL_P3_KHp6_H_b0.3` (3,589 B): 0.8562 / 0.4915 / 0.8809 /
0.8911 / 0.4162 (5/5). Best-all-panel `PL_T3_KH01_C1_b0.85`: 0.8492 / 0.5197 / 0.9009 / 0.9066 /
0.4125. **B-ruler corrected** (the same-pair restriction was size-correlated, +0.05–0.07 on B's
family axes): runtime-era B on the FULL keyed slices = 0.8821 / 0.5186 / 0.8505 / 0.8609 / 0.3496
= **2/5** — the 3.6 KB linears beat runtime-B on nonphoto/imazen26/hfnl; B keeps cid22 (+0.026)
and dial range (86 vs 59). **Pool block's first clear WIN: +14.7–21.7 dial dynamic range** (rank
effect stays hundredths, matching the carrier lane). NOT a ship candidate: all 5/5 arms have
compressed dials (hf leg = a ≥0.90 band — data-coverage limit); selected arm fails G-RANGE by one
row. Leg collapse: tbig_hf ⊂ tbig, tsafesyn/ttbig = target twins ⇒ ONE extraction (208,169 rows;
G-K1/G-J 0.0 / G-R1 216/216 / G-X 944/944; triple-mirrored). tbig cost MEASURED: fetch 72 min
@48 c/s (dominant), decode 31 min @112 c/s, extract 5m39s @614 c/s — LOCAL by necessity
(103,585/208,169 cells exist only as tar members; not URI-expressible). R7-A1: the shared τ 0.005
zeroed the hf head to a constant bake — caught by pre-registration. **Slicing shortcut FALSIFIED:
fold f0..371 ≠ current-era buffered v1 on 60 % of tbig rows** (handed to the 372-subset lane);
kon-504 features bit-identical across eras. R1b §9 CLOSED: B's ledger kon 0.5935 vs keyed 0.5183 =
two extractions, 371/372 columns differ. SRC0437: |Δ| ≤ 0.0003, no bar changes.

### ROUND 6b — board era rows + **the features-root flip (USER DIRECTIVE, DONE)** (`a25d1b80` + 2 more)
- 11 `@cur372` rows promoted (4 curated pairs incl. B and the new composite leader
  `mlp_2L_diverse_H128@cur372` 0.88191; 7 grid-interior behind a family toggle); never-overwrite
  gate PASS; gauntlet regenerated + full gates PASS →
  `http://localhost:3300/zensim/reports/summer_gauntlet.html` (prior HTML preserved).
- **DEFAULT `--features-root` = `/mnt/v/zen/zensim-training/2026-08-30-full-features-372`** via ONE
  owner constant (`zensim_validate::eval_roots`; was a literal in ten files); every run prints its
  era line; 4 pin tests; flagless vs explicit **byte-identical full-json**; stored-era verdicts
  remain valid as stored-era reads; dial/corruption grids excluded.
- **FINDING: 7 of the 9 "stored-era" board rows were never stored-root reads** — they are
  `regime:"720"` ext720 reads (post-fix root; reproduce bit-exactly), already ≤2e-4 from the
  current read. Only shipped B + `T_appT_b372_lam1e-3` are genuine stored-root reads ⇒ **only B's
  pair is a clean era A/B on the board** (registry `board372-row-read-on-ext720-root-2026-08-30`;
  round-4b §7 attribution handed back to its lane). Two honesty annotations added: 6/14 corpora in
  the current root are byte-copies (39.5 % of product_composite weight rides on them), and the dial
  grid is pre-fix era for ALL rows.
- `family_of()` gate scoping bug fixed (`era_base_name()`); OPEN: board 18.87 MB vs the 12 MB cap
  (was already 18.30 before this round — cap needs a deliberate decision); `--full-json` does not
  record `features_root` (era line is stderr-only — provenance gap, handed to round-4b's lane);
  `LOOP_BAKE_MAP.blend2L_base` left unmapped (era-consistency of the loop columns is one decision).
## 2026-08-30 ~23:1xZ — ROUND 8: extraction perf/removability lane (6 commits; doc `benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md`)

- **Shipped lever: H-blur gather ring** (redundant strided remove-side gathers eliminated across 24
  sites, bit-exact by construction, negative controls run): callgrind Ir @576² buffered-372 −8.29 %,
  zeroed-944 −5.57 %, live-944 −6.22 % (per-kernel `box_blur_h` −21…−22 %). Prior ranked lever #1
  (`box_blur_h_of_abs_diff`) is MIS-RANKED (re-adds a gather to a gather-bound kernel); #2 ≈ 0.4 %.
  Next real target: `dense_block_kernel` (needs re-profiling at v4x).
- **Buffered NOT removable — four blockers, none perf**: the fold has no `score()` (every scoring
  entry + attribution's basic canvas runs the buffered walk); no ref-cached fold form; pool values
  differ at production widths; zensim-gpu's CPU oracle is `compute_extended_features`. And the fold's
  MT scaling is structurally weaker: fixed degree 3 (1T→8T 1.1–1.5×) vs buffered band-per-strip
  (2–4×) ⇒ **buffered is FASTER at 8T for the same pools (1.09–3.30×) and cheaper serial is the
  fold (0.63–0.91×)** — "is buffered slower than streaming" inverts with thread count. α+β fits
  return negative intercepts (per-pixel cost rises with size — the linear model failing).
- **★ PREMISE INVERSION on the "372 segment" mandate: BUFFERED pools over PADDED (phantom)
  columns; the FOLD is the unpadded/clean one.** Pre-padding makes 17/20 geometries bit-exact
  (incl. the 81.6 %-divergent 200×150 and 576/1152 classes; tight widths 0/552 = the control) but
  moves 505–508/552 v2 slots up to 36× — so "fix 372" and "leave 944 alone" cannot both be had by
  padding. FORK → USER: **A** pre-pad everything (invalidates every 944-trained MODEL, beyond
  re-extraction); **B** two plane sets (944 untouched, 372 reproduces buffered; cost unmeasured —
  being priced); **C** stop v1 padding (fold already correct; buffered fixed to match; a NEW 372
  era — tables AND shipped-B runtime move again; B retrain already registered). Residual either
  way: 3 cells at h=93 (≤1.1e-6, pad-column × row-group tiling) — under root-cause. Block-skipping
  does not exist yet (naive 944-then-project = 3.9–4.8× buffered at 28T) — being built now, option-
  independent, along with fold band-parallelism.
- **Lane error, flagged**: `714da506` pushed 54.4 MiB of build artifacts to main (gitignore gap,
  fixed + removed in `7d8ac808`); blobs remain in history — any filter-repo pass is a coordinated,
  user-gated history rewrite (NOT attempted). Benches ran not-load-clean (box shared all day; CV
  24–90 %, effect sizes 1–2 orders above) — callgrind Ir numbers are the quotable ones.

### Round-4b follow-ups landed (`5d393734`, `5507d443`, `4af30b97`)
§7 board-row attribution corrected with INDEPENDENT verification (cl_tfm board row bit-exact to a
fresh `--regime 720` read; only B + `T_appT` bit-exact to the stored-372 run — B's pair confirmed
the board's only clean era A/B; basic-only rows show the same offset vs BOTH eras = folded-720
feature space, not an era). Registry scopes narrowed in place to the 2 genuine rows. §6.1 marked
SUPERSEDED by the `a25d1b80` default flip. **`--full-json` now records `features_root`**
{path, era label (UNKNOWN never guessed), manifest sha, declared regime, per-corpus file sha256s} —
additive, recomputes nothing, 2 tests; stored root honestly reports `manifest_sha256: null`
(it ships `_MANIFEST.md`). Stale-working-copy incident handled without loss (pre-backup +
selective re-apply; no other lane's edits clobbered).

## 2026-08-31 ~00:1xZ — ROUND 9: W-LIN 7b — VERDICT PASS (`d9ead488` pre-reg, `cda35fe8` results; 158 arms)

- **Diagnosis corrected round 7 honestly**: the blends' dial compression was ~40 % SPLINE ANCHOR
  (single-variable `shared-anchor` swap: +22.44 dyn on the round-7 winner), not pure coverage —
  "no monotone spline can repair it" WITHDRAWN for the blends, stands for the hf head (which IS
  saturated: raw span 0.366). **Confound caught pre-quote**: the canonical 944 dial grid is
  `foldapp2` (pools ZEROED), so round-7 pools arms were dialed at the wrong regime →
  `build_dial944.py DIAL944_MODE` rebuilt the twin from the persisted 2026-07-27 pixels (no
  re-encode, G-DIAL 4,817/4,817); measured confound +0.23…+0.78 ⇒ round-7's pool-dial claim stands.
- **Coverage = a RE-CUT, not a re-extraction**: tbig already spans q5→q95 (19,430 rows < 0.10);
  band-stratified cuts (12 registered edges, deterministic stride, no RNG) on both substrates.
- **Pre-registered dial gates added**: G-DYN ≥ 60.0 (70 % of B's 86.08; reachable — Hp_lasso_w10
  80.5 — and discriminating — no round-7 arm cleared it) + G-RANGE PASS; same maximin rule.
- **Results: 15 arms clear all five rank bars + G-DYN (and G-RANGE)**. Winner
  `Q7b_pools_g0.2_a0.2_b0.97` (3,583 B): 0.8588 / 0.5118 / 0.8778 / 0.8873 / 0.4056, dyn 61.72,
  G-RANGE PASS — beats the round-7 winner on EVERY registered axis at the same size; sibling
  g0.25 variant reaches dyn 71.67. B stored/runtime: 0.8764/0.8821, 0.5186, 0.8505, 0.8609,
  0.3496, 2/5, dyn 86.08. Head-level: the 7b hf head's dial 25.94 → 89.78 (past B) at p5 3.57.
- **Non-obvious mechanics**: swapping the hf head outright FAILS (32 arms, 0 at 5/5 — the two hf
  heads are COMPLEMENTS; the four-way HG = H7b × H closes it); and the two fixes are SUBSTITUTES —
  after the re-cut, the anchor that bought +22.4 costs 8–13 and plain safesyn is best (fixing
  coverage removes the anchor's leverage — the user's directive was right).
- Corrected-fold (option-C) confirmation pass for the winners: REGISTERED, not run (sequencing).

## 2026-08-31 ~00:4xZ — ROUND 10: extraction lane, option-independent items DONE (4 commits; doc §6-§8 extended)

- **Block-skipping SHIPPED** (`V2NewFeatureToggles::v1_only`, additive): 372-only fold = **249.2M Ir
  = 0.743× buffered** (25.7 % below) and 0.466× of 944-full — 53.4 % of the walk removed; gate
  `folded_v1_only_matches_full_walk` (bit-identical, 5 geometries × 3 pool modes × serial+rayon;
  caught a regime-from-compute-flags bug — layout and compute now separate).
- **MT structural cap FIXED**: the fold saturated at exactly 3 threads (2.26×) — band parallelism
  (4 bands/strip, SEQUENTIAL-ORDER merge so f64 sums stay bit-exact) lifts 944-full to 2.57× @8T
  (−14 % wall best-case), regression-past-3T gone in all modes. Trap caught: `map_init` scratch
  re-allocation LOST 15 % — persistent per-band slots fixed it ("parallelism that allocates isn't
  parallelism").
- **h=93 residual DISSOLVED** — it was an artifact of the option-A pre-pad workaround; under C the
  divergent classes go to exactly 0. Resolved by deletion.
- **Option C measures as a PERF WIN, not a cost**: fixed-buffered Ir −9.02 % @576 / −7.37 % @1152 /
  +0.00 % at the tight-width control. **And the v1 golden fixtures are 64×64 = tight-width —
  structurally BLIND to the defect C fixes** → C rollout requires a non-tight golden geometry.
- **Which world (user's steer answered)**: at ≥8T **944-full ≈ 944-zeroed** (identical @576², +9.7 %
  @1152²; band parallelism absorbed the pool work) — 944-full's overhead does NOT justify a separate
  path by itself; `v1_only` stays as ONE boolean (0.7× buffered serial) for the ~25 v1-372
  consumers, not a third pipeline.
- **Fusion trap measured** (user's register-spill steer): folding activity abs-diff into the H-blur
  LOSES (+1.0/+2.0 %) post-rem-ring — reverted and DELETED. Spill audit: rem-ring clean (1 store /
  2 loads); `fused_vblur_ssim_inner_v4x` carries 28 spill loads = the standing fission candidate.
- **Named MT blocker for retiring buffered**: the serial `StripPlaneProducer` (buffered still ~2-3×
  ahead at 8T; its parallelism grows with image height, the fold's tops out at 3×4). Next axes
  (sent): C implementation + 372 era step (non-tight golden, era-3 eval root, B-under-C delta),
  producer parallelization, the fission experiment, profile-driven work re-based on corrected
  semantics.

## 2026-08-31 ~01:2xZ — ROUND 11: option C LANDED; era-3; B-under-C a non-event; single-mode taxonomy (5 commits, `56bbcda2`..`f769e7b9`)

- **C shipped** (`pyramid_plane_stride` = the one owner; `mirror_pad_columns` + 3 call sites DELETED):
  buffered v1-372 Ir −9.02 % @576 / −7.37 % @1152 / +0.00 % @tight-control 592; the fold needed no
  change (verified structurally — its production path never references the owner) ⇒ **no 944 table
  or model invalidated**. Three gates inverted meaning, each a TIGHTENING (differ-bounded → equal-
  exactly; §3.30 table); no tolerance widened.
- **Goldens: re-VERIFIED, not re-pinned** — 64×64/96×96 are stride-invariant so C leaves them
  bit-unchanged (that is the finding); `GOLDEN_NONTIGHT` (200×150, the 81.6 %-divergence geometry)
  added with the one-line negative control: restoring era-2 padding fails ONLY the new fixture.
- **Era-3 root** re-extracted (8 corpora + kon504; six copied corpora stay prior-era, registered;
  `pipal` byte-identical = the tight-class self-validation). Found+fixed a silent-loss bug: `zv
  cid22` at the dataset parent wrote a 34-byte empty cache and EXITED 0 ("4292 pairs in 0.0 s");
  the era-2 build had hit the same and been hand-patched.
- **Shipped B under C: a NON-EVENT** — cid22 +0.000024, konjnd −0.0046 (|·| ⇒ improves), pipal
  exactly +0.000000 (control); features move materially at non-tight widths, pooled rank stats
  barely move. `DEFAULT_FEATURES_ROOT_372` NOT flipped — user decision; caveat: six era-3 corpora
  are copied prior-era rows, so a flip makes the default root ERA-MIXED.
- **Taxonomy per the user's single-mode decision**: `v1_only` = `#[doc(hidden)]` instrumentation
  (pub(crate) impossible — `..Default::default()` visibility from external crates, incl.
  zenmetrics); `V1PoolsMode` untouched (fleet harvests); `fold372_only` bench arm deleted;
  `toggles_off` survives only as the pool-block price control.
- **Final 944-full table (1/8/16T)**: 576² 17.33/7.17/6.83 ms (best 2.54×), 1152² 76.0/33.0/34.5
  (2.30×), 2304² 381.7/180/180 (2.12×); **Ir/MP flat (1628→1546 M) across 16× pixels ⇒ the wall
  convexity is memory-system**; RSS thread-independent (39/76/163 MB). Scaling peaks at 8T.
- **Producer parallelization NOT done — measured ~8 % of the walk** (the earlier "named blocker"
  re-priced); the real ceiling is `dense_block_kernel` (23 %, 3-way, not bit-exactly row-splittable
  as accumulated today) + Y-channel imbalance → next: per-band self-contained accumulators, ONLY
  under bit-exact sequential merge (a byte change to 944 outputs is an ERA decision, not a lever).
- Score: **4 levers shipped** (rem-ring, block-skipping, band-parallel fold hook, C) / **4
  implemented-then-rejected on measurement** (activity fusion +1.04 %, map_init scratch, row-
  parallel blur, option-A pre-pad) — the rejections are the measurements working.

## 2026-08-31 ~01:5xZ — ROUND 12: the dense fork priced and STOPPED (`091dc35a`; no code changed — the decision is the deliverable)

- **Grouping obstacle located exactly**: dense's f64 accumulator takes one add per row ONLY on
  `v4x` with `width % 8 == 0` (`POOL_SIMD` is v4x-only by design — 16-register tiers would spill;
  the scalar tail fires at 3 of 4 scales on 200×150). Measured ulps: 0 (v4x, w%8==0) / −2 (tails) /
  13 (per-pixel pools). A restructure bit-exact on one tier at one width class = a second output
  regime hiding inside the kernel.
- **Gain vs cost**: dense = 23.2 % of the walk; Amdahl UPPER bound 1.17× @8T / 1.23× @16T — against
  making prior-era the 11 canonical legs, tbig 5.7M + 21 views, kadis700k_924, the eval
  instruments, AND every 944-trained model. Not traded silently. If ever wanted: ONE-TIME byte
  change giving every tier a POOL_SIMD-equivalent path + tails folded into row-local accumulators
  (= permanently band-parallelisable), paid once — best bundled with the next unavoidable era break
  (e.g. the future HDR-features regime).
- **Y-channel imbalance: no free rebalancing exists** — disjoint accumulators already work-steal;
  the imbalance IS Y-only work (append/BANDVIS/CSFW); pipelining X/B rejected on expected value.
- **Fission probe retired pre-implementation**: of the 28 spill loads, ONE is in the innermost loop
  (188 of 1750 insns) — no hot-loop pressure to relieve; cost of the answer = one objdump.
- **Buffered-retirement checklist (§15) updated to what is true**: C closed blocker 2 (width
  divergence). Remaining: 1 fold `score()` (the gating one), 3 ref-cached form, 4 attribution's
  basic canvas, 5 zensim-gpu CPU oracle — all additive-API/oracle work, no measurement risk.
  Blocker 6 (MT) now BOUNDED: ≤ ~1.2× headroom, era-gated ⇒ **retirement plans on the API blockers
  and must not wait for buffered-class scaling** (unreachable without changing 944 bytes).
- Extraction-lane scorecard final: 4 shipped / 5 rejected-on-measurement / 2 era-gated decisions
  surfaced (root flip; the one-time accumulation-shape change).

## 2026-08-31 ~02:1xZ — ROUND 13: PERF ERA BREAK AUTHORIZED (user directive)

User, verbatim: "we can do an era break in order to push performance higher, remember the point of
zensim is to be extremely fast, and as good or better than ssim, and be good at hdr."
⇒ The era-2 lane is launched: all-tier POOL_SIMD-equivalent accumulation + scalar tails folded
row-local (dense permanently band-parallelisable; the 8T plateau re-attacked on the new shape);
the rejected-for-byte-stability pile re-opened ONCE (art-L4 fused sums etc. — each must still WIN
on measurement); ONE batched era, layout/slots unchanged (append-only holds); era gates =
same-binary determinism (bit-exact) + thread-count invariance BY CONSTRUCTION + measured/declared
numerical equivalence to era-1 + rank preservation (B + roster on same-pair era-1-vs-era-2);
blast radius + re-extraction/retraining waves (944 roster + the registered ~227k B re-extraction
fold in) to be REGISTERED with priorities, launched on user go; hdr944 route inherits the kernels
(enumerated); the future HDR-feature append lands ON era-2. Sequencing with the fold-engine lane
(score()/ref-cache/attribution/oracle, running on era-1 parity gates) coordinated explicitly.
Mission memory saved (user_zensim_mission).

## 2026-08-31 ~02:5xZ — ROUND 14: era-2 S0/S0b + the S1 accumulator decision

- **S0 design-before-code** (`78f3f988`): batched-era principle; the re-opened pile is honestly ONE
  item deep (only art-L4 fused sums were rejected for byte stability alone); era stamp; blast
  radius R1–R7 registered, launched by nobody; HDR-route cleanliness; fold-engine sequencing.
- **S0b — the oracle landed FIRST** (`41f0b44d`; Neumaier L1 + Shewchuk-exact L2, test-only
  `oracle` feature) **and caught two errors in the lane's own bound analysis before judging any
  kernel**: (1) the first model bounded only summation — the f32 term-evaluation error was missing
  (measured coefficient 40 vs predicted 20); (2) cancellation amplification: `d = max(1−local, 0)`
  is a cancelling difference, so moment bounds go as Σ|d^(k−1)| not Σ|dᵏ| — 283× apart here.
  One uniformly-valid cancellation-safe bound replaced the patchwork; gate green, worst case at
  18.93 % of its proven bound. A relative A/B could not have caught either — the whole argument
  for the user's oracle demand.
- **Arch-dependence enumerated in source**: zero `mul_add` in dense, no transcendentals, `rsqrt`
  already rejected in-tree with the vendor-seed-table reason ⇒ cross-ARCH identity is plausible,
  neon/wasm declared not-verifiable-from-this-box (never asserted).
- **S1 DECIDED (orchestrator, on the user's no-slower-accumulators directive): 8 f32 virtual lanes
  + f64 band layer.** Measured: f64 chunked lanes +132–147 % on the accumulation step; naive
  `lane[x % 8]` +780–915 % (modulo defeats vectorization — `as_chunks::<8>` required, trap noted);
  f32 lane error ~4.3e-6 relative = 3 orders under dial materiality (5e-3) and ~100× under the
  5e-4 pool policy; IEEE f32 `+` correctly rounded on every arch ⇒ the bit-identity theorem holds
  unchanged. v4x's 16-wide accumulation narrowing to the fixed 8 IS part of the break.
- Sequencing: era-2 kernels land AFTER the fold-engine lane pins its remaining parity stages (it
  reached stage 4 — **the fold has `score()`**); then the byte change + both paths' gate re-pins go
  in one/adjacent commits. Fleet metric suffix remains a registered zenmetrics item for the
  re-extraction fleet lane.
