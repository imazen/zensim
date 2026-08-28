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
