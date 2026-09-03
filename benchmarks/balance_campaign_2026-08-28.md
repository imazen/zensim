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

## 2026-08-31 ~03:1xZ — ROUND 15: R2→LAN mirror COMPLETE (zenmetrics lane, 13 commits, latest `10493507`)

**457.54 GiB moved, 32 prefixes verified equal on both stores, 0 mismatches; nothing deleted from
R2.** Inventory from USAGE (48 rows): P0 369.6 GiB (all 61 bigcodec box tars — the round-7 fetch
bottleneck — + all 53 variant indexes + canonical/2026-06-27 metadata + zenwebp_lossless encodes
40,473 + eval-grids + canonical-2026-05-21 + both kadis canonicals + every ext/tbig/kadis/hdr
table) + P1 213.5 GiB; 609 GiB / 1.45 M objects measured and deliberately NOT mirrored; refs/jobs/
blobs already present (verified, not re-copied). **6 of 7 bigcodec datasets are now byte-complete
+ fetchable via tar+index on the LAN store** (each proven by hashing a real member offset from
both stores); `zenjpeg_lossy` has tars but NO index exists anywhere (needs building). Four
`encodes/` plain-GET prefixes deferred (2.77 M objects; bytes covered by tars; do tower-LOCALLY
after the Unraid mover — running 3 d 21 h — drains). Capacity: array 20 T free (42 %);
SeaweedFS writes round-robin across tiers ⇒ NVMe cache took +210 GiB (849 G free) — first
snapshot said untouched, corrected in the doc.

**Traps found (all recorded in `zenmetrics/benchmarks/r2_lan_mirror_2026-08-30.md` + the reusable
`scripts/lanstore/mirror_r2.sh` with the rules baked in):** `s5cmd ls` UNDERCOUNTS
non-deterministically against SeaweedFS (40,450/40,466/40,467 vs true 40,473 — any s5cmd-diffed
mirror re-transfers phantoms forever; aws-lister only); **`s3env.sh`'s LAN arm re-exports the LAN
key AS `R2_ACCESS_KEY_ID`** — anything sourcing it then talking to real R2 gets EMPTY listings,
not an auth error (registered for a zenfleet fix, unassigned); ~333 small-object PUT/s took the
tower to load 30.7 / 48.9 % iowait / S3 unresponsive (recovered ~12 min; load gate now in the
tool; large sequential is fine); `zen-lanstore` runs volumeSizeLimitMB=1024 (30× under default →
1,123 volume files; flagged, unchanged); pre-existing dangling `volume 473` filer retry loop.

## 2026-08-31 ~04:1xZ — ROUND 16: THE FOLD IS THE ENGINE (8 zensim commits `7a843375`..`a2918021` + zenmetrics `92bdec00`)

All six stages landed. **Additive API (for approval): 4 items, ALL `#[doc(hidden)]` and ALL behind
`feature-regime-v2` (not a default feature)** — `zensim::fold_engine` (module),
`fold_engine::ScoringEngine {Buffered|Fold}`, `Zensim::with_engine`, `Zensim::engine`. Default
build's surface AND behaviour byte-for-byte unchanged; two pre-registered tier-2 items were NOT
built (stage 3 needed no new reference type); no existing signature changed.

**Gates:** fold-backed score = `to_bits()` equality on score/raw_distance/all features/mean_offset
across 18 geometries × {serial, rayon} × profiles B + PreviewV0_2, incl. `classify` and rayon pools
1/2/3/8/16; the golden gate states the fold against the **PINNED golden arrays**, not the buffered
path (a cross-path test would pass if both drifted together); ref-cache N-vs-1 **bit-identical**
(cross-engine worst |Δmean_offset| 8.674e-19); attribution density + SAT block sums bit-identical
across engines with all 26 pre-existing attribution gates unchanged; GPU oracle A/B same 6 names,
same CPU values. Full suite **378/0**, clippy clean at 3 feature combos.

**Perf (same `ZensimResult` bit-for-bit in both arms — not a 944-vs-372 table):** serial PARITY
(1.03×, CI crosses 0) at 1152²/2304²; under threads the fold is 2.30×/2.54× at 8T and 2.93×/3.25×
at 16T slower — entirely thread SCALING (1T→16T at 2304²: buffered 6.15×, fold 1.95×, flat past
8T), not work. Serial parity is NEW: a v1 score now asks the fold for `v1_only + PoolsMode::Full`
(53 % of the walk skipped). `Buffered` remains the DEFAULT — no regression ships. Ref-cache saving
fold −7.3…−9.7 % vs buffered −13.7…−20.4 % (cause measured: the fold allocates a fresh `V2Scratch`
per compare — no fold-side `compute_with_ref_into`). Fused-vs-split 2.1× serial / 3.05–3.40× at
8–16T. 576² cells CV 58–221 % — not quoted.

**Retirement: REGISTERED PROPOSAL, nothing deleted.** Blockers 1/3/4/5 CLOSED (score, ref-cache,
attribution canvas, GPU oracle). **The new gating blocker: `feature-regime-v2` is not a default
feature — a default `cargo add zensim` build contains no fold at all**, independent of every parity
gate. §9 carries 4 prerequisites, 7 entries still routing to buffered by design, a 7-step deletion
order with the pinning test for each, and what must SURVIVE the deletion (the pyramid cache — move
it out of `streaming.rs` first; the XYB front end; `compute_delta_stats`; `compute_xyb_mean_offset`
as the DEFINITION `MeanOffsetRows::finish` reproduces).

Open (measured): the fused compare is not re-hosted (its in-strip fold is tiled to
`BAND_ROWS == STRIP_INNER`; falls back, costing 2.1–3.4×); no fold-side `compute_with_ref_into`
(additive API, registered, outside the approved list); PU-linear, >16 MP strips, weight-skipping
linear profiles, `with_stop`, `num_scales != 4` all named in §9.3. The predecessor's §15 and
`zensim/CLAUDE.md`'s "buffered is not removable today" block were stale in four ways and are
corrected in place.

## 2026-08-31 ~05:0xZ — ROUND 17: era-2 — THE CROSS-VENDOR HYPOTHESIS IS MEASURED; the flip is blocked on a perf number

**`era2_vendor_probe`** (one binary, no `target-cpu=native`, all 35 dense slots as raw f64 bits,
both eras, run on this box AND i134 over ssh):

| box | CPU | vendor | tier |
|---|---|---|---|
| dev | AMD Zen 4 | AuthenticAMD | **v4x (AVX-512)** |
| i134 | Intel i5-13400F | GenuineIntel | **v3 (SSE4.2)** |

**era-1 differs on 66 of 105 slots (63 %); era-2 on 0 of 105.** The `reduce_add` hypothesis
graduates from plausible to **MEASURED on one pair** — and the confound is named, not hidden:
vendor and tier are not separable here (AMD picks v4x, Intel v3), so it is honestly a CROSS-TIER
result that a vendor difference induced. That is the same shape as the historical failure
(non-AMD classes agreeing with each other = clustering by reduction shape), which is why it is
consistent — "consistent with" is as far as one pair goes; `neon`/`wasm` still unverifiable here.
**Registered, not taken:** era-1's golden policy is a TOLERANCE precisely because exactness never
held cross-vendor; if era-2 holds on the CI matrix, **re-tightening the golden gate to EXACT
becomes a user option** — recovering a property abandoned 2026-08-05.

**Perf: 10.3× → 4.4× → 2.24×, and two of the three causes were the lane's own.** (a) The 10× was
the trap its own §13 had written down — `let n = (width - x).min(8)` is a RUNTIME bound LLVM will
not vectorize; writing a trap down is not the same as not falling into it, and only the bench
caught it. (b) The 4.4× was ISA: plain Rust compiles to baseline SSE2 while era-1 sits inside an
`#[arcane]` `target_feature` region — fixed by wrapping the identical body in `#[magetypes]` +
`incant!`. Both fixes verified BYTE-NEUTRAL (oracle deviations identical to the digit; the vendor
probe re-run under the final build still 66/105 vs 0/105). **Remaining 2.24× is attributed**:
era-1 one pass vs era-2 two, with a four-plane scratch round-trip per row adopted to fit
16-register tiers — and the lane's own §14.4 licenses the fix (pass split is byte-NEUTRAL, so it
may differ per tier): single-pass on v4x (32 registers; era-1's `POOL_SIMD` path proves a fused
pass fits), two-pass elsewhere, byte-identical by construction.

**Accuracy (§10.5 filled from measurement):** era-2 is slightly MORE accurate on the two dominant
families (core −3.8 %, pools −3.6 %), marginally worse on two much smaller ones; every variant
under its proven bound (worst overall is era-1's `pools_scalar` at 18.9 % of bound).
**The flip is correctly BLOCKED on the perf number** — a break justified by speed does not ship
before its speed is known. Next: the per-tier pass split, then flip + era stamp + rank
preservation + golden re-pins.

## 2026-08-31 ~05:3xZ — ROUND 18: KB-43 — 28 of 31 screen cells closed; the wave is 125,973/126,000 (zenav1-aom `58204b29`..`ff0dbc99`, CI green ×2)

- **Root #24 — `get_tx_mask`'s mandatory DCT_DCT fallback never ported** (`tx_search.c:1948-1952`):
  both est-rd prune arms can legitimately clear the INTER tx-type mask (`prune_txk_type_separ`
  returns 0xFFFF when its best horizontal candidate is skipped; the combine loop can end
  `num_cand == 0` and then read a tail slot outside the mask). C pins DCT_DCT; the port evaluated
  nothing → `search_tx_type_inter` = None → the whole IntraBC candidate dropped (238 empty-mask
  bails in one 256² frame). **Closes 13 cells = exactly the cpu-4 subset** (that prune is the
  speed-4/5 arm).
- **Root #25 — `av1_allow_intrabc(cm)` was read off "the DV search runs"**: root #10 nulls
  `PickFrameCfg::intrabc` at speed ≥ 8, which also dropped `intra_mode_info_cost_y`'s
  `intrabc_cost[0]`; C zeroes the header flag only AFTER the tiles. Every C YMODE rate was exactly
  3 units above the port's, `best_rd` 58 tighter, tipping a V_PRED bail. Explicit
  `PickFrameCfg::search_allow_intrabc`. **+15 ⇒ 28 of 31 screen cells byte-exact.**
- **Root #26 — `av1_nn_predict` is RTCD-specialized; the port had transcribed `av1_nn_predict_c`**:
  AVX2 order ported and pinned against the DISPATCHED C (new `shim_nn_predict_dispatched`; 408
  cases, dispatched ≠ `_c` on 167, port matches dispatched on all). Deliberately NOT wired into
  `finish_decision` (an AVX2 DNN over #27's scalar CNN models neither chain).
- **Root #27 — REGISTERED OPEN, the carrier for the last 13**:
  `av1_cnn_convolve_no_maxpool_padding_valid` is dispatched too. Proven on `2765x4096_cq6_s6`
  mi(0,352): the port's 25 DNN features are BIT-IDENTICAL to the oracle under `AOM_SIMD_CAPS=0`;
  raw logits −3.86037111 vs −3.8603348731994629 land on adjacent 1/512 quanta straddling
  `no_split_thresh = −3.858222961`, flipping `do_square_split`. That family's gate rationale ("the
  gap stays inside the prec-reduce bucket so flags never flip") was FALSE — a gap need only
  straddle a boundary; corrected, nothing relaxed. Cost to close: bit-exact `cnn_avx2.c` (two
  reachable specializations) + a NEON twin.
- Bonus: a real CI flake fixed (the forced-scalar dispatch sweep asserted a round trip against a
  snapshot another test could invalidate).
- **Gates**: census 104/104 across 14 dirs (`unexplained: []` everywhere), `just gate-encode`
  152/0, test-fast + test-fast-scalar 564/0, probes stripped, **CI green on all seven legs** (runs
  `33337936828`, `33344816114`).
- **Fleet round** on image `exec-zensim944hdr-8662064f` (built from a verified-clean
  zenav1-aom `58204b29`): all 59 pardoned, drained in one pass per box (18/9/5) ⇒ ledger
  **125,941 → 125,973 done, 27 failing**, cell-for-cell the two registered classes (**14 tiny**
  C3 — four of the original 18 closed as a side effect — and **13 large** = root #27: 3 screen
  cpu-6 + all 10 photo). Score runs gap4 16/16 both, 0 failed. Write-back over all 8 runs →
  125,973 scores × 10 / 125,972 features × 944, `miss_sha=0 miss_score=0`; **views train 102,405 /
  eval8 23,567**; prior harvest kept as `…-pre-round3.bak`.

## 2026-08-31 ~06:0xZ — ROUND 19: bigcodec retired from R2 (user-gated deletion; zenmetrics `fdc473fe`)

**2,815,191 objects / 311,894,986,018 B (290.47 GiB) deleted; the tree re-lists as `Total Objects:
0`; nothing outside it touched.** Enumeration was taken BEFORE the first delete (15 groups; it
caught that `_regroup/` had never been mirrored — mirrored + verified 1/14,561 first).
Verification per class: (a) 54 mirrored objects re-checked count+bytes on BOTH stores at delete
time, 0 missing / 0 differing; (b) **zenjpeg had no index anywhere, so one was BUILT** — its 8 box
tars streamed from the LAN store through the owner tool (extended with `ZEN_INDEX_ONLY=1`, since
the declaring path also writes the `manifest.json`/`control.json` that make a run claimable):
`bf-zjpeg-t0..7`, **1,484,010 members = the exact encodes count**, 13 member reads byte-for-byte
identical to the R2 objects; then set-level coverage on all five `encodes/` prefixes = **0
uncovered** (the first run was DISCARDED, not reported — `comm` had warned "not in sorted order"
from mixed collations; recomputed under `LC_ALL=C` with `sort -c` self-checks); (c) all 2,815,191
keys classified before the first delete: 2,815,122 tar-covered + 54 mirrored + 15 box-tar-covered,
**0 unclassified**. Post-deletion: **6/6 datasets read end-to-end from the LAN store via
index→tar byte range with R2 already gone**. Deleted-keys manifest (31.4 MB gz, sha256
`66f30cd5…`, 2,815,191 lines) uploaded and round-trip verified BEFORE the first delete.

**Cost, read live:** `DeleteObject` is FREE on R2 — the deletion cost nothing; the recurring
saving is storage-line only, **$4.36–4.68/month** (311.895 GB × $0.015/GB-mo), which does NOT
touch the operations spend the LAN migration targeted. Lane defect recorded: 353 delete batches
failed `MalformedXML` because its driver globbed temp payloads a parallel shard was writing — R2
rejected every one (nothing deleted), unmarked batches retried, driver fixed, final state settled
by the tree listing at 0 rather than a batch counter.

**Durability, corrected:** the "one physical machine" worry was WRONG — the 61 box tars carrying
every encode byte were outside the deletion scope and remain on R2 *and* tower, so the compact-form
off-site copy already exists. The real gap this created: **`originals/` (3 tars, 1.03 GiB) is now
LAN-ONLY** (no R2, no `/mnt/v`) — cheap and acute; `bf-zjpeg-t0..7` is also LAN-only but
regenerable in ~8 min. If the R2 box tars are ever retired too, a second compact copy becomes
urgent. DATA_PROVENANCE carries the stop-block (bigcodec = LAN-primary, R2 removed 2026-08-30,
tar+index is the fetch path).

### Round 19 addendum — the single-copy gap closed (zenmetrics `f240bb93`)

`originals/` (3 tars, 1,100,933,120 B) + `bf-zjpeg-t0..7` (8 index files, 242,272,708 B) copied to
`s3://zentrain/_archive/bigcodec-2026-08-30/{originals,indexes}/`; **11/11 sha256 identical to the
LAN source**, LAN copies untouched, per-object shas recorded in the mirror doc §7.7. R2 (not
`/mnt/v`) because the risk being closed is "the tower is lost" — `/mnt/v` is the same building,
power and network on a 74 %-full volume, so it would have made the count two while leaving the
failure mode intact; and `originals/` is IRREPLACEABLE (every encode re-derives from originals +
a codec; nothing re-derives the originals). `bf-zjpeg` was second-copied rather than left
regenerable for symmetry (the other 53 `bf-*` index families were already on R2) and filed under
`_archive/` so a bare index cannot be mistaken for a run; the `ZEN_INDEX_ONLY=1` regeneration
command + its correctness bar (8 boxes must total 1,484,010 members) are recorded regardless.
Cost 1.343 GB × $0.015 = **$0.02/mo against the $4.68 freed**. Lane self-correction: a first
upload silently did nothing (`nice` cannot invoke the `aws_r2` shell function) — caught by
`--summarize` reading `Total Objects: 0`, redone; the "uploaded" echoes in that pass were false.

**R2 accounting after this round (measured, for the retire-more decision):** deleted =
`canonical/2026-06-27/` 290.47 GiB (now 0 objects). STILL ON R2 and mirrored to tower: the **61
box tars, 219.3 GiB**, under `zentrain/jxl-lossy/runs/<run>/variants/` (mandfix4-zenavif 47.20 /
jxl-lossy-vardct 30.80 / mandfix2-zenjpeg 44.19 / mandfix2-zenwebp 29.71 / jxl-modular 54.60 /
mandfix2-zenpng 12.81) — these are the SAME bytes as the deleted `encodes/`, in compact form, and
are now the ONLY off-site copy of the corpus. Measured, un-mirrored, no active LAN reader:
`kadis-700k-gpu/distorted/` 197.43 GiB (699,999 PNGs, regenerable from links),
`codec-corpus/picker-sweep-2026-06-22/` 165.91 GiB (superseded by canonical/2026-06-27),
`codec-corpus/kadis-hdr-2026-07-13/` 66.54 GiB, `codec-corpus/synthetic-v2/` 38.22 GiB (archived
on tower). Recommendation to the user: KEEP the 61 box tars as the off-site copy; retire the
superseded picker-sweep + the regenerable kadis distorted PNGs instead (~363 GiB, no durability
loss).

## 2026-08-31 ~07:0xZ — ROUND 20: box-tar dedup + KADIS originals, MEASURED (user questions; zenmetrics `50899f9d`, doc `benchmarks/bigcodec_dedup_and_kadis_originals_2026-08-31.md`)

**Q1 — the box tars are NOT deduplicated; 22.62 % of their payload is duplicate.** Corpus shape,
exact from the 61 indexes: **5,742,669 members / 231,078,618,620 B payload (215.21 GiB)** inside
235,474,984,960 B of tar (4.09 GiB header/padding), from **4,497 distinct source renditions**;
per-family payload reconciles to the byte with the R2 `encodes/` totals. The indexes carry **no
content hash** (schema `name⇥offset⇥size⇥name`; the trailing 16-hex token in a member name is the
KNOB-TUPLE hash, not content), so: group by length, then PPS-by-potential-dup-bytes over
(family, rendition, size), 2,400 draws over 2,241 groups, every member of every sampled group
sha256'd via ranged GETs (2.21 GiB read, no whole-tar pulls). Duplicate bytes **52,268,279,641 =
48.68 GiB [48.30, 49.03]**; per family: jxl-lossless **63.9 %**, webp 23.7 %, jxl-lossy 19.1 %,
png 3.0 %, avif 2.1 %, **jpeg 0.0 %**. Cross-rendition stratum: 2,310 members hashed, **0
collisions** (≤ ~0.40 GiB by rule of three). **Mechanism: 90.7 % of multi-member groups have q
fixed and only the knob tuple varying — knob tuples that emit byte-identical bitstreams.**
Content-dependent, NOT a prunable grid defect (25 zjxlm renditions → 20 distinct (q,knob)→content
partitions), so only content-addressing recovers it. **Name-level duplication is exactly ZERO**
(5,742,669 members, 5,742,669 distinct names); rendition-level is 100 % by design (all 4,497
renditions in all 6 families) — rendition reuse ≠ byte duplication. Unique content **166.53 GiB**;
dedup opportunity **48.68 GiB per copy**. The tars exist in **THREE** copies (R2 + LAN SeaweedFS +
tower `zen924/tars/`, re-verified 235,474,984,976 B) ⇒ **dropping a copy is the bigger lever
(219.30 GiB each)** than deduplicating one.

**Q2 — 140,000 KADIS originals, they exist twice, and the distorted PNGs do NOT regenerate.**
140,000 verified four ways (distinct `source_id` 0…139,999 and `source_filename` in the GPU
canonical, the 2026-06-30 canonical and `kadis700k_924.parquet`; plus `/mnt/v/datasets/kadis700k/
refs/` = exactly 140,000 files, set-matching the parquet with 0 missing / 0 extra both ways).
Originals live in **two byte-identical copies** — `/mnt/v/datasets/kadis700k/refs/` and
`s3://zentrain/kadis-700k/refs/`, both 140,000 objects / 44,632,022,736 B (41.57 GiB), 3/3 sha
spot-checks identical; NOT on tower, NOT on the LAN store. **`kadis-700k-gpu/distorted/` =
699,999 objects / 211,990,306,109 B (197.43 GiB), R2-only.** ⚠ **NOT faithfully regenerable**:
measured 2026-07-24, `kadis_distort.io.rng_for()` is not the seed the 2026-06/07 generator used
(stochastic types come out mean |Δ| ≈ 9.8 different; `serve.py` carries two seeding schemes and
picks the name-based v2 when `ref_name` is present), and even bit-exact regen would not restore
validity — all 7 metric scores and every 372/720/924/944 feature vector were computed on THOSE
pixels, and `negrich` is *defined* as `score_zensim_gpu < 0` on them. Retiring them costs no
existing work (all four regimes extracted + triple-mirrored) — it costs the option to extract a
FUTURE regime (e.g. the anticipated HDR append): 197.43 GiB / ~$3.18/mo against re-labelling 700k
cells. **Middle path: keep only `negrich` (167,034 of 699,999 = 23.9 %).**
⚠ Flagged, other repo: `~/work/kadis-distort/docs/DATASET.md` still claims the distorted PNGs are
"cheap to regenerate deterministically" — falsified by the 2026-07-24 measurement.

## 2026-08-31 ~07:3xZ — ROUND 21: FOLD THREAD SCALING — 1.95× → 3.71×, and the machine bound is now the limit (6 commits `ae83a5ca`..`469c5a2c`; doc `benchmarks/fold_mt_scaling_2026-08-31.md`)

**Shipped, zero bytes moved:** 2304² fold-backed scoring 1→16T scaling **1.95× → 3.71×**, ratio to
buffered **3.25× → 1.69×** (108.6 → **54.47 ms**); at 8T **2.54× → 1.46×** (113.1 → 59.67 ms);
1152² **1.94× → 3.49×**, ratio **2.93× → 1.61×**. Serial parity preserved (0.96×/1.04×).

- **Profiled first, and the inherited premise was WRONG**: new `zensim/src/fold_timing.rs`
  (env-gated per-phase wall AND task-busy, so occupancy is measured) puts the 2304²/16T critical
  path at producer 29.0 % (serial), phase A 35.6 % (occupancy **0.157**), phase B 28.6 % —
  **`dense_block_kernel` never appears**, because a fold-backed SCORE runs `v1_only`, which gates
  the whole dense/gradient/append block off. The predecessor's "23 %, era-locked, ~1.2× MT
  ceiling" applies to the 944 EXTRACTION, not this path; its "row-parallel H-blur is NEUTRAL" was
  measured where phase A is small.
- **Six byte-neutral levers**: H-blur row bands aligned to the kernels' own row-transpose group
  (16 on v4/v4x, 8 elsewhere); two-sided producer conversion + 6-way downscale cascade;
  `ADVANCE_ROWS` 128→256; fused per-channel fan-out (the only cross-channel edge is absent under
  `v1_only`); **self-blur bands** (each band blurs the rows it consumes into private scratch —
  buffered's shape, +40 % blur compute and faster anyway); and the serial `mean_offset` pass
  parallelised.
- **The ceiling is MEASURED, not asserted**: N independent single-threaded PROCESSES saturate this
  box at **3.5× (fold) vs 10.9× (buffered)** from the same serial speed — the fold's larger working
  set is memory-bandwidth-bound. Against that latency bound the threaded implementation sits at
  **94 % (8T) / 108 % (16T)**: there is no scheduling left. The bound is also what FOUND the last
  lever — 8T sat at 82 %, which sent the lane back to the profile and located the serial
  `mean_offset` pass (7.1 ms "unaccounted" that survived scratch reuse); the profile now accounts
  for **100.0 %** of the walk.
- **Two measured negatives, kept as tests not ships**: lowering the conversion chunk height MOVES
  BYTES (one ULP at 97×51 ⇒ chunk height is SEMANTICS, now pinned two-sidedly); doubling
  `scale_capacity_rows` buys ~2 ms for ~20 MB — rejected.
- **Ref-cache gap CLOSED** (the fold-engine lane's open item): `compute_with_ref_into` did not
  route to the fold at all, and the fold allocated a fresh `V2Scratch` per compare — both fixed
  with a new PRIVATE field on the existing public `ZensimScratch`, so **no public type or method
  was added**.
- Gates 385/385, clippy clean, `--no-default-features` builds;
  `both_engines_are_bit_identical_across_rayon_pool_sizes` WIDENED from 4 shapes to all 18
  geometries + 4 large × pools 1/2/3/8/16.
- **Weigh**: RSS — `ADVANCE_ROWS` 256 + band-local H planes cost footprint; at 2304² the fold is
  still 0.75×/0.85× buffered, but at 1152² it went 1.13× → **1.32–1.38×** (§5.2). No era-2 byte
  flip landed during the lane, so its bit-identity is relative to current main throughout.

## 2026-08-31 ~09:0xZ — ROUND 22: THE FEATURE/MODEL COST FRONTIER (user: "what features should we drop" + "drop linear/blend/MLPs if it buys 2×"; doc `benchmarks/feature_cost_frontier_2026-08-31.md` §0.1, head `f035aede`)

**Slot counts are a LIE about cost.** B reads 95/372 but you cannot save 74 % — the families share
passes, read from source: **peaks (f156..228) are FREE** (`acc.ssim_d8 +=` at 10 sites,
`acc.edge_art_max =` at 20, NONE behind a predicate; `V1PoolsMode::Off` was already computing them
and merely declining to emit), and **masked + IW are ONE pass group** (one activity chain, one
`store_mu`/`store_sigma`, three `*_inline_both` kernels doing both strengths in a single sweep —
dropping one saves ~nothing, dropping both removes the arm). So f0..372 has **exactly one compute
boundary: peaks vs masked-and-IW**, worth **+2.4 / +9.8 / +44.2 ms** at 576²/1152²/2304² (1T) =
+33–36 % of the peaks-only walk.

**Rank cost of that boundary** (exact rank-|K| ablation via `bake_contrib`; baseline parity vs
`score_row` max|diff| **0.000e0** over 47,511 rows): shipped **B** CID22 **−0.399** / KonJND
**−0.525** (it needs them); **W-LIN 7b** −0.005 / −0.048; **944 MLPs exactly 0** (layer 0 is
exact-zero there). W-LIN also: whole v1-372 = CID22 −0.027 but LIVE **+0.117**; v2-348 = CID22
**−0.745**.

**⇒ The model CLASS is the lever, not the family.** At 2304² the basic-only walk is
**2.65×/3.46×/3.57×** (1/8/16T) the W-LIN blend, **2.26×/2.95×/3.54×** the 944 MLP, and
1.60×/0.98×/1.12× today's buffered path (the win over TODAY's default is low-thread only).
**Recommended drop set: NOTHING inside shipped B** — every family it computes, it uses. The real
recommendation is to **evaluate the basic-only class (`ADD156`) seriously**: within **0.019** pooled
CID22 of B, and on WITHIN-IMAGE ranking (what a dial actually consumes) it matches or beats B on
**7 of 8 corpora**, incl. HF-NL/ref **0.799 vs 0.765**. Only the 944 MLP is at-or-above the board's
`peer_ssim2` row on every human corpus; ADD156 is the closest tie. **Cost to realize: a profile
slot and a ship call — no retrain, no era.**

**Shipped**: `V1PoolsMode::Peaks` + `BandPoolWork{HOnly,Carriers,Full}` (the compute boundary made
expressible, band-local self-blur kept); per-profile weight-skipping
(`fold_engine::{V1PoolNeed, bake_pool_need, cached_bake_pool_need, caller_col_spans,
score_pool_mode}` + `Zensim::with_unread_feature_skipping`, **opt-in, default off**);
`caller_col_spans` handles PRUNED bakes — without it the policy would never have fired on any
packed bake (pruning has been on by default since 2026-08-04), including the two 944 MLPs where it
is exact. Gates: `folded_peaks_mode_is_pure_compute_skipping` (19 geom × {v1_only,944} ×
{serial,rayon}, bit-exact), raw-distance bit-identity on both the plain and ref-cached entries,
`unread_feature_skipping_is_inert_…` (23 geom × pools 1/2/3/8/16 × both engines) + 5 policy tests;
zensim lib **240/0**, zensim-validate 160+40, clippy clean. **Working set: `Peaks` halves the plane
count (10→4 per slot)** — per-band-task 4.43→2.21 MiB, per-process 11.1→4.4 MiB at 2304², and the
footprint lane's saturation test shows that term governs the thread ceiling (**3.38 → 5.85×**).

**New public items for approval** (all `feature-regime-v2`-gated, methods `#[doc(hidden)]`):
`V1PoolsMode::Peaks` (variant on a `pub` non-`#[non_exhaustive]` enum),
`Zensim::with_unread_feature_skipping`, `Zensim::score_pool_mode`.

**Open**: the `fold_v1` lever (`feature_v2.rs:7395`, hardcoded `true`; the walk already branches on
it in four places) is the biggest remaining block for the W-LIN class but is NOT byte-neutral
(turns f0..372 into zeros in a 944 vector) ⇒ an ERA decision, priced not pulled; the 944-MLP
v2/append ablation was not run (~2 TFLOP, would have contended with the wall-clock measurement);
HDR untouched (the fold falls back to buffered for declared-HDR) though the free part is measured —
`BHdr` reads 28 masked / 17 IW, `c_hdr_l1t1944` reads 0/72, so the SDR split repeats exactly.
⚠ Process note: **`cargo … | grep -E '^error'` NEVER matches** — cargo's ANSI colorizing puts an
escape before `error`; it hid a bench compile failure for ~30 min and made a first clippy "clean"
false.

## 2026-08-31 ~09:3xZ — ROUND 23: THE FOLD FOOTPRINT, DECOMPOSED — and the fold is now FASTER than buffered serially (12 commits; doc `benchmarks/fold_footprint_2026-08-31.md`)

**Why the fold was ever heavier: buffered sizes its band scratch by WORKER COUNT, the fold sized
its by FAN-OUT SHAPE.** Buffered's `ScaleBuffers` comes from a rayon `map_init` — one buffer per
worker, so ONE at one thread; the fold pre-allocated a `FoldPoolScratch` per (channel × band slot)
= **12 band buffers regardless of the pool**, plus three full 14-plane `ScratchV2Strip` sets of
which a `v1_only` score writes **two planes**. At 1 thread: 12 band buffers against buffered's 1,
on identical work.
**Two inherited premises FALSIFIED**: there is **no 1+1/4+1/16 pyramid series** (`downscale_2x_inplace`
writes over the plane prefix, `Vec::truncate` keeps capacity — the term is exactly `24·W·H`,
confirmed 31.85 MB/18 calls), and widths are **not** SIMD-padded (`pyramid_plane_stride(w)==w`).
Closed-form model committed; every term matches heaptrack to **<0.1 %**; per-cell RSS error over 42
cells buffered −4.4…+9.4 %, fold +3.8…−9.0 %, with per-worker glibc arena churn isolated rather
than fitted.
**Pool-block answer (the user's sub-question): 33–34 % of the fold's working set at 1T, 39–43 % at
16T** — the largest single term, width-scaled and height-independent, which is exactly why the fold
lost below ~3 MP. After the fix, turning pools ON at 1T makes the walk **smaller** (self-blur's
1-slot pool is cheaper than the phase-A planes it replaces).
**Shipped (4 fixes, zero bytes moved)**: `StripPlaneNeeds` (strip scratch sized to the plane groups
actually written: 2 of 14), `band_slots_for` (`min(bands, threads)` slots + chunked fan-out
preserving the band-order f64 merge), `advance_rows_for` (`ADVANCE_ROWS` becomes a CEILING; `32·T`
on the 64-row lattice), `FoldPoolScratch::ensure` sized to the max band. 366/0.
**Results — working set** 1152² **53,880 → 23,188 KiB** (1T), 64,668 → 55,208 (16T); 2304² 99,520 →
**43,832** / 116,724 → 104,748; buffered control moved ≤1.5 %. **Crossover 1T ~3.2 → ~0.5 MP**, 8T
→ ~2.35, 16T ~2.2 → **~1.4 MP**. **Speed 1T: `score_fold` −26.5 % (1152²) / −13.1 % (2304²) — now
FASTER than buffered, 0.78×/0.87×** (was 1.03×); 16T −12.4 % / −7.2 %, ratio 1.593→1.366 and
1.656→1.589.
**★ The machine is NOT a 7950X — it is a Ryzen 9 9950X3D with ASYMMETRIC L3**: CCD0 (cpus
0-7,16-23) **96 MiB**, CCD1 (8-15,24-31) **32 MiB**; `getconf LEVEL3_CACHE_SIZE` reports 32 MiB and
is wrong for half the box. Per-thread budget is set by CCD1: 8 MiB (8T) / 4 (16T) / 2 (32T); fold
band task = `2,016·W` vs buffered `1,512·W`, so at 2304²×8T that is 35.4 vs 26.6 MiB against
32 MiB — **the threshold falls between them.** Per-CCD N-process saturation, TESTED: pre-fix the
fold was CCD-insensitive (3.38×/3.33×, reproducing the predecessor's "3.5× machine bound"),
post-fix it is CCD-SENSITIVE (**5.85× / 4.54×**) ⇒ **the predecessor's "machine's own bound" was
the fold's OWN footprint**; throughput +80 % on the 96 MiB CCD. Ablation attributes the whole gain
to `band_slots_for`.
**Column tiling: PRICED, not implemented** — halo derived from the kernel chain (two chained H
passes ⇒ 10 columns/side, buffer `Tw+20`, redundant work 15.6/7.8/3.9/1.0 % at Tw=128/256/512/2048);
any `Tw ≤ 2048` holds the per-thread hot set under CCD1's 16T budget at every width; it reorders
the f64 pooled accumulation ⇒ moves bytes today, so it is recorded as an **era-2-enabled design**
(fixed virtual-lane grouping makes it byte-safe by construction) with predicted gain 2304²/16T
**4.43 → 1.02 MiB/thread at Tw=512**, CCD1 occupancy 35.4 → 8.2 MiB.
**Open**: the workspace Environment doc names the wrong CPU (user-gated edit, NOT made);
`Zensim::compute` builds a fresh `V2Scratch` per call so a hot loop churns per-worker arenas
(+19 % RSS over 20 compares at 16T — `compute_with_ref_into` already avoids it; fixing plain
`compute` is an API/ownership decision); cross-channel band-slot pooling priced (−7.4 MB at
1152²/8T, nothing at 16T), not built; 512²/8T still 1.84× (four band slots are genuine there and a
180-row rolling floor is 85 % of a 512-row image).

## 2026-08-31 ~10:3xZ — ROUND 24: era-2 stage A — the lane's OWN hypothesis falsified, and the 2.12× isolated (`e1a0f724`)

- **Design absorbed (§18)**: era-2 is now the umbrella for all byte-changing speed work, ordered by
  gain-per-risk — A pass split (lowest risk), B column tiling (biggest gain), C `fold_v1` flag.
  Two constraints fell out of composing tiling with the lanes/bands, and they are what make tiling
  SAFE rather than another grouping hazard: **`TILE_WIDTH` must be a multiple of 8** (a pixel's lane
  is `x mod 8` on the GLOBAL x; for tile `t` at offset `k`, `x mod 8 = k mod 8` iff `Tw ≡ 0 mod 8`,
  so lane assignment stays tile-invariant and the identity proof carries over untouched — without
  it tiling would silently permute which terms share a lane), and the merge is a fixed function of
  **(tile, band), tile-major/band-minor**, which is also the natural loop order ⇒ merge order and
  loop order coincide and **no per-tile partial storage is needed**. `TILE_WIDTH` joins
  `ERA2_BAND_ROWS` as semantics-not-a-knob.
- **Stage A falsified its own §17.2 hypothesis.** Re-baselined on current main first (the other
  lanes moved these kernels; era-1 106.8/231.8 µs vs era-2 two-pass 226.1/454.9 = 2.12×/1.96×,
  ≈ §17 ⇒ their work did not change the era relationship). Then the proposed single-pass-on-v4x
  fusion, implemented behind `const FUSED` and measured: **445.5 / 911.2 µs — 2× WORSE than the
  two-pass, 3.98× of era-1**. Arithmetic says why: 29 `Lanes8` accumulators = **232 live f32
  values** against 32 registers — the same wall that capped `POOL_SIMD` at 16 originally. **The
  two-pass split is not a 16-register concession; it is the right structure on EVERY tier**
  (the user's "multiple passes per row can beat fusion when it spills" steer, confirmed in code).
  Byte-neutrality of pass structure is now proven ON the configuration that turned out slower:
  `era2_fused_and_two_pass_are_bit_identical`, all 35 slots `to_bits()`, 6 geometries × 3 channels.
- **The remaining 2.12× is attributed**: era-1 accumulates in `V8<T>` magetypes SIMD types, era-2 in
  plain `[f32; 8]` arrays — with fusion ruled out and the `target_feature` region already in place,
  the gap is the accumulator REPRESENTATION, not pass structure or ISA. Next lever specified:
  rewrite the body against `V8<T>` keeping every semantic (V8 IS 8 lanes; explicit `to_array` +
  `era2_reduce8`, never `reduce_add`; tail folded; band-order merge). **Tiling is deliberately
  queued BEHIND this** — its win is a cache-footprint effect a 2× compute deficit would mask.
  The flip stays blocked.
- ⚠ For other lanes: `cargo clippy --tests --benches -D warnings` **fails on main** with 14
  warnings, **none in era-2 code** — `feature_v2_stream.rs` (83, 774, 1073), `fold_timing.rs` (45,
  63, 65, 154, 182), `feature_v2.rs` (639, 2078). Left for their owners rather than a conflicting
  drive-by.

## 2026-08-31 ~11:3xZ — ROUND 25: era-2 stage B — PARITY REACHED, and a 55× lesson about closures

| geometry | era-1 | era-2 (V8, macros) | ratio |
|---|---:|---:|---:|
| 576×128 | 98.4 µs | **101.8 µs** | +4.0…+5.8 % |
| 1152×128 | 202.7 µs | **210.1 µs** | −7.1…+16.2 % (CI spans zero) |

**The 2.12× is closed** — but the route matters more than the number:

| step | 576×128 | vs era-1 |
|---|---:|---:|
| runtime chunk bound | 1007.5 µs | 10.3× |
| `as_chunks::<8>` | 482.1 µs | 4.4× |
| + `target_feature` | 226.1 µs | 2.12× |
| **V8 via closures** | **5610.1 µs** | **36×** |
| **V8 via `macro_rules!`** | **101.8 µs** | **1.04×** |

**The V8 rewrite made things 17× WORSE before better, and era-1's own kernel comment had already
documented the mechanism** — a prior 5.3× regression when "the body passed LLVM's inline-cost
threshold, the hint stopped being honored and every V8 operator compiled into a CALL… outside the
feature region". The lane's two closures hit exactly that, harder (bigger body). Converting them to
`macro_rules!` — textual expansion, so there is no inline-cost decision to lose — was a **55× step**.
**`#[inline(always)]` on the enclosing fn is NOT sufficient when the hot work sits in closures it
contains.** Fifth catch of the campaign, and the first the code's own comments had already warned of.
- **All three instruments re-verified on the V8 kernel**: oracle unchanged except `hf` IMPROVED
  (3.77e-5 → 3.44e-5 — `bounded_excess_pair_v` uses a true division where the lane's scalar helper
  used reciprocal-multiply, so era-2's term math is now literally era-1's `_v` helpers); **vendor
  probe era-1 66/105 differ, era-2 0/105** ⇒ cross-tier identity SURVIVES per-tier V8 compilation
  (the load-bearing question); all bit-identity gates green. Trap recorded: the row tail must be
  **MASKED, not zero-padded** — padded lanes are not accumulator-neutral for the pool WEIGHTS
  (`saturate(0)` ⇒ `mask_w = 1.0`), and since the core families ARE zero on padded lanes only the
  pool denominators would have exposed it. Superseded scalar `e2_*` helpers + `E2Pools` deleted.
- **Widened charter absorbed, with the threshold registered BEFORE any candidate exists**: the bar
  moves from bit-identity to proven UTILITY PRESERVATION (bit-identity remains the standard WITHIN
  the era). **PASS iff no corpus loses more than 0.005 SROCC and the composite does not fall** —
  ~200× the option-C precedent (+0.000024 cid22) and an order below the 0.5-point materiality step;
  a failing redefinition is reverted, not renegotiated. Drops ship as **structural zeros, never
  renumbering** (the f156-371 precedent) with a registered reason. HDR gets a **compute-set
  descriptor that REPLACES `V1PoolsMode` + `v1_only`** rather than adding a third ad-hoc instance;
  stated explicitly: **the fold still falls back to buffered for declared-HDR and era-2 does not
  change that** (fold-native HDR is the fold-engine lane's decision).
- Two places the evidence cuts against the ask, recorded as such: stage B is evidence **FOR** the
  magetypes rule (generic `V8<T>` beat plain arrays), so raw intrinsics are reserved for measured
  residues; and const generics must report **code-size + compile-time cost** beside the speed win,
  because this lane just measured one "obviously faster" structure at 36× slower.
- Re-ordered by risk: **C (tiling) and D (compute-set descriptor) next** — pure skipping/reordering,
  no utility question; **E (drop set) and F (redefinitions) last**, as the only items where faster
  can mean worse — and they come to the USER as a measured table, not a lane decision. Flip still
  blocked pending rank preservation, the v2-block hand-off, blast-radius registration and the
  gate-re-pin enumeration.

## 2026-08-31 ~12:1xZ — ROUND 26: what is costly inside v2-348+append — IT IS THE PLANE PIPELINE, NOT THE FEATURE MATH (user question; `40fdcd86`, doc `benchmarks/v2_block_cost_2026-08-31.md`)

**★ Methodological correction first: `dense_block_kernel`'s `POOL_SIMD` path is v4x-ONLY, and
valgrind masks AVX-512 out of CPUID — so callgrind physically CANNOT profile the path that ships.**
Every prior Ir number for this block is the v3 scalar-pool form. The lane did the wall-clock
re-profile (7 new `fold_timing` phases); attribution additive to ≤1.2 %, reproduced twice ~20 min
apart at ≤1 % 1T agreement. **`dense` is 13.5 % of the block and 7.3 % of the walk on the shipping
tier — NOT the 22–26 % this ledger has been repeating** (at v3 it is 47.9 % of the block's Ir;
`POOL_SIMD` is the entire gap). Everyone citing the old profile should stop.

**Ranked cost, 2304²/1T (the +200.1 ms block):**

| item | ms | % | ns/px 576→2304 |
|---|---:|---:|---|
| **H-plane shape** (strip-wide `blur_h` − the fold's self-blur saving) | 65.83 | **32.9** | 1.34→4.94 (**3.7×**) |
| **`planesA`** (4× V-blur + activity chain) | 49.25 | **24.6** | 1.11→2.01 (1.8×) |
| `dense_block_kernel` | 27.10 | 13.5 | 1.21→1.28 (1.06×) |
| `append_block_kernel` | 21.32 | 10.7 | 0.94→1.01 |
| `gradient_block_kernel` | 16.89 | 8.4 | 0.88→0.80 |
| `planesApp` (`bs2` chain) | 13.00 | 6.5 | 0.43→0.53 |
| `blockiness` | 4.46 | 2.2 | 0.17→0.18 |

**Every v2 feature kernel is FLAT in ns/px across a 16× pixel range; every PLANE pass is not.** The
block's composition inverts with size — kernels are 70.6 % of it at 576², 34.8 % at 2304².
`planesA` runs at **29.8 GB/s = the single-thread DRAM ceiling** and is ~75 % of the block's 1T→8T
CPU-time growth; `blur_h` at 4.88 GB/s is NOT bandwidth-bound — its 16-row × 6-plane transpose set
hits the 1 MiB L2 at width ≳ 2304.
**Structural read:** all four kernels read the SAME six strip-wide planes, and producing them costs
**1.84×** what every formula on them costs (128.08 vs 69.77 ms). Dense's 11 weighted-pool slots are
**byproducts** (extra per-pixel arithmetic, no extra pass — the v2 analogue of v1's free peaks);
gradient, append and blockiness each force their own sweep; append also OWNS a plane (`bs2`, not
derivable from `ssq`, which is the SUM `blur(src²+dst²)`); BANDVIS rides gradient's sweep but its
instantiation costs 1.56×. **The redundancy hunt came back genuinely empty**: v1's activity is
`|src − blur_h(src)|`, v2's is `|src − mu1|` — different quantities.
**Shipped (byte-neutral):** the decomposition instrument itself, and
`folded944_is_bit_identical_across_rayon_pool_sizes` — a REAL gap, since the existing pool sweep
covers the scoring path, which runs `v1_only` and touches not one v2 kernel; the new gate is 944
slots `to_bits()` × 22 geometries × pools {1,2,3,8,16}, and it is load-bearing right now because
era-2 and tiling are about to re-schedule exactly the fan-out whose strip grouping moves f372..943.
369/0, clippy rc=0.
**Falsified, four levers, none shipped:** rayon band split at 1T (serial is FASTER, 114.5 vs
121.5 ms); smaller `STRIP_ROWS` (128→378.0, 64→421.8, 32→418.5 ms — the halo tax beats the cache
win, so **the fix cannot come from strip height**); row-major V blur (proven BIT-IDENTICAL over 21
geometries × 3 radii, then **9 % slower** everywhere — the column-major form keeps its accumulator
in a register across 148 rows; implementation and gate REVERTED, not parked); bounds checks (zero
`panic_bounds_check` in any v2 kernel or blur, 61 symbols re-verified post-rebase).
**⇒ Retarget handed to era-2, measured:** phase A's strip-wide H blur and the fold's band-local
self-blur run the SAME kernel on the SAME data, and band-local is **1.49× cheaper end-to-end on
13 % MORE blur work** while callgrind independently shows the strip form executes 9.1 % FEWER
instructions — two instruments, opposite signs ⇒ it is memory. Band-local phase A is worth
**−65.8 ms**: block 1.49×, walk 367.7 → ~302 ms (**1.22×**), *serial*, so it composes with the
thread work; ~275 ms (1.34×) if `planesA` gains the same once cache-resident (labelled a
projection). The shape is a **rolling row window**, not smaller strips.
Honesty note from the lane: its era-2 dense calibration (a 2× regression) **went stale the same
day** — stage B landed parity while it measured; §7.3 carries both and says which is live.

## 2026-08-31 ~13:2xZ — ROUND 27: era-2 — band-local phase A FALSIFIED, COLUMN TILING is the win, and "tile means PACK" (6 commits → `d56c670a`)

- **Band-local phase A: FALSIFIED.** Round 26's hand-off (−65.8 ms, 1.22×) is not there. Built at
  three band heights with a sound estimator (one binary, runtime arms, byte-identical env blocks,
  interleaved, min of 7 walks × min over **15 ASLR layouts**): `B=32` +13.1 %, `B=64` +3.0 %,
  `B=128` (the bit-identical control, 956/956) 0. **Why the proxy lied: it compared unlike
  closures.** The fold's self-blur needs ±`V1_BAND_OVERLAP` = 5 rows (1.31× halo); **phase A ends
  at `activity = blur(|src − blur(src)|)` — TWO CHAINED BLURS, ±2·BLUR_RADIUS = 20 rows out of 32 =
  62 % redundancy**, and it applies to the V blurs and `bs2` too. Band-local H *is* 1.25× more
  efficient per unit work; the tax is simply bigger. The rolling-row-window variant needs the
  row-major V blur round 26 already measured at +9 % and reverted. Reverted, not parked.
- **★ COLUMN TILING IS THE WIN — on the axis where the halo is 0.6 %.** `blur_h`
  **1532.7 → 179.6 ms (8.5×)** at 4608²; whole 944 walk **1.15× @5 MP, 1.73× @21 MP** at 1T,
  1.23× @8T/4608², 1.11× @16T. Untiled ms/MP climbs 22.2 → 108.9; tiled 22.2 → 61.3 — **tiling
  removes most of the walk's superlinear term, so the win GROWS with size.** Behind
  `ZENSIM_H_TILE`, default off pending the era-2 re-pin. Found by a zero-code probe: at a FIXED
  5.31 MP, `blur_h` costs **104.99 ms at width 2304 vs 34.58 at width 1152** — the cost is a
  function of WIDTH, not pixels.
- **The transferable result: TILE MEANS PACK.** The lane then built the "clean" version anyone
  would reach for next — `x0/x1` threaded through all 16 H-blur bodies (38 initialisers, 38 loops,
  48 ring warm-ups), byte-neutral, suite green — and in the same binary it **bought NOTHING**:
  1.06× @1T/2304², **0.96× @1T/4608²**, against packed's 1.26× / 1.71×. Restricting `x` does not
  change which cache lines are walked — the planes are still full-width. **The copies are not
  overhead; they ARE the optimisation.** All 16 bodies deleted rather than parked. Same pass: an
  activity-chain tiling committed mid-session as "+1.5 %" is a **wash** when isolated properly
  (A-only 0.975–1.034× over four cells) — removed, claim superseded.
- **⚠ MEASUREMENT HAZARD for every lane**: 944@2304² is **bimodal over 10.1 % in the ASLR base
  alone** (±0.13 % under `setarch -R`); THP, heap-base shift, plane stagger and CCD placement were
  all ruled out BY MEASUREMENT. **The environment block size is itself a layout input** — adding a
  provably-dead env var flipped one build 359 → 328 ms. This invalidated three of the lane's own
  earlier sweeps of the same experiment. Protocol + the identical-code-path control (tile > width
  cells must read 1.000×, and instead give the noise floor: **±0.3 % @1T, up to 6.5 % @8T**) are
  now in `zensim/CLAUDE.md`.
- **Next, specified**: post-tiling at 4608² the leaders are `fold` 373.6 ms (28.3 %) and `planesA`
  222.7 (19.1 %) — same width disease, neither reachable by another copying tile (they are
  one-plane-in/one-plane-out; the H blur paid because it is six plane-touches behind two copies).
  The shape that gets both is the **packed column slab**: copy `src`/`dst`/`refy` into slab-width
  buffers once per slab, run phase A AND phase B at slab width, never copy out (kernels accumulate;
  the planes never need to be full-width). Needs an x-offset for `blockiness_sparse_strip_wide` and
  a per-slab X/B activity stash — no kernel signature changes. The compute-set descriptor (item D)
  is not started. 369/0, clippy clean.

## 2026-08-31 ~14:1xZ — ROUND 28: era-2 — the slab's PREMISE falsified before building it; the pool pass is now the largest item (`09464b84`, `64d89213`, `31b76af6`)

- **The lane measured its own hand-off premise and killed it.** "fold and planesA are the two
  remaining WIDTH-DISEASED leaders" was its own claim from round 27, never measured. The same
  fixed-5.31 MP width probe prices every phase for free:

  | phase | w2304 | w1152 | w576 | width-driven headroom |
  |---|---:|---:|---:|---|
  | `blur_h` | 131.96 | 35.34 | 31.75 | −100.2 ms — already taken |
  | **`fold`** | **79.86** | **77.33** | **75.01** | **−4.85 ms (6.1 %)** |
  | `planesA` | 39.31 | 27.37 | 26.08 | −13.23 ms |
  | `dense` | 26.76 | 27.06 | 27.40 | none — *worse* narrow |
  | `gradient` | 18.41 | 19.42 | 21.22 | none — **15 % worse at w576** |

  **The fold is not width-diseased — it is 28 % of the walk because it is 28 % of the WORK.** So the
  slab's whole prize is planesA+planesApp ≈ **16.4 ms of a 302 ms tiled walk (5.4 %)**, against a
  3-plane copy-in AND a measured penalty on the pointwise kernels (they get SLOWER narrow: 66 ms of
  kernels, −3.3 ms at 5 % eats a fifth of the prize before copies). "No kernel signature changes"
  was also wrong: a column slab's halo is **interleaved in every row** and cannot be sliced off the
  way a row band's can, so `stride`/`width` must thread through six phase-B kernels plus the fold.
  **Not built.**
- **★ Where the fold's mass actually is**: `944full` fold = 78.79 ms vs `924` (pools **Off**) =
  37.59 ⇒ **the masked/IW/soft-peak POOL PASS is 41.2 ms = 52 % of the fold and 13.6 % of the whole
  tiled walk** — the single largest remaining item, it is feature ARITHMETIC, and no tiling or
  slabbing touches it. **That is the item-E cost column: the 11 pool pairs cost 13.6 % of every
  compare, 2.5× larger than either layout lever left.**
- **Next layout build, priced with its obstacle**: reading `fused_vblur_features_ssim` REVERSED the
  lane's own §24.2 proposal — it does not run a V-blur pass; it keeps the four recurrences in
  registers and already stores all four into `FoldPoolScratch` under `store_sigma`, so **the
  duplicate to delete is phase A's**. The four `box_blur_v_from_copy` sweeps are **22.05 ms (55 % of
  planesA)**; the three the fold already has are **16.69 ms = 5.5 %** (mu1's must stay — the
  activity chain needs it on halo rows the band-scoped fold never produces). Obstacle worked out
  BEFORE building: the fold's band buffers **overlap by 10 rows**, exactly the rows each band's V
  blur mirrors at its buffer edge, so whole-buffer writes corrupt a neighbour in either band order
  (same corruption as §22.1, different costume). Two costed exits: copy the inner rows out (no
  kernel change, ≈ HALF the 16.69 ms) or an inner-only store offset (full 16.69, six tier bodies) —
  and price them rather than assume, since that assumption was falsified once already today.
- **`ZENSIM_H_TILE` default-on: RECOMMENDED unconditionally as part of era-2.** It **costs nothing
  below the tile width structurally** (`width <= tile` is a no-op, so the 576²/1152² cells are not
  small regressions — they ARE the noise floor and they bound every other cell in their row); gains
  1.151×/1.733× at 1T from 2304² up and 1.234×/1.109× at 4608² on 8/16T. Cannot flip standalone
  (byte change). **`TILE_WIDTH` should be DERIVED, not frozen at 1536**: the live window is
  `H_BLUR_ROW_GROUP × 6 × tile × 4 B` ⇒ ≤1365 on this part's 1 MiB L2, so **1024** leaves headroom
  on 512 KiB-L2 parts and sits inside the measured-flat 512–1536 band.
- **Cross-lane hand-off done** (the blur/radius/branch lane's doc is not on main yet, so the numbers
  live in era-2's): the B=32/64/128 band-local results with the bit-identical control; **a radius
  cut does NOT rescue row banding** — phase A's closure is ±2R, so a 32-row band is 1.625× at R=5
  and still **1.375× at R=3**, viable only near R=2, while the COLUMN closure is ±R and tiling is
  essentially radius-insensitive (0.65 % → 0.26 %); tile-edge branch behaviour is measurable today
  on main via `ZENSIM_H_TILE=0000` vs `0512`/`1536` (3× the edge count, same binary); plus the ASLR
  protocol. Item D (compute-set descriptor) not started.

## 2026-08-31 ~14:5xZ — ROUND 29: radius, locality, branches — the three user questions ANSWERED (`c2df4be7`; doc `benchmarks/blur_radius_locality_branches_2026-08-31.md`, library byte-for-byte UNCHANGED)

**(c) Trailing pixels / branch mispredictions: NO — measured for the first time, hypothesis
falsified.** Enabler worth carrying: `perf_event_paranoid` was 4 (all events blocked) and the
`perf` first on PATH is a stale binary; with the sysctl at 1 and `/usr/bin/perf`, **hardware
counters work on the shipping `v4x` tier — which callgrind structurally cannot execute.**
1T misprediction rate **0.015–0.050 %**; the ENTIRE budget, charging every miss 20 cycles with no
overlap credit, is **0.14–0.50 % of cycles** (IPC 2.5–3.9). **Row tails falsified**: the worst tail
class (2303 ≡ 7 mod 8) costs **+0.06 percentage points of cycles** over 2304 and actually LOWERS
the V blur's share of misses (37.5 → 34.8 %); disassembly shows the reflect-mirror index math is
already `cmovae`/`cmovb`/`cmovs` — there is no branchy edge handling left to remove. Blur edge
clamping is the top source (37.5 % of misses) at **0.18 % of cycles**; band/strip boundaries 9.0 %
of misses = 0.017 %. The only regime near 1 % is 576²/16T (1.2–1.65 %), where **48.6 % of misses
are `crossbeam_epoch`** — the rayon runtime, not our kernels. Four candidate fixes retired, nothing
shipped.
**(a) Radius: real but small, and the one setting that passes quality is FREE.** `HALO_P = 2R`, and
the blur is a running sum ⇒ **O(1)/px at any radius** — radius buys only halo, prologue and working
set. A single-build A/B was inadmissible: a control (second R=5 build, identical semantics)
measured the **cross-build layout floor at 4.67 %**, so every radius got TWO independent builds,
n=30 draws/arm.

| R | 2304²/1T | 2304²/16T | peak RSS | C944 worst corpus | composite | BAR |
|---|---|---|---|---|---|---|
| 4 | +0.68 % | −0.17 % | −1.35 % | −0.0007 | +0.0038 | **PASS** |
| 3 | −5.53 % | −4.42 % | −2.90 % | −0.0059 | +0.0070 | FAIL |
| 2 | −4.71 % | −7.14 % | −4.12 % | −0.0221 | +0.0090 | FAIL |

**R=4 passes era-2's registered bar for the shipped 944 flagship and costs nothing measurable in
time**; R=3/2 buy 4.4–7.1 % at 16T and fail. Mechanism confirmed: every feature kernel and the
producer are radius-invariant to ≤0.3 % while `planesA` −23.7 %, `planesApp` −14.7 %, `blur_h`
−13.7 %. Quality is a **redistribution, not a degradation** — gains on cid22/aic3/aic4 and hugely
on **KonJND (Profile C 0.5006 → 0.5896 at R=2)**, losses on TID/KADID; all models were trained at
R=5, so a **radius-4 retrain is registered**, not launched. Validation bonus: the lane's radius-5
re-extraction is **bit-identical to the canonical root — 19,367,104 / 19,367,104 cells, max diff 0**
across 9 legs, which incidentally byte-neutrality-checks the **155 commits** since that root's
`build_commit`. Suite 369/0 at R=5 and 362/7 at R=3 with **6 of 7 failures being stored radius-5
expectations**; all 13 fold-engine parity tests pass at R=3. No test relaxed.
**(b) Locality: the RADIUS is what unlocks the axis — the sign flips.** At R=5 a 32-row strip is
**+12.0 %** (reproducing the v2-block L2 falsification); at **R=2 it is −4.7 %**. Best cell
`R=2 × STRIP_ROWS 32`: **301.8 ms vs 326.5 (−7.6 %) at 61.0 MB vs 97.6 (−37.6 %)** — the locality
prize, reached through the radius (not shippable: R=2 fails quality, `STRIP_ROWS` is not
byte-neutral). This **reconciles the one-step arithmetic difference with round 28**: the 1.25 %
per-unit gain must be compared against the halo RELATIVE TO THE SHAPE IT REPLACES — 1.406/1.257/
1.176× at R=5/3/2 — so break-even sits between R=3 (a wash: round 28's read, correct) and R=2, and
round 28's own figure then predicts +12.5 % at R=5 (measured +12.0/+13.1) and −5.9 % at R=2
(measured −4.7): **three numbers, two lanes, two knobs, inside 1.1 points.** **Column tiling is
confirmed radius-INSENSITIVE** (1.229/1.189/1.203× at 2304², 1.837/1.874× at 4608², tile width
128→2048 within 1.4 %) ⇒ radius drops out of era-2's `TILE_WIDTH` grid, one fewer dimension — and
the levers **COMPOSE**: 335.2 → 272.8 (tile) → 255.8 (both) = **1.311×**, 98 % of the product;
1.968× at 4608².
Open: the bar's dial clause (grids are radius-blind — verified byte-identical across all four
roots), 3 of 12 corpora, the band shape at small radius (needs era-2 to rebuild what it reverted),
threaded shape knobs, no PEBS-class sampling on this host.

## 2026-08-31 ~15:5xZ — ROUND 30: era-2 — a real DEFECT found while enumerating re-pins; the E+F table exists; item D landed (`512c66c2`, `08c1ac07`, `ab49d4b7`)

- **★ Defect found by forcing the tile on for the whole suite**: tiling only SELECTED call sites
  **split the v1 reference path from the fold** — 4 cross-path `to_bits()` gates failed at ~1.3e-9.
  The tile now lives on **all four H entries** (`fused_blur_h_ssim`, `_ssim3`, `box_blur_h`,
  `box_blur_h_into_abs_diff`, `fused_blur_h_mu`) with private `*_untiled` forms. **The rule: either
  EVERY H entry tiles or none does — this crate's strongest gates are cross-path bit-identity, so a
  partial tile is a silent regime split.** Three ring-vs-regather tests were **re-pointed at the
  kernel, not relaxed**. Perf unaffected (1.183× / 1.673× / 1.299×).
- **Gate re-pin enumeration DONE for the tiling flip** (one of the three flip prerequisites):
  default (off) **370 passed / 0 failed**; tile forced on **365 / 5**. All five are ABSOLUTE-VALUE
  goldens (the four `v1_golden_bytes` fixtures, `f0` −5.149e-6; and `hardcoded_reference_scores`,
  1.8e-4 … 1.0e-2 score points). **Zero internal-consistency gates fail — that is what makes the
  flip safe in one step.** Radius 4 re-pins the SAME five and is additive; the V-redirect may
  re-pin none.
- **(a) Both V-redirect exits priced, and the lane's own "≈ half" estimate was WRONG**:

  | | 2304² | 4608² |
  |---|---:|---:|
  | baseline | 330.48 | 1939.21 |
  | **exit 1** (inner-row copy, no kernel change) | **318.51 (−3.6 %)** | **1840.03 (−5.1 %)** |
  | exit 2 (no copy, six tier bodies) | 315.50 (−4.5 %) | 1824.38 (−5.9 %) |

  `planesA` 39.27 → 22.42 and 248.50 → 139.21, independently reproducing round 28's −16.69 ms.
  **Exit 1 captures 80 %/86 % of exit 2 for zero kernel change** — the estimate had assumed the copy
  costs as much as the plane write it replaces, but its source is an L2-hot band buffer while the
  removed sweep was a COLD full-plane read plus the same write. **Decision: build exit 1**; exit 2's
  extra 3.0/15.7 ms registered with its price. Two facts recorded so they are not re-derived:
  round 28's overlap obstacle is **smaller than written** (the fused kernel writes only INNER rows,
  so bands are disjoint), and the real blocker is a **borrow**, not the algorithm.
- **(b) Item D landed**: `ComputeSet` (`pub(crate)`) replaces six ad-hoc locals and the walk reads
  it; behaviour is **gated, not asserted** — `compute_set_matches_legacy_derivation` sweeps **1,024
  combinations** against the legacy expressions verbatim plus two invariants the old code only
  implied. **No public API added**; §26.1 lists the proposed surface for approval and recommends the
  cheaper form (keep it `pub(crate)`, derive inside the existing entries from a model handle the
  caller already passes) — which unlocks item E with no new public types.
- **(c) The E+F decision table exists**: `benchmarks/era2_drop_redefine_table_2026-08-31.md` (linked
  from INDEX). Headline shape: **E1 is MODEL-CONDITIONAL, not global** — the pool pass is worth
  exactly 0 to the 944 MLPs and 0.399 CID22 to B, so its shipping form is "let the request say",
  which IS item D. **F1 (radius 4) is the one redefinition that has passed the registered bar on
  the shipped flagship.** Every number attributed to its lane; not-measured cells say so.
- Flip prerequisites remaining: **rank preservation across the roster** (needs model eval — not that
  lane) and **blast-radius / wave registration**.

## 2026-08-31 ~16:5xZ — ROUND 31: era-2 RANK PRESERVATION measured — the shippable subset is TILING; radius 4 is the whole decision (`34a30831`, `3e1bc89f`, `1115b3ef`; doc `benchmarks/era2_rank_preservation_2026-08-31.md`)

**★ Scope check first, and it re-prices the break**: at `9e52fb16` the break has **exactly ONE
merged byte-changing component — the column tile** (`ZENSIM_H_TILE`, default off, five H dispatch
sites). **The fixed 8-lane accumulation + `era2_reduce8` + `ERA2_BAND_ROWS` is IN TREE but NOT
WIRED** — every `dense_block_kernel_era2` call site sits inside `mod tests` or
`#[cfg(any(test, feature="oracle"))]`, while the four PRODUCTION dense sites call era-1
unconditionally with no flag. **So the largest semantic component of the break cannot clear the bar
until that kernel sits behind a runtime switch** (an env knob on those four sites would make it a
70 s re-run). Radius 4 is not in the break (main is R=5); the V-redirect exit 1 is priced, not
landed; item D is byte-neutral.
- **Tiling at the production width 1024: 5 PASS / 1 FAIL.** Eight of nine corpora are
  **byte-identical BY CONSTRUCTION** — every H entry guards `width > tile` and the corpora are
  narrow (four top out at 512 px, eight at ≤768; only AIC-3 has refs >1024) — so the panel sees the
  flip on ONE corpus and moves it by ≤2.0e-4. The single FAIL is `BHdr` on the bar's
  **zero-tolerance composite clause by 3.2e-6** (13 % of the +0.000024 non-event the bar itself
  cites); its worst corpus loss is 4e-5, **125× inside** the 0.005 clause. **Reported as FAIL — the
  threshold was not renegotiated.** Stress arms: tile 256 costs ≤1.8e-4 anywhere; **tile 32** gives
  the study's only real corpus-clause failure (BHdr/sdr25 −0.0162), mechanism measured — BHdr
  amplifies the perturbation **>30×** (max |Δscore| 1.82 vs C944's 0.054) and sdr25 is the smallest
  corpus.
- **Radius 4: 2 PASS / 4 FAIL** — reproduces the blur lane's four cells to the digit and adds a
  **new second passing model, `ADD156`** (worst −0.0027 aic3, composite +0.0049) beside C944;
  B / both W-LINs / BHdr fail.
- **★ ISOLATION: the components are separable and RADIUS DOMINATES.** Every model's worst-corpus
  delta is identical to five decimals between "radius 4, tile off" and "radius 4 + tile 1024"; the
  tile contributes ~1e-5 to a composite against radius's ~4e-3. **So whatever the break decides
  about radius 4 IS the break's verdict, and the shippable subset today is TILING ALONE at
  tile ≥ 1024.**
- **Dial clause satisfied by construction, and the radius lane's was CLOSED too**: grids are read
  from stored parquets so cross-arm identity proves nothing — the lane rebuilt them per arm. Every
  dial-grid reference is ≤1024 px and the corruption grid is one 576² image, so the tile-1024 twin
  is **byte-identical** (sha `1bed24cf…`, 4,547,248/4,547,248 cells). At tile 32 the grid moves and
  the panel was run: no G3 bound crossed. **Radius 4's clause 3 — registered by the blur lane and
  never run — is now measured: 52 % of dial and corruption cells move and NO model flips a gate.**
- **Controls**: the era-1 arm is sha256-identical leg-for-leg to the blur lane's R=5 root (hence
  19,367,104/19,367,104 cells to canonical r1b), which **incidentally byte-verifies `ab49d4b7`
  (the tile onto all H entries) as byte-neutral on 20,516 real pairs**; the radius-4 rebuild
  reproduces their R=4 byte-for-byte; revert verified two ways; row alignment exact across all 7
  arms and both blur-lane roots.
- **Unmeasured, with reasons**: era-2 accumulation (not wired), V-redirect (not landed),
  imazen26/nonphoto/hfnlproxy (bigcodec views, no local pairs TSV — absent, NOT counted as passes),
  BHdr's HDR panel (SDR corpora only), E1's rank cost (frontier lane's, attributed not re-derived).
  **All six bakes were trained at era-1, so every FAIL is an UPPER BOUND on cost, not an estimate**
  — the registered radius-4 retrain was deliberately not launched. Incidental: the stored canonical
  `dial_grid_944col_2026-08-01` is slightly stale vs HEAD (104,107 of 4.55M cells, max |Δ| 7.2e-9,
  inside golden policy).

## 2026-08-31 ~17:5xZ — ROUND 32: era-2 — accumulation WIRED + measured, V-redirect built-then-REVERTED, blast radius registered ⇒ ALL FLIP PREREQUISITES CLOSED (`c2282c8c`, `3d9ea485`, `068df97c`)

- **(1) The accumulation is measurable: `ZENSIM_ERA2_DENSE=1`.** The switch is on the **ENTRY**, for
  the reason round 30 established the hard way (a selected-call-site tile silently split the v1 path
  from the fold); era-1's body is now the explicit `dense_block_kernel_era1` arm for the four
  test/harness sites, so their comparison cannot be silently turned into an identity. **Live and
  confined: 714/956 slots bit-identical, 242 move, max 5.073e-6 at `f548` — and the movers are 241
  in `f372..719` + 1 in `f720+`, ZERO in `f0..371`.** **Blast radius zero: 370/0 at BOTH settings**
  (the five goldens tiling moves all pin v1's 372, which this does not touch) — the two components
  have very different profiles. **Perf: NEUTRAL** (1.026×/1.015×, corroborated by phase attribution;
  `v2:dense` marginally worse) — **its value is the fixed grouping (66/105 → 0/105 cross-vendor),
  not speed.** §28.5 records a caught mistake: the lane's first measurement said 1.296×, impossible
  since Amdahl caps a 7.9 % component at 1.086× even at zero cost; cause = arms run in sequential
  blocks, which its own §22.5 forbids. **Third unsound estimator this session, same tell every time:
  a result larger than the component it claims to come from.**
- **(2) V-redirect: built, enumerated, REVERTED.** Four couplings, each caught by an existing gate,
  each with a number: (i) **not byte-neutral**, 677/956 — the lane's own §24.5 call was wrong (the
  doc says bit-identical *from the same inputs*, and the fold V-blurs its BAND BUFFER while phase A
  V-blurs the WIDE WINDOW); (ii) `mu2` cannot be redirected where BANDVIS runs — it reads `mu2` over
  the wide window, and redirecting moved `f939` by **68 %**; (iii) it must fire in EVERY pool mode or
  `v1_pools` moves v2-era slot 372 (two independent gates); (iv) then it must fire in the PARALLEL
  arm too (failures 2 → 11, including `streamed_parallel_matches_serial`). Shippable design fully
  specified; remaining work is the rayon band chunking. **The prize (−3.6 %/−5.1 %) is SMALLER than
  the v1 pool pass at 13.6 %, which needs no new code — only item D's per-model derivation. That
  ordering is the finding.** Round 28 *reasoned* about this change and got the obstacle wrong in
  both directions.
- **(3) Blast radius + wave REGISTERED** (`benchmarks/era2_blast_radius_2026-08-31.md`) — **the last
  outstanding flip prerequisite is closed. And the radius decides its size:**

  | | tiling | accumulation | radius 4 |
  |---|---|---|---|
  | test re-pins | 5 goldens | **0** | 5 (same set) |
  | re-extraction | no | no | **required** |
  | retrain | no (5/6) | not yet checked | **required** (4/6) |

  Tiling and the accumulation have **essentially no data-side blast radius**. Radius 4 carries all
  of it: the **5.74 M-row bigcodec table is ~97 %** of the re-extraction (a fleet wave), the 11 local
  legs + 2 eval grids are a local job, plus **378 board fullevals to RESCORE** (not re-extract)
  under a new era stamp, plus a 6-arm retrain wave. Append-only: a new dated root, never an in-place
  overwrite. **BHdr is the one model constraining BOTH components.**
- The E/F table now carries the F6 accumulation row, the corrected F4 row, and a "what is shippable
  today" section: **tiling alone; radius 4 only behind the registered retrain.**

## 2026-08-31 ~18:4xZ — ROUND 33: ★ ERA-2 IS FLIPPED (`515001dc`, `b58b3ade`) + the fast-profile subset (`0a9e7113`)

**USER DECISION EXECUTED: column tiling + fixed-lane accumulation are DEFAULT-ON in one step.**
Radius 4 not flipped (stays behind the registered retrain wave).
- **Gate re-pins, old → new: ZERO — and not by luck.** Round 30's enumeration predicted five
  goldens, but that was measured at `tile = 32`, the forced-on setting used to FIND the set. At the
  shipped width the fixtures are 64×64, 96×96, 200×150, 128×128 — **all narrower than 1024** — so
  every H entry's `width > tile` guard runs them untiled; and the accumulation re-pins zero
  independently (it moves only `f372+`; every golden pins v1's 372). No tolerance widened, nothing
  ignored, every internal-consistency gate unchanged at both settings.
- **A hole the flip CREATED, closed in the same commit**: if no fixture is wider than the tile, no
  test exercises the shipped configuration — `CELLS` topped out at 592. Added **(1153, 72)** and
  **(2049, 40)** — odd widths crossing one and two tile boundaries, the second leaving a 1-column
  remainder tile — so the mandated pool-size gate now runs where tiling actually fires. **370/0.**
- `H_TILE_WIDTH = 1024` is **DERIVED**: the live window is `H_BLUR_ROW_GROUP × 6 × tile × 4 B` ⇒ a
  1 MiB L2 caps it at 1365; 1024 is the power of two below, fits a 512 KiB L2 (393 KiB), and sits
  inside the measured-flat 512–1536 band. Semantics-not-a-knob warning ships at the constant.
- **★ THE ERA'S HEADLINE PROPERTY, on the shipped build** — same binary, dev (`v4x` AVX-512) vs
  i134 (`v3` SSE4.2): **era-1 `reduce_add` diverges on 66 of 105 slots across vendors; era-2
  `era2_reduce8` diverges on 0 of 105.** (This is what would let the golden gate be re-tightened
  from tolerance to EXACT — registered as a user option, not taken.)
- **BHdr recorded as a USER-ACCEPTED EXCEPTION, not dropped**: `eval_annotations.json` →
  `era2-tiling-bhdr-accepted-exception`, carrying the standing warning that **any future roll-up
  counting tiling as 6/6 is WRONG — it is 5/6 plus this exception** — and naming retrain arm
  W-R4-4 as the fix path. Blemish recorded: a backtick in the flip commit message triggered shell
  substitution (stray empty file, dropped phrase); since `515001dc` was already on origin the lane
  landed a follow-up rather than rewriting pushed history.

### The fast-profile subset — measured on the era-2 build, 2304², min over 5 process starts

| compute set | 1T | 8T | 16T | RSS | vs `944full` |
|---|---:|---:|---:|---:|---|
| `944full` | 278.0 | 124.0 | 113.6 | 105 MB | — |
| `944carriers` | 286.3 | 102.8 | 108.0 | 105 MB | **SLOWER at 1T** |
| `944peaks` | 246.8 | 101.8 | 96.6 | 98 MB | 1.13 / 1.22 / 1.18× |
| **`156`** | **109.6** | **28.0** | **32.2** | **83 MB** | **2.54 / 4.43 / 3.52×** |

Three things the table said that the prose did not: dropping the pool pass independently reproduces
the 13.6 % figure; **`944carriers` is NOT a cheap middle** (slower than full at 1T — it still runs
the masked/IW kernel at scales 0–1); and **basic-only SCALES BEST** (2.54× → 4.43×) because what it
removes is what contends for bandwidth.
**Recommendation — two points**: **`156` + an ADD156-class model** for a real fast profile (within
0.019 CID22 of B and BEATING it on within-image ranking on 7/8 corpora — the criterion a codec loop
actually consumes); **`944peaks`** for callers keeping a 944 MLP (1.13–1.22× at **exactly zero**
rank cost). Keep the peaks (free byproducts). NOT recommended: `944carriers`, dropping peaks, and
`372` (dominated). **API: no new public types** — `ComputeSet::from_block_profile(model)` is the one
new function; it makes the `944peaks` case automatic, and a fast profile is then just a profile
carrying an ADD156 bake. Retrain **W-FAST-1 registered, not launched** (not required — ADD156 already
trains on basic families only — but warranted, since every current bake was trained at era-1; it
shares the blast-radius wave's re-extraction). Per-corpus deltas vs B and ssim2 are marked
NOT-MEASURED-HERE / ATTACH from the frontier lane rather than estimated.

## 2026-08-31 ~19:3xZ — ROUND 34: ADD156 SHIP AUDIT — the model is fine; the PRODUCT PATH and the INSTRUMENTS are not (`b4ddb927`, `f6ebb3b3`; doc `benchmarks/add156_ship_audit_2026-08-31.md`)

**Verdict: ADD156 cannot ship as a fast profile today, and the reason is NOT the model.** It passes
what the use case needs: **dial fully green and UNCOMPRESSED** (G-DYN **85.5** vs B's 86.1,
monotonicity 98.5 %, zero dead-zone, **0 ladder inversions** on real zenjpeg ladders at 512 AND
2048 px), identity exactly **100.000000**, buffered-vs-streaming agreeing to **0.000e0 at every
point**, thread-invariant bit-exact at 1/4/8/16/32, **M3a 0.9641 / 27-of-27 GOLD with M2 = 1.0000**
(structural — a single linear layer's gradient IS the model), packs to **837 B** rank-identical on
all 14 corpora, and it is the most era-robust bake in the roster. The CID22 gap replicates at 0.0187.

| battery | result |
|---|---|
| `freeze_check` §5 | 2 FAIL (CID22 0.8634 < 0.89 — **B fails too**; repro MISSING), 9 ATTACH |
| balanced-2026-08-04 | **6/8 floors** (F1 CID22, F3 nonphoto fail) |
| `bake_verdict` scorecard | G5 + G-IM26 fail (**B fails G-IM26 too**) |
| DIAL / G-DYN | **all green** |
| **G-RANGE** | **FAIL 4 of 8 corpora — NEW finding** |
| M3a / block profile / product API / thread-invariance | PASS |
| corruption head, HDR, loop map | **ABSENT / N-A**, not failed |

**14 defects; the five that matter:**
1. **D1 (BLOCKING) — there is no product path at all.** `ComputeSet` is `pub(crate)`,
   **`from_block_profile` DOES NOT EXIST** (round 33 reported it as the one new function; it was
   specified, not written), and `ZensimProfile::Custom` sits behind a non-default feature. **The
   advertised 2.54× is unreachable by any caller.**
2. **D2 (high) — `bake_verdict`'s default KonJND corpus is the DILUTED 1008-ref file** while the
   correct JPEG-504 ruler sits in the same directory. **It inverts the headline**: ADD156 0.4462 vs
   B 0.6497 on the default; **0.5332 vs 0.5194 on the right ruler.**
3. **D3 (high)** — the registered selection rule stamps ADD156 *"era-bridge — never shortlisted"*:
   structurally unselectable.
4. **D4 (high)** — `bake_dial_refit pack` without `--neg-tail` **silently deletes the negative
   tail** (p5 −12.43 → 0.0000, up to −0.021 SROCC on LIVE) **while the prune identity gate still
   reports bit-identical**, because it only checks the network in-domain.
5. **D7 (high)** — G-RANGE fails 4 of 8 corpora including **100 % of HF near-lossless above the top
   knot** — the exact zone the profile is sold on (narrow [0.301, 0.968] domain,
   `n_feature_bounds: 0`).
Also: **D9** the builder's defaults return **0.000000 for every q from 10 to 100 with no error**
(reproduced first try); **D10** three registry entries sit OUTSIDE `entries[]` so `freeze_check`
silently drops them — **including the one documenting D2**; **D14** zenmetrics hard-codes
`latest_preview()` everywhere, so **the fleet cannot score ADD156 at all**.

**Corrections to published claims (mine included):** "beats B on within-image ranking on 7 of 8"
replicates as **6 of 8** — though the substance is STRONGER than stated: **B ranks 21 % of HF
near-lossless ladders backwards; ADD156 0 %.** AIC-4's ⛔INVERTED −0.9325 is a **CORPUS** property
(shipped B reads −0.8906, also 100 % backwards, same root). Era-2 exposure separated for the first
time: **the accumulation moves ADD156 by exactly zero; tiling by ≤0.0013 dial points** at 2048 px.
Not measured, with reason: the campaign battery (G-OUT v2 / G-GRAN v2) is Python-owned and
unreachable from `freeze_check` (`dial_range_gate.py` has a hardcoded `BAKES` dict needing a hand
edit) — recorded as NOT RUN, never as passed. No gate, threshold, default or public API changed; no
retrain run. Two API changes requested in the doc: promote `ComputeSet` + write
`from_block_profile`, and a fast-profile variant.

## 2026-08-31 ~20:4xZ — ROUND 35: the PRINCIPLED defects fixed (user directive) — and three audit findings did not survive contact with the code (`a2074bae..a188b0b3`, zenmetrics `e4bb566b`/`fc971815`)

Every fix carries a **failing-first** test; nothing was relaxed.

| # | outcome | failing-first test |
|---|---|---|
| **D9** silent 0.000000 | **FIXED** `5dd4d4e9` | `d9_spline_bake_without_skip_score_mapping_is_refused_not_silently_zeroed` — pre-fix panics *"returned Ok and score 0 — silent garbage instead of a diagnostic"* |
| **D4** pack deletes the neg tail | **FIXED** `be6ba6c2` | `d4_flat_bottom_spline_deletes_the_negative_tail_and_is_now_caught` |
| **D10** dropped registry entries | **FIXED** `cb76bd5c` | `d10_registry_findings_are_never_silently_dropped` (pre-migration registry panics, naming all 3 orphans) |
| **D2** diluted KonJND ruler | **FIXED** `6e508793` | `konjnd_default_ruler_is_jpeg504_not_the_diluted_file` |
| **D14** fleet cannot score a bake | **FIXED** (zenmetrics) | `cpu_umbrella_scores_a_runtime_selected_bake`, proven failing-first twice |
| **D3** "never shortlisted" | **CLAIM FALSIFIED**; misleading note fixed `0753275a` | `d3_era_bridge_class_is_a_label_not_an_exclusion` |
| **D7** OOD guard | **MEASURED — free half validated, costly half DECLINED** `7cb78395` | n/a (artifact lane) |

**Three audit findings did not survive contact with the code:**
- **D3 is FALSIFIED.** `class` is compared only against `"944-ensemble"`; `"era-bridge"` is tested
  nowhere, and selectability is `m3a != Unmeasured && n_pass > 0`. Running `--select` over the board
  fullevals prints **`SELECTED: ADD156_safesyn_only_raw_lasso — 6/8 floors, selection_composite
  0.9644`, ahead of B at 0.9151** — both stamped `era-bridge`. The audit's "NO" came from **its own
  fulleval omitting `m3a_coherence`** — the value its own §1.6 had measured at 0.9641. **ADD156 was
  always selectable.**
- **D10 is ~7× larger than filed**: beyond the 3 out-of-array findings, **19 of 42 entries had a
  scope the matcher cannot evaluate**, so they matched zero cells ⇒ **22 of 45 findings were
  invisible**.
- **D7 understates and mis-names**: G-RANGE fails **8 of 14** corpora, not 4 of 8; and
  `n_feature_bounds: 0` is NOT the difference from B — **B reports 0 too** (guards ride in
  `feature_transforms`).

**D2 blast radius, measured**: only **17 of 378** board cells read the diluted ruler; **361 already
read n=504**, so the fix makes the board *internally consistent* rather than moving it. The 17 are
enumerated by name in the new registry entry `konjnd-372-diluted-ruler-pre-2026-08-31` and include
all four `peer_*` reference metrics. Nothing retro-edited. The memory note
`feedback_konjnd_human_score_two_columns` is about column SEMANTICS and is unaffected (both rulers
carry the same PJND scale, verified). `bake_compare` had the same defect and is fixed behind the
same owner.
**D4 reproduction safety**: the lane chose **refusal over flipping the default** so nothing changes
silently — verified against the pinned pre-fix binary (`--neg-tail` → `879409d3…` identical to old
`--neg-tail`; `--no-neg-tail` → `3f03a734…` identical to the old DEFAULT). **No committed artifact
depended on the old default** — every byte-repro doc and all 16 `pack` call sites in `scripts/*.sh`
already pass `--neg-tail`. The gate now prints `dial tail: ⚠ DELETED …` beside `PASS — BIT-identical`,
which the old binary never did.
**Left open**: **D1 untouched** (needs the user's API approval). **NEW defect, not fixed:** `pack`
always refits the spline, so it **erases** a D7 spline extension, and `extend-top` after a pruned
pack is structurally impossible ⇒ **no packed ADD156 can pass G-RANGE today — ship unpacked, as B
does.** A spline BOTTOM extension has no owner (building it would make D7's remaining half free).
G-RANGE over-counts on any bake whose bottom knot is the dial's zero. Pre-existing, not this lane's:
three `blur.rs` ring tests panic `subtract with overflow` in DEBUG only (release green); a stale
`push-qqkqluuttltu` bookmark conflict; zenmetrics#51.

## 2026-08-31 ~21:5xZ — ROUND 36: board regenerated on the corrected KonJND ruler — the column's leader was an ARTEFACT, and three pre-existing board defects fixed (`5d38820c`; doc `benchmarks/board_regen_2026-08-31.md`)

- **17 of 17 diluted-ruler cells re-scored; all 378 board cells now read KonJND at n=504** (verified
  by scanning `rank.konjnd.n` board-wide; the other 361 untouched). 13 bake cells got a fresh
  `bake_verdict --full-json` on sha-verified bytes at their own root, grafted with
  `--graft-into … --reslice-rank konjnd` (which refuses to write unless every other key is
  byte-identical): **12 of 13 reproduced every non-KonJND rank block BIT-IDENTICALLY** — which
  independently answers the era-2 question, **the flip does not move a `bake_verdict` rank read**
  because that path scores stored parquet features. The eleven `@cur372` values match round-4b's
  independent shim run (different binary, different path) at **max |Δ| = 0.0e+00**. The 4 `peer_*`
  rows were rebuilt through the owner after proving the `/jpeg/` subset IS the ruler population
  (504 rows, `pjnd` bit-identical to the parquet's `human_score`).
- **★ ORDERING: 355 of 378 cells change position in the KonJND column — its leader was a ruler
  artefact.** `v02_bvls_NO_shaping@cur372` **1 → 295** (0.7275 → 0.3296); `peer_iwssim` **364 → 7**
  (0.1859 → 0.5704); shipped `B@cur372` 2 → 25; **ADD156@cur372 130 → 14, and now reads ABOVE B
  (0.5332 vs 0.5194)** — the D2 inversion, corrected on the board. Composite: 188 cells move,
  **top-10 unchanged**; `freeze_check --select` winner **unchanged** (`W10L9P_s4005_packed`, 8/8,
  0.9876); three cells cross the F2 KonJND floor.
- **Correction to my own brief on ADD156**: `SELECTED: ADD156 — 6/8, 0.9644` reproduces exactly **in
  the two-candidate pool it was run on**. Over the full 374-cell board ADD156 is rank **101** and
  the rule selects `W10L9P_s4005_packed`; **within era-bridge it is 3rd**. It still leads shipped
  **B** by **+0.0457** (B is 0.9187 post-fix, not the 0.9151 I quoted). The board has **no place to
  show `selection_composite`/floors** — its "Gate scorecard" is the unrelated CODEC_TARGET_GOALS
  system — so the lane did NOT invent a panel; the table is in its doc.
- **Badge premise CORRECTED by measurement**: the registry repair added **ZERO** badges — all 22
  recovered findings were `{"manual": …}`, which by construction matches no cell (counted with the
  gauntlet's own matcher: 42→47 entries, 2,959→3,355 matches, and the +396 is round 35's two
  MACHINE-scoped entries, not the repair). What the repair bought is **readability** — the 3
  formerly out-of-array findings are now in the page's `annRegistry`. Badge rendering verified
  end-to-end incl. a no-finding control (`W10L9P_s4005_packed` clean in every column).
- **Three PRE-EXISTING board defects found and fixed at the owner**: (1) **276 cells' `composite`
  was STALE** — the 2026-08-28 reslice replaced two of its six terms and never updated it (|Δ| up to
  0.0237 **on the default sort key**); `--reslice-rank` now carries the verdict's composite in the
  same gated write (registered `composite-stale-after-rank-graft-2026-08-28`). (2) **Peer rows led
  the composite sort at 1.11–1.42** vs the best bake's 0.872 — the legacy unnormalised fallback; now
  NOT MEASURED (em-dash, sorted last). (3) **`peer_cvvdp.fulleval.json` is not valid strict JSON**
  (73 bare `NaN`), which aborts `freeze_check --select` over a glob; also fixed
  `build_peer_fullevals.py` silently deleting other lanes' grafts on re-run. The render gate got
  **stricter**, not relaxed (it never checked where nulls sort and its sentinel encoded the opposite
  of the page's rule; negative control fails all three assertions).
- **Curation**: ADD156, shipped B (both era halves), the 944 flagship, and the newly-promoted
  **`Q7b_pools_g0.2_a0.2_b0.97`** — the W-LIN 7b candidate the registered rule names, **which was
  not on the board at all** (7 of 14 corpora; rest em-dashed) — plus a new `W-LIN` `family_of`
  branch (those stems had been falling through to "pre-944 era"). **Gates PASS** (362 bakes, 18
  tables, sort clicks, ECharts SSR, **960 ⚠-badged cells**). Board 19.95 MB →
  http://localhost:3300/zensim/reports/summer_gauntlet.html ; prior board + pre-change cells
  preserved with SHA256SUMS.

### Round 36 addendum — the 2026-08-31 discussion set, and two silently-inert sets repaired (`0aec701e`)

**Appended** `2026-08-31-era2-fast-profile` (era-2 flip + fast-profile subset + the ADD156 ship
audit): `ADD156_safesyn_only_raw_lasso`, its `@cur372` half,
`b_sdr_linear_cid80_inclwinsor_dense_dial@cur372`, `Q7b_pools_g0.2_a0.2_b0.97`,
`W10L9P_s4005_packed` — **all five resolve, nothing dropped**; incumbents unioned by the page, not
duplicated.
**The two null-id legacy sets were broken WORSE than suspected, measured on the built payload:**
(1) **they selected NOTHING** — the page reads `d.bakes||[]` and both carried their models under
`members`, so `discussionSets[5].bakes.length == 0` and `[6] == 0`: selecting either showed
**incumbents ∪ peers only (7 rows)**, never the 6 and 7 models they name, and the options rendered
with correct labels so it was invisible; (2) **sorting was broken** — `gauntlet.py:882` sorts on
`x.get("date","")` descending, so a missing date sinks: two sets labelled 2026-08-29 rendered
BELOW the 2026-08-28 set, violating the `_schema`'s own latest-first contract. Repaired in place as
`2026-08-29-r6-fresh-b-linear-pair` and `2026-08-29-b-swap-decision` with dates, **and** `members`
→ `bakes` (ids+dates alone would have fixed their POSITION while leaving them inert — a cosmetic
fix reported as complete); `note` kept as provenance, nothing removed or reordered. All 8 sets now
resolve every name they list and the date order is monotone.
**Dropdown verified through the SHIPPED handler**, not a re-implementation — the `sel.onchange`
block was extracted verbatim from the emitted HTML and run against the emitted `DATA`: the new
option is present and FIRST, and selecting it yields **exactly 12 visible rows = 5 set + 3
incumbents + 4 peers**, disjoint, no unresolved name. Gates PASS with counts identical to the
previous regen (962 badged cells, 362 bakes, 18 tables, ECharts SSR); byte-comparing old vs new
HTML around the `discussionSets` array, prefix and suffix are **byte-identical** and the file grows
by exactly **+477 B**. Prior board preserved. Board:
http://localhost:3300/zensim/reports/summer_gauntlet.html

## 2026-08-31 ~23:3xZ — ROUND 37: THE FAILURE-PROFILE PANEL (user: "minimal info on its flaws and where they will hurt") — ladder inversions board-wide, and pooled numbers proven to hide the risk (`926e8020`..`dc477a08`; doc `benchmarks/failure_profiles_2026-08-31.md`)

**The board now says what a model gets WRONG**: a "Failure profile — what breaks, how big, where
you meet it" section under the scoreboard — findings ranked by product impact in the four-part form
(what breaks / how big / where you meet it / evidence), a side-by-side comparison for the
discussion set, named worst ladders, the honest inverse ("reliably good at"), and an explicit
NOT-MEASURED list. Nothing recomputed — stored numbers thresholded into severity + situation.
**`frac_negative` was the buried lede**: present on 374 of 379 cells and rendered on no panel — it
is the share of images where a per-image codec loop walks the WRONG WAY.

**Ladder inversions, board-wide, built into the owner** (`bake_verdict`'s dial panel: codec × zone
+ content × zone buckets, ladders-ending-backwards, worst ladders BY NAME; the `all` rows re-derive
the pooled G3 counters and the run ASSERTS it to the last digit). **322 of 379 cells measured, 0
graft failures**; the driver accepts only the run whose pooled dial block is byte-identical to the
board's; every skip carries a reason. **q≥85 is where the board fails**: median inversion 2.83 %
(vs 0.76/0.79 below), **189 of 322 models carry a ladder that ends backwards there**; avif worst
(median 3.64 %, p90 14.03 %), webp nearly clean. Discussion set: shipped **B**'s worst single
reversal at aggressive quality is **30.2 dial points** — a statement its pooled mono 0.9792 does
not contain; `Q7b_pools` has NO backwards ladders but the deepest single step (91.3) — a different
risk shape a mono column cannot distinguish; the 944 flagship is cleanest at q≥85 (0.36 %, 0
backwards).

**Two honest negatives, both registered**: (1) the byte-identity gate found **7 board cells cut on
the un-quarantined 2026-05-29 grid** (their inversion counts inflated BY THE GRID) — and this
FALSIFIED the pass's own draft headline: the ~19-point "collapse" at a JXL rung is CORRECT scoring
of a broken encode (66.7 % of features grow 5–8 orders of magnitude on the bad grid vs 0.6 % on the
healthy one); (2) `aic4-corpus-wide-per-ref-inversion` measured on 373 cells (median 60 % of refs
backwards) and reported as a CORPUS property, never a model defect. **Extended the audit's HF
finding**: on the real 48-ref `hf_nearlossless`, shipped B ranks **20.8 %** of references backwards
where ADD156 ranks 0 % — and `mlp_2L_diverse_H128@cur372` / `winner_dial…@cur372` invert **~86 %
of references with NEGATIVE per-ref means while publishing positive pooled SROCCs**; that shape is
now a `blocker` row on the board.

Render harness EXTENDED (a blank failure panel or a NOT-MEASURED cell drawn as zero now FAILS the
gate — it caught a real crash pre-ship); gates PASS (974 badges, 29 tables). Board +3.76 % to
20.7 MB (already past the documented cap before this pass; the trim lever is the registered size
rule). Not measured, stated: per-ref-by-name on rank corpora (zenstats returns aggregates; the
interned-id fix is a sibling-repo item), KonJND has no per-ref statistic, G-RANGE/G-RD/G-TARGET are
other tools' rows.

## 2026-09-01 ~00:3xZ — ROUND 38: ★ THE SSIM2-REPLACEMENT BAR — defined, measured, and the honest per-axis verdict (4 commits; doc `benchmarks/ssim2_replacement_bar_2026-08-31.md`, 1,121 lines)

**Has real progress been made toward "the new ssim2"? Per axis:**

| axis | verdict | the numbers |
|---|---|---|
| **SPEED** | **WON, 1.2–7× — and it had NEVER been measured** | fast-ssim2 **21.7 ms** @576²/1T vs the 944 walk **18.3** (1.19×) and the 372 fold **9.4** (2.31×); public API end-to-end ~1.95× @1T, **5.9–9.9× @8T** |
| **RANK** | **statistical TIE at the top** (progress on July's clear-#2, not a win) | CID22: ssim2 **0.8894**, W10L9PH **0.8927** (+0.0032, CI [−0.007, +0.013]; reference-clustered CI ±0.010) |
| **DIAL** | **944 class WINS; the shipped default LOSES** | ssim2 mono 0.9930 / 14 % NL-ladder inversions / 0 % backwards; W10L9P **0.9947 / 6 % / 0 %**; shipped B **0.9792 / 17 % / 2 %** |
| **HDR** | **shipped BHdr WINS; the frozen candidate loses to BOTH** | BHdr **0.7536** > ssim2-PU **0.7044** > HDR944_L1T1 **0.6664** (UPIQ, n=380 human JOD) |

**The user's worry is PARTLY TRUE and precisely locatable**: the 944 class ties the opponent on the
gold holdout and beats it on the dial, and the whole crate is faster — but **the shipped default B
is measurably WORSE than ssim2 within-image on CID22 (−0.0079, CI excludes 0) and on the dial. No
gate could have said so, because until today `bake_verdict` refused to run on a reference metric
and NO gate in the stack contained the opponent at all.** **Nobody passes the exam (W1–W7).**
Closest: `W10L9PH_s4004_packed` — fails W1 on KonJND (−0.027), and W2 partly because the
near-lossless HUMAN axis is **NOT MEASURED (not failed) for every 944 model**.

**Premise ledger, top rows (the 944 story corrected at the source):** the literal "only slightly
slower than 156" sentence does not exist — the real claim was **"≤3–4 % … the v1 block is NOT where
the time goes"**, built on a table that compared a FOLDED 944 walk against a BUFFERED 372 walk
(different engines). Truth: v2-348+append is **+76–101 % @1T, +114–152 % @8–16T**. Dollar/hour cost
of the wrong premise: **not recoverable — stated, never estimated.** Also corrected: "dense is
22–26 %" (13.5 % of the block / 7.3 % of the walk; callgrind profiled a tier we don't ship); "3.5×
is the machine's bound" (the fold's own footprint); the 2026-05-20 "bit-equivalent" audit sampled
only f0..f99 — the one block that didn't drift — so **B was fit on pre-fix features and serves
post-fix ones**; and **the orchestrator's own "ssim2 has no HDR" claim is FALSE** (fast-ssim2 has a
PU path — caught by reading the code). **Balanced by what 944 BOUGHT**: the only class at-or-above
ssim2 across human corpora AND the only one beating it on the dial; the lift is mostly the DATA
(E-M6b), durable at any width.

**Sharpest instrument finding:** `peer_ssim2` holds the **HIGHEST `balanced_composite` on the board
(0.8979, above every model) while ranking LAST at 4/8 floors** — the first by arithmetic (nonphoto,
whose target IS ssim2, at weight 0.30), the second because four floors were structurally
unmeasurable for a peer (two now measurable; ssim2 passes both → 6/8). The board's stored CID22
`srocc_ci` is a pair bootstrap and **understates reference-clustered uncertainty ~2×**.
**Instruments landed** (extensions of owners, both gated): `bake_verdict --dial-peer-scores` (the
dial panel on a reference metric — reproduces a bake's own dial section line-for-line from dumped
cells) and `panel --per-group` (canonical per-group SROCC, parity-exact vs `per_ref_mean`); plus
the fast-ssim2 speed bar. **Seven proposed instrument changes in §4.5 — proposed, not implemented;
adoption of the exam as THE bar is the user's call.**

## 2026-09-01 ~01:3xZ — ROUND 39: the near-lossless "human" axis was NEVER HUMAN — and the exam's only strict win appears (`f2681f5a`; APPENDIX A of the ssim2-bar doc)

- **★ `hf_nearlossless` is an ssim2 SELF-TARGET, not a human corpus**: its `human_score` column IS
  `ssim2_gpu / 100` — float-equal on **1200/1200 rows, max |Δ| = 0.0**; scored by the owner, ssim2
  as predictor: pooled SROCC 1.0000, per-ref 1.0000. **W2's near-lossless clause was UNWINNABLE at
  any feature width** — a 944 extraction would have produced an axis the opponent wins by
  definition. (`hfnlproxy` was already declared `self_target` on the board; `hf_nearlossless`
  escaped only because `peer_ssim2` carries no row for it.) The extraction is also unreachable
  regardless: the 1,200 distorted JXL bitstreams of the 2026-07-06 sweep are GONE (blank
  `encoded_filename` on 1200/1200 of the sweep's own pareto.tsv, both `refit/distorted/` mirrors
  empty, refs rooted in wiped /tmp) — regeneration priced and REJECTED as a substitution (encoder 2
  months past the pinned rev; the GPU-ssim2 target unreproducible here). Bonus defect: **four row
  populations have shipped under the name `hfnlproxy`** (7,224/7,717/9,167/11,356) and the
  opponent's 9,167 is on no surviving root — W10L9PH and ssim2 never shared rows there either.
- **The non-circular replacement, registered**: `hfnl_cid22band` = CID22's top MOS band
  (merged-decile-2026-08-06, MOS ≥ 0.80, **n = 1,425 over all 49 refs**), reference-clustered
  paired bootstrap (B = 10,000):

  | arm | pooled | Δ vs ssim2 | per-ref | Δ vs ssim2 |
  |---|--:|---|--:|---|
  | ssim2 | 0.5058 | — | 0.7099 | — |
  | B | 0.5089 | +0.003 [−0.031,+0.039] | 0.7020 | −0.008 [−0.027,+0.011] |
  | ADD156 | 0.4349 | **−0.070 [−0.103,−0.033]** | 0.6691 | **−0.041 [−0.062,−0.021]** |
  | W10L9PH | 0.4984 | −0.007 [−0.043,+0.023] | 0.7060 | −0.004 [−0.016,+0.010] |
  | **Q7b** | 0.4584 | −0.045 | **0.7250** | **+0.0151 [+0.0006,+0.0301], P=0.980** |

- **Exam update**: W10L9PH **TIES** ssim2 in the near-lossless human zone (both CIs straddle 0) —
  W2 still FAILS but for a measured structural reason, not an instrument gap; its line is W1 FAIL
  (KonJND −0.027) / W2 FAIL / W3-W4-W6 PASS / W7 FAIL. **Nobody passes.** Two rows move: **Q7b now
  holds the exam's ONLY strict win over ssim2 on a named non-circular axis** (within-image +0.0151;
  marginal by its own lower bound, one axis of six, and nominally behind pooled on the same rows);
  **ADD156 gains a new W1 failing axis** (−0.070 pooled in the zone the fast profile is sold on).
  Cross-checks reproduced exactly (B backwards on 0.208333 of `hf_nearlossless` refs, ADD156 0).
- Landed as `rank.hfnl_cid22band` on the six candidate cells (sha-gated grafts; board rendering
  unchanged — the axis is not in CORP_ORDER). 9 gates pass incl. flagless-reproduction of the
  committed bootstrap and arm-is-its-own-board-vector identity. Four superseded exam claims + the
  project CLAUDE.md "targets human_score + ssim2_gpu" wording corrected in place. Two defects
  reported not fixed: `panel --json --per-group` emits two concatenated JSON documents;
  `peer_ssim2`'s stored `per_pair.cid22.mos` is on 0–100 where every model cell's is [0,1].

## 2026-09-01 — ROUNDS 40-47 (consolidated from the orchestration log; Opus weekly limit mid-day → Sonnet resumes)

- ROUND 40 (parked; primary held by the wave lane): HF ingestion lane DONE (487c7bb2..c559c23f). ★ NO TRAINING-LEGAL SIDE EXISTS: AIC-3 BTC/IPTC + AIC-4/PTC + SDR25 are ONE experiment on TEN source images; three board axes (aic3/aic4/sdr25 — the SELECTION COMPARATOR) are built from it; training on any member contaminates all three; would buy only ~900 rows on 10 images. HOLDOUT-ONLY family-wide, registered (`jpeg-aic-family-holdout-2026-09-01`); the CLAUDE.md "untapped HF human training data" premise is DEAD for training — eval-only. New exam axes at scale: 515,250 human forced choices; `panel --pairwise` + `zensim_validate::pairwise` (majority-oracle ceiling beside every accuracy). VERDICTS: nobody strictly beats ssim2 under BOTH boosted/native readings (the lane withdrew its own first headline — 14/36 verdicts flip between readings; the W10L9PH JPEG-AI-leg win is arm-dependent); shipped B BEHIND under both readings (btc −0.0025 CI excl. 0) AND on aic4_all rank (−0.0223 CI excl. 0); Q7b never loses; ★ ADD156 BEATS ssim2 on aic4_all rank (0.9331 vs 0.9127). ext_aic3's target is the CTC DESIGN value (−0.25×level exactly, 600/600); ext_sdr25 ⊂ ext_aic4 with reconstructions differing up to 1.79 JND. Overlap audit clean (d≥12 vs 1,221 refs) BUT `check_holdout_overlap` was BLIND to .bmp — LIVE/TID audits protected nothing (fixed); `panel --json --per-group` double-document fixed; the scipy parity gate had been silently skipped under per-agent CARGO_TARGET_DIR (fixed). NOT REACHABLE: the 130 IPTC_* stimuli (51,870 responses; would 6× the only native-scale unboosted axis) — upstream aicdb.jpeg.org; recovery lane dispatched. Board integration of pairwise axes = a design call, written up, not grafted (correct — accuracies are not SROCCs).
- ROUND 41 (parked; wave holds primary): IPTC lane DONE (779a63d9, 15dcd04c). The 130 "missing" IPTC stimuli were NEVER a separate artifact — IPTC is the June-2024 campaign's method tag for the paper's PTC experiment; the stimuli are the PTC crops ALREADY ON DISK. Identification proven, not assumed: G8 join gate 0 disagreements in 155,610 checks; four falsifiable negative controls (level reversal flips agreement to exactly the mirror −0.9686; rotation collapses it; the two order-preserving controls are provably blind because AIC ladders are JND-equalised — stated, not hidden). No wall upstream (CC BY 4.0, open HTTP; probed anonymously). Extended axis: iptc_native 130/900/35,044 responses; pooled native_all 41,973 decided = the 6× forecast measured. ★ ARM QUESTION SETTLED: two independent NATIVE readings agree 22/24; native-vs-BOOSTED flips 9/24, every flip TIE→LOSS ⇒ the earlier 14/36 flips belong to the boosted rendering — the native axis is the trustworthy one. RESULTS: NOBODY beats ssim2 on native (best W10L9PH +0.0004, P 0.755); §5.4's ADD156/B +0.0051 DOES NOT REPLICATE at 5× triplets (sign reversed; registered invalidated); Q7b takes a real pooled loss (−0.0013, CI excl. 0 at the boundary); ★ on native cross_codec ssim2 captures only A THIRD of the headroom (acc_norm 0.3315) — the near-lossless cross-codec zone is genuinely OPEN; whoever learns it wins an axis nobody holds (but the AIC family is holdout-only — the win must come from training elsewhere). Lane self-caught writing a fabricated commit tail into its manifest — now asserted equal to git rev-parse.
- ROUND 42 (parked; wave holds primary): hybrid lane DONE (ee0baa33..106a51e2). W4 AMENDED (B2): <=1.25x the 156-walk class at BOTH 1T and 8T, margin DERIVED (372 class 1.55-1.85x, 944 2.06-2.68x, ADD156 1.00x — the cut sits in the class gap); fast-ssim2 demoted to context row and measures 2.78-3.40x the 156 walk. Under B2 only 156-class arms pass W4; shipped B FAILS W4 (1.55-1.85x). ★ HYA_w084 (0.84*W10L9PH + 0.16*Q7b) PASSES W1 — neither parent does — with KonJND SUPER-ADDITIVE (0.5390 > parents 0.5006/0.5118 > ssim2 0.5272), window w in [0.76,0.86] (LIVE binds below, KonJND above); but W2's named win lives at w~0.4-0.6 — DISJOINT, structural — and it fails W4 2.6x => teacher/ceiling. NOBODY passes (best = 3 clauses). ★ The 156-class W1 lever is DATA: safesyn 111k->196k rows = +0.057 CID22 on the identical recipe (196k reproduces ADD156 as subset control) -> handed to the wave (re-extract safesyn at pools-944). Distillation: pure-teacher costs KonJND (-0.008..-0.115) -> MIXED targets mandated. Owner bug fixed (bake_dial_refit predict fed pruned bakes a 667-col PREFIX of 944 rows); LIVE/AIC-4 pairable at 944 (exclusion was root-scoped); one self-caught cross-grid wrong claim corrected (W3 is hybrid-neutral for the 156 class). 8T roster speeds NOT measured (box loaded) — wave must. Board 381 bakes w/ HYA + SADD promoted, gates pass.
- ROUND 43 (parked; wave holds primary): free-features RESUMED on Sonnet and DONE (d82b5dfb, 7b54dbea). Recovery clean (verified the predecessor's mid-commit optimization before trusting it: batched raw-moments reduce, precision 5.35e-6 worst vs 5e-4 tol). Classification complete: free set = 109 slots (72 peaks + 37 raw-moments; class B finalize-only = 0, empirical); raw_moments accumulator priced honestly at +0.8-1.6% @1T (not "free"). Quality probe vs ssim2 at ADD156's operating point: 156+free buys +0.0527 pooled / +0.0291 per-ref on hfnl_cid22band (~30% of the 156 class's remaining gap to ssim2 there), CID22 +0.0405, KonJND +0.0822, cost SDR25 −0.058 / AIC4 −0.023; probe on the 111k leg (relative read licensed by wave-r4's own 111k-vs-196k finding). Hand-off sent to the wave (train students on 156+free beside plain 156; folded720append2pools regime required — the *_pure roots zero the peaks half). ⚠ USER CALL: slot_classification.tsv (48KB, plain-text, script-reproducible) landed in-repo over the 30KB rule — surfaced, not resolved; options = leave, or follow-up commit moving to /mnt/v with a pointer.
- ROUND 44 (parked; wave holds primary): squintly prep RESUMED on Sonnet and DONE (squintly 7fa11d15/e9dbda62/5022056 + zensim protocol/literature docs). Literature report RESCUED verbatim to benchmarks/squintly_literature_basis_2026-09-01.md; fatigue sub-question answered from zenpapers (block=120 trials + mandatory 3-min break per AIC-HDR2025/BT.500; NO n=1 test-retest standard exists in the corpus — reported NOT FOUND). Study: 2,536 pairs — gMAD disagreement 1,200 (both directions x both candidates x 3 zones, 100/cell per UPIQ), random control 400, ladder contests 600, calibration 120, spaced repeats 216; candidates verified on native regimes (the --regime hazard avoided). Disjointness: ZERO genuine contamination (two independent sweeps converge; the one extreme match is a dHash-degenerate blank-margin scan) — and the reconciliation CAUGHT A REAL BUG: the mechanical exclusion never fired (base-id vs rendition-filename mismatch; count printed correctly while excluding zero rows) — fixed, corpus re-mined (795 refs, 3,564 blobs, 220 MB), re-smoke-tested. Staged + end-to-end verified; start command in the protocol doc. Honest gap left open: squintly's toggle lacks PTC's exact min-interaction/rate-cap enforcement. ZERO real judgments yet — the 10 hours are the user's.
- ROUND 45 (parked): wave-r4 RESUMED on Sonnet and CLOSED (726503dd, 3974f5d4, feec9634, 81e2e95d; §19 closing table). W4 8T measured: flagship/156 = 2.97/3.40/4.06x at 576/1152/2304 — threading WIDENS the 944 gap; both 944 columns fail W4 by 2.4-3.2x. A6 (196k safesyn) = best plain-156 profile, still loses CID22; caught+fixed a real pack --anchor scale bug (raw 0-1 human_score anchor produced a broken dial). W3 measured for every arm from existing data (ssim2's bar independently re-derived 0.9930/0% to 4dp); all 944 arms pass W3, only A2ctrl + A6 λ0.3 in the 156 family. ★ A4b (156+free small MLP) = the standout: fails W1 on KonJND ALONE (the flagship's own shape) with the wave's HIGHEST composite 0.8664 > teacher 0.8601, at ~5-7% over the cheapest walk (evidence-inferred; direct bench registered). Nothing passes outright (pre-registered expectation). Follow-up lane launched: close A4b's KonJND via kon-data mass + mixed HYA-distill (kon-super-additive teacher) + combined, k=2, exam verdict incl. whether anything passes W1-W7 outright and rides Profile D in a default build; + the direct W4 bench for the basic+peaks+raw_moments shape. NOTE --keep-features already existed (the wave doc's "needs trainer change" claim was wrong, corrected).

## ROUND 46 — Profile D SHIPPED + the benchmark verdict (`0f7eb2ea`, `f466c805`, `c98a5920`)
Profile D: one public addition (`ZensimProfile::D`, non-exhaustive, default-on `candidate-profiles`),
ADD156 arm-A spline-extended UNPACKED; default build = buffered fallback (B-class speed), the
156-class fast form behind `feature-regime-v2` (un-gating deferred, stated). Wide-bake fix
(`fold_engine::wide_bake_v2_read`): free-set 944-wide bakes now get the CHEAP set through
`from_block_profile`. **Benchmark verdict ("I remember a 10")**: the claim = README's 18×@4K
(MT-vs-1T C++) + 4× 1T. **NO regression — every cell faster than published — but both ratios
HALVED** (4.57×→2.34× 1T; 18.4×→8.7–9.5× MT) because the OPPONENTS got 36–69 % faster while
zensim barely moved; matched-thread honest frame ~1.6–1.7×; **Profile D = 8.2× @4K matched-MT —
closest to the 10× band**. Doc-integrity flagged: both README claims + the 7950X box name.
(Post-completion resume of that lane died on a `[bio]` safety-filter false positive — work all
landed first; agent not resumable, noted.)

## ROUND 47 — A4b's KonJND: ALL THREE CLOSERS FALSIFIED; A4b stands as-is (9 commits `38d948ce`..`2c348ad6`; wave doc §24)
| arm | KonJND | Δ vs A4b-control 0.4327 |
|---|--:|--:|
| K1 kon-weight 1.8 / 2.4 | 0.3472 / 0.3524 | **−0.086 / −0.080 — the certified lever BACKFIRES on this class** |
| K2 HYA-teacher leg (100 % key-join, 192,714/192,714) | 0.4317 | wash, and costs composite (0.861 vs 0.866) |
| K3 combined (selected w=2.4) | 0.3553 | −0.077 |
**The kon-data-mass lever (+0.034 on the 944 flagship) is class-conditional — it INVERTS on
156+free.** 7 of 8 new arms fail a within-image CID22 axis the control passes; K1 w=2.4 fails LIVE
outright on both seeds. **A4b/K4 remains the best 156-class profile of the campaign.** W4 measured
directly (new `free156_peaks_raw` bench arm, ASLR, N=3×{1,8,16}T): **1T clean PASS every size**
(4.3–8.5 % over bar, matching the estimate); 8T passes 576²/2304², **FAILS 1152² at a tight
repeatable 1.44–1.46×** (not noise); 16T noisy (not part of the bar). **Exam: nothing passes
W1–W7** — W1/W2/W3/W7 fail for every arm, W6 passes, W4 mostly passes with the one measured
exception. Remaining honest KonJND paths for the 156 class: new human near-threshold data (the
staged squintly study targets exactly this zone and is data we would OWN), architecture beyond
this recipe, the unexploited class-C in-register free slots, or shipping D with the kon weakness
stated while B/944 serves kon-sensitive uses.

## ROUND 48 — board closure (both closure lanes onto the gauntlet, `5f9f2376`/`4b95b4a2`/`52932525`..)

Two closure lanes landed the same session and both are now on the board, per the standing
'gauntlet always current' rule (every verdict cell belongs on the board, negative results
included).

**(a) a4bkon exam stamp + K1-K4 promotion.** `5f9f2376` stamps APPENDIX C onto
`ssim2_replacement_bar_2026-08-31.md`: the a4bkon lane's three KonJND closers (K1 kon-weight
1.8/2.4, K2 the ttbig mixed-teacher leg, K3 combined) all fail to close A4b's KonJND-only W1
gap — K1/K3 make KonJND *worse* (−0.0855/−0.0804/−0.0774 vs A4b's 0.4327), K2 is a
statistical wash (−0.0010) but costs composite and a within-image CID22 axis A4b itself
passes. A4b/K4 stands as the best 156+free profile produced (composite 0.8664, the highest of
the whole wave). `4b95b4a2` then promoted the 9 a4bkon fullevals (K1 w1.8/w2.4 × 2 seeds, K2 ×
2 seeds, K3 × 2 seeds, K4) via `scripts/promote_fulleval.py --strip-per-pair` +
`--set-block-profile` — K4 under its true identity `A4b_s4004` (sha256-confirmed identical to
wave-r4's own file, not a duplicate name), K1-K3 as `a4bkon_K*`. All 9 round-tripped exactly
against wave doc §24.7; board grew 21,035,320 → 21,386,638 bytes; both mandatory gates PASSED.

**(b) wave-r4's own 16 arms (this round).** Inventoried `wave_r4_2026-09-01.md` §19 (the
closing exam table) against the artifact root `/mnt/v/output/zensim/wave-r4-2026-09-01/`:
16 `.fulleval.json` files on disk, matching exactly the 13 §19 table rows plus 3 arms that
exist on disk but aren't in the closing table (`A1_r4_s4004`, the pre-fix diagnostic; `A2_r4`,
superseded in the table by A6's bigger leg but never re-cited; `A2b_l0.002`, the sparser-λ
sibling of the promoted `A2b_l0.3`). sha256-cross-checked all 16 against the 390 existing
board fullevals first: only `A4b_s4004` was already present (byte-identical, promoted minutes
earlier in (a) as K4) — not re-promoted. Of the remaining 15: **14 promoted**, **1 skipped**.

Skipped: `A1_r4_s4004` — the wave doc (§7.3/§7.6) explicitly disclaims this file as a
wrong-regime diagnostic: the first A1 retrain was scored with a bare `bake_verdict` call that
silently fell back to the era-1 canonical features root instead of the wave's own
`foldapp2_views/` root, and the doc says outright the resulting `.verdict.md`/fulleval "are
kept on disk as that historical (wrong-root) record; they are **not** cited below" — best_val
0.3058 (vs the corrected retrain's 0.9242), dial-mono 0.9271 (below the 0.93 gate). This is
bug-diagnosis evidence, not a candidate arm; the doc's real A1 flagship citation is
`A1foldapp2_r4_s4004`/`A1foldapp2_r4_s4005`, both promoted under their own on-disk names.

Promoted (own on-disk stem as board name, matching the K4-keeps-its-own-identity precedent —
none of these needed a `waver4_` disambiguation prefix, they're already wave-scoped and
distinct): `A1foldapp2_r4_s4004`, `A1foldapp2_r4_s4005`, `A2_r4`, `A2b_l0.002`, `A2b_l0.3`,
`A2ctrl_l0.3`, `A3_r4_s4004`, `A3_r4_s4005`, `A3b_s4004`, `A4_r4_s4004`, `A5_r4_s4004`,
`A5_r4_s4005`, `A6_l002`, `A6_l3` — all via `promote_fulleval.py --strip-per-pair` then
`--set-block-profile`; M3a is `null` on all 14 (never measured this wave, `harvest_bakes.sh`
was not run — correctly rendered as an em-dash, never a fabricated zero). Spot-checked 4 cells'
CID22/KonJND/composite against the wave doc's own tables — **exact matches on every figure**:
`A1foldapp2_r4_s4004` CID22 0.8897/KonJND 0.4773/composite 0.8601 (§8.1+§23); `A6_l3` CID22
0.8495/KonJND 0.4870/composite 0.8150 (§21.2+§23); `A3b_s4004` KonJND 0.3540/composite 0.8598
(§23), and its promoted `block_profile` reproduces §23's fingerprint exactly (`v1_basic`
156/156, `v1_peaks` 72/72, `v1_masked`+`v1_iw` 0/72 each, `f720_943` 37/224); `A2_r4` CID22
0.7760/KonJND 0.3964/composite 0.7650 (§9).

Appended discussion set `2026-09-01-wave-r4` to `board_discussion_sets.json` (current
id/date/label/bakes/note schema) listing the 14 new names + `A4b_s4004` + the incumbent/peer
comparators (`W10L9PH_s4004_packed`, `ADD156_safesyn_only_raw_lasso`, `peer_ssim2`), with a
note summarizing §19's closing verdict and the two honest rankings (A5 s4005 cleanest 944-class
W1 profile; A4b cleanest 156-class-ish + highest composite of the wave).

Regenerated the board (`bandwise_dashboard.py --fulleval-dir ... --out summer_gauntlet.html`,
404 fullevals now on disk, 53.8s wall): 21,386,638 → **21,872,718 bytes (+486,080 B, +0.46 MB,
20.86 MB total)**. Both mandatory gates (`gauntlet_gates.sh`) **PASS**: GATE 1 (`node --check`,
2 script blocks) and GATE 2 (DOM-shim render harness: 387 bakes rendered, 13 sections, 31
tables, 641 rows, 11 svgs, loop panel 11 models, failure panel 16 rows/345 findings, 1360
registry-annotated cells). Still well over the registered 12 MB cap — the trim decision is
still the user's, unchanged and untouched here, reported not acted on.

### ROUND 48 addendum — origin `push-*` branch audit (2026-09-01)
The wave-r4 board lane flagged a conflicted local bookmark; auditing it found **26 `push-*`
branches on origin** (the `jj git push --change @-` failure-mode residue). Classified by
`git merge-base --is-ancestor` against origin/main: **15 MERGED** (pure noise, safe to delete
whenever) and **11 ORPHANS** — every one from the closed April–May V0_x era (V0_4 ssim2-holdout,
V0_7 340k results, V_24/25 mix experiments incl. the EX-4 FALSIFIED record, ex-mix3 scaffolding,
hybrid+NiN, balanced-tilt launchers, a v2-era repath fix). **No current-era work is stranded.**
One provenance note: `push-znwzswtnrvqx` @ `5401f839` uniquely carries
`zensim-bench/examples/extract_features_372col_zenpng.rs` (229 lines, absent from main) — the
zenpng-decode extractor variant from the canonical-2026-05-21 LARGE rebuild, i.e. likely the tool
that produced `cvvdp_iwssim_LARGE_372col.parquet`; main's own `extract_features_372col.rs`
carries the pairstsv capability via its later lineage. Per the found-in-the-wild rule nothing was
deleted — the branch list itself is the archive; deletion of the 15 merged ones is a user call.
Local-only fix applied: `jj bookmark forget push-qqkqluuttltu` (stale conflicted tracking state;
commits + remote untouched).

**2026-09-01 follow-up (user-directed):** the 15 merged `push-*` branches were DELETED from origin (each re-verified `merge-base --is-ancestor` origin/main immediately before deletion; commits remain reachable via main). Exactly the 11 V0_x-era orphan branches remain, held per the found-in-the-wild rule.

## 2026-09-01 — ROUND 49: PERMUTATION RETROFIT — the 2026-09-01 AVIF subsample sweep was 17.8 % duplicate work

*(Parked by the permutation-retrofit lane while `zensim` was claimed by `claude-dnotax`;
applied verbatim by the post-outage recovery lane. Belongs with "FRESH AVIF WAVE — SVT
BACKEND" / "AOM ARM" above, appended here to keep the ledger chronological.)*

The `avifsub-{svt,aom}-enc-20260901` sweep (successor to this campaign's "FRESH
AVIF WAVE — SVT BACKEND" / "AOM ARM") was declared as a naive Cartesian product
and **3,424 of its 19,200 cells are byte-wise duplicates**. Two alias classes,
both confirmed against `output_sha` in the run's own ledger and by a controlled
encode probe:

- **zenavif svt-rs speeds 7, 8, 9 and 10 are ONE encode** — all resolve to SVT
  preset 9, because C remaps every all-intra preset above M9 down to M9
  (`enc_handle.c:4416-4419`). 30 of 30 `(image, q)` cells covered at ≥2 of those
  speeds have identical `output_sha`; a direct probe gives identical
  encoded_bytes AND identical SSIMULACRA2 to six decimals across the class, with
  speed 6 differing on every cell as the control. **The svt-rs speed dial has 7
  distinct settings, not 10** (presets 2, 5, 8 are unreachable through it).
- **q 98 ≡ q 100 on BOTH backends** — each clamps its lossy dial off the
  lossless quantizer (svt QP 1 via `quality_to_qp_gated`, aom cq 1 via
  `aom_rs_cq_level`). 60 identical-`output_sha` groups, every one exactly
  `{98, 100}`. **29 distinct quantizers, not 30.**

**The q alias applies retroactively to the 2026-08-30 wave recorded above**,
which used the same `avifsvt_cells.py` emitter and the same 30-point q grid: its
q=100 column duplicates its q=98 column on both backends, so **~1/30 of its
130,590 svt + 125,688 aom harvested cells are byte-identical repeats**
(~4,353 + ~4,190 rows). Anything trained on those tables should de-duplicate on
`output_sha` rather than treat `(speed, q)` as distinct samples — a duplicate
row is not a free extra sample, it is a mislabelled one.

Its **speed** sampling `{4, 6, 8}` is unaffected: those map to presets 4, 7 and
9, which are genuinely distinct. Worth knowing for interpretation, though, is
that **speed 8 is the M9 saturation point** — it is the FASTEST setting the dial
reaches, identical to 9 and 10 (and to 7), not a midpoint between 6 and 10. A
speed model fit on `{4, 6, 8}` therefore has no sample between preset 7 and the
saturation point, and none at presets 0–3 (speeds 1–3) at all.

Fixed at the owners, not worked around: `zenavif::sweep` gained a backend axis
and svt-rs resolved-state fingerprints (`292582fb`), and `zenmetrics-cli` gained
`sweep::dedup::knob_cell_identity` plus `--dry-run/--emit-cells` on the
`--knob-grid` path (`e91a03b4`). A drifted mirror was repaired on the way:
`speed_to_svt_preset` returned the un-clamped 0..=13 value while the upstream
`AvifEncoder::speed_to_preset` it documents itself as mirroring clamps `.min(9)`
(byte-neutral, and its test asserted the drifted value).

Full record: `zenmetrics/benchmarks/avif_sweep_permutation_retrofit_2026-09-01.md`.

## 2026-09-01/02 — ROUND 50: AVIF DOE lane — the 2026-09-01 subsample wave becomes a control arm, and the aom naive arm is cut by 80 % of its CPU

*(Parked by the `claude-avifdoe` lane for the same reason; applied verbatim by the
post-outage recovery lane, which then executed the lane's resume file — see ROUND 51.)*

The `avifsub-{svt,aom}-enc-20260901` wave (32 k-means imazen26 picks × the
speed/cpu-used dial × a 30-point q grid, both pure-Rust AV1 backends) is now the
**control arm** of a registered design of experiments for encoder-knob tuning:
`zenmetrics/benchmarks/avif_doe_plan_2026-09-01.md` (commit `6bc743b8`,
`zenmetrics master@origin`).

What changed for anyone reading the wave's tables:

- **`avifsub-aom-enc-20260901` was shrunk from 9,280 to 5,179 declared cells.**
  Kept: `cpu-used {4,6,8,9}` at all 29 q; `{2,3,5,7}` at a 9-point ladder;
  `{0,1}` at a 5-point ladder. Removed: 4,096 cells by design + 5 known-poison
  cells (`1432.scale3000x4000.png × cpu-used 0`, `encoder_panic`, a zenav1-aom
  KB-41 lead). Verified a **strict subset with byte-identical retained jobs**,
  so no `JobId` moved and every completed cell is reused; the pre-shrink
  manifest is preserved at
  `s3://zentrain/jobs/avifsub-aom-enc-20260901/manifest.pre-doeshrink-2026-09-01.json`.
  Ledger rows and blobs for the 310 completed cells that fall outside the new
  grid are untouched and remain valid data — they are simply no longer declared.
- **Justification is measured, not aesthetic**: aom encode time is flat in `q`
  (every q column 14.27–14.62 % of 7 sampled, uniform 14.29 %), so a speed model
  gets nothing from q density, and the shrink keeps the **full 10-value
  `cpu-used` ladder**. The saving is **128.8 CPU-h of the 161.2 CPU-h** the
  declared arm would have cost — 80 %, almost all of it `cpu-used` 0–1.
- **A cost-model note for anyone re-deriving these numbers:** the aom per-speed
  costs descend from a smoke test whose two images were **both screen-detected
  screenshots** (the class that pays the IntraBC DV-search tax), so every aom
  CPU-h figure in this campaign is an **upper bound**. The svt figures are
  direct measurements on 5 corpus images and reproduce the launch doc's own
  total to 17 %.

Two findings that bear on any table built from this wave:

1. **`encode_ms` is not persisted by the wave at all.** `jobexec` emits it only
   on the `metric` job kind, which re-encodes; the `encode` kind writes the
   encoded bytes (they are the content-addressed output) and `score_file` scores
   persisted blobs without re-encoding, and the ledger schema has no timing
   column. **Any speed/`encode_ms` analysis of these runs has no input data.**
   The DOE adds a separate single-host uncontended timing instrument rather than
   trusting fleet-contended times.
2. **Nothing was scoring.** Over 2026-09-02T01:35Z→01:55Z the encode blob counts
   grew by 1,279 while score blobs stayed at 29 (svt) and 7 (aom); no score
   worker was running on any host. The score runs were declared and their
   manifests refreshed every 5 minutes, with no consumer.

## 2026-09-02 — ROUND 51: THE OUTAGE — `/tmp` quota, and the root cause was ~30× larger than either paused lane diagnosed

Two lanes (`claude-dnotax` in this repo, `claude-avifdoe` in `zenmetrics`/`zenavif`)
stopped mid-flight when every Bash invocation began returning `exit 1` with zero
output. Both wrote resume files, parked their uncommitted work in the working tree,
and stopped honestly rather than guessing. Both diagnosed the *class* of fault
correctly and the *occupant* incorrectly. This round is the recovery lane's
measurement of what actually happened, because the wrong number is the one a future
session would otherwise budget against.

**The mechanism (measured, not inferred).** `/tmp` is a 31 GB tmpfs mounted
**`usrquota`** — so the failure mode is `EDQUOT` against a *per-user* limit, not
`ENOSPC` against the filesystem. The harness appends a `pwd -P >| /tmp/claude-*-cwd`
write to every command; once the quota was hit that write failed, and the failure
surfaced as `exit 1` with no captured output — **on commands whose bodies had
already run to completion.**

**⇒ The load-bearing lesson: `exit 1` with no output ≠ "did not execute".** The
command ran; only the harness's own bookkeeping write failed. Corroborating
evidence: the during-outage `kill` of the two spinning local workers *took effect*
(both PIDs were confirmed dead on recovery) even though the invocation reported
failure and printed nothing. A lane that reads exit-1-no-output as "nothing
happened" will re-run destructive or non-idempotent work. The recovery channel is
**redirect output to a file on `/home`** (`cmd > ~/tmp/out.log 2>&1`) — a different
filesystem, so the command's own output survives even while `/tmp` refuses writes.

**The occupant, corrected.** The working diagnosis on entry was "~730 MB of stale
`svt_census` scratch plus worker parquet staging". Measured, the actual occupant was
**22.93 GB in 10,988 files sitting directly in `/tmp`**: `zenmetrics`' `jobexec`
source-image cache. `resolve_source_raw`
(`zenmetrics/crates/zenmetrics-cli/src/jobexec.rs:289`) stages every source image as
`std::env::temp_dir()/jobexec_src_{pid}_{basename}` after a verified-complete
download, as a deliberate warm-process cache — and **nothing ever removes it**:
not per cell, not on process exit. Because the filename is PID-scoped, every worker
process re-downloads and re-caches its own private copy. The measured shape:
**10,413 distinct owning PIDs (zero still alive) holding only 60 distinct source
images** — roughly 46 duplicate copies of each image. A further **7,639 truncated
`.part` files** were left behind once the quota hit: the code writes a `.part`
sibling and renames on success, so a quota-failed write leaks the partial with no
cleanup path.

**Why both lanes missed it, and the diagnostic that finds it.** Both ran
`du -x --max-depth=1 /tmp`, which rolls depth-1 *files* into the summary line without
listing them: it reported `24G /tmp` while every child it printed summed to ~500 MB,
so both concluded the space was somewhere they couldn't see (root-owned dirs,
deleted-but-open handles) and moved on. Neither is true here. **`find /tmp -maxdepth 1
-type f -printf '%s\n' | awk '{s+=$1} END{print s}'` is the diagnostic that finds it**
— use it whenever `du`'s total and its children disagree. (`quota`/`repquota` are not
installed on this box, so `usrquota` is only visible in `mount`.)

**Reclaimed, conservatively.** 22.77 GB freed; `/tmp` went 78 % → **3 %** (735 MB
used, 30 GB free). Every deletion was gated on recoverability first: all 7,639
`.part` files (truncated by construction — never valid data), plus cached copies for
the **59 of 60** source basenames independently confirmed present on `/mnt/v`
(`avifsvt-subsample-2026-09-01/sources`, `imazen-26-hdr-grid-2026-06-14`,
`imazen-26-variants`). The 1 basename that could not be confirmed
(`5309_noaa_nhc-al022024-beryl_p26_2550x3300.scale297x384.png`) and its 8 files were
**left in place**, per the never-delete-unverified rule.

**Not fixed in code, deliberately, and registered here instead.** The jobexec cache
leak is a real defect (unbounded, PID-duplicated, no eviction, leaks `.part` on
error), but the fleet runs a pinned `ghcr.io/imazen/zenfleet-worker:exec-*` image, so
a local source edit would not reach a single remote worker — it would only change
this box while leaving the actual fleet behaviour identical, which is the worse of
the two states to be in. The mitigation applied instead: relaunched local workers get
`TMPDIR=/home/lilith/tmp/zfw-scratch`, which **is** honoured, because the staging path
goes through `std::env::temp_dir()` (verified in source, not assumed). A proper fix —
evict on process exit, share the cache across PIDs, and unlink the `.part` on write
failure — belongs at the owner, with a rebuilt-and-pushed worker image.

## 2026-09-02 — ROUND 52: the DOE gap-closure lane — R51's root cause is FIXED at the owner, and a gate that was satisfiable by a no-op is now un-fakeable

*(Full record: `zenmetrics/benchmarks/avif_doe_plan_2026-09-01.md` §12. One commit,
`ad4d44b3`, verified on `zenmetrics` origin/master.)*

R51 diagnosed the outage and deliberately left the fix at the owner. This round is
that fix, plus the two other gaps the recovery lane recorded honestly rather than
papering over.

**The `/tmp` occupant is fixed structurally, not relocated.** R51's mitigation moved
`TMPDIR` to `/home`, which bought room while the growth continued.
`jobexec::resolve_source_raw` now keys its source cache on **`sha256(resolved URI)`**
instead of `{pid}_{basename}`, collapsing copies from (processes × images) to
(images) — the 46× that produced 22.93 GB from 60 images. A lazily-once
`sweep_src_cache_once()` ages out entries untouched for 24 h (env-tunable), which
also collects R51's 10,413 legacy PID-scoped files **and** the 7,639 orphaned
`.part` files R51 measured; `.part` is now per-writer so a shared `dst` cannot be
interleaved, and a cache hit touches its entry so "age" is time-since-last-**use**.
R51 proposed "evict on process exit"; content-addressing is strictly better — it
also fixes a **latent wrong-image bug** the PID key was hiding, because two corpora
in this very wave hold different pixels under one filename (the DOE plan keeps the
corpus key unchanged across its 1024² crop), and two such fetches inside one
`--serve` process collided. 4 tests, 24 pass.

**A gate that could be cleared by a no-op is the reusable lesson.** The budget-corpus
builder documented `--features-cmd` as satisfying its feature-re-extraction gate and
flipped the manifest off `PENDING` when it was passed — while importing no
`subprocess` and executing nothing. The flag is removed; the extractor is now really
run, and — the part worth copying — **the gate validates itself against data it
cannot fake**: 19 of the 32 references are symlinks, bit-identical to the clustered
parents, so re-extraction must reproduce those parents' own clusters, and a missing,
no-op, wrong-schema or garbage extractor fails that control and exits non-zero.
Measured drift control first (re-extracting the 32 parents moves them **0.0253 max /
0.0012 mean** in the clustering z-space, against decision margins of 0.3–8, so
extractor drift cannot flip an assignment); then the gate: **native control 19/19,
11 of 13 crops preserved their cluster, 2 moved**. `1442` moved out of a **singleton**
cluster at a displacement of 24.41 — cropping removed exactly what made it an
outlier. And `6604` reads PRESERVED at a displacement of **67.72**, so *preserved is
not unchanged*: read `parent_z_dist` beside the verdict, never the verdict alone.

**Two `+10 %`-class findings that only appear when you recheck arithmetic against
what was actually built.** (a) The corpus is 13 cropped / 19 native, not the
registered 23/9 — **AMENDED, not rebuilt**, because 13,532 cells (25,383 twenty
minutes later) were already encoded against those references, cell identity is
content-addressed on the reference bytes (so a rebuild orphans rather than corrupts),
and the deviation costs +10.3 % pixels on a budget the design itself makes a free
variable plus exactly **one** reference's superblock purity. (b) The pre-registered
cross-size transfer gate is **degenerate on the as-built corpus**: 19 of its 32
comparisons are the same encode against itself, so its `T3` median-of-32 has 19 exact
zeros and **cannot fail for any arm**, and its `T1` "80 % of 32" is an effective
7-of-13 (54 %). Registered correction: compute it over the 13 cropped references
only. A pre-registered bar is not automatically a live bar — it has to be rechecked
against the population that actually got built.

## 2026-09-02 — ROUND 53: AVIF-DOE scoring scale-up — the backlog was UNDECLARED, and the first "measurement" measured failure

**Not a zensim-model round — a fleet-ops round on the AVIF DOE, recorded here
because two of its lessons are measurement-hygiene lessons this campaign keeps
re-learning.** Full record: `zenmetrics/benchmarks/avif_doe_plan_2026-09-01.md`
§13.

**The trickle had three causes and the dominant one was not throughput.** With
~36 k encode blobs on disk, score progress read 148 + 256 blobs. Cause: the four
`avifdoe-*` encode runs had **zero** score runs declared — `aws s3 ls
s3://zentrain/jobs/` shows four `avifdoe-*-enc` prefixes and no `avifdoe-*-sf-*`
at all. The 5-minute gap-fill loop everyone assumed was dead was **alive**
(round 51), but its `for backend in svt aom` covers only the *naive* wave's two
runs; it never had a DOE branch to lose. Second cause: the fleet held **one**
score worker, on **2 cores**, serving one of the two declared score runs — while
a 32-thread box ran none. Per-pair cost and store I/O were both measured and
were never the constraint.

**The lesson that belongs in this campaign: a fast number is a suspicious
number.** The first local timing read **6.72 s per 12-pair chunk** and was
entirely fake — every row carried `chooser: no measurements for metric 'ssim2'`
and **zero numeric scores**. Acting on it, a worker wrote **322 `done` ledger
rows over 333 error blobs in under three minutes** and *looked like the fix*:
exit 0, well-formed JSONL, right row count, right blob size, 257 blobs in 134 s.
Nothing failed. The only thing that caught it was noticing the rate was ~13×
what the CPU budget allows and then **opening a blob and counting numeric
scores**. Cause: the local binary was built without `--no-default-features`, so
it carries the orchestrator, whose persistent capability profile has an empty
`[metrics]` table (the box's CPU changed — it now reports a **9950X3D / 32
logical cores**, not the 7950X/28 these docs assume, so its `machine_hash` is new
and the older populated profiles do not apply). `zenmetrics score --metric ssim2`
on the same pair returns `19.796691`, which is exactly why it is hard to see: the
binary is correct everywhere except the path the fleet uses. Resolution: run the
**proven fleet image**, not a speculative rebuild. The 322 poisoned rows were
verified to be one worker's (`{'wsl-score-smoke': 323}`, no other worker in the
run) and cleared, not pardoned — `requeue` keys on failed/poison `error_class`
and `reassert` reverses buried *done* rows, so neither targets a row that is
genuinely `done` over garbage.

**Result.** One new run `avifdoe-svt-sf-cpu-20260902` (A1+A2, ~3.4 k jobs /
~41 k pairs) plus a recurring declaration loop that **fails loud** (heartbeat file
+ error counter, the thing its silent predecessor lacked), and four local workers
inside cpuset `0-23` (8 of 32 threads left free). Score blob rate went from
**~20/h to 4,508/h (3-metric)**, and the mid-lane user directive *"you can skip
butteraugli even"* then **1.9×'d** it again to **8,514/h = 102 k pairs/h** (487 s window). Because `JobId::of(&kind,
&inputs)` hashes the metric list, the metric change could not be an in-place
edit — it is a disjoint job set, so the switch was made **going forward** at a
recorded boundary (907 blobs) and **rows are now heterogeneous**: key on the
metrics present per row, never on a fixed row count. AG stays undeclared on
purpose — it encodes the *native* corpus under the same filenames (§12.3/§12.4),
so declaring it against the crop prefix would have silently scored wrong pixels.

## 2026-09-02 — ROUND 54: AVIF-DOE wave COMPLETE + Stage A — two knobs were never wired, and the plan's own isolation rule made a third of its design undeclarable

**A fleet + analysis round on the AVIF knob DOE, recorded here for three
measurement-hygiene lessons.** Full record:
`zenmetrics/benchmarks/avif_doe_stageA_2026-09-02.md`; plan §14.

**The wave finished: 49,120 cells across four svt-rs runs, live-gap 0, ZERO
failed cells, all scored.** A0R (the control arm every effect is differenced
against) and AG had no encode workers at all when the round opened.

**Lesson 1 — a stale read of a FINISHED run looked exactly like a gap.** The
plan's §13.8 recorded A1 as `live_done 6,432, gap 480, failed-only`, and the
brief for this round carried that forward as work to triage. A1's ledger shows
the run finished at **04:47:21Z, ~57 minutes before that snapshot**, 6,912/6,912
live-done, with **zero post-fix `encoder_panic`** — the §11.7 pardon worked and
laundered nothing. Nothing needed requeueing. The triage that had been queued
was for a gap that did not exist; the ledger, not the report line, is the
evidence.

**Lesson 2 — the fingerprint said the knob changed and the bytes said it did
not.** `tune=0` and `screen_content_mode=Some(3)` produce **byte-identical
bitstreams to the default on 288/288 cells at both presets**, while zenavif's
own resolved-state fingerprint separates them (`525f0219…` vs `98b71736…`).
`tune=3` *does* move the bytes, so tune is partly wired. Consequence beyond the
port bug: **all 27 A2 pairs containing an inert knob are byte-identical aliases
of the other knob's single arm** (verified 288/288 each), so 29 of A2's 118
strata carry no information and **8,972 cells of the wave measured nothing**.
Content-addressed storage is what made this visible at all — it is the same
property that turns "two configs" into one blob. **When a knob axis is
declared, check byte-identity against the control before trusting any effect
size computed from it.**

**Lesson 3 — a registered cell count that its own constraint forbids.** §3.2
registered A1 as "17 arms × all 7 effective presets" = 34,272 cells. At
`--max-deviations 1` a non-default preset **is** the one permitted deviation, so
a knob×preset cell is inexpressible; zenavif's own test fixes the design at
**24 strata** = the declared 6,912. Nobody reconciled the 5× discrepancy for a
day. The load-bearing consequence is not the count — it is that **knob main
effects exist at two presets, not seven**, so the preset-inversion trigger
(B-5) has one preset pair to work with and the slow end is untested.

**Lesson 4 — the pre-registered gate earned its keep, and not through its
headline.** The cross-size transfer gate certifies only **2 of 16** knobs for
reduced-size screening (`mtx32`, `qml1.8.15`), holds direction on 7, and fails
2 outright. Its most valuable output came from its *sign* test: **tiling's
bitrate cost is largely a reduced-size artefact** — `tl1.0` +0.65 % at the 1024²
budget vs **−0.12 % at native** (sign flip), `tl1.1` 8.6× smaller, carrying the
only two significant T3 binomials in the set (p 0.012, 0.001). The main-effects
table alone says "tiling costs bits on every image at both presets, tight CI",
and that statement does not survive a size change. **A knob measured at one size
is a knob measured at one size.**

**Lesson 5 — a "free" measurement that was never free.** §3.9's bytes
decomposition (`total = α + β·pixels`, from running the control at two sizes)
**is not identifiable from a crop/native pair**: a crop is a different image,
not a smaller one, so the two-point intercept absorbs the content difference
instead of a container header. **SROCC(α, q) median 0.943** over 91
(image, speed) groups, α climbing **731 → 59,176 bytes** across the ladder (81×)
and going **negative on 781 of 2,639 fits**. Cropping was chosen over downscaling
to preserve native HF content — right for the transfer gate, fatal for the bytes
model, and the plan did not notice the two goals were in tension.

**Result.** Stage-A tables + the mechanical Stage-B trigger list are published;
honouring every trigger costs **447,636 cells against a registered 60,000
envelope (7.5×)**, so prioritisation is a coordinator decision and **no Stage-B
wave was declared**. Fleet returned to its pre-round state (five encode workers
created, all torn down).

---

## 2026-09-02 — ROUND 55: DOE close-out — a recurring declaration was re-doing its own work forever, and the "drained" run could never drain

**Two bounded cleanups left over from ROUND 54's Stage A.** Records:
`zenmetrics` `08215e84` (fix) + `986899c6` (doc correction);
`zenmetrics/benchmarks/avif_doe_stageA_2026-09-02.md` §1.4.

**The fix.** `zenfleet-ctl pairs` emitted its DONE rows in `HashMap` order, so a
recurring `declare-scorefiles` loop re-minted every `job_id` each round and the
fleet redid finished work. Now sorted on a TOTAL key (cell identity, `job_id`
last as the tie-break — unique by construction, being the `LedgerView` map key).
Verified live: three separate `pairs` processes over a frozen 6,496-row ledger
gave byte-identical `.tsv` **and** `.parquet` (sha `148165c7…`), and two
`declare-scorefiles` runs gave byte-identical manifests (sha `2ce81289…`). Four
new tests; both gate tests confirmed to FAIL with the sort defeated.

**Lesson 1 — the mechanism was mis-stated, and the correct version is what
decides who else is exposed.** §1.4 blamed input order. But `JobId::of` sorts and
dedups its inputs, so member order *within* a chunk cannot move an id. What moves
it is chunk MEMBERSHIP: a permutation re-cuts which members share a chunk, and
**only when a ref has more members than `--chunk`**. This run was maximally
exposed — 32 refs × 203 members at chunk 12, ~17 chunks per ref — while a run
whose refs fit one chunk was never affected at all. It also explains the shape of
the symptom: `declared` stayed pinned at exactly 4,128 while the identities
rotated, because a permutation changes which members pair up, never how many
chunks fall out. Pinned by `only_refs_larger_than_the_chunk_can_remint`.

**Lesson 2 — "drained" was read from a gap that closed every round and reopened
every round.** The brief for this lane, and §1.4's decision to hold the fix, both
rested on the wave being finished and the churn being a bounded historical cost.
It was live. Over rounds 37-40 the encode side sat **frozen** at 49,120 DONE
cells and `declared` stayed pinned at 4,128, with four workers re-scoring
finished cells. Re-measured with the owning tool at the moment the fix landed:
`declared=4128 ever_done=29664 live_done=29664 rescore tax 1.01x errors=0` — the
multiplier had gone **4.0× → 7.19×**, about seven full passes over one pass of
work, and was still rising. **A run in that state never settles**, so holding did
not avoid churn — it extended it.

*(Corrected in the same lane: the first figure published here was **6.7×**, taken
as score-blobs ÷ declared. That is not a multiplier — it happens to look like one
because blobs track distinct completed `job_id`s about 1:1, 29,608 against
`ever_done` 29,664. The defensible ratio is `ever_done ÷ declared`, which is what
the wave's original 4.0× was. Measure the multiplier with the tool that owns it,
not from a blob count that merely correlates.)*

The generalisable form: *live-gap 0 is not evidence of completion when something
re-declares on a timer; check whether the denominator is being re-minted.* Here
`report` printed `VERDICT: COMPLETE — every run live-gap==0` throughout the
churn, because the gap really did close every round — and the next round re-minted
it. **Post-fix, confirmed live:** rounds 43, 44 **and 45** uploaded byte-identical
manifests (sha `aeb915e2…`, `declared=4,128` each), so the id set has stopped
rotating. The loop picked the fix up unaided — it re-execs its binary each round
— so no restart was needed. One caveat for whoever reads the counter next: blob
count keeps rising for one more pass (`ever_done` 29,664 → 30,264) because round
43 minted a brand-new stable set and the fleet is completing its 4,128 jobs
once; every later round re-declares the same set and is a no-op. **Check the
manifest sha, not the slope.**

**Lesson 3 — no restart was needed, because the loop re-execs its binary.** The
gap-fill loops spawn `./target/release/zenfleet-ctl` fresh each round, so the
rebuilt binary was picked up automatically; the `.pid` files held wrapper PIDs
(the live `bash` was a different pid), which is why the loops first read as dead.
Verified by observation rather than by resetting the round counter.

**Task 2.** The Stage-A dead-knob finding (`tune=0` and
`screen_content_mode=Some(3)` byte-identical to baseline on 288/288 cells at both
presets, while `tune=3` moves bytes) is filed as
[imazen/zenav1-svt#17](https://github.com/imazen/zenav1-svt/issues/17) with the
repro, the 8,972-cell blast radius, and one consumer-side lead: only tunes 3/4
rewrite config via the port's `apply_tune_overrides`, and the existing parity
test compares **config to config**, so it cannot catch a field that resolves
correctly and is then never consumed.

## 2026-09-02 — ROUND 56: Stage B opens with B-6 — the two arms that provably do not screen at reduced size, at native

**User decision, verbatim: "go with B-6 first".** Stage B is opened with **one**
trigger. The other 53 (§10 of the DOE plan: B-1 17, B-2 23, B-3 13) stay
undeclared and the rest of the 60,000-cell envelope is unspent. Record: DOE plan
doc **§15**; commits zenavif `43423054` + `386b82f8`, zenmetrics `8d5d3d93` +
`6b3c41fe` + `c1208d29`.

**What B-6 is.** Stage A's cross-size transfer gate certified only `mtx32` and
`qml1.8.15` for reduced-size screening. Two arms — `acb3` (`ac_bias` 3.0) and
`shp3` (`sharpness` 3) — **failed T1**: their direction at the 1024² screening
budget does not carry to native. So leaving them at budget is *known* to be
wrong, which is what makes them the cheapest high-value cells in the trigger
list. Their follow-up runs the B-1 dense grid **at native size**.

**The arithmetic, and why two right numbers disagree.** Registered 27,840
reproduces exactly as `2 × (5 levels × 29 q × 32 img × 3 speeds)`. Declared is
**25,056**. The 2,784 gap is the two knobs' **shared default level** —
`ac_bias` 0.0 and `sharpness` 0 are the *same configuration*, so one axis
carries 9 levels, not 10. The trigger list counts that control block once per
knob; the declaration counts it once in total. Nothing was re-registered.

**Five gates, none assumed.** 32/32 native refs sha-match local (0 mismatches;
13 differ / 19 passthrough vs the budget twin, the as-built split); the
native/crop **filename collision is impossible at CELL level, not merely
avoided** — 6604's declared `source_sha` is `769b0df4` native vs `4ac38273`
crop; declaration is deterministic (two rounds sha-equal); `pairs` is
deterministic (`82aea675` — R55's fix holds, so R55's 4.0× re-declare
multiplier does not apply here); G-FIRSTCELL passed on **both** halves — the
first encode blob is `ftypavif`/`mif1miaf`, `file(1)` AVIF, and *decodes and
scores* (ssim2 59.42), and the first score blob carries 12 ssim2 scalars + 12
zensim 720-wide feature rows.

**Cost measured, not extrapolated** — through `zenmetrics jobexec`, the
worker's own entry point, on the real corpus, before the fleet was scaled:
**13.2 CPU-h encode = 0.22× the 60 CPU-h envelope**, 14.3 GB blobs. Three
things the plan did not have:

- **`ac_bias = 8.0` is not clamped.** `SvtParams::clamped()` clamps the
  variance-boost pair, the QM levels, `max_tx_size` and `sharpness` — but not
  `ac_bias`. The top of the documented range was therefore a genuine
  release-mode out-of-range risk (the **H-10** class). It encodes cleanly and
  *moves bytes* (29,049 vs the default's 28,516), so the level is live, not a
  third inert knob.
- **Preset dominates and cost is super-linear in pixels.** Preset 4 runs
  **1.842 MP/s at 1.57 MP but 0.591 MP/s at 16 MP**, and is **17–27×** presets
  7/9 — 88 % of the wave's CPU. So **cell-count fill overstates real progress**
  (measured 1.28×: 8.02 % of cells was 6.27 % of the work). Report this wave
  work-weighted.
- **Bytes and time have different q shapes** — the 29-q ladder factor is 53–56
  for bytes but 27–30 for time. High-q cells are cheap to compute, expensive to
  store; one factor cannot serve both plans.

**A diagnostic that would have lied.** zenmetrics' "unknown zenavif plan"
message hand-typed the plan list and its own comment called itself a
"human-readable mirror" — it drifted the instant `svt_doe_b6` landed and
omitted it. That message **is** the control arm §11.5 uses to prove a fleet
image is not stale, the check whose absence cost the DOE wave 93 % of its
main-effects arm. A reader running it against a *good* image would have been
told the plan they just declared does not exist. Fixed at the owner: one static
`PLANS` table behind `by_name` + `names()`, so a second copy is no longer
expressible. It was caught only because the control arm was actually run — and
its first run measured nothing (an empty sources dir makes the real and bogus
arms return the same "no source files found", exactly the trap §11.5
documents).

**Two corrections to assumed state.** The corpus has **no 16320×7612 pano** —
32 refs spanning **0.25–16.00 MP**, 161.59 MP total, median 1.57. And the fleet
was **not** idle: `avifsub-aom-{r7900x,tower}` had been up 11 h re-touching
poison on a run already at live-gap 0 (zenfleet's own `idle` waste). Both were
stopped; nothing was destroyed.

**Live at report time (14:14Z):** 3,030/25,056 cells (12.09 %) = **11.27 % of
the work**; effective throughput **8.6 CPU-h per wall-h**; **ETA 1.35 h**
(~15:32Z). Encode on r7900x (`0-19`) + tower (`0-19`, shares 256, 24g — the
tower rule, it is a live media server); scoring on dev (`20-29`) into
`avifdoe-svt-b6-sf-cpu-20260902` (`ssim2,zensim`), fed by the DOE gap-fill loop
now **parameterised** (`ZEN_DOE_RUNS`) so B-6 runs as a second instance of that
loop rather than a fork. B-6 *analysis* is a later lane.

---

## 2026-09-02 — ROUND 57: the AVIF HDR tripwire — a PQ AVIF scored to a plausible number for as long as the path has existed, and the tiled/plain split was the wrong diagnosis

**TODO-0 of the AVIF-HDR-arm plan, a zero-tolerance silent-corruption defect.**
zenmetrics commits `e9e2ef71` (fix + gates) and `cece471e` (plan doc); record:
`zenmetrics/benchmarks/avif_hdr_arm_plan_2026-09-02.md` §3.2, resolution block.

**The defect.** An AVIF whose container `colr`/`nclx` box signalled PQ (16) or
HLG (18) was decoded, narrowed to 8 bits and relabelled sRGB by the
`RowConverter` funnel in `decode::pixel_slice_to_rgb8` — **no error, no
warning**. PNG had refused exactly this since its cICP tripwire
(`decode.rs:151-166`); AVIF had no equivalent, and the second-line zenpixels
`HdrSourceRequiresPeak` guard structurally cannot fire, because
`ManagedAvifDecoder`'s buffered path never calls `descriptor_with_cicp` (only
the row-sink paths do), so the buffer arrives tagged
`TransferFunction::Unknown` and the conversion is a byte passthrough.

**Measured, on the then-current release binary, before touching anything.**
`ref_64.avif` patched to `tc=16` and to `tc=18` each scored
`ssim2=96.137450` — **bit-identical to the sRGB original**, i.e. the transfer
signalling was ignored end to end. Genuine 10-bit PQ files from zenavif's
vectors (`cosmos1650_yuv444_10bpc_p3pq.avif`, `colors_hdr_rec2020.avif`) scored
`100.000000` against themselves. The same PNG content with a cICP PQ chunk was
refused loudly by the existing tripwire — one question, two answers, in one
binary.

**The plan's own diagnosis was half wrong, and measuring it first is what
caught it.** §3.2 predicted the behaviour differed between tiled and plain
files — that a grid-tiled AVIF "is tagged (`sink.rs:290`) and would be handled
correctly", so an experiment could produce correct and corrupt cells side by
side. A real 1×5 grid AVIF (`sofa_grid1x5_420`) patched to `tc=16` scored
**silently too** (`100.000000`): `sink.rs:290` is the *row-sink* grid path,
while `decoder.rs::decode_grid` — the one the buffered decode actually uses —
does not tag either. **Both shapes were corrupt.** That is a worse defect and a
*simpler* fix: there was no correct-for-grid behaviour to preserve, so one guard
serves both. Had the tiled/plain split been taken on trust, the fix would have
been designed around a divergence that does not exist.

**The fix, and what bounds it.** `decode_avif` now takes
`ManagedAvifDecoder::decode_full` — the same decoder `decode_with` already
selected here and the one `sweep::hdr::decode_avif_to_nits` drives, so decoded
pixels are unchanged — and refuses on an HDR transfer, naming the code, the
transfer and the **real bit depth** and pointing at `--hdr`. `decode_full`
branches on `grid_config()` internally and returns an `ImageInfo` either way, so
the single guard site covers both shapes by construction. ~19 non-comment lines
of production change; no public API change; no zenpixels or zenavif change.

**The guard is deliberately narrow — only 16 and 18, and that is load-bearing.**
Narrowing a 10-bit *SDR* AVIF to 8 bits is `decode.rs`'s documented contract and
the 8-vs-10-bit SDR track (`bd10`) depends on it; BT.2020's SDR transfers (14,
15) sit adjacent to PQ/HLG in the CICP table. An over-broad guard would have
refused live sweep cells. SDR non-regression was measured pre-fix vs post-fix
binary — **identical output** on 8-bit AVIF vs PNG, the `tc=1` variant,
grid-tiled SDR, and genuine **10-bit SDR** AVIFs (`plum-blossom` profile0 10bpc
4:2:0 and profile1 10bpc 4:4:4).

**Gates.** `zenmetrics/crates/zenmetrics-cli/tests/avif_hdr_tripwire.rs` —
`pq_transfer_is_refused` and `hlg_transfer_is_refused` **fail at the parent
commit** with the recorded message *"expected a loud refusal, got a silent 64x64
decode (12288 bytes)"*; plus `committed_sdr_fixture_decodes`,
`patching_transfer_alone_does_not_change_pixels` (pins that the fixture helper
perturbs signalling only) and `refusal_is_scoped_to_hdr_transfers_only`. In
`decode.rs`: `only_pq_and_hlg_are_refused` pins the policy over **all 256**
transfer codes, and `bt2020_sdr_transfers_are_not_hdr` pins the two neighbours.
Fixtures are built in-test by rewriting one `u16` in the committed 445-byte
`ref_64.avif`; nothing binary was added to git.

**What it does NOT do — stated so no one over-reads it.** It refuses; it does
not route. A cell mis-sent to the SDR path now fails loudly instead of returning
a number, which is what makes the arm's **G5** satisfiable — but G5 still
requires *positive* evidence in the run log that the PQ `--hdr` route was taken,
because a refusal proves only that the wrong route was rejected.

---

## ROUND 58 — the issue-17 recheck: `tune=0` still dead, `scm=3` was never a bug, and a third arm nobody swept

**Trigger.** The user's read that zenav1-svt#17 "might be fixed". It is not — but
the recheck splits the issue three ways and **one third of it retracts a claim
this campaign filed**, so the DOE's own conclusions move.

**What was probed.** `zenav1-svt` at `0284b855c` (308 commits past the
`30cf4b3d0`-era binary the wave encoded with, including `188948556` sc-detector
tier-1 and `b8e5e1c11` tune-vmaf), then re-run at `6fe01232` — **byte-identical
TSV both times**, so nothing in between moved these arms. 288 encodes at the
**port boundary**, not through zenavif: `EncodePipeline::new(w, h, preset, Cqp,
0, 1).with_chroma_420(true)`, `hdr = mainline()`, exactly ONE field overwritten
— the construction `encoder_svt_rs.rs:690` + `apply_svt_params` uses, pinned by
zenavif's own `svt_params_default_leaves_the_pipeline_at_mainline`. 3 content
classes × presets **4, 6, 8** × qp 20/32/45/55 × 8 arms. Landed as
`rust/svtav1/examples/knob_byte_identity.rs` (zenav1-svt `70883fbe8`) — the
bitstream-level assertion §3 said would have caught this.

**Positive controls carry the null**, which is the whole reason to believe it:
`tune=3` **0/36** identical, `sharpness=7` **0/36** identical. The probe is not
blind.

| arm | p4 | p6 | p8 | verdict |
|---|--:|--:|--:|---|
| `tn0` (mainline) | 12/12 | 12/12 | 12/12 | **still dead** |
| `fork_tn0` (same delta, `hdr_fork()`) | 4/12 | 0/12 | 0/12 | moves on 32/36 |
| `scm3` | 12/12 | 12/12 | **8/12** | live at p8 |
| `scm0` | 12/12 | 12/12 | 12/12 | **dead everywhere** |

**1. `tune=0` — NOT fixed, and now localised to one predicate.** The `tn0` and
`fork_tn0` rows differ in exactly one thing: `if self.hdr.is_fork()` guarding
`tune::lf_sharpness_for_tune` in `pipeline.rs`. Walking every `tune` read in
`svtav1-encoder`, that is the **only** site that can separate tune 0 from tune 1
— each other one keys on `TUNE_IQ`, `TUNE_MS_SSIM`, or `tune_uses_ssim_rdmult`
(`SSIM|IQ|MS_SSIM`), all of which exclude both (`hdr_mode.rs:358,369`,
`chroma_q.rs:89,122`, `pipeline.rs:2322,2382,2546,2578,2977,2985,10020`,
`pd0.rs:252`, `md_config.rs:927`, `mds3.rs:2843`). The fork arm's differences
are **same-length, different-bytes** (photo p4 q20: 2846 B → 2846 B), the
signature of a deblocking-strength change — which is what `TUNE_VQ ⇒ min(7,
sharpness+2)` should produce. Whether the fix is dropping the gate or
documenting 0 ≡ 1 as faithful turns on whether `deblocking_filter.c:1157`'s VQ
arm is mainline v4.2.0; **not settled** (the C submodule is not checked out
locally), so it is filed as a lead with a measurement behind it, not a
diagnosis. The adjacent `apply_tune_overrides` call site carries a comment
saying this exact `is_fork()` gating was already a bug once and was fixed there.

**2. `screen_content_mode = Some(3)` is NOT a bug — §3's filing was wrong, and
this is the retraction.** It reaches the bitstream. It is inert at presets ≤ 7
because C's allintra default is **already 3** there (`sc_detect.rs`:
`Allintra => preset <= 7`; `Some(3) => preset.min(7)` is the identity at 4 and
6). At preset 8 it moves screen content hard — 2,898 → **930 B** at q20, 2,186 →
833, 1,615 → 833, 1,984 → 807 — while photo and detail stay identical because
the detector correctly finds no screen content in them. **The DOE swept presets
4 and 6 only, which is exactly the range where the knob is a semantic
identity.** The wasted cells are real; the cause is the sweep design, not the
port. Consequence for Stage B: `scm3` is **not** a dead arm to drop — it is a
preset-≥8 knob, worth nothing at 4/6 and worth measuring above.

**3. NEW: `screen_content_mode = Some(0)` is genuinely unplumbed** — 36/36
identical at **every** preset including 8. The match is `Some(3) => …, _ =>
preset`, so `Some(0)` falls through the wildcard and is indistinguishable from
`None`; at presets ≤ 7 it fails to *disable* the tools. Never swept, so it cost
this wave nothing — but it is the real instance of the bug class §3 opened on.

**No rerun declared.** The task's rerun branch was conditional on the arms
moving; `tn0` does not, and `scm3` does not at the swept presets, so
re-declaring the 27 aliased A2 arm-pairs would reproduce the same 8,972
byte-identical cells at the same cost. The fleet stays on B-6/T1/T2. The
original §3 measurement **stands unamended** as a fact about the `30cf4b3d0`-era
binary — only its interpretation of `scm3` changes.

**Ancillary, same commit** (`70883fbe8`): `bd10.rs`'s header claimed *"UNWIRED
(add `pub mod bd10;` when integration starts) … no build run yet"*. It described
the 2026-07-17 bulk-write directive and was never updated — `lib.rs` declares
the module, `cdef.rs` and `quant.rs` read it, `tests/c_parity_bd10_quant.rs`
gates it. Corrected against those call sites; the "inert for bd8" line sharpened
to *byte-inert by construction* (`qzbin_factor`'s `8 => 148` arm reproduces
`build_quant_table`'s own threshold), which is a different claim from
"unreached".

**Evidence.** zenav1-svt#17 comments
[5511181545](https://github.com/imazen/zenav1-svt/issues/17#issuecomment-5511181545)
+ [5511234180](https://github.com/imazen/zenav1-svt/issues/17#issuecomment-5511234180);
issue left **OPEN** on the `tune=0` gate and the `scm=0` wildcard. Reproduce:
`cargo run --release -p zenav1-svt --example knob_byte_identity -- <outdir>`.

---

## 2026-09-02 — ROUND 59: the AVIF high-bit-depth arm, Track T1 — why `bd10` had no s6 number was structural, and two of the registered cell counts were wrong

> **Numbering note.** This row is **59**, not 58: a concurrent lane landed its
> own ROUND 58 (`d29e97ff`, the issue-17 recheck) while this one was being
> written, and it was renumbered on discovery. ⚠ The commit that first added
> this row (`bf7284ce`) still carries **"ROUND 58" in its subject line** — a
> `jj squash --use-destination-message` kept the parent's message, and the
> corrected message landed on a commit that lost the race to the bookmark. The
> **file** has always said 59; only that one subject line is wrong, and it is
> left rather than force-pushed.

**Track T1 of the AVIF-HBD arm plan declared, gated and staged.** zenavif
`bcd79789`; zenmetrics `32e68a8f` (declare), `c863fd30` (the G3 gate tool),
`e6959efe` (T2 picks), `8670ea55` (execution record). Record:
`zenmetrics/benchmarks/avif_hdr_arm_plan_2026-09-02.md` **§10**.

**The finding that made the block cheap to build.** Stage A recorded `bd10` as
"the worst-covered arm in the wave — no s6 main effect, no interaction coverage
and no transfer evidence," which reads like an omission. It is not: it is
**forced by the axis layout**. `svt_doe_main` carries
`bit_depths = [Auto, Ten]`, so under `with_max_deviations(1)` a 10-bit cell has
**already spent its one deviation on depth** and can only be emitted at the
default speed. No amount of re-running A1 would ever have produced a `bd10` s6
cell. The fix is one line of axis algebra rather than a new deviation budget:
pin `bit_depths = [Ten]`, which makes `Ten` **index 0** and therefore costs
**zero** deviations (`cross()`'s `idxs` array contains the bit-depth index),
freeing the speed — or the knob — to be the isolated deviation. Three plans
follow from that: `svt_doe_t1_bd10_ladder` (7 speeds × 1 arm),
`svt_doe_t1_bd10_knobs` (15 live arms at s6), `svt_doe_t1_bd10_transfer`
(s4, native).

**"15 live single-deviation arms" means the default plus 14, not 15
non-default.** The knob set holds 17 configurations, 16 of them non-default,
and Stage A proved two byte-inert (`tn0`, `scm3` — 288/288 identical to the
default). 16 − 2 = 14 live non-default arms, so the registered 4,320 =
15 × 9 × 32 only closes once the default stratum is counted as one of the 15.
The live set is expressed as a **filter over `svt_doe_knob_sets`** rather than a
re-listing, so a level added to the owner arrives automatically — the opposite
choice from `svt_doe_b6_knob_sets`, which needed different levels and had to
re-list.

**Two count corrections, both measured at declare time, neither anticipated.**
6,432 job ids declared (matching the plan's block sum exactly) but only **6,087
DISTINCT cells**. 288 of the gap is the s6 knob-default stratum, which the plan
counts three times over. The other **57** is the interesting one: **19 of the 32
corpus images are sub-budget passthroughs**, so their "native" and "1024² budget"
files are byte-identical, share a `source_sha`, and therefore share a `CellId`.
Stage A had measured the same 19/13 split and *relies* on it (A0-native ≡ A0R on
3,857/3,857 cells) — but nobody had carried it forward into a native-vs-budget
cell count. **The consequence is not bookkeeping: T1-d's cross-size transfer
gate has n = 13 IMAGES, NOT 32.** On the 19 passthroughs there is no size
transfer to measure. Q4 ("does −1.02 % survive native size") is answerable only
at that n, and any statement that does not report it is overstating its sample
by 2.5×.

**G3 — the gate that exists because a request is not evidence.**
`AvifEncoder::with_bit_depth` was documented to coerce unknown depths to 8
**silently**, so a typo'd depth yields a valid 8-bit encode labelled 10-bit and
every BD-rate measures nothing. `avif_depth_verify` answers it from the stored
blob with **three independent reads** — av1C box, AV1 sequence header, decoder
`ImageInfo` — and treats disagreement as a FAIL on the blob's own evidence, with
no expected depth supplied. Verified on **207 conformance vectors** (zero
mismatches, zero disagreements) and, crucially, **watched to fail** on five
negative controls: 8-bit and 12-bit blobs against `--expect-depth 10`, a
byte-identical `--control`, an empty directory, and an **av1C patched to claim 8
while the sequence header still says 10** — which fails without any expectation
flag and proves the three reads are genuinely independent, since the container
flipped and the bitstream reads did not. On the arm's own first cell: **7/7 at
depth 10**, all three reads agreeing, and **0/7 byte-identical** to the 8-bit
control, which itself reads depth 8 on all three.

**A hazard that may have already been retired, read from source and NOT
measured.** H-BD-3's stated mechanism looks stale at current `zenav1-svt` HEAD:
`svtav1/src/avif.rs:218` and `lib.rs:135` both read
`{ self.bit_depth = depth; self }`, with a doc comment saying the function
"deliberately does NOT coerce … It used to". The plan cites the knob dossier
§605 for the coercing behaviour. This does **not** retire G3 — the
zenavif/zenrav1e arm reaches depth by a different path, the new refusal was
never exercised end to end, and G3's byte-identity half is untouched either way.

**Track T2 is blocked, and the blocker is a wiring gap the plan did not
record.** `validate_hdr_sweep` admits only `Zenjxl` and `Zenavif`; `zenav1-svt`
is not a `CodecKind` at all, and the only route to `HdrCodec::Zenav1Svt` is
`from_name`, whose sole caller is the fleet path — whose ssim2/zensim scoring is
the **u8 shell** gate G5 forbids for T2. So T2-a's two requirements (svt
encodes, f32 scoring) are individually reachable and **jointly are not**, which
is precisely what TODO-4 exists to fix. A concurrent capability lane owns that
file and was writing it during this execution, so T2-a is **sequenced, not
dead**. T2-b was **deliberately not smoke-tested**: it calls into the same
in-flight file, so a G5 route assertion produced from this tree would be
evidence about a tree that has never existed on `master` — worse than no
measurement, because it looks like one. **NOT MEASURED — never a null, never a
zero.** T2's corpus gates are complete and committed regardless (76/76 G0.2;
K=16 picks at 6 BT.709 / 10 P3, deterministic across three runs, sha256
`fb805707…`).

**Scope on every number this arm will produce (H-BD-4):** at presets ≤ 8 the
zenav1-svt port is not byte-identical to C SVT-AV1, so T1 measures **this
port's** 10-bit encoder and no result may be stated as a property of SVT-AV1.

# ROUND — bit-depth capability lane (`claude-bitdepth`), 2026-09-02

| field | value |
|---|---|
| **round** | bit-depth capability layer under the AVIF HBD arm |
| **directive** | "identify and fix what is needed for least-bitdepth lossy encoding" (2026-09-02) |
| **scope** | make lossy encode + its evaluation honest across 8/10/12, so depth can eventually be a picker knob |
| **repos** | `zenav1-svt` (1 commit), `zenmetrics` (4 commits). `zenavif` surveyed, deliberately unmodified |
| **commits (all VERIFIED on their remote)** | `6fe01232` zenav1-svt/main · `7051921a`, `0155c165`, `fdce651e`, `58e10310` zenmetrics/master |
| **coordination** | shared checkouts with the live `claude-hbdexec` lane; additive `.workongoing.bitdepth` markers, file ownership negotiated in both directions (their marker names `crates/zenmetrics-cli/src/hdr.rs` as mine), every commit `jj split` to my files only |

## Outcomes

| gap | outcome | evidence |
|---|---|---|
| **b** silent `with_bit_depth` coercion | **FIXED** | `avif.rs:207` stores verbatim; the existing encode-time guard refuses 8\|10 violations as a typed error. `unsupported_bit_depth_is_not_silently_coerced_to_8` (0/1/7/9/11/16/255) FAILS at the parent |
| **a** aom-rs hardcoded `bd: 8` | **FIXED** | `bd` knob 8/10/12; refused BY NAME outside `--cpu-used {0,7,8,9}`; bit-replication promotion; **bitstream read-back at 8/10/12 via `av1C`** |
| **d** fleet scoring depth-blind | **FIXED** | `score-pairs` / `batch` / jobexec now take the umbrella's f32 feeding; u8 shell retained as fallback only |
| **c** `EncodeBitDepth::Twelve` | **REGISTERED, not added** — would lie on `encode_rgb16` today, and is a 0.1.7→0.2.0 semver break | capability matrix §3 |

## Key measurements

- **The u8 shell erases 94.17 %** of a 10-bit-vs-8-bit difference (99.75 % of f32
  samples differ; 5.82 % of u8 bytes do). Pre-fix all four metrics were **bit-identical**
  to the shell.
- Post-fix deviation from identity, shell → f32: ssim2 **1.65×**, zensim **1.21×**,
  iwssim **1.29×**, butteraugli **3.03×**.
- Cost: the f32 route is **≈2× per pair** at every size (64² → 2048², paired arms in
  one binary).
- **12-bit AVIF encode works end to end** through zenav1-aom + zenavif-serialize,
  verified from the emitted `av1C`.

## Carried forward (in the matrix, not in a handoff)

1. ⚠ **NEW DEFECT**: `zenavif::encode_rgb16`/`encode_rgba16` ignore `config.bit_depth`
   → `bit_depth: Eight` + a 16-bit buffer is a silent 10-bit file. Published path,
   reachable from the generic zencodec route. Registered with the reason it was not
   fixed under a concurrent lane.
2. ⚠ **SCORING-ERA BREAK**: stored `--hdr` ssim2/zensim/iwssim/butteraugli-CPU numbers
   from before `7051921a` are u8-shelled — do not join across it.
3. ⚠ `--hdr-transfer` is now **inert** on the faithful route (pinned by a test).
4. zensim HDR **feature** vectors still use the v1 PU21 u8-shell regime — a DATA
   decision, deliberately not made here; the scalar now follows the feature path's
   regime so a sidecar row cannot disagree with itself.
5. Environment: `zenav1-svt`'s `tier_invariance` corpus test wants
   `$ZENAV1_CORPUS_ROOT/gb82-sc/graph.png`; the corpus is at `~/work/codec-corpus`,
   not `~/work/zen/codec-corpus`. Fails before and after this lane's change.

## The honest state of the goal

Encode is depth-honest (8/10/12 reachable, gated, read back from the bitstream).
Scoring is depth-honest (f32 end to end on every route). **The corpus is not**: every
depth cell today is 8-bit content at a deeper coded depth (the `Rgb8Image` funnel), and
the wired HDR references are gain-map-reconstructed from 8-bit bases. **A depth picker
cannot yet be trained honestly** — that is now a corpus problem, not a wiring one.

# ROUND — av1C container-metadata lane (`claude-av1cfix`), 2026-09-02

**One defect, found by another lane on real fleet artifacts, fixed at its owner.**
The HBD executor's G3 pass over `avifhbd-t2a-20260902` blobs found an `av1C` box
claiming `seq_profile = 1` / `ssx = ssy = 0` (4:4:4) over an AV1 sequence header
that says `seq_profile = 0` (Main, 4:2:0-only) — not a legal pair
(`zenmetrics/benchmarks/avif_hdr_arm_plan_2026-09-02.md` §10.4c). The encode is
genuinely 4:2:0 10-bit, so the **bitstream is right and the container lies**. This
is the wrong-metadata-consumers-trust class: our own decoder reads the sequence
header and scores fine; a strict consumer keying on `av1C` mis-handles the file.

**The owner was not the call site.** `zenavif-serialize`'s
`Aviffy::build_color_ipma` (`zenavif-serialize/src/lib.rs:750`) took the `av1C`
profile from the caller's `min_seq_profile` — **default 1** — and the chroma from
`chroma_subsampling` — **default `NONE` (4:4:4)** — and never looked at the payload
it was handed. Two of the three call sites in the workspace never set either:
`zenmetrics` `sweep/hdr.rs:614` (the HDR svt arm, the one that was measured) and
`sweep/encode.rs:1397` (the aom-rs SDR port arm, **a second instance nobody had
reported** — it was never G3'd for profile). The third,
`zenavif/src/encoder_svt_rs.rs:725-727`, states both explicitly, which is exactly
why the SDR svt blobs and the whole T2-b arm muxed correctly from the same port.
**A default that produces an illegal file when a caller forgets is the defect; the
forgetting is the symptom.**

**Fix = derive, not patch a literal** (`zenavif ae9a354f`, 95 production LOC).
`stream_seq_profile` walks the OBUs to the first `OBU_SEQUENCE_HEADER` and reads its
first three bits (AV1 5.5.1) — the whole parse needed, because AV1 5.5.2 then
*determines* chroma for the two profiles that admit one format only: profile 0 is
4:2:0-or-mono (`ssx = ssy = 1`), profile 1 is 4:4:4 and never mono. Profile 2
(4:2:2, or any 12-bit) is genuinely ambiguous and still takes the caller's
subsampling, as does a payload with no readable sequence header — that fallback is
what keeps every previously-correct mux byte-identical.

**Measured, not argued.**
- Failing-test-first: `av1c_profile_and_chroma_follow_the_sequence_header_hdr_420_10bit`
  fails at the parent with `left: 1, right: 0` — the exact T2-a signature — and
  passes after. Five more cases pin the converse (a real profile-1 payload must
  still read profile 1), the explicit-profile SDR call-site shape, monochrome, and
  the no-sequence-header fallback. `test_seq_header_obu` hand-builds spec-valid
  reduced-still-picture headers, so the assertion is against a real bitstream.
- **The other lane's own G3 tool, on a mux of a real 10-bit 4:2:0 conformance
  payload through the `hdr.rs:614` shape:** `chroma` **444 → 420**, `seq_profile`
  **0 on both sides**, and `av1c_depth`/`seqhdr_depth`/`decoder_depth` **10/10/10
  unchanged** with `decoder_transfer = 16`. Both blobs PASS `--expect-depth 10` —
  which is precisely why **G3 passed on the defective blobs and still stands**, and
  why a depth-only gate could never have caught this.
- **SDR byte-identity:** six representative muxes (explicit-profile call-site shape,
  full container metadata, monochrome, alpha, 4:4:4, and both no-sequence-header
  fixtures) hash to `1bb2ffef62e6c7a3ca98e728858048a0e1847f593eab752858e020b1202be9a4`
  **on both sides of the fix**. Adding the defect shape moves the hash. The change
  touches exactly the broken shape and nothing else.
- `cargo public-api`: **0 of 418 items differ.** `set_seq_profile` survives as the
  profile-2 hint and the no-sequence-header fallback. zenavif-serialize is published
  at **0.1.4** and the manifest is already pre-bumped to 0.2.0, so the behaviour
  change needs no new version pressure.

**Blast radius: ANNOTATE, do not re-encode.** `avifhbd-t2a-20260902` (3,248 cells)
carries the bad `av1C`; `avifhbd-t2b-20260902` does not (it routes through
`zenavif::AvifEncoderConfig`). The AV1 payloads and every score derived from them
are **valid** — decoders read the sequence header — so re-encoding buys nothing.
No consumer-facing use exists: no reference outside the plan doc in zenmetrics'
`scripts/` or `benchmarks/`, no `site/` reference, no `/mnt/v/output` publication
directory. They are internal DOE artifacts. Annotated in place at
`zenmetrics/benchmarks/avif_hdr_arm_plan_2026-09-02.md` §10.4c (`10665c6e`) and in
`~/work/zen/DATA_PROVENANCE.md`'s t2 entry.

**Forward rule** (beside the executor's existing "future T2 image must re-run G3"):
the fix rides in automatically on any image rebuild — `zenavif-serialize` is a
**path** dep of the zenmetrics workspace (`Cargo.toml:362`) — so a worker image
built from source on or after `ae9a354f` mints correct `av1C`. **A rebuilt image's
G3 should assert profile/chroma agreement as well as depth**: require the `chroma`
column to read `420` with `seq_profile` `0` on the svt HDR arm. The defective blobs
passed the depth-only gate; a gate that only checks depth cannot see this class.

**Not done, deliberately:** the two zenmetrics call sites were left alone. With the
derivation in the muxer they need no setters, and editing sweep code under a running
wave buys nothing. The **alpha** `av1C` (`lib.rs:861`) still writes profile from
depth alone — correct by construction (alpha is monochrome, so profile 0 or 2 with
`ssx = ssy = 1` is the only legal shape), so it was not changed.

---

## 2026-09-02 — ROUND 61: the imazen-only correction lane — a C oracle was the encoder, the probe AND the gate in the aom tuning path, and ImageMagick was inside model selection

USER RULE **"IMAZEN-ONLY IMAGING/CODEC SOFTWARE"** landed in
`~/work/zen/CLAUDE.md` this round: never reach for imaging/codec software not
written by imazen — as encoder, reference, admission gate, or probe —
**especially in a pipeline that develops predictive models designed to tune
imazen software.** C references are for differential port validation **inside
the port repos only**. It was minted after a session wrongly declared
zenav1-aom *"validation-only, not a backend"*. Four findings, three commits in
zenmetrics (`6f1f1f22`, `cbc2ab00`, `b590aa6d`) and one here.

**1. The claim that started it was wrong, and its likely SOURCE is a stale
comment.** zenav1-aom **is** a wired decode backend — `DecodeBackend::AomRs`
behind `aom-backend`, reaching the public API for stills, alpha, grid,
animation, gain-map, 8/10/12-bit and mono, with 15 tests. The trap is that
zenavif has **two** backend enums: `Av1Backend` (encode) and `DecodeBackend`
(decode). aom lives in the second, so checking the encode enum gives a true
negative to a question you did not mean to ask. And
`zenavif/src/decoder_managed/aom.rs:25-27` still claims *"grid images and
animation return honest `Unsupported`"* — **false**, contradicted by code 50
lines below it and by two passing tests. It is the first thing a reader hits.
**Date correction:** PR #31 merged **2026-08-11**, not 2026-07-13 (that is the
branch commit); anything dating these seams "on main in mid-July" is off by a
month. Full truth table + 5 doc-vs-source discrepancies:
`zenmetrics/benchmarks/bitdepth_capability_matrix_2026-09-02.md` §7.

**2. In the aom sweep arm the C oracle is not merely a gate — it is partly the
encoder and wholly the probe.** MEASURED at
`zenmetrics-cli/src/sweep/encode.rs:1307-1380`: every aom-rs cell runs a full C
libaom encode as its **sequence/frame header source**, reads the
**screen-content decision out of the C stream** and feeds it to the port's
`ToggleKnobs` (so C's detector chooses the *port's* coding tools), refuses
unless the payloads match byte-for-byte, then **splices the port's payload into
the oracle's OBU frame**. Root cause is an API-shape mismatch, not a policy
slip: zenmetrics drives the port through `aom-bench`, which is the port's
**differential-validation harness**, and every `port_encode*` takes a C-encoded
bootstrap **by signature** (`aom-bench/src/lib.rs:1150,:1176`). **So deleting
the byte-identity compare would not purge C** — the header would still be C's.
The aom arm cannot be de-oracled at the zenmetrics layer at all.

**3. The HBD refusal was keyed on the wrong thing, and the port's own record
says so.** The `bd>8` × `cpu-used` 1..=6 refusal was justified by divergence
from C. But zenav1-aom files that band under tier **T4 — "measured, pinned,
unlocalized (byte divergences, NO REFUSAL)"** (`zenav1-aom/CLAUDE.md` T4 row
`HBD_OPEN`/`b10_64`; pinned set `aom-bench/tests/s4cov_qm_axis.rs:380`): the
port **encodes** there and does not call its own output invalid. Neither branch
of the brief's if/else applied — the refusal **stays**, because finding 2 makes
it structurally forced at *every* depth, but it is now keyed on the port's own
pin plus the harness constraint, with C-parity demoted to metadata and an
explicit disclaimer that the port's HBD encode is not claimed wrong.
Failing-test-first: the pinning test went 2 → 4 assertions and was observed
failing on the old message for exactly that reason.

**4. ⚠ ImageMagick was inside MODEL SELECTION, and its removal is an era
break.** `scripts/run_full_eval.sh` shelled `magick` for **both** M3 axes —
`-filter Mitchell -resize` and `-quality Q` (ImageMagick's libjpeg) — and M3a
is a first-class selection input (`WAVE_PLAYBOOK` step 6; the `freeze_check
--select` tie-break is `balanced_composite + 0.15·M3a`). It also carried a
**graceful skip**: with no `magick` on the box the size axis silently collapsed
to 576-only, so **M3a changed meaning without failing**. Now
`zensim-bench/examples/m3_fixture_gen.rs` (zenpng + zenresize `Filter::Mitchell`
+ zenjpeg), and a missing generator is **fatal, exit 3**.

**MEASURED era hazard — the reason nothing was regenerated.** zenjpeg's `q` is
not ImageMagick-libjpeg's `q`. Same `city_384`, same nominal quality:

| q | ImageMagick-era B | zenjpeg B | ratio |
|--:|--:|--:|--:|
| 20 | 11,852 | 10,752 | **0.907** |
| 50 | 20,513 | 16,307 | **0.795** |
| 75 | 30,113 | 24,897 | **0.827** |

The Mitchell downscales differ too (241,630 vs 241,828 B at 384; not
byte-identical at either size). A fixture from the new owner is a **different
rate point**, so an M3a measured on it is not comparable to any M3a in the
record. The 48 ImageMagick-era fixtures under
`/mnt/v/output/zensim/diffmap-coherence-2026-07-18/` are therefore **left
alone**, the default `$FIX` still points at them, and the rewired loop only
fills a **missing** file. **VERIFIED:** with the generator deliberately absent
the loop needs **zero** regenerations and the fixture-set digest is
**unchanged** — no published M3/M3a number moves. Regenerating means a **new
era-stamped directory** (same discipline as `2026-05-15-full-features` vs
`2026-08-30-full-features-372`) and expecting the axis to re-base.

**A3 is REDESIGNED, not declared** (`zenmetrics/benchmarks/avif_doe_plan_2026-09-01.md`
§16): arms drive zenav1-aom's own encoder over the **port's** knob surface (so
`tune=iq` and `deltaq_mode2/3`, excluded only because C could not be driven to
match, become eligible); the port's emitted bitstream is ground truth; validity
is the port's **own** decode-verify (`aom_decode::frame::decode_frame_obus` +
`zenavif-parse` read-back) — a stronger product statement than byte-equality
with C, which never proved a stream decodes. G-AOM-BASE/G-AOM-ARM are
**withdrawn as tuning-data gates** and re-registered unchanged as port-repo
checks. **Blocked on `PREREQ-AOM-STANDALONE`** (zenav1-aom exposing an encode
entry that derives its own header) — owned by zenav1-aom, not built here. Until
then the draining A0 aom arm is **re-labelled, not discarded**: valid
port-parity evidence, **not admissible tuning data** for a model that tunes
zenav1-aom. The **svt arm has no C anywhere in its path** and now carries the
DOE as the load-bearing plan rather than a fallback.

**Tool sweep:** 31 hits across the campaign paths (12 live, 8 peripheral, 5
dead, 6 mention-only). The Rust sweep dir is **clean**. Two fixed (the M3 path
above; and `zenmetrics/scripts/sweep/CLAUDE.md`'s *"libjxl is the authority…
always test with `djxl` directly"* — a committed instruction that reproduced the
violation on every read, now scoped to port-repo triage). Nine registered with
owners (§8.2), seven of which share one missing piece: **a `zenmetrics image
{probe,decode,resize,encode}` subcommand**, the highest-leverage build on the
list. Two deliberately not drive-by: the DoE budget-corpus PIL decode would
change every `crop_sha256` under a **live** wave, and `synth_nonphoto.py`
**rasterises** training sources with PIL/matplotlib — imazen has no rasteriser,
so that needs a decision, not a swap. Trap registered before it bites: the
frozen `asrun/{avthdr,hdrvdc}` ffmpeg drivers are the only recorded route to
re-extract those HDR-video reads, and imazen has **no demuxer and no HEVC/VVC
video decoder** — that study is currently un-re-runnable under the rule.

# ROUND row — blessqueue lane (2026-09-02)

Written to ~/tmp rather than committed: the zensim `.workongoing` was NOT free
(foreign `claude-imazenonly` marker, path-scoped coexistence used), per brief.

| round | lane | outcome | artifacts | verified |
|---|---|---|---|---|
| 2026-09-02 blessqueue | 3 user decisions: bless frozen ffmpeg reads; queue PREREQ-AOM-STANDALONE; open its issue | ALL 3 LANDED. Provenance corrected on 2 points while verifying (codec facts; SCM-detector framing) | zensim `47a7233c` (eval_annotations entry `extreads-ffmpeg-blessed-avthdr-hdrvdc-2026-09-02` + `scripts/external_reads/asrun/README.md`); zenav1-aom `64f3cdb8` (CLAUDE.md coverage-queue T3 row); issue imazen/zenav1-aom#15 | both commits confirmed ancestors of their `origin/main`; annotation load exercised via `freeze_check --profile balanced-2026-08-04` (surfaces as 1 of 30 documentation-only findings) |

## Corrections made against the brief (measured, not assumed)

1. **"both domains ship only as HEVC-encoded video" — FALSE on both counts.**
   HDR-VDC ships **AV1** (SVT-AV1 v1.5.0 preset 4, 10-bit yuv420p limited range;
   `hdrvdc/PROTOCOL.md:63`), decoded via libdav1d (`:96`). AVT-VQDB-UHD-1-HDR ships
   **three** coded codecs — 65 av1 `.mkv` + 65 hevc `.mp4` + 65 vvc `.266` = 195
   bitstreams — plus FFVHUFF lossless `.mkv` references (`avthdr/PROTOCOL.md:68`),
   and needs **two** pinned ffmpeg builds because system 4.4.2 has no VVC decoder
   (`:131-137`). Also corrected in `~/work/zen/CLAUDE.md` line 41 (not a git repo,
   so no commit); the blessing text itself is unchanged.
2. **ffmpeg's role is wider than demux+decode** — swscale also does the bt2020
   tv->full csc to rgb48le AND a Lanczos a=3 resample to the 4K display frame
   (+ a 1920x1080 far-viewing leg for hdrvdc). A foreign colour converter and
   resampler are in the chain; both registered common-mode across legs.
3. **"C's SCM detector drives the port's ToggleKnobs" — the detector is ALREADY
   PORTED.** `estimate_screen_content_antialiasing_aware` runs at
   `aom-bench/src/lib.rs:1904` but is only ASSERTED equal to the bootstrap-parsed
   `p.allow_screen_content_tools` (`:1914`); the value threaded into the search is
   the bootstrap's (`:1720`, `:2044`, `:2069`). It is a differential gate, not the
   driver. Narrows the standalone work.
4. **"port payload is spliced into the oracle's OBU frame"** — the port DOES
   re-serialize the frame header itself (`:2487`, `:2494`; the multi-tile path
   says "nothing is spliced from the bootstrap"). What is missing is a
   sequence-header OBU emitter and temporal-unit framing; the return value is a
   bare frame OBU payload.

## Registered, not executed

- zensim: a from-scratch re-extraction of the avthdr/hdrvdc 944 tables needs
  fresh user sign-off (or an imazen video-decode capability). Hazard recorded:
  the vvc leg staged its ffmpeg n7.1.5 binary under `~/tmp/tools/` (volatile).
- zenav1-aom: PREREQ-AOM-STANDALONE itself is queued, not implemented; DOE arm
  A3 stays BLOCKED, svt arm carries the DOE.

---

## 2026-09-02 — ROUND 62: B-6 ANALYSED — the reduced-size screening failure was MISCALIBRATION, not a wrong direction, and both knobs are out of the tuning model

**This is ROUND 56's analysis lane.** Record: zenmetrics
`benchmarks/avif_doe_stageB6_analysis_2026-09-02.md` (+ `.pointer.md`), DOE plan
**§17**, commit zenmetrics `693787a0` (verified on `master@origin`).
**Analysis only — nothing declared, launched or stopped.**

**The wave.** `avifdoe-svt-b6-20260902` (native, 9 levels × 29 q × 32 images ×
speeds {4,6,7}) **25,056/25,056, live-gap 0**; its score run COMPLETE with
**23,489/23,489 distinct bitstreams scored, 0 cells missing `ssim2` or bytes**.
The score run's `done 5907 > declared 2112` is the chunk-key **rework echo**, not
3,795 extra cells — score jobs are chunk-keyed, so a re-declaration after the
sort changed mints new job identities over the same cells. Counted by distinct
scored cell it is exactly 100%.

**The registered question, answered.** B-6 existed to find out whether 1024²
screening was *miscalibrated* or *directionally wrong* for the two arms that
failed the transfer gate's T1. **Miscalibrated.** On every (speed, knob) cell
where **both** legs carry an effect above ±0.5%, budget and native agree on
**sign 1.000 of the time** (6/6, 10/10, 11/11, 11/11, 1/1).

- **`acb3` is NOT-MEASURED at native, not FAIL.** Only **2 of 11** cropped refs
  clear ±0.5% at speed 4 and **0 of 11** at speed 6 — there is no direction to
  transfer. Stage A's T1 = 0.25 was four references that cleared the floor on a
  **3-point** ladder.
- **`shp3`'s failure is real and speed-4-specific**: FAIL-T1 at s4 (T1 **0.75**
  vs the 0.80 bar, up from 0.62) and **PASS at s6** (1.000). Stage A's gate ran
  at speed 4 only and structurally could not see this.
- **T1 has a construction defect**, registered and deliberately **not** fixed
  (amending a pre-registered bar after seeing results is the coordinator's call):
  its denominator counts a reference by its *native* effect only and asks nothing
  of the budget leg, so it **grades sign agreement against budget-side noise**.
  Stage A §8.2 named the vanishing-effect half; the surviving-effect half is
  worse and produced one of B-6's two arms.
- **Magnitude, honestly:** per-image overstatement is **not** established
  (pooled 29/46, p = 0.104); suggestive at s6 only (`shp3` 9/11 p = 0.065, median
  ratio 1.63). A *ratio of medians* would have read 1.02–1.63 and told a tidier,
  less true story — the per-image test is the one reported. No relation to the
  size jump (SROCC 0.068).

**The knob verdicts — both OUT of the per-image knob set.**

- **`ac_bias` has no native effect at any level.** 12 (speed, level) medians span
  **−0.03% to +0.39%** BD-rate; at speed 7 every level is inside ±0.02%. It is
  also **not learnable** — per-image sign survives the three presets on **6–15 of
  31** images, at or barely above chance. `acb8`, the unclamped **H-10** level,
  *is* genuinely live (byte-identical to control on only 6.7% of cells, median
  **+0.559%** bytes, p95 +5.67%) — and it is the only `ac_bias` level with a
  defensible effect and it is a **LOSS**. All 2,784 `acb8` cells flagged.
- **`sharpness` is a pure bit cost that rises toward the fast presets** —
  `shp7` **+7.15 / +7.99 / +9.46%** at s4/s6/s7, `shp5` +5.57/+5.67/+7.52,
  `shp3` +1.20/+1.49/+2.75, and **`shp1` free** (+0.00…+0.23%; it moves the
  bitstream on 90% of cells and its *size* by +0.000%). Class spread is
  **10.97 pp** (`shp7` @s6: plot +0.17%, ai-gen +11.13%) but **B-3 does not
  fire** — no B-6 knob has opposite-signed class medians past ±1%.
- **The stakes.** A perfect per-image oracle over all 8 levels buys a corpus
  median **−0.23 / −0.09 / −0.01 pp**; a realistic speed-4-trained rule buys
  **−0.05 to −0.24 pp**. **One lead, n = 1–2**: `7004` and `7058` (both `plot`,
  1.05 MP) are where sharpness pays — `shp7` −7.41% and −5.02% at s7 — while the
  other four plots lose +0.54% to +8.72% on the same knob at the same size and
  speed. Two images is a lead, not a rule.

**QM × sharpness at native: NOT MEASURED, and it stays that way.** B-6 carries
no QM axis and no cross-wave join can make one (Stage A's pair cells are 1024²
crops whose native twins are provably different pixels: **0/2,535**
byte-identical). Only the **sharpness half of the additive baseline** is
re-measured, and it shifts **−0.39 pp** at corpus level (−1.86 pp on cropped
refs alone, but 19 of 32 refs are passthroughs contributing exactly zero). A
−0.4 pp shift does not overturn Stage A's −5.2…−5.5% residual, so **size is not
a plausible explanation for the synergy** — that is an argument about one input,
not a measurement of the output.

**Four gates, none assumed.** BD-rate parity vs `zenavif/scripts/rd_gap/bd_arm.py`
(max |Δ| **0.0 exactly**); a **new cross-run byte-identity gate** (AG's native leg
≡ B-6 on **576/576** shared cells; a0r/a1/a2 ≡ B-6 on **3,705/3,705**
passthroughs and **0/2,535** cropped — exactly as the corpus design requires);
the naive `avifsub-svt-enc` sweep as an **independent** native-defaults control
(21 of 24 medians identical to 4 dp, max |Δ| **0.0015 pp** — because they are the
*same bitstreams*, **928/928**); passthrough null **0 violations of 19** on all 8
q-matched cells.

**Three corrections owed to the record.** (1) DOE plan §15.4's preset column is
wrong — `s6` is preset **6** and `s7` is preset **7**, proven by 928/928 byte
identity and 0/928 against every other naive speed (the CPU-cost numbers are
unaffected; only the labels). (2) **Stage A §5.3's `6006`/`6018` exclusion is
CROP-specific** — at native `6006` yields BD-rates at all three speeds and is the
*largest* effect in the wave (`shp7` @s7 **+18.72%**); a native wave must re-test
degeneracy at native rather than inherit the budget exclusion list. (3) The naive
sweep's speeds **7, 8, 9 and 10 are byte-identical** (1–6 are mutually distinct)
— the preset saturates at 7, so 630 of its cells measure preset 7 three extra
times. Same *shape* as the Stage-A inert-knob finding but on the **speed** axis;
flagged for the port program, **no issue opened by this lane**.

**Two owners extended, not forked** (per the no-duplicate-implementations rule):
`avifdoe_harvest.py` now reads the naive sweep's knob-tuple shape (its label
synthesis **asserts chroma**, backed by the 928/928 identity measurement, which
the code names as the check to re-run for a future backend); and
`avifdoe_stagea_analyze.py`'s control-source label was hardcoded `"in-run-9q"`
and would have misreported B-6's **29-q** control — **verified non-regressive,
all five Stage-A `a0r` tables reproduce BYTE-IDENTICALLY**. The new
`avifdoe_stageb6_analyze.py` imports `bd_rate`/`frontier`/`median_ci`/`q1q3` from
the Stage-A analyzer, `binom_two_sided` from the Stage-A gates and SROCC from
`zenstats` — **it re-implements no statistic.**

**Limitations that matter for anyone citing these numbers.** `ssim2` is the
**only** corpus-wide scalar response — `zensim` is emitted as a 720-wide feature
vector with **no scalar**, butteraugli is dropped by standing directive — and
`sharpness` is a *perceptual sharpening* control, so *"it costs bits at matched
ssim2"* and *"it is not worth enabling"* are different claims and **only the
first is measured**. Whether sharpness buys anything a sharpness-blind metric
cannot see is **NOT MEASURED** — not zero, not disproven. Also: one backend
(svt-rs), no speed axis (`encode_ms` still unpersisted, B-4 still NOT EVALUABLE),
three presets with the slow end untested while the sharpness cost *rises* toward
the fast end, class n = 5–9 with `plot` bimodal, and every T1/T2/T3 figure at
**n = 11** because 19 of 32 references have no size contrast at all.

**Outputs.** `/mnt/v/output/zensim-avifdoe-b6/` +
`s3://zentrain/analysis/avif-doe-stageB6-2026-09-02/` (20 objects, 1.77 MB), with
a sha256 pointer in the zenmetrics record. **Tower was NOT mirrored** — `/mnt/tower`
returned a stale NFS handle; recorded as unavailable, not as done.

**What this buys the remaining 53 triggers.** B-6 cost **13.2 CPU-h** of the
60 CPU-h Stage-B envelope (0.22×) and retired the two arms §10 called "the
cheapest high-value cells". Anything keyed on `acb3`/`shp3` at reduced size
should be re-scoped or dropped, and the same question should be put to the three
NOT-MEASURED arms (`acb1`, `tl1.0`, `tl1.1`) — their verdict may likewise be *"no
effect to transfer"* rather than *"untested"*. **B-2's QM × sharpness cluster is
now the only route to the synergy question**, and B-6 has removed the concern
that its residual is a size artefact. Prioritisation remains the coordinator's.

## 2026-09-02 — ROUND 63: zenav1-svt #18 LOCALISED AND FIXED — bd10's ">8 MP" cliff was AV1's FORCED tile grid, and it reproduces at 0.27 MP

Differential-localisation lane for **imazen/zenav1-svt#18**, opened by the HBD
executor of the AVIF/HDR arm (`zenmetrics/benchmarks/avif_hdr_arm_plan_2026-09-02.md`
§10.4d) after a high-bit-depth wave was stopped at 120/3,248 cells. Record lives
in zenav1-svt (commit **`3121b6a8`**, verified on `main@origin`; issue comment
[#18](https://github.com/imazen/zenav1-svt/issues/18#issuecomment-5516202540));
this is the zensim-side ledger row. **Nothing declared, launched or stopped
here.**

**The reported shape was a proxy, and following it would have been the wrong
search.** The issue bracketed a correctness cliff between **8.01 MP (healthy)**
and **12.00 MP (broken)** at `bd10`, with 8-bit and zenrav1e clean at the same
sizes — a shape that reads as a 16/32-bit overflow. It is not. AV1 **forces** a
multi-tile grid once a frame passes `MAX_TILE_WIDTH` (**width > 4096 px**) or
`MAX_TILE_AREA` (**SB-aligned area > 4096·2304 = 9,437,184 px**), and it
**clamps a `(0,0)` tile request UP** to that minimum. `AvifEncoder` never
requests a tile, so every AVIF encode past ~9.44 MP had been multi-tile all
along. Intra prediction is tile-scoped in AV1; two `bd10` sites were not
(`extract_neighbors_hbd` frame-absolute at preset ≤ 8; `TileMi::whole_frame` in
the preset ≥ 9 level re-encode), so the encoder predicted across tile edges a
conforming decoder cannot see.

**Discriminating cell, chosen before the size bisect.** `4096×64` vs `4160×64`
— one superblock column apart, **0.26 vs 0.27 MP**, twenty times *below* the
reported bracket. bd10: **clean vs 65,054 of 399,360 samples wrong**; bd8:
identical in both. That single pair rules out area, width, height and every
accumulator-overflow hypothesis in one 4-second encode, and it is now the CI
regression cell. The predicted flip — `ceil(w/64)·ceil(h/64) > 2304` — was then
tested on a **0.19 MP** bracket (`2944×3200` = 2300 SB clean vs `2944×3264` =
2346 SB, 3,448,059 of 14,413,824 wrong) and confirmed exactly.

**Oracle note that generalises past this issue.** The defect was invisible to
every byte-parity gate the port owns, because byte-parity against C cannot see
an encoder/decoder prediction **mismatch** — both encoders being wrong the same
way stays green. It was found by comparing the encoder's own published
reconstruction against the reference decoder's output, which is the same
two-oracle discipline `alignment_gate.sh` already uses. A 2026-07-22 note in
that repo recording this exact threading as "byte-inert" was **correct on the
byte oracle and blind to the class** — and after the fix, 7 of its 8
pinned-diverging cells became byte-exact with C anyway (axis 4/12 → 11/12).

**What this means for our data.** Any `bd10` AVIF result produced by this
backend on a source whose SB-aligned area exceeds **9,437,184 px**, **or whose
width exceeds 4096 px**, is invalid and needs re-encoding — the width limit is
independent of area, so a 6 MP panorama wider than 4096 px was affected while
the issue's bracket said "healthy below 8 MP". Everything below both limits was
single-tile and is unaffected. The stopped HBD wave can resume against
`3121b6a8`.

**Gates now standing** (all measured, this lane): `issue18_repro.rs` — 3
multi-tile cells + a single-tile control, on the decoder oracle. Written before
the fix and run at the parent commit: **2 fail, the control passes** (the third
multi-tile cell is the preset-9 re-encode arm, added after fixing the funnel arm
exposed it, and measured failing at that intermediate state — 49,606/98,304).
Plus 4 `bd10ReconEq` cells in
`regression_spotcheck.sh`; workspace **2,458/2,458** (189 binaries), spotcheck
**71/71**, coverage combos **40/40**, bd10 recon parity vs C **13/13**,
alignment **74/74**, bd10 matrix **36/36**, non-flat **309/309**, partial-SB
**159/159**, hbd-src **26/26**, tile gate **29/29**. Byte-inertness A/B: **26 of
28** cells emit identical OBUs (every single-tile cell at both depths, every bd8
multi-tile cell); only bd10 × multi-tile moved.

**One gate is NOT green and it is NOT this lane's** — recorded so nobody reads
the list above as "everything passes". `bd10_hbd_pq_gate.sh` reads **48/60** on
this box, the 12 failures all at preset 6. Those are the cells
`docs/SUSPECTED-C-BUGS.md` #9 pins as `uname -m`-scoped to aarch64, failing here
on x86-64 against a **locally built** C reference rather than CI's. Ruled out as
this lane's doing by direct A/B: the port's OBU is **byte-identical pre-fix vs
post-fix on all 60 cells**, so the verdict cannot have moved. Left as found and
not silently annotated — the C-oracle host-divergence entry is the owner.

---

## 2026-09-02 — ROUND 64: the #18 validity re-audit — no published number was contaminated, and the fix repairs only half the shapes

**Owning record:** `zenmetrics/benchmarks/avif_hdr_arm_plan_2026-09-02.md`
§10.4e. Follows ROUND 63 (which localised and fixed `zenav1-svt#18`).

**The audit had to be redone, because the first pass checked the wrong thing.**
`#18`'s root cause is AV1's **forced tile grid**, and it has **two** triggers:
`width > 4096` — *area-independent* — and sb-aligned area > `4096·2304`. My
original clearance of Track T1 was **area-only**, so the width limit was
genuinely unchecked. Re-run against both, from final encode dimensions:

| wave | backend | cells | valid | invalid |
|---|---|--:|--:|--:|
| t1ac | zenav1-svt | 2,016 | **2,016** | **0** |
| t1b (unencoded) | zenav1-svt | 4,320 | 4,320 | 0 |
| t1d | zenav1-svt | 96 | 72 | **24** |
| t2a | zenav1-svt | 3,248 | 203 | **3,045** |
| t2b | **zenrav1e** | 432 | **432** | 0 |

**No published number consumed an invalid cell**, and the reason is a property
of the corpus rather than luck: the budget corpus is **0/32 invalid on both
triggers**, its widest file being 1536×1024 — the width limit is not merely
unexceeded, it is not approached. Both A1's `bd10` median and this arm's t1b
refusal ran entirely on it. **Verified rather than asserted:** the gate was
re-run with an explicit per-cell validity filter and came back
**byte-identical** (A1 repro n=31 median −1.025 %, 23/31; T1-c n=31 median
−0.214 %, CI [−0.836, +1.321], 18/31). **The t1b refusal stands.**

The 8 native images the triggers predict broken are **exactly** the 8 measured
broken — root cause and observation agree cell-for-cell, which is the strongest
confirmation available that #18 is the whole story for T1-d.

**But the fix repairs only one of the two tile axes.** Composed `3121b6a8` in
via a workspace-root `[patch]` — necessary because zenavif moved `zenav1-svt`
to a git-rev pin *the same day*, and that pin **predates the fix**, so building
against it reproduced `−57.049692` **exactly**. With the fix in, and with the
fix's own four `issue18_repro` tests **passing** in the same tree:

| dims | orientation | sb-area | post-fix |
|---|---|--:|--:|
| **4000×3000** | landscape | 12,128,256 | **+88.96 ✅** |
| **3000×4000** | **portrait** | 12,128,256 | **−10.61 ❌** |
| 3302×4844 | portrait | 16,187,392 | +1.56 ❌ |
| 3286×4868 | portrait | 16,400,384 | −15.45 ❌ |

**Identical area, identical sb-aligned area; only the long axis differs, and
only one is repaired.** The 8-bit control on that cell is 86.57. Reported as a
[#18 follow-up](https://github.com/imazen/zenav1-svt/issues/18#issuecomment-5516506667).

**So t2a was NOT restarted.** 14 of its 16 references are portrait *and* over
the area limit, so a restart would still mis-encode **88 %** of the corpus —
the same waste a second time — and the 2 usable refs would destroy the K=16
stratification and G0.5 primaries balance that make T2-a a baseline at all.

**t2b is complete and valid**: 432/432 encoded, 48/48 scored, zero poison, on
the zenrav1e backend that #18 never touched. It is the workspace's first HDR-10
RD dataset, currently standing alone rather than as the contrast §4.3 designed.

**Transferable lesson.** A synthetic repro passing is not the same as the
production path being repaired: four targeted tests covering *both* tile axes
went green while the real portrait path stayed broken. The check that caught it
was a real corpus cell with a known-good 8-bit control beside it — and the
tightest control in the whole audit was a **pair differing only in
orientation**.

## 2026-09-02 — ROUND 65: zenav1-svt #18 ROUND 2 — the residual was a predictor that took a tile and ignored it, and my own round-1 tests were blind on the PRESET axis

Follow-on to ROUND 63. The HBD executor's independent re-verification (issue #18
follow-up) showed the first fix repaired only part of the class: with it in
tree and all four ROUND-63 tests passing, `4000x3000` landscape scored **+88.96**
while `3000x4000` **portrait** still scored **−10.61** against an 8-bit control
of +86.57. Record in zenav1-svt (commit **`2ca060f4`**, verified on
`main@origin`; issue comment 5516796458); this is the zensim-side row. **Nothing
declared, launched or stopped.**

**The residual, and it is a lesson about reading code rather than about tiles.**
`intra_edge::dr_predict_hbd` — the DIRECTIONAL 10-bit predictor — received the
correct `DrGeom.tile` and then derived **all four** availability predicates from
the FRAME (`have_top` from `g.mi_row > 0`, `right_available` against `mi_cols`,
…), while its u8 twin `dr_predict` scoped every one to `g.tile`. ROUND 63's own
note had cleared this arm on the grounds that *"it passed `tile: geom.tile`"* —
**passing a tile is not using one**, and that arm turned out to be the larger
half of the defect, because it is the one real photographs reach. Four lines.

**The orientation reading was a coincidence, and disproving it mattered.**
`4000x3000` and `3000x4000` both resolve to **1 tile column × 2 tile rows** —
the same grid — so "the row axis is unrepaired" could not have been the
mechanism. Sweeping the ROUND-63 test geometry instead (256×256, 2 tile rows,
bd10) found the real axis:

| | q6 | q12 | q20 | q40 |
|---|---|---|---|---|
| **p0 / p2 / p3 / p4 / p5** | FAIL | FAIL | FAIL | FAIL |
| p6 / p7 / p8 / p9 | ok | ok | ok | ok |
| `uniform`, any preset | ok | ok | ok | ok |

**It is the PRESET axis** — a predictor is reached only when mode decision picks
it, and the intra candidate set narrows with preset. ROUND 63's cells were
presets 6 and 9: the passing side. Not content, not qp, not the tile axis, not
orientation; each was varied and none discriminates. `AvifEncoder` **speed 4 →
preset 4** and **quality 90 → qp 6** sit dead centre of the failing band, which
is the entire distance between "synthetic gates green" and "product broken".

**Measured, before → after.** The reported cell — the real `3000x4000`
photograph at qp 6 / preset 4 — **6,468,452 of 18,000,000 samples wrong, first
at Y r2048** (= the 32-SB tile-row boundary) **→ 0**. Forced-by-AREA portrait
control `gradient 2920x3270` (9.55 MP, 2392 SB, partial SB on both axes)
4,185,160/14,322,600, first Y r1664 (= 26 SB × 64) **→ 0**. The 60-cell
{gradient,diag,uniform} × preset{0,2,4,6,9} × qp{6,12,20,40} sweep is clean at
2 tile rows and at 2×2 tiles; **bd8 was clean throughout**. Byte-inertness A/B:
**30 of 32** cells identical (every single-tile cell at both depths across nine
presets, every bd8 multi-tile cell).

**A green gate that proves nothing, named so it is not trusted again.**
`coverage_combos_gate`'s bd10×tiles axis read **11/12 identically before and
after** this fix, because every cell on it is preset 6, 10 or 13. That axis was
green over a live wrong-pixels bug for the whole interval. Recorded in
`docs/coverage-combos-map.md`: extending it means extending it **down** into
presets 0-5.

**For our data, unchanged from ROUND 63 and now actually true**: any `bd10`
AVIF result from this backend whose source exceeds **9,437,184 px** of
SB-aligned area **or 4096 px of width** is invalid. t2a (3,248 HDR cells, 88 %
portrait over the area limit) re-encodes from scratch against `2ca060f4`.

**Two rollout steps remain and neither is this lane's:** `zenavif` pins
`zenav1-svt` at rev `ef0b122bd`, which predates both fixes (ancestry-confirmed —
that is why the executor needed a workspace `[patch]`), so it needs a
**deliberate pin-bump commit in zenavif, never a `[patch]` left on master**;
then a new fleet image tag carrying the bumped seam.

**Gates:** `issue18_repro.rs` is now 8 cells — a preset-BAND sweep, a
directional forced-tile-column cell, a single-tile band control that passes
before and after (so "low preset is just broken" and "a fix that disabled
directional prediction" both fail it), and a forced-by-area **portrait** cell at
the reported shape rather than a stand-in. `regression_spotcheck.sh` **81/81**
(+5). Workspace **2,486/2,486** (189 binaries), coverage combos 40/40, bd10
recon parity vs C 13/13, alignment 74/74, matrix 36/36, non-flat 309/309,
partial-SB 159/159, tile gate 29/29.

---

## 2026-09-02 — ROUND 66: #18 closed at the second fix, t2a restarted, and the axis I named was the wrong one

**Owning record:** `zenmetrics/benchmarks/avif_hdr_arm_plan_2026-09-02.md`
§10.4f. Follows ROUND 64 (the validity re-audit).

**I got the residual's cause wrong, and the way I got it wrong is the useful
part.** After `3121b6a8` I reported the remaining breakage as
**orientation-dependent**, on the strength of one pair: 4000×3000 healthy
(+88.96) against 3000×4000 broken (−10.61), identical area, identical
sb-aligned area. I described it as "a pair differing only in orientation." **It
wasn't.** They are different source images — `1442.scale4000x3000` and
`1008.scale3000x4000` — so content varied too, and 1442 was already the mildest
of the eight broken cells before any fix (+7.88 vs −59.84). Both shapes resolve
to the same 1×2 tile grid, so orientation was never the discriminator.

The real axis was **preset**. `dr_predict_hbd` — the *directional* HBD intra
path — derived all four availability predicates from the **frame** while its u8
twin was tile-scoped, and directional modes only enter the candidate set at
**presets 0–5**. Every cell I measured ran speed 4 = preset 4, dead centre of
that band, which is why the first fix looked shape-dependent instead of
mode-dependent.

> **The transferable error:** from "same area, one repaired" the correct move is
> *find the variable that actually separates the two*, not *name the one that
> visibly differs*. My "tightest control in the audit" was not a control at all
> — it varied two things and I reported one.

**META-FINDING, and it outlives this bug.** The C-parity axis stayed green
through **both** rounds because every cell it ran was preset 6/10/13 — outside
the directional band. A parity or recon gate that samples only fast presets is
structurally blind to this defect class. **Extend recon/parity coverage DOWN
the preset band, not merely across sizes.** The new gate
(`scripts/jobsys/avifhbd_recon_gate.sh`) is built on exactly that: bd10 ×
forced-multi-tile × low preset, refusing any source that does not force
multi-tile, asserting **bd10 ≥ 8-bit − TOL on the same cell** so it calibrates
itself. Verified both ways — PASS at +4.182 on the fixed binary, FAIL (exit 1)
at −143.623 on the recorded pre-fix value.

**Repair, measured** (bd10 q90, preset 4, portrait over-limit):

| dims | old pin | `+3121b6a8` | **`+2ca060f4`** |
|---|--:|--:|--:|
| 3000×4000 | −57.05 | −10.61 | **+90.75** |
| 3302×4844 | broken | +1.56 | **+75.57** |
| 3286×4868 | broken | −15.45 | **+81.65** |

bd10 now beats its 8-bit twin (+4.18 over an 86.57 control) — the arm's premise,
finally observable.

**Rollout.** Pin bumped **in zenavif** (`56179fcb`, `ef0b122bd`→`2ca060f42`,
`cargo test --workspace` green), and the interim workspace-root `[patch]`
**reverted** — left on master it would silently redirect every other lane's
build to a local working copy, the hazard that pin exists to prevent. t2a
restarted as a **fresh run** (`avifhbd-t2a-fix-20260902`), not a requeue:
cells are content-addressed, so re-declaring into the old run would have
counted its 120 **invalid** cells as done. Fleet blobs now score
**+76.67 / +81.07 / +83.08** at q88/90/92 against the pre-fix wave's
**−67.80 / −66.85 / −64.26**.

---

## 2026-09-03 — ROUND 67: new-era delta audit + sweep, `claude-newera3` (Sonnet
failover after two Opus-service-incident deaths mid-early-phase)

**Owning records:** `zenmetrics/benchmarks/avif_newera_delta_2026-09-03.md`
(the audit) + `avif_newera_sweep_2026-09-03.md` (the grid + Stage-B
reconciliation). Follows ROUND 66 — the pin this round audits IS the
`2ca060f4` fix ROUND 66 shipped.

**t2a-fix + t2b: both COMPLETE** (live gap 0, 0 errors, verified against the
live ledger — the stored snapshot predates the run and lies). Nothing left to
avoid disturbing; scored capacity freed.

**The era delta is real but narrow: 33 zenav1-svt commits, 2 of them
AVIF-reachable.** The other 31 are inter/video-mode port work (PD0, NSQ
motion search, conformance) with zero surface on the intra-only still-image
path. Three items in this lane's own brief — sc-detector tier-1, a
depth-coercion fix, "new tune-vmaf" — all **pre-date** the pin (checked by
ancestry, not assumed) and are corrected out of the delta. tune-vmaf is
additionally **falsified as a knob at all**: the ported preprocessing chain
is called from nowhere in the encode pipeline.

**Re-probed the two known-dead knobs at HEAD** (`knob_byte_identity`, 3
presets → widened to 5 to find the edge): `tn0`/`scm0` still fully dead.
**`scm3` is not** — 32/288 divergent, all screen content, all at raw SVT
preset 8–9. Preset 8 turned out **unreachable** through zenavif's product
speed dial (`speed_to_svt_preset` skips it entirely, 1→10 maps to
`{0,1,3,4,6,7,9,9,9,9}`); preset 9 **is** reachable, via speed ∈{7,8,9,10}.
`tn3` (tune=IQ, Stage-A's biggest win and least certain one) forces this
exact field as one of 9 aliased fields and has never been measured past
speed 6 — flagged top priority for the stability re-run this round declares.

**Found mid-audit, before writing any new code (the mandatory duplication
check): `avifdoe-svt-t1d-20260902` already asked the bd10-native question**
(committed 08:30 the same day as ROUND 66's fix work) **and its data is
corrupted** — encoded 09:57–10:16, 4.5–7h before either #18 fix, 24 of 96
cells on the two known-broken images (`6602`/`6604`) plus six more
12 MP forced-multi-tile images. Re-declared fresh as
`avifdoe-svt-eradelta-c1-20260903` rather than written from scratch.

**Declared + launched, smoke-verified grinding, zero errors:**
`avifdoe-svt-eradelta-{a1,b1,c1}-20260903` (6,912 + 8,640 + 96 = 15,648
cells; new plan `svt_doe_era_delta_r1` in zenavif `b552418e`, additive, 251/251
lib tests green; image `exec-avifhbd-eradelta-e015344f`, control-arm
verified with a real mounted source per the empty-dir trap ROUND 66's own
lineage documented). **G3 + the recon gate on the re-run bd10-native cells,
the whole point of arm-set C**, both known-broken images, era-delta binary:
`6602` bd10 **75.573** vs 8-bit **75.114** (Δ+0.459, PASS); `6604` bd10
**81.651** vs 8-bit **80.856** (Δ+0.795, PASS, essentially reproducing ROUND
66's own +81.65). The fix holds on a fresh, independently-built binary.

**Stage-B B-1/B-2/B-3**: 55 triggers registered, 447,636 cells if all
honoured against a 60,000-cell envelope (7.5× over) — unchanged by this era's
findings, since none of B-2's QM×sharpness cluster or most of B-1/B-3's
other knobs sit in the AVIF-reachable 2-commit delta. `tn3`'s B-1 trigger and
`bd10`'s B-3 trigger are partially absorbed by this round's arm-sets B/C; the
other 51 remain the coordinator's budget call, stated plainly rather than
silently deferred.

**aom tranche**: registered PLANNED-BLOCKED per the addendum, not declared.
zenav1-aom#15 is OPEN but far more advanced than briefed — the concurrent
lane's self-contained encode path covers the real ALLINTRA-default envelope
186/186 byte-exact + dual-decoder-verified, already wired into zenavif main
(`000ac9a`, zenav1-aom rev `c3e1b4ab`) — squarely that lane's live work in a
shared repo, moving in real time throughout this round.

**One incident this round owes the record, unrelated to the above**: an
early `env` dump (before the LAN-store env vars were understood) printed a
live `OPENAI_API_KEY` in cleartext — the `secret|access_key|token` filter
this lane was handed does not match `API_KEY`. Switched to allowlist-only
env inspection for the rest of the round; flagged to the user for rotation.

## 2026-09-03 — ROUND 68: era-delta + T2-HDR analysis, `claude-deltaanal` (Opus lane)

**Owning records:** `zenmetrics/benchmarks/avif_eradelta_analysis_2026-09-03.md`
+ `avif_hdr_rd_baseline_2026-09-03.md` (+ their pointer docs). Two docs, not
one: the waves share no corpus, no instrument and no pin.

**STABILITY — the era changed NOTHING, provably, and the proof is bytes not
statistics.** Arm-set A replicated `svt_doe_main` at the new pin and reproduced
Stage-A's bitstreams on **6,912 / 6,912 shared cells, byte-identical**.
Arm-set B — a *different* plan written this era — reproduced the same bytes
independently on **3,456** speed-4/control cells (vs `a1`) and **2,880**
speed-6 knob cells (vs `a2`). **13,248 verified cell-pairs, 9,792 distinct cell
identities, 0 differing.** Because the bytes are identical *and the scorers
agree exactly*, every covered effect is the SAME NUMBER, not "within CI":
`shp7` 5.4676/7.3077, `tn3` −7.0342/−4.4807, `qml1.2.10` −0.2895/−2.5853 all
reproduce Stage-A's `stagea_inrun` to every digit. **≥1 pp movement: 0.0000 pp
on every replicated cell.** Scope stated rather than implied — 7 of the 16
speed-6 knobs (`acb1 acb3 mtx32 qml1.8.15 tl1.0 tl1.1 tn0`) were **not**
re-measured this wave and keep their Stage-A numbers unverified.

**SCORER-vs-ENCODER drift, separated.** Identical bytes make score differences
attributable: on 6,912 provably-identical bitstreams the era-delta scorer image
and Stage-A's agree on `ssim2` to **exactly 0** (max |Δ bytes| 0 too). So the
byte result and the effect result are not two ways of saying the same thing —
the second needed the first plus a measured instrument.

**★ scm3 at speed 7 (preset 9): a knob believed dead is worth a MEDIAN −50 %
BD-rate on screen content.** 90 of 288 cells differ from the control at speed 7;
**0/288 at speed 4 and 0/288 at speed 6** (reproducing the dossier's "dead at
preset ≤7" on a different instrument). Divergence is content-EXCLUSIVE:
photo 0/63, AI-gen 0/81, scan 18/45, screenshot 27/45, plot 45/54. Conditional
on firing (10 of 32 images) the BD-rate is **median −50.08 %, range −88.86 % to
−18.57 %, 10 of 10 wins**. Grounded in bytes: `6018` (a scan) at q90 goes
434,757 B → 29,536 B (**14.7×**) while ssim2 goes 94.636 → **100.000**. The
corpus median is 0.0000 — a knob confined to a content class is invisible in a
corpus-median, so the conditional statistic + firing count is what is reported.
`tn3` tracks it closely on the firing images (it aliases the same field),
confirming the mechanism the delta audit predicted.

**★ Arm-set C: the bd10-native answer INVERTS, and t1d is superseded
everywhere.** 72/96 cells reproduce byte-identically; **24 differ, on exactly
the 8 images #18's tile-forcing predicate selects — zero false positives, zero
false negatives.** Worse than "24 of 96": the registered n for the cross-size
question is **13 images**, and **8 of those 13 (61.5 %) are the corrupt ones**.
Clean read at n=13, q45: **−0.69 % bytes for +0.244 ssim2, bd10 DOMINATES on
10/13**; the superseded block read **−104.80 ssim2** at q90 with 8/13
DOMINATED. Δbytes is unchanged pre-to-post on all 24 cells — the broken encoder
emitted normally-sized bitstreams containing wrong pixels, which is why only the
quality column shows it. **Q4 answered: bd10 DOES survive native resolution, and
is marginally stronger on the large images** (10/13 DOMINATES vs 8/19 on the
passthroughs). Independent cross-check: `1008` q90 reads **+4.182**, the same
value §10.4f recorded from the local recon gate, reached through fleet encode +
fleet scoring instead. **A BD-rate is NOT MEASURED for this block in either
era** — a 3-point ladder cannot satisfy the ≥4-point guard, and the guard was
not loosened. t1d is annotated SUPERSEDED in the plan doc (3 sites) and
DATA_PROVENANCE.

**★ HDR T2 — an instrument check, and a CORRECTION this lane owes the record.**
Two scoring images touched the T2 corpus. Measured on **351 (encode_sha, metric)
pairs scored by BOTH**: `ssim2` 0 differing, max |Δ| **0.0000000000**; `zensim`
225 differing, max |Δ| **1.156e-7**. So the images agree and there is no
material era split. An earlier draft of the doc claimed an 11.14-point zensim
split between images, inferred from ONE hand-run — **wrong**: that gap is
between the **fleet ScoreFile executor** and a hand-run `score-pairs --metric
zensim` (the CLI's default profile), and a third build reproduces the shape in
the other direction. **`score-pairs --metric zensim` and the ScoreFile
executor's `zensim` are different quantities** — a trap that generalises. Every
T2 number was already `ssim2` so nothing published moves; the 48 pre-era blobs
stay excluded as hygiene (last-write-wins over a sha-sorted glob) rather than
necessity. G5 route proof re-run on this wave's own cell:
`--hdr-transfer pq` ≡ `pu-rescale` (Δ 0.0) ⇒ faithful f32, and the value is
bit-equal to the fleet's stored score; the no-`--hdr` tripwire refuses loudly.

**★ THE HDR-10 RD BASELINE EXISTS — and svt beats zenrav1e by a median
−43.01 % BD-rate.** 3,680/3,680 cells scored (16 PQ refs × 7 svt presets × 29 q
+ 3 zenrav1e speeds × 9 q, all 10-bit PQ, faithful f32). **Q5** (a baseline, no
PASS/FAIL bar by registration): the per-(arm, q) reference curve is on disk;
svt's median ceiling is **ssim2 86.1** vs **77.0**, its rate floor **9–10 KB**
vs **91–93 KB**, and at the *same* dial position svt is both cheaper and
lower-quality (q45: 0.083 bpp / 2.6 vs 0.235 bpp / 29.0) — the two dials are not
comparable, so only matched-quality reads mean anything. **Q6** (a CONTRAST —
backend, chroma 4:2:0-vs-4:4:4 and matrix all differ): per-backend Pareto
envelope **−43.01 %**, 95 % CI [−48.63, −37.97], **16/16 images**, range −57.00
to −18.22. **The ladder-density objection is resolved by measurement, not
argument**: all 21 cross-backend arm pairs land between −36.70 % and −45.38 %,
the envelope sits inside that range, and the most conservative single-arm read
available — svt's SLOWEST preset vs zenrav1e's FASTEST speed — is still
−36.70 % on 16/16. Density buys a few points at most; it is not the gap.

**Two fleet gaps found and closed, not worked around.** (1) The era-delta wave
was declared with **no score gapfill loop and no scoring worker** — 0 of 15,648
cells scored at pickup, and none would have been. (2) The T2 scorefiles were
declared correctly (with `--hdr`) but **`avifhbd-t2a-fix` was 0/3,248 scored**
while containers on tower AND r7900x sat in idle-drain restart loops against
the already-COMPLETE *encode* run. Both repurposed to score runs.

**Tooling (committed, each non-regression-measured against the owner's own
outputs):** `avifdoe_era_compare.py` (new — cross-era cell identity,
scorer-vs-encoder drift separation, effect-stability verdicts with n on both
sides); `avifhbd_t2_analyze.py` (new — Q5/Q6, importing frontier/bd_rate/
median_ci from the BD-rate owner, refusing to map preset↔speed);
`avifdoe_harvest.py` (+3 additive columns for HDR single-dial knob tuples — 14
pre-existing columns byte-identical); `avifdoe_stagea_analyze.py` (+ the paired
matched-q read, the `transform` column, and `--runs` — all six owned tables
byte-identical to the committed `stagea_inrun/` outputs, verified three times).

**Method notes worth keeping.** (a) Stability uses `--control inrun` on BOTH
sides, matching `stagea_inrun/` not `stagea_a0r/` — differencing an in-run-9q
effect against an a0r-dense one confounds instrument with era. (b) `--runs`
exists because arm-sets A and B share 3,456 cell identities; pooling them would
enter each image twice with the same BD-rate and halve the bootstrap CI on
duplicate values. (c) Medians reproduce exactly across eras but bootstrap CI
*bounds* can differ in the third decimal — `median_ci` resamples in insertion
order. Registered, not fixed: making it order-invariant would move every
published Stage-A CI. (d) n is 30–32 not 32 because two scanned documents have
a Pareto frontier that collapses to 2 points at speeds 4/6; Stage-A carries the
same n on the identical image set.
