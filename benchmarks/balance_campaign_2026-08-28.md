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
