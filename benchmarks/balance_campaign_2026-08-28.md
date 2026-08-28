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
