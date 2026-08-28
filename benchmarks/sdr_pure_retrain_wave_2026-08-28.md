# SDR purity retrain wave — pre-registered (2026-08-28)

REGISTERED BEFORE ANY FIT. Trigger: the user's SDR-freeze answer ("SDR purity
retrain first") — W10L9_s4003 is the unanimous two-lens SDR selection, but its
training is campaign-era (saw the channel-A synthetic-v2 files and the
pre-family bigcodec splits). This wave retrains the EXACT winner recipe on
policy-clean views; the freeze moves to its result.

## Data (frozen; `/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/_MANIFEST.json`)
Recipe inputs from the winner's embedded repro, sha-verified against the
originals, with the policy filters applied (one substantive, one measured no-op):
- `safesyn_pure` 111,068 → **111,068 — the purge is a MEASURED NO-OP**: the
  safesyn training table contains ZERO gen-token refs and its ref set
  intersects the d≤2 sharing sources ZERO times (verified against
  `canon_vs_train_synth.tsv`, all 79 sharing sources gen-named). The
  channel-A files lived in the sources DIRECTORY, never in this table —
  so the campaign winner's training NEVER saw channel-A content, and the
  earlier "training saw the shared files" framing is CORRECTED here.
- `safesyn_teacher944_pure` — same, no-op by the same measurement
- `tbig_944_200k_pure` 208,169 → **192,714** (−15,455 rows whose origin's
  FAMILY bucket ≠ train per `split_map_family.tsv`)
- `tbig_teacher944_pure` — same predicate, −15,455
Unchanged groups (separate corpora, standing clean audits): cid22_train201
(metric-anchored, never human-MOS), kadid, tid, kadis 50k, konjnd_bpg
train/val.

## Recipe (frozen = the W10L9 embedded argv verbatim, paths swapped, seeds {4003,4004,4005})
L0 (0-hidden), target human_score ×100, epochs 120, pairs/epoch 50k,
coarse-decay 1e-5, max-features 944, the full winsor/signed_cbrt transform
list from the repro, group weights identical.

## Gates (frozen)
Same as the campaign selection: freeze_check E.4 over the fullevals
(--regime 944; the resliced boards), M3a measured, packed via
`bake_dial_refit pack` (default anchor/verify as W10L9_s4003_packed).
**Comparison row:** the incumbent W10L9_s4003 on the same panels — the wave
answers "does purity-clean training cost or gain?" with the E.4 + gate-panel
numbers side by side. No auto-freeze: the result returns to the user.

## RESULTS — the purity retrain WINS on both lenses (2026-08-28)

3/3 seeds trained (verbatim recipe, pure views), packed with the exact
campaign parity invocation (identity gates BIT-identical on all 2,035
anchors). Selection (freeze_check E.4, incumbent included):

| rank | bake | floors | bal_comp | M3a | sel_comp | sdr25 |
|---|---|---|---|---|---|---|
| **1** | **W10L9P_s4005_packed** | **8/8** | 0.8565 | **0.8744** | **0.9876** | 0.9612 |
| 2-3 | W10L9_s4003(_packed) — incumbent | 8/8 | 0.8549 | 0.8626 | 0.9843 | 0.9527 |
| 4 | W10L9P_s4003_packed | 8/8 | 0.8567 | 0.8278 | 0.9808 | 0.9539 |
| 5 | W10L9P_s4004_packed | 7/8 | 0.8491 | 0.8822 | 0.9814 | 0.9678 |

Gate panel AGREES: winner weighted_goal **0.764 vs 0.727**, g1=g7=1.0, dial
mono 0.9947, reach 90.6. Axis detail: cid22 0.8901 (+0.003), imazen26 0.9298
(+0.009), nonphoto 0.9342 (+0.008); honest dips: konjnd 0.4446 (−0.054),
hfnlproxy 0.3781 (−0.042) — the family-filtered rows carried near-threshold/
near-lossless signal; floors hold 8/8 regardless. **Answer to the wave's
question: family-clean training GAINS on the headline axes and the selection
rule — the freeze proposal moves to `W10L9P_s4005_packed`** (user-gated).
Sha: recorded in the board fulleval; bakes at
`/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/`.

## G-OUT: worst-outlier gates per chart — REGISTERED before computing (2026-08-28, user: "new gates for worst outliers per chart, extreme outliers are bad signs")

Per (candidate × rank axis), from the fulleval `per_pair` blocks (the same
data the scatter charts draw):
- **chart-z** = residual from the OLS fit line (what the chart shows),
  normalized by the MAD-σ of residuals (robust). The 4PL-mapped OR/Z-RMSE
  remain the panel's own stats; chart-z is the chart-visual outlier measure,
  method stated.
- **rank-displacement** = |rank(pred) − rank(target)| / n per pair.
Bars (first-pass, provisional): **EXTREME: zero pairs with |chart-z| > 6 on
any axis (n≥500)**; SEVERE: frac(|chart-z| > 4) ≤ 0.2% per axis;
rank-displacement: no pair > 0.6 on any axis with n≥1000. Plus the DIAL
chart: worst backward step magnitude per ladder (report; bar = no backward
step > 5 dial points). Worst-5 pairs per axis are LISTED with their targets
so outliers are inspectable, not just counted. Candidates: SDR purity winner
+ incumbent; HDR incumbent-hfpack + t2 retrain.

## G-OUT RESULTS (2026-08-28) — all four candidates FAIL the registered bars; the bars fail the base-rate test; the lens still ranks

**Registered-form verdicts:** SDR purity winner FAIL 4 axes; SDR incumbent
FAIL 5; HDR incumbent FAIL 9; HDR t2 FAIL 9. **But the PEER base-rates show
the zero-extreme bar is unpassable by any known metric on these axes** —
cvvdp: 341/327/353 extremes on hfnl/imazen26/nonphoto (max|z| 91.8!);
butteraugli: 278/24/25. Comparative extreme counts (hfnl / imazen26 / nonphoto):

| candidate | extremes | reading |
|---|---|---|
| SDR purity winner | 71 / **2** / **4** | the cleanest tails ever measured on these axes |
| SDR incumbent | **61** / 8 / 8 | comparable; hfnl slightly cleaner, imazen26/nonphoto heavier |
| butteraugli peer | 278 / 24 / 25 | 4-6× the models' tails |
| cvvdp peer | 341 / 327 / 353 | catastrophic tails |
| HDR incumbent (cross-domain) | 362 / 372 / 345 | SDR axes are out-of-route; bounded [−28, 100] |
| HDR t2 (cross-domain) | 524 / 196 / 170 | **UNBOUNDED out-of-domain: emits −387 (kadid), −301 (live)** |

**Named outlier classes (the product signal the gate exists for):**
1. **HF-NL catastrophics** (every candidate + every peer): near-lossless
   cells (target ≈ 0.91+) scored 28–55 — the HF weak zone's worst tail.
2. **Corruption-cell high-scores**: negative-target cells (−45…−66) scored
   58–100 by the SDR models — exactly the corruption-HEAD territory (the
   dial's known blind spot; the head is the shipped mitigation).
3. **Unbounded out-of-domain emission** (HDR t2 only): −387/−301 raw values
   on SDR content vs the incumbent's bounded saturation — a production-safety
   differentiator; a dial must fail bounded.

**Proposed calibrated gate form (user decides):** per axis, (a) candidate
extreme count ≤ the best peer's count (comparative — beats the field), AND
(b) absolute ceiling max|z| ≤ 35 (excludes cvvdp-class blowups), AND
(c) bounded output (predictions within the dial's declared range ±5) — the
boundedness clause alone formalizes finding 3. Under (a)+(b)+(c): both SDR
candidates PASS, HDR incumbent PASSES on-route (bounded; cross-domain axes
annotated), HDR t2 FAILS (c).

## G-OUT VARIANT STUDY (2026-08-27, user-requested: "try a few variants based on or and zrmse and p99 and p1")

Instrument: `scripts/v_next/outlier_gate_variants.py` — per (model × axis):
**V1** `or` + **V2** `z_rmse` READ from the rank blocks (panel-owned stats, no
new math), **V3a** p99(|chart-z|), **V3b** signed tails p1(z)/p99(z), **V4**
max|z| + emitted pred range. Computed on both SDR candidates, both HDR
candidates, and both classical peers (cvvdp, butteraugli) over
cid22/imazen26/nonphoto/hfnlproxy/kadid/live.

### What each variant can and cannot see (all MEASURED)

- **V1 OR (rate)** — cleanly monotone on the product axes (imazen26: purity
  .009 < incumbent .011 < L1T1 .032 < t2 .034 < cvvdp .038 < butter .045;
  nonphoto same shape; SDR candidates 0.000 on cid22/live). But it is a
  *rate*, blind to *depth*: on hfnlproxy every model sits at .031–.036
  (non-discriminating on exactly the catastrophic axis), and **t2 PASSES OR
  on kadid (.008, better than L1T1's .010) while emitting −387**.
- **V2 Z-RMSE (bulk fidelity)** — good bulk ordering on cid22/imazen26/
  nonphoto, but **actively unsuitable as an outlier gate**: the panel's 4PL
  map-before-compare SATURATES unbounded emissions to the logistic floor
  before the stat is computed. Decisive cell: t2 on kadid emits < −50 on
  **8.78% of pairs** (439/5000, floor −387) yet its z_rmse (0.756) reads
  BETTER than L1T1's (0.802, range [−1,100]). On hfnlproxy the tiny target
  span z-normalizes everyone to ~0.9 and the 47-z cvvdp catastrophics
  vanish. Z-RMSE stays a reported fidelity stat; it cannot carry G-OUT.
  (This measurement is also the panel-native justification for G-OUT
  existing at all: the mapped-space panel structurally cannot see
  out-of-range emissions.)
- **V3a p99|z| (severity, n-stable)** — the workhorse. Catches every named
  class the registered form found, without max|z|'s single-pair noise:
  hfnl depth separates SDR (6.8–6.9) from HDR (21.9/28.2) from peers
  (butter 12.2, cvvdp 47.5); t2's kadid/live problems show (9.5/10.6 vs
  L1T1 2.7/4.9) where OR and Z-RMSE both missed them. Sibling seeds
  s4003/s4005 agree on p99 to ≤0.03 where max|z| is single-pair-driven
  (L1T1 vs t2 nonphoto RANK FLIPS between max 32.1>14.3 and p99 10.4>8.2).
- **V3b p1/p99 signed (diagnostic)** — separates the failure modes: hfnl is
  pure UNDER-prediction (p1 carries everything, p99_z ≈ 2.5–4.2 for all);
  the HDR corruption-cell class is the OVER-prediction tail (p99_z ≈ +9
  on imazen26/nonphoto where SDR sits +2.7); t2's cid22 failure is
  over-prediction (p99_z 6.56 vs p1 −3.29) — scoring bad things good, the
  worse mode for a dial. Keep as reported diagnosis, not a gated bar.
- **V4 max|z| + range** — max|z| demoted to a backstop (only stat that sees
  ONE catastrophic pair; p99 needs ~n/100). The emitted-range check is
  irreplaceable: NO statistical form reliably catches unboundedness
  (OR/Z-RMSE saturate it away; p99 only when frequent).

### Calibrated G-OUT v2 (the synthesis — PROPOSED)

Per candidate class; peers are the calibration, never gated:

| clause | bar |
|---|---|
| **R** rate | axis OR ≤ best-peer OR + 0.005 |
| **S** severity | axis p99\|z\| ≤ min(best-peer p99, 12.0) |
| **B** backstop | axis max\|z\| ≤ 35 |
| **D** bounded | emitted preds within declared dial range ±5 (every axis, on- or off-route) |

SDR candidates gate on all six axes; HDR candidates gate R/S/B on-route
(cid22 + the HDR-route panel) + D everywhere (unboundedness is a model
property, visible anywhere).

**Results under v2:** `W10L9P_s4005_packed` **PASS 6/6 axes, all four
clauses** (p99 margins 1.4–1.8× vs best peer everywhere; hfnl OR .036 ≤
butter .034+.005). Incumbent `W10L9_s4003_packed` PASS (purity better on 4
of 6 p99 axes; incumbent on kadid/hfnl by hairs). `HDR944_L1T1_s4005_hfpack`
PASS on-route (cid22: OR .003 = cvvdp; p99 4.64 ≤ cvvdp 4.68 — hair-thin,
noted; max 7.9; bounded [−28,100] everywhere incl. off-route). `HDR944R_t2`
**FAIL three ways** (R: cid22 OR .017 = 5.7× cvvdp; S: p99 6.6/9.5/10.6;
D: −387/−302). No classical peer would clear S+B on the SDR axes (cvvdp
max|z| 91.8/33.2/37.0; butter unbounded-below by construction). The gate is
therefore simultaneously peer-calibrated, passable by the best models, and
failed by the known-bad candidate — the sense-making criterion.

### G-OUT v2 ACCEPTED AS FINAL FORM (user, 2026-08-27)

The user accepted v2 R+S+B+D. Registered and implemented at the owner —
`scripts/v_next/outlier_gate.py` is now the v2 evaluator (peers passed as
calibration via `--peer`; declared dial ranges decoded from each bake's
`zentrain.output_calibration_spline`; D floor = bottom_knot − span/3, the
neg-tail design's sanctioned extrapolation zone; ceiling 105). **Gated axis
scope = the six axes of the study** (cid22, imazen26, nonphoto, hfnlproxy,
kadid, live — the scope the acceptance was presented over); every other axis
is reported, never silently gated. G-OUT joins freeze_check as an
externally-owned ATTACH row (the existing mechanism for panel-external gates).

Final verdicts (2026-08-27 run): `W10L9P_s4005_packed` **PASS**,
`W10L9_s4003_packed` **PASS**, `HDR944_L1T1_s4005_hfpack` **PASS** (on-route
= cid22 + D everywhere; bounded [−28,100] ≥ floor −32.0),
`HDR944R_t2_s4003_hfpack` **FAIL cid22:R, cid22:S, kadid:D (−387 < −227),
live:D (−302)**.

**Named finding, NOT folded into the gate without a user call:** on konjnd
(outside the gated scope) BOTH SDR candidates exceed the peer rate bar
(OR .038 purity / .032 incumbent vs best-peer+tol .011). KonJND targets are
PJND thresholds, not MOS — the panel's OR band semantics differ there — but
the signal was recorded as ~3× the peer outlier [AMENDED 2026-08-28: INVALID comparison — candidates n=504 vs peers n=1,008, unaligned pair sets; own-set severe outliers show candidates CLEANER than peers 1/0 vs 8/26; campaign H-KON] outlier
rate on the JND axis. Declared spline ranges of record: purity (5.11→87.09),
incumbent (5.42→86.96), L1T1 (0.00→96.14), t2 (−146.92→93.43 — and it still
overshoots its own declared floor by 240).

### SDR freeze — user REQUIREMENT recorded (2026-08-27, not a pick)

"lf and hf both matter and neither can be sacrificed or lose granularity."
⇒ No SDR freeze until a TWO-ZONE scorecard shows, for purity vs incumbent:
LF zone (low-q/heavy-distortion: low bands, corruption blocks, negative
reach) AND HF zone (near-lossless: hfnlproxy, top bands, q≥90 step-1 dial
granularity) — rank + dial granularity per zone, neither zone sacrificed
relative to the other candidate. Scorecard follows below when computed.

## SEED EXTENSION — registered 2026-08-27 BEFORE any new seed trains

Trigger: under the user's two-zone requirement + the accepted G-OUT v2, NO
existing candidate qualifies (all paired-bootstrap, aligned rescored rows,
B=5000, seed 11; instrument `bake_verdict --per-pair-output` on the resliced
ext944 slices — the fulleval per_pair subsamples were row-misaligned across
bakes, so rescoring is the alignment method of record):

| candidate | hfnl Δ vs incumbent | cid22 Δ vs incumbent | G-OUT v2 | verdict |
|---|---|---|---|---|
| s4005P | **−0.0423 [−0.0489,−0.0357]** | +0.0034 [+0.0018,+0.0050] | PASS | HF SACRIFICED |
| s4004P | **+0.0169 [+0.0092,+0.0244]** | −0.0093 [−0.0113,−0.0074] | **FAIL hfnl:B (max\|z\| 45.9)** | gate-ineligible |
| incumbent | — | — | PASS | LF-extreme floor 5.4 (all worse-than-worst collapses to one value); trails imazen26/nonphoto; pre-family training |

Two-zone dial granularity is EQUIVALENT across all three (HF q88-100 med
step 0.36-0.42, tied 0.32; LF q0-45 tied 0.077, mono 1.0) — the zones
separate on RANK and on LF-extreme reach (only s4005P emits below zero,
−5.8; the others floor at their bottom knot). hfnl seed spread within the
recipe is 0.30→0.44 — the recipe reaches HF-strong seeds; whether one seed
can hold cid22 + hfnl + the gate simultaneously is exactly what a seed
extension answers.

**Arms (frozen):** the identical recipe (`~/tmp/sdrpure_argv.txt`, the
embedded-repro argv verbatim, pure views), seeds **s4006..s4011** (6 new;
9 total). Pack = the campaign parity invocation. Harvest = the standard
chain (sota944_verdict + run_full_eval + M3a).

**Selection rule (frozen):** ELIGIBLE = passes G-OUT v2 (gated six axes)
AND two-zone non-sacrifice vs the incumbent: (i) hfnl paired-Δ CI does not
sit wholly below 0, (ii) cid22 paired-Δ CI does not sit wholly below 0,
(iii) LF: cid22/tid bottom-band srocc ≥ incumbent − 0.01 and LF dial mono
= 1.0, (iv) HF dial tied-frac ≤ incumbent + 0.02. Among eligible, PRIMARY =
freeze_check E.4 (profile floors, tie-break bal_comp + 0.15·M3a); below-zero
reach breaks remaining ties. If NO seed in the 9 is eligible ⇒ the recipe
cannot satisfy the requirement; escalate to the konjnd/hfnl-preserving
variant lever WITH the user (no silent recipe change).

### SEED EXTENSION RESULTS — NO ELIGIBLE SEED IN 9; escalation fires (2026-08-28)

6 new seeds trained/packed/harvested clean (driver exit 0). G-OUT v2:
s4006/s4007/s4010/s4011 PASS; s4008 FAIL (hfnl:B max|z|), s4009 FAIL
(live:R+S). Frozen eligibility on the passers (paired-Δ vs incumbent,
aligned rescore, B=5000):

| seed | hfnl Δ (CI) | cid22 Δ (CI) | LF bot-band c22 | verdict |
|---|---|---|---|---|
| s4006P | −0.0033 [−0.0101,+0.0034] **NOT-WORSE — first seed to hold HF** | −0.0028 [−0.0052,−0.0005] ✗ | 0.682 (bar 0.697) ✗ | fails (ii)+(iii) |
| s4007P | −0.2231 ✗ | −0.0118 ✗ | 0.687 | fails (i) |
| s4010P | −0.0813 ✗ | **+0.0059 BETTER** | **0.711 ✓** (tid 0.885) | fails (i) |
| s4011P | −0.2269 ✗ | −0.0045 ✗ | 0.686 | fails (i) |

With the original 3: **0 of 9 seeds satisfies G-OUT v2 + two-zone
non-sacrifice.** The axes anti-separate across the recipe's seed
distribution — the cid22-best seeds (s4010 0.8926, s4005 0.8901) sacrifice
hfnl; the hfnl-holding seed (s4006) pays cid22+LF-band. The recipe's seed
Pareto front does not contain a point dominating the incumbent on both
zones, consistent with the wave's own observation that the family-filtered
tbig rows carried near-threshold/near-lossless signal. **Per the frozen
rule: the recipe cannot satisfy the requirement; the
konjnd/hfnl-preserving VARIANT lever goes to the user** (candidate design,
mirroring the HDR L1 pattern that passed: same pure views + an HF-band
extra group drawn from tbig_944_200k_pure human_score ≥ 0.90 at weight
~1.0 — purity-costless since it is the same family-clean table).

## HF-PRESERVING VARIANT ARM (SPH1) — registered 2026-08-28 BEFORE any fit (user call)

The user chose the escalation's recommended lever. **Arm (frozen):** the
purity recipe verbatim + ONE extra group
`tbig_hf:/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/tbig_hf_pure.parquet:1.0:0.0:both`
— the family-clean tbig table filtered to human_score ≥ 0.90 (11,941 rows,
1,973 refs, targets 0.90-0.984; built this session, same table = zero purity
cost by construction; the HDR L1 target-coherent pattern, NOT GH1's
cross-era mix). Seeds {4003,4004,4005}, stems `W10L9PH_s*`. Pack + harvest
identical to the extension.

**Gates (frozen, unchanged):** the extension's eligibility rule verbatim —
G-OUT v2 + hfnl paired-Δ CI not wholly below 0 + cid22 paired-Δ CI not
wholly below 0 + LF bottom-bands ≥ incumbent − 0.01 + LF mono 1.0 + HF tied
≤ incumbent + 0.02. Selection among eligible: freeze_check E.4; below-zero
reach breaks ties. If 0/3 eligible: report; no further silent arms.

## H-BAL ARM — registered 2026-08-28 BEFORE any fit (balance campaign)

Mechanism twin of SPH1: the SAME purity recipe + the SAME tbig_hf leg but
**val-weighted only** (`tbig_hf:...:0.0:1.0:both`) — HF enters epoch
SELECTION (best_val geomean), never the gradient. Seeds {4003,4004,4005},
stems `W10L9PB_s*`, pack/harvest identical. Gates: the extension eligibility
rule verbatim. Reading: SPH1-vs-H-BAL separates "HF skill needs training
signal" from "HF skill exists in some epochs and selection loses it" — if
H-BAL alone closes the HF gap, the anti-separation was a checkpoint-selection
artifact; if only SPH1 does, it is a data-mix property.

## SPH1 RESULTS — W10L9PH_s4004_packed is the FIRST fully-eligible candidate (2026-08-28)

3/3 seeds clean. The HF leg transforms the recipe — seed-consistent, not a
lottery (hfnl 0.729/0.752/0.734 vs purity twins 0.30-0.44):

| seed | cid22 Δ vs incumbent (CI) | hfnl Δ (CI) | LF bots c22/tid | HF tied | G-OUT | negmin |
|---|---|---|---|---|---|---|
| **PH_s4004** | **+0.0060 [+0.0044,+0.0077] BETTER** | **+0.3320 [+0.3150,+0.3487] BETTER** | **0.714 ✓ / 0.857 ✓** | **0.228 ✓** | **PASS** | **−18.1** |
| PH_s4003 | −0.0033 ✗ | +0.3086 BETTER | 0.694 ✗ / 0.871 | 0.446 ✗ | PASS | 7.8 |
| PH_s4005 | −0.0079 ✗ | +0.3133 BETTER | 0.683 ✗ / 0.871 | 0.435 ✗ | PASS | 7.8 |

**`W10L9PH_s4004_packed` (sha 61ebc4562c2c4f78…) satisfies EVERY frozen clause — by
domination, not trade**: better than the incumbent on BOTH zones with CIs
excluding zero, LF bands over bars, best HF tied-rate in the field, G-OUT
v2 PASS with the cleanest hfnl tail ever measured (OR 0.013, p99 3.87 —
half the previous best), below-zero reach (−18.1, the LF-extreme contract),
konjnd 0.501, imazen26/nonphoto above incumbent. m3a 0.7628 is its one
soft spot (tie-break input; the campaign's terminal read stands between it
and any freeze proposal regardless).

Provenance note (asked and answered before celebrating): the HF train leg's
target is the ssim2-derived label — the SAME metric family as the hfnl eval
target — so this is on-task skill on disjoint families, not cross-metric
proof; cross-metric transfer is measured ONCE by the sealed H-HID panel
(7 independent targets). The +0.006 on cid22 (human MOS, fully held out) is
the first human-labeled evidence the HF skill costs humans nothing.

Mechanism reading (with H-BAL pending): the cid22↔hfnl anti-separation was
a DATA-MIX property — the family purge removed the recipe's only HF-band
signal; restoring it from the family-clean side of the SAME table both
lifts HF ~3× and cleans the HF outlier tail. H-BAL (val-only twin) will
show whether selection alone could have found any of this (prediction: no).

### Instrument disagreement recorded + SPH1-M3a seed extension registered (2026-08-28, pre-fit)

freeze_check E.4 over the full 14-candidate pool selects `W10L9P_s4005_packed`
(sel_comp 0.9876) — but ONLY via the M3a tie-break term: on floors+bal_comp
alone `W10L9PH_s4004_packed` LEADS the pool (8/8, bal_comp 0.8599, highest
sdr25 0.9770). The HF leg costs M3a systematically (SPH1 seeds 0.81/0.76/0.73
vs purity twins 0.83/0.88/0.87 — the HF gradient reshapes coarse
attribution). The wave's registered selection rule (eligibility-first, E.4
among eligible) stands: PH_s4004 is the sole eligible candidate and thus the
wave's selection; the whole-pool E.4 row is the comparator record. M3a is a
real steering-coherence property — so, using the measured 42.3% M3a seed
variance: **SPH1-M3a EXTENSION (frozen): 3 more SPH1 seeds {4006,4007,4008};
eligibility clauses unchanged; among eligible seeds prefer highest M3a.**
Queued behind H-BAL. If no new seed is both eligible and ≥0.83 M3a, PH_s4004
stands as the wave candidate with its M3a honestly recorded.
LF-severe addendum: PH_s4004 corruption q10 0.0774 / q20 0.1920 — best of all candidates (incumbent 0.061/0.188); worst families unchanged (aliasing, channel swaps — corruption-head territory). The two-zone LF case is complete on every measure.

## H-BAL RESULTS — prediction confirmed; the mechanism question is closed (2026-08-28)

Val-only HF signal (identical leg, 0:1.0) recovers only ~20-25% of the HF
gap (hfnl 0.476/0.469/0.398 vs SPH1's 0.729/0.752/0.734) while CRATERING
the product axes (imazen26 0.8755/0.8701/0.9239 vs ≥0.92 for every other
family member; cid22 0.870-0.881) — epoch selection under an HF-weighted
geomean lands on checkpoints that TRADE product skill for HF, because no
gradient ever taught both. **The cid22↔hfnl anti-separation is decisively a
data-mix property: HF skill must be in the GRADIENT (SPH1), where it is
additive; in the SELECTION (H-BAL) it is only exchangeable.** All three
H-BAL seeds ineligible on their face (imazen26 collapse).

Unexpected observation, recorded not chased: H-BAL checkpoints carry the
POOL'S HIGHEST M3a (0.9308/0.9323/0.8689 vs SPH1 0.73-0.81, purity
0.83-0.88) — earlier/HF-val-selected epochs have far more coherent
attribution maps, consistent with the E-M coarse-mass-drift mechanism. The
M3a↔rank trade across training epochs is a real structure worth a future
registered study (checkpoint-level M3a trajectory), not an ad-hoc chase.

### SPH1-M3a EXTENSION RESULTS — fallback fires; PH_s4004 stands (2026-08-28)

| seed | cid22 | hfnl | M3a | G-OUT | eligibility |
|---|---|---|---|---|---|
| s4006 | 0.8814 | 0.7647 | 0.8219 | FAIL kadid:S (p99 4.48) | out |
| s4007 | 0.8929 | 0.7580 | 0.8386 | **FAIL kadid:D, live:D (−71/−27 vs floor −18.3)** | out |
| s4008 | 0.8799 | 0.7457 | 0.7781 | PASS | cid22 −0.0069 CI-sig → out |

s4007 is the instructive cell: pool-best cid22 + high M3a, but it emits 63
points below its declared floor — the t2 unboundedness class in miniature,
and exactly what clause D exists to catch. Across 6 SPH1 seeds the frozen
eligibility passes EXACTLY ONE (s4004) — the filter is strict, not a rubber
stamp. **Registered fallback: `W10L9PH_s4004_packed` stands as the wave
candidate with M3a 0.7628 honestly recorded** (sha 61ebc4562c2c4f78…).

## SPH1-BROAD ARM — registered 2026-08-28 ~04:5xZ BEFORE any fit (user call)

Tests breadth-vs-purity on the hidden non-codec robustness axis. **Arm
(frozen):** the SPH1 recipe with the UN-PURGED bigcodec base tables
(`tbig_944_200k.parquet` 208,169 rows + `sota944/teacher/tbig_teacher944.parquet`
— the incumbent's own training breadth) + the FAMILY-CLEAN HF leg
(`tbig_hf_pure`, so the HF signal carries no new leakage; the discount is
confined to the base tables). Seeds {4003,4004,4005}, stems `W10L9PBR_s*`,
pack/harvest identical. **Leakage discount, stated up front:** its
imazen26/nonphoto/hfnl board reads are family-exposed (the un-purged base
holds 5 test + 12 validate ids) and are REPORTED but not rankable against
family-clean candidates; the arm's DECIDING axes are cid22 (human,
fully-held-out), the semi-hidden secondaries, and post-hoc codec-like/
non-codec subgroup reads of the (already-seen) hidden panel. Gates:
G-OUT v2 + cid22 paired-CI vs incumbent AND vs PH_s4004.

## CLAUSE-(iv) AMENDED BY USER → G-GRAN; eligible set = {PH_s4004, e060} (2026-08-28)

User ratified the granularity goals and the amendment ("yes, amend the gate
to span+reach+mono"; operationalization + measured basis in the campaign
md). Recomputed: **W10L9PH_s4004_packed G-GRAN PASS** (tied diag 0.421) and
**PH_s4004_e060 G-GRAN PASS** (tied diag 0.474) against incumbent-reach
non-inferiority (avif 96.2 / jpeg 94.4 / jxl 96.6 / webp 91.9). e070 stays
out (cid22 CI-sacrificed — clause ii, untouched by the amendment). The
final's earlier 0.228 tied figure was the cross-codec pool; per-codec max
is 0.421 (webp) — recorded for honesty. Selection between the two eligible
candidates: E.4 among eligible per the frozen wave rule, pending the
user-called FINE trajectory pass (may add a dominating checkpoint).
