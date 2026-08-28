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
