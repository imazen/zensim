# Dial addressability expectations + G-GRAN v2 (peer-anchored dial gate) — 2026-08-28

User directive: "do all of this, and also establish addressability
expectations at the ends of codec qualities - also consider distance for
jxl, not its quality mapping unless that spline is truly fair. some codecs
are integer qualities and some having meaningful floating point and we
should support both well."

Owner: `scripts/v_next/dial_range_gate.py` (supersedes
`webp_ceiling_audit.py`, same map-inversion construction generalized).
Machine record: `benchmarks/dial_addressability_2026-08-28.json`.
Data: stored 944 dial grid + refmetrics ssim2/butteraugli sidecars.

## Measured structural facts (bake-independent)

**The jxl q display map (q = 100 − 4·distance, linear) is UNFAIR — jxl
runs natively on DISTANCE everywhere in the gate.** At display q88, true
quality is ssim2 75.2 for jxl vs 84.7 jpeg / 83.0 webp / 89.1 avif; jxl's
"q≥88" zone spans 22 true-quality points where webp's spans 7.8. Any
uniform-q zone or bar is cross-codec incomparable.

**Effective knob quantum ≠ grid quantum (measured by per-image f0
encode-identity on adjacent grid steps):**

| codec | knob kind | grid → effective steps | collapsed classes |
|---|---|---|---|
| jpeg | integer-quantum q | 40 → 33 | q0≡5≡10 (bottom floor); q99≡99.25≡…≡100 (top, 6-way) |
| webp | integer-quantum q | 40 → 33 | 96.5≡97, 97.5≡98, 98.5≡99≡99.25, 99.5→100 (4-way) |
| avif | fine-quantum q | 40 → 39 | only 99.9≡100 |
| jxl | float distance (continuous) | 49 → 49 | none |

Forced ties on duplicate-encode knobs are CORRECT behavior — the previous
tied-rate diagnostic counted them against the model (part of webp's
0.42-0.47 tied diagnostics was this artifact). All shape stats (gaps,
mono, ties) now run on the EFFECTIVE ladder.

## Addressability expectations at the ends (peer truth, committed)

| codec | bottom knob → ssim2 / butter | HF entry (truth ≥ 83) | top knob → ssim2 / butter |
|---|---|---|---|
| jpeg | q0 (≡5≡10) → **+22.35** / 7.20 | q86 → 83.24 | q100 (≡99..) → **93.79** / 0.547 |
| webp | q0 → **−12.03** / 14.18 | q89 → 83.91 | q100 (≡99.5..) → **90.76** / 1.187 |
| avif | q0 → **−14.78** / 19.49 | q76 → 83.26 | q100 (≡99.9) → **95.13** / 0.259 |
| jxl | d25 → **+11.30** / 18.26 | d1.6 → 84.38 | d0.025 → **97.04** / 0.066 |

Key end-expectations: jpeg cannot go below ssim2 ~22 (its bottom floor
quantizes at q10-class) — models should NOT emit deep-negative for jpeg
q0; webp/avif bottoms genuinely reach ~−13; jxl's top (d0.025) is the
highest attainable rung of any codec and jpeg's top is ~3 below avif's.
HF-zone truth spans: jxl 12.7, avif 11.9, jpeg 10.6, webp 6.9 — a uniform
span bar across codecs was never fair (webp's old span ≥ 8 exceeded its
zone's true span).

## G-GRAN v2 gate design (provenance-tagged; model-derived bars banned)

1. **Two-sided calibration at three peer anchors** per codec (bottom /
   HF-entry / top): |actual − map⁻¹(truth)| ≤ tol under the bake's own
   monotone translation. tol_top = tol_entry = 1.0 [noise-derived: 2× the
   ±0.5 cross-map spread]. **Cap-aware top**: the portion of `honest`
   above the bounded-scale cap (100) is excused — a [0,100] metric cannot
   emit 101.6 for jxl d0.025; only in-scale under-report fails. Bottom
   tol derives in-run from cross-map honest-bottom spread (23.2 this
   pool) — the bottom anchor is a SANITY BAND only (bottom extrapolation
   of the maps is too uncertain for a tight bar; G-OUT clause D remains
   the real floor guard).
2. **Gaps** [goal-derived: ±2 loop tolerance ⇒ bar 4]: per-image max
   emission step between adjacent effective HF steps, gate p90 ≤ 4 —
   integer-quantum codecs only. For jxl (continuous knob) grid gaps are
   sampling artifacts: diagnostic, with the attainability proxy as jxl's
   gate.
3. **Mono ≥ 0.93** [convention, unchanged] on the effective ladder —
   for jxl this is now the NATIVE-distance ladder.
4. **Attainability** [goal-derived, the unifying product gate]:
   seeded-secant proxy (k=3), integer targets across the honest range,
   gate median |achieved − target| ≤ 2; also reported in translated
   ssim2 units (gaming resistance). Caveat: the proxy quantizes to grid
   cells, UNDER-estimating jxl's true continuous-knob attainability.

## Verdicts on the six finalists (this run)

| bake | fails | reading |
|---|---|---|
| **w11_s4014_e050** | avif:hf_entry(−1.24) | **best-calibrated dial in the pool** — one anchor, 0.24 over tol |
| A_PH_s4004 | jpeg:top(−1.81), jxl:hf_entry(−1.02) | jxl entry hairline; jpeg top real |
| w11_s4014_final | jpeg:top(−2.22), webp:top(−1.74) | top-zone compression |
| B_e060 | jpeg:top(−1.67), webp:top(−1.90) | map champion, weakest top calibration — consistent with its coherence-first trajectory |
| w11_s4012_e080 | jpeg:top(−1.91), jpeg:entry(+1.11), webp:top(−1.32) | only cell with a POSITIVE entry delta (stretch) |
| incumbent_s4003 | jpeg:top(−1.09), jxl:top(−2.46), jxl:**mono 0.89** | jxl dial NON-MONOTONE on the native distance ladder — invisible to the old q-axis check |

Cross-cutting findings: (a) **every bake under-reports jpeg's top zone**
(−0.74..−2.22, honest 95.5-99.2 all inside the scale — no cap excuse):
the family systematically compresses jpeg near-lossless; the top W12 data
target together with webp top. (b) Attainability passes everywhere
(0.26-1.56 vs bar 2) — the dial-gap fear is NOT realized on any finalist;
gaps p90 1.75-3.68, all under the bar. (c) No stretch anywhere at the top
(all deltas ≤ 0) — the era's failure mode is compression, not inflation.

**Scope guard:** this is the DIAL-side gate. Rank-side verdicts are
unchanged — e050's tid-LF regression (paired CI wholly negative) and its
cid22-vs-A trade stand; v2 flipping e050's webp-reach fail to a pass does
NOT make it eligible under the frozen two-zone battery.

**Registration:** G-GRAN v2 is the registered REPLACEMENT CANDIDATE for
G-GRAN v1 in W12's frozen gate set — adoption is user-gated (it swaps
incumbent-derived bars for peer-derived ones; strictly a re-founding, not
a relaxation: v2 fails cells v1 passed, incl. the incumbent itself).
