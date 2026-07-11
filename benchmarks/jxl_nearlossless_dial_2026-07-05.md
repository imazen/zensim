# zenjxl near-lossless sweep → B dial-top characterization (2026-07-05)

_AI-authored (Claude). Numbers are measured, not estimated. Raw data under
`/mnt/v/output/zensim-jxl-nearlossless/`._

## Motivation

B's dense-dial (`b_sdr_linear_cid80_dense_dial_2026-07-05.bin`, sha `b78adb15`)
had its **95–100 segment fit on a 600-row multiband-anchor extrapolation**, never
on real near-lossless codec output. The 5.7M picker sweeps stop at q90–95, so
pred_b topped at ~91–95 there with 0% of cells reaching 95. To calibrate the
dial-top on real data we swept genuine near-lossless zenjxl.

## Part 1 — zenjxl distance curve + a real encoder/decoder bug (issue #18)

`zenmetrics sweep --codec zenjxl --knob-grid '{"distance":[…]}'`, decoded via
`zenjxl-decoder`, scored with ssim2 (mean over 4 diverse sRGB refs; identical at
generic_quality q=90 and q=99):

| distance | ssim2 | file KB | note |
|--:|--:|--:|---|
| 0.01 | 33.94 | ~730 | **BROKEN** |
| 0.02 | 33.95 | 594 | **BROKEN** (largest file, worst quality) |
| 0.03 | 96.02 | 518 | lossy ceiling |
| 0.04 | 95.77 | 465 | |
| 0.05 | 95.65 | 427 | |
| 0.10 | 94.95 | 319 | |
| 0.20 | 94.06 | 235 | |
| 0.50 | 92.49 | 161 | |
| 1.00 | 87.50 | 125 | |

**zenjxl round-trip is broken at distance ≤ 0.02**: it spends the *most* bits yet
produces ssim2 ~34, a sharp cliff below 0.03. Filed as
[imazen/zenjxl#18](https://github.com/imazen/zenjxl/issues/18); fix in flight
(encoder-vs-decoder disambiguation via jxl-oxide reference decode).

**Consequence:** lossy zenjxl tops at ssim2 ~96 (distance 0.03); ssim2 97–100 is
reachable only via true lossless (=100 via `is_identical`) **until #18 is fixed**.

## Part 2 — B dial gap on real near-lossless (2200 cells: 200 refs × 11 distances)

Forwarded B and A over the stored 372-feature vectors (`rescore_parquet`,
bit-exact), joined to the sweep's ssim2:

| dist | n | ssim2 | pred_b | pred_a | B−ssim2 | A−ssim2 |
|--:|--:|--:|--:|--:|--:|--:|
| 0.03 | 200 | 95.58 | 91.39 | 95.37 | **−4.19** | −0.21 |
| 0.04 | 200 | 95.44 | 91.46 | 95.24 | −3.99 | −0.21 |
| 0.05 | 200 | 95.24 | 91.42 | 95.08 | −3.82 | −0.16 |
| 0.07 | 200 | 94.97 | 91.38 | 94.83 | −3.59 | −0.14 |
| 0.10 | 200 | 94.68 | 91.33 | 94.52 | −3.35 | −0.16 |
| 0.15 | 200 | 94.29 | 91.26 | 94.11 | −3.03 | −0.17 |
| 0.20 | 200 | 94.00 | 91.22 | 93.83 | −2.77 | −0.17 |
| 0.30 | 200 | 93.45 | 91.14 | 93.31 | −2.31 | −0.14 |
| 0.50 | 200 | 92.39 | 90.91 | 92.49 | −1.47 | +0.10 |
| 0.70 | 200 | 91.19 | 89.64 | 91.35 | −1.55 | +0.16 |
| 1.00 | 200 | 89.88 | 88.24 | 90.30 | −1.65 | +0.42 |

**Findings:**

1. **A is essentially ssim2-calibrated at near-lossless** (A−ssim2 within ±0.42
   across the whole range).
2. **B under-scores by 1.5–4.2 points, worst at the top** (distance 0.03:
   −4.19). In the ssim2≥93 zone (n=1488): pred_b mean **91.36** vs ssim2 **95.00**
   — B is **3.6 points low**. B reaches 95+ in only **5 of 2200 cells**
   (pred_b max 96.43).
3. This is a **dial/calibration defect, not a rank defect**: pred_b is monotonic
   in distance (91.39→88.24), just compressed into an 88–91 band instead of
   90–96. A rank-invariant output-spline refit stretches it back.

## Plan

1. **[in flight]** Fix zenjxl #18 → unlocks distance 0.01–0.02 (the true ssim2
   96–100 lossy top), or explicitly routes sub-floor distance to lossless.
2. **Re-sweep** including the now-working near-lossless distances → complete real
   curve to ssim2 ~99.
3. **Refit B's dial-top** (rank-invariant PCHIP/concave-saturation spline) on the
   real `(raw_b, ssim2)` near-lossless pairs, replacing the 600-row extrapolation,
   so near-lossless resolves toward 95–100 instead of piling at 91.
4. Re-validate via `bake_verdict` — rank panel unchanged (spline is monotone),
   dial-top corrected, G-RANGE tail gate still PASS.

## Part 3 — zenjxl #18 fixed + dial-top refit attempt (extend-top FALSIFIED)

**#18 fixed** (jxl-encoder@`008499e1`, on origin/main; #18 closed): i16→i32 DC
widening + `VARDCT_MIN_LOSSY_DISTANCE = 0.03` floor. Root cause was DC stored as
i16 saturating at distance ≲0.025.

**UPDATE 2026-07-06 — the floor is now REMOVED (jxl-encoder@`eeb52735`, #94 fixed).**
The real root cause wasn't the ANS coder: it was a **header lie** —
`modular_16bit_buffer_sufficient = true` was set unconditionally even when DC > i16,
so spec-conformant decoders (jxl-oxide) reconstructed DC into i16 buffers and
**wrapped** (frymire DC max 43280 > i16 32767), desyncing the DC predictor → the ANS
final-state check failed. zenjxl-decoder used wide buffers so never saw it. Fix:
`force_modular_32bit` flag emits `modular_16bit_buffer_sufficient = false` when DC
overflows i16; floor deleted; Huffman alphabet sized to the actual max token.
Verified: jxl-oxide now ACCEPTS distance 0.005 (PSNR 77.1 dB ≈ ssim2 99+), 0.01
(74 dB), 0.02 (67.5 dB); **≥0.03 byte-identical (hash-proven across ANS+Huffman)**.

So lossy zenjxl **now reaches ssim2 ~96.85** (distance 0.005; measured end-to-end in
Part 5 — the earlier "96–100" over-read the 77 dB PSNR). This **STRENGTHENS the A-vs-B
conclusion**: B's projection saturates near dial 91 even harder as distortion → 0
(§ Part 4 feature-vanishing), so as lossy quality climbs to ssim2 ~96.85, B's dial gap
*widens* (Part 5 measured −4.24 → −5.30) — B is an even worse near-lossless knob than
Part 2 showed, A even more clearly the better one.

**The exact-ssim2 re-sweep over the newly-unlocked range is ABANDONED as cosmetic
(2026-07-06).** It needs a zenmetrics rebuild, which requires completing the in-flight
**zencodec#103 (Pattern-B) migration** — local zenjpeg 0.9.0 / zenwebp 0.5.0 / zenjxl
0.3.0 all consume the *unreleased* `zencodec 0.1.26` `CategorizedError`/`CodecError`
API via a git-rev `[patch.crates-io]` (`fde07d0`, only zenmetrics carries it;
standalone consumers fail with 16 errors *inside zenjpeg*), and zenwebp 0.5.0 dropped
its `zencodec` feature gate. That's an ecosystem migration partly blocked on the
unreleased crate — not a pin bump, and not worth it here: the PSNR table above already
confirms near-lossless works (0.005 → 77 dB ≈ ssim2 99+), and the exact curve only
strengthens A-over-B. Speculative consumer pin-bumps were made and reverted; sibling
repos left clean. Redo as a deliberate project only when `zencodec 0.1.26` is ready.

**extend-top on real near-lossless — FALSIFIED.** Built a near-lossless anchor
(2200 cells, `feat_0..371` + `target_score`=ssim2, target 78.3–100) and ran
`bake_dial_refit extend-top` from the winsor bake. Result: **0/2200 near-lossless
cells moved** (cand−ship = +0.00 at every distance; candidate 12988 B). extend-top
fits a concave saturation *above* the in-distribution top knot (linear-raw 1.138,
which maps to dial 95.9), but real zenjxl near-lossless sits *below* it (dial
88–91). The top extension never touches it — the under-scoring is a **mid-spline /
raw-projection** property, not a top-knot pile-up.

**What the compression is.** B's linear projection ranks zenjxl near-lossless
(ssim2 89.9–95.6) at dial 88–91 (~0.55× the resolution of ssim2's spread). The
top-knot dial (95.9) sits at linear-raw 1.138, reserved for near-perfect content;
lossy near-lossless (linear-raw ~1.0–1.1) maps just below it. This is B
**disagreeing with ssim2 on absolute near-lossless quality** (B: "very good, not
perfect"; ssim2: 95.6). Whether B's 88–91 or ssim2's 95.6 is closer to human MOS
is **UNVERIFIED** (no human MOS on zenjxl near-lossless), but it matches the
established pattern: B human-MOS-aligned, A/ssim2 ssim2-tracking.

**Resolution — reinforces the A-vs-B tradeoff.** A tracks ssim2 at near-lossless
(91–96, full resolution) → **A is the better near-lossless codec KNOB**; B
compresses it (its human-MOS-aligned judgment) → B is the human-MOS RANK metric.
The near-lossless data thus **reinforces keeping A as `codec_target`**. Making B
match ssim2 at near-lossless would need a mid-spline remap (shared-anchor, with the
documented bottom-knot pitfall) AND would erode B's human-MOS distinction — not
recommended unless B is chosen as the codec knob, in which case the remap is the
lever (feasibility of stretching the projection's near-lossless resolution is the
open question).

## Part 4 — why B compresses near-lossless: winsor vs feature-vanishing (2026-07-06)

Question (user): a linear projection is continuous — how does it produce the
near-lossless ties/compression? Winsor clamps? **Measured decomposition:**

- **Winsor IS clamping hard at near-lossless** — but via the LOWER `p1` clamp, not
  `p99`. At distance 0.03, ~310/372 features fall below their `p1` floor (mean 249.9
  low-clamped vs 0.6 high-clamped) because near-lossless distortion features are
  *tiny* — below the 1st percentile of the heavier-distortion training set. Clamp
  count is monotonic in distance: 310 @ d0.03 → 95 @ d1.0.
- **But isolating winsor** (B's real ens weights/scaler, clip(F) vs F) it costs only
  **8–21%** of the near-lossless projection variance (11% @ d0.03, peak 21% @ d0.1).
- **The dominant cause is feature-vanishing.** Even UNWINSORED, B's near-lossless
  projection std is 0.004–0.012 and the range collapses to `[1.00, 1.02]` at
  d0.04–0.05. As distortion features → their zero-distortion values, a linear
  dot-product → a near-constant output. There's no residual signal to spread.

So B's near-lossless compression ≈ **~80% feature-vanishing + ~15–20% winsor**. **A
does better not by avoiding winsor** (same 372 features) **but because its
nonlinearity up-weights the few features that retain near-lossless signal** —
spreading 91–96 where B's linear fit compresses to 88–91.

**Fix implication:** removing winsor recovers only ~15–20% AND reintroduces the f155
tiny-screen pathology it guards (raw −1131 → "webp −80"). The real lever for B's
near-lossless resolution is up-weighting the near-lossless-discriminating features (a
near-lossless-weighted re-fit, or the mid-spline remap) — the "make B ssim2-shaped at
the top" tradeoff. A pure winsor removal is not the fix.

## Part 5 — real near-lossless re-sweep, fix confirmed end-to-end (2026-07-06)

Rebuilt zenmetrics against the fixed jxl-encoder and re-swept **200 refs × 6
newly-unlocked distances → 1200/1200 cells, 0 failures**. (Build path in the note
below; sources were converted PNG→lossless-JXL so the minimal png-less build could
decode them — exact pixels, so the reference is unchanged.)

| dist | ssim2 | pred_b | pred_a | B−ss | A−ss |
|--:|--:|--:|--:|--:|--:|
| 0.005 | 96.85 | 91.56 | 96.26 | −5.30 | −0.60 |
| 0.01  | 96.53 | 91.55 | 96.05 | −4.97 | −0.48 |
| 0.015 | 96.18 | 91.54 | 95.83 | −4.64 | −0.35 |
| 0.02  | 96.00 | 91.53 | 95.69 | −4.47 | −0.31 |
| 0.025 | 95.88 | 91.51 | 95.58 | −4.37 | −0.30 |
| 0.03  | 95.73 | 91.49 | 95.44 | −4.24 | −0.29 |

**Findings:**
1. **Distance 0.005–0.02 (previously broken at ssim2 ~34) now produce ssim2
   96.0–96.85** — the header-lie fix (#94) is validated end-to-end through the actual
   sweep pipeline, not just the agent's standalone jxl-oxide test. 0.03 = 95.73
   matches Part 2's 95.58 (sanity ✓).
2. **CORRECTION to the Part-3 "96–100" claim: the lossy ceiling is ssim2 ~96.85
   (distance 0.005), NOT 100.** The 77 dB PSNR ≈ ssim2 ~97, not 99+. The whole lossy
   near-lossless band is compressed into ssim2 95.7–96.9; true 97–100 still needs
   actual lossless (distance 0). So the fix extends the valid lossy curve by only
   ~+1.1 ssim2 above the old 0.03 floor — the practical near-lossless win is
   conformance/monotonicity, not a big ssim2-range gain.
3. **B saturates at 91.5 — nearly FLAT (91.49→91.56) across the whole range** — while
   A tracks ssim2 within ±0.6. As quality climbs 95.7→96.85, **B's gap WIDENS
   −4.24 → −5.30** — exactly the Part-4 feature-vanishing prediction. A is the better
   near-lossless knob; this reinforces keeping A as `codec_target`.

**Build note (zencodec#103 migration).** The zenmetrics rebuild needed the in-flight
Pattern-B migration patched in — zenmetrics already carries the `zencodec` git-rev
`[patch.crates-io]`; on top of that it needed +3 consumer `zenjpeg ^0.8→^0.9` pins and
removal of one stale `zenwebp?/zencodec` feature ref (zenwebp 0.5.0 made `zencodec`
always-on). Two genuine zenmetrics bugs surfaced, logged for a deliberate fix:
(a) the stale `zenwebp?/zencodec` ref; (b) `zenmetrics-cli/src/hdr.rs::rgb16_hlg_to_nits`
does `use cvvdp::params` under `#[cfg(feature="png")]` but `cvvdp` is never a direct
CLI dependency → any png-without-cvvdp build can't compile. The speculative consumer
pins were reverted (siblings clean); the sweep used PNG→lossless-JXL sources to
sidestep (b) via the already-working minimal (png-less) build.

## Part 6 — "fix B's flaw": the linear re-fit FAILS (human-MOS ⊥ near-lossless), 2026-07-07

User directive: "fix this flaw of B." Full diagnosis + a 3-way falsification that the
flaw is **intrinsic to B being a linear projection**, not a spline/calibration bug.

**The flaw is the projection, not the spline.** B's near-lossless cells project to raw
~1.06 → dial 91.5 with only 22% per-image monotonicity (vs A's 84%; per-image
SROCC-vs-ssim2 0.46 vs A 0.93). Decoding B's 30-knot spline: the near-lossless region
(raw 1.03–1.14) is **monotone** (slope 45–72 dial/raw), *not* flat (flats are only at
raw >2.5 → dial ~100, which real zenjxl never reaches). A monotone spline preserves
rank exactly ⇒ **a rank-invariant spline remap cannot manufacture rank the projection
lacks.** (The prior session's `full/b_nl_candidate.bin` was exactly this remap — same
sparse layer, near_zero 0.745 — and was falsified: 0/2200 cells moved.)

**A linear re-fit is antagonistic — FALSIFIED (POC, ridge on 5 human-MOS groups +
zenjxl near-lossless sweep at weight w):**

| w_nl | CID22 | KonJND | nl per-img SROCC | nl raw span |
|--:|--:|--:|--:|--:|
| 0.0 | 0.843 | 0.246 | +0.771 | 0.094 |
| 0.25 | 0.822 | 0.210 | **−0.143** | 0.092 |
| 0.5 | 0.799 | 0.181 | −0.657 | 0.092 |
| 1.0 | 0.759 | 0.138 | **−0.829** | 0.095 |
| 5.0 | 0.611 | 0.023 | −0.543 | 0.100 |
| 10.0 | 0.528 | 0.016 | +0.657 | 0.112 |

At usable weights, near-lossless supervision **flips the near-lossless ranking negative**
(the fit can't satisfy it linearly → it corrupts) AND costs CID22; the dial raw-span
barely moves (reach not fixed). A **1.4%-weight** near-lossless contribution (w=0.25)
already flips the sign ⇒ B's near-lossless ranking is **ill-conditioned / knife's edge**
(near-constant features → prediction is a tiny difference of large terms). Only w=10
recovers near-lossless (+0.657) — but CID22 is destroyed (0.53). A *pure*-ssim2 linear
fit ranks near-lossless (0.829, held-out) but abandons human-MOS: **you cannot have both
in one linear projection.**

**Ceiling context.** ssim2's OWN per-image monotonicity across these distances is only
**46%** (d≤0.03) — the encoder's near-lossless RD isn't monotonic at fine steps, so NO
metric gives a truly clean near-lossless dial. A reaches 84%/0.943 only by nonlinear
over-smoothing past what the encoder/ssim2 actually do.

**Conclusion + recommendation.** B's near-lossless flaw is **intrinsic to linearity**,
falsified 3 ways (spline-remap, strict-mono, antagonistic re-fit). The only fixes are:
(a) bolt a **nonlinear near-lossless head** onto B with a hard regime switch — complex,
introduces a dial discontinuity, and is ssim2-shaped there anyway (no human MOS on
zenjxl near-lossless), i.e. it **recreates A** in that regime; or (b) **use A as the
near-lossless codec knob** (already 0.943, smooth monotone dial) and keep B as the
human-MOS RANK metric. **Recommend (b)** — the "flaw" is B honestly failing to fake a
razor-thin band that isn't linearly separable; making B nonlinear just rebuilds A at
high cost.

## Part 7 — the REAL fix (Part 6 REVISED): B's winsor bounds are miscalibrated, 2026-07-07

User: "the winsorization of every single feature is suspicious, and might contribute
to this." **Correct — it's the wound, and Part 6's "intrinsic to linearity" verdict is
wrong in an important way.** B applies `winsor_p99` (clamp each feature to a baked
`[lo,hi]`) to ALL 372 features, and those bounds are **miscalibrated**.

**Winsor destroys near-lossless discrimination.** Applying B's shipped bounds to the
1200 near-lossless cells: **245/372 features become CONSTANT** (all clamp to `lo`;
pre-winsor 0 constant); 247 lose >90% of their within-near-lossless variance. A linear
projection cannot distinguish cells whose inputs are 245-way identical.

**Recomputing the bounds is a Pareto win (POC: ridge on 5 human-MOS groups):**

| winsor bounds | CID22 | KonJND | nl per-img SROCC |
|--|--|--|--|
| B's shipped | 0.812 | 0.225 | 0.286 |
| clean p1/p99, train-only | 0.861 | 0.291 | 0.829 |
| clean p1/p99 **+ near-lossless** | 0.861 | 0.294 | **0.886** |
| (A nonlinear reference) | — | — | 0.943 |

Two separable effects: (1) B's shipped bounds are simply BADLY CALIBRATED — a clean
p1/p99 recompute alone recovers near-lossless 0.286→0.829 AND lifts CID22 +0.049;
(2) including near-lossless in the bound computation closes the rest (→0.886) at ~0
human-MOS cost. −1131 (the f155 tiny-screen pathology the lower clamp guards) is still
an outlier below the new p1 ⇒ the guard survives. **So the linear ceiling is ~0.83
(still < A's 0.943), but B was crippled to 0.286/0.714 BELOW its own ceiling by
miscalibrated winsor bounds — self-inflicted, not a fundamental limit.** Part 6's
antagonistic-refit was real but was fighting this wound; and Part 3's rank-invariant
spline remap failed only because the OLD near-lossless rank was garbage — with the rank
now correct (0.886) though still razor-thin (raw span ~0.01), a spline refit CAN now
stretch it to the full dial.

**Full fix:** regenerate the auto_transforms TSV with inclusive p1/p99 bounds
(`load_transforms`, `scripts/v_next/train_v02_bvls_shaped.py`) → re-run
gram→fit→ensemble(`Pline-cid80`)→finalize → spline refit.

**CAVEAT (do not over-claim yet):** these are POC numbers — a single ridge on a 5-group
subset, NOT B's actual BVLS ensemble (CID22 0.876). The near-lossless recovery is robust
across every test; the CID22/KonJND *gains* need validating on a real `ens-Pline-cid80`
rebake (the POC's 0.812 B-bounds baseline may be low because B's shipped bounds don't
match the subset). **The rebake IS the validation — DONE, see Part 8.**

## Part 8 — SHIPPED: inclusive-winsor B rebake (2026-07-07)

The Part 7 fix, validated on B's REAL `ens-Pline-cid80` pipeline and shipped as
`ZensimProfile::B`. **No re-fit was needed** — B's winsor is a post-hoc
`add-winsor` guard on raw-fitted weights (weights/scaler/spline copied verbatim
through `winsorize_bake` → `dense_dial_refit_b`), so ONLY the winsor fit corpus
changed. This is why the human-MOS panel is preserved by construction.

**Pipeline** (all deterministic; reproduced by `scripts/reproduce_b.sh`):

```
raw  b_sdr_linear_cid80_anchored_2026-07-04.bin              (sha 7b326ac5)
  → bake_dial_refit add-winsor --fit-corpus <inclusive> --lo-pct 0.1 --hi-pct 99.9   (sha 92189ea1)
  → bake_dial_refit extend-top --anchor multiband_anchor_dial100.parquet --target-col target_score
  → b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin  (7,325 B, sha b6fe5233)  [SHIPPED]
```

**Inclusive fit corpus** (`scripts/v_next/build_inclusive_winsor_corpus.py`, sha
352e7a55): hdr_v3mix (7,410 rows) + zenjxl near-lossless SDR sweep refit+full
(3,400 rows) = 10,810 × 372. The predecessor used hdr_v3mix ALONE → its
[p0.1,p99.9] bounds sat above the SDR near-lossless feature range → 245/372
features clamped CONSTANT there. Adding the near-lossless sweep drops 340/372
lower bounds so those features pass through; f155's upper p99.9 rises only
0.479→0.776, still ≪ the 14,532 offender, so the pathology guard survives.

**Measured** (`bake_verdict` full panel + `predict_features_with_bake` dial +
`ensemble_score_rows` rank, all on the fixed bake bytes):

| | shipped B (2026-07-05) | FIXED B (inclusive-winsor) |
|--|--|--|
| CID22 SROCC | 0.8763 | **0.8764** |
| KonJND SROCC | 0.5474 | 0.5466 |
| TID / AIC-3 SROCC | 0.7869 / 0.7774 | 0.7868 / 0.7774 |
| near-lossless dial (ssim2 tgt ~96) | **91.5** pinned | **96.1** climbs |
| near-lossless per-img rank-vs-ssim2 | 0.657 | **0.771** |

Human-MOS panel preserved within noise; the near-lossless dial band is corrected
from a 91.5 pin to ~96.1 (matching ssim2's 96.0–96.85); near-lossless rank
lifts +0.114.

**Honest gaps.** (1) The per-image near-lossless dial SPAN stays ~0.06
(razor-thin) — the encoder-RD-noise ceiling ssim2 itself shares (Part 6); only
the ABSOLUTE level is corrected, not the fine within-band resolution. (2) Rank
0.771 < A's 0.943 — the residual is the linear ceiling (Part 7), not the winsor.
(3) The fit corpus is HDR + one-codec near-lossless; a broader multi-codec SDR
sweep could tighten bounds further, but the current corpus is a strict
improvement (validated panel + dial). Part 6's "use A as the near-lossless knob"
is SUPERSEDED for the dial LEVEL (B now reaches 96); A remains better for fine
within-band resolution and overall rank.

## Part 9 — why NOT fit the winsor corpus on "the entire quality range + all million jxl encodes" (2026-07-10)

Natural follow-up: if too-narrow a corpus caused the bug, why not fit the bounds on
everything? MEASURED — two reasons, and the minimal corpus is the sweet spot.

| winsor fit corpus | f155 bound (p99.9) | f155=14532 offender raw | guard | CID22 | near-lossless dial |
|--|--|--|--|--|--|
| shipped fix: hdr_v3mix + near-lossless | 0.776 | **+1.09** | ✓ | **0.8764** | 96.1 |
| broad, tiny-screens filtered (valdigits f155≤10 + nl) | 8.7 | +0.48 | ✓ | 0.8732 | 96.1 |
| UNFILTERED everything (valdigits + nl) | 126.2 | **−8.63** | ✗ dial pins 0 | 0.8733 | 96.1 |

("everything" proxied by valdigits 147k for speed; the full 2.95M bigcodec train has
f155 p99.9 = 65.2, max 2975, ~97k rows > 1 — same conclusion a fortiori. The full
add-winsor timed out at 2 min; the percentile shape is identical.)

1. **The guard needs the pathology to stay an OUTLIER in the fit corpus.** The winsor
   clamps the f155=14,532 tiny-screen offender to a small value so its large negative
   weight can't crater the raw score. "All million encodes" INCLUDES those tiny-screen
   renditions (f155 up to 2975), so p99.9 blows out to 65–126, the clamp lands too high,
   and the offender's raw output falls to **−8.63** — far below the spline floor (−1.97)
   — so the dial pins at 0 and the pathology returns. MEASURED, not theorized. Filtering
   the tiny-screens (f155≤10) restores the guard (offender raw +0.48).
2. **Even filtered, there is NO upside.** Near-lossless is already maxed at 96.1 by the
   current fix — broadening doesn't move it. And CID22 is BEST with the tight hdr_v3mix
   corpus (0.8764 vs 0.8732/0.8733 for both broad variants), because B's weights are
   ~67% HDR-trained (the cid head is fit on hdr_v3mix); the hdr_v3mix winsor bounds MATCH
   the distribution those fixed weights expect, and looser bounds let through feature
   variance the weights don't rank as well. Broadening the winsor corpus WITHOUT re-fitting
   the weights is a small net loss (and re-fitting is the Part 6 antagonism).

Conclusion: the minimal corpus (the weights' own training distribution + the specific
near-lossless band that was missing) is correct. "Use everything" breaks the guard
unfiltered and gains nothing filtered. Recorded so it isn't re-tried.

## Data

Part 5 re-sweep: `/mnt/v/output/zensim-jxl-nearlossless/refit/` (pareto.tsv,
features.parquet, nl_b/nl_a.parquet, distorted/ persisted).

- Features (2200×377): `/mnt/v/output/zensim-jxl-nearlossless/full/features.parquet`
- ssim2 + bytes: `/mnt/v/output/zensim-jxl-nearlossless/full/pareto.tsv`
- B/A forwards: `full/nl_b.parquet`, `full/nl_a.parquet`
- Distance-curve smoke: `/mnt/v/output/zensim-jxl-nearlossless/smoke/`
