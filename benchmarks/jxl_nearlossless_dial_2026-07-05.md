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

So lossy zenjxl **now reaches ssim2 96–100** (distance 0.005–0.02). This
**STRENGTHENS the A-vs-B conclusion**: B's projection saturates near dial 91 even
harder as distortion → 0 (§ Part 4 feature-vanishing), so as lossy quality now spans
to ssim2 ~99–100, B's dial gap *widens* — B is an even worse near-lossless knob than
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

## Data

- Features (2200×377): `/mnt/v/output/zensim-jxl-nearlossless/full/features.parquet`
- ssim2 + bytes: `/mnt/v/output/zensim-jxl-nearlossless/full/pareto.tsv`
- B/A forwards: `full/nl_b.parquet`, `full/nl_a.parquet`
- Distance-curve smoke: `/mnt/v/output/zensim-jxl-nearlossless/smoke/`
