# Issue #50 — the ~95.7 top cliff: mechanism, measured (2026-08-02)

**Verdict: the cliff and the plateau are MODEL-RAW saturation — a
training-data gap in the sub-threshold region — not a spline, winsor, or
clamp artifact. A spline-only refit cannot fix it. The fix is near-top
training anchors (the issue's own proposal), which belongs in the next
model wave (PLAN_SOTA944 P3).**

## Instrument

`zensim/examples/issue50_topcliff.rs` (committed; `custom-profiles`
feature). Reproduces the issue's perturbation class — shift `frac` of
pixels by ±`codes` on all three channels, random sign, clamped,
deterministic xorshift64* seed `0x5EED_0050` — and scores every pair
twice through the PUBLIC pipeline:

1. `ZensimProfile::B` (the shipped default, `codec_target()`): B sets
   `skip_score_mapping` + `extrapolate_score`, so
   `score == spline(model_raw)`.
2. `ZensimProfile::Custom` with B's exact runtime flags but the bake's
   `zentrain.output_calibration_spline` **stripped via the canonical
   `bake_dial_refit strip`** (no ad-hoc byte edits; no duplicated
   forward): output = the PRE-spline model raw.

Bakes: deployed B `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`
(sha256 `b6fe5233ee9c752d…`); stripped twin sha256 `5ec68b1f828615ad…`
(matches the byte-identity fixture recorded for `bake_dial_refit strip`
in CLAUDE.md — independent provenance cross-check).

Images: CID22 validation originals (never trained), picked as the
darkest / brightest of the first 40 by mean luma — `1544947.png`
(mean 53.1, 512×512) and `1418519.png` (mean 194.9, 512×512).

Repro:

```sh
cargo build --release -p zensim-validate --bin bake_dial_refit
target/release/bake_dial_refit strip \
    --in zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin \
    --out /tmp/b_nospline.bin
cargo run --release --features custom-profiles --example issue50_topcliff -- \
    --stripped-bake /tmp/b_nospline.bin <dark.png> <bright.png>
```

## Measurements

| image | frac | codes | score (B) | model raw (pre-spline) |
|---|---:|---:|---:|---:|
| 1544947 (dark) | identical | 0 | 100.0000 | (identity short-circuit) |
| | 0.0001 | 1 | 96.0680 | 1.146917 |
| | 0.001 | 1 | 95.9967 | 1.143123 |
| | 0.01 | 1 | 95.7990 | 1.134714 |
| | 0.05 | 1 | 95.1706 | 1.124822 |
| | 0.25 | 1 | 94.0881 | 1.112500 |
| | 1.0 | 1 | 94.0030 | 1.111496 |
| | 0.05 | 2 | 92.8596 | 1.092839 |
| | 0.25 | 2 | 91.5449 | 1.063768 |
| | 1.0 | 2 | 90.8168 | 1.046171 |
| | 1.0 | 4 | 80.7246 | 0.839290 |
| | 1.0 | 8 | 48.0608 | 0.110303 |
| 1418519 (bright) | identical | 0 | 100.0000 | (identity short-circuit) |
| | 0.0001 | 1 | 96.0591 | 1.146435 |
| | 0.001 | 1 | 95.9839 | 1.142455 |
| | 0.01 | 1 | 95.8220 | 1.135313 |
| | 0.05 | 1 | 95.1640 | 1.124741 |
| | 0.25 | 1 | 94.3525 | 1.115512 |
| | 1.0 | 1 | 94.4416 | 1.116505 |
| | 0.05 | 2 | 92.9290 | 1.094200 |
| | 0.25 | 2 | 91.8833 | 1.071773 |
| | 1.0 | 2 | 91.5899 | 1.064847 |
| | 1.0 | 4 | 84.2442 | 0.900506 |
| | 1.0 | 8 | 52.4665 | 0.200180 |

Raw grid TSV: `~/tmp` copy is ephemeral; the table above is the record.
(Issue table replicates: the issue's harness reported the cliff at
~95.7–95.75 for 0.1%×1 on 480×640 content; here 95.98–96.00 on 512×512
CID22 content — same shape, same order.)

The deployed B spline (30 knots, decoded from
`zentrain.output_calibration_spline`): the **dense data-fit knots end at
raw = 1.13786 → score 95.894**; above that sit the synthetic
`extend-top` concave-saturation knots (1.383→98.09, 1.628→99.12, …,
4.076→99.9996). The last two dense segments are steep: 45 then **72
score-points per raw unit** — versus 9.0/unit in the first extension
segment.

## Mechanism (measured, not hypothesized)

1. **The model's raw output saturates at ≈1.147 as perturbation → 0.**
   26 perturbed pixels in 262,144 (0.01% × 1 code) → raw 1.1469/1.1464
   on both images. The smallest representable non-identity distortion
   lands the raw AT the end of the spline's data-fit domain (1.1379).
   `spline(≈1.147) ≈ 96.07` — that IS the cliff. Score 100 is produced
   only by the `is_identical` byte-equality short-circuit, so the
   (≈96.1, 100) score band is structurally unreachable for
   perturbation-class inputs.
2. **The plateau is raw-space compression.** From 0.01%×1 to 100%×2 —
   a ~20,000× distortion-mass range — the raw moves only
   1.147 → 1.046 (Δ ≈ 0.10). The spline maps that sliver to
   [90.8, 96.1]. The model barely discriminates within the
   sub-threshold class; the spline faithfully reports what little raw
   variation exists.
3. **The mild non-monotonicity is upstream of the spline too**: on both
   images 100%×1 vs 25%×1 invert by ≤0.09 score points, and the
   inversion is present in the raw column (1.1115 vs 1.1125 dark;
   1.1165 vs 1.1155 bright — opposite signs on the two images, i.e.
   feature-noise-level, not systematic).
4. **What it is NOT:** not the ≤100 extrapolation cap (raws never reach
   the extension domain for this input class); not the winsor guards
   (they clip the p99 upper FEATURE tail — near-lossless features are
   tiny); not the identity short-circuit malfunctioning (it fires only
   on byte-identical pairs, correctly).

Consistency with the profile record: `profile.rs`'s B documentation
already notes real-content raw tops out ≈1.12 and that near-lossless
*codec-knob* configs reach raw up to ~2.8 (multiband-anchor rows, dial
climbing to ~96.1 there). Codec residuals are structured and can push
some features harder than uniform random ±1-code noise, so parts of the
codec near-lossless ladder DO climb into the extension — the
perturbation class of this issue does not.

## Why a spline-only fix is wrong (and was not applied)

A monotone spline refit is rank-invariant, so it is *allowed* — but the
sub-threshold information simply is not in the raw: Δraw ≈ 0.10 across a
20,000× distortion range, with per-image feature noise ≈0.001–0.005 raw
(the non-monotone cells). Remapping [1.05, 1.15] onto [90, 100] would
amplify that noise ~2× in score units, shift every consumer's scores in
the top band, and still not make 0.1%×1 distinguishable from 1%×1 beyond
what the raw carries. The bottleneck is the model, not the mapping.
The deployed extend-top extension already covers raws above 1.14
correctly for the codec-ladder cases that produce them.

## Recommended fix (for the P3 / next training wave)

Exactly the issue's proposal, now with measured shape:

1. **Add near-top synthetic anchors to training**: `(ref, ref ± k-code
   perturbation)` pairs spanning frac ∈ [1e-4, 1] × codes ∈ {1, 2},
   with ssim2-tracked targets (ssim2 spreads this region 98.6 → 90.9)
   and ≈100 pins for imperceptible cells (≤1 code, ≤~1% of pixels).
   This teaches the model to keep raising raw as distortion → 0 instead
   of flattening at the training corpus's ceiling (the ssim2-anchored
   corpus tops out where codecs stop producing pairs — the dense knots
   end at raw 1.1379 because the DATA ends there).
2. KonJND-1k PJND pairs are the natural external calibration for where
   "visually lossless" should sit on the dial (already a first-class
   `bake_verdict` corpus).
3. Gate: held-out CID22/KADID must not regress in the 60–95 band
   (`bake_verdict` before/after), HF-NL band not below B's 0.614.

No production behavior was changed by this diagnosis; the deployed bake
and its gates are untouched (`bake_verdict` unaffected by an
instrument-only commit).
