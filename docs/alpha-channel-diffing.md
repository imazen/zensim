# Alpha Channel Separation in Image Diffing

## The Bug That Taught This Lesson

In imageflow's dual-CMS backend comparison (moxcms vs lcms2), a CMYK→BGRA
transform had an alpha corruption bug that went undetected because the diff
analysis lumped all four channels together.

The comparison function computed per-pixel max delta across B, G, R, and A,
then checked that against a single threshold. The actual per-channel breakdown
was:

```
ch_max=B3/G2/R2/A255
```

RGB divergence was 3 — normal rounding. Alpha divergence was 255 — a
structural bug. But the single threshold saw max=255, someone set threshold=17
to suppress the warnings, and the bug lived for months.

### Root Cause

lcms2's `apply_transform` saved and restored the high byte of each BGRA u32
(the alpha channel) before/after every color transform, to work around lcms2
zeroing alpha on RGB profiles. For CMYK input, byte 3 is K (black), not alpha.
After CMYK→BGRA transform, the function overwrote the correct output alpha
(255) with the saved K value. Black pixels (K=255, stored inverted as 0)
became fully transparent.

## The Principle

**Alpha divergence between two color management backends is always a bug,
never a rounding tolerance.**

Legitimate CMS rounding differences come from LUT interpolation, float
precision, and gamut mapping — these affect R, G, B channels roughly equally
and are bounded (typically 1-3 for 8-bit). Alpha is never touched by color
management:

- **RGB input**: alpha is passed through unchanged. Both backends should
  produce identical alpha.
- **CMYK input**: there is no alpha. Both backends should produce 255.
- **Grayscale input**: alpha is passed through. Same as RGB.

Any alpha delta > 0 between backends means one of them has a pixel format bug.

## What To Do About It

When comparing two image processing pipelines pixel-by-pixel, separate the
analysis by channel role:

```rust
let max_rgb = max(b_diff, g_diff, r_diff);
let a_diff = a_source.abs_diff(a_reference);

// Alpha divergence is structural — zero tolerance
if a_diff > 0 {
    alpha_bugs += 1;
}
// Color divergence is numerical — bounded tolerance
if max_rgb > rgb_threshold {
    color_divergent += 1;
}
```

This catches two fundamentally different failure modes:

1. **Color rounding** (expected, bounded): LUT grid interpolation, float→int
   rounding, different gamut mapping strategies. Set a small threshold (1-3
   for 8-bit).

2. **Alpha corruption** (unexpected, unbounded): pixel format confusion,
   wrong byte offset, missing alpha restoration. Zero tolerance.

## Generalization for Zensim

The same principle applies to any image similarity metric that handles RGBA:

- If comparing two encoders/decoders and one produces alpha=0 where the other
  produces alpha=255, that's not a quality difference — it's a format bug.
  The metric shouldn't report it as visual divergence; it should flag it
  separately.

- When pooling per-channel errors into a single score, alpha errors can
  dominate and mask real color differences (or vice versa). Keeping them
  separate in the analysis pipeline gives more actionable diagnostics.

- For opaque images (all alpha=255), any alpha divergence in the output is
  immediately suspicious regardless of the color channel results.
