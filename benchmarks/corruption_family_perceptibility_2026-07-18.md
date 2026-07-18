# Corruption-family perceptibility classification (2026-07-18)

Classifies the 44 gb82_dog corruption families by whether the corruption is PERCEPTIBLE,
using butteraugli-max as the structural oracle and q10 (a visibly-lossy honest JPEG — a
"definitely perceptible" calibration point) as the yardstick. Data: the May-28 multimetric
TSV (ssim2/butter-max/cvvdp/dssim on corruption/q10/q20 per recipe).

- **p-butt** = frac(butter_max(corruption) > butter_max(q10)) — corruption worse than visibly-lossy q10 → perceptible
- **p-cvvdp** = frac(cvvdp(corruption) < cvvdp(q10))
- **ssim2✓** = frac(ssim2 ranks corruption below honest q20) — does ssim2 catch it?

## Three buckets

**SUBTLE — plausibly sub-perceptible, should NOT be force-gated (8 families):**
chroma_boundary, noise_bit_flip_n1 (both 0% every metric), noise_bit_flip_n16,
noise_salt_pepper_n1, aliasing, geometric_shift1px, noise_bit_flip_n256, channel_swap_rb.
butteraugli itself says these are subtler than a visibly-lossy q10. Scoring them below q20
would be WRONG (they look near-lossless). This is why butteraugli sits at 72%, not 100% —
and that 72% is closer to correct than cl_tfm's 100% (which over-gates the invisible).

**OBVIOUS — every metric catches (17 families):** block_garbage/zero/gray/copy_wrong,
channel_invert/max_r/zero_g, composite_premul/wrong_bg_*, tone_gamma_*, overlay_rect,
geometric_flip/rotate, edge_border_all_k4.

**ssim2-BLIND — perceptible per butteraugli, MISSED by ssim2 (19 families) = THE gate's job:**
edge_border_* (100% butter / 0% ssim2), overlay_glyph (83/6), overlay_line (67/6),
tone_brightness_* (89/33), channel_swap_gb/rg (83/44, 67/22), noise_salt_pepper_n16/256
(67-83/0-17), channel_zero_r/b, tone_contrast, block_repeat_neighbor, edge_shift/duplicate.
These are real structural breaks (borders, overlaid text/lines, tone/gamma errors, channel
swaps, impulse noise) that a mean-SSIM-shaped scalar is blind to — where corruption-awareness
earns its keep.

## Key findings

1. **~8/44 families are sub-perceptible** → a 100% gate rate is OVER-sensitive, not a win.
   The gate target should be the ~36 perceptible families, not all 44.
2. **cvvdp is NOT a corruption oracle** — p-cvvdp is 0% almost everywhere; cvvdp (a
   contrast/masking model) is even blinder to structural breakage than ssim2, consistent
   with its 32% gate rate. butteraugli-MAX is the reliable severe-regime signal.
3. **butteraugli's 72% ≈ the perceptible set.** It gates obvious+ssim2-blind, skips subtle —
   which is the correct behavior, not a shortfall.

## Implication for the gate

Redefine the corruption gate on the PERCEPTIBLE subset (drop/downweight the 8 SUBTLE
families, or weight each recipe by butter-max perceptibility). Then "catch it" means the
36 perceptible families rank below honest q20; the 8 subtle families staying near-lossless
is CORRECT, not a miss. Confirming the SUBTLE set by eye / against a human-JND anchor is the
next step before locking the gate.
