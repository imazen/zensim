# Known Bugs

## `fused.rs:154` — subtraction overflow in `vblur_add_idx()`
- **File**: `zensim/src/fused.rs:154`
- **Symptom**: `attempt to subtract with overflow` panic
- **Trigger**: Various image sizes including 53x70, 70x70, 100x99, 450x450, 800x450, 1409x1922
- **Cause**: `2 * (height - 1) - add_raw` underflows when `add_raw >= 2 * (height - 1)`
- **Impact**: Prevents processing many common image sizes. Found via imageflow checksum analysis (61 panics out of 205 attempted comparisons).
- **Fix**: Clamp the mirror-reflect index calculation to handle edge cases where the blur radius is large relative to the downscaled image dimension.
