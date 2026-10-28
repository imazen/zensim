//! O(1)-per-pixel box blur using running sums.
//!
//! Unlike recursive Gaussian IIR (used in ssimulacra2, ~60-70% of runtime),
//! box blur is O(1) per pixel regardless of radius.
#![allow(
    clippy::assign_op_pattern,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
use archmage::magetypes;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
use magetypes::simd::generic::f32x8 as GenericF32x8;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x16;

/// Capacity of the horizontal-blur "rem ring" (see the reuse note at each
/// gather site). Sized for `radius <= 16`; a larger radius simply falls back
/// to the original two-gather form, so this is a perf bound, never a
/// correctness one.
const H_RING_CAP: usize = 33;

/// Right-edge mirror for the horizontal sliding window: the source column the
/// add-side gather reads when the window's leading edge `add_raw = x + r + 1`
/// has walked past the last column.
///
/// **The `saturating_sub` is load-bearing.** The one-step reflection
/// `2*(width - 1) - add_raw` is only non-negative for `width >= radius + 2`,
/// and the era-2 column tile (see [`H_TILE_WIDTH`]) makes narrower slabs
/// reachable from a perfectly ordinary image: the last tile stages
/// `width % tile` interior columns plus a `radius`-wide left halo, so at
/// `width % tile == 1` the slab is exactly `radius + 1` columns and the
/// reflection underflows. Before 2026-09-04 this was a bare `usize`
/// subtraction — a debug build panicked with `attempt to subtract with
/// overflow` (12 tests, incl. `Zensim::compute` through
/// `tests/fold_engine_parity.rs`'s `(2049, 40)` cell) and a release build
/// wrapped to `usize::MAX - k`, saved from an out-of-bounds gather only by
/// the `.min(width - 1)` here.
///
/// The value in that slot is **not observable**: `add_raw > 2*(width - 1)`
/// requires `x > 2*width - radius - 3`, which for `width == radius + 1` is
/// satisfied only by `x == width - 1`, and every kernel writes its outputs
/// BEFORE the running-sum update, so the last update is dead.
/// `blur::tests::h_entries_are_bit_exact_at_a_degenerate_last_column_tile`
/// MEASURES that (its reference feeds a *different* index into the same slot
/// and every plane still matches to the bit), which is why this returns 0 —
/// matching the identical `saturating_sub` the vertical kernels have carried
/// since they hit the same shape at small pyramid scales — rather than
/// paying for a full multi-period reflection in the hot loop.
#[inline(always)]
fn h_mirror_add_idx(add_raw: usize, width: usize) -> usize {
    if add_raw < width {
        add_raw
    } else {
        (2 * (width - 1)).saturating_sub(add_raw)
    }
}

/// 1-pass blur: rectangular kernel.
/// Use with larger radius to approximate same effective width.
pub fn box_blur_1pass_into(
    input: &[f32],
    output: &mut [f32],
    temp: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    box_blur_h(input, temp, width, height, radius);
    box_blur_v_from_copy(temp, output, width, height, radius);
}

/// One-pass blur using the request's arithmetic on both axes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn box_blur_1pass_at_revision(
    input: &[f32],
    output: &mut [f32],
    temp: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    revision: crate::feature_defs::FormulaRevision,
) {
    box_blur_h_at_revision(input, temp, width, height, radius, revision);
    box_blur_v_from_copy_at_revision(temp, output, width, height, radius, revision);
}

/// Vertical box blur: read from `src`, write to `dst`.
pub fn box_blur_v_from_copy(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    box_blur_v_from_copy_at_revision(
        src,
        dst,
        width,
        height,
        radius,
        crate::ssim_form::active_revision(),
    )
}

pub(crate) fn box_blur_v_from_copy_at_revision(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    revision: crate::feature_defs::FormulaRevision,
) {
    if matches!(
        crate::featcanon::canon_blur_axis_for(crate::featcanon::mode(revision), revision),
        crate::featcanon::BlurMode::Local
    ) {
        box_blur_v_local(src, dst, width, height, radius);
        return;
    }

    incant!(
        box_blur_v_copy_inner(src, dst, width, height, radius),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// AVX-512 vertical blur: process 16 columns at a time.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn box_blur_v_copy_inner_v4(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    // One slice construction per plane: every access below is
    // `idx * width + col_base` with `idx <= height - 1` (all index helpers
    // clamp with `.min(height - 1)`) and `col_base <= width - LANES` —
    // statically `< height * width`, so the per-load/store range checks
    // fold away. Same values, same order.
    let n_px = height * width;
    let src = if src.len() >= n_px { &src[..n_px] } else { src };
    let dst = if dst.len() >= n_px {
        &mut dst[..n_px]
    } else {
        dst
    };

    for cg in 0..col_groups {
        let col_base = cg * 16;

        let mut sum = f32x16::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x16::from_array_t(token, src[base..][..16].try_into().unwrap());
        }

        for y in 0..height {
            let base = y * width + col_base;
            dst[base..base + 16].copy_from_slice(&(sum * inv_v).to_array());

            let add_raw = y + r + 1;
            // Boundary mirror: at the bottom edge, reflect the index
            // back into the image (`2*(height-1) - add_raw`). When the
            // mirror would itself go negative (i.e. `add_raw` exceeds
            // `2*(height-1)` — only possible with `height < r + 2`,
            // e.g. `r = 5` on `height = 6`), the row is clamped to 0
            // via the `saturating_sub`. The final `.min(height - 1)`
            // still caps above. Without the `saturating_sub`, the
            // expression panics on `attempt to subtract with overflow`
            // when used at very small pyramid scales.
            let add_idx = if add_raw < height {
                add_raw
            } else {
                (2 * (height - 1)).saturating_sub(add_raw)
            };
            let add_idx = add_idx.min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);

            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let add_v = f32x16::from_array_t(token, src[add_base..][..16].try_into().unwrap());
            let rem_v = f32x16::from_array_t(token, src[rem_base..][..16].try_into().unwrap());
            sum = sum + add_v - rem_v;
        }
    }

    // Remainder with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum = f32x8::zero_t(v3);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x8::from_array_t(v3, src[base..][..8].try_into().unwrap());
        }
        for y in 0..height {
            let base = y * width + col_base;
            dst[base..base + 8].copy_from_slice(&(sum * inv_v8).to_array());
            let add_raw = y + r + 1;
            // Boundary mirror: at the bottom edge, reflect the index
            // back into the image (`2*(height-1) - add_raw`). When the
            // mirror would itself go negative (i.e. `add_raw` exceeds
            // `2*(height-1)` — only possible with `height < r + 2`,
            // e.g. `r = 5` on `height = 6`), the row is clamped to 0
            // via the `saturating_sub`. The final `.min(height - 1)`
            // still caps above. Without the `saturating_sub`, the
            // expression panics on `attempt to subtract with overflow`
            // when used at very small pyramid scales.
            let add_idx = if add_raw < height {
                add_raw
            } else {
                (2 * (height - 1)).saturating_sub(add_raw)
            };
            let add_idx = add_idx.min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum = sum + f32x8::from_array_t(v3, src[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, src[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            sum += src[idx * width + x];
        }
        for y in 0..height {
            dst[y * width + x] = sum * inv;
            let add_raw = y + r + 1;
            // Boundary mirror: at the bottom edge, reflect the index
            // back into the image (`2*(height-1) - add_raw`). When the
            // mirror would itself go negative (i.e. `add_raw` exceeds
            // `2*(height-1)` — only possible with `height < r + 2`,
            // e.g. `r = 5` on `height = 6`), the row is clamped to 0
            // via the `saturating_sub`. The final `.min(height - 1)`
            // still caps above. Without the `saturating_sub`, the
            // expression panics on `attempt to subtract with overflow`
            // when used at very small pyramid scales.
            let add_idx = if add_raw < height {
                add_raw
            } else {
                (2 * (height - 1)).saturating_sub(add_raw)
            };
            let add_idx = add_idx.min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);
            sum = sum + src[add_idx * width + x] - src[rem_idx * width + x];
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn box_blur_v_copy_inner_v4x(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    // One slice construction per plane: every access below is
    // `idx * width + col_base` with `idx <= height - 1` (all index helpers
    // clamp with `.min(height - 1)`) and `col_base <= width - LANES` —
    // statically `< height * width`, so the per-load/store range checks
    // fold away. Same values, same order.
    let n_px = height * width;
    let src = if src.len() >= n_px { &src[..n_px] } else { src };
    let dst = if dst.len() >= n_px {
        &mut dst[..n_px]
    } else {
        dst
    };

    for cg in 0..col_groups {
        let col_base = cg * 16;

        let mut sum = f32x16::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x16::from_array_t(token, src[base..][..16].try_into().unwrap());
        }

        for y in 0..height {
            let base = y * width + col_base;
            dst[base..base + 16].copy_from_slice(&(sum * inv_v).to_array());

            let add_raw = y + r + 1;
            // Boundary mirror: at the bottom edge, reflect the index
            // back into the image (`2*(height-1) - add_raw`). When the
            // mirror would itself go negative (i.e. `add_raw` exceeds
            // `2*(height-1)` — only possible with `height < r + 2`,
            // e.g. `r = 5` on `height = 6`), the row is clamped to 0
            // via the `saturating_sub`. The final `.min(height - 1)`
            // still caps above. Without the `saturating_sub`, the
            // expression panics on `attempt to subtract with overflow`
            // when used at very small pyramid scales.
            let add_idx = if add_raw < height {
                add_raw
            } else {
                (2 * (height - 1)).saturating_sub(add_raw)
            };
            let add_idx = add_idx.min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);

            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let add_v = f32x16::from_array_t(token, src[add_base..][..16].try_into().unwrap());
            let rem_v = f32x16::from_array_t(token, src[rem_base..][..16].try_into().unwrap());
            sum = sum + add_v - rem_v;
        }
    }

    // Remainder with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum = f32x8::zero_t(v3);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x8::from_array_t(v3, src[base..][..8].try_into().unwrap());
        }
        for y in 0..height {
            let base = y * width + col_base;
            dst[base..base + 8].copy_from_slice(&(sum * inv_v8).to_array());
            let add_raw = y + r + 1;
            // Boundary mirror: at the bottom edge, reflect the index
            // back into the image (`2*(height-1) - add_raw`). When the
            // mirror would itself go negative (i.e. `add_raw` exceeds
            // `2*(height-1)` — only possible with `height < r + 2`,
            // e.g. `r = 5` on `height = 6`), the row is clamped to 0
            // via the `saturating_sub`. The final `.min(height - 1)`
            // still caps above. Without the `saturating_sub`, the
            // expression panics on `attempt to subtract with overflow`
            // when used at very small pyramid scales.
            let add_idx = if add_raw < height {
                add_raw
            } else {
                (2 * (height - 1)).saturating_sub(add_raw)
            };
            let add_idx = add_idx.min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum = sum + f32x8::from_array_t(v3, src[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, src[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            sum += src[idx * width + x];
        }
        for y in 0..height {
            dst[y * width + x] = sum * inv;
            let add_raw = y + r + 1;
            // Boundary mirror: at the bottom edge, reflect the index
            // back into the image (`2*(height-1) - add_raw`). When the
            // mirror would itself go negative (i.e. `add_raw` exceeds
            // `2*(height-1)` — only possible with `height < r + 2`,
            // e.g. `r = 5` on `height = 6`), the row is clamped to 0
            // via the `saturating_sub`. The final `.min(height - 1)`
            // still caps above. Without the `saturating_sub`, the
            // expression panics on `attempt to subtract with overflow`
            // when used at very small pyramid scales.
            let add_idx = if add_raw < height {
                add_raw
            } else {
                (2 * (height - 1)).saturating_sub(add_raw)
            };
            let add_idx = add_idx.min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);
            sum = sum + src[add_idx * width + x] - src[rem_idx * width + x];
        }
    }
}

/// SIMD vertical blur: read from src (const), write to dst.
/// Processes 8 columns at a time for cache efficiency.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn box_blur_v_copy_inner_v3(
    token: archmage::X64V3Token,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    // One slice construction per plane: every access below is
    // `idx * width + col_base` with `idx <= height - 1` (all index helpers
    // clamp with `.min(height - 1)`) and `col_base <= width - LANES` —
    // statically `< height * width`, so the per-load/store range checks
    // fold away. Same values, same order.
    let n_px = height * width;
    let src = if src.len() >= n_px { &src[..n_px] } else { src };
    let dst = if dst.len() >= n_px {
        &mut dst[..n_px]
    } else {
        dst
    };

    for cg in 0..col_groups {
        let col_base = cg * 8;

        // Initialize running sums for 8 columns
        let mut sum = f32x8::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x8::from_array_t(token, src[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            let base = y * width + col_base;
            dst[base..base + 8].copy_from_slice(&(sum * inv_v).to_array());

            let add_raw = y + r + 1;
            // Boundary mirror: at the bottom edge, reflect the index
            // back into the image (`2*(height-1) - add_raw`). When the
            // mirror would itself go negative (i.e. `add_raw` exceeds
            // `2*(height-1)` — only possible with `height < r + 2`,
            // e.g. `r = 5` on `height = 6`), the row is clamped to 0
            // via the `saturating_sub`. The final `.min(height - 1)`
            // still caps above. Without the `saturating_sub`, the
            // expression panics on `attempt to subtract with overflow`
            // when used at very small pyramid scales.
            let add_idx = if add_raw < height {
                add_raw
            } else {
                (2 * (height - 1)).saturating_sub(add_raw)
            };
            let add_idx = add_idx.min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);

            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let add_v = f32x8::from_array_t(token, src[add_base..][..8].try_into().unwrap());
            let rem_v = f32x8::from_array_t(token, src[rem_base..][..8].try_into().unwrap());
            sum = sum + add_v - rem_v;
        }
    }

    // Scalar remainder columns
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            sum += src[idx * width + x];
        }

        for y in 0..height {
            dst[y * width + x] = sum * inv;
            let add_raw = y + r + 1;
            // Boundary mirror: at the bottom edge, reflect the index
            // back into the image (`2*(height-1) - add_raw`). When the
            // mirror would itself go negative (i.e. `add_raw` exceeds
            // `2*(height-1)` — only possible with `height < r + 2`,
            // e.g. `r = 5` on `height = 6`), the row is clamped to 0
            // via the `saturating_sub`. The final `.min(height - 1)`
            // still caps above. Without the `saturating_sub`, the
            // expression panics on `attempt to subtract with overflow`
            // when used at very small pyramid scales.
            let add_idx = if add_raw < height {
                add_raw
            } else {
                (2 * (height - 1)).saturating_sub(add_raw)
            };
            let add_idx = add_idx.min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);
            sum = sum + src[add_idx * width + x] - src[rem_idx * width + x];
        }
    }
}

#[magetypes(neon, wasm128, scalar)]
fn box_blur_v_copy_inner(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    for cg in 0..col_groups {
        let col_base = cg * 8;

        // Initialize running sums for 8 columns
        let mut sum = f32x8::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x8::from_array_t(token, src[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            let base = y * width + col_base;
            dst[base..base + 8].copy_from_slice(&(sum * inv_v).to_array());

            let add_raw = y + r + 1;
            // Boundary mirror: at the bottom edge, reflect the index
            // back into the image (`2*(height-1) - add_raw`). When the
            // mirror would itself go negative (i.e. `add_raw` exceeds
            // `2*(height-1)` — only possible with `height < r + 2`,
            // e.g. `r = 5` on `height = 6`), the row is clamped to 0
            // via the `saturating_sub`. The final `.min(height - 1)`
            // still caps above. Without the `saturating_sub`, the
            // expression panics on `attempt to subtract with overflow`
            // when used at very small pyramid scales.
            let add_idx = if add_raw < height {
                add_raw
            } else {
                (2 * (height - 1)).saturating_sub(add_raw)
            };
            let add_idx = add_idx.min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);

            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let add_v = f32x8::from_array_t(token, src[add_base..][..8].try_into().unwrap());
            let rem_v = f32x8::from_array_t(token, src[rem_base..][..8].try_into().unwrap());
            sum = sum + add_v - rem_v;
        }
    }

    // Scalar remainder columns
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            sum += src[idx * width + x];
        }

        for y in 0..height {
            dst[y * width + x] = sum * inv;
            let add_raw = y + r + 1;
            // Boundary mirror: at the bottom edge, reflect the index
            // back into the image (`2*(height-1) - add_raw`). When the
            // mirror would itself go negative (i.e. `add_raw` exceeds
            // `2*(height-1)` — only possible with `height < r + 2`,
            // e.g. `r = 5` on `height = 6`), the row is clamped to 0
            // via the `saturating_sub`. The final `.min(height - 1)`
            // still caps above. Without the `saturating_sub`, the
            // expression panics on `attempt to subtract with overflow`
            // when used at very small pyramid scales.
            let add_idx = if add_raw < height {
                add_raw
            } else {
                (2 * (height - 1)).saturating_sub(add_raw)
            };
            let add_idx = add_idx.min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);
            sum = sum + src[add_idx * width + x] - src[rem_idx * width + x];
        }
    }
}

/// Horizontal box blur using running sum. O(1) per pixel.
pub(crate) fn box_blur_h(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    box_blur_h_at_revision(
        input,
        output,
        width,
        height,
        radius,
        crate::ssim_form::active_revision(),
    )
}

pub(crate) fn box_blur_h_at_revision(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    revision: crate::feature_defs::FormulaRevision,
) {
    if matches!(
        crate::featcanon::canon_blur_axis_for(crate::featcanon::mode(revision), revision),
        crate::featcanon::BlurMode::Local
    ) {
        box_blur_h_local(input, output, width, height, radius);
        return;
    }

    let tile = h_blur_tile_width();
    if tile > 0 && width > tile {
        h_tile_1in1out(input, output, width, height, radius, tile, |i, o, w, h| {
            box_blur_h_untiled(i, o, w, h, radius)
        });
        return;
    }
    box_blur_h_untiled(input, output, width, height, radius)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn box_blur_h_untiled(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    incant!(
        box_blur_h_inner(input, output, width, height, radius),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// AVX-512 horizontal blur: process 16 rows simultaneously.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn box_blur_h_inner_v4(
    token: archmage::X64V4Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    // One slice construction per plane: every gather/store below is
    // `(row_base + ro) * width + idx` with `ro < LANES`,
    // `row_base + LANES <= height`, and every column index clamped by
    // `.min(width - 1)` — statically `< height * width`, so the per-element
    // range checks fold away. Same values, same order.
    let n_px = height * width;
    let input = if input.len() >= n_px {
        &input[..n_px]
    } else {
        input
    };
    let output = if output.len() >= n_px {
        &mut output[..n_px]
    } else {
        output
    };

    let mut ring = [[0.0f32; 16]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 16;
        let mut ring_pos = 0usize;

        let mut sum = f32x16::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 16];
            for ro in 0..16 {
                arr[ro] = input[(row_base + ro) * width + idx];
            }
            sum = sum + f32x16::from_array_t(token, arr);
        }

        for x in 0..width {
            let result = (sum * inv_v).to_array();
            for ro in 0..16 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 16];
            for ro in 0..16 {
                add_arr[ro] = input[(row_base + ro) * width + add_idx];
            }
            // rem-ring (2026-08-30): for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped — so the remove-side gather re-reads the exact
            // bytes the add-side gather loaded `diam` steps ago. Keep the last
            // `diam` add-vectors in a stack ring and reuse them: same memory,
            // same values, same `sum + add - rem` order, so the output is
            // BIT-IDENTICAL (gated by `tests::box_blur_h_ring_matches_
            // regathered_reference` plus the v1 golden + fold-pool parity).
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 16];
                for ro in 0..16 {
                    a[ro] = input[(row_base + ro) * width + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x16::from_array_t(token, add_arr) - f32x16::from_array_t(token, rem_arr);
        }
    }

    // Remainder rows: use v3 (f32x8) path
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    let mut ring = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut ring_pos = 0usize;
        let mut sum = f32x8::zero_t(v3);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 8];
            for ro in 0..8 {
                arr[ro] = input[(row_base + ro) * width + idx];
            }
            sum = sum + f32x8::from_array_t(v3, arr);
        }
        for x in 0..width {
            let result = (sum * inv_v8).to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            for ro in 0..8 {
                add_arr[ro] = input[(row_base + ro) * width + add_idx];
            }
            // rem-ring (2026-08-30): for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped — so the remove-side gather re-reads the exact
            // bytes the add-side gather loaded `diam` steps ago. Keep the last
            // `diam` add-vectors in a stack ring and reuse them: same memory,
            // same values, same `sum + add - rem` order, so the output is
            // BIT-IDENTICAL (gated by `tests::box_blur_h_ring_matches_
            // regathered_reference` plus the v1 golden + fold-pool parity).
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 8];
                for ro in 0..8 {
                    a[ro] = input[(row_base + ro) * width + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x8::from_array_t(v3, add_arr) - f32x8::from_array_t(v3, rem_arr);
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for row in (remaining_start + remaining_8groups * 8)..height {
        let row_off = row * width;
        let inp = &input[row_off..row_off + width];
        let out = &mut output[row_off..row_off + width];
        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum += inp[idx];
        }
        for x in 0..width {
            out[x] = sum * inv;
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum = sum + inp[add_idx] - inp[rem_idx];
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn box_blur_h_inner_v4x(
    token: archmage::X64V4xToken,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    box_blur_h_v4x_body(token, input, output, width, height, radius);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn box_blur_h_v4x_body(
    token: archmage::X64V4xToken,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    if width == 0 || !width.is_multiple_of(256) || height < 16 {
        box_blur_h_v4x_strided(token, input, output, width, height, radius, width);
        return;
    }
    thread_local! {
        static PADDED: core::cell::RefCell<Vec<f32>> = const { core::cell::RefCell::new(Vec::new()) };
    }
    let rows = height / 16 * 16;
    PADDED.with(|a| {
        let mut arena = a.borrow_mut();
        let stride = width + 16;
        let per = 16 * stride;
        arena.resize(2 * per, 0.0);
        let (src, dst) = arena.split_at_mut(per);
        for first in (0..rows).step_by(16) {
            for row in 0..16 {
                let from = (first + row) * width;
                let to = row * stride;
                src[to..to + width].copy_from_slice(&input[from..from + width]);
            }
            box_blur_h_v4x_strided(token, src, dst, width, 16, radius, stride);
            for row in 0..16 {
                let from = row * stride;
                let to = (first + row) * width;
                output[to..to + width].copy_from_slice(&dst[from..from + width]);
            }
        }
    });
    if rows < height {
        let off = rows * width;
        box_blur_h_v4x_strided(
            token,
            &input[off..],
            &mut output[off..],
            width,
            height - rows,
            radius,
            width,
        );
    }
}

// Physical row pitch changes storage only; preserve the original recurrence.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn box_blur_h_v4x_strided(
    token: archmage::X64V4xToken,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    stride: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    // One slice construction per plane: every gather/store below is
    // `(row_base + ro) * stride + idx` with `ro < LANES`,
    // `row_base + LANES <= height`, and every column index clamped by
    // `.min(width - 1)` — statically `< (height - 1) * stride + width`
    // (stride >= width at every call site), so the per-element range
    // checks fold away. Same values, same order.
    let n_px = height.saturating_sub(1) * stride + width;
    let input = if input.len() >= n_px {
        &input[..n_px]
    } else {
        input
    };
    let output = if output.len() >= n_px {
        &mut output[..n_px]
    } else {
        output
    };

    let mut ring = [[0.0f32; 16]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 16;
        let mut ring_pos = 0usize;

        let mut sum = f32x16::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 16];
            for ro in 0..16 {
                arr[ro] = input[(row_base + ro) * stride + idx];
            }
            sum = sum + f32x16::from_array_t(token, arr);
        }

        for x in 0..width {
            let result = (sum * inv_v).to_array();
            for ro in 0..16 {
                output[(row_base + ro) * stride + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 16];
            for ro in 0..16 {
                add_arr[ro] = input[(row_base + ro) * stride + add_idx];
            }
            // rem-ring (2026-08-30): for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped — so the remove-side gather re-reads the exact
            // bytes the add-side gather loaded `diam` steps ago. Keep the last
            // `diam` add-vectors in a stack ring and reuse them: same memory,
            // same values, same `sum + add - rem` order, so the output is
            // BIT-IDENTICAL (gated by `tests::box_blur_h_ring_matches_
            // regathered_reference` plus the v1 golden + fold-pool parity).
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 16];
                for ro in 0..16 {
                    a[ro] = input[(row_base + ro) * stride + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x16::from_array_t(token, add_arr) - f32x16::from_array_t(token, rem_arr);
        }
    }

    // Remainder rows: use v3 (f32x8) path
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    let mut ring = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut ring_pos = 0usize;
        let mut sum = f32x8::zero_t(v3);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 8];
            for ro in 0..8 {
                arr[ro] = input[(row_base + ro) * stride + idx];
            }
            sum = sum + f32x8::from_array_t(v3, arr);
        }
        for x in 0..width {
            let result = (sum * inv_v8).to_array();
            for ro in 0..8 {
                output[(row_base + ro) * stride + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            for ro in 0..8 {
                add_arr[ro] = input[(row_base + ro) * stride + add_idx];
            }
            // rem-ring (2026-08-30): for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped — so the remove-side gather re-reads the exact
            // bytes the add-side gather loaded `diam` steps ago. Keep the last
            // `diam` add-vectors in a stack ring and reuse them: same memory,
            // same values, same `sum + add - rem` order, so the output is
            // BIT-IDENTICAL (gated by `tests::box_blur_h_ring_matches_
            // regathered_reference` plus the v1 golden + fold-pool parity).
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 8];
                for ro in 0..8 {
                    a[ro] = input[(row_base + ro) * stride + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x8::from_array_t(v3, add_arr) - f32x8::from_array_t(v3, rem_arr);
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for row in (remaining_start + remaining_8groups * 8)..height {
        let row_off = row * stride;
        let inp = &input[row_off..row_off + width];
        let out = &mut output[row_off..row_off + width];
        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum += inp[idx];
        }
        for x in 0..width {
            out[x] = sum * inv;
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum = sum + inp[add_idx] - inp[rem_idx];
        }
    }
}

/// SIMD horizontal blur: process 8 rows simultaneously.
/// Each f32x8 lane holds the running sum for one row.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn box_blur_h_inner_v3(
    token: archmage::X64V3Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    // One slice construction per plane: every gather/store below is
    // `(row_base + ro) * width + idx` with `ro < LANES`,
    // `row_base + LANES <= height`, and every column index clamped by
    // `.min(width - 1)` — statically `< height * width`, so the per-element
    // range checks fold away. Same values, same order.
    let n_px = height * width;
    let input = if input.len() >= n_px {
        &input[..n_px]
    } else {
        input
    };
    let output = if output.len() >= n_px {
        &mut output[..n_px]
    } else {
        output
    };

    let mut ring = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut ring_pos = 0usize;

        // Initialize running sums for 8 rows
        let mut sum = f32x8::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 8];
            for ro in 0..8 {
                arr[ro] = input[(row_base + ro) * width + idx];
            }
            sum = sum + f32x8::from_array_t(token, arr);
        }

        for x in 0..width {
            let result = (sum * inv_v).to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 8];
            for ro in 0..8 {
                add_arr[ro] = input[(row_base + ro) * width + add_idx];
            }
            // rem-ring (2026-08-30): for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped — so the remove-side gather re-reads the exact
            // bytes the add-side gather loaded `diam` steps ago. Keep the last
            // `diam` add-vectors in a stack ring and reuse them: same memory,
            // same values, same `sum + add - rem` order, so the output is
            // BIT-IDENTICAL (gated by `tests::box_blur_h_ring_matches_
            // regathered_reference` plus the v1 golden + fold-pool parity).
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 8];
                for ro in 0..8 {
                    a[ro] = input[(row_base + ro) * width + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x8::from_array_t(token, add_arr) - f32x8::from_array_t(token, rem_arr);
        }
    }

    // Scalar remainder rows
    let inv = 1.0 / diam as f32;
    for row in (row_groups * 8)..height {
        let row_off = row * width;
        let inp = &input[row_off..row_off + width];
        let out = &mut output[row_off..row_off + width];

        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum += inp[idx];
        }

        for x in 0..width {
            out[x] = sum * inv;
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum = sum + inp[add_idx] - inp[rem_idx];
        }
    }
}

#[magetypes(neon, wasm128, scalar)]
fn box_blur_h_inner(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    let mut ring = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut ring_pos = 0usize;

        // Initialize running sums for 8 rows
        let mut sum = f32x8::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 8];
            for ro in 0..8 {
                arr[ro] = input[(row_base + ro) * width + idx];
            }
            sum = sum + f32x8::from_array_t(token, arr);
        }

        for x in 0..width {
            let result = (sum * inv_v).to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 8];
            for ro in 0..8 {
                add_arr[ro] = input[(row_base + ro) * width + add_idx];
            }
            // rem-ring (2026-08-30): for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped — so the remove-side gather re-reads the exact
            // bytes the add-side gather loaded `diam` steps ago. Keep the last
            // `diam` add-vectors in a stack ring and reuse them: same memory,
            // same values, same `sum + add - rem` order, so the output is
            // BIT-IDENTICAL (gated by `tests::box_blur_h_ring_matches_
            // regathered_reference` plus the v1 golden + fold-pool parity).
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 8];
                for ro in 0..8 {
                    a[ro] = input[(row_base + ro) * width + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x8::from_array_t(token, add_arr) - f32x8::from_array_t(token, rem_arr);
        }
    }

    // Scalar remainder rows
    let inv = 1.0 / diam as f32;
    for row in (row_groups * 8)..height {
        let row_off = row * width;
        let inp = &input[row_off..row_off + width];
        let out = &mut output[row_off..row_off + width];

        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum += inp[idx];
        }

        for x in 0..width {
            out[x] = sum * inv;
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum = sum + inp[add_idx] - inp[rem_idx];
        }
    }
}

/// Phase 2 Lever 3 (2026-05-22): fused horizontal blur with abs-diff
/// activity output. Computes `|src[i] - box_blur_h(src)[i]|` in one pass —
/// the intermediate H-blur plane is never materialized.
///
/// Replaces `box_blur_h(src, h_blur_src) + abs_diff_into(src, h_blur_src,
/// activity)` (two separate full-plane passes) with a single SIMD kernel.
/// Saves one full plane read of src + one full plane write/read of
/// h_blur_src per channel per scale on the activity path.
pub(crate) fn box_blur_h_into_abs_diff(
    src: &[f32],
    out_activity: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let revision = crate::ssim_form::active_revision();
    if matches!(
        crate::featcanon::canon_blur_axis_for(crate::featcanon::mode(revision), revision),
        crate::featcanon::BlurMode::Local
    ) {
        box_blur_h_local(src, out_activity, width, height, radius);
        for (v, &s) in out_activity.iter_mut().zip(src) {
            *v = (s - *v).abs();
        }
        return;
    }

    let tile = h_blur_tile_width();
    if tile > 0 && width > tile {
        h_tile_1in1out(
            src,
            out_activity,
            width,
            height,
            radius,
            tile,
            |i, o, w, h| box_blur_h_into_abs_diff_untiled(i, o, w, h, radius),
        );
        return;
    }
    box_blur_h_into_abs_diff_untiled(src, out_activity, width, height, radius)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn box_blur_h_into_abs_diff_untiled(
    src: &[f32],
    out_activity: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    incant!(
        box_blur_h_into_abs_diff_inner(src, out_activity, width, height, radius),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn box_blur_h_into_abs_diff_inner_v4(
    token: archmage::X64V4Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    let mut ring = [[0.0f32; 16]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 16;
        let mut ring_pos = 0usize;

        let mut sum = f32x16::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 16];
            for ro in 0..16 {
                arr[ro] = input[(row_base + ro) * width + idx];
            }
            sum = sum + f32x16::from_array_t(token, arr);
        }

        for x in 0..width {
            // Gather src[col=x] for each of the 16 rows. These reads are
            // shared with the running-sum updates below, so they're hot
            // in L1 either way.
            let mut src_arr = [0.0f32; 16];
            for ro in 0..16 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x16::from_array_t(token, src_arr);
            let blur_v = sum * inv_v;
            let act_v = (src_v - blur_v).abs();
            let result = act_v.to_array();
            for ro in 0..16 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 16];
            for ro in 0..16 {
                add_arr[ro] = input[(row_base + ro) * width + add_idx];
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 16];
                for ro in 0..16 {
                    a[ro] = input[(row_base + ro) * width + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x16::from_array_t(token, add_arr) - f32x16::from_array_t(token, rem_arr);
        }
    }

    // Remainder rows with v3 (f32x8)
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    let mut ring = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut ring_pos = 0usize;
        let mut sum = f32x8::zero_t(v3);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 8];
            for ro in 0..8 {
                arr[ro] = input[(row_base + ro) * width + idx];
            }
            sum = sum + f32x8::from_array_t(v3, arr);
        }
        for x in 0..width {
            let mut src_arr = [0.0f32; 8];
            for ro in 0..8 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x8::from_array_t(v3, src_arr);
            let act_v = (src_v - sum * inv_v8).abs();
            let result = act_v.to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            for ro in 0..8 {
                add_arr[ro] = input[(row_base + ro) * width + add_idx];
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 8];
                for ro in 0..8 {
                    a[ro] = input[(row_base + ro) * width + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x8::from_array_t(v3, add_arr) - f32x8::from_array_t(v3, rem_arr);
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for row in (remaining_start + remaining_8groups * 8)..height {
        let row_off = row * width;
        let inp = &input[row_off..row_off + width];
        let out = &mut output[row_off..row_off + width];
        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum += inp[idx];
        }
        for x in 0..width {
            out[x] = (inp[x] - sum * inv).abs();
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum = sum + inp[add_idx] - inp[rem_idx];
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn box_blur_h_into_abs_diff_inner_v4x(
    token: archmage::X64V4xToken,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    let mut ring = [[0.0f32; 16]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 16;
        let mut ring_pos = 0usize;

        let mut sum = f32x16::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 16];
            for ro in 0..16 {
                arr[ro] = input[(row_base + ro) * width + idx];
            }
            sum = sum + f32x16::from_array_t(token, arr);
        }

        for x in 0..width {
            // Gather src[col=x] for each of the 16 rows. These reads are
            // shared with the running-sum updates below, so they're hot
            // in L1 either way.
            let mut src_arr = [0.0f32; 16];
            for ro in 0..16 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x16::from_array_t(token, src_arr);
            let blur_v = sum * inv_v;
            let act_v = (src_v - blur_v).abs();
            let result = act_v.to_array();
            for ro in 0..16 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 16];
            for ro in 0..16 {
                add_arr[ro] = input[(row_base + ro) * width + add_idx];
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 16];
                for ro in 0..16 {
                    a[ro] = input[(row_base + ro) * width + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x16::from_array_t(token, add_arr) - f32x16::from_array_t(token, rem_arr);
        }
    }

    // Remainder rows with v3 (f32x8)
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    let mut ring = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut ring_pos = 0usize;
        let mut sum = f32x8::zero_t(v3);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 8];
            for ro in 0..8 {
                arr[ro] = input[(row_base + ro) * width + idx];
            }
            sum = sum + f32x8::from_array_t(v3, arr);
        }
        for x in 0..width {
            let mut src_arr = [0.0f32; 8];
            for ro in 0..8 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x8::from_array_t(v3, src_arr);
            let act_v = (src_v - sum * inv_v8).abs();
            let result = act_v.to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            for ro in 0..8 {
                add_arr[ro] = input[(row_base + ro) * width + add_idx];
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 8];
                for ro in 0..8 {
                    a[ro] = input[(row_base + ro) * width + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x8::from_array_t(v3, add_arr) - f32x8::from_array_t(v3, rem_arr);
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for row in (remaining_start + remaining_8groups * 8)..height {
        let row_off = row * width;
        let inp = &input[row_off..row_off + width];
        let out = &mut output[row_off..row_off + width];
        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum += inp[idx];
        }
        for x in 0..width {
            out[x] = (inp[x] - sum * inv).abs();
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum = sum + inp[add_idx] - inp[rem_idx];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn box_blur_h_into_abs_diff_inner_v3(
    token: archmage::X64V3Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    let mut ring = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut ring_pos = 0usize;

        let mut sum = f32x8::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 8];
            for ro in 0..8 {
                arr[ro] = input[(row_base + ro) * width + idx];
            }
            sum = sum + f32x8::from_array_t(token, arr);
        }
        for x in 0..width {
            let mut src_arr = [0.0f32; 8];
            for ro in 0..8 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x8::from_array_t(token, src_arr);
            let act_v = (src_v - sum * inv_v).abs();
            let result = act_v.to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            for ro in 0..8 {
                add_arr[ro] = input[(row_base + ro) * width + add_idx];
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 8];
                for ro in 0..8 {
                    a[ro] = input[(row_base + ro) * width + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x8::from_array_t(token, add_arr) - f32x8::from_array_t(token, rem_arr);
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for row in (row_groups * 8)..height {
        let row_off = row * width;
        let inp = &input[row_off..row_off + width];
        let out = &mut output[row_off..row_off + width];
        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum += inp[idx];
        }
        for x in 0..width {
            out[x] = (inp[x] - sum * inv).abs();
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum = sum + inp[add_idx] - inp[rem_idx];
        }
    }
}

#[magetypes(neon, wasm128, scalar)]
fn box_blur_h_into_abs_diff_inner(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    let mut ring = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut ring_pos = 0usize;
        let mut sum = f32x8::zero_t(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut arr = [0.0f32; 8];
            for ro in 0..8 {
                arr[ro] = input[(row_base + ro) * width + idx];
            }
            sum = sum + f32x8::from_array_t(token, arr);
        }
        for x in 0..width {
            let mut src_arr = [0.0f32; 8];
            for ro in 0..8 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x8::from_array_t(token, src_arr);
            let act_v = (src_v - sum * inv_v).abs();
            let result = act_v.to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            for ro in 0..8 {
                add_arr[ro] = input[(row_base + ro) * width + add_idx];
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let rem_arr = if use_ring && x >= diam {
                ring[ring_pos]
            } else {
                let mut a = [0.0f32; 8];
                for ro in 0..8 {
                    a[ro] = input[(row_base + ro) * width + rem_idx];
                }
                a
            };
            if use_ring {
                ring[ring_pos] = add_arr;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum = sum + f32x8::from_array_t(token, add_arr) - f32x8::from_array_t(token, rem_arr);
        }
    }

    let inv = 1.0 / diam as f32;
    for row in (row_groups * 8)..height {
        let row_off = row * width;
        let inp = &input[row_off..row_off + width];
        let out = &mut output[row_off..row_off + width];
        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum += inp[idx];
        }
        for x in 0..width {
            out[x] = (inp[x] - sum * inv).abs();
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum = sum + inp[add_idx] - inp[rem_idx];
        }
    }
}

/// Fused horizontal blur for means: computes blur(src) and blur(dst) in a single pass.
/// Reads each pixel of src/dst exactly once, replacing two separate box_blur_h calls.
/// Used for edge-only channels that need mu1/mu2 but not sigma planes.
pub(crate) fn fused_blur_h_mu(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let tile = h_blur_tile_width();
    if tile > 0 && width > tile {
        h_tile_2in2out(
            src,
            dst,
            out_mu1,
            out_mu2,
            width,
            height,
            radius,
            tile,
            |s2, d2, a, b, w, h| fused_blur_h_mu_untiled(s2, d2, a, b, w, h, radius),
        );
        return;
    }
    fused_blur_h_mu_untiled(src, dst, out_mu1, out_mu2, width, height, radius)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_blur_h_mu_untiled(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    incant!(
        fused_blur_h_mu_inner(src, dst, out_mu1, out_mu2, width, height, radius),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_mu_inner_v4(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    let mut ring_s = [[0.0f32; 16]; H_RING_CAP];
    let mut ring_d = [[0.0f32; 16]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 16;
        let mut ring_pos = 0usize;
        let mut sum_s = f32x16::zero_t(token);
        let mut sum_d = f32x16::zero_t(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 16];
            let mut d_arr = [0.0f32; 16];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 16 interior ones: `ro < 16` and
                // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + idx;
                let cs = &src[off..off + 15 * width + 1];
                let cd = &dst[off..off + 15 * width + 1];
                for ro in 0..16 {
                    s_arr[ro] = cs[ro * width];
                    d_arr[ro] = cd[ro * width];
                }
            }
            sum_s = sum_s + f32x16::from_array_t(token, s_arr);
            sum_d = sum_d + f32x16::from_array_t(token, d_arr);
        }

        for x in 0..width {
            let mu1_result = (sum_s * inv_v).to_array();
            let mu2_result = (sum_d * inv_v).to_array();
            for ro in 0..16 {
                let base = (row_base + ro) * width + x;
                out_mu1[base] = mu1_result[ro];
                out_mu2[base] = mu2_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 16];
            let mut d_add = [0.0f32; 16];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 16 interior ones: `ro < 16` and
                // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + add_idx;
                let cs = &src[off..off + 15 * width + 1];
                let cd = &dst[off..off + 15 * width + 1];
                for ro in 0..16 {
                    s_add[ro] = cs[ro * width];
                    d_add[ro] = cd[ro * width];
                }
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 16];
                let mut da = [0.0f32; 16];
                {
                    // The column's extent is hoisted to ONE range check per gather and
                    // LLVM folds the 16 interior ones: `ro < 16` and
                    // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                    // slice ends exactly one past the highest index the loop below
                    // reads, so it is in bounds precisely when the old form was.
                    // Same loads, same values, same order — BIT-EXACT; this moves
                    // WHERE the bound is proven, not what is read.
                    let off = row_base * width + rem_idx;
                    let cs = &src[off..off + 15 * width + 1];
                    let cd = &dst[off..off + 15 * width + 1];
                    for ro in 0..16 {
                        sa[ro] = cs[ro * width];
                        da[ro] = cd[ro * width];
                    }
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum_s = sum_s + f32x16::from_array_t(token, s_add) - f32x16::from_array_t(token, s_rem);
            sum_d = sum_d + f32x16::from_array_t(token, d_add) - f32x16::from_array_t(token, d_rem);
        }
    }

    // Remainder with f32x8
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    let mut ring_s = [[0.0f32; 8]; H_RING_CAP];
    let mut ring_d = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut ring_pos = 0usize;
        let mut sum_s = f32x8::zero_t(v3);
        let mut sum_d = f32x8::zero_t(v3);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + idx;
                let cs = &src[off..off + 7 * width + 1];
                let cd = &dst[off..off + 7 * width + 1];
                for ro in 0..8 {
                    s_arr[ro] = cs[ro * width];
                    d_arr[ro] = cd[ro * width];
                }
            }
            sum_s = sum_s + f32x8::from_array_t(v3, s_arr);
            sum_d = sum_d + f32x8::from_array_t(v3, d_arr);
        }

        for x in 0..width {
            let mu1_result = (sum_s * inv_v8).to_array();
            let mu2_result = (sum_d * inv_v8).to_array();
            for ro in 0..8 {
                let base = (row_base + ro) * width + x;
                out_mu1[base] = mu1_result[ro];
                out_mu2[base] = mu2_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + add_idx;
                let cs = &src[off..off + 7 * width + 1];
                let cd = &dst[off..off + 7 * width + 1];
                for ro in 0..8 {
                    s_add[ro] = cs[ro * width];
                    d_add[ro] = cd[ro * width];
                }
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 8];
                let mut da = [0.0f32; 8];
                {
                    // The column's extent is hoisted to ONE range check per gather and
                    // LLVM folds the 8 interior ones: `ro < 8` and
                    // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                    // slice ends exactly one past the highest index the loop below
                    // reads, so it is in bounds precisely when the old form was.
                    // Same loads, same values, same order — BIT-EXACT; this moves
                    // WHERE the bound is proven, not what is read.
                    let off = row_base * width + rem_idx;
                    let cs = &src[off..off + 7 * width + 1];
                    let cd = &dst[off..off + 7 * width + 1];
                    for ro in 0..8 {
                        sa[ro] = cs[ro * width];
                        da[ro] = cd[ro * width];
                    }
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum_s = sum_s + f32x8::from_array_t(v3, s_add) - f32x8::from_array_t(v3, s_rem);
            sum_d = sum_d + f32x8::from_array_t(v3, d_add) - f32x8::from_array_t(v3, d_rem);
        }
    }

    // Scalar remainder rows
    let inv = 1.0 / diam as f32;
    for row in (remaining_start + remaining_8groups * 8)..height {
        let row_off = row * width;
        let s_row = &src[row_off..row_off + width];
        let d_row = &dst[row_off..row_off + width];
        let mut sum_s = 0.0f32;
        let mut sum_d = 0.0f32;

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum_s += s_row[idx];
            sum_d += d_row[idx];
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum_s = sum_s + s_row[add_idx] - s_row[rem_idx];
            sum_d = sum_d + d_row[add_idx] - d_row[rem_idx];
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_mu_inner_v4x(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    let mut ring_s = [[0.0f32; 16]; H_RING_CAP];
    let mut ring_d = [[0.0f32; 16]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 16;
        let mut ring_pos = 0usize;
        let mut sum_s = f32x16::zero_t(token);
        let mut sum_d = f32x16::zero_t(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 16];
            let mut d_arr = [0.0f32; 16];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 16 interior ones: `ro < 16` and
                // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + idx;
                let cs = &src[off..off + 15 * width + 1];
                let cd = &dst[off..off + 15 * width + 1];
                for ro in 0..16 {
                    s_arr[ro] = cs[ro * width];
                    d_arr[ro] = cd[ro * width];
                }
            }
            sum_s = sum_s + f32x16::from_array_t(token, s_arr);
            sum_d = sum_d + f32x16::from_array_t(token, d_arr);
        }

        for x in 0..width {
            let mu1_result = (sum_s * inv_v).to_array();
            let mu2_result = (sum_d * inv_v).to_array();
            for ro in 0..16 {
                let base = (row_base + ro) * width + x;
                out_mu1[base] = mu1_result[ro];
                out_mu2[base] = mu2_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 16];
            let mut d_add = [0.0f32; 16];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 16 interior ones: `ro < 16` and
                // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + add_idx;
                let cs = &src[off..off + 15 * width + 1];
                let cd = &dst[off..off + 15 * width + 1];
                for ro in 0..16 {
                    s_add[ro] = cs[ro * width];
                    d_add[ro] = cd[ro * width];
                }
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 16];
                let mut da = [0.0f32; 16];
                {
                    // The column's extent is hoisted to ONE range check per gather and
                    // LLVM folds the 16 interior ones: `ro < 16` and
                    // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                    // slice ends exactly one past the highest index the loop below
                    // reads, so it is in bounds precisely when the old form was.
                    // Same loads, same values, same order — BIT-EXACT; this moves
                    // WHERE the bound is proven, not what is read.
                    let off = row_base * width + rem_idx;
                    let cs = &src[off..off + 15 * width + 1];
                    let cd = &dst[off..off + 15 * width + 1];
                    for ro in 0..16 {
                        sa[ro] = cs[ro * width];
                        da[ro] = cd[ro * width];
                    }
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum_s = sum_s + f32x16::from_array_t(token, s_add) - f32x16::from_array_t(token, s_rem);
            sum_d = sum_d + f32x16::from_array_t(token, d_add) - f32x16::from_array_t(token, d_rem);
        }
    }

    // Remainder with f32x8
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    let mut ring_s = [[0.0f32; 8]; H_RING_CAP];
    let mut ring_d = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut ring_pos = 0usize;
        let mut sum_s = f32x8::zero_t(v3);
        let mut sum_d = f32x8::zero_t(v3);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + idx;
                let cs = &src[off..off + 7 * width + 1];
                let cd = &dst[off..off + 7 * width + 1];
                for ro in 0..8 {
                    s_arr[ro] = cs[ro * width];
                    d_arr[ro] = cd[ro * width];
                }
            }
            sum_s = sum_s + f32x8::from_array_t(v3, s_arr);
            sum_d = sum_d + f32x8::from_array_t(v3, d_arr);
        }

        for x in 0..width {
            let mu1_result = (sum_s * inv_v8).to_array();
            let mu2_result = (sum_d * inv_v8).to_array();
            for ro in 0..8 {
                let base = (row_base + ro) * width + x;
                out_mu1[base] = mu1_result[ro];
                out_mu2[base] = mu2_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + add_idx;
                let cs = &src[off..off + 7 * width + 1];
                let cd = &dst[off..off + 7 * width + 1];
                for ro in 0..8 {
                    s_add[ro] = cs[ro * width];
                    d_add[ro] = cd[ro * width];
                }
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 8];
                let mut da = [0.0f32; 8];
                {
                    // The column's extent is hoisted to ONE range check per gather and
                    // LLVM folds the 8 interior ones: `ro < 8` and
                    // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                    // slice ends exactly one past the highest index the loop below
                    // reads, so it is in bounds precisely when the old form was.
                    // Same loads, same values, same order — BIT-EXACT; this moves
                    // WHERE the bound is proven, not what is read.
                    let off = row_base * width + rem_idx;
                    let cs = &src[off..off + 7 * width + 1];
                    let cd = &dst[off..off + 7 * width + 1];
                    for ro in 0..8 {
                        sa[ro] = cs[ro * width];
                        da[ro] = cd[ro * width];
                    }
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum_s = sum_s + f32x8::from_array_t(v3, s_add) - f32x8::from_array_t(v3, s_rem);
            sum_d = sum_d + f32x8::from_array_t(v3, d_add) - f32x8::from_array_t(v3, d_rem);
        }
    }

    // Scalar remainder rows
    let inv = 1.0 / diam as f32;
    for row in (remaining_start + remaining_8groups * 8)..height {
        let row_off = row * width;
        let s_row = &src[row_off..row_off + width];
        let d_row = &dst[row_off..row_off + width];
        let mut sum_s = 0.0f32;
        let mut sum_d = 0.0f32;

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum_s += s_row[idx];
            sum_d += d_row[idx];
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum_s = sum_s + s_row[add_idx] - s_row[rem_idx];
            sum_d = sum_d + d_row[add_idx] - d_row[rem_idx];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_mu_inner_v3(
    token: archmage::X64V3Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    let mut ring_s = [[0.0f32; 8]; H_RING_CAP];
    let mut ring_d = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut ring_pos = 0usize;
        let mut sum_s = f32x8::zero_t(token);
        let mut sum_d = f32x8::zero_t(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + idx;
                let cs = &src[off..off + 7 * width + 1];
                let cd = &dst[off..off + 7 * width + 1];
                for ro in 0..8 {
                    s_arr[ro] = cs[ro * width];
                    d_arr[ro] = cd[ro * width];
                }
            }
            sum_s = sum_s + f32x8::from_array_t(token, s_arr);
            sum_d = sum_d + f32x8::from_array_t(token, d_arr);
        }

        for x in 0..width {
            let mu1_result = (sum_s * inv_v).to_array();
            let mu2_result = (sum_d * inv_v).to_array();
            for ro in 0..8 {
                let base = (row_base + ro) * width + x;
                out_mu1[base] = mu1_result[ro];
                out_mu2[base] = mu2_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + add_idx;
                let cs = &src[off..off + 7 * width + 1];
                let cd = &dst[off..off + 7 * width + 1];
                for ro in 0..8 {
                    s_add[ro] = cs[ro * width];
                    d_add[ro] = cd[ro * width];
                }
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 8];
                let mut da = [0.0f32; 8];
                {
                    // The column's extent is hoisted to ONE range check per gather and
                    // LLVM folds the 8 interior ones: `ro < 8` and
                    // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                    // slice ends exactly one past the highest index the loop below
                    // reads, so it is in bounds precisely when the old form was.
                    // Same loads, same values, same order — BIT-EXACT; this moves
                    // WHERE the bound is proven, not what is read.
                    let off = row_base * width + rem_idx;
                    let cs = &src[off..off + 7 * width + 1];
                    let cd = &dst[off..off + 7 * width + 1];
                    for ro in 0..8 {
                        sa[ro] = cs[ro * width];
                        da[ro] = cd[ro * width];
                    }
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum_s = sum_s + f32x8::from_array_t(token, s_add) - f32x8::from_array_t(token, s_rem);
            sum_d = sum_d + f32x8::from_array_t(token, d_add) - f32x8::from_array_t(token, d_rem);
        }
    }

    // Scalar remainder rows
    let inv = 1.0 / diam as f32;
    for row in (row_groups * 8)..height {
        let row_off = row * width;
        let s_row = &src[row_off..row_off + width];
        let d_row = &dst[row_off..row_off + width];
        let mut sum_s = 0.0f32;
        let mut sum_d = 0.0f32;

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum_s += s_row[idx];
            sum_d += d_row[idx];
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum_s = sum_s + s_row[add_idx] - s_row[rem_idx];
            sum_d = sum_d + d_row[add_idx] - d_row[rem_idx];
        }
    }
}

#[magetypes(neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_mu_inner(
    token: Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;

    // Masked 8-row groups — same construction as `fused_blur_h_ssim_inner`:
    // lanes past `valid` alias the last real row, only real rows are
    // stored, so tail rows carry the vector body's exact numerics (the
    // former scalar tail summed `sum += add - rem`, a different association
    // from the vector `sum + add - rem`).
    let mut run_group = |row_base: usize, valid: usize| {
        let mut ring_s = [[0.0f32; 8]; H_RING_CAP];
        let mut ring_d = [[0.0f32; 8]; H_RING_CAP];
        let use_ring = diam <= H_RING_CAP;
        let mut ring_pos = 0usize;
        let mut row_off = [0usize; 8];
        for (ro, off) in row_off.iter_mut().enumerate() {
            *off = (row_base + ro.min(valid - 1)) * width;
        }
        let mut sum_s = f32x8::zero_t(token);
        let mut sum_d = f32x8::zero_t(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for (ro, &off) in row_off.iter().enumerate() {
                s_arr[ro] = src[off + idx];
                d_arr[ro] = dst[off + idx];
            }
            sum_s = sum_s + f32x8::from_array_t(token, s_arr);
            sum_d = sum_d + f32x8::from_array_t(token, d_arr);
        }

        for x in 0..width {
            let mu1_result = (sum_s * inv_v).to_array();
            let mu2_result = (sum_d * inv_v).to_array();
            for (ro, &off) in row_off.iter().enumerate().take(valid) {
                let base = off + x;
                out_mu1[base] = mu1_result[ro];
                out_mu2[base] = mu2_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            for (ro, &off) in row_off.iter().enumerate() {
                s_add[ro] = src[off + add_idx];
                d_add[ro] = dst[off + add_idx];
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 8];
                let mut da = [0.0f32; 8];
                for (ro, &off) in row_off.iter().enumerate() {
                    sa[ro] = src[off + rem_idx];
                    da[ro] = dst[off + rem_idx];
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            sum_s = sum_s + f32x8::from_array_t(token, s_add) - f32x8::from_array_t(token, s_rem);
            sum_d = sum_d + f32x8::from_array_t(token, d_add) - f32x8::from_array_t(token, d_rem);
        }
    };

    let row_groups = height / 8;
    for rg in 0..row_groups {
        run_group(rg * 8, 8);
    }
    let tail = height - row_groups * 8;
    if tail > 0 {
        run_group(row_groups * 8, tail);
    }
}

/// Column-tile a 1-in / 1-out H pass (`box_blur_h`,
/// `box_blur_h_into_abs_diff`): stage the tile plus its `±radius` halo into a
/// compact thread-local buffer, run `f`, copy the interior back.
///
/// These entries are tiled for CONSISTENCY, not for speed — the A/B on the
/// activity chain was a wash (§23.5). Tiling only `fused_blur_h_ssim` left the
/// v1 reference path and the fold reading planes produced under different
/// summation groupings, which failed `folded720_v1_pools_match_v1_path` and
/// `v1_372_bit_exact_to_fold_at_every_width`. Either every H entry tiles or
/// none does.
fn h_tile_1in1out(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    rows: usize,
    radius: usize,
    tile: usize,
    mut f: impl FnMut(&[f32], &mut [f32], usize, usize),
) {
    thread_local! {
        static ARENA: core::cell::RefCell<Vec<f32>> = const { core::cell::RefCell::new(Vec::new()) };
    }
    ARENA.with(|a| {
        let mut arena = a.borrow_mut();
        let tw_max = (tile + 2 * radius).min(width);
        let per = rows * tw_max;
        if arena.len() < 2 * per {
            arena.resize(2 * per, 0.0);
        }
        let (tin, tout) = arena.split_at_mut(per);
        let mut x0 = 0usize;
        while x0 < width {
            let x1 = (x0 + tile).min(width);
            let xl = x0.saturating_sub(radius);
            let xr = (x1 + radius).min(width);
            let (tw, keep, kn) = (xr - xl, x0 - xl, x1 - x0);
            for y in 0..rows {
                let s = y * width + xl;
                tin[y * tw..y * tw + tw].copy_from_slice(&input[s..s + tw]);
            }
            f(&tin[..rows * tw], &mut tout[..rows * tw], tw, rows);
            for y in 0..rows {
                let o = y * width + x0;
                let t = y * tw + keep;
                output[o..o + kn].copy_from_slice(&tout[t..t + kn]);
            }
            x0 = x1;
        }
    });
}

/// Column-tile the 2-in / 2-out `fused_blur_h_mu`, same contract as
/// [`h_tile_1in1out`].
#[allow(clippy::too_many_arguments)]
fn h_tile_2in2out(
    src: &[f32],
    dst: &[f32],
    o1: &mut [f32],
    o2: &mut [f32],
    width: usize,
    rows: usize,
    radius: usize,
    tile: usize,
    mut f: impl FnMut(&[f32], &[f32], &mut [f32], &mut [f32], usize, usize),
) {
    thread_local! {
        static ARENA: core::cell::RefCell<Vec<f32>> = const { core::cell::RefCell::new(Vec::new()) };
    }
    ARENA.with(|a| {
        let mut arena = a.borrow_mut();
        let tw_max = (tile + 2 * radius).min(width);
        let per = rows * tw_max;
        if arena.len() < 4 * per {
            arena.resize(4 * per, 0.0);
        }
        let (ts, rest) = arena.split_at_mut(per);
        let (td, rest) = rest.split_at_mut(per);
        let (t1, t2) = rest.split_at_mut(per);
        let mut x0 = 0usize;
        while x0 < width {
            let x1 = (x0 + tile).min(width);
            let xl = x0.saturating_sub(radius);
            let xr = (x1 + radius).min(width);
            let (tw, keep, kn) = (xr - xl, x0 - xl, x1 - x0);
            for y in 0..rows {
                let s = y * width + xl;
                ts[y * tw..y * tw + tw].copy_from_slice(&src[s..s + tw]);
                td[y * tw..y * tw + tw].copy_from_slice(&dst[s..s + tw]);
            }
            f(
                &ts[..rows * tw],
                &td[..rows * tw],
                &mut t1[..rows * tw],
                &mut t2[..rows * tw],
                tw,
                rows,
            );
            for y in 0..rows {
                let o = y * width + x0;
                let t = y * tw + keep;
                o1[o..o + kn].copy_from_slice(&t1[t..t + kn]);
                o2[o..o + kn].copy_from_slice(&t2[t..t + kn]);
            }
            x0 = x1;
        }
    });
}

/// The H-blur column-tile width: [`H_TILE_WIDTH`]. It is semantics, not a tuning knob.
#[inline]
pub(crate) fn h_blur_tile_width() -> usize {
    H_TILE_WIDTH
}

/// Default column-tile width for the H blur. **ERA-2 DEFAULT (2026-08-31).**
///
/// **DERIVED, not picked.** The quantity that matters is the H blur's live
/// window: it holds `H_BLUR_ROW_GROUP` rows × **6** planes (src, dst, four
/// outputs) × `tile` columns × 4 B, and that wants to sit inside L2. At the
/// 1 MiB L2 on the development part that is
/// `1_048_576 / (16 × 6 × 4)` = **1365 columns**; **1024** is the power of two
/// below it, which also leaves headroom on parts with a 512 KiB L2
/// (`16 × 6 × 1024 × 4` = 393 KiB, still inside) — and it sits inside the
/// **measured-flat** band: 512 / 1024 / 1536 were 271.25 / 271.64 / 270.57 ms
/// at 2304²/1T, i.e. within noise of each other
/// (`benchmarks/era2_perf_break_2026-08-31.md` §23.3).
///
/// **This is SEMANTICS, not a tuning knob.** The running sum along x restarts
/// at every tile boundary, so changing this value changes the emitted feature
/// bytes — the tile boundaries are part of the era-2 definition, exactly as
/// `ERA2_BAND_ROWS` is. Re-tuning it for speed on a different part is a NEW
/// era, not a configuration change. `ZENSIM_H_TILE` overrides it for
/// measurement only (`0` disables tiling and reproduces era-1's H planes).
///
/// A width at or below the tile is untiled by construction (`width > tile`
/// guards every H entry), so images narrower than this are byte-for-byte what
/// era-1 produced.
pub(crate) const H_TILE_WIDTH: usize = 1024;

/// [`fused_blur_h_ssim`] over column tiles of `tile` output columns, each
/// staged with a `±radius` halo into a compact thread-local buffer, blurred
/// there, and its interior copied back.
///
/// The PACKING is the optimisation, not the loop restriction — an in-place
/// column range measured 0.96–1.06× against this form's 1.26–1.71×
/// (`benchmarks/era2_perf_break_2026-08-31.md` §23.5).
///
/// Not bit-exact: the running sum along x restarts at each tile. Tile
/// boundaries are a pure function of `(width, tile)`, so the result stays
/// thread- and schedule-invariant.
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_column_tiled(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    rows: usize,
    radius: usize,
    tile: usize,
    err: bool,
) {
    thread_local! {
        static ARENA: core::cell::RefCell<Vec<f32>> = const { core::cell::RefCell::new(Vec::new()) };
    }
    ARENA.with(|a| {
        let mut arena = a.borrow_mut();
        let tw_max = (tile + 2 * radius).min(width);
        let per = rows * tw_max;
        if arena.len() < 6 * per {
            arena.resize(6 * per, 0.0);
        }
        let (tsrc, rest) = arena.split_at_mut(per);
        let (tdst, rest) = rest.split_at_mut(per);
        let (tm1, rest) = rest.split_at_mut(per);
        let (tm2, rest) = rest.split_at_mut(per);
        let (tsq, rest) = rest.split_at_mut(per);
        let (ts12, _) = rest.split_at_mut(per);

        let mut x0 = 0usize;
        while x0 < width {
            let x1 = (x0 + tile).min(width);
            let xl = x0.saturating_sub(radius);
            let xr = (x1 + radius).min(width);
            let tw = xr - xl;
            let keep = x0 - xl;
            let kn = x1 - x0;
            for y in 0..rows {
                let s = y * width + xl;
                let t = y * tw;
                tsrc[t..t + tw].copy_from_slice(&src[s..s + tw]);
                tdst[t..t + tw].copy_from_slice(&dst[s..s + tw]);
            }
            fused_blur_h_ssim_untiled(
                &tsrc[..rows * tw],
                &tdst[..rows * tw],
                &mut tm1[..rows * tw],
                &mut tm2[..rows * tw],
                &mut tsq[..rows * tw],
                &mut ts12[..rows * tw],
                tw,
                rows,
                radius,
                err,
            );
            for y in 0..rows {
                let o = y * width + x0;
                let t = y * tw + keep;
                out_mu1[o..o + kn].copy_from_slice(&tm1[t..t + kn]);
                out_mu2[o..o + kn].copy_from_slice(&tm2[t..t + kn]);
                out_sigma_sq[o..o + kn].copy_from_slice(&tsq[t..t + kn]);
                out_sigma12[o..o + kn].copy_from_slice(&ts12[t..t + kn]);
            }
            x0 = x1;
        }
    });
}

/// Same SIMD kernel with explicit request-local arithmetic; no global mutation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_blur_h_ssim_at_revision(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    revision: crate::feature_defs::FormulaRevision,
) {
    let revision = crate::ssim_form::effective_revision(revision);
    let err = revision >= crate::feature_defs::FormulaRevision::Rev5;
    // Canonical arithmetic follows THIS computation's revision (featcanon D1),
    // never the process switch.
    let mode = crate::featcanon::mode(revision);
    let canon = mode.active();
    if canon {
        fused_blur_h_ssim_canon(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            width,
            height,
            radius,
            err,
            mode,
            revision,
        );
        return;
    }
    let tile = h_blur_tile_width();
    if tile > 0 && width > tile {
        fused_blur_h_ssim_column_tiled(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            width,
            height,
            radius,
            tile,
            err,
        );
        return;
    }
    fused_blur_h_ssim_untiled(
        src,
        dst,
        out_mu1,
        out_mu2,
        out_sigma_sq,
        out_sigma12,
        width,
        height,
        radius,
        err,
    )
}

/// [`fused_blur_h_ssim`] with the column tile bypassed — the tiled form's own
/// per-tile call, and the path every caller took before the tile existed.
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_untiled(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    incant!(
        fused_blur_h_ssim_inner(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            width,
            height,
            radius,
            err
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_inner_v4(
    token: archmage::X64V4Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    // One slice construction per plane: every access below is
    // `(row_base + ro) * width + col` or `row_base * width + idx` with
    // `ro < LANES`, `row_base + LANES <= height`, and every column index
    // clamped by `.min(width - 1)` — so each index is statically `<
    // height * width` and the per-gather/per-store range checks fold away.
    // Same values, same order; this only moves where the bound is proven.
    let n_px = height * width;
    let src = if src.len() >= n_px { &src[..n_px] } else { src };
    let dst = if dst.len() >= n_px { &dst[..n_px] } else { dst };
    let out_mu1 = if out_mu1.len() >= n_px {
        &mut out_mu1[..n_px]
    } else {
        out_mu1
    };
    let out_mu2 = if out_mu2.len() >= n_px {
        &mut out_mu2[..n_px]
    } else {
        out_mu2
    };
    let out_sigma_sq = if out_sigma_sq.len() >= n_px {
        &mut out_sigma_sq[..n_px]
    } else {
        out_sigma_sq
    };
    let out_sigma12 = if out_sigma12.len() >= n_px {
        &mut out_sigma12[..n_px]
    } else {
        out_sigma12
    };

    let mut ring_s = [[0.0f32; 16]; H_RING_CAP];
    let mut ring_d = [[0.0f32; 16]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 16;
        let mut ring_pos = 0usize;

        let mut sum_s = f32x16::zero_t(token);
        let mut sum_d = f32x16::zero_t(token);
        let mut sum_sq = f32x16::zero_t(token);
        let mut sum_prod = f32x16::zero_t(token);

        // Initialize running sums
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 16];
            let mut d_arr = [0.0f32; 16];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 16 interior ones: `ro < 16` and
                // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + idx;
                let cs = &src[off..off + 15 * width + 1];
                let cd = &dst[off..off + 15 * width + 1];
                for ro in 0..16 {
                    s_arr[ro] = cs[ro * width];
                    d_arr[ro] = cd[ro * width];
                }
            }
            let sv = f32x16::from_array_t(token, s_arr);
            let dv = f32x16::from_array_t(token, d_arr);
            sum_s = sum_s + sv;
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = if err {
                let e = sv - dv;
                e.mul_add(e, sum_prod)
            } else {
                sv.mul_add(dv, sum_prod)
            };
        }

        // Slide window
        for x in 0..width {
            let mu1_result = (sum_s * inv_v).to_array();
            let mu2_result = (sum_d * inv_v).to_array();
            let sq_result = (sum_sq * inv_v).to_array();
            let prod_result = (sum_prod * inv_v).to_array();
            for ro in 0..16 {
                let base = (row_base + ro) * width + x;
                out_mu1[base] = mu1_result[ro];
                out_mu2[base] = mu2_result[ro];
                out_sigma_sq[base] = sq_result[ro];
                out_sigma12[base] = prod_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 16];
            let mut d_add = [0.0f32; 16];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 16 interior ones: `ro < 16` and
                // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + add_idx;
                let cs = &src[off..off + 15 * width + 1];
                let cd = &dst[off..off + 15 * width + 1];
                for ro in 0..16 {
                    s_add[ro] = cs[ro * width];
                    d_add[ro] = cd[ro * width];
                }
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 16];
                let mut da = [0.0f32; 16];
                {
                    // The column's extent is hoisted to ONE range check per gather and
                    // LLVM folds the 16 interior ones: `ro < 16` and
                    // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                    // slice ends exactly one past the highest index the loop below
                    // reads, so it is in bounds precisely when the old form was.
                    // Same loads, same values, same order — BIT-EXACT; this moves
                    // WHERE the bound is proven, not what is read.
                    let off = row_base * width + rem_idx;
                    let cs = &src[off..off + 15 * width + 1];
                    let cd = &dst[off..off + 15 * width + 1];
                    for ro in 0..16 {
                        sa[ro] = cs[ro * width];
                        da[ro] = cd[ro * width];
                    }
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            let sa = f32x16::from_array_t(token, s_add);
            let da = f32x16::from_array_t(token, d_add);
            let sr = f32x16::from_array_t(token, s_rem);
            let dr = f32x16::from_array_t(token, d_rem);
            sum_s = sum_s + sa - sr;
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = if err {
                let ea = sa - da;
                let er = sr - dr;
                ea.mul_add(ea, (-er).mul_add(er, sum_prod))
            } else {
                sa.mul_add(da, (-sr).mul_add(dr, sum_prod))
            };
        }
    }

    // Remainder with f32x8
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    let mut ring_s = [[0.0f32; 8]; H_RING_CAP];
    let mut ring_d = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut ring_pos = 0usize;
        let mut sum_s = f32x8::zero_t(v3);
        let mut sum_d = f32x8::zero_t(v3);
        let mut sum_sq = f32x8::zero_t(v3);
        let mut sum_prod = f32x8::zero_t(v3);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + idx;
                let cs = &src[off..off + 7 * width + 1];
                let cd = &dst[off..off + 7 * width + 1];
                for ro in 0..8 {
                    s_arr[ro] = cs[ro * width];
                    d_arr[ro] = cd[ro * width];
                }
            }
            let sv = f32x8::from_array_t(v3, s_arr);
            let dv = f32x8::from_array_t(v3, d_arr);
            sum_s = sum_s + sv;
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = if err {
                let e = sv - dv;
                e.mul_add(e, sum_prod)
            } else {
                sv.mul_add(dv, sum_prod)
            };
        }

        for x in 0..width {
            let mu1_result = (sum_s * inv_v8).to_array();
            let mu2_result = (sum_d * inv_v8).to_array();
            let sq_result = (sum_sq * inv_v8).to_array();
            let prod_result = (sum_prod * inv_v8).to_array();
            for ro in 0..8 {
                let base = (row_base + ro) * width + x;
                out_mu1[base] = mu1_result[ro];
                out_mu2[base] = mu2_result[ro];
                out_sigma_sq[base] = sq_result[ro];
                out_sigma12[base] = prod_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + add_idx;
                let cs = &src[off..off + 7 * width + 1];
                let cd = &dst[off..off + 7 * width + 1];
                for ro in 0..8 {
                    s_add[ro] = cs[ro * width];
                    d_add[ro] = cd[ro * width];
                }
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 8];
                let mut da = [0.0f32; 8];
                {
                    // The column's extent is hoisted to ONE range check per gather and
                    // LLVM folds the 8 interior ones: `ro < 8` and
                    // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                    // slice ends exactly one past the highest index the loop below
                    // reads, so it is in bounds precisely when the old form was.
                    // Same loads, same values, same order — BIT-EXACT; this moves
                    // WHERE the bound is proven, not what is read.
                    let off = row_base * width + rem_idx;
                    let cs = &src[off..off + 7 * width + 1];
                    let cd = &dst[off..off + 7 * width + 1];
                    for ro in 0..8 {
                        sa[ro] = cs[ro * width];
                        da[ro] = cd[ro * width];
                    }
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            let sa = f32x8::from_array_t(v3, s_add);
            let da = f32x8::from_array_t(v3, d_add);
            let sr = f32x8::from_array_t(v3, s_rem);
            let dr = f32x8::from_array_t(v3, d_rem);
            sum_s = sum_s + sa - sr;
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = if err {
                let ea = sa - da;
                let er = sr - dr;
                ea.mul_add(ea, (-er).mul_add(er, sum_prod))
            } else {
                sa.mul_add(da, (-sr).mul_add(dr, sum_prod))
            };
        }
    }

    // Scalar remainder rows
    let inv = 1.0 / diam as f32;
    for row in (remaining_start + remaining_8groups * 8)..height {
        let row_off = row * width;
        let s_row = &src[row_off..row_off + width];
        let d_row = &dst[row_off..row_off + width];
        let mut sum_s = 0.0f32;
        let mut sum_d = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_prod = 0.0f32;

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let s = s_row[idx];
            let d = d_row[idx];
            sum_s += s;
            sum_d += d;
            sum_sq = s.mul_add(s, d.mul_add(d, sum_sq));
            sum_prod = if err {
                let e = s - d;
                e.mul_add(e, sum_prod)
            } else {
                s.mul_add(d, sum_prod)
            };
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;
            out_sigma_sq[row_off + x] = sum_sq * inv;
            out_sigma12[row_off + x] = sum_prod * inv;

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let sa = s_row[add_idx];
            let da = d_row[add_idx];
            let sr = s_row[rem_idx];
            let dr = d_row[rem_idx];
            sum_s = sum_s + sa - sr;
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = if err {
                let ea = sa - da;
                let er = sr - dr;
                ea.mul_add(ea, (-er).mul_add(er, sum_prod))
            } else {
                sa.mul_add(da, (-sr).mul_add(dr, sum_prod))
            };
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_inner_v4x(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    fused_blur_h_ssim_v4x_body::<true>(
        token,
        src,
        dst,
        out_mu1,
        out_mu2,
        out_sigma_sq,
        out_sigma12,
        width,
        height,
        radius,
        err,
    );
}

/// 3-output variant (no `mu1`): the ref-side mean is read from a
/// [`V2PreparedReference`-style cache](crate::feature_v2), so the `sum_s`
/// accumulator chain and the `out_mu1` stores are pure waste on that
/// path. `out_mu1` is accepted but untouched (callers pass their scratch
/// slice; non-v4x tiers of the public dispatcher fall back to the
/// 4-output kernel writing into it).
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim3_inner_v4x(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    fused_blur_h_ssim_v4x_body::<false>(
        token,
        src,
        dst,
        out_mu1,
        out_mu2,
        out_sigma_sq,
        out_sigma12,
        width,
        height,
        radius,
        err,
    );
}

/// Shared v4x body — `MU1` const-guards the `sum_s` chain + `out_mu1`
/// stores; every other accumulator chain is textually identical in both
/// monomorphizations, so the 3-output variant's `mu2`/`ssq`/`s12` are
/// bit-identical to the 4-output kernel's (independent accumulators).
/// `#[inline(always)]`: must fuse into the `#[arcane]` wrappers'
/// target_feature region (see `dense_block_kernel_generic`'s doc for the
/// measured 5.3x cliff when this inlining is left to the cost model).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
// callers live in the ssim3 v4x chain (v2-walk); the arcane-registered siblings mask their own liveness
fn fused_blur_h_ssim_v4x_body<const MU1: bool>(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    // A cache-line pad spreads the 16 row streams across cache sets. Logical
    // width and horizontal running sums remain unchanged (unlike column
    // tiling). Stage one vector group, not a whole image, and reuse storage.
    if width == 0 || !width.is_multiple_of(256) || height < 16 {
        fused_blur_h_ssim_v4x_strided::<MU1>(
            token,
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            width,
            height,
            radius,
            width,
            err,
        );
        return;
    }
    thread_local! {
        static PADDED: core::cell::RefCell<Vec<f32>> = const { core::cell::RefCell::new(Vec::new()) };
    }
    let rows = height / 16 * 16;
    PADDED.with(|a| {
        let mut arena = a.borrow_mut();
        let stride = width + 16;
        let per = 16 * stride;
        arena.resize(6 * per, 0.0);
        let (s, rest) = arena.split_at_mut(per);
        let (d, rest) = rest.split_at_mut(per);
        let (m1, rest) = rest.split_at_mut(per);
        let (m2, rest) = rest.split_at_mut(per);
        let (sq, prod) = rest.split_at_mut(per);
        for first in (0..rows).step_by(16) {
            for row in 0..16 {
                let from = (first + row) * width;
                let to = row * stride;
                s[to..to + width].copy_from_slice(&src[from..from + width]);
                d[to..to + width].copy_from_slice(&dst[from..from + width]);
            }
            fused_blur_h_ssim_v4x_strided::<MU1>(
                token, s, d, m1, m2, sq, prod, width, 16, radius, stride, err,
            );
            for row in 0..16 {
                let from = row * stride;
                let to = (first + row) * width;
                if MU1 {
                    out_mu1[to..to + width].copy_from_slice(&m1[from..from + width]);
                }
                out_mu2[to..to + width].copy_from_slice(&m2[from..from + width]);
                out_sigma_sq[to..to + width].copy_from_slice(&sq[from..from + width]);
                out_sigma12[to..to + width].copy_from_slice(&prod[from..from + width]);
            }
        }
    });
    if rows < height {
        let off = rows * width;
        // The three-output path permits an empty, untouched mu1 slice.
        let tail_mu1 = if MU1 { &mut out_mu1[off..] } else { &mut [] };
        fused_blur_h_ssim_v4x_strided::<MU1>(
            token,
            &src[off..],
            &dst[off..],
            tail_mu1,
            &mut out_mu2[off..],
            &mut out_sigma_sq[off..],
            &mut out_sigma12[off..],
            width,
            height - rows,
            radius,
            width,
            err,
        );
    }
}

// Original arithmetic, with physical row pitch independent of logical width.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_v4x_strided<const MU1: bool>(
    token: archmage::X64V4xToken,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    stride: usize,
    err: bool,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    // One slice construction per plane: every access below is
    // `(row_base + ro) * stride + col` or `row_base * stride + idx` with
    // `ro < LANES`, `row_base + LANES <= height`, and every column index
    // clamped by `.min(width - 1)` — so each index is statically `<
    // (height - 1) * stride + width` (stride >= width at every call site)
    // and the per-gather/per-store range checks fold away.
    // Same values, same order; this only moves where the bound is proven.
    let n_px = height.saturating_sub(1) * stride + width;
    let src = if src.len() >= n_px { &src[..n_px] } else { src };
    let dst = if dst.len() >= n_px { &dst[..n_px] } else { dst };
    let out_mu1 = if out_mu1.len() >= n_px {
        &mut out_mu1[..n_px]
    } else {
        out_mu1
    };
    let out_mu2 = if out_mu2.len() >= n_px {
        &mut out_mu2[..n_px]
    } else {
        out_mu2
    };
    let out_sigma_sq = if out_sigma_sq.len() >= n_px {
        &mut out_sigma_sq[..n_px]
    } else {
        out_sigma_sq
    };
    let out_sigma12 = if out_sigma12.len() >= n_px {
        &mut out_sigma12[..n_px]
    } else {
        out_sigma12
    };

    let mut ring_s = [[0.0f32; 16]; H_RING_CAP];
    let mut ring_d = [[0.0f32; 16]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 16;
        let mut ring_pos = 0usize;

        let mut sum_s = f32x16::zero_t(token);
        let mut sum_d = f32x16::zero_t(token);
        let mut sum_sq = f32x16::zero_t(token);
        let mut sum_prod = f32x16::zero_t(token);

        // Initialize running sums
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 16];
            let mut d_arr = [0.0f32; 16];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 16 interior ones: `ro < 16` and
                // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * stride + idx;
                let cs = &src[off..off + 15 * stride + 1];
                let cd = &dst[off..off + 15 * stride + 1];
                for ro in 0..16 {
                    s_arr[ro] = cs[ro * stride];
                    d_arr[ro] = cd[ro * stride];
                }
            }
            let sv = f32x16::from_array_t(token, s_arr);
            let dv = f32x16::from_array_t(token, d_arr);
            if MU1 {
                sum_s = sum_s + sv;
            }
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = if err {
                let e = sv - dv;
                e.mul_add(e, sum_prod)
            } else {
                sv.mul_add(dv, sum_prod)
            };
        }

        // Slide window
        for x in 0..width {
            let mu2_result = (sum_d * inv_v).to_array();
            let sq_result = (sum_sq * inv_v).to_array();
            let prod_result = (sum_prod * inv_v).to_array();
            if MU1 {
                let mu1_result = (sum_s * inv_v).to_array();
                for ro in 0..16 {
                    let base = (row_base + ro) * stride + x;
                    out_mu1[base] = mu1_result[ro];
                }
            }
            for ro in 0..16 {
                let base = (row_base + ro) * stride + x;
                out_mu2[base] = mu2_result[ro];
                out_sigma_sq[base] = sq_result[ro];
                out_sigma12[base] = prod_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 16];
            let mut d_add = [0.0f32; 16];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 16 interior ones: `ro < 16` and
                // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * stride + add_idx;
                let cs = &src[off..off + 15 * stride + 1];
                let cd = &dst[off..off + 15 * stride + 1];
                for ro in 0..16 {
                    s_add[ro] = cs[ro * stride];
                    d_add[ro] = cd[ro * stride];
                }
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 16];
                let mut da = [0.0f32; 16];
                {
                    // The column's extent is hoisted to ONE range check per gather and
                    // LLVM folds the 16 interior ones: `ro < 16` and
                    // `col.len() == 15*width + 1` give `ro*width < col.len()`. The
                    // slice ends exactly one past the highest index the loop below
                    // reads, so it is in bounds precisely when the old form was.
                    // Same loads, same values, same order — BIT-EXACT; this moves
                    // WHERE the bound is proven, not what is read.
                    let off = row_base * stride + rem_idx;
                    let cs = &src[off..off + 15 * stride + 1];
                    let cd = &dst[off..off + 15 * stride + 1];
                    for ro in 0..16 {
                        sa[ro] = cs[ro * stride];
                        da[ro] = cd[ro * stride];
                    }
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            let sa = f32x16::from_array_t(token, s_add);
            let da = f32x16::from_array_t(token, d_add);
            let sr = f32x16::from_array_t(token, s_rem);
            let dr = f32x16::from_array_t(token, d_rem);
            if MU1 {
                sum_s = sum_s + sa - sr;
            }
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = if err {
                let ea = sa - da;
                let er = sr - dr;
                ea.mul_add(ea, (-er).mul_add(er, sum_prod))
            } else {
                sa.mul_add(da, (-sr).mul_add(dr, sum_prod))
            };
        }
    }

    // Remainder with f32x8
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    let mut ring_s = [[0.0f32; 8]; H_RING_CAP];
    let mut ring_d = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut ring_pos = 0usize;
        let mut sum_s = f32x8::zero_t(v3);
        let mut sum_d = f32x8::zero_t(v3);
        let mut sum_sq = f32x8::zero_t(v3);
        let mut sum_prod = f32x8::zero_t(v3);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * stride + idx;
                let cs = &src[off..off + 7 * stride + 1];
                let cd = &dst[off..off + 7 * stride + 1];
                for ro in 0..8 {
                    s_arr[ro] = cs[ro * stride];
                    d_arr[ro] = cd[ro * stride];
                }
            }
            let sv = f32x8::from_array_t(v3, s_arr);
            let dv = f32x8::from_array_t(v3, d_arr);
            if MU1 {
                sum_s = sum_s + sv;
            }
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = if err {
                let e = sv - dv;
                e.mul_add(e, sum_prod)
            } else {
                sv.mul_add(dv, sum_prod)
            };
        }

        for x in 0..width {
            let mu2_result = (sum_d * inv_v8).to_array();
            let sq_result = (sum_sq * inv_v8).to_array();
            let prod_result = (sum_prod * inv_v8).to_array();
            if MU1 {
                let mu1_result = (sum_s * inv_v8).to_array();
                for ro in 0..8 {
                    let base = (row_base + ro) * stride + x;
                    out_mu1[base] = mu1_result[ro];
                }
            }
            for ro in 0..8 {
                let base = (row_base + ro) * stride + x;
                out_mu2[base] = mu2_result[ro];
                out_sigma_sq[base] = sq_result[ro];
                out_sigma12[base] = prod_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * stride + add_idx;
                let cs = &src[off..off + 7 * stride + 1];
                let cd = &dst[off..off + 7 * stride + 1];
                for ro in 0..8 {
                    s_add[ro] = cs[ro * stride];
                    d_add[ro] = cd[ro * stride];
                }
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 8];
                let mut da = [0.0f32; 8];
                {
                    // The column's extent is hoisted to ONE range check per gather and
                    // LLVM folds the 8 interior ones: `ro < 8` and
                    // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                    // slice ends exactly one past the highest index the loop below
                    // reads, so it is in bounds precisely when the old form was.
                    // Same loads, same values, same order — BIT-EXACT; this moves
                    // WHERE the bound is proven, not what is read.
                    let off = row_base * stride + rem_idx;
                    let cs = &src[off..off + 7 * stride + 1];
                    let cd = &dst[off..off + 7 * stride + 1];
                    for ro in 0..8 {
                        sa[ro] = cs[ro * stride];
                        da[ro] = cd[ro * stride];
                    }
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            let sa = f32x8::from_array_t(v3, s_add);
            let da = f32x8::from_array_t(v3, d_add);
            let sr = f32x8::from_array_t(v3, s_rem);
            let dr = f32x8::from_array_t(v3, d_rem);
            if MU1 {
                sum_s = sum_s + sa - sr;
            }
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = if err {
                let ea = sa - da;
                let er = sr - dr;
                ea.mul_add(ea, (-er).mul_add(er, sum_prod))
            } else {
                sa.mul_add(da, (-sr).mul_add(dr, sum_prod))
            };
        }
    }

    // Scalar remainder rows
    let inv = 1.0 / diam as f32;
    for row in (remaining_start + remaining_8groups * 8)..height {
        let row_off = row * stride;
        let s_row = &src[row_off..row_off + width];
        let d_row = &dst[row_off..row_off + width];
        let mut sum_s = 0.0f32;
        let mut sum_d = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_prod = 0.0f32;

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let s = s_row[idx];
            let d = d_row[idx];
            if MU1 {
                sum_s += s;
            }
            sum_d += d;
            sum_sq = s.mul_add(s, d.mul_add(d, sum_sq));
            sum_prod = if err {
                let e = s - d;
                e.mul_add(e, sum_prod)
            } else {
                s.mul_add(d, sum_prod)
            };
        }

        for x in 0..width {
            if MU1 {
                out_mu1[row_off + x] = sum_s * inv;
            }
            out_mu2[row_off + x] = sum_d * inv;
            out_sigma_sq[row_off + x] = sum_sq * inv;
            out_sigma12[row_off + x] = sum_prod * inv;

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let sa = s_row[add_idx];
            let da = d_row[add_idx];
            let sr = s_row[rem_idx];
            let dr = d_row[rem_idx];
            if MU1 {
                sum_s = sum_s + sa - sr;
            }
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = if err {
                let ea = sa - da;
                let er = sr - dr;
                ea.mul_add(ea, (-er).mul_add(er, sum_prod))
            } else {
                sa.mul_add(da, (-sr).mul_add(dr, sum_prod))
            };
        }
    }
}

/// AVX2 fallback for fused SSIM horizontal blur.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_inner_v3(
    token: archmage::X64V3Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    // One slice construction per plane: every access below is
    // `(row_base + ro) * width + col` or `row_base * width + idx` with
    // `ro < LANES`, `row_base + LANES <= height`, and every column index
    // clamped by `.min(width - 1)` — so each index is statically `<
    // height * width` and the per-gather/per-store range checks fold away.
    // Same values, same order; this only moves where the bound is proven.
    let n_px = height * width;
    let src = if src.len() >= n_px { &src[..n_px] } else { src };
    let dst = if dst.len() >= n_px { &dst[..n_px] } else { dst };
    let out_mu1 = if out_mu1.len() >= n_px {
        &mut out_mu1[..n_px]
    } else {
        out_mu1
    };
    let out_mu2 = if out_mu2.len() >= n_px {
        &mut out_mu2[..n_px]
    } else {
        out_mu2
    };
    let out_sigma_sq = if out_sigma_sq.len() >= n_px {
        &mut out_sigma_sq[..n_px]
    } else {
        out_sigma_sq
    };
    let out_sigma12 = if out_sigma12.len() >= n_px {
        &mut out_sigma12[..n_px]
    } else {
        out_sigma12
    };

    let mut ring_s = [[0.0f32; 8]; H_RING_CAP];
    let mut ring_d = [[0.0f32; 8]; H_RING_CAP];
    let use_ring = diam <= H_RING_CAP;
    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut ring_pos = 0usize;
        let mut sum_s = f32x8::zero_t(token);
        let mut sum_d = f32x8::zero_t(token);
        let mut sum_sq = f32x8::zero_t(token);
        let mut sum_prod = f32x8::zero_t(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + idx;
                let cs = &src[off..off + 7 * width + 1];
                let cd = &dst[off..off + 7 * width + 1];
                for ro in 0..8 {
                    s_arr[ro] = cs[ro * width];
                    d_arr[ro] = cd[ro * width];
                }
            }
            let sv = f32x8::from_array_t(token, s_arr);
            let dv = f32x8::from_array_t(token, d_arr);
            sum_s = sum_s + sv;
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = if err {
                let e = sv - dv;
                e.mul_add(e, sum_prod)
            } else {
                sv.mul_add(dv, sum_prod)
            };
        }

        for x in 0..width {
            let mu1_result = (sum_s * inv_v).to_array();
            let mu2_result = (sum_d * inv_v).to_array();
            let sq_result = (sum_sq * inv_v).to_array();
            let prod_result = (sum_prod * inv_v).to_array();
            for ro in 0..8 {
                let base = (row_base + ro) * width + x;
                out_mu1[base] = mu1_result[ro];
                out_mu2[base] = mu2_result[ro];
                out_sigma_sq[base] = sq_result[ro];
                out_sigma12[base] = prod_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            {
                // The column's extent is hoisted to ONE range check per gather and
                // LLVM folds the 8 interior ones: `ro < 8` and
                // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                // slice ends exactly one past the highest index the loop below
                // reads, so it is in bounds precisely when the old form was.
                // Same loads, same values, same order — BIT-EXACT; this moves
                // WHERE the bound is proven, not what is read.
                let off = row_base * width + add_idx;
                let cs = &src[off..off + 7 * width + 1];
                let cd = &dst[off..off + 7 * width + 1];
                for ro in 0..8 {
                    s_add[ro] = cs[ro * width];
                    d_add[ro] = cd[ro * width];
                }
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 8];
                let mut da = [0.0f32; 8];
                {
                    // The column's extent is hoisted to ONE range check per gather and
                    // LLVM folds the 8 interior ones: `ro < 8` and
                    // `col.len() == 7*width + 1` give `ro*width < col.len()`. The
                    // slice ends exactly one past the highest index the loop below
                    // reads, so it is in bounds precisely when the old form was.
                    // Same loads, same values, same order — BIT-EXACT; this moves
                    // WHERE the bound is proven, not what is read.
                    let off = row_base * width + rem_idx;
                    let cs = &src[off..off + 7 * width + 1];
                    let cd = &dst[off..off + 7 * width + 1];
                    for ro in 0..8 {
                        sa[ro] = cs[ro * width];
                        da[ro] = cd[ro * width];
                    }
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            let sa = f32x8::from_array_t(token, s_add);
            let da = f32x8::from_array_t(token, d_add);
            let sr = f32x8::from_array_t(token, s_rem);
            let dr = f32x8::from_array_t(token, d_rem);
            sum_s = sum_s + sa - sr;
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = if err {
                let ea = sa - da;
                let er = sr - dr;
                ea.mul_add(ea, (-er).mul_add(er, sum_prod))
            } else {
                sa.mul_add(da, (-sr).mul_add(dr, sum_prod))
            };
        }
    }

    // Scalar remainder rows
    let inv = 1.0 / diam as f32;
    for row in (row_groups * 8)..height {
        let row_off = row * width;
        let s_row = &src[row_off..row_off + width];
        let d_row = &dst[row_off..row_off + width];
        let mut sum_s = 0.0f32;
        let mut sum_d = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_prod = 0.0f32;

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let s = s_row[idx];
            let d = d_row[idx];
            sum_s += s;
            sum_d += d;
            sum_sq = s.mul_add(s, d.mul_add(d, sum_sq));
            sum_prod = if err {
                let e = s - d;
                e.mul_add(e, sum_prod)
            } else {
                s.mul_add(d, sum_prod)
            };
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;
            out_sigma_sq[row_off + x] = sum_sq * inv;
            out_sigma12[row_off + x] = sum_prod * inv;

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let sa = s_row[add_idx];
            let da = d_row[add_idx];
            let sr = s_row[rem_idx];
            let dr = d_row[rem_idx];
            sum_s = sum_s + sa - sr;
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = if err {
                let ea = sa - da;
                let er = sr - dr;
                ea.mul_add(ea, (-er).mul_add(er, sum_prod))
            } else {
                sa.mul_add(da, (-sr).mul_add(dr, sum_prod))
            };
        }
    }
}

#[magetypes(neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_inner(
    token: Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;

    // One 8-row group per call. `valid` (1..=8) is the number of real rows
    // starting at `row_base`; lanes past it alias the LAST real row, so
    // every lane runs the identical vector arithmetic on in-bounds data and
    // only the real rows are stored. The tail rows (height % 8) therefore
    // get bit-for-bit the same numerics as full-group rows on every tier.
    // A separate scalar tail (`f32::mul_add`, i.e. a fused FMA) diverged
    // from the vector body wherever the tier's `mul_add` is the unfused
    // `a * b + c` (scalar polyfill, wasm128), which made the H-blur output
    // depend on how the caller banded the plane — the streaming strips
    // (32 + 2·r rows) and the whole-plane attribution walk (h rows) then
    // disagreed on tail rows. Caught by the i686 CI run (scalar-only
    // dispatch) in `attribution::tests::sum_preservation_*`.
    let mut run_group = |row_base: usize, valid: usize| {
        let mut ring_s = [[0.0f32; 8]; H_RING_CAP];
        let mut ring_d = [[0.0f32; 8]; H_RING_CAP];
        let use_ring = diam <= H_RING_CAP;
        let mut ring_pos = 0usize;
        let mut row_off = [0usize; 8];
        for (ro, off) in row_off.iter_mut().enumerate() {
            *off = (row_base + ro.min(valid - 1)) * width;
        }
        let mut sum_s = f32x8::zero_t(token);
        let mut sum_d = f32x8::zero_t(token);
        let mut sum_sq = f32x8::zero_t(token);
        let mut sum_prod = f32x8::zero_t(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for (ro, &off) in row_off.iter().enumerate() {
                s_arr[ro] = src[off + idx];
                d_arr[ro] = dst[off + idx];
            }
            let sv = f32x8::from_array_t(token, s_arr);
            let dv = f32x8::from_array_t(token, d_arr);
            sum_s = sum_s + sv;
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = if err {
                let e = sv - dv;
                e.mul_add(e, sum_prod)
            } else {
                sv.mul_add(dv, sum_prod)
            };
        }

        for x in 0..width {
            let mu1_result = (sum_s * inv_v).to_array();
            let mu2_result = (sum_d * inv_v).to_array();
            let sq_result = (sum_sq * inv_v).to_array();
            let prod_result = (sum_prod * inv_v).to_array();
            for (ro, &off) in row_off.iter().enumerate().take(valid) {
                let base = off + x;
                out_mu1[base] = mu1_result[ro];
                out_mu2[base] = mu2_result[ro];
                out_sigma_sq[base] = sq_result[ro];
                out_sigma12[base] = prod_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            for (ro, &off) in row_off.iter().enumerate() {
                s_add[ro] = src[off + add_idx];
                d_add[ro] = dst[off + add_idx];
            }
            // rem-ring: for every `x >= diam`, `rem_idx(x)` and
            // `add_idx(x - diam)` BOTH resolve to column `x - r`, unmirrored
            // and unclamped, so the remove-side gather re-reads bytes the
            // add-side gather already loaded. Reuse them from a stack ring —
            // same memory, same values, same `sum + add - rem` order, so
            // BIT-IDENTICAL. Full note at `box_blur_h_inner_v4`.
            let (s_rem, d_rem) = if use_ring && x >= diam {
                (ring_s[ring_pos], ring_d[ring_pos])
            } else {
                let mut sa = [0.0f32; 8];
                let mut da = [0.0f32; 8];
                for (ro, &off) in row_off.iter().enumerate() {
                    sa[ro] = src[off + rem_idx];
                    da[ro] = dst[off + rem_idx];
                }
                (sa, da)
            };
            if use_ring {
                ring_s[ring_pos] = s_add;
                ring_d[ring_pos] = d_add;
                ring_pos += 1;
                if ring_pos == diam {
                    ring_pos = 0;
                }
            }
            let sa = f32x8::from_array_t(token, s_add);
            let da = f32x8::from_array_t(token, d_add);
            let sr = f32x8::from_array_t(token, s_rem);
            let dr = f32x8::from_array_t(token, d_rem);
            sum_s = sum_s + sa - sr;
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = if err {
                let ea = sa - da;
                let er = sr - dr;
                ea.mul_add(ea, (-er).mul_add(er, sum_prod))
            } else {
                sa.mul_add(da, (-sr).mul_add(dr, sum_prod))
            };
        }
    };

    let row_groups = height / 8;
    for rg in 0..row_groups {
        run_group(rg * 8, 8);
    }
    let tail = height - row_groups * 8;
    if tail > 0 {
        run_group(row_groups * 8, tail);
    }
}

/// One row's Rec64 horizontal SSIM window — the scalar canonical body,
/// extracted verbatim so the 8-row vector body and the scalar/wasm stubs
/// share exactly this op sequence for the `height % 8` tail.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn fused_blur_h_rec64_row(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    y: usize,
    width: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    let row = y * width;
    let mut sum_s = 0.0f64;
    let mut sum_d = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum_prod = 0.0f64;
    for i in 0..diam {
        let idx = if i <= r {
            (r - i).min(width - 1)
        } else {
            (i - r).min(width - 1)
        };
        let s = src[row + idx] as f64;
        let d = dst[row + idx] as f64;
        sum_s += s;
        sum_d += d;
        sum_sq = s.mul_add(s, d.mul_add(d, sum_sq));
        sum_prod = if err {
            let e = s - d;
            e.mul_add(e, sum_prod)
        } else {
            s.mul_add(d, sum_prod)
        };
    }
    for x in 0..width {
        out_mu1[row + x] = (sum_s * inv_v64) as f32;
        out_mu2[row + x] = (sum_d * inv_v64) as f32;
        out_sigma_sq[row + x] = (sum_sq * inv_v64) as f32;
        out_sigma12[row + x] = (sum_prod * inv_v64) as f32;

        let add_raw = x + r + 1;
        let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
        let rem_i = x as isize - r as isize;
        let rem_idx = (if rem_i < 0 {
            rem_i.unsigned_abs()
        } else {
            rem_i as usize
        })
        .min(width - 1);
        let sa = src[row + add_idx] as f64;
        let da = dst[row + add_idx] as f64;
        let sr = src[row + rem_idx] as f64;
        let dr = dst[row + rem_idx] as f64;
        sum_s = sum_s + sa - sr;
        sum_d = sum_d + da - dr;
        sum_sq = sa.mul_add(
            sa,
            da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
        );
        sum_prod = if err {
            let ea = sa - da;
            let er = sr - dr;
            ea.mul_add(ea, (-er).mul_add(er, sum_prod))
        } else {
            sa.mul_add(da, (-sr).mul_add(dr, sum_prod))
        };
    }
}

// Vector types for the Rec64 lane-parallel bodies' signatures — the
// intrinsic ops themselves come from each `#[rite]` body's
// `import_intrinsics` scope; the types are re-exports of `core::arch`'s.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use archmage::intrinsics::x86_64::__m512d;
#[cfg(target_arch = "x86_64")]
use archmage::intrinsics::x86_64::{__m256, __m256d};

/// `f64x8` as two `__m256d` halves — the lane layout the AVX2 (v3) body
/// uses (lanes 0..4 in `.0`, 4..8 in `.1`).
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
struct Rec64D8(__m256d, __m256d);

/// 8×8 f32 transpose in registers — the standard 24-op `unpack`/`shuffle`/
/// `perm2f128` network. Written at v3 level so every fused-FMA tier body
/// (v3's own and the wider ones, whose feature sets cover v3) can call it.
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3, import_intrinsics)]
fn rec64_t8x8(m: [__m256; 8]) -> [__m256; 8] {
    let t0 = _mm256_unpacklo_ps(m[0], m[1]);
    let t1 = _mm256_unpackhi_ps(m[0], m[1]);
    let t2 = _mm256_unpacklo_ps(m[2], m[3]);
    let t3 = _mm256_unpackhi_ps(m[2], m[3]);
    let t4 = _mm256_unpacklo_ps(m[4], m[5]);
    let t5 = _mm256_unpackhi_ps(m[4], m[5]);
    let t6 = _mm256_unpacklo_ps(m[6], m[7]);
    let t7 = _mm256_unpackhi_ps(m[6], m[7]);
    let u0 = _mm256_shuffle_ps(t0, t2, 0x44);
    let u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    let u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    let u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    let u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    let u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    let u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    let u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    [
        _mm256_permute2f128_ps(u0, u4, 0x20),
        _mm256_permute2f128_ps(u1, u5, 0x20),
        _mm256_permute2f128_ps(u2, u6, 0x20),
        _mm256_permute2f128_ps(u3, u7, 0x20),
        _mm256_permute2f128_ps(u0, u4, 0x31),
        _mm256_permute2f128_ps(u1, u5, 0x31),
        _mm256_permute2f128_ps(u2, u6, 0x31),
        _mm256_permute2f128_ps(u3, u7, 0x31),
    ]
}

/// Load the 8×8 f32 block `rows[*][col .. col + 8]` and return its
/// transpose: `out[j][lane]` is `rows[lane][col + j]`. The caller
/// guarantees `col + 8 <= rows[l].len()` (the block-phase x schedule
/// keeps every column in-range; `try_into` stays as the cheap guard).
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3, import_intrinsics)]
fn rec64_ld8x8(rows: [&[f32]; 8], col: usize) -> [__m256; 8] {
    let mut v = [_mm256_setzero_ps(); 8];
    for l in 0..8 {
        let c: &[f32; 8] = rows[l][col..col + 8].try_into().unwrap();
        v[l] = _mm256_loadu_ps(c);
    }
    rec64_t8x8(v)
}

/// Store 8 columns (`cols[j][lane]`) into `out[lane][x0 .. x0 + 8]` —
/// transpose back, then one vector store per row.
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3, import_intrinsics)]
fn rec64_store8x8(out: &mut [&mut [f32]; 8], x0: usize, cols: [__m256; 8]) {
    let t = rec64_t8x8(cols);
    for l in 0..8 {
        let c: &mut [f32; 8] = (&mut out[l][x0..x0 + 8]).try_into().unwrap();
        _mm256_storeu_ps(c, t[l]);
    }
}

/// One column `idx` of the eight `rows`, widened to `__m512d` — the
/// window-init / mirrored-edge feed where the add/remove index is not part
/// of a contiguous block. Same per-lane `as f64` values as the scalar body.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4, import_intrinsics)]
fn rec64_gather8_512(rows: [&[f32]; 8], idx: usize) -> __m512d {
    let a: [f64; 8] = std::array::from_fn(|l| rows[l][idx] as f64);
    _mm512_loadu_pd(&a)
}

/// Store `v`'s lanes into `out[lane][x]` — the scalar-path emit for the
/// head/tail steps: `vcvtpd2ps` is per-lane `(x) as f32` (round-to-nearest).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4, import_intrinsics)]
fn rec64_emit8_512(out: &mut [&mut [f32]; 8], x: usize, v: __m512d) {
    let mut a = [0.0f32; 8];
    _mm256_storeu_ps(&mut a, _mm512_cvtpd_ps(v));
    for l in 0..8 {
        out[l][x] = a[l];
    }
}

/// `Rec64D8` counterparts of the AVX-512 helpers — the same values, split
/// over two `__m256d` halves.
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3, import_intrinsics)]
fn rec64_gather8_256(rows: [&[f32]; 8], idx: usize) -> Rec64D8 {
    let a: [f64; 8] = std::array::from_fn(|l| rows[l][idx] as f64);
    Rec64D8(
        _mm256_loadu_pd(a[0..4].try_into().unwrap()),
        _mm256_loadu_pd(a[4..8].try_into().unwrap()),
    )
}

#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3, import_intrinsics)]
fn rec64_widen8_256(v: __m256) -> Rec64D8 {
    Rec64D8(
        _mm256_cvtps_pd(_mm256_castps256_ps128(v)),
        _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)),
    )
}

#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3, import_intrinsics)]
fn rec64_narrow8_256(a: Rec64D8) -> __m256 {
    _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm256_cvtpd_ps(a.0)),
        _mm256_cvtpd_ps(a.1),
        1,
    )
}

#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3, import_intrinsics)]
fn rec64_emit8_256(out: &mut [&mut [f32]; 8], x: usize, v: Rec64D8) {
    let mut a = [0.0f32; 8];
    _mm256_storeu_ps(&mut a, rec64_narrow8_256(v));
    for l in 0..8 {
        out[l][x] = a[l];
    }
}

/// `plane`'s `G` consecutive 8-row groups as `G × 8` mutable row slices.
#[inline(always)]
fn rec64_rows8g_mut<const G: usize>(plane: &mut [f32], width: usize) -> [[&mut [f32]; 8]; G] {
    let mut it = plane.chunks_exact_mut(width);
    std::array::from_fn(|_| std::array::from_fn(|_| it.next().unwrap()))
}

/// **canon64 vector body** — the Rec64 horizontal window run on `G × 8`
/// independent ROWS at a time (`G` interleaved 8-row groups share the same
/// x schedule so their recurrence chains overlap). Every lane is one row's
/// own scalar recurrence (`fused_blur_h_rec64_row`), so the per-row op
/// sequence — the `+add −rem` sum slide, the `mul_add`-fused
/// square/product updates in their exact order — is the scalar body's
/// verbatim; lanes never mix.
///
/// The x range splits three ways: the `x < r` head and the tail keep the
/// scalar-gather body (mirrored add/remove indices); the interior, where
/// `x0 - r .. x0 + 7 - r` and `x0 + r + 1 .. x0 + r + 8` are all in-range
/// columns, runs in 8-column blocks — 8 row loads + one 8×8 transpose per
/// plane per block instead of per-step gathers, and the four
/// `(sum * inv_v64) as f32` stores likewise go through a staged transpose.
/// `height % 8` tail rows take the scalar row body. `_scalar`/`_wasm128`
/// keep the scalar body: their `mul_add` is unfused either way.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4, import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn rec64_run512<const G: usize>(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    let inv = _mm512_set1_pd(inv_v64);
    let zero = _mm512_setzero_pd();
    let neg = _mm512_set1_pd(-0.0);
    let wm1 = width - 1;
    let head_end = r.min(width);
    let n_blocks = if width >= 2 * r + 9 {
        (width - 2 * r - 9) / 8 + 1
    } else {
        0
    };
    let srows: [[&[f32]; 8]; G] = std::array::from_fn(|g| {
        std::array::from_fn(|l| &src[(g * 8 + l) * width..(g * 8 + l + 1) * width])
    });
    let drows: [[&[f32]; 8]; G] = std::array::from_fn(|g| {
        std::array::from_fn(|l| &dst[(g * 8 + l) * width..(g * 8 + l + 1) * width])
    });
    let mut om1 = rec64_rows8g_mut::<G>(out_mu1, width);
    let mut om2 = rec64_rows8g_mut::<G>(out_mu2, width);
    let mut osq = rec64_rows8g_mut::<G>(out_sigma_sq, width);
    let mut os12 = rec64_rows8g_mut::<G>(out_sigma12, width);
    let mut sum_s = [zero; G];
    let mut sum_d = [zero; G];
    let mut sum_sq = [zero; G];
    let mut sum_prod = [zero; G];
    for i in 0..diam {
        let idx = if i <= r {
            (r - i).min(wm1)
        } else {
            (i - r).min(wm1)
        };
        for g in 0..G {
            let sa = rec64_gather8_512(srows[g], idx);
            let da = rec64_gather8_512(drows[g], idx);
            sum_s[g] = _mm512_add_pd(sum_s[g], sa);
            sum_d[g] = _mm512_add_pd(sum_d[g], da);
            sum_sq[g] = _mm512_fmadd_pd(sa, sa, _mm512_fmadd_pd(da, da, sum_sq[g]));
            sum_prod[g] = if err {
                let e = _mm512_sub_pd(sa, da);
                _mm512_fmadd_pd(e, e, sum_prod[g])
            } else {
                _mm512_fmadd_pd(sa, da, sum_prod[g])
            };
        }
    }
    for x in 0..head_end {
        let add_idx = h_mirror_add_idx(x + r + 1, width).min(wm1);
        let rem_idx = (r - x).min(wm1);
        for g in 0..G {
            let sa = rec64_gather8_512(srows[g], add_idx);
            let da = rec64_gather8_512(drows[g], add_idx);
            let sr = rec64_gather8_512(srows[g], rem_idx);
            let dr = rec64_gather8_512(drows[g], rem_idx);
            rec64_emit8_512(&mut om1[g], x, _mm512_mul_pd(sum_s[g], inv));
            rec64_emit8_512(&mut om2[g], x, _mm512_mul_pd(sum_d[g], inv));
            rec64_emit8_512(&mut osq[g], x, _mm512_mul_pd(sum_sq[g], inv));
            rec64_emit8_512(&mut os12[g], x, _mm512_mul_pd(sum_prod[g], inv));
            sum_s[g] = _mm512_sub_pd(_mm512_add_pd(sum_s[g], sa), sr);
            sum_d[g] = _mm512_sub_pd(_mm512_add_pd(sum_d[g], da), dr);
            sum_sq[g] = _mm512_fmadd_pd(
                sa,
                sa,
                _mm512_fmadd_pd(
                    da,
                    da,
                    _mm512_fmadd_pd(
                        _mm512_xor_pd(sr, neg),
                        sr,
                        _mm512_fmadd_pd(_mm512_xor_pd(dr, neg), dr, sum_sq[g]),
                    ),
                ),
            );
            sum_prod[g] = if err {
                let ea = _mm512_sub_pd(sa, da);
                let er = _mm512_sub_pd(sr, dr);
                _mm512_fmadd_pd(
                    ea,
                    ea,
                    _mm512_fmadd_pd(_mm512_xor_pd(er, neg), er, sum_prod[g]),
                )
            } else {
                _mm512_fmadd_pd(
                    sa,
                    da,
                    _mm512_fmadd_pd(_mm512_xor_pd(sr, neg), dr, sum_prod[g]),
                )
            };
        }
    }
    // KERNOPT2: an add-column history ring for the remove side was
    // measured (A/B, Rev4 scalar 1024, 4 rounds interleaved) at < 0.5 %
    // of the cell — below the 3 % keep bar, so the remove side loads
    // like the add side.
    let mut x0 = r;
    for _ in 0..n_blocks {
        // `from_fn` builds the block columns in place — a `[zero; 8]`
        // fill ahead of the loads shows up as dead `vmovdqa64` spills in
        // the hot loop.
        let sadd: [[__m256; 8]; G] = std::array::from_fn(|g| rec64_ld8x8(srows[g], x0 + r + 1));
        let dadd: [[__m256; 8]; G] = std::array::from_fn(|g| rec64_ld8x8(drows[g], x0 + r + 1));
        let srem: [[__m256; 8]; G] = std::array::from_fn(|g| rec64_ld8x8(srows[g], x0 - r));
        let drem: [[__m256; 8]; G] = std::array::from_fn(|g| rec64_ld8x8(drows[g], x0 - r));
        let mut stage = [[[_mm256_setzero_ps(); 8]; 4]; G];
        for j in 0..8 {
            for g in 0..G {
                let sa = _mm512_cvtps_pd(sadd[g][j]);
                let da = _mm512_cvtps_pd(dadd[g][j]);
                let sr = _mm512_cvtps_pd(srem[g][j]);
                let dr = _mm512_cvtps_pd(drem[g][j]);
                stage[g][0][j] = _mm512_cvtpd_ps(_mm512_mul_pd(sum_s[g], inv));
                stage[g][1][j] = _mm512_cvtpd_ps(_mm512_mul_pd(sum_d[g], inv));
                stage[g][2][j] = _mm512_cvtpd_ps(_mm512_mul_pd(sum_sq[g], inv));
                stage[g][3][j] = _mm512_cvtpd_ps(_mm512_mul_pd(sum_prod[g], inv));
                sum_s[g] = _mm512_sub_pd(_mm512_add_pd(sum_s[g], sa), sr);
                sum_d[g] = _mm512_sub_pd(_mm512_add_pd(sum_d[g], da), dr);
                sum_sq[g] = _mm512_fmadd_pd(
                    sa,
                    sa,
                    _mm512_fmadd_pd(
                        da,
                        da,
                        _mm512_fmadd_pd(
                            _mm512_xor_pd(sr, neg),
                            sr,
                            _mm512_fmadd_pd(_mm512_xor_pd(dr, neg), dr, sum_sq[g]),
                        ),
                    ),
                );
                sum_prod[g] = if err {
                    let ea = _mm512_sub_pd(sa, da);
                    let er = _mm512_sub_pd(sr, dr);
                    _mm512_fmadd_pd(
                        ea,
                        ea,
                        _mm512_fmadd_pd(_mm512_xor_pd(er, neg), er, sum_prod[g]),
                    )
                } else {
                    _mm512_fmadd_pd(
                        sa,
                        da,
                        _mm512_fmadd_pd(_mm512_xor_pd(sr, neg), dr, sum_prod[g]),
                    )
                };
            }
        }
        for g in 0..G {
            rec64_store8x8(&mut om1[g], x0, stage[g][0]);
            rec64_store8x8(&mut om2[g], x0, stage[g][1]);
            rec64_store8x8(&mut osq[g], x0, stage[g][2]);
            rec64_store8x8(&mut os12[g], x0, stage[g][3]);
        }
        x0 += 8;
    }
    for x in x0..width {
        let add_idx = h_mirror_add_idx(x + r + 1, width).min(wm1);
        let rem_idx = x - r;
        for g in 0..G {
            let sa = rec64_gather8_512(srows[g], add_idx);
            let da = rec64_gather8_512(drows[g], add_idx);
            let sr = rec64_gather8_512(srows[g], rem_idx);
            let dr = rec64_gather8_512(drows[g], rem_idx);
            rec64_emit8_512(&mut om1[g], x, _mm512_mul_pd(sum_s[g], inv));
            rec64_emit8_512(&mut om2[g], x, _mm512_mul_pd(sum_d[g], inv));
            rec64_emit8_512(&mut osq[g], x, _mm512_mul_pd(sum_sq[g], inv));
            rec64_emit8_512(&mut os12[g], x, _mm512_mul_pd(sum_prod[g], inv));
            sum_s[g] = _mm512_sub_pd(_mm512_add_pd(sum_s[g], sa), sr);
            sum_d[g] = _mm512_sub_pd(_mm512_add_pd(sum_d[g], da), dr);
            sum_sq[g] = _mm512_fmadd_pd(
                sa,
                sa,
                _mm512_fmadd_pd(
                    da,
                    da,
                    _mm512_fmadd_pd(
                        _mm512_xor_pd(sr, neg),
                        sr,
                        _mm512_fmadd_pd(_mm512_xor_pd(dr, neg), dr, sum_sq[g]),
                    ),
                ),
            );
            sum_prod[g] = if err {
                let ea = _mm512_sub_pd(sa, da);
                let er = _mm512_sub_pd(sr, dr);
                _mm512_fmadd_pd(
                    ea,
                    ea,
                    _mm512_fmadd_pd(_mm512_xor_pd(er, neg), er, sum_prod[g]),
                )
            } else {
                _mm512_fmadd_pd(
                    sa,
                    da,
                    _mm512_fmadd_pd(_mm512_xor_pd(sr, neg), dr, sum_prod[g]),
                )
            };
        }
    }
}

/// AVX-512 driver for [`rec64_run512`]: 16-row interleaved chunks, an
/// 8-row chunk for the `height % 16` remainder, scalar rows under 8.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4)]
#[allow(clippy::too_many_arguments)]
fn rec64_rows8x8(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    let mut y0 = 0;
    while y0 + 16 <= height {
        rec64_run512::<2>(
            &src[y0 * width..(y0 + 16) * width],
            &dst[y0 * width..(y0 + 16) * width],
            &mut out_mu1[y0 * width..(y0 + 16) * width],
            &mut out_mu2[y0 * width..(y0 + 16) * width],
            &mut out_sigma_sq[y0 * width..(y0 + 16) * width],
            &mut out_sigma12[y0 * width..(y0 + 16) * width],
            width,
            r,
            diam,
            inv_v64,
            err,
        );
        y0 += 16;
    }
    if y0 + 8 <= height {
        rec64_run512::<1>(
            &src[y0 * width..(y0 + 8) * width],
            &dst[y0 * width..(y0 + 8) * width],
            &mut out_mu1[y0 * width..(y0 + 8) * width],
            &mut out_mu2[y0 * width..(y0 + 8) * width],
            &mut out_sigma_sq[y0 * width..(y0 + 8) * width],
            &mut out_sigma12[y0 * width..(y0 + 8) * width],
            width,
            r,
            diam,
            inv_v64,
            err,
        );
        y0 += 8;
    }
    for y in y0..height {
        fused_blur_h_rec64_row(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            y,
            width,
            r,
            diam,
            inv_v64,
            err,
        );
    }
}

/// Without `avx512` the `_v4x`/`_v4` dispatch names can't be reached (the
/// tokens are unsummonable) but still must exist — `rec64_rows8x8` keeps
/// the same name as the AVX-512 driver and delegates to the AVX2 body.
#[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
#[archmage::rite(v3)]
#[allow(clippy::too_many_arguments)]
fn rec64_rows8x8(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    rec64_rows8x8_avx2(
        src,
        dst,
        out_mu1,
        out_mu2,
        out_sigma_sq,
        out_sigma12,
        width,
        height,
        r,
        diam,
        inv_v64,
        err,
    );
}

/// The AVX2 (`v3`) lane-parallel body: identical x schedule and op order
/// to [`rec64_run512`], with `Rec64D8` pairs standing in for `__m512d`.
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3, import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn rec64_run256<const G: usize>(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    let zero = _mm256_setzero_pd();
    let neg = _mm256_set1_pd(-0.0);
    let inv = Rec64D8(_mm256_set1_pd(inv_v64), _mm256_set1_pd(inv_v64));
    let wm1 = width - 1;
    let head_end = r.min(width);
    let n_blocks = if width >= 2 * r + 9 {
        (width - 2 * r - 9) / 8 + 1
    } else {
        0
    };
    let srows: [[&[f32]; 8]; G] = std::array::from_fn(|g| {
        std::array::from_fn(|l| &src[(g * 8 + l) * width..(g * 8 + l + 1) * width])
    });
    let drows: [[&[f32]; 8]; G] = std::array::from_fn(|g| {
        std::array::from_fn(|l| &dst[(g * 8 + l) * width..(g * 8 + l + 1) * width])
    });
    let mut om1 = rec64_rows8g_mut::<G>(out_mu1, width);
    let mut om2 = rec64_rows8g_mut::<G>(out_mu2, width);
    let mut osq = rec64_rows8g_mut::<G>(out_sigma_sq, width);
    let mut os12 = rec64_rows8g_mut::<G>(out_sigma12, width);
    let zero8 = Rec64D8(zero, zero);
    let mut sum_s = [zero8; G];
    let mut sum_d = [zero8; G];
    let mut sum_sq = [zero8; G];
    let mut sum_prod = [zero8; G];
    for i in 0..diam {
        let idx = if i <= r {
            (r - i).min(wm1)
        } else {
            (i - r).min(wm1)
        };
        for g in 0..G {
            let sa = rec64_gather8_256(srows[g], idx);
            let da = rec64_gather8_256(drows[g], idx);
            sum_s[g] = Rec64D8(
                _mm256_add_pd(sum_s[g].0, sa.0),
                _mm256_add_pd(sum_s[g].1, sa.1),
            );
            sum_d[g] = Rec64D8(
                _mm256_add_pd(sum_d[g].0, da.0),
                _mm256_add_pd(sum_d[g].1, da.1),
            );
            sum_sq[g] = Rec64D8(
                _mm256_fmadd_pd(sa.0, sa.0, _mm256_fmadd_pd(da.0, da.0, sum_sq[g].0)),
                _mm256_fmadd_pd(sa.1, sa.1, _mm256_fmadd_pd(da.1, da.1, sum_sq[g].1)),
            );
            sum_prod[g] = if err {
                let e = Rec64D8(_mm256_sub_pd(sa.0, da.0), _mm256_sub_pd(sa.1, da.1));
                Rec64D8(
                    _mm256_fmadd_pd(e.0, e.0, sum_prod[g].0),
                    _mm256_fmadd_pd(e.1, e.1, sum_prod[g].1),
                )
            } else {
                Rec64D8(
                    _mm256_fmadd_pd(sa.0, da.0, sum_prod[g].0),
                    _mm256_fmadd_pd(sa.1, da.1, sum_prod[g].1),
                )
            };
        }
    }
    for x in 0..head_end {
        let add_idx = h_mirror_add_idx(x + r + 1, width).min(wm1);
        let rem_idx = (r - x).min(wm1);
        for g in 0..G {
            let sa = rec64_gather8_256(srows[g], add_idx);
            let da = rec64_gather8_256(drows[g], add_idx);
            let sr = rec64_gather8_256(srows[g], rem_idx);
            let dr = rec64_gather8_256(drows[g], rem_idx);
            rec64_emit8_256(
                &mut om1[g],
                x,
                Rec64D8(
                    _mm256_mul_pd(sum_s[g].0, inv.0),
                    _mm256_mul_pd(sum_s[g].1, inv.1),
                ),
            );
            rec64_emit8_256(
                &mut om2[g],
                x,
                Rec64D8(
                    _mm256_mul_pd(sum_d[g].0, inv.0),
                    _mm256_mul_pd(sum_d[g].1, inv.1),
                ),
            );
            rec64_emit8_256(
                &mut osq[g],
                x,
                Rec64D8(
                    _mm256_mul_pd(sum_sq[g].0, inv.0),
                    _mm256_mul_pd(sum_sq[g].1, inv.1),
                ),
            );
            rec64_emit8_256(
                &mut os12[g],
                x,
                Rec64D8(
                    _mm256_mul_pd(sum_prod[g].0, inv.0),
                    _mm256_mul_pd(sum_prod[g].1, inv.1),
                ),
            );
            sum_s[g] = Rec64D8(
                _mm256_sub_pd(_mm256_add_pd(sum_s[g].0, sa.0), sr.0),
                _mm256_sub_pd(_mm256_add_pd(sum_s[g].1, sa.1), sr.1),
            );
            sum_d[g] = Rec64D8(
                _mm256_sub_pd(_mm256_add_pd(sum_d[g].0, da.0), dr.0),
                _mm256_sub_pd(_mm256_add_pd(sum_d[g].1, da.1), dr.1),
            );
            sum_sq[g] = Rec64D8(
                _mm256_fmadd_pd(
                    sa.0,
                    sa.0,
                    _mm256_fmadd_pd(
                        da.0,
                        da.0,
                        _mm256_fmadd_pd(
                            _mm256_xor_pd(sr.0, neg),
                            sr.0,
                            _mm256_fmadd_pd(_mm256_xor_pd(dr.0, neg), dr.0, sum_sq[g].0),
                        ),
                    ),
                ),
                _mm256_fmadd_pd(
                    sa.1,
                    sa.1,
                    _mm256_fmadd_pd(
                        da.1,
                        da.1,
                        _mm256_fmadd_pd(
                            _mm256_xor_pd(sr.1, neg),
                            sr.1,
                            _mm256_fmadd_pd(_mm256_xor_pd(dr.1, neg), dr.1, sum_sq[g].1),
                        ),
                    ),
                ),
            );
            sum_prod[g] = if err {
                let ea = Rec64D8(_mm256_sub_pd(sa.0, da.0), _mm256_sub_pd(sa.1, da.1));
                let er = Rec64D8(_mm256_sub_pd(sr.0, dr.0), _mm256_sub_pd(sr.1, dr.1));
                Rec64D8(
                    _mm256_fmadd_pd(
                        ea.0,
                        ea.0,
                        _mm256_fmadd_pd(_mm256_xor_pd(er.0, neg), er.0, sum_prod[g].0),
                    ),
                    _mm256_fmadd_pd(
                        ea.1,
                        ea.1,
                        _mm256_fmadd_pd(_mm256_xor_pd(er.1, neg), er.1, sum_prod[g].1),
                    ),
                )
            } else {
                Rec64D8(
                    _mm256_fmadd_pd(
                        sa.0,
                        da.0,
                        _mm256_fmadd_pd(_mm256_xor_pd(sr.0, neg), dr.0, sum_prod[g].0),
                    ),
                    _mm256_fmadd_pd(
                        sa.1,
                        da.1,
                        _mm256_fmadd_pd(_mm256_xor_pd(sr.1, neg), dr.1, sum_prod[g].1),
                    ),
                )
            };
        }
    }
    let mut x0 = r;
    for _ in 0..n_blocks {
        let sadd: [[__m256; 8]; G] = std::array::from_fn(|g| rec64_ld8x8(srows[g], x0 + r + 1));
        let dadd: [[__m256; 8]; G] = std::array::from_fn(|g| rec64_ld8x8(drows[g], x0 + r + 1));
        let srem: [[__m256; 8]; G] = std::array::from_fn(|g| rec64_ld8x8(srows[g], x0 - r));
        let drem: [[__m256; 8]; G] = std::array::from_fn(|g| rec64_ld8x8(drows[g], x0 - r));
        let mut stage = [[[_mm256_setzero_ps(); 8]; 4]; G];
        for j in 0..8 {
            for g in 0..G {
                let sa = rec64_widen8_256(sadd[g][j]);
                let da = rec64_widen8_256(dadd[g][j]);
                let sr = rec64_widen8_256(srem[g][j]);
                let dr = rec64_widen8_256(drem[g][j]);
                stage[g][0][j] = rec64_narrow8_256(Rec64D8(
                    _mm256_mul_pd(sum_s[g].0, inv.0),
                    _mm256_mul_pd(sum_s[g].1, inv.1),
                ));
                stage[g][1][j] = rec64_narrow8_256(Rec64D8(
                    _mm256_mul_pd(sum_d[g].0, inv.0),
                    _mm256_mul_pd(sum_d[g].1, inv.1),
                ));
                stage[g][2][j] = rec64_narrow8_256(Rec64D8(
                    _mm256_mul_pd(sum_sq[g].0, inv.0),
                    _mm256_mul_pd(sum_sq[g].1, inv.1),
                ));
                stage[g][3][j] = rec64_narrow8_256(Rec64D8(
                    _mm256_mul_pd(sum_prod[g].0, inv.0),
                    _mm256_mul_pd(sum_prod[g].1, inv.1),
                ));
                sum_s[g] = Rec64D8(
                    _mm256_sub_pd(_mm256_add_pd(sum_s[g].0, sa.0), sr.0),
                    _mm256_sub_pd(_mm256_add_pd(sum_s[g].1, sa.1), sr.1),
                );
                sum_d[g] = Rec64D8(
                    _mm256_sub_pd(_mm256_add_pd(sum_d[g].0, da.0), dr.0),
                    _mm256_sub_pd(_mm256_add_pd(sum_d[g].1, da.1), dr.1),
                );
                sum_sq[g] = Rec64D8(
                    _mm256_fmadd_pd(
                        sa.0,
                        sa.0,
                        _mm256_fmadd_pd(
                            da.0,
                            da.0,
                            _mm256_fmadd_pd(
                                _mm256_xor_pd(sr.0, neg),
                                sr.0,
                                _mm256_fmadd_pd(_mm256_xor_pd(dr.0, neg), dr.0, sum_sq[g].0),
                            ),
                        ),
                    ),
                    _mm256_fmadd_pd(
                        sa.1,
                        sa.1,
                        _mm256_fmadd_pd(
                            da.1,
                            da.1,
                            _mm256_fmadd_pd(
                                _mm256_xor_pd(sr.1, neg),
                                sr.1,
                                _mm256_fmadd_pd(_mm256_xor_pd(dr.1, neg), dr.1, sum_sq[g].1),
                            ),
                        ),
                    ),
                );
                sum_prod[g] = if err {
                    let ea = Rec64D8(_mm256_sub_pd(sa.0, da.0), _mm256_sub_pd(sa.1, da.1));
                    let er = Rec64D8(_mm256_sub_pd(sr.0, dr.0), _mm256_sub_pd(sr.1, dr.1));
                    Rec64D8(
                        _mm256_fmadd_pd(
                            ea.0,
                            ea.0,
                            _mm256_fmadd_pd(_mm256_xor_pd(er.0, neg), er.0, sum_prod[g].0),
                        ),
                        _mm256_fmadd_pd(
                            ea.1,
                            ea.1,
                            _mm256_fmadd_pd(_mm256_xor_pd(er.1, neg), er.1, sum_prod[g].1),
                        ),
                    )
                } else {
                    Rec64D8(
                        _mm256_fmadd_pd(
                            sa.0,
                            da.0,
                            _mm256_fmadd_pd(_mm256_xor_pd(sr.0, neg), dr.0, sum_prod[g].0),
                        ),
                        _mm256_fmadd_pd(
                            sa.1,
                            da.1,
                            _mm256_fmadd_pd(_mm256_xor_pd(sr.1, neg), dr.1, sum_prod[g].1),
                        ),
                    )
                };
            }
        }
        for g in 0..G {
            rec64_store8x8(&mut om1[g], x0, stage[g][0]);
            rec64_store8x8(&mut om2[g], x0, stage[g][1]);
            rec64_store8x8(&mut osq[g], x0, stage[g][2]);
            rec64_store8x8(&mut os12[g], x0, stage[g][3]);
        }
        x0 += 8;
    }
    for x in x0..width {
        let add_idx = h_mirror_add_idx(x + r + 1, width).min(wm1);
        let rem_idx = x - r;
        for g in 0..G {
            let sa = rec64_gather8_256(srows[g], add_idx);
            let da = rec64_gather8_256(drows[g], add_idx);
            let sr = rec64_gather8_256(srows[g], rem_idx);
            let dr = rec64_gather8_256(drows[g], rem_idx);
            rec64_emit8_256(
                &mut om1[g],
                x,
                Rec64D8(
                    _mm256_mul_pd(sum_s[g].0, inv.0),
                    _mm256_mul_pd(sum_s[g].1, inv.1),
                ),
            );
            rec64_emit8_256(
                &mut om2[g],
                x,
                Rec64D8(
                    _mm256_mul_pd(sum_d[g].0, inv.0),
                    _mm256_mul_pd(sum_d[g].1, inv.1),
                ),
            );
            rec64_emit8_256(
                &mut osq[g],
                x,
                Rec64D8(
                    _mm256_mul_pd(sum_sq[g].0, inv.0),
                    _mm256_mul_pd(sum_sq[g].1, inv.1),
                ),
            );
            rec64_emit8_256(
                &mut os12[g],
                x,
                Rec64D8(
                    _mm256_mul_pd(sum_prod[g].0, inv.0),
                    _mm256_mul_pd(sum_prod[g].1, inv.1),
                ),
            );
            sum_s[g] = Rec64D8(
                _mm256_sub_pd(_mm256_add_pd(sum_s[g].0, sa.0), sr.0),
                _mm256_sub_pd(_mm256_add_pd(sum_s[g].1, sa.1), sr.1),
            );
            sum_d[g] = Rec64D8(
                _mm256_sub_pd(_mm256_add_pd(sum_d[g].0, da.0), dr.0),
                _mm256_sub_pd(_mm256_add_pd(sum_d[g].1, da.1), dr.1),
            );
            sum_sq[g] = Rec64D8(
                _mm256_fmadd_pd(
                    sa.0,
                    sa.0,
                    _mm256_fmadd_pd(
                        da.0,
                        da.0,
                        _mm256_fmadd_pd(
                            _mm256_xor_pd(sr.0, neg),
                            sr.0,
                            _mm256_fmadd_pd(_mm256_xor_pd(dr.0, neg), dr.0, sum_sq[g].0),
                        ),
                    ),
                ),
                _mm256_fmadd_pd(
                    sa.1,
                    sa.1,
                    _mm256_fmadd_pd(
                        da.1,
                        da.1,
                        _mm256_fmadd_pd(
                            _mm256_xor_pd(sr.1, neg),
                            sr.1,
                            _mm256_fmadd_pd(_mm256_xor_pd(dr.1, neg), dr.1, sum_sq[g].1),
                        ),
                    ),
                ),
            );
            sum_prod[g] = if err {
                let ea = Rec64D8(_mm256_sub_pd(sa.0, da.0), _mm256_sub_pd(sa.1, da.1));
                let er = Rec64D8(_mm256_sub_pd(sr.0, dr.0), _mm256_sub_pd(sr.1, dr.1));
                Rec64D8(
                    _mm256_fmadd_pd(
                        ea.0,
                        ea.0,
                        _mm256_fmadd_pd(_mm256_xor_pd(er.0, neg), er.0, sum_prod[g].0),
                    ),
                    _mm256_fmadd_pd(
                        ea.1,
                        ea.1,
                        _mm256_fmadd_pd(_mm256_xor_pd(er.1, neg), er.1, sum_prod[g].1),
                    ),
                )
            } else {
                Rec64D8(
                    _mm256_fmadd_pd(
                        sa.0,
                        da.0,
                        _mm256_fmadd_pd(_mm256_xor_pd(sr.0, neg), dr.0, sum_prod[g].0),
                    ),
                    _mm256_fmadd_pd(
                        sa.1,
                        da.1,
                        _mm256_fmadd_pd(_mm256_xor_pd(sr.1, neg), dr.1, sum_prod[g].1),
                    ),
                )
            };
        }
    }
}

/// AVX2 driver for [`rec64_run256`] — same chunking as [`rec64_rows8x8`].
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
#[allow(clippy::too_many_arguments)]
fn rec64_rows8x8_avx2(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    let mut y0 = 0;
    while y0 + 16 <= height {
        rec64_run256::<2>(
            &src[y0 * width..(y0 + 16) * width],
            &dst[y0 * width..(y0 + 16) * width],
            &mut out_mu1[y0 * width..(y0 + 16) * width],
            &mut out_mu2[y0 * width..(y0 + 16) * width],
            &mut out_sigma_sq[y0 * width..(y0 + 16) * width],
            &mut out_sigma12[y0 * width..(y0 + 16) * width],
            width,
            r,
            diam,
            inv_v64,
            err,
        );
        y0 += 16;
    }
    if y0 + 8 <= height {
        rec64_run256::<1>(
            &src[y0 * width..(y0 + 8) * width],
            &dst[y0 * width..(y0 + 8) * width],
            &mut out_mu1[y0 * width..(y0 + 8) * width],
            &mut out_mu2[y0 * width..(y0 + 8) * width],
            &mut out_sigma_sq[y0 * width..(y0 + 8) * width],
            &mut out_sigma12[y0 * width..(y0 + 8) * width],
            width,
            r,
            diam,
            inv_v64,
            err,
        );
        y0 += 8;
    }
    for y in y0..height {
        fused_blur_h_rec64_row(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            y,
            width,
            r,
            diam,
            inv_v64,
            err,
        );
    }
}

/// `#[arcane]` dispatch entry points — the same names the `incant!` site
/// has always called, now backed by the transposed-block bodies. Like the
/// other `_v4x`/`_v4` arcanes they exist on every x86_64 build (the token
/// is unsummonable without `avx512`); `rec64_rows8x8` itself resolves to
/// the `v4` rite under `avx512` and to a `v3` rite delegating to the AVX2
/// driver otherwise — the canon is tier-invariant either way.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_rec64_rows_v4x(
    _token: archmage::X64V4xToken,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    rec64_rows8x8(
        src,
        dst,
        out_mu1,
        out_mu2,
        out_sigma_sq,
        out_sigma12,
        width,
        height,
        r,
        diam,
        inv_v64,
        err,
    );
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_rec64_rows_v4(
    _token: archmage::X64V4Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    rec64_rows8x8(
        src,
        dst,
        out_mu1,
        out_mu2,
        out_sigma_sq,
        out_sigma12,
        width,
        height,
        r,
        diam,
        inv_v64,
        err,
    );
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_rec64_rows_v3(
    _token: archmage::X64V3Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    rec64_rows8x8_avx2(
        src,
        dst,
        out_mu1,
        out_mu2,
        out_sigma_sq,
        out_sigma12,
        width,
        height,
        r,
        diam,
        inv_v64,
        err,
    );
}

/// Scalar-tier siblings of [`fused_blur_h_ssim_rec64_rows`]: scalar and
/// wasm128 keep the scalar Rec64 row body — their `mul_add` is unfused.
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_rec64_rows_scalar(
    _token: archmage::ScalarToken,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    for y in 0..height {
        fused_blur_h_rec64_row(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            y,
            width,
            r,
            diam,
            inv_v64,
            err,
        );
    }
}

/// Wasm128 sibling of [`fused_blur_h_ssim_rec64_rows_scalar`].
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_rec64_rows_wasm128(
    _token: archmage::Wasm128Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    for y in 0..height {
        fused_blur_h_rec64_row(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            y,
            width,
            r,
            diam,
            inv_v64,
            err,
        );
    }
}

/// NEON runs the scalar canonical body (2026-10-03): NEON vector `max`/`min` propagate NaN where the canon's `f32::max`/
/// `f64::max` drop it (the REV4VEC CI aarch64 failure), so only the x86_64 fused tiers run a lane-parallel body.
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_rec64_rows_neon(
    _token: archmage::NeonToken,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    r: usize,
    diam: usize,
    inv_v64: f64,
    err: bool,
) {
    for y in 0..height {
        fused_blur_h_rec64_row(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            y,
            width,
            r,
            diam,
            inv_v64,
            err,
        );
    }
}

// ============================================================================
// Rev5 `localwin` — per-output f32 pair-tree windows (THE REV5 CANON blur)
// ============================================================================

/// The `localwin` window reduce — the Rev5 spec's fixed pair tree over `t`'s
/// taps. The largest power-of-two prefix reduces as a balanced pairwise tree;
/// the remainder chunks fold on left-to-right. For the production 11-tap
/// window this is exactly the spec's `s2 = x[i]+x[i+1]; s4 = s2[i]+s2[i+2];
/// s8 = s4[i]+s4[i+4]; w = s8 + s2[8] + x[10]` — `(s8 + s89) + x10`. Every
/// `localwin` kernel funnels through this one function so SIMD bodies can
/// only diverge by lane, never by order.
#[inline(always)]
pub(crate) fn local_sum_taps(t: &[f32]) -> f32 {
    if t.len() == 11 {
        let s8 = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
        return s8 + (t[8] + t[9]) + t[10];
    }
    let mut rest = t;
    let mut acc = 0.0f32;
    let mut first = true;
    while !rest.is_empty() {
        // Largest power of two ≤ `rest.len()` — for 11 taps this yields the
        // 8 + 2 + 1 chunking the spec tree writes out.
        let sz = 1usize << (usize::BITS - 1 - rest.len().leading_zeros());
        let c = match sz {
            1 => rest[0],
            2 => rest[0] + rest[1],
            4 => (rest[0] + rest[1]) + (rest[2] + rest[3]),
            8 => {
                ((rest[0] + rest[1]) + (rest[2] + rest[3]))
                    + ((rest[4] + rest[5]) + (rest[6] + rest[7]))
            }
            _ => {
                // Chunks ≥ 16 taps never occur at `BLUR_RADIUS = 5`; keep a
                // fixed left fold of pairwise leaves for larger windows.
                let mut s = rest[0] + rest[1];
                let mut i = 2;
                while i + 1 < sz {
                    s += rest[i] + rest[i + 1];
                    i += 2;
                }
                if i < sz {
                    s += rest[i];
                }
                s
            }
        };
        acc = if first { c } else { acc + c };
        first = false;
        rest = &rest[sz..];
    }
    acc
}

/// **Rev5 `localwin` fused H-blur** — [`fused_blur_h_ssim_canon`]'s Rev5 arm:
/// each output's 11-tap window is evaluated independently in f32 through
/// [`local_sum_taps`] — no sliding recurrence, so no rounding can travel
/// along a row. `tap_mirror` padding, identical to every other axis; the
/// per-tap leaves are the same products the Rec64 body feeds its chain
/// (`s·s + d·d` fused, `(s−d)²` or `s·d`). Tier-identical by construction:
/// one op sequence, every tier.
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_local_general(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    let r = radius;
    let diam = 2 * r + 1;
    let inv = 1.0f32 / diam as f32;
    let mut stack = [0.0f32; 4 * H_RING_CAP];
    let mut heap;
    let t = if diam <= H_RING_CAP {
        &mut stack[..4 * diam]
    } else {
        heap = vec![0.0; 4 * diam];
        &mut heap[..]
    };
    let (ts, rest) = t.split_at_mut(diam);
    let (td, rest) = rest.split_at_mut(diam);
    let (tq, tp) = rest.split_at_mut(diam);
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            for k in 0..diam {
                let idx = crate::featcanon::tap_mirror(x as isize + k as isize - r as isize, width);
                let s = src[row + idx];
                let d = dst[row + idx];
                ts[k] = s;
                td[k] = d;
                tq[k] = s.mul_add(s, d * d);
                tp[k] = if err {
                    let e = s - d;
                    e * e
                } else {
                    s * d
                };
            }
            out_mu1[row + x] = local_sum_taps(ts) * inv;
            out_mu2[row + x] = local_sum_taps(td) * inv;
            out_sigma_sq[row + x] = local_sum_taps(tq) * inv;
            out_sigma12[row + x] = local_sum_taps(tp) * inv;
        }
    }
}

/// Rev5 `localwin` H blur — [`box_blur_h`]'s Rev5 arithmetic: per-output
/// pair tree, `tap_mirror` padding. Bit-identical to the `mu1`/`mu2` plane a
/// [`fused_blur_h_ssim_local`] feed of `(input, input)` writes, the parity
/// every activity-chain consumer relies on.
pub(crate) fn box_blur_h_local_general(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    let r = radius;
    let diam = 2 * r + 1;
    let inv = 1.0f32 / diam as f32;
    let mut stack = [0.0f32; H_RING_CAP];
    let mut heap;
    let t = if diam <= H_RING_CAP {
        &mut stack[..diam]
    } else {
        heap = vec![0.0; diam];
        &mut heap[..]
    };
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            for k in 0..diam {
                t[k] = input[row
                    + crate::featcanon::tap_mirror(x as isize + k as isize - r as isize, width)];
            }
            output[row + x] = local_sum_taps(t) * inv;
        }
    }
}

/// Rev5 `localwin` V blur — [`box_blur_v_from_copy`]'s Rev5 arithmetic:
/// per-output pair tree over the `tap_mirror` row set. Only inner rows need
/// evaluating (no recurrence to feed), but this entry — like the shipped
/// copy — writes every row.
pub(crate) fn box_blur_v_local_general(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    let r = radius;
    let diam = 2 * r + 1;
    let inv = 1.0f32 / diam as f32;
    let mut stack = [0.0f32; H_RING_CAP];
    let mut heap;
    let t = if diam <= H_RING_CAP {
        &mut stack[..diam]
    } else {
        heap = vec![0.0; diam];
        &mut heap[..]
    };
    let mut rb_stack = [0usize; H_RING_CAP];
    let mut rb_heap;
    let rb = if diam <= H_RING_CAP {
        &mut rb_stack[..diam]
    } else {
        rb_heap = vec![0; diam];
        &mut rb_heap[..]
    };
    for y in 0..height {
        for k in 0..diam {
            rb[k] =
                crate::featcanon::tap_mirror(y as isize + k as isize - r as isize, height) * width;
        }
        let base = y * width;
        for x in 0..width {
            for k in 0..diam {
                t[k] = src[rb[k] + x];
            }
            dst[base + x] = local_sum_taps(t) * inv;
        }
    }
}

/// Radius-five vector windows use exactly the scalar pair tree. Scalar and
/// non-x86 targets retain correctly rounded inherent `mul_add` leaves.
fn fused_blur_h_ssim_local(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    if radius == 5 {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(token) = <archmage::X64V4xToken as archmage::SimdToken>::summon() {
            fused_blur_h_local11_wide_v4x(
                token,
                src,
                dst,
                out_mu1,
                out_mu2,
                out_sigma_sq,
                out_sigma12,
                width,
                height,
                radius,
                err,
            );
            return;
        }
        incant!(
            fused_blur_h_local11(
                src,
                dst,
                out_mu1,
                out_mu2,
                out_sigma_sq,
                out_sigma12,
                width,
                height,
                radius,
                err
            ),
            [v4x, v4, v3, scalar]
        );
    } else {
        fused_blur_h_ssim_local_general(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            width,
            height,
            radius,
            err,
        );
    }
}
fn fused_blur_h_local11_scalar(
    _token: archmage::ScalarToken,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    debug_assert_eq!(radius, 5);
    if width == 0 || height == 0 {
        return;
    }
    let mut taps = [[0.0f32; 16]; 4];
    for y in 0..height {
        let row = y * width;
        let leaf = |column: isize| {
            let i = row + crate::featcanon::tap_mirror(column, width);
            let s = src[i];
            let d = dst[i];
            let p = if err {
                let e = s - d;
                e * e
            } else {
                s * d
            };
            [s, d, s.mul_add(s, d * d), p]
        };
        for k in 0..11 {
            let values = leaf(k as isize - 5);
            for plane in 0..4 {
                taps[plane][k] = values[plane];
            }
        }
        for x in 0..width {
            out_mu1[row + x] = local_sum_ring11(&taps[0], x) * (1.0 / 11.0);
            out_mu2[row + x] = local_sum_ring11(&taps[1], x) * (1.0 / 11.0);
            out_sigma_sq[row + x] = local_sum_ring11(&taps[2], x) * (1.0 / 11.0);
            out_sigma12[row + x] = local_sum_ring11(&taps[3], x) * (1.0 / 11.0);
            if x + 1 < width {
                let values = leaf(x as isize + 6);
                for plane in 0..4 {
                    taps[plane][(x + 11) & 15] = values[plane];
                }
            }
        }
    }
}

/// Cache leaves, never partial sums: each output still executes the frozen
/// 11-tap pair tree. Power-of-two storage makes the wrap bounds provable.
#[inline(always)]
fn local_sum_ring11(t: &[f32; 16], base: usize) -> f32 {
    let at = |k| t[(base + k) & 15];
    let s8 = ((at(0) + at(1)) + (at(2) + at(3))) + ((at(4) + at(5)) + (at(6) + at(7)));
    s8 + (at(8) + at(9)) + at(10)
}
#[magetypes(define(f32x8), v4x, v4, v3, -scalar)]
fn fused_blur_h_local11(
    token: Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    let _ = radius;
    let inv = f32x8::splat_t(token, 1.0 / 11.0);
    // The two boundary blocks use the same mirrored columns on every row.
    // Gather each distinct leaf once, then load overlapping windows from it.
    let left: [usize; 18] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror(i as isize - 5, width));
    let last = (width / 8).saturating_sub(1) * 8;
    let right: [usize; 18] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror((last + i) as isize - 5, width));
    // One 8-output block of the four planes from its 18 leaves of each input:
    // the frozen tree over taps at constant offsets (one range check per block
    // and input, not one per tap), as `fused_blur_h_local11_wide` does at 16.
    let block = |a_t: &[f32; 18], b_t: &[f32; 18]| {
        let tap = |k: usize| {
            let a = f32x8::load_t(token, a_t[k..k + 8].try_into().unwrap());
            let b = f32x8::load_t(token, b_t[k..k + 8].try_into().unwrap());
            let q = a.mul_add(a, b * b);
            let p = if err {
                let e = a - b;
                e * e
            } else {
                a * b
            };
            (a, b, q, p)
        };
        let add = |a: (f32x8, f32x8, f32x8, f32x8), b: (f32x8, f32x8, f32x8, f32x8)| {
            (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3)
        };
        let pair = |k| add(tap(k), tap(k + 1));
        let s01 = add(pair(0), pair(2));
        let s45 = add(pair(4), pair(6));
        add(add(add(s01, s45), pair(8)), tap(10))
    };
    for y in 0..height {
        let row = y * width;
        let (sr, dr) = (&src[row..row + width], &dst[row..row + width]);
        let o1 = &mut out_mu1[row..row + width];
        let o2 = &mut out_mu2[row..row + width];
        let oq = &mut out_sigma_sq[row..row + width];
        let op = &mut out_sigma12[row..row + width];
        let mut x = 0;
        while x + 8 <= width {
            let sum = if x >= 5 && x + 8 + 5 <= width {
                block(
                    sr[x - 5..x + 13].try_into().unwrap(),
                    dr[x - 5..x + 13].try_into().unwrap(),
                )
            } else {
                let indices = if x == 0 { &left } else { &right };
                block(&indices.map(|i| sr[i]), &indices.map(|i| dr[i]))
            };
            (sum.0 * inv).store((&mut o1[x..x + 8]).try_into().unwrap());
            (sum.1 * inv).store((&mut o2[x..x + 8]).try_into().unwrap());
            (sum.2 * inv).store((&mut oq[x..x + 8]).try_into().unwrap());
            (sum.3 * inv).store((&mut op[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            let mut a = [0.0; 11];
            let mut b = a;
            let mut q = a;
            let mut p = a;
            for k in 0..11 {
                let i = crate::featcanon::tap_mirror(x as isize + k as isize - 5, width);
                a[k] = sr[i];
                b[k] = dr[i];
                q[k] = a[k].mul_add(a[k], b[k] * b[k]);
                p[k] = if err {
                    let e = a[k] - b[k];
                    e * e
                } else {
                    a[k] * b[k]
                };
            }
            o1[x] = local_sum_taps(&a) * (1.0 / 11.0);
            o2[x] = local_sum_taps(&b) * (1.0 / 11.0);
            oq[x] = local_sum_taps(&q) * (1.0 / 11.0);
            op[x] = local_sum_taps(&p) * (1.0 / 11.0);
            x += 1;
        }
    }
}

/// [`fused_blur_h_local11`] and then [`box_blur_v_local11`] on each of its
/// four planes, a 256-column panel at a time through `hbuf` (four panel-wide
/// planes, row stride = the panel's span), as [`fused_blur_hv_local11_wide`]
/// does at sixteen lanes. Every output keeps both kernels' trees, operands and
/// roundings.
#[cfg(feature = "threads")]
#[magetypes(define(f32x8), v4x, v4, v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fused_blur_hv_local11(
    token: Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    err: bool,
    hbuf: [&mut [f32]; 4],
) {
    if width == 0 || height == 0 {
        return;
    }
    let inv = f32x8::splat_t(token, 1.0 / 11.0);
    let left: [usize; 18] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror(i as isize - 5, width));
    let last = (width / 8).saturating_sub(1) * 8;
    let right: [usize; 18] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror((last + i) as isize - 5, width));
    // `fused_blur_h_local11`'s 8-output block.
    let block = |a_t: &[f32; 18], b_t: &[f32; 18]| {
        let tap = |k: usize| {
            let a = f32x8::load_t(token, a_t[k..k + 8].try_into().unwrap());
            let b = f32x8::load_t(token, b_t[k..k + 8].try_into().unwrap());
            let q = a.mul_add(a, b * b);
            let p = if err {
                let e = a - b;
                e * e
            } else {
                a * b
            };
            (a, b, q, p)
        };
        let add = |a: (f32x8, f32x8, f32x8, f32x8), b: (f32x8, f32x8, f32x8, f32x8)| {
            (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3)
        };
        let pair = |k| add(tap(k), tap(k + 1));
        let s01 = add(pair(0), pair(2));
        let s45 = add(pair(4), pair(6));
        add(add(add(s01, s45), pair(8)), tap(10))
    };
    let [h1, h2, hq, hp] = hbuf;
    let mut outs = [out_mu1, out_mu2, out_sigma_sq, out_sigma12];
    const PANEL: usize = 32;
    let blocks = width / 8;
    let mut pairs = [[f32x8::zero_t(token); PANEL]; 4];
    let mut quads = [[f32x8::zero_t(token); PANEL]; 8];
    for tile in (0..blocks).step_by(PANEL) {
        let count = (blocks - tile).min(PANEL);
        let (x0, span) = (tile * 8, count * 8);
        for y in 0..height {
            let row = y * width;
            let (sr, dr) = (&src[row..row + width], &dst[row..row + width]);
            let o = y * span;
            let r1 = &mut h1[o..o + span].as_chunks_mut::<8>().0[..count];
            let r2 = &mut h2[o..o + span].as_chunks_mut::<8>().0[..count];
            let rq = &mut hq[o..o + span].as_chunks_mut::<8>().0[..count];
            let rp = &mut hp[o..o + span].as_chunks_mut::<8>().0[..count];
            for c in 0..count {
                let x = x0 + c * 8;
                let sum = if x >= 5 && x + 8 + 5 <= width {
                    block(
                        sr[x - 5..x + 13].try_into().unwrap(),
                        dr[x - 5..x + 13].try_into().unwrap(),
                    )
                } else {
                    let indices = if x == 0 { &left } else { &right };
                    block(&indices.map(|i| sr[i]), &indices.map(|i| dr[i]))
                };
                (sum.0 * inv).store(&mut r1[c]);
                (sum.1 * inv).store(&mut r2[c]);
                (sum.2 * inv).store(&mut rq[c]);
                (sum.3 * inv).store(&mut rp[c]);
            }
        }
        for (input, output) in [&*h1, &*h2, &*hq, &*hp].into_iter().zip(outs.iter_mut()) {
            for k in 0..9 {
                let a = crate::featcanon::tap_mirror(k as isize - 5, height) * span;
                let b = crate::featcanon::tap_mirror(k as isize - 4, height) * span;
                let ra = &input[a..a + span].as_chunks::<8>().0[..count];
                let rb = &input[b..b + span].as_chunks::<8>().0[..count];
                for c in 0..count {
                    pairs[k & 3][c] = f32x8::load_t(token, &ra[c]) + f32x8::load_t(token, &rb[c]);
                }
                if k >= 2 {
                    for c in 0..count {
                        quads[k - 2][c] = pairs[(k - 2) & 3][c] + pairs[k & 3][c];
                    }
                }
            }
            for y in 0..height {
                let a = crate::featcanon::tap_mirror(y as isize + 4, height) * span;
                let b = crate::featcanon::tap_mirror(y as isize + 5, height) * span;
                let ra = &input[a..a + span].as_chunks::<8>().0[..count];
                let rb = &input[b..b + span].as_chunks::<8>().0[..count];
                let o = y * width + x0;
                let ro = &mut output[o..o + span].as_chunks_mut::<8>().0[..count];
                let (q0, q4, q7) = (y & 7, (y + 4) & 7, (y + 7) & 7);
                let (p7, p8, p9) = ((y + 7) & 3, (y + 8) & 3, (y + 9) & 3);
                for c in 0..count {
                    let last = f32x8::load_t(token, &rb[c]);
                    pairs[p9][c] = f32x8::load_t(token, &ra[c]) + last;
                    quads[q7][c] = pairs[p7][c] + pairs[p9][c];
                    let sum = ((quads[q0][c] + quads[q4][c]) + pairs[p8][c]) + last;
                    (sum * inv).store(&mut ro[c]);
                }
            }
        }
    }
    // The incomplete vector: both kernels' per-output scalar trees, the
    // horizontal one per row into `hbuf` (row stride = the tail's width).
    let xt = blocks * 8;
    let tw = width - xt;
    if tw == 0 {
        return;
    }
    for y in 0..height {
        let row = y * width;
        let (sr, dr) = (&src[row..row + width], &dst[row..row + width]);
        for x in xt..width {
            let mut a = [0.0; 11];
            let mut b = a;
            let mut q = a;
            let mut p = a;
            for k in 0..11 {
                let i = crate::featcanon::tap_mirror(x as isize + k as isize - 5, width);
                a[k] = sr[i];
                b[k] = dr[i];
                q[k] = a[k].mul_add(a[k], b[k] * b[k]);
                p[k] = if err {
                    let e = a[k] - b[k];
                    e * e
                } else {
                    a[k] * b[k]
                };
            }
            let o = y * tw + (x - xt);
            h1[o] = local_sum_taps(&a) * (1.0 / 11.0);
            h2[o] = local_sum_taps(&b) * (1.0 / 11.0);
            hq[o] = local_sum_taps(&q) * (1.0 / 11.0);
            hp[o] = local_sum_taps(&p) * (1.0 / 11.0);
        }
    }
    for (input, output) in [&*h1, &*h2, &*hq, &*hp].into_iter().zip(outs.iter_mut()) {
        for y in 0..height {
            let rows: [usize; 11] = std::array::from_fn(|k| {
                crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * tw
            });
            for x in 0..tw {
                let taps: [f32; 11] = std::array::from_fn(|k| input[rows[k] + x]);
                output[y * width + xt + x] = local_sum_taps(&taps) * (1.0 / 11.0);
            }
        }
    }
}

/// The scalar tier's form of [`fused_blur_hv_local11`]: the two scalar kernels
/// in turn, `hbuf` holding the four image-wide horizontal planes.
#[cfg(feature = "threads")]
#[allow(clippy::too_many_arguments)]
fn fused_blur_hv_local11_scalar(
    token: archmage::ScalarToken,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    err: bool,
    hbuf: [&mut [f32]; 4],
) {
    let [h1, h2, hq, hp] = hbuf;
    fused_blur_h_local11_scalar(token, src, dst, h1, h2, hq, hp, width, height, 5, err);
    for (input, output) in
        [&*h1, &*h2, &*hq, &*hp]
            .into_iter()
            .zip([out_mu1, out_mu2, out_sigma_sq, out_sigma12])
    {
        box_blur_v_local11_scalar(token, input, output, width, height, 5);
    }
}

/// The Rev5 `localwin` moment planes of one window in one pass: what
/// [`fused_blur_h_ssim_at_revision`] and then [`box_blur_v_from_copy_at_revision`]
/// on each of its four planes compute, bit for bit, without writing four
/// image-wide horizontal planes. `hbuf` (each `width × height`) holds the
/// panels' horizontal rows. Returns false, computing nothing, where either
/// kernel would not run its per-output (`Local`) tree.
#[cfg(feature = "threads")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_blur_hv_ssim_at_revision(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    revision: crate::feature_defs::FormulaRevision,
    hbuf: [&mut [f32]; 4],
) -> bool {
    use crate::featcanon::BlurMode;
    // The horizontal kernel's selection (`fused_blur_h_ssim_at_revision`).
    let revision_h = crate::ssim_form::effective_revision(revision);
    let mode_h = crate::featcanon::mode(revision_h);
    let canon_h = mode_h.active();
    let local = |mode, revision| {
        matches!(
            crate::featcanon::canon_blur_axis_for(mode, revision),
            BlurMode::Local
        )
    };
    // The vertical kernel's (`box_blur_v_from_copy_at_revision`).
    let local_v = local(crate::featcanon::mode(revision), revision);
    if radius != 5 || !canon_h || !local(mode_h, revision_h) || !local_v {
        return false;
    }
    let err = revision_h >= crate::feature_defs::FormulaRevision::Rev5;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(token) = <archmage::X64V4xToken as archmage::SimdToken>::summon() {
        fused_blur_hv_local11_wide_v4x(
            token,
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            width,
            height,
            err,
            hbuf,
        );
        return true;
    }
    incant!(
        fused_blur_hv_local11(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            width,
            height,
            err,
            hbuf
        ),
        [v4x, v4, v3, scalar]
    );
    true
}

/// [`activity_local11_wide`] at eight lanes: `|src − mu|` through
/// [`box_blur_h_local11`] then [`box_blur_v_local11`], a 256-column panel at a
/// time through `row` and `hbuf`.
#[cfg(feature = "threads")]
#[magetypes(define(f32x8), v4x, v4, v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn activity_local11(
    token: Token,
    src: &[f32],
    mu: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    hbuf: &mut [f32],
    row: &mut [f32],
) {
    if width == 0 || height == 0 {
        return;
    }
    let inv = f32x8::splat_t(token, 1.0 / 11.0);
    let left: [usize; 18] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror(i as isize - 5, width));
    let last = (width / 8).saturating_sub(1) * 8;
    let right: [usize; 18] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror((last + i) as isize - 5, width));
    // `box_blur_h_local11`'s 8-output block.
    let block = |t: &[f32; 18]| {
        let l = |k: usize| f32x8::load_t(token, t[k..k + 8].try_into().unwrap());
        (((l(0) + l(1)) + (l(2) + l(3))) + ((l(4) + l(5)) + (l(6) + l(7)))) + (l(8) + l(9)) + l(10)
    };
    const PANEL: usize = 32;
    let blocks = width / 8;
    let mut pairs = [[f32x8::zero_t(token); PANEL]; 4];
    let mut quads = [[f32x8::zero_t(token); PANEL]; 8];
    for tile in (0..blocks).step_by(PANEL) {
        let count = (blocks - tile).min(PANEL);
        let (x0, span) = (tile * 8, count * 8);
        let lo = x0.saturating_sub(5);
        let hi = (x0 + span + 5).min(width);
        for y in 0..height {
            let r = y * width;
            let (s, m) = (&src[r + lo..r + hi], &mu[r + lo..r + hi]);
            let d = &mut row[..hi - lo];
            let full = (hi - lo) / 8;
            for ((dc, sc), mc) in d
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(s.as_chunks::<8>().0)
                .zip(m.as_chunks::<8>().0)
            {
                (f32x8::load_t(token, sc) - f32x8::load_t(token, mc))
                    .abs()
                    .store(dc);
            }
            for i in full * 8..hi - lo {
                d[i] = (s[i] - m[i]).abs();
            }
            let o = y * span;
            let rh = &mut hbuf[o..o + span].as_chunks_mut::<8>().0[..count];
            for c in 0..count {
                let x = x0 + c * 8;
                let sum = if x >= 5 && x + 8 + 5 <= width {
                    block(d[x - 5 - lo..x + 13 - lo].try_into().unwrap())
                } else {
                    let indices = if x == 0 { &left } else { &right };
                    block(&indices.map(|i| d[i - lo]))
                };
                (sum * inv).store(&mut rh[c]);
            }
        }
        let input = &*hbuf;
        for k in 0..9 {
            let a = crate::featcanon::tap_mirror(k as isize - 5, height) * span;
            let b = crate::featcanon::tap_mirror(k as isize - 4, height) * span;
            let ra = &input[a..a + span].as_chunks::<8>().0[..count];
            let rb = &input[b..b + span].as_chunks::<8>().0[..count];
            for c in 0..count {
                pairs[k & 3][c] = f32x8::load_t(token, &ra[c]) + f32x8::load_t(token, &rb[c]);
            }
            if k >= 2 {
                for c in 0..count {
                    quads[k - 2][c] = pairs[(k - 2) & 3][c] + pairs[k & 3][c];
                }
            }
        }
        for y in 0..height {
            let a = crate::featcanon::tap_mirror(y as isize + 4, height) * span;
            let b = crate::featcanon::tap_mirror(y as isize + 5, height) * span;
            let ra = &input[a..a + span].as_chunks::<8>().0[..count];
            let rb = &input[b..b + span].as_chunks::<8>().0[..count];
            let o = y * width + x0;
            let ro = &mut output[o..o + span].as_chunks_mut::<8>().0[..count];
            let (q0, q4, q7) = (y & 7, (y + 4) & 7, (y + 7) & 7);
            let (p7, p8, p9) = ((y + 7) & 3, (y + 8) & 3, (y + 9) & 3);
            for c in 0..count {
                let last = f32x8::load_t(token, &rb[c]);
                pairs[p9][c] = f32x8::load_t(token, &ra[c]) + last;
                quads[q7][c] = pairs[p7][c] + pairs[p9][c];
                let sum = ((quads[q0][c] + quads[q4][c]) + pairs[p8][c]) + last;
                (sum * inv).store(&mut ro[c]);
            }
        }
    }
    let xt = blocks * 8;
    let tw = width - xt;
    if tw == 0 {
        return;
    }
    for y in 0..height {
        let r = y * width;
        for x in xt..width {
            let t: [f32; 11] = std::array::from_fn(|k| {
                let i = r + crate::featcanon::tap_mirror(x as isize + k as isize - 5, width);
                (src[i] - mu[i]).abs()
            });
            hbuf[y * tw + (x - xt)] = local_sum_taps(&t) * (1.0 / 11.0);
        }
    }
    for y in 0..height {
        let rows: [usize; 11] = std::array::from_fn(|k| {
            crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * tw
        });
        for x in 0..tw {
            let taps: [f32; 11] = std::array::from_fn(|k| hbuf[rows[k] + x]);
            output[y * width + xt + x] = local_sum_taps(&taps) * (1.0 / 11.0);
        }
    }
}

/// The scalar tier's form of [`activity_local11`]: the three kernels in turn,
/// `hbuf` and `row` as image-wide planes.
#[cfg(feature = "threads")]
#[allow(clippy::too_many_arguments)]
fn activity_local11_scalar(
    token: archmage::ScalarToken,
    src: &[f32],
    mu: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    hbuf: &mut [f32],
    row: &mut [f32],
) {
    let n = width * height;
    crate::simd_ops::abs_diff_into(&src[..n], &mu[..n], &mut row[..n]);
    box_blur_h_local11_scalar(token, &row[..n], &mut hbuf[..n], width, height, 5);
    box_blur_v_local11_scalar(token, &hbuf[..n], &mut output[..n], width, height, 5);
}

/// The activity plane `box_blur(|src − mu|)` of one window in one pass: what
/// `abs_diff_into` and then [`box_blur_1pass_at_revision`] compute, bit for
/// bit, without the two image-wide intermediates (`hbuf` and `row`, each
/// `width × height`, hold a panel's rows). Returns false, computing nothing,
/// where the blurs would not run their per-output (`Local`) trees.
#[cfg(feature = "threads")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn activity_at_revision(
    src: &[f32],
    mu: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    revision: crate::feature_defs::FormulaRevision,
    hbuf: &mut [f32],
    row: &mut [f32],
) -> bool {
    let local = matches!(
        crate::featcanon::canon_blur_axis_for(crate::featcanon::mode(revision), revision),
        crate::featcanon::BlurMode::Local
    );
    if radius != 5 || !local {
        return false;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(token) = <archmage::X64V4xToken as archmage::SimdToken>::summon() {
        activity_local11_wide_v4x(token, src, mu, output, width, height, hbuf, row);
        return true;
    }
    incant!(
        activity_local11(src, mu, output, width, height, hbuf, row),
        [v4x, v4, v3, scalar]
    );
    true
}

pub(crate) fn box_blur_h_local(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    if radius == 5 {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(token) = <archmage::X64V4xToken as archmage::SimdToken>::summon() {
            box_blur_h_local11_wide_v4x(token, input, output, width, height, radius);
            return;
        }
        incant!(
            box_blur_h_local11(input, output, width, height, radius),
            [v4x, v4, v3, scalar]
        );
    } else {
        box_blur_h_local_general(input, output, width, height, radius);
    }
}
fn box_blur_h_local11_scalar(
    _token: archmage::ScalarToken,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    debug_assert_eq!(radius, 5);
    if width == 0 || height == 0 {
        return;
    }
    let mut taps = [0.0f32; 16];
    for y in 0..height {
        let row = y * width;
        for k in 0..11 {
            taps[k] = input[row + crate::featcanon::tap_mirror(k as isize - 5, width)];
        }
        for x in 0..width {
            output[row + x] = local_sum_ring11(&taps, x) * (1.0 / 11.0);
            if x + 1 < width {
                taps[(x + 11) & 15] =
                    input[row + crate::featcanon::tap_mirror(x as isize + 6, width)];
            }
        }
    }
}
#[magetypes(define(f32x8), v4x, v4, v3, -scalar)]
fn box_blur_h_local11(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let _ = radius;
    let inv = f32x8::splat_t(token, 1.0 / 11.0);
    let left: [usize; 18] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror(i as isize - 5, width));
    let last = (width / 8).saturating_sub(1) * 8;
    let right: [usize; 18] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror((last + i) as isize - 5, width));
    // One 8-output block from its 18 leaves: the frozen tree over taps at
    // constant offsets (one range check per block, not one per tap).
    let block = |t: &[f32; 18]| {
        let l = |k: usize| f32x8::load_t(token, t[k..k + 8].try_into().unwrap());
        (((l(0) + l(1)) + (l(2) + l(3))) + ((l(4) + l(5)) + (l(6) + l(7)))) + (l(8) + l(9)) + l(10)
    };
    for y in 0..height {
        let rin = &input[y * width..(y + 1) * width];
        let rout = &mut output[y * width..(y + 1) * width];
        let mut x = 0;
        while x + 8 <= width {
            let sum = if x >= 5 && x + 8 + 5 <= width {
                block(rin[x - 5..x + 13].try_into().unwrap())
            } else {
                let indices = if x == 0 { &left } else { &right };
                block(&indices.map(|i| rin[i]))
            };
            (sum * inv).store((&mut rout[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            let t: [f32; 11] = std::array::from_fn(|k| {
                rin[crate::featcanon::tap_mirror(x as isize + k as isize - 5, width)]
            });
            rout[x] = local_sum_taps(&t) * (1.0 / 11.0);
            x += 1;
        }
    }
}

pub(crate) fn box_blur_v_local(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    if radius == 5 {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(token) = <archmage::X64V4xToken as archmage::SimdToken>::summon() {
            box_blur_v_local11_wide_v4x(token, input, output, width, height, radius);
            return;
        }
        incant!(
            box_blur_v_local11(input, output, width, height, radius),
            [v4x, v4, v3, scalar]
        );
    } else {
        box_blur_v_local_general(input, output, width, height, radius);
    }
}
fn box_blur_v_local11_scalar(
    _token: archmage::ScalarToken,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    debug_assert_eq!(radius, 5);
    if width == 0 || height == 0 {
        return;
    }
    for y in 0..height {
        let rows: [&[f32]; 11] = std::array::from_fn(|k| {
            let start = crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * width;
            &input[start..start + width]
        });
        for x in 0..width {
            let taps: [f32; 11] = std::array::from_fn(|k| rows[k][x]);
            output[y * width + x] = local_sum_taps(&taps) * (1.0 / 11.0);
        }
    }
}
#[magetypes(define(f32x8), v4x, v4, v3, -scalar)]
fn box_blur_v_local11(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    debug_assert_eq!(radius, 5);
    if width == 0 || height == 0 {
        return;
    }
    let inv = f32x8::splat_t(token, 1.0 / 11.0);
    // Cache the frozen tree's pair and quadruple nodes, never an output
    // recurrence. Each node has the same two operands and rounding as before.
    // A panel of 2 KiB row segments walks the whole height, so every row's
    // reads and writes are contiguous runs the stream prefetchers follow (the
    // former 512-byte tiles moved one row stride per few cache lines). A pair
    // node lives three rows and a quadruple node eight, so rings of four and
    // eight segments hold every live node.
    const PANEL: usize = 64;
    let blocks = width / 8;
    let mut pairs = [[f32x8::zero_t(token); PANEL]; 4];
    let mut quads = [[f32x8::zero_t(token); PANEL]; 8];
    for tile in (0..blocks).step_by(PANEL) {
        let count = (blocks - tile).min(PANEL);
        // The panel's span of a tap row and of the output row, cut to exact
        // length once per row, so the column loops have no range checks.
        let (x0, span) = (tile * 8, count * 8);
        for k in 0..9 {
            let a = crate::featcanon::tap_mirror(k as isize - 5, height) * width;
            let b = crate::featcanon::tap_mirror(k as isize - 4, height) * width;
            let ra = &input[a + x0..a + x0 + span].as_chunks::<8>().0[..count];
            let rb = &input[b + x0..b + x0 + span].as_chunks::<8>().0[..count];
            for c in 0..count {
                pairs[k & 3][c] = f32x8::load_t(token, &ra[c]) + f32x8::load_t(token, &rb[c]);
            }
            if k >= 2 {
                for c in 0..count {
                    quads[k - 2][c] = pairs[(k - 2) & 3][c] + pairs[k & 3][c];
                }
            }
        }
        for y in 0..height {
            let a = crate::featcanon::tap_mirror(y as isize + 4, height) * width;
            let b = crate::featcanon::tap_mirror(y as isize + 5, height) * width;
            let ra = &input[a + x0..a + x0 + span].as_chunks::<8>().0[..count];
            let rb = &input[b + x0..b + x0 + span].as_chunks::<8>().0[..count];
            let o = y * width + x0;
            let ro = &mut output[o..o + span].as_chunks_mut::<8>().0[..count];
            let (q0, q4, q7) = (y & 7, (y + 4) & 7, (y + 7) & 7);
            let (p7, p8, p9) = ((y + 7) & 3, (y + 8) & 3, (y + 9) & 3);
            for c in 0..count {
                let last = f32x8::load_t(token, &rb[c]);
                pairs[p9][c] = f32x8::load_t(token, &ra[c]) + last;
                quads[q7][c] = pairs[p7][c] + pairs[p9][c];
                let sum = ((quads[q0][c] + quads[q4][c]) + pairs[p8][c]) + last;
                (sum * inv).store(&mut ro[c]);
            }
        }
    }
    // The incomplete vector uses the unchanged per-output scalar tree.
    if blocks * 8 == width {
        return;
    }
    for y in 0..height {
        let rows: [usize; 11] = std::array::from_fn(|k| {
            crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * width
        });
        for x in blocks * 8..width {
            let taps: [f32; 11] = std::array::from_fn(|k| input[rows[k] + x]);
            output[y * width + x] = local_sum_taps(&taps) * (1.0 / 11.0);
        }
    }
}

/// **FEATCANON canonical fused H-blur** — the same sliding-moment arithmetic
/// as [`fused_blur_h_ssim_inner`] written as one plain-Rust body over inherent
/// `f32` ops. Every product-sum keeps the vector body's fused chain
/// (`f32::mul_add` is fused on every target), every sum keeps the same order,
/// so the outputs equal the v3 (AVX2) tier's bit-for-bit on every tier —
/// including scalar/wasm128, where the magetypes backend `mul_add` is unfused
/// and produced different planes.
///
/// With the `oracle` feature and `ZENSIM_FEATCANON=exact`, the same sliding
/// structure runs in f64 (fused `f64::mul_add`) and rounds once at the f32
/// plane store — the exact arm of the accuracy oracle.
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_canon(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
    mode: crate::featcanon::Mode,
    revision: crate::feature_defs::FormulaRevision,
) {
    let diam = 2 * radius + 1;
    let inv_v = 1.0f32 / diam as f32;
    let inv_v64 = 1.0f64 / diam as f64;
    let r = radius;
    // rev4canon: the blur axis owns this kernel's arithmetic entirely — the
    // window sums ARE its elements. `canon_blur_axis_for(mode, revision)`
    // selects it: `Local` (the per-output f32 pair tree) is THE REV5 CANON —
    // it runs in product builds at Rev5; `Rec64` (the f64 sliding form) is
    // THE REV4 CANON; `Rec` is the shipped f32 sliding form, replayed for the
    // `c32` oracle arm (the superseded canon) and explicit `BLUR=rec` runs;
    // `Fresh` re-sums every window in f64 (oracle ruler). `mode` routes the
    // default per mode (an `exact` run with `BLUR=rec` still measures
    // exactly the shipped blur's contribution).
    let blur = crate::featcanon::canon_blur_axis_for(mode, revision);

    // Rev5 `localwin`: the per-output f32 pair tree — no recurrence at all.
    // Scalar on purpose for now; every tier runs this same body so the planes
    // are tier-identical by construction (the vector body arrives with the
    // speed loop and must match it bit-for-bit).
    if matches!(blur, crate::featcanon::BlurMode::Local) {
        fused_blur_h_ssim_local(
            src,
            dst,
            out_mu1,
            out_mu2,
            out_sigma_sq,
            out_sigma12,
            width,
            height,
            radius,
            err,
        );
        return;
    }

    // rev4vec2: Rec64 → the 8-row f64x8 body on the fused-FMA tiers
    // (bit-identical per-row op sequence); scalar + wasm128 run the same
    // code as scalar rows. Other (oracle) axes stay in the loop below.
    if matches!(blur, crate::featcanon::BlurMode::Rec64) {
        incant!(
            fused_blur_h_ssim_rec64_rows(
                src,
                dst,
                out_mu1,
                out_mu2,
                out_sigma_sq,
                out_sigma12,
                width,
                height,
                r,
                diam,
                inv_v64,
                err,
            ),
            [v4x, v4, v3, neon, wasm128, scalar]
        );
        return;
    }

    for y in 0..height {
        let row = y * width;

        let mut ss = 0.0f32;
        let mut sd = 0.0f32;
        let mut ssq = 0.0f32;
        let mut sprod = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let s = src[row + idx];
            let d = dst[row + idx];
            ss += s;
            sd += d;
            ssq = s.mul_add(s, d.mul_add(d, ssq));
            sprod = if err {
                let e = s - d;
                e.mul_add(e, sprod)
            } else {
                s.mul_add(d, sprod)
            };
        }
        for x in 0..width {
            out_mu1[row + x] = ss * inv_v;
            out_mu2[row + x] = sd * inv_v;
            out_sigma_sq[row + x] = ssq * inv_v;
            out_sigma12[row + x] = sprod * inv_v;

            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = (if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            })
            .min(width - 1);
            let sa = src[row + add_idx];
            let da = dst[row + add_idx];
            let sr = src[row + rem_idx];
            let dr = dst[row + rem_idx];
            ss = ss + sa - sr;
            sd = sd + da - dr;
            ssq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, ssq))),
            );
            sprod = if err {
                let ea = sa - da;
                let er = sr - dr;
                ea.mul_add(ea, (-er).mul_add(er, sprod))
            } else {
                sa.mul_add(da, (-sr).mul_add(dr, sprod))
            };
        }
    }
}

/// Downscale in-place: writes to beginning of buffer, truncates.
/// Safe because output index < source index for all elements.
///
/// In-place proof: output index y*new_w+x < source index (2y)*width+(2x) for all y,x.
/// Because 2y*width + 2x = 2*(y*width + x) >= 2*(y*new_w + x) > y*new_w + x.
pub fn downscale_2x_inplace(plane: &mut Vec<f32>, width: usize, height: usize) -> (usize, usize) {
    let new_w = width / 2;
    let new_h = height / 2;
    downscale_2x(plane, width, new_w, new_h);
    plane.truncate(new_w * new_h);
    (new_w, new_h)
}

fn downscale_2x(plane: &mut [f32], width: usize, new_w: usize, new_h: usize) {
    incant!(
        downscale_2x_inner(plane, width, new_w, new_h),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// Out-of-place 2× downscale: read from `src` (`src_w × src_h`), write to
/// `dst` (`new_w × new_h` where `new_w = src_w / 2`, `new_h = src_h / 2`).
///
/// Compared to [`downscale_2x_inplace`] this avoids reading and writing the
/// same buffer — useful when callers want to keep the source data alive
/// (e.g. multi-scale pyramid construction with all levels owned).
pub fn downscale_2x_into(src: &[f32], src_w: usize, dst: &mut [f32], new_w: usize, new_h: usize) {
    // aarch64: use the SCALAR variant. On AArch64, NEON is baseline, so LLVM
    // autovectorises the scalar body anyway — the hand-written NEON variant is
    // competing with the autovectoriser, not with scalar code, and it LOSES.
    // Measured on Apple M4 Pro at 1024x1024 (zensim-bench/benches/stage_isolation.rs,
    // within-group A/B): NEON 0.149ms vs scalar 0.099ms — the NEON variant is 1.5x SLOWER.
    // Verified BIT-IDENTICAL to the NEON variant before switching (differing
    // lanes = 0), so this is a pure speed change with no effect on any score.
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if archmage::NeonToken::summon().is_some() {
            return downscale_2x_into_inner_scalar(
                archmage::ScalarToken::summon().expect("infallible"),
                src,
                src_w,
                dst,
                new_w,
                new_h,
            );
        }
    }
    incant!(
        downscale_2x_into_inner(src, src_w, dst, new_w, new_h),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// AVX-512 downscale: process 16 output pixels per iteration.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn downscale_2x_inner_v4(
    token: archmage::X64V4Token,
    plane: &mut [f32],
    width: usize,
    new_w: usize,
    new_h: usize,
) {
    let quarter = f32x16::splat_t(token, 0.25);
    let v3 = token.v3();
    let quarter8 = f32x8::splat_t(v3, 0.25);

    for y in 0..new_h {
        let row0 = y * 2 * width;
        let row1 = row0 + width;
        let out_row = y * new_w;

        let chunks16 = new_w / 16;
        for chunk in 0..chunks16 {
            let ox = chunk * 16;
            let sx = ox * 2;
            let mut arr = [0.0f32; 16];
            for i in 0..16 {
                let s = sx + i * 2;
                arr[i] =
                    plane[row0 + s] + plane[row0 + s + 1] + plane[row1 + s] + plane[row1 + s + 1];
            }
            let result = f32x16::from_array_t(token, arr) * quarter;
            plane[out_row + ox..][..16].copy_from_slice(&result.to_array());
        }

        let base8 = chunks16 * 16;
        let chunks8 = (new_w - base8) / 8;
        for chunk in 0..chunks8 {
            let ox = base8 + chunk * 8;
            let sx = ox * 2;
            let mut arr = [0.0f32; 8];
            for i in 0..8 {
                let s = sx + i * 2;
                arr[i] =
                    plane[row0 + s] + plane[row0 + s + 1] + plane[row1 + s] + plane[row1 + s + 1];
            }
            let result = f32x8::from_array_t(v3, arr) * quarter8;
            plane[out_row + ox..][..8].copy_from_slice(&result.to_array());
        }

        for x in (base8 + chunks8 * 8)..new_w {
            let sx = x * 2;
            plane[out_row + x] =
                (plane[row0 + sx] + plane[row0 + sx + 1] + plane[row1 + sx] + plane[row1 + sx + 1])
                    * 0.25;
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn downscale_2x_inner_v4x(
    token: archmage::X64V4xToken,
    plane: &mut [f32],
    width: usize,
    new_w: usize,
    new_h: usize,
) {
    let quarter = f32x16::splat_t(token, 0.25);
    let v3 = token.v3();
    let quarter8 = f32x8::splat_t(v3, 0.25);

    for y in 0..new_h {
        let row0 = y * 2 * width;
        let row1 = row0 + width;
        let out_row = y * new_w;

        let chunks16 = new_w / 16;
        for chunk in 0..chunks16 {
            let ox = chunk * 16;
            let sx = ox * 2;
            let mut arr = [0.0f32; 16];
            for i in 0..16 {
                let s = sx + i * 2;
                arr[i] =
                    plane[row0 + s] + plane[row0 + s + 1] + plane[row1 + s] + plane[row1 + s + 1];
            }
            let result = f32x16::from_array_t(token, arr) * quarter;
            plane[out_row + ox..][..16].copy_from_slice(&result.to_array());
        }

        let base8 = chunks16 * 16;
        let chunks8 = (new_w - base8) / 8;
        for chunk in 0..chunks8 {
            let ox = base8 + chunk * 8;
            let sx = ox * 2;
            let mut arr = [0.0f32; 8];
            for i in 0..8 {
                let s = sx + i * 2;
                arr[i] =
                    plane[row0 + s] + plane[row0 + s + 1] + plane[row1 + s] + plane[row1 + s + 1];
            }
            let result = f32x8::from_array_t(v3, arr) * quarter8;
            plane[out_row + ox..][..8].copy_from_slice(&result.to_array());
        }

        for x in (base8 + chunks8 * 8)..new_w {
            let sx = x * 2;
            plane[out_row + x] =
                (plane[row0 + sx] + plane[row0 + sx + 1] + plane[row1 + sx] + plane[row1 + sx + 1])
                    * 0.25;
        }
    }
}

/// AVX2 downscale: process 8 output pixels per iteration.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn downscale_2x_inner_v3(
    token: archmage::X64V3Token,
    plane: &mut [f32],
    width: usize,
    new_w: usize,
    new_h: usize,
) {
    let quarter = f32x8::splat_t(token, 0.25);

    for y in 0..new_h {
        let row0 = y * 2 * width;
        let row1 = row0 + width;
        let out_row = y * new_w;

        let chunks8 = new_w / 8;
        for chunk in 0..chunks8 {
            let ox = chunk * 8;
            let sx = ox * 2;
            let mut arr = [0.0f32; 8];
            for i in 0..8 {
                let s = sx + i * 2;
                arr[i] =
                    plane[row0 + s] + plane[row0 + s + 1] + plane[row1 + s] + plane[row1 + s + 1];
            }
            let result = f32x8::from_array_t(token, arr) * quarter;
            plane[out_row + ox..][..8].copy_from_slice(&result.to_array());
        }

        for x in (chunks8 * 8)..new_w {
            let sx = x * 2;
            plane[out_row + x] =
                (plane[row0 + sx] + plane[row0 + sx + 1] + plane[row1 + sx] + plane[row1 + sx + 1])
                    * 0.25;
        }
    }
}

#[magetypes(neon, wasm128, scalar)]
fn downscale_2x_inner(token: Token, plane: &mut [f32], width: usize, new_w: usize, new_h: usize) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let quarter = f32x8::splat_t(token, 0.25);

    for y in 0..new_h {
        let row0 = y * 2 * width;
        let row1 = row0 + width;
        let out_row = y * new_w;

        let chunks8 = new_w / 8;
        for chunk in 0..chunks8 {
            let ox = chunk * 8;
            let sx = ox * 2;
            let mut arr = [0.0f32; 8];
            for i in 0..8 {
                let s = sx + i * 2;
                arr[i] =
                    plane[row0 + s] + plane[row0 + s + 1] + plane[row1 + s] + plane[row1 + s + 1];
            }
            let result = f32x8::from_array_t(token, arr) * quarter;
            plane[out_row + ox..][..8].copy_from_slice(&result.to_array());
        }

        for x in (chunks8 * 8)..new_w {
            let sx = x * 2;
            plane[out_row + x] =
                (plane[row0 + sx] + plane[row0 + sx + 1] + plane[row1 + sx] + plane[row1 + sx + 1])
                    * 0.25;
        }
    }
}

// ─── Out-of-place 2× downscale (separate src and dst slices) ─────────────
//
// Two magetypes blocks — each tier uses its native SIMD width, so we get
// hand-tuned-per-tier perf without duplicating bodies via #[arcane]:
//   - v4 / v4x (AVX-512 family): f32x16 native body, 16 outputs per inner iter
//   - v3 / neon / wasm128 / scalar: f32x8 native body, 8 outputs per inner iter
// Both blocks emit the same base name `downscale_2x_into_inner`; the
// suffixed variants have disjoint tier sets, so `incant!` resolves cleanly.

// Tier-natural SIMD widths via two #[magetypes] blocks: AVX-512 family
// uses f32x16 native; everything else uses f32x8 (native on AVX2 / NEON
// polyfilled to 2× / WASM SIMD128 polyfilled to 2×). The first block uses
// additive-mode (`+v4, +v4x, -v3, -neon, -wasm128, -scalar`) to leave only
// the AVX-512 tiers and explicitly drop the auto-scalar fallback —
// requires magetypes >= 0.9.23, which honors `-scalar` removal.
// The second block emits the `_scalar` fallback for everyone else.

#[magetypes(define(f32x16), +v4, +v4x, -v3, -neon, -wasm128, -scalar)]
fn downscale_2x_into_inner(
    token: Token,
    src: &[f32],
    src_w: usize,
    dst: &mut [f32],
    new_w: usize,
    new_h: usize,
) {
    let quarter = f32x16::splat_t(token, 0.25);

    for y in 0..new_h {
        let row0 = y * 2 * src_w;
        let row1 = row0 + src_w;
        let out_row = y * new_w;

        let chunks16 = new_w / 16;
        for chunk in 0..chunks16 {
            let ox = chunk * 16;
            let sx = ox * 2;
            // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance Optimization"):
            // two range checks at the boundary, ZERO interior. The previous
            // form indexed the `src` SLICE four times per output lane, so
            // LLVM emitted 64 bounds-checked scalar loads per 16 outputs and
            // could not see the strided pattern; against fixed-size arrays it
            // can, and emits the even/odd de-interleave shuffles.
            //
            // BIT-EXACT BY CONSTRUCTION: the per-lane add order is unchanged
            // — `((row0[s] + row0[s+1]) + row1[s]) + row1[s+1]`, then the
            // same `* 0.25`. Nothing is summed across lanes, so f32
            // non-associativity has nothing to bite.
            let r0: &[f32; 32] = src[row0 + sx..row0 + sx + 32]
                .try_into()
                .expect("32 source columns per 16 outputs");
            let r1: &[f32; 32] = src[row1 + sx..row1 + sx + 32]
                .try_into()
                .expect("32 source columns per 16 outputs");
            let mut arr = [0.0f32; 16];
            for i in 0..16 {
                arr[i] = r0[2 * i] + r0[2 * i + 1] + r1[2 * i] + r1[2 * i + 1];
            }
            let result = f32x16::from_array_t(token, arr) * quarter;
            dst[out_row + ox..][..16].copy_from_slice(&result.to_array());
        }

        for x in (chunks16 * 16)..new_w {
            let sx = x * 2;
            dst[out_row + x] =
                (src[row0 + sx] + src[row0 + sx + 1] + src[row1 + sx] + src[row1 + sx + 1]) * 0.25;
        }
    }
}

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn downscale_2x_into_inner(
    token: Token,
    src: &[f32],
    src_w: usize,
    dst: &mut [f32],
    new_w: usize,
    new_h: usize,
) {
    let quarter = f32x8::splat_t(token, 0.25);

    for y in 0..new_h {
        let row0 = y * 2 * src_w;
        let row1 = row0 + src_w;
        let out_row = y * new_w;

        let chunks8 = new_w / 8;
        for chunk in 0..chunks8 {
            let ox = chunk * 8;
            let sx = ox * 2;
            // Same fixed-size-array de-interleave as the f32x16 block above,
            // at this tier's natural width. Bit-exact for the same reason:
            // the per-lane add order and the `* 0.25` are unchanged.
            let r0: &[f32; 16] = src[row0 + sx..row0 + sx + 16]
                .try_into()
                .expect("16 source columns per 8 outputs");
            let r1: &[f32; 16] = src[row1 + sx..row1 + sx + 16]
                .try_into()
                .expect("16 source columns per 8 outputs");
            let mut arr = [0.0f32; 8];
            for i in 0..8 {
                arr[i] = r0[2 * i] + r0[2 * i + 1] + r1[2 * i] + r1[2 * i + 1];
            }
            let result = f32x8::from_array_t(token, arr) * quarter;
            dst[out_row + ox..][..8].copy_from_slice(&result.to_array());
        }

        for x in (chunks8 * 8)..new_w {
            let sx = x * 2;
            dst[out_row + x] =
                (src[row0 + sx] + src[row0 + sx + 1] + src[row1 + sx] + src[row1 + sx + 1]) * 0.25;
        }
    }
}

/// Row stride of a v1 pyramid plane — **equal to the image width**.
///
/// OPTION C (2026-08-30, user decision; `docs/DATASET_HISTORY.md` era-3).
/// This used to round the width up to a multiple of 16, plus a further 16
/// whenever that landed on an even multiple of 16 at or above 512 (an
/// L1d set-aliasing dodge). The extra columns were mirror-filled by
/// `mirror_pad_columns` and then **POOLED** — so every v1 feature at a
/// non-tight width was a statistic over columns that are not in the image.
/// Measured, that was worth up to **81.6 % relative** on a pool slot, it put
/// 512/576/768/1024/1152/2304 all in the divergent class, and it was the
/// entire reason the fold's `f0..372` disagreed with the buffered path.
///
/// Removing it is exact AND faster: buffered v1-372 instruction counts moved
/// **−9.02 % at 576 and −7.37 % at 1152**, with the tight-width control at
/// 592 unchanged (+0.00 %) — so the aliasing dodge was costing more than it
/// saved on the current kernels. The fold needed no change; it never padded.
///
/// Kept as a named function rather than inlined at the ~18 call sites so the
/// decision stays greppable and so a future stride experiment has one owner.
#[inline]
pub(crate) fn pyramid_plane_stride(width: usize) -> usize {
    width
}

/// Compute `padded_width * height` for plane-allocation sites with overflow
/// detection. Used by streaming-pipeline paths whose dimensions originate
/// from a public API that already validated `width × height` but not
/// `padded_width × height`.
pub(crate) fn checked_padded_plane_len(
    padded_width: usize,
    height: usize,
) -> Result<usize, crate::ZensimError> {
    padded_width
        .checked_mul(height)
        .ok_or(crate::ZensimError::ImageTooLarge)
}

#[cfg(test)]
mod tests {
    /// The fused moment pass ([`fused_blur_hv_ssim_at_revision`]) against the
    /// horizontal kernel followed by four vertical ones, bit for bit, at every
    /// SIMD tier permutation: widths with and without remainder columns, on and
    /// across the 256-column panel edge, and short, odd and strip heights.
    #[cfg(feature = "threads")]
    #[test]
    fn fused_moment_pass_matches_the_two_passes() {
        if std::env::var("FUSED_HV_CHILD").as_deref() != Ok("1") {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "blur::tests::fused_moment_pass_matches_the_two_passes",
                    "--exact",
                ])
                .env("FUSED_HV_CHILD", "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }
        use super::*;
        use crate::feature_defs::FormulaRevision;
        let _guard = archmage::testing::lock_token_testing();
        let mut runs = 0;
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |_| {
                for (w, h) in [
                    (1, 1),
                    (7, 3),
                    (8, 12),
                    (15, 2),
                    (16, 11),
                    (17, 9),
                    (255, 13),
                    (256, 21),
                    (257, 21),
                    (300, 148),
                    (513, 31),
                    (1031, 17),
                ] {
                    let src: Vec<f32> = (0..w * h)
                        .map(|i| ((i * 137 + 19) % 1009) as f32 / 1009.0 - 0.2)
                        .collect();
                    let dst: Vec<f32> = src
                        .iter()
                        .enumerate()
                        .map(|(i, v)| if i % 23 == 0 { *v } else { 0.9 * v + 0.02 })
                        .collect();
                    for revision in [FormulaRevision::Rev5, FormulaRevision::Rev7] {
                        let n = w * h;
                        let mut horizontal: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; n]);
                        let mut expected = horizontal.clone();
                        {
                            let [a, b, c, d] = &mut horizontal;
                            fused_blur_h_ssim_at_revision(
                                &src, &dst, a, b, c, d, w, h, 5, revision,
                            );
                        }
                        for (hp, out) in horizontal.iter().zip(expected.iter_mut()) {
                            box_blur_v_from_copy_at_revision(hp, out, w, h, 5, revision);
                        }
                        let mut actual: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; n]);
                        let mut hbuf: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; n]);
                        let [a, b, c, d] = &mut actual;
                        let [h1, h2, h3, h4] = &mut hbuf;
                        assert!(fused_blur_hv_ssim_at_revision(
                            &src,
                            &dst,
                            a,
                            b,
                            c,
                            d,
                            w,
                            h,
                            5,
                            revision,
                            [h1, h2, h3, h4],
                        ));
                        for plane in 0..4 {
                            for i in 0..n {
                                assert_eq!(
                                    actual[plane][i].to_bits(),
                                    expected[plane][i].to_bits(),
                                    "{revision:?} {w}x{h} plane {plane} index {i}"
                                );
                            }
                        }
                    }
                }
                runs += 1;
            },
        );
        assert!(runs >= 1, "{report:?}");
    }

    /// The fused activity pass ([`activity_at_revision`]) against `|src − mu|`
    /// followed by [`box_blur_1pass_at_revision`], bit for bit, at every SIMD
    /// tier permutation and the moment test's geometries.
    #[cfg(feature = "threads")]
    #[test]
    fn fused_activity_pass_matches_the_two_passes() {
        if std::env::var("FUSED_ACT_CHILD").as_deref() != Ok("1") {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "blur::tests::fused_activity_pass_matches_the_two_passes",
                    "--exact",
                ])
                .env("FUSED_ACT_CHILD", "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }
        use super::*;
        use crate::feature_defs::FormulaRevision;
        let _guard = archmage::testing::lock_token_testing();
        let mut runs = 0;
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |_| {
                for (w, h) in [
                    (1, 1),
                    (7, 3),
                    (8, 12),
                    (15, 2),
                    (16, 11),
                    (17, 9),
                    (21, 5),
                    (255, 13),
                    (256, 21),
                    (257, 21),
                    (300, 148),
                    (513, 31),
                    (1031, 17),
                ] {
                    let n = w * h;
                    let src: Vec<f32> = (0..n)
                        .map(|i| ((i * 137 + 19) % 1009) as f32 / 1009.0 - 0.2)
                        .collect();
                    let mu: Vec<f32> = (0..n)
                        .map(|i| {
                            if i % 29 == 0 {
                                src[i]
                            } else {
                                ((i * 71 + 5) % 997) as f32 / 997.0
                            }
                        })
                        .collect();
                    for revision in [FormulaRevision::Rev5, FormulaRevision::Rev7] {
                        let mut abs = vec![0.0; n];
                        let mut temp = vec![0.0; n];
                        let mut expected = vec![0.0; n];
                        crate::simd_ops::abs_diff_into(&src, &mu, &mut abs);
                        box_blur_1pass_at_revision(
                            &abs,
                            &mut expected,
                            &mut temp,
                            w,
                            h,
                            5,
                            revision,
                        );
                        let mut actual = vec![0.0; n];
                        let (mut hbuf, mut row) = (vec![0.0; n], vec![0.0; n]);
                        assert!(activity_at_revision(
                            &src,
                            &mu,
                            &mut actual,
                            w,
                            h,
                            5,
                            revision,
                            &mut hbuf,
                            &mut row,
                        ));
                        for i in 0..n {
                            assert_eq!(
                                actual[i].to_bits(),
                                expected[i].to_bits(),
                                "{revision:?} {w}x{h} index {i}"
                            );
                        }
                    }
                }
                runs += 1;
            },
        );
        assert!(runs >= 1, "{report:?}");
    }

    #[test]
    fn rev5_simd_edges_match_frozen_windows() {
        if std::env::var("REV5PERF_EDGES_CHILD").as_deref() != Ok("1") {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "blur::tests::rev5_simd_edges_match_frozen_windows",
                    "--exact",
                ])
                .env("REV5PERF_EDGES_CHILD", "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }
        use super::*;
        let _guard = archmage::testing::lock_token_testing();
        let report = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |_| {
                for (w, h) in [
                    (1, 1),
                    (7, 3),
                    (8, 9),
                    (15, 2),
                    (16, 9),
                    (17, 9),
                    (31, 33),
                    (32, 33),
                    (64, 64),
                    (97, 63),
                    (131, 65),
                    (1919, 9),
                ] {
                    let src: Vec<f32> = (0..w * h)
                        .map(|i| {
                            if i % 19 == 0 {
                                -0.0
                            } else {
                                ((i * 137 + 19) % 1009) as f32 / 1009.0 - 0.5
                            }
                        })
                        .collect();
                    let dst: Vec<f32> = src
                        .iter()
                        .enumerate()
                        .map(|(i, v)| if i % 23 == 0 { *v } else { 0.9 * v + 0.02 })
                        .collect();
                    for err in [false, true] {
                        let mut expected: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; w * h]);
                        let mut actual = expected.clone();
                        let [a, b, c, d] = &mut expected;
                        fused_blur_h_ssim_local_general(&src, &dst, a, b, c, d, w, h, 5, err);
                        let [a, b, c, d] = &mut actual;
                        fused_blur_h_ssim_local(&src, &dst, a, b, c, d, w, h, 5, err);
                        for plane in 0..4 {
                            assert_eq!(
                                actual[plane]
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                expected[plane]
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                "{w}x{h} plane {plane} err={err}"
                            );
                        }
                    }
                    let mut expected = vec![0.0; w * h];
                    let mut actual = expected.clone();
                    box_blur_h_local_general(&src, &mut expected, w, h, 5);
                    box_blur_h_local(&src, &mut actual, w, h, 5);
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "horizontal tree {w}x{h}"
                    );
                    box_blur_v_local_general(&src, &mut expected, w, h, 5);
                    box_blur_v_local(&src, &mut actual, w, h, 5);
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "vertical tree {w}x{h}"
                    );
                }
            },
        );
        assert!(report.permutations_run >= 1);
    }

    #[test]
    fn rev5_scalar_cached_leaves_match_frozen_windows() {
        use super::*;
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for (w, h) in [
            (0, 0),
            (1, 1),
            (3, 2),
            (10, 9),
            (11, 1),
            (12, 2),
            (15, 9),
            (16, 9),
            (17, 9),
            (31, 33),
            (32, 33),
            (64, 64),
            (97, 63),
            (131, 65),
            (1919, 9),
        ] {
            let src: Vec<f32> = (0..w * h)
                .map(|i| {
                    if i % 19 == 0 {
                        -0.0
                    } else {
                        ((i * 137 + 19) % 1009) as f32 / 1009.0 - 0.5
                    }
                })
                .collect();
            let dst: Vec<f32> = src
                .iter()
                .enumerate()
                .map(|(i, v)| if i % 23 == 0 { *v } else { 0.9 * v + 0.02 })
                .collect();
            for err in [false, true] {
                let mut expected: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; w * h]);
                let mut actual = expected.clone();
                let [a, b, c, d] = &mut expected;
                fused_blur_h_ssim_local_general(&src, &dst, a, b, c, d, w, h, 5, err);
                let [a, b, c, d] = &mut actual;
                fused_blur_h_local11_scalar(
                    archmage::ScalarToken::summon().unwrap(),
                    &src,
                    &dst,
                    a,
                    b,
                    c,
                    d,
                    w,
                    h,
                    5,
                    err,
                );
                for plane in 0..4 {
                    assert_eq!(
                        bits(&actual[plane]),
                        bits(&expected[plane]),
                        "H {w}x{h} plane {plane} err={err}"
                    );
                }
            }
            let mut expected = vec![0.0; w * h];
            let mut actual = expected.clone();
            box_blur_h_local_general(&src, &mut expected, w, h, 5);
            box_blur_h_local11_scalar(
                archmage::ScalarToken::summon().unwrap(),
                &src,
                &mut actual,
                w,
                h,
                5,
            );
            assert_eq!(bits(&actual), bits(&expected), "plain H {w}x{h}");
            box_blur_v_local_general(&src, &mut expected, w, h, 5);
            box_blur_v_local11_scalar(
                archmage::ScalarToken::summon().unwrap(),
                &src,
                &mut actual,
                w,
                h,
                5,
            );
            assert_eq!(bits(&actual), bits(&expected), "plain V {w}x{h}");
        }
    }

    #[test]
    fn rev5_local_windows_match_f64_and_have_local_support() {
        use super::*;
        for (w, h) in [(1, 1), (3, 2), (17, 9), (97, 63)] {
            let src: Vec<f32> = (0..w * h)
                .map(|i| ((i * 137 + 19) % 1009) as f32 / 1009.0)
                .collect();
            let dst: Vec<f32> = src.iter().map(|v| 0.9 * v + 0.02).collect();
            let mut outputs = [
                vec![0.0; w * h],
                vec![0.0; w * h],
                vec![0.0; w * h],
                vec![0.0; w * h],
            ];
            let [a, b, c, d] = &mut outputs;
            fused_blur_h_ssim_local(&src, &dst, a, b, c, d, w, h, 5, true);
            for y in 0..h {
                for x in 0..w {
                    let mut exact = [0.0f64; 4];
                    for k in -5..=5 {
                        let i = y * w + crate::featcanon::tap_mirror(x as isize + k, w);
                        let (s, d) = (src[i], dst[i]);
                        for (sum, v) in
                            exact
                                .iter_mut()
                                .zip([s, d, s.mul_add(s, d * d), (s - d) * (s - d)])
                        {
                            *sum += v as f64;
                        }
                    }
                    for j in 0..4 {
                        let e = exact[j] / 11.0;
                        assert!(
                            (outputs[j][y * w + x] as f64 - e).abs()
                                <= 8.0 * f32::EPSILON as f64 * e.abs().max(1e-30),
                            "plane {j} {w}x{h} {x},{y}"
                        );
                    }
                }
            }
            let mut plain = vec![0.0; w * h];
            box_blur_h_local(&src, &mut plain, w, h, 5);
            assert_eq!(plain, outputs[0]);
            box_blur_v_local(&src, &mut plain, w, h, 5);
            for y in 0..h {
                for x in 0..w {
                    let e = (-5..=5)
                        .map(|k| {
                            src[crate::featcanon::tap_mirror(y as isize + k, h) * w + x] as f64
                        })
                        .sum::<f64>()
                        / 11.0;
                    assert!(
                        (plain[y * w + x] as f64 - e).abs()
                            <= 8.0 * f32::EPSILON as f64 * e.abs().max(1e-30)
                    );
                }
            }
            if w > 11 && h > 11 {
                let mut changed = src.clone();
                changed[20 * w + 20] += 1.0;
                let mut other = vec![0.0; w * h];
                box_blur_v_local(&changed, &mut other, w, h, 5);
                for y in 0..h {
                    for x in 0..w {
                        if x != 20 || y.abs_diff(20) > 5 {
                            assert_eq!(plain[y * w + x].to_bits(), other[y * w + x].to_bits());
                        }
                    }
                }
                box_blur_h_local(&changed, &mut other, w, h, 5);
                for y in 0..h {
                    for x in 0..w {
                        if y != 20 || x.abs_diff(20) > 5 {
                            assert_eq!(outputs[0][y * w + x].to_bits(), other[y * w + x].to_bits());
                        }
                    }
                }
            }
        }
    }

    #[allow(unused_imports)]
    use archmage::SimdToken as _;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn padded_box_rows_preserve_contiguous_bits() {
        use archmage::SimdToken;
        let Some(token) = archmage::X64V4xToken::summon() else {
            return;
        };
        for width in [255, 256, 512, 1024, 1026] {
            for height in [15, 16, 17, 24, 31, 32, 48] {
                let len = width * height;
                let src: Vec<_> = (0..len)
                    .map(|i| ((i * 127 + i / width * 31) % 4093) as f32 / 1024.0 - 2.0)
                    .collect();
                for radius in [0, 1, 4, 16, 17] {
                    let mut expected = vec![0.0; len];
                    let mut actual = vec![0.0; len];
                    super::box_blur_h_v4x_strided(
                        token,
                        &src,
                        &mut expected,
                        width,
                        height,
                        radius,
                        width,
                    );
                    super::box_blur_h_v4x_body(token, &src, &mut actual, width, height, radius);
                    assert!(
                        expected
                            .iter()
                            .zip(&actual)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{width}x{height} radius={radius}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn padded_ssim_rows_preserve_contiguous_bits() {
        use archmage::SimdToken;
        let Some(token) = archmage::X64V4xToken::summon() else {
            return;
        };
        for width in [255, 256, 512, 1024, 1026] {
            for height in [15, 16, 17, 24, 31, 32, 48] {
                let len = width * height;
                let src: Vec<_> = (0..len)
                    .map(|i| ((i * 127 + i / width * 31) % 4093) as f32 / 1024.0 - 2.0)
                    .collect();
                let dst: Vec<_> = (0..len)
                    .map(|i| ((i * 337 + i / width * 17) % 4091) as f32 / 1024.0 - 2.0)
                    .collect();
                for radius in [0, 1, 4, 16, 17] {
                    let [mut a, mut b, mut c, mut d] = core::array::from_fn(|_| vec![0.0; len]);
                    let [mut e, mut f, mut g, mut h] = core::array::from_fn(|_| vec![0.0; len]);
                    super::fused_blur_h_ssim_v4x_strided::<true>(
                        token, &src, &dst, &mut a, &mut b, &mut c, &mut d, width, height, radius,
                        width, false,
                    );
                    super::fused_blur_h_ssim_v4x_body::<true>(
                        token, &src, &dst, &mut e, &mut f, &mut g, &mut h, width, height, radius,
                        false,
                    );
                    for (old, new) in [&a, &b, &c, &d].into_iter().zip([&e, &f, &g, &h]) {
                        assert!(
                            old.iter().zip(new).all(|(x, y)| x.to_bits() == y.to_bits()),
                            "{width}x{height} radius={radius}"
                        );
                    }
                    e.fill(-123.0);
                    super::fused_blur_h_ssim_v4x_body::<false>(
                        token, &src, &dst, &mut e, &mut f, &mut g, &mut h, width, height, radius,
                        false,
                    );
                    assert!(e.iter().all(|x| *x == -123.0));
                    for (old, new) in [&b, &c, &d].into_iter().zip([&f, &g, &h]) {
                        assert!(
                            old.iter().zip(new).all(|(x, y)| x.to_bits() == y.to_bits()),
                            "three-output {width}x{height} radius={radius}"
                        );
                    }
                }
            }
        }
    }
    use super::*;

    // Synthetic references for the pixel-response prerequisite. These are not
    // production attribution or finite-removal implementations.
    fn adjoint_probe_blur(input: &[f32], w: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0.0; input.len()];
        box_blur_1pass_into(input, &mut out, &mut vec![0.0; input.len()], w, h, 5);
        out
    }

    fn adjoint_probe_weight(x: usize, y: usize, w: usize, h: usize) -> f32 {
        let wx = if x == 0 || x + 1 == w { 0.5 } else { 1.0 };
        let wy = if y == 0 || y + 1 == h { 0.5 } else { 1.0 };
        wx * wy
    }

    fn adjoint_probe_transpose(input: &[f32], w: usize, h: usize) -> Vec<f32> {
        // Detailed balance for reflect-101: D K = K^T D. In two dimensions
        // D has edge weights 1/2 and corner weights 1/4. Reuse the SIMD K.
        let scaled: Vec<_> = input
            .iter()
            .enumerate()
            .map(|(i, v)| v / adjoint_probe_weight(i % w, i / w, w, h))
            .collect();
        adjoint_probe_blur(&scaled, w, h)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v * adjoint_probe_weight(i % w, i / w, w, h))
            .collect()
    }

    fn adjoint_probe_direct(input: &[f32], w: usize, h: usize, transpose: bool) -> Vec<f64> {
        let reflect = |i: isize, len: usize| {
            let period = 2 * (len - 1);
            let k = i.unsigned_abs() % period;
            if k < len { k } else { period - k }
        };
        let mut out = vec![0.0; input.len()];
        for y in 0..h {
            for x in 0..w {
                for dy in -5..=5 {
                    for dx in -5..=5 {
                        let source = reflect(y as isize + dy, h) * w + reflect(x as isize + dx, w);
                        let center = y * w + x;
                        let (dest, src) = if transpose {
                            (source, center)
                        } else {
                            (center, source)
                        };
                        out[dest] += f64::from(input[src]) / 121.0;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn reflected_box_pixel_adjoint_matches_direct_scatter() {
        let mut max_error = 0.0_f64;
        let mut max_dot_error = 0.0_f64;
        for (w, h) in [(8, 8), (17, 9), (65, 97)] {
            let a: Vec<_> = (0..w * h)
                .map(|i| ((i * 31 + i / w * 13) % 257) as f32 / 128.0 - 1.0)
                .collect();
            let mut b: Vec<_> = (0..w * h)
                .map(|i| ((i * 11 + i / w * 29) % 251) as f32 / 125.0 - 1.0)
                .collect();
            for impulse in [false, true] {
                if impulse {
                    b.fill(0.0);
                    b[0] = 1.0;
                }
                let ka = adjoint_probe_blur(&a, w, h);
                let ktb = adjoint_probe_transpose(&b, w, h);
                for (got, expected) in ka
                    .iter()
                    .zip(adjoint_probe_direct(&a, w, h, false))
                    .chain(ktb.iter().zip(adjoint_probe_direct(&b, w, h, true)))
                {
                    let error = (f64::from(*got) - expected).abs();
                    max_error = max_error.max(error);
                    assert!(error <= 2e-5, "{w}x{h}: {got} vs {expected}");
                }
                let lhs: f64 = ka
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| f64::from(*x) * f64::from(*y))
                    .sum();
                let rhs: f64 = a
                    .iter()
                    .zip(&ktb)
                    .map(|(x, y)| f64::from(*x) * f64::from(*y))
                    .sum();
                let error = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1.0);
                max_dot_error = max_dot_error.max(error);
                assert!(error <= 2e-5, "adjoint pairing {w}x{h}: {lhs} vs {rhs}");
                if impulse {
                    let wrong = adjoint_probe_blur(&b, w, h);
                    assert!(
                        wrong.iter().zip(&ktb).any(|(x, y)| (x - y).abs() > 0.005),
                        "plain reflected K must be rejected as K^T at boundaries"
                    );
                }
            }
        }
        eprintln!("ADJOINT vector_max={max_error:.9e} normalized_dot_max={max_dot_error:.9e}");
    }

    #[test]
    fn residual_pixel_gradient_has_nonzero_repair_contraction() {
        for (w, h) in [(8, 8), (17, 9), (65, 97)] {
            let d: Vec<_> = (0..w * h)
                .map(|i| ((i * 31 + i / w * 13) % 257) as f32 / 128.0 - 1.0)
                .collect();
            let kd = adjoint_probe_blur(&d, w, h);
            let residual: Vec<_> = d.iter().zip(&kd).map(|(x, y)| x - y).collect();
            let ktr = adjoint_probe_transpose(&residual, w, h);
            let gradient: Vec<_> = residual
                .iter()
                .zip(&ktr)
                .map(|(x, y)| 2.0 * f64::from(x - y))
                .collect();
            let energy: f64 = residual.iter().map(|v| f64::from(*v).powi(2)).sum();
            let sum: f64 = gradient.iter().sum();
            let contraction: f64 = gradient
                .iter()
                .zip(&d)
                .map(|(g, d)| -g * f64::from(*d))
                .sum();
            assert!(sum.abs() <= 2e-5 * energy.max(1.0));
            assert!((contraction + 2.0 * energy).abs() <= 2e-5 * energy.max(1.0));
            assert!(
                energy > 1.0 && (contraction + energy).abs() > 0.9 * energy,
                "a correct local derivative does not equal finite full erasure"
            );
            eprintln!(
                "RESIDUAL {w}x{h} energy={energy:.9} gradient_sum={sum:.9e} repair_derivative={contraction:.9}"
            );
        }
    }

    fn adjoint_probe_edge_pool(
        source: &[f32],
        distorted: &[f32],
        source_mean: &[f32],
        w: usize,
        h: usize,
        branch: f64,
        p: i32,
    ) -> f64 {
        let means = adjoint_probe_blur(distorted, w, h);
        let unused = vec![0.0; w * h];
        // Radius zero consumes the already blurred means. The existing fused
        // owner computes and pools the actual f32 edge signals; SSIM output is
        // ignored. This is a synthetic kernel test, not full pixel scoring.
        let acc = crate::fused::fused_vblur_features_ssim(
            source_mean,
            &means,
            &unused,
            &unused,
            source,
            distorted,
            w * h,
            1,
            0,
            1,
            0,
            &mut [],
            &mut [],
            false,
            &mut [],
            false,
            &mut [],
            &mut [],
            false,
            crate::fused::FreeExtrasWork::default(),
            crate::fused::ExtPoolsWork::default(),
            &[],
        );
        let sum = match (branch > 0.0, p) {
            (true, 1) => acc.edge_art,
            (true, 2) => acc.edge_art2,
            (true, 4) => acc.edge_art4,
            (true, 8) => acc.edge_art8,
            (false, 1) => acc.edge_det,
            (false, 2) => acc.edge_det2,
            (false, 4) => acc.edge_det4,
            (false, 8) => acc.edge_det8,
            _ => unreachable!(),
        };
        (sum / (w * h) as f64).powf(1.0 / f64::from(p))
    }

    #[test]
    fn edge_pixel_adjoint_matches_smooth_directional_differences() {
        let mut checks = 0;
        let mut max_error = 0.0_f64;
        for (w, h) in [(8, 8), (17, 9), (65, 97)] {
            let n = w * h;
            let a: Vec<_> = (0..n)
                .map(|i| {
                    let sign = if (i % w + i / w) % 2 == 0 { 1.0 } else { -1.0 };
                    sign * (0.3 + 0.03 * (i % 7) as f32)
                })
                .collect();
            let ka = adjoint_probe_blur(&a, w, h);
            for (factor, branch) in [(1.3, 1.0_f64), (0.7, -1.0)] {
                let d: Vec<_> = a.iter().map(|v| factor * v + 0.02).collect();
                let kd = adjoint_probe_blur(&d, w, h);
                let signals = |pixels: &[f32]| {
                    let means = adjoint_probe_blur(pixels, w, h);
                    (0..n)
                        .map(|i| {
                            let ed = (1.0 + (pixels[i] - means[i]).abs())
                                / (1.0 + (a[i] - ka[i]).abs())
                                - 1.0;
                            (branch * f64::from(ed)).max(0.0)
                        })
                        .collect::<Vec<_>>()
                };
                let signal = signals(&d);
                assert!(signal.iter().all(|v| *v > 0.02), "smooth active branch");
                for p in [1, 2, 4, 8] {
                    let pool = |values: &[f64]| {
                        (values.iter().map(|v| v.powi(p)).sum::<f64>() / n as f64)
                            .powf(1.0 / f64::from(p))
                    };
                    let root = pool(&signal);
                    let canonical_pool =
                        |pixels: &[f32]| adjoint_probe_edge_pool(&a, pixels, &ka, w, h, branch, p);
                    assert!(
                        (canonical_pool(&d) - root).abs() <= 2e-6,
                        "independent root versus canonical edge pool"
                    );
                    let z: Vec<_> = (0..n)
                        .map(|i| {
                            (root.powi(1 - p) * signal[i].powi(p - 1) / n as f64
                                * branch
                                * f64::from((d[i] - kd[i]).signum())
                                / f64::from(1.0 + (a[i] - ka[i]).abs()))
                                as f32
                        })
                        .collect();
                    let ktz = adjoint_probe_transpose(&z, w, h);
                    let g: Vec<_> = z.iter().zip(&ktz).map(|(x, y)| f64::from(x - y)).collect();
                    for rect in [
                        [0, 0, w, h],
                        [0, 0, 4, 4],
                        [w / 3, h / 3, 2 * w / 3, 2 * h / 3],
                    ] {
                        let direction: Vec<_> = (0..n)
                            .map(|i| {
                                if i % w >= rect[0]
                                    && i % w < rect[2]
                                    && i / w >= rect[1]
                                    && i / w < rect[3]
                                {
                                    ((i * 11 + i / w * 29) % 31) as f32 / 15.0 - 1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        let predicted: f64 = g
                            .iter()
                            .zip(&direction)
                            .map(|(g, v)| g * f64::from(*v))
                            .sum();
                        for epsilon in [0.001, 0.0005] {
                            let plus: Vec<_> = d
                                .iter()
                                .zip(&direction)
                                .map(|(d, v)| d + epsilon * v)
                                .collect();
                            let minus: Vec<_> = d
                                .iter()
                                .zip(&direction)
                                .map(|(d, v)| d - epsilon * v)
                                .collect();
                            let observed = (canonical_pool(&plus) - canonical_pool(&minus))
                                / (2.0 * f64::from(epsilon));
                            let error = (observed - predicted).abs();
                            max_error = max_error.max(error);
                            assert!(
                                error <= 2e-5_f64.max(0.01 * observed.abs().max(predicted.abs())),
                                "edge {w}x{h} factor={factor} p={p} rect={rect:?} eps={epsilon}: predicted={predicted} observed={observed}"
                            );
                            checks += 1;
                        }
                    }
                }
            }
        }
        eprintln!("EDGE_ADJOINT checks={checks} max_abs_directional_error={max_error:.9e}");
    }

    /// Helper: 1-pass box blur (H then V) on a plane.
    fn blur_1pass(input: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
        let n = width * height;
        let mut temp = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];
        box_blur_h(input, &mut temp, width, height, radius);
        box_blur_v_from_copy(&temp, &mut output, width, height, radius);
        output
    }

    /// `downscale_2x_into` must be BIT-IDENTICAL to `downscale_2x_inplace`
    /// on the same input — both compute `(a + b + c + d) * 0.25` per output
    /// element in the same order; only the storage differs. The v2
    /// prepared-reference pyramid is built with `_into` while the pair
    /// path's distorted walk uses `_inplace`, and the v2 feature
    /// bit-identity guarantee (`prepared_ref_bit_identical_to_pair_path`)
    /// rests on this equivalence.
    #[test]
    fn downscale_into_bit_identical_to_inplace() {
        for &(w, h) in &[(64usize, 64usize), (97, 65), (200, 137), (66, 130)] {
            // Structured + pseudo-noise content, incl. negative values.
            let mut state = 0x1234_5678u32;
            let plane: Vec<f32> = (0..w * h)
                .map(|i| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let noise = (state >> 8) as f32 / (1u32 << 24) as f32;
                    let x = (i % w) as f32;
                    let y = (i / w) as f32;
                    (x * 0.013 + y * 0.007).sin() * 0.4 + noise * 0.2 - 0.1
                })
                .collect();

            let new_w = w / 2;
            let new_h = h / 2;
            let mut out_into = vec![0.0f32; new_w * new_h];
            downscale_2x_into(&plane, w, &mut out_into, new_w, new_h);

            let mut inplace = plane.clone();
            let (iw, ih) = downscale_2x_inplace(&mut inplace, w, h);
            assert_eq!((iw, ih), (new_w, new_h));

            for (i, (a, b)) in out_into.iter().zip(inplace.iter()).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "{w}x{h}: downscaled element {i} diverged: into={a:e} inplace={b:e}"
                );
            }
        }
    }

    /// The 2x2 downscale's per-lane ADD ORDER is the whole reason the
    /// fixed-size-array de-interleave (2026-09-05, lane 2) could be landed
    /// bit-exactly, so pin it against a literal reference rather than against
    /// another implementation — `downscale_into_bit_identical_to_inplace`
    /// compares two of our own kernels and would stay green if BOTH were
    /// reordered together.
    ///
    /// Contract: `dst[y][x]` is
    /// `(((src[2y][2x] + src[2y][2x+1]) + src[2y+1][2x]) + src[2y+1][2x+1]) * 0.25`,
    /// evaluated left to right in f32. A SIMD rewrite that sums the two ROWS
    /// first (a natural shape) is a different expression and moves bytes; this
    /// test is what says so.
    ///
    /// Widths are chosen to cover every path: a whole number of 16-wide
    /// chunks with no tail, chunks plus a scalar tail, a width entirely below
    /// one chunk, and odd source widths (where `new_w = src_w / 2` truncates
    /// and the chunk's 32-column fixed-array read must still stay in bounds).
    #[test]
    fn downscale_add_order_is_pinned_left_to_right() {
        // Values with mismatched exponents, so f32 addition is genuinely
        // non-associative here and the negative control below can fire.
        let mk = |w: usize, h: usize| -> Vec<f32> {
            let mut state = 0x9E37_79B9u32;
            (0..w * h)
                .map(|i| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let m = (state >> 8) as f32 / (1u32 << 24) as f32;
                    // Spread across ~7 binades so (a+b)+c != a+(b+c) often.
                    let scale = [1.0f32, 64.0, 1.0 / 128.0, 4096.0][i % 4];
                    (m - 0.5) * scale
                })
                .collect()
        };

        let mut order_sensitive_cells = 0usize;
        for &(w, h) in &[
            (64usize, 64usize), // new_w 32 = 2 chunks of 16, no tail
            (70, 10),           // new_w 35 = 2 chunks + 3 scalar tail
            (20, 8),            // new_w 10 = below one chunk, all tail
            (65, 9),            // odd source width, new_w 32
            (33, 7),            // odd, new_w 16 = exactly one chunk
            (1153, 4),          // odd and above H_TILE_WIDTH
        ] {
            let src = mk(w, h);
            let (new_w, new_h) = (w / 2, h / 2);
            let mut got = vec![0.0f32; new_w * new_h];
            downscale_2x_into(&src, w, &mut got, new_w, new_h);

            for y in 0..new_h {
                for x in 0..new_w {
                    let r0 = y * 2 * w;
                    let r1 = r0 + w;
                    let (a, b) = (src[r0 + 2 * x], src[r0 + 2 * x + 1]);
                    let (c, d) = (src[r1 + 2 * x], src[r1 + 2 * x + 1]);
                    let want = (((a + b) + c) + d) * 0.25;
                    let got_v = got[y * new_w + x];
                    assert!(
                        got_v.to_bits() == want.to_bits(),
                        "{w}x{h} out ({x},{y}): got {got_v:e} ({:08x}) want {want:e} ({:08x})",
                        got_v.to_bits(),
                        want.to_bits()
                    );
                    // NEGATIVE CONTROL: a row-first sum is a real, different
                    // number on this data, so the assertion above has
                    // discriminating power rather than passing vacuously.
                    if (((a + c) + b) + d) * 0.25 != want {
                        order_sensitive_cells += 1;
                    }
                }
            }
        }
        assert!(
            order_sensitive_cells > 0,
            "negative control never fired: the fixture is order-INSENSITIVE, so \
             this test would pass against a reordered kernel — fix the fixture"
        );
    }

    /// Blur of a uniform plane must return the same uniform value everywhere,
    /// including at edges. Any boundary handling that biases edges will fail.
    #[test]
    fn blur_uniform_plane_stays_uniform() {
        for &(w, h) in &[(32, 32), (17, 13), (64, 48)] {
            let val = 0.42f32;
            let input = vec![val; w * h];
            let output = blur_1pass(&input, w, h, 5);
            for (i, &v) in output.iter().enumerate() {
                let y = i / w;
                let x = i % w;
                assert!(
                    (v - val).abs() < 1e-4,
                    "uniform plane {w}x{h}: pixel ({x},{y}) = {v}, expected {val}"
                );
            }
        }
    }

    /// Blur of a plane with a single impulse at each corner should produce
    /// symmetric response for opposite corners (top-left vs bottom-right,
    /// top-right vs bottom-left). Asymmetric boundary handling (mirror left,
    /// clamp right) will make corners behave differently.
    #[test]
    fn blur_corner_impulse_symmetry() {
        let w = 32;
        let h = 32;
        let r = 5;

        let corners = [
            (0, 0),         // top-left
            (w - 1, 0),     // top-right
            (0, h - 1),     // bottom-left
            (w - 1, h - 1), // bottom-right
        ];

        let mut blurred = Vec::new();
        for &(cx, cy) in &corners {
            let mut input = vec![0.0f32; w * h];
            input[cy * w + cx] = 1.0;
            blurred.push(blur_1pass(&input, w, h, r));
        }

        // The blurred value AT the impulse corner should be the same for all 4 corners
        // if boundary handling is symmetric. With clamp-right/mirror-left, the corner
        // values will differ because clamping repeats the edge pixel more than mirroring.
        let corner_vals: Vec<f32> = corners
            .iter()
            .zip(blurred.iter())
            .map(|(&(cx, cy), b)| b[cy * w + cx])
            .collect();

        // Print for diagnostic visibility
        eprintln!("Corner impulse blur values at impulse point:");
        eprintln!("  top-left:     {:.6}", corner_vals[0]);
        eprintln!("  top-right:    {:.6}", corner_vals[1]);
        eprintln!("  bottom-left:  {:.6}", corner_vals[2]);
        eprintln!("  bottom-right: {:.6}", corner_vals[3]);

        // Diagonal pairs should match if boundary handling is fully symmetric.
        // Currently they won't (asymmetric clamp vs mirror), but this test
        // documents the actual asymmetry magnitude.
        let tl_br_diff = (corner_vals[0] - corner_vals[3]).abs();
        let tr_bl_diff = (corner_vals[1] - corner_vals[2]).abs();
        eprintln!("  TL-BR asymmetry: {:.6}", tl_br_diff);
        eprintln!("  TR-BL asymmetry: {:.6}", tr_bl_diff);

        // All 4 corners must produce identical blur response (symmetric mirror boundaries).
        assert!(
            tl_br_diff < 1e-6,
            "TL-BR corner asymmetry {tl_br_diff:.6} exceeds tolerance"
        );
        assert!(
            tr_bl_diff < 1e-6,
            "TR-BL corner asymmetry {tr_bl_diff:.6} exceeds tolerance"
        );
        for i in 1..4 {
            let diff = (corner_vals[0] - corner_vals[i]).abs();
            assert!(
                diff < 1e-6,
                "Corner {} differs from TL by {diff:.6}",
                ["TL", "TR", "BL", "BR"][i]
            );
        }
    }

    /// Blur of horizontally-mirrored input should give horizontally-mirrored output
    /// if boundary handling is symmetric. With clamp-right/mirror-left, the
    /// relationship breaks near edges.
    #[test]
    fn blur_horizontal_mirror_symmetry() {
        let w = 32;
        let h = 16;
        let r = 5;

        // Create an asymmetric ramp: bright on left, dark on right
        let mut input = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                input[y * w + x] = (w - 1 - x) as f32 / (w - 1) as f32;
            }
        }

        // Create horizontally mirrored version
        let mut mirrored = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                mirrored[y * w + x] = input[y * w + (w - 1 - x)];
            }
        }

        let blurred = blur_1pass(&input, w, h, r);
        let blurred_mirror = blur_1pass(&mirrored, w, h, r);

        // If boundary handling is symmetric, blurred[y][x] == blurred_mirror[y][w-1-x]
        let mut max_diff = 0.0f32;
        let mut max_diff_pos = (0, 0);
        for y in 0..h {
            for x in 0..w {
                let diff = (blurred[y * w + x] - blurred_mirror[y * w + (w - 1 - x)]).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_pos = (x, y);
                }
            }
        }

        eprintln!(
            "H-mirror symmetry: max diff = {max_diff:.6} at ({}, {})",
            max_diff_pos.0, max_diff_pos.1
        );
        assert!(
            max_diff < 1e-6,
            "H-mirror blur asymmetry {max_diff:.6} at ({}, {}) exceeds tolerance",
            max_diff_pos.0,
            max_diff_pos.1
        );
    }

    /// Same test but vertical: blur of vertically-mirrored input should give
    /// vertically-mirrored output if boundary handling is symmetric.
    #[test]
    fn blur_vertical_mirror_symmetry() {
        let w = 16;
        let h = 32;
        let r = 5;

        let mut input = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                input[y * w + x] = (h - 1 - y) as f32 / (h - 1) as f32;
            }
        }

        let mut mirrored = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                mirrored[y * w + x] = input[(h - 1 - y) * w + x];
            }
        }

        let blurred = blur_1pass(&input, w, h, r);
        let blurred_mirror = blur_1pass(&mirrored, w, h, r);

        let mut max_diff = 0.0f32;
        let mut max_diff_pos = (0, 0);
        for y in 0..h {
            for x in 0..w {
                let diff = (blurred[y * w + x] - blurred_mirror[(h - 1 - y) * w + x]).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_pos = (x, y);
                }
            }
        }

        eprintln!(
            "V-mirror symmetry: max diff = {max_diff:.6} at ({}, {})",
            max_diff_pos.0, max_diff_pos.1
        );
        assert!(
            max_diff < 1e-6,
            "V-mirror blur asymmetry {max_diff:.6} at ({}, {}) exceeds tolerance",
            max_diff_pos.0,
            max_diff_pos.1
        );
    }

    /// Edge-concentrated distortion: compare metric sensitivity to distortions
    /// at the right/bottom edges vs left/top edges. With symmetric handling,
    /// the scores should be equal. Tests the full metric pipeline.
    #[test]
    fn edge_distortion_left_vs_right() {
        let w = 64;
        let h = 64;
        let n = w * h;

        // Uniform gray source
        let src: Vec<[u8; 3]> = vec![[128, 128, 128]; n];

        // Distortion on left 8 columns (within blur radius of edge)
        let mut dst_left = src.clone();
        for y in 0..h {
            for x in 0..8 {
                dst_left[y * w + x] = [180, 128, 128]; // +52 in red
            }
        }

        // Same distortion on right 8 columns
        let mut dst_right = src.clone();
        for y in 0..h {
            for x in (w - 8)..w {
                dst_right[y * w + x] = [180, 128, 128];
            }
        }

        let score_left = crate::metric::compute_zensim_with_config(
            &src,
            &dst_left,
            w,
            h,
            crate::metric::ZensimConfig::default(),
        )
        .unwrap();
        let score_right = crate::metric::compute_zensim_with_config(
            &src,
            &dst_right,
            w,
            h,
            crate::metric::ZensimConfig::default(),
        )
        .unwrap();

        eprintln!("Edge distortion sensitivity:");
        eprintln!(
            "  Left  8 cols distorted: score={:.4}, raw_dist={:.6}",
            score_left.score(),
            score_left.raw_distance()
        );
        eprintln!(
            "  Right 8 cols distorted: score={:.4}, raw_dist={:.6}",
            score_right.score(),
            score_right.raw_distance()
        );
        let ratio = score_left.raw_distance() / score_right.raw_distance();
        eprintln!("  Left/Right raw_distance ratio: {ratio:.4} (1.0 = symmetric)");

        // Left and right edge distortions must produce nearly identical scores
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "Left/Right edge distortion asymmetry: ratio {ratio:.4}, expected ~1.0"
        );
    }

    /// Same as above but for top vs bottom edge distortion.
    #[test]
    fn edge_distortion_top_vs_bottom() {
        let w = 64;
        let h = 64;
        let n = w * h;

        let src: Vec<[u8; 3]> = vec![[128, 128, 128]; n];

        // Distortion on top 8 rows
        let mut dst_top = src.clone();
        for y in 0..8 {
            for x in 0..w {
                dst_top[y * w + x] = [128, 180, 128];
            }
        }

        // Same distortion on bottom 8 rows
        let mut dst_bottom = src.clone();
        for y in (h - 8)..h {
            for x in 0..w {
                dst_bottom[y * w + x] = [128, 180, 128];
            }
        }

        let score_top = crate::metric::compute_zensim_with_config(
            &src,
            &dst_top,
            w,
            h,
            crate::metric::ZensimConfig::default(),
        )
        .unwrap();
        let score_bottom = crate::metric::compute_zensim_with_config(
            &src,
            &dst_bottom,
            w,
            h,
            crate::metric::ZensimConfig::default(),
        )
        .unwrap();

        eprintln!("Edge distortion sensitivity (vertical):");
        eprintln!(
            "  Top    8 rows distorted: score={:.4}, raw_dist={:.6}",
            score_top.score(),
            score_top.raw_distance()
        );
        eprintln!(
            "  Bottom 8 rows distorted: score={:.4}, raw_dist={:.6}",
            score_bottom.score(),
            score_bottom.raw_distance()
        );
        let ratio = score_top.raw_distance() / score_bottom.raw_distance();
        eprintln!("  Top/Bottom raw_distance ratio: {ratio:.4} (1.0 = symmetric)");

        // Top and bottom edge distortions must produce nearly identical scores
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "Top/Bottom edge distortion asymmetry: ratio {ratio:.4}, expected ~1.0"
        );
    }

    /// Small image where boundary pixels are a large fraction.
    /// At 16x16 with radius=5, the blur window (11 pixels) exceeds the dimension,
    /// so boundary handling dominates every pixel.
    #[test]
    fn small_image_edge_dominance() {
        let w = 16;
        let h = 16;
        let n = w * h;

        // Gradient source: value increases left-to-right
        let src: Vec<[u8; 3]> = (0..n)
            .map(|i| {
                let x = i % w;
                let v = ((x * 255) / (w - 1)) as u8;
                [v, v, v]
            })
            .collect();

        // Horizontally mirrored: value increases right-to-left
        let src_mirror: Vec<[u8; 3]> = (0..n)
            .map(|i| {
                let x = i % w;
                let v = (((w - 1 - x) * 255) / (w - 1)) as u8;
                [v, v, v]
            })
            .collect();

        // Uniform distortion: +20 everywhere
        let dst: Vec<[u8; 3]> = src
            .iter()
            .map(|&[r, g, b]| {
                [
                    r.saturating_add(20),
                    g.saturating_add(20),
                    b.saturating_add(20),
                ]
            })
            .collect();
        let dst_mirror: Vec<[u8; 3]> = src_mirror
            .iter()
            .map(|&[r, g, b]| {
                [
                    r.saturating_add(20),
                    g.saturating_add(20),
                    b.saturating_add(20),
                ]
            })
            .collect();

        // With 2 scales (avoid hitting minimum at scale 3 for 16x16)
        let config = crate::metric::ZensimConfig {
            num_scales: 2,
            ..Default::default()
        };

        let result = crate::metric::compute_zensim_with_config(&src, &dst, w, h, config).unwrap();
        let result_mirror =
            crate::metric::compute_zensim_with_config(&src_mirror, &dst_mirror, w, h, config)
                .unwrap();

        eprintln!("Small image (16x16) mirror symmetry:");
        eprintln!(
            "  Original:  score={:.4}, raw_dist={:.6}",
            result.score(),
            result.raw_distance()
        );
        eprintln!(
            "  H-mirrored: score={:.4}, raw_dist={:.6}",
            result_mirror.score(),
            result_mirror.raw_distance()
        );
        let diff_pct = ((result.raw_distance() - result_mirror.raw_distance())
            / result.raw_distance()
            * 100.0)
            .abs();
        eprintln!("  Score difference: {diff_pct:.2}%");

        // Horizontally mirrored image+distortion must produce nearly identical scores
        assert!(
            diff_pct < 2.5,
            "Small image H-mirror asymmetry: {diff_pct:.2}%, expected < 2.5%"
        );
    }
    /// The rem-ring reuse in `box_blur_h_inner_*` must be BIT-IDENTICAL to
    /// re-gathering the remove-side column every step. This gates the
    /// identity `rem_idx(x) == add_idx(x - diam)` directly, independently of
    /// the v1 golden / fold-pool tests (which only reach the production
    /// radius on production geometries).
    ///
    /// The reference below is the pre-2026-08-30 kernel body, scalar, run per
    /// row — the H-blur is an independent per-row recurrence, so a scalar
    /// row-at-a-time reference is the exact same arithmetic in the exact same
    /// order as any lane width. Sizes deliberately straddle every lane
    /// boundary (16-row groups, 8-row remainder, scalar remainder) and every
    /// `width` vs `diam` relation, including `width <= diam` where the ring
    /// never engages.
    #[test]
    fn box_blur_h_ring_matches_regathered_reference() {
        fn reference(input: &[f32], out: &mut [f32], width: usize, height: usize, radius: usize) {
            let diam = 2 * radius + 1;
            let inv = 1.0 / diam as f32;
            let r = radius;
            for row in 0..height {
                let off = row * width;
                let inp = &input[off..off + width];
                let o = &mut out[off..off + width];
                let mut sum = 0.0f32;
                for i in 0..diam {
                    let idx = if i <= r {
                        (r - i).min(width - 1)
                    } else {
                        (i - r).min(width - 1)
                    };
                    sum += inp[idx];
                }
                for x in 0..width {
                    o[x] = sum * inv;
                    let add_raw = x + r + 1;
                    let add_idx = super::h_mirror_add_idx(add_raw, width).min(width - 1);
                    let rem_i = x as isize - r as isize;
                    let rem_idx = if rem_i < 0 {
                        rem_i.unsigned_abs()
                    } else {
                        rem_i as usize
                    };
                    let rem_idx = rem_idx.min(width - 1);
                    sum = sum + inp[add_idx] - inp[rem_idx];
                }
            }
        }

        for &(w, h) in &[
            (7usize, 3usize), // width <= diam: ring never engages
            (11, 16),         // width == diam exactly
            (12, 16),         // first width where the ring fires
            (13, 17),         // scalar remainder rows
            (31, 24),         // 16-group + 8-group
            (64, 42),         // the fold's band shape (32 + 2*5 overlap)
            (127, 8),         // odd width, pure 8-group
            (208, 26),        // 16-group + 8-group + scalar remainder
            (592, 33),        // the padded production width
        ] {
            let mut input = vec![0.0f32; w * h];
            for (i, v) in input.iter_mut().enumerate() {
                // Deterministic, wide dynamic range, no repeats — a value the
                // ring pulled from the wrong slot cannot coincide with the
                // right one.
                let k = i.wrapping_mul(2654435761usize) % 65521;
                *v = (k as f32) * (1.0 + 1.0 / 1024.0) - 30000.0;
            }
            for &radius in &[1usize, 2, 5, 8] {
                if 2 * radius + 1 > w.max(1) * 4 {
                    continue;
                }
                let mut got = vec![0.0f32; w * h];
                let mut want = vec![0.0f32; w * h];
                super::box_blur_h_untiled(&input, &mut got, w, h, radius);
                reference(&input, &mut want, w, h, radius);
                for i in 0..w * h {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "{w}x{h} r={radius} idx {i}: ring {} != regathered {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    /// Shared geometry set for the rem-ring reference tests: straddles the
    /// 16-row group, the 8-row remainder and the scalar/masked tail, and
    /// every `width` vs `diam` relation including `width <= diam` (where the
    /// ring never engages) and `width == diam` (the boundary).
    const RING_GEOM: &[(usize, usize)] = &[
        (7, 3),
        (11, 16),
        (12, 16),
        (13, 17),
        (31, 24),
        (64, 42),
        (127, 8),
        (208, 26),
        (592, 33),
        // v4x strided-tile path: `width % 256 == 0 && height >= 16` stages
        // the blur through `stride = width + 16` tiles — the plane slices
        // inside `box_blur_h_v4x_strided` must use the strided extent, not
        // `height * width`.
        (256, 32),
        (512, 17),
        (256, 40),
    ];

    fn ring_plane(w: usize, h: usize, salt: usize) -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let k = (i + salt).wrapping_mul(2654435761usize) % 65521;
                (k as f32) * (1.0 + 1.0 / 1024.0) - 30000.0
            })
            .collect()
    }

    /// Per-row scalar reference for the horizontal sliding window. The
    /// H-blur is an independent per-row recurrence and every lane does the
    /// same elementwise IEEE ops, so a scalar row-at-a-time walk reproduces
    /// any lane width BIT-FOR-BIT — provided the association is copied
    /// exactly: the kernels evaluate `(sum + add) - rem`, NOT
    /// `sum += add - rem` (a real distinction the authors already hit when
    /// they replaced the old scalar tails with masked vector groups).
    fn ring_walk(width: usize, radius: usize, mut step: impl FnMut(usize, usize, usize)) {
        let diam = 2 * radius + 1;
        let r = radius;
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            step(usize::MAX, idx, usize::MAX);
        }
        for x in 0..width {
            let add_raw = x + r + 1;
            let add_idx = h_mirror_add_idx(add_raw, width).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            step(x, add_idx, rem_idx.min(width - 1));
        }
    }

    /// `box_blur_h_into_abs_diff` — the v1 activity kernel.
    #[test]
    fn abs_diff_h_ring_matches_regathered_reference() {
        for &(w, h) in RING_GEOM {
            let input = ring_plane(w, h, 0);
            for &radius in &[1usize, 2, 5, 8] {
                let diam = 2 * radius + 1;
                let inv = 1.0 / diam as f32;
                let mut got = vec![0.0f32; w * h];
                super::box_blur_h_into_abs_diff_untiled(&input, &mut got, w, h, radius);
                let mut want = vec![0.0f32; w * h];
                for row in 0..h {
                    let off = row * w;
                    let mut sum = 0.0f32;
                    ring_walk(w, radius, |x, a, rm| {
                        if x == usize::MAX {
                            sum += input[off + a];
                        } else {
                            want[off + x] = (input[off + x] - sum * inv).abs();
                            sum = sum + input[off + a] - input[off + rm];
                        }
                    });
                }
                for i in 0..w * h {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "abs_diff {w}x{h} r={radius} idx {i}"
                    );
                }
            }
        }
    }

    /// REV4VEC2 bit-exactness gate: the 8-row f64x8 Rec64 body must
    /// reproduce `fused_blur_h_rec64_row` on every row **bit for bit** —
    /// each compiled tier variant invoked directly on its summoned token,
    /// so no global dispatch state is touched. Odd widths/heights, tiny
    /// images, every radius, `err` both ways, specials-injected planes.
    #[test]
    fn rec64_rows_vec_matches_scalar_body_on_every_compiled_tier() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                x.wrapping_mul(0x2545_F491_4F6C_DD1D)
            }
            fn f32v(&mut self) -> f32 {
                // Mostly sane magnitudes, a few raw bit patterns (±0,
                // denormals, inf, NaN) injected every 257th value.
                if self.next().is_multiple_of(257) {
                    f32::from_bits(self.next() as u32)
                } else {
                    (self.next() % 4096) as f32 / 128.0 - 16.0
                }
            }
        }

        fn run_scalar(
            src: &[f32],
            dst: &[f32],
            width: usize,
            height: usize,
            r: usize,
            diam: usize,
            inv_v64: f64,
            err: bool,
        ) -> [Vec<f32>; 4] {
            let mut out = std::array::from_fn(|_| vec![0.0; width * height]);
            for y in 0..height {
                let (m1, rest) = out.split_at_mut(1);
                let (m2, rest) = rest.split_at_mut(1);
                let (sq, s12) = rest.split_at_mut(1);
                super::fused_blur_h_rec64_row(
                    src,
                    dst,
                    &mut m1[0],
                    &mut m2[0],
                    &mut sq[0],
                    &mut s12[0],
                    y,
                    width,
                    r,
                    diam,
                    inv_v64,
                    err,
                );
            }
            out
        }

        fn run_vec(
            f: impl Fn(&[f32], &[f32], &mut [f32], &mut [f32], &mut [f32], &mut [f32]),
            src: &[f32],
            dst: &[f32],
            len: usize,
        ) -> [Vec<f32>; 4] {
            let mut out = std::array::from_fn(|_| vec![0.0; len]);
            let (m1, rest) = out.split_at_mut(1);
            let (m2, rest) = rest.split_at_mut(1);
            let (sq, s12) = rest.split_at_mut(1);
            f(src, dst, &mut m1[0], &mut m2[0], &mut sq[0], &mut s12[0]);
            out
        }

        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let cases: Vec<(usize, usize, usize, bool)> = vec![
            (3, 9, 1, true),
            (8, 8, 2, false),
            (9, 17, 4, true),
            (15, 1, 0, false),
            (16, 33, 5, true),
            (17, 8, 3, false),
            (33, 16, 2, true),
            (5, 24, 11, false),
            (64, 9, 7, true),
            // KERNOPT2 transposed-block paths: widths around the 8-column
            // block boundary and the `2r + 9` minimum, heights covering
            // 16-row / 8-row / scalar-tail remainders, radii on both sides
            // of the `diam <= 16` add-history ring, and the v4x
            // strided-tile geometries from KERNOPT.
            (256, 32, 5, true),
            (512, 17, 5, false),
            (256, 40, 5, true),
            (255, 16, 5, false),
            (257, 9, 5, true),
            (24, 15, 5, true),
            (23, 31, 5, false),
            (21, 33, 5, true),
            (20, 8, 5, false),
            (19, 40, 5, true),
            (18, 25, 2, false),
            (248, 17, 5, false),
            (264, 23, 8, true),
            (96, 11, 7, true),
            (97, 40, 9, false),
            (128, 7, 6, true),
            (13, 16, 6, false),
            (41, 24, 3, true),
            (2, 16, 2, false),
            (1, 8, 0, true),
        ];
        let mut compiled = 0usize;
        // Each candidate receives every run-time argument — the closure is
        // defined once per tier, invoked per case.
        type RowFn = dyn Fn(
            &[f32],
            &[f32],
            &mut [f32],
            &mut [f32],
            &mut [f32],
            &mut [f32],
            usize,
            usize,
            usize,
            usize,
            f64,
            bool,
        );
        macro_rules! check_variant {
            ($name:literal, $f:expr) => {{
                compiled += 1;
                let f: &RowFn = &$f;
                for &(w, h, r, err) in &cases {
                    let diam = 2 * r + 1;
                    let inv = 1.0f64 / diam as f64;
                    let len = w * h;
                    let src: Vec<f32> = (0..len).map(|_| rng.f32v()).collect();
                    let dst: Vec<f32> = (0..len).map(|_| rng.f32v()).collect();
                    let want = run_scalar(&src, &dst, w, h, r, diam, inv, err);
                    let got = run_vec(
                        |s, dd, a, b, c, d| f(s, dd, a, b, c, d, w, h, r, diam, inv, err),
                        &src,
                        &dst,
                        len,
                    );
                    for (p, (g, e)) in got.iter().zip(&want).enumerate() {
                        for i in 0..len {
                            assert_eq!(
                                g[i].to_bits(),
                                e[i].to_bits(),
                                "{}: plane {} differs at {i} (w={w} h={h} r={r} err={err})",
                                $name,
                                p
                            );
                        }
                    }
                }
            }};
        }
        check_variant!("scalar", |s: &[f32],
                                  dd: &[f32],
                                  m1: &mut [f32],
                                  m2: &mut [f32],
                                  sq: &mut [f32],
                                  s12: &mut [f32],
                                  w: usize,
                                  h: usize,
                                  r: usize,
                                  diam: usize,
                                  inv: f64,
                                  err: bool| {
            super::fused_blur_h_ssim_rec64_rows_scalar(
                archmage::ScalarToken::summon().expect("infallible"),
                s,
                dd,
                m1,
                m2,
                sq,
                s12,
                w,
                h,
                r,
                diam,
                inv,
                err,
            )
        });
        #[cfg(target_arch = "x86_64")]
        if let Some(t) = archmage::X64V3Token::summon() {
            check_variant!("v3", move |s: &[f32],
                                       dd: &[f32],
                                       m1: &mut [f32],
                                       m2: &mut [f32],
                                       sq: &mut [f32],
                                       s12: &mut [f32],
                                       w: usize,
                                       h: usize,
                                       r: usize,
                                       diam: usize,
                                       inv: f64,
                                       err: bool| {
                super::fused_blur_h_ssim_rec64_rows_v3(
                    t, s, dd, m1, m2, sq, s12, w, h, r, diam, inv, err,
                )
            });
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(t) = archmage::X64V4Token::summon() {
            check_variant!("v4", move |s: &[f32],
                                       dd: &[f32],
                                       m1: &mut [f32],
                                       m2: &mut [f32],
                                       sq: &mut [f32],
                                       s12: &mut [f32],
                                       w: usize,
                                       h: usize,
                                       r: usize,
                                       diam: usize,
                                       inv: f64,
                                       err: bool| {
                super::fused_blur_h_ssim_rec64_rows_v4(
                    t, s, dd, m1, m2, sq, s12, w, h, r, diam, inv, err,
                )
            });
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(t) = archmage::X64V4xToken::summon() {
            check_variant!("v4x", move |s: &[f32],
                                        dd: &[f32],
                                        m1: &mut [f32],
                                        m2: &mut [f32],
                                        sq: &mut [f32],
                                        s12: &mut [f32],
                                        w: usize,
                                        h: usize,
                                        r: usize,
                                        diam: usize,
                                        inv: f64,
                                        err: bool| {
                super::fused_blur_h_ssim_rec64_rows_v4x(
                    t, s, dd, m1, m2, sq, s12, w, h, r, diam, inv, err,
                )
            });
        }
        assert!(compiled >= 1, "no SIMD tier compiled on this host");
    }
}

// Rev5 AVX-512 windows preserve the per-output tree while processing sixteen pixels.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[magetypes(define(f32x16), v4x, -v4, -v3, -scalar)]
fn fused_blur_h_local11_wide(
    token: Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    err: bool,
) {
    let _ = radius;
    let inv = f32x16::splat_t(token, 1.0 / 11.0);
    // The two boundary blocks use the same mirrored columns on every row.
    // Gather each distinct leaf once, then load overlapping windows from it.
    let left: [usize; 26] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror(i as isize - 5, width));
    let last = (width / 16).saturating_sub(1) * 16;
    let right: [usize; 26] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror((last + i) as isize - 5, width));
    // One 16-output block of the four planes from its 26 leaves of each
    // input: the frozen tree over taps at constant offsets (one range check
    // per block and input, not one per tap).
    let block = |a_t: &[f32; 26], b_t: &[f32; 26]| {
        let tap = |k: usize| {
            let a = f32x16::load_t(token, a_t[k..k + 16].try_into().unwrap());
            let b = f32x16::load_t(token, b_t[k..k + 16].try_into().unwrap());
            let q = a.mul_add(a, b * b);
            let p = if err {
                let e = a - b;
                e * e
            } else {
                a * b
            };
            (a, b, q, p)
        };
        let add = |a: (f32x16, f32x16, f32x16, f32x16), b: (f32x16, f32x16, f32x16, f32x16)| {
            (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3)
        };
        let pair = |k| add(tap(k), tap(k + 1));
        let s01 = add(pair(0), pair(2));
        let s45 = add(pair(4), pair(6));
        add(add(add(s01, s45), pair(8)), tap(10))
    };
    for y in 0..height {
        let row = y * width;
        let (sr, dr) = (&src[row..row + width], &dst[row..row + width]);
        let o1 = &mut out_mu1[row..row + width];
        let o2 = &mut out_mu2[row..row + width];
        let oq = &mut out_sigma_sq[row..row + width];
        let op = &mut out_sigma12[row..row + width];
        let mut x = 0;
        while x + 16 <= width {
            let sum = if x >= 5 && x + 16 + 5 <= width {
                block(
                    sr[x - 5..x + 21].try_into().unwrap(),
                    dr[x - 5..x + 21].try_into().unwrap(),
                )
            } else {
                let indices = if x == 0 { &left } else { &right };
                block(&indices.map(|i| sr[i]), &indices.map(|i| dr[i]))
            };
            (sum.0 * inv).store((&mut o1[x..x + 16]).try_into().unwrap());
            (sum.1 * inv).store((&mut o2[x..x + 16]).try_into().unwrap());
            (sum.2 * inv).store((&mut oq[x..x + 16]).try_into().unwrap());
            (sum.3 * inv).store((&mut op[x..x + 16]).try_into().unwrap());
            x += 16;
        }
        while x < width {
            let mut a = [0.0; 11];
            let mut b = a;
            let mut q = a;
            let mut p = a;
            for k in 0..11 {
                let i = crate::featcanon::tap_mirror(x as isize + k as isize - 5, width);
                a[k] = sr[i];
                b[k] = dr[i];
                q[k] = a[k].mul_add(a[k], b[k] * b[k]);
                p[k] = if err {
                    let e = a[k] - b[k];
                    e * e
                } else {
                    a[k] * b[k]
                };
            }
            o1[x] = local_sum_taps(&a) * (1.0 / 11.0);
            o2[x] = local_sum_taps(&b) * (1.0 / 11.0);
            oq[x] = local_sum_taps(&q) * (1.0 / 11.0);
            op[x] = local_sum_taps(&p) * (1.0 / 11.0);
            x += 1;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[magetypes(define(f32x16), v4x, -v4, -v3, -scalar)]
fn box_blur_h_local11_wide(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let _ = radius;
    let inv = f32x16::splat_t(token, 1.0 / 11.0);
    let left: [usize; 26] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror(i as isize - 5, width));
    let last = (width / 16).saturating_sub(1) * 16;
    let right: [usize; 26] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror((last + i) as isize - 5, width));
    // One 16-output block from its 26 leaves: the frozen tree over taps at
    // constant offsets (one range check per block, not one per tap).
    let block = |t: &[f32; 26]| {
        let l = |k: usize| f32x16::load_t(token, t[k..k + 16].try_into().unwrap());
        (((l(0) + l(1)) + (l(2) + l(3))) + ((l(4) + l(5)) + (l(6) + l(7)))) + (l(8) + l(9)) + l(10)
    };
    for y in 0..height {
        let rin = &input[y * width..(y + 1) * width];
        let rout = &mut output[y * width..(y + 1) * width];
        let mut x = 0;
        while x + 16 <= width {
            let sum = if x >= 5 && x + 16 + 5 <= width {
                block(rin[x - 5..x + 21].try_into().unwrap())
            } else {
                let indices = if x == 0 { &left } else { &right };
                block(&indices.map(|i| rin[i]))
            };
            (sum * inv).store((&mut rout[x..x + 16]).try_into().unwrap());
            x += 16;
        }
        while x < width {
            let t: [f32; 11] = std::array::from_fn(|k| {
                rin[crate::featcanon::tap_mirror(x as isize + k as isize - 5, width)]
            });
            rout[x] = local_sum_taps(&t) * (1.0 / 11.0);
            x += 1;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[magetypes(define(f32x16), v4x, -v4, -v3, -scalar)]
fn box_blur_v_local11_wide(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    debug_assert_eq!(radius, 5);
    if width == 0 || height == 0 {
        return;
    }
    let inv = f32x16::splat_t(token, 1.0 / 11.0);
    // Cache the frozen tree's pair and quadruple nodes, never an output
    // recurrence. Each node has the same two operands and rounding as before.
    // A panel of 2 KiB row segments walks the whole height, so every row's
    // reads and writes are contiguous runs the stream prefetchers follow (the
    // former 512-byte tiles moved one row stride per few cache lines). A pair
    // node lives three rows and a quadruple node eight, so rings of four and
    // eight segments hold every live node.
    const PANEL: usize = 32;
    let blocks = width / 16;
    let mut pairs = [[f32x16::zero_t(token); PANEL]; 4];
    let mut quads = [[f32x16::zero_t(token); PANEL]; 8];
    for tile in (0..blocks).step_by(PANEL) {
        let count = (blocks - tile).min(PANEL);
        // The panel's span of a tap row and of the output row, cut to exact
        // length once per row, so the column loops have no range checks.
        let (x0, span) = (tile * 16, count * 16);
        for k in 0..9 {
            let a = crate::featcanon::tap_mirror(k as isize - 5, height) * width;
            let b = crate::featcanon::tap_mirror(k as isize - 4, height) * width;
            let ra = &input[a + x0..a + x0 + span].as_chunks::<16>().0[..count];
            let rb = &input[b + x0..b + x0 + span].as_chunks::<16>().0[..count];
            for c in 0..count {
                pairs[k & 3][c] = f32x16::load_t(token, &ra[c]) + f32x16::load_t(token, &rb[c]);
            }
            if k >= 2 {
                for c in 0..count {
                    quads[k - 2][c] = pairs[(k - 2) & 3][c] + pairs[k & 3][c];
                }
            }
        }
        for y in 0..height {
            let a = crate::featcanon::tap_mirror(y as isize + 4, height) * width;
            let b = crate::featcanon::tap_mirror(y as isize + 5, height) * width;
            let ra = &input[a + x0..a + x0 + span].as_chunks::<16>().0[..count];
            let rb = &input[b + x0..b + x0 + span].as_chunks::<16>().0[..count];
            let o = y * width + x0;
            let ro = &mut output[o..o + span].as_chunks_mut::<16>().0[..count];
            let (q0, q4, q7) = (y & 7, (y + 4) & 7, (y + 7) & 7);
            let (p7, p8, p9) = ((y + 7) & 3, (y + 8) & 3, (y + 9) & 3);
            for c in 0..count {
                let last = f32x16::load_t(token, &rb[c]);
                pairs[p9][c] = f32x16::load_t(token, &ra[c]) + last;
                quads[q7][c] = pairs[p7][c] + pairs[p9][c];
                let sum = ((quads[q0][c] + quads[q4][c]) + pairs[p8][c]) + last;
                (sum * inv).store(&mut ro[c]);
            }
        }
    }
    // The incomplete vector uses the unchanged per-output scalar tree.
    if blocks * 16 == width {
        return;
    }
    for y in 0..height {
        let rows: [usize; 11] = std::array::from_fn(|k| {
            crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * width
        });
        for x in blocks * 16..width {
            let taps: [f32; 11] = std::array::from_fn(|k| input[rows[k] + x]);
            output[y * width + x] = local_sum_taps(&taps) * (1.0 / 11.0);
        }
    }
}

/// [`fused_blur_h_local11_wide`] and then [`box_blur_v_local11_wide`] on each
/// of its four planes, a 256-column panel at a time: the panel's horizontal
/// rows go to `hbuf` (four panel-wide planes, row stride = the panel's span,
/// small enough for L2) instead of four image-wide planes. Every output has
/// both kernels' trees, operands and roundings, so the moments are unchanged.
#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "threads"))]
#[magetypes(define(f32x16), v4x, -v4, -v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fused_blur_hv_local11_wide(
    token: Token,
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    err: bool,
    hbuf: [&mut [f32]; 4],
) {
    if width == 0 || height == 0 {
        return;
    }
    let inv = f32x16::splat_t(token, 1.0 / 11.0);
    let left: [usize; 26] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror(i as isize - 5, width));
    let last = (width / 16).saturating_sub(1) * 16;
    let right: [usize; 26] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror((last + i) as isize - 5, width));
    // `fused_blur_h_local11_wide`'s 16-output block.
    let block = |a_t: &[f32; 26], b_t: &[f32; 26]| {
        let tap = |k: usize| {
            let a = f32x16::load_t(token, a_t[k..k + 16].try_into().unwrap());
            let b = f32x16::load_t(token, b_t[k..k + 16].try_into().unwrap());
            let q = a.mul_add(a, b * b);
            let p = if err {
                let e = a - b;
                e * e
            } else {
                a * b
            };
            (a, b, q, p)
        };
        let add = |a: (f32x16, f32x16, f32x16, f32x16), b: (f32x16, f32x16, f32x16, f32x16)| {
            (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3)
        };
        let pair = |k| add(tap(k), tap(k + 1));
        let s01 = add(pair(0), pair(2));
        let s45 = add(pair(4), pair(6));
        add(add(add(s01, s45), pair(8)), tap(10))
    };
    let [h1, h2, hq, hp] = hbuf;
    let mut outs = [out_mu1, out_mu2, out_sigma_sq, out_sigma12];
    const PANEL: usize = 16;
    let blocks = width / 16;
    let mut pairs = [[f32x16::zero_t(token); PANEL]; 4];
    let mut quads = [[f32x16::zero_t(token); PANEL]; 8];
    for tile in (0..blocks).step_by(PANEL) {
        let count = (blocks - tile).min(PANEL);
        let (x0, span) = (tile * 16, count * 16);
        // The panel's horizontal rows.
        for y in 0..height {
            let row = y * width;
            let (sr, dr) = (&src[row..row + width], &dst[row..row + width]);
            let o = y * span;
            let r1 = &mut h1[o..o + span].as_chunks_mut::<16>().0[..count];
            let r2 = &mut h2[o..o + span].as_chunks_mut::<16>().0[..count];
            let rq = &mut hq[o..o + span].as_chunks_mut::<16>().0[..count];
            let rp = &mut hp[o..o + span].as_chunks_mut::<16>().0[..count];
            for c in 0..count {
                let x = x0 + c * 16;
                let sum = if x >= 5 && x + 16 + 5 <= width {
                    block(
                        sr[x - 5..x + 21].try_into().unwrap(),
                        dr[x - 5..x + 21].try_into().unwrap(),
                    )
                } else {
                    let indices = if x == 0 { &left } else { &right };
                    block(&indices.map(|i| sr[i]), &indices.map(|i| dr[i]))
                };
                (sum.0 * inv).store(&mut r1[c]);
                (sum.1 * inv).store(&mut r2[c]);
                (sum.2 * inv).store(&mut rq[c]);
                (sum.3 * inv).store(&mut rp[c]);
            }
        }
        // `box_blur_v_local11_wide`'s panel walk over each of them.
        for (input, output) in [&*h1, &*h2, &*hq, &*hp].into_iter().zip(outs.iter_mut()) {
            for k in 0..9 {
                let a = crate::featcanon::tap_mirror(k as isize - 5, height) * span;
                let b = crate::featcanon::tap_mirror(k as isize - 4, height) * span;
                let ra = &input[a..a + span].as_chunks::<16>().0[..count];
                let rb = &input[b..b + span].as_chunks::<16>().0[..count];
                for c in 0..count {
                    pairs[k & 3][c] = f32x16::load_t(token, &ra[c]) + f32x16::load_t(token, &rb[c]);
                }
                if k >= 2 {
                    for c in 0..count {
                        quads[k - 2][c] = pairs[(k - 2) & 3][c] + pairs[k & 3][c];
                    }
                }
            }
            for y in 0..height {
                let a = crate::featcanon::tap_mirror(y as isize + 4, height) * span;
                let b = crate::featcanon::tap_mirror(y as isize + 5, height) * span;
                let ra = &input[a..a + span].as_chunks::<16>().0[..count];
                let rb = &input[b..b + span].as_chunks::<16>().0[..count];
                let o = y * width + x0;
                let ro = &mut output[o..o + span].as_chunks_mut::<16>().0[..count];
                let (q0, q4, q7) = (y & 7, (y + 4) & 7, (y + 7) & 7);
                let (p7, p8, p9) = ((y + 7) & 3, (y + 8) & 3, (y + 9) & 3);
                for c in 0..count {
                    let last = f32x16::load_t(token, &rb[c]);
                    pairs[p9][c] = f32x16::load_t(token, &ra[c]) + last;
                    quads[q7][c] = pairs[p7][c] + pairs[p9][c];
                    let sum = ((quads[q0][c] + quads[q4][c]) + pairs[p8][c]) + last;
                    (sum * inv).store(&mut ro[c]);
                }
            }
        }
    }
    // The incomplete vector: both kernels' per-output scalar trees, the
    // horizontal one per row into `hbuf` (row stride = the tail's width).
    let xt = blocks * 16;
    let tw = width - xt;
    if tw == 0 {
        return;
    }
    for y in 0..height {
        let row = y * width;
        let (sr, dr) = (&src[row..row + width], &dst[row..row + width]);
        for x in xt..width {
            let mut a = [0.0; 11];
            let mut b = a;
            let mut q = a;
            let mut p = a;
            for k in 0..11 {
                let i = crate::featcanon::tap_mirror(x as isize + k as isize - 5, width);
                a[k] = sr[i];
                b[k] = dr[i];
                q[k] = a[k].mul_add(a[k], b[k] * b[k]);
                p[k] = if err {
                    let e = a[k] - b[k];
                    e * e
                } else {
                    a[k] * b[k]
                };
            }
            let o = y * tw + (x - xt);
            h1[o] = local_sum_taps(&a) * (1.0 / 11.0);
            h2[o] = local_sum_taps(&b) * (1.0 / 11.0);
            hq[o] = local_sum_taps(&q) * (1.0 / 11.0);
            hp[o] = local_sum_taps(&p) * (1.0 / 11.0);
        }
    }
    for (input, output) in [&*h1, &*h2, &*hq, &*hp].into_iter().zip(outs.iter_mut()) {
        for y in 0..height {
            let rows: [usize; 11] = std::array::from_fn(|k| {
                crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * tw
            });
            for x in 0..tw {
                let taps: [f32; 11] = std::array::from_fn(|k| input[rows[k] + x]);
                output[y * width + xt + x] = local_sum_taps(&taps) * (1.0 / 11.0);
            }
        }
    }
}

/// The activity plane `box_blur(|src − mu|)` as [`box_blur_h_local11_wide`]
/// and then [`box_blur_v_local11_wide`] compute it from an `|src − mu|` plane,
/// a 256-column panel at a time: each row's panel span of `|src − mu|` goes to
/// `row` and its horizontal sums to `hbuf` (row stride = the span) instead of
/// two image-wide planes. Every output keeps the three kernels' operations.
#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "threads"))]
#[magetypes(define(f32x16), v4x, -v4, -v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn activity_local11_wide(
    token: Token,
    src: &[f32],
    mu: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    hbuf: &mut [f32],
    row: &mut [f32],
) {
    if width == 0 || height == 0 {
        return;
    }
    let inv = f32x16::splat_t(token, 1.0 / 11.0);
    let left: [usize; 26] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror(i as isize - 5, width));
    let last = (width / 16).saturating_sub(1) * 16;
    let right: [usize; 26] =
        std::array::from_fn(|i| crate::featcanon::tap_mirror((last + i) as isize - 5, width));
    // `box_blur_h_local11_wide`'s 16-output block.
    let block = |t: &[f32; 26]| {
        let l = |k: usize| f32x16::load_t(token, t[k..k + 16].try_into().unwrap());
        (((l(0) + l(1)) + (l(2) + l(3))) + ((l(4) + l(5)) + (l(6) + l(7)))) + (l(8) + l(9)) + l(10)
    };
    const PANEL: usize = 16;
    let blocks = width / 16;
    let mut pairs = [[f32x16::zero_t(token); PANEL]; 4];
    let mut quads = [[f32x16::zero_t(token); PANEL]; 8];
    for tile in (0..blocks).step_by(PANEL) {
        let count = (blocks - tile).min(PANEL);
        let (x0, span) = (tile * 16, count * 16);
        // Every column this panel's blocks read, mirrored ones included.
        let lo = x0.saturating_sub(5);
        let hi = (x0 + span + 5).min(width);
        for y in 0..height {
            let r = y * width;
            let (s, m) = (&src[r + lo..r + hi], &mu[r + lo..r + hi]);
            let d = &mut row[..hi - lo];
            let full = (hi - lo) / 16;
            for ((dc, sc), mc) in d
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(s.as_chunks::<16>().0)
                .zip(m.as_chunks::<16>().0)
            {
                (f32x16::load_t(token, sc) - f32x16::load_t(token, mc))
                    .abs()
                    .store(dc);
            }
            for i in full * 16..hi - lo {
                d[i] = (s[i] - m[i]).abs();
            }
            let o = y * span;
            let rh = &mut hbuf[o..o + span].as_chunks_mut::<16>().0[..count];
            for c in 0..count {
                let x = x0 + c * 16;
                let sum = if x >= 5 && x + 16 + 5 <= width {
                    block(d[x - 5 - lo..x + 21 - lo].try_into().unwrap())
                } else {
                    let indices = if x == 0 { &left } else { &right };
                    block(&indices.map(|i| d[i - lo]))
                };
                (sum * inv).store(&mut rh[c]);
            }
        }
        // `box_blur_v_local11_wide`'s panel walk.
        let input = &*hbuf;
        for k in 0..9 {
            let a = crate::featcanon::tap_mirror(k as isize - 5, height) * span;
            let b = crate::featcanon::tap_mirror(k as isize - 4, height) * span;
            let ra = &input[a..a + span].as_chunks::<16>().0[..count];
            let rb = &input[b..b + span].as_chunks::<16>().0[..count];
            for c in 0..count {
                pairs[k & 3][c] = f32x16::load_t(token, &ra[c]) + f32x16::load_t(token, &rb[c]);
            }
            if k >= 2 {
                for c in 0..count {
                    quads[k - 2][c] = pairs[(k - 2) & 3][c] + pairs[k & 3][c];
                }
            }
        }
        for y in 0..height {
            let a = crate::featcanon::tap_mirror(y as isize + 4, height) * span;
            let b = crate::featcanon::tap_mirror(y as isize + 5, height) * span;
            let ra = &input[a..a + span].as_chunks::<16>().0[..count];
            let rb = &input[b..b + span].as_chunks::<16>().0[..count];
            let o = y * width + x0;
            let ro = &mut output[o..o + span].as_chunks_mut::<16>().0[..count];
            let (q0, q4, q7) = (y & 7, (y + 4) & 7, (y + 7) & 7);
            let (p7, p8, p9) = ((y + 7) & 3, (y + 8) & 3, (y + 9) & 3);
            for c in 0..count {
                let last = f32x16::load_t(token, &rb[c]);
                pairs[p9][c] = f32x16::load_t(token, &ra[c]) + last;
                quads[q7][c] = pairs[p7][c] + pairs[p9][c];
                let sum = ((quads[q0][c] + quads[q4][c]) + pairs[p8][c]) + last;
                (sum * inv).store(&mut ro[c]);
            }
        }
    }
    // The incomplete vector: both kernels' per-output scalar trees.
    let xt = blocks * 16;
    let tw = width - xt;
    if tw == 0 {
        return;
    }
    for y in 0..height {
        let r = y * width;
        for x in xt..width {
            let t: [f32; 11] = std::array::from_fn(|k| {
                let i = r + crate::featcanon::tap_mirror(x as isize + k as isize - 5, width);
                (src[i] - mu[i]).abs()
            });
            hbuf[y * tw + (x - xt)] = local_sum_taps(&t) * (1.0 / 11.0);
        }
    }
    for y in 0..height {
        let rows: [usize; 11] = std::array::from_fn(|k| {
            crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * tw
        });
        for x in 0..tw {
            let taps: [f32; 11] = std::array::from_fn(|k| hbuf[rows[k] + x]);
            output[y * width + xt + x] = local_sum_taps(&taps) * (1.0 / 11.0);
        }
    }
}

#[cfg(feature = "threads")]
#[path = "blur_local_rows.rs"]
mod local_rows;
#[cfg(feature = "threads")]
pub(crate) use local_rows::box_blur_v_local_rows;
