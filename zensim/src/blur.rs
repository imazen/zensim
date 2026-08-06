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

/// Vertical box blur: read from `src`, write to `dst`.
pub fn box_blur_v_from_copy(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
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
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    for cg in 0..col_groups {
        let col_base = cg * 16;

        let mut sum = f32x16::zero(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x16::from_array(token, src[base..][..16].try_into().unwrap());
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
            let add_v = f32x16::from_array(token, src[add_base..][..16].try_into().unwrap());
            let rem_v = f32x16::from_array(token, src[rem_base..][..16].try_into().unwrap());
            sum = sum + add_v - rem_v;
        }
    }

    // Remainder with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum = f32x8::zero(v3);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x8::from_array(v3, src[base..][..8].try_into().unwrap());
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
            sum = sum + f32x8::from_array(v3, src[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, src[rem_base..][..8].try_into().unwrap());
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
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    for cg in 0..col_groups {
        let col_base = cg * 16;

        let mut sum = f32x16::zero(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x16::from_array(token, src[base..][..16].try_into().unwrap());
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
            let add_v = f32x16::from_array(token, src[add_base..][..16].try_into().unwrap());
            let rem_v = f32x16::from_array(token, src[rem_base..][..16].try_into().unwrap());
            sum = sum + add_v - rem_v;
        }
    }

    // Remainder with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum = f32x8::zero(v3);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x8::from_array(v3, src[base..][..8].try_into().unwrap());
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
            sum = sum + f32x8::from_array(v3, src[add_base..][..8].try_into().unwrap())
                - f32x8::from_array(v3, src[rem_base..][..8].try_into().unwrap());
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
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    for cg in 0..col_groups {
        let col_base = cg * 8;

        // Initialize running sums for 8 columns
        let mut sum = f32x8::zero(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x8::from_array(token, src[base..][..8].try_into().unwrap());
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
            let add_v = f32x8::from_array(token, src[add_base..][..8].try_into().unwrap());
            let rem_v = f32x8::from_array(token, src[rem_base..][..8].try_into().unwrap());
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
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    for cg in 0..col_groups {
        let col_base = cg * 8;

        // Initialize running sums for 8 columns
        let mut sum = f32x8::zero(token);
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            let base = idx * width + col_base;
            sum = sum + f32x8::from_array(token, src[base..][..8].try_into().unwrap());
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
            let add_v = f32x8::from_array(token, src[add_base..][..8].try_into().unwrap());
            let rem_v = f32x8::from_array(token, src[rem_base..][..8].try_into().unwrap());
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
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    for rg in 0..row_groups {
        let row_base = rg * 16;

        let mut sum = f32x16::zero(token);
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
            sum = sum + f32x16::from_array(token, arr);
        }

        for x in 0..width {
            let result = (sum * inv_v).to_array();
            for ro in 0..16 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 16];
            let mut rem_arr = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x16::from_array(token, add_arr) - f32x16::from_array(token, rem_arr);
        }
    }

    // Remainder rows: use v3 (f32x8) path
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut sum = f32x8::zero(v3);
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
            sum = sum + f32x8::from_array(v3, arr);
        }
        for x in 0..width {
            let result = (sum * inv_v8).to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            let mut rem_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x8::from_array(v3, add_arr) - f32x8::from_array(v3, rem_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    for rg in 0..row_groups {
        let row_base = rg * 16;

        let mut sum = f32x16::zero(token);
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
            sum = sum + f32x16::from_array(token, arr);
        }

        for x in 0..width {
            let result = (sum * inv_v).to_array();
            for ro in 0..16 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 16];
            let mut rem_arr = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x16::from_array(token, add_arr) - f32x16::from_array(token, rem_arr);
        }
    }

    // Remainder rows: use v3 (f32x8) path
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut sum = f32x8::zero(v3);
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
            sum = sum + f32x8::from_array(v3, arr);
        }
        for x in 0..width {
            let result = (sum * inv_v8).to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            let mut rem_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x8::from_array(v3, add_arr) - f32x8::from_array(v3, rem_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    for rg in 0..row_groups {
        let row_base = rg * 8;

        // Initialize running sums for 8 rows
        let mut sum = f32x8::zero(token);
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
            sum = sum + f32x8::from_array(token, arr);
        }

        for x in 0..width {
            let result = (sum * inv_v).to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 8];
            let mut rem_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x8::from_array(token, add_arr) - f32x8::from_array(token, rem_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    for rg in 0..row_groups {
        let row_base = rg * 8;

        // Initialize running sums for 8 rows
        let mut sum = f32x8::zero(token);
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
            sum = sum + f32x8::from_array(token, arr);
        }

        for x in 0..width {
            let result = (sum * inv_v).to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 8];
            let mut rem_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x8::from_array(token, add_arr) - f32x8::from_array(token, rem_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    for rg in 0..row_groups {
        let row_base = rg * 16;

        let mut sum = f32x16::zero(token);
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
            sum = sum + f32x16::from_array(token, arr);
        }

        for x in 0..width {
            // Gather src[col=x] for each of the 16 rows. These reads are
            // shared with the running-sum updates below, so they're hot
            // in L1 either way.
            let mut src_arr = [0.0f32; 16];
            for ro in 0..16 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x16::from_array(token, src_arr);
            let blur_v = sum * inv_v;
            let act_v = (src_v - blur_v).abs();
            let result = act_v.to_array();
            for ro in 0..16 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 16];
            let mut rem_arr = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x16::from_array(token, add_arr) - f32x16::from_array(token, rem_arr);
        }
    }

    // Remainder rows with v3 (f32x8)
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut sum = f32x8::zero(v3);
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
            sum = sum + f32x8::from_array(v3, arr);
        }
        for x in 0..width {
            let mut src_arr = [0.0f32; 8];
            for ro in 0..8 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x8::from_array(v3, src_arr);
            let act_v = (src_v - sum * inv_v8).abs();
            let result = act_v.to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            let mut rem_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x8::from_array(v3, add_arr) - f32x8::from_array(v3, rem_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    for rg in 0..row_groups {
        let row_base = rg * 16;

        let mut sum = f32x16::zero(token);
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
            sum = sum + f32x16::from_array(token, arr);
        }

        for x in 0..width {
            // Gather src[col=x] for each of the 16 rows. These reads are
            // shared with the running-sum updates below, so they're hot
            // in L1 either way.
            let mut src_arr = [0.0f32; 16];
            for ro in 0..16 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x16::from_array(token, src_arr);
            let blur_v = sum * inv_v;
            let act_v = (src_v - blur_v).abs();
            let result = act_v.to_array();
            for ro in 0..16 {
                output[(row_base + ro) * width + x] = result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut add_arr = [0.0f32; 16];
            let mut rem_arr = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x16::from_array(token, add_arr) - f32x16::from_array(token, rem_arr);
        }
    }

    // Remainder rows with v3 (f32x8)
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut sum = f32x8::zero(v3);
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
            sum = sum + f32x8::from_array(v3, arr);
        }
        for x in 0..width {
            let mut src_arr = [0.0f32; 8];
            for ro in 0..8 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x8::from_array(v3, src_arr);
            let act_v = (src_v - sum * inv_v8).abs();
            let result = act_v.to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            let mut rem_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x8::from_array(v3, add_arr) - f32x8::from_array(v3, rem_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    for rg in 0..row_groups {
        let row_base = rg * 8;

        let mut sum = f32x8::zero(token);
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
            sum = sum + f32x8::from_array(token, arr);
        }
        for x in 0..width {
            let mut src_arr = [0.0f32; 8];
            for ro in 0..8 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x8::from_array(token, src_arr);
            let act_v = (src_v - sum * inv_v).abs();
            let result = act_v.to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            let mut rem_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x8::from_array(token, add_arr) - f32x8::from_array(token, rem_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut sum = f32x8::zero(token);
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
            sum = sum + f32x8::from_array(token, arr);
        }
        for x in 0..width {
            let mut src_arr = [0.0f32; 8];
            for ro in 0..8 {
                src_arr[ro] = input[(row_base + ro) * width + x];
            }
            let src_v = f32x8::from_array(token, src_arr);
            let act_v = (src_v - sum * inv_v).abs();
            let result = act_v.to_array();
            for ro in 0..8 {
                output[(row_base + ro) * width + x] = result[ro];
            }
            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            let mut add_arr = [0.0f32; 8];
            let mut rem_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                add_arr[ro] = input[base + add_idx];
                rem_arr[ro] = input[base + rem_idx];
            }
            sum = sum + f32x8::from_array(token, add_arr) - f32x8::from_array(token, rem_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_blur_h_mu(
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
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    for rg in 0..row_groups {
        let row_base = rg * 16;
        let mut sum_s = f32x16::zero(token);
        let mut sum_d = f32x16::zero(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 16];
            let mut d_arr = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            sum_s = sum_s + f32x16::from_array(token, s_arr);
            sum_d = sum_d + f32x16::from_array(token, d_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 16];
            let mut d_add = [0.0f32; 16];
            let mut s_rem = [0.0f32; 16];
            let mut d_rem = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            sum_s = sum_s + f32x16::from_array(token, s_add) - f32x16::from_array(token, s_rem);
            sum_d = sum_d + f32x16::from_array(token, d_add) - f32x16::from_array(token, d_rem);
        }
    }

    // Remainder with f32x8
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut sum_s = f32x8::zero(v3);
        let mut sum_d = f32x8::zero(v3);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            sum_s = sum_s + f32x8::from_array(v3, s_arr);
            sum_d = sum_d + f32x8::from_array(v3, d_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            let mut s_rem = [0.0f32; 8];
            let mut d_rem = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            sum_s = sum_s + f32x8::from_array(v3, s_add) - f32x8::from_array(v3, s_rem);
            sum_d = sum_d + f32x8::from_array(v3, d_add) - f32x8::from_array(v3, d_rem);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum_s += s_row[add_idx] - s_row[rem_idx];
            sum_d += d_row[add_idx] - d_row[rem_idx];
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
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    for rg in 0..row_groups {
        let row_base = rg * 16;
        let mut sum_s = f32x16::zero(token);
        let mut sum_d = f32x16::zero(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 16];
            let mut d_arr = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            sum_s = sum_s + f32x16::from_array(token, s_arr);
            sum_d = sum_d + f32x16::from_array(token, d_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 16];
            let mut d_add = [0.0f32; 16];
            let mut s_rem = [0.0f32; 16];
            let mut d_rem = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            sum_s = sum_s + f32x16::from_array(token, s_add) - f32x16::from_array(token, s_rem);
            sum_d = sum_d + f32x16::from_array(token, d_add) - f32x16::from_array(token, d_rem);
        }
    }

    // Remainder with f32x8
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut sum_s = f32x8::zero(v3);
        let mut sum_d = f32x8::zero(v3);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            sum_s = sum_s + f32x8::from_array(v3, s_arr);
            sum_d = sum_d + f32x8::from_array(v3, d_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            let mut s_rem = [0.0f32; 8];
            let mut d_rem = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            sum_s = sum_s + f32x8::from_array(v3, s_add) - f32x8::from_array(v3, s_rem);
            sum_d = sum_d + f32x8::from_array(v3, d_add) - f32x8::from_array(v3, d_rem);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum_s += s_row[add_idx] - s_row[rem_idx];
            sum_d += d_row[add_idx] - d_row[rem_idx];
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
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut sum_s = f32x8::zero(token);
        let mut sum_d = f32x8::zero(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            sum_s = sum_s + f32x8::from_array(token, s_arr);
            sum_d = sum_d + f32x8::from_array(token, d_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            let mut s_rem = [0.0f32; 8];
            let mut d_rem = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            sum_s = sum_s + f32x8::from_array(token, s_add) - f32x8::from_array(token, s_rem);
            sum_d = sum_d + f32x8::from_array(token, d_add) - f32x8::from_array(token, d_rem);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum_s += s_row[add_idx] - s_row[rem_idx];
            sum_d += d_row[add_idx] - d_row[rem_idx];
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
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut sum_s = f32x8::zero(token);
        let mut sum_d = f32x8::zero(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            sum_s = sum_s + f32x8::from_array(token, s_arr);
            sum_d = sum_d + f32x8::from_array(token, d_arr);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            let mut s_rem = [0.0f32; 8];
            let mut d_rem = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            sum_s = sum_s + f32x8::from_array(token, s_add) - f32x8::from_array(token, s_rem);
            sum_d = sum_d + f32x8::from_array(token, d_add) - f32x8::from_array(token, d_rem);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum_s += s_row[add_idx] - s_row[rem_idx];
            sum_d += d_row[add_idx] - d_row[rem_idx];
        }
    }
}

/// Fused horizontal blur for SSIM: computes blur(src), blur(dst), blur(src²+dst²), blur(src*dst)
/// in a single pass. Reads each pixel of src/dst exactly once, eliminating 3 extra H-passes
/// and 2 element-wise ops (sq_sum_into, mul_into).
#[allow(clippy::too_many_arguments)]
pub fn fused_blur_h_ssim(
    src: &[f32],
    dst: &[f32],
    out_mu1: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
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
            radius
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// 3-output fused H-blur — `mu2`/`sigma_sq`/`sigma12` only, for the
/// cached-reference-moments path (`mu1` comes from a
/// `V2PreparedReference` cache, so its accumulator chain + stores are
/// pure waste there). On the v4x tier the mu1 chain is compiled out
/// (`fused_blur_h_ssim_v4x_body::<false>`); every other tier falls back
/// to the 4-output kernel with `mu1_scratch` receiving the unused plane
/// — identical output planes either way (the accumulator chains are
/// independent; gated by `ssim3_matches_ssim4_bitwise`).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(any(feature = "feature-regime-v2", test)), allow(dead_code))] // v2-walk + tests only
pub fn fused_blur_h_ssim3(
    src: &[f32],
    dst: &[f32],
    mu1_scratch: &mut [f32],
    out_mu2: &mut [f32],
    out_sigma_sq: &mut [f32],
    out_sigma12: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken as _;
        if let Some(token) = archmage::X64V4xToken::summon() {
            fused_blur_h_ssim3_inner_v4x(
                token,
                src,
                dst,
                mu1_scratch,
                out_mu2,
                out_sigma_sq,
                out_sigma12,
                width,
                height,
                radius,
            );
            return;
        }
    }
    fused_blur_h_ssim(
        src,
        dst,
        mu1_scratch,
        out_mu2,
        out_sigma_sq,
        out_sigma12,
        width,
        height,
        radius,
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
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    for rg in 0..row_groups {
        let row_base = rg * 16;

        let mut sum_s = f32x16::zero(token);
        let mut sum_d = f32x16::zero(token);
        let mut sum_sq = f32x16::zero(token);
        let mut sum_prod = f32x16::zero(token);

        // Initialize running sums
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 16];
            let mut d_arr = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            let sv = f32x16::from_array(token, s_arr);
            let dv = f32x16::from_array(token, d_arr);
            sum_s = sum_s + sv;
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = sv.mul_add(dv, sum_prod);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 16];
            let mut d_add = [0.0f32; 16];
            let mut s_rem = [0.0f32; 16];
            let mut d_rem = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            let sa = f32x16::from_array(token, s_add);
            let da = f32x16::from_array(token, d_add);
            let sr = f32x16::from_array(token, s_rem);
            let dr = f32x16::from_array(token, d_rem);
            sum_s = sum_s + sa - sr;
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = sa.mul_add(da, (-sr).mul_add(dr, sum_prod));
        }
    }

    // Remainder with f32x8
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut sum_s = f32x8::zero(v3);
        let mut sum_d = f32x8::zero(v3);
        let mut sum_sq = f32x8::zero(v3);
        let mut sum_prod = f32x8::zero(v3);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            let sv = f32x8::from_array(v3, s_arr);
            let dv = f32x8::from_array(v3, d_arr);
            sum_s = sum_s + sv;
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = sv.mul_add(dv, sum_prod);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            let mut s_rem = [0.0f32; 8];
            let mut d_rem = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            let sa = f32x8::from_array(v3, s_add);
            let da = f32x8::from_array(v3, d_add);
            let sr = f32x8::from_array(v3, s_rem);
            let dr = f32x8::from_array(v3, d_rem);
            sum_s = sum_s + sa - sr;
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = sa.mul_add(da, (-sr).mul_add(dr, sum_prod));
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
            sum_prod = s.mul_add(d, sum_prod);
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;
            out_sigma_sq[row_off + x] = sum_sq * inv;
            out_sigma12[row_off + x] = sum_prod * inv;

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
            sum_prod = sa.mul_add(da, (-sr).mul_add(dr, sum_prod));
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
#[cfg_attr(not(any(feature = "feature-regime-v2", test)), allow(dead_code))] // callers live in the ssim3 v4x chain (v2-walk); the arcane-registered siblings mask their own liveness
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
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 16;

    for rg in 0..row_groups {
        let row_base = rg * 16;

        let mut sum_s = f32x16::zero(token);
        let mut sum_d = f32x16::zero(token);
        let mut sum_sq = f32x16::zero(token);
        let mut sum_prod = f32x16::zero(token);

        // Initialize running sums
        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 16];
            let mut d_arr = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            let sv = f32x16::from_array(token, s_arr);
            let dv = f32x16::from_array(token, d_arr);
            if MU1 {
                sum_s = sum_s + sv;
            }
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = sv.mul_add(dv, sum_prod);
        }

        // Slide window
        for x in 0..width {
            let mu2_result = (sum_d * inv_v).to_array();
            let sq_result = (sum_sq * inv_v).to_array();
            let prod_result = (sum_prod * inv_v).to_array();
            if MU1 {
                let mu1_result = (sum_s * inv_v).to_array();
                for ro in 0..16 {
                    let base = (row_base + ro) * width + x;
                    out_mu1[base] = mu1_result[ro];
                }
            }
            for ro in 0..16 {
                let base = (row_base + ro) * width + x;
                out_mu2[base] = mu2_result[ro];
                out_sigma_sq[base] = sq_result[ro];
                out_sigma12[base] = prod_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 16];
            let mut d_add = [0.0f32; 16];
            let mut s_rem = [0.0f32; 16];
            let mut d_rem = [0.0f32; 16];
            for ro in 0..16 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            let sa = f32x16::from_array(token, s_add);
            let da = f32x16::from_array(token, d_add);
            let sr = f32x16::from_array(token, s_rem);
            let dr = f32x16::from_array(token, d_rem);
            if MU1 {
                sum_s = sum_s + sa - sr;
            }
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = sa.mul_add(da, (-sr).mul_add(dr, sum_prod));
        }
    }

    // Remainder with f32x8
    let v3 = token.v3();
    let inv_v8 = f32x8::splat(v3, 1.0 / diam as f32);
    let remaining_start = row_groups * 16;
    let remaining_8groups = (height - remaining_start) / 8;

    for rg in 0..remaining_8groups {
        let row_base = remaining_start + rg * 8;
        let mut sum_s = f32x8::zero(v3);
        let mut sum_d = f32x8::zero(v3);
        let mut sum_sq = f32x8::zero(v3);
        let mut sum_prod = f32x8::zero(v3);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            let sv = f32x8::from_array(v3, s_arr);
            let dv = f32x8::from_array(v3, d_arr);
            if MU1 {
                sum_s = sum_s + sv;
            }
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = sv.mul_add(dv, sum_prod);
        }

        for x in 0..width {
            let mu2_result = (sum_d * inv_v8).to_array();
            let sq_result = (sum_sq * inv_v8).to_array();
            let prod_result = (sum_prod * inv_v8).to_array();
            if MU1 {
                let mu1_result = (sum_s * inv_v8).to_array();
                for ro in 0..8 {
                    let base = (row_base + ro) * width + x;
                    out_mu1[base] = mu1_result[ro];
                }
            }
            for ro in 0..8 {
                let base = (row_base + ro) * width + x;
                out_mu2[base] = mu2_result[ro];
                out_sigma_sq[base] = sq_result[ro];
                out_sigma12[base] = prod_result[ro];
            }

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            let mut s_rem = [0.0f32; 8];
            let mut d_rem = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            let sa = f32x8::from_array(v3, s_add);
            let da = f32x8::from_array(v3, d_add);
            let sr = f32x8::from_array(v3, s_rem);
            let dr = f32x8::from_array(v3, d_rem);
            if MU1 {
                sum_s = sum_s + sa - sr;
            }
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = sa.mul_add(da, (-sr).mul_add(dr, sum_prod));
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
            if MU1 {
                sum_s += s;
            }
            sum_d += d;
            sum_sq = s.mul_add(s, d.mul_add(d, sum_sq));
            sum_prod = s.mul_add(d, sum_prod);
        }

        for x in 0..width {
            if MU1 {
                out_mu1[row_off + x] = sum_s * inv;
            }
            out_mu2[row_off + x] = sum_d * inv;
            out_sigma_sq[row_off + x] = sum_sq * inv;
            out_sigma12[row_off + x] = sum_prod * inv;

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
            sum_prod = sa.mul_add(da, (-sr).mul_add(dr, sum_prod));
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
) {
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut sum_s = f32x8::zero(token);
        let mut sum_d = f32x8::zero(token);
        let mut sum_sq = f32x8::zero(token);
        let mut sum_prod = f32x8::zero(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            let sv = f32x8::from_array(token, s_arr);
            let dv = f32x8::from_array(token, d_arr);
            sum_s = sum_s + sv;
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = sv.mul_add(dv, sum_prod);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            let mut s_rem = [0.0f32; 8];
            let mut d_rem = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            let sa = f32x8::from_array(token, s_add);
            let da = f32x8::from_array(token, d_add);
            let sr = f32x8::from_array(token, s_rem);
            let dr = f32x8::from_array(token, d_rem);
            sum_s = sum_s + sa - sr;
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = sa.mul_add(da, (-sr).mul_add(dr, sum_prod));
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
            sum_prod = s.mul_add(d, sum_prod);
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;
            out_sigma_sq[row_off + x] = sum_sq * inv;
            out_sigma12[row_off + x] = sum_prod * inv;

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
            sum_prod = sa.mul_add(da, (-sr).mul_add(dr, sum_prod));
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
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let row_groups = height / 8;

    for rg in 0..row_groups {
        let row_base = rg * 8;
        let mut sum_s = f32x8::zero(token);
        let mut sum_d = f32x8::zero(token);
        let mut sum_sq = f32x8::zero(token);
        let mut sum_prod = f32x8::zero(token);

        for i in 0..diam {
            let idx = if i <= r {
                (r - i).min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            let mut s_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width + idx;
                s_arr[ro] = src[base];
                d_arr[ro] = dst[base];
            }
            let sv = f32x8::from_array(token, s_arr);
            let dv = f32x8::from_array(token, d_arr);
            sum_s = sum_s + sv;
            sum_d = sum_d + dv;
            sum_sq = sv.mul_add(sv, dv.mul_add(dv, sum_sq));
            sum_prod = sv.mul_add(dv, sum_prod);
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
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);

            let mut s_add = [0.0f32; 8];
            let mut d_add = [0.0f32; 8];
            let mut s_rem = [0.0f32; 8];
            let mut d_rem = [0.0f32; 8];
            for ro in 0..8 {
                let base = (row_base + ro) * width;
                s_add[ro] = src[base + add_idx];
                d_add[ro] = dst[base + add_idx];
                s_rem[ro] = src[base + rem_idx];
                d_rem[ro] = dst[base + rem_idx];
            }
            let sa = f32x8::from_array(token, s_add);
            let da = f32x8::from_array(token, d_add);
            let sr = f32x8::from_array(token, s_rem);
            let dr = f32x8::from_array(token, d_rem);
            sum_s = sum_s + sa - sr;
            sum_d = sum_d + da - dr;
            sum_sq = sa.mul_add(
                sa,
                da.mul_add(da, (-sr).mul_add(sr, (-dr).mul_add(dr, sum_sq))),
            );
            sum_prod = sa.mul_add(da, (-sr).mul_add(dr, sum_prod));
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
            sum_prod = s.mul_add(d, sum_prod);
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;
            out_sigma_sq[row_off + x] = sum_sq * inv;
            out_sigma12[row_off + x] = sum_prod * inv;

            let add_raw = x + r + 1;
            let add_idx = if add_raw < width {
                add_raw
            } else {
                2 * (width - 1) - add_raw
            };
            let add_idx = add_idx.min(width - 1);
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
            sum_prod = sa.mul_add(da, (-sr).mul_add(dr, sum_prod));
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
    let quarter = f32x16::splat(token, 0.25);
    let v3 = token.v3();
    let quarter8 = f32x8::splat(v3, 0.25);

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
            let result = f32x16::from_array(token, arr) * quarter;
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
            let result = f32x8::from_array(v3, arr) * quarter8;
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
    let quarter = f32x16::splat(token, 0.25);
    let v3 = token.v3();
    let quarter8 = f32x8::splat(v3, 0.25);

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
            let result = f32x16::from_array(token, arr) * quarter;
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
            let result = f32x8::from_array(v3, arr) * quarter8;
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
    let quarter = f32x8::splat(token, 0.25);

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
            let result = f32x8::from_array(token, arr) * quarter;
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
    let quarter = f32x8::splat(token, 0.25);

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
            let result = f32x8::from_array(token, arr) * quarter;
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
    let quarter = f32x16::splat(token, 0.25);

    for y in 0..new_h {
        let row0 = y * 2 * src_w;
        let row1 = row0 + src_w;
        let out_row = y * new_w;

        let chunks16 = new_w / 16;
        for chunk in 0..chunks16 {
            let ox = chunk * 16;
            let sx = ox * 2;
            let mut arr = [0.0f32; 16];
            for i in 0..16 {
                let s = sx + i * 2;
                arr[i] = src[row0 + s] + src[row0 + s + 1] + src[row1 + s] + src[row1 + s + 1];
            }
            let result = f32x16::from_array(token, arr) * quarter;
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
    let quarter = f32x8::splat(token, 0.25);

    for y in 0..new_h {
        let row0 = y * 2 * src_w;
        let row1 = row0 + src_w;
        let out_row = y * new_w;

        let chunks8 = new_w / 8;
        for chunk in 0..chunks8 {
            let ox = chunk * 8;
            let sx = ox * 2;
            let mut arr = [0.0f32; 8];
            for i in 0..8 {
                let s = sx + i * 2;
                arr[i] = src[row0 + s] + src[row0 + s + 1] + src[row1 + s] + src[row1 + s + 1];
            }
            let result = f32x8::from_array(token, arr) * quarter;
            dst[out_row + ox..][..8].copy_from_slice(&result.to_array());
        }

        for x in (chunks8 * 8)..new_w {
            let sx = x * 2;
            dst[out_row + x] =
                (src[row0 + sx] + src[row0 + sx + 1] + src[row1 + sx] + src[row1 + sx + 1]) * 0.25;
        }
    }
}

/// Compute SIMD-aligned width that also avoids L1d cache set aliasing.
///
/// The basic alignment rounds up to a multiple of 16 (SIMD lane count).
/// But when `padded_width * 4` (stride in bytes) causes power-of-2 aliasing
/// in the 32KB 8-way L1d cache (512 sets), rows 0 and 2 map to the same
/// cache set, causing catastrophic conflict misses in H-blur (which writes
/// to 4 output buffers × 16 rows simultaneously).
///
/// Fix: for widths >= 512 (where the H-blur working set exceeds L1),
/// ensure `padded_width / 16` is odd so the cache-line stride between
/// rows is odd, spreading all 16 rows across distinct cache sets.
/// Below 512, the working set fits in L1 and aliasing doesn't matter.
pub(crate) fn simd_padded_width(width: usize) -> usize {
    // For widths near `usize::MAX` the original `(width + 15) & !15` would
    // wrap silently. Saturate to `usize::MAX` instead — every downstream
    // allocation site that derives from `padded_width` is guarded by
    // `padded_width.checked_mul(height)` (see [`checked_padded_plane_len`])
    // and will surface `ImageTooLarge` rather than wrap.
    let aligned = match width.checked_add(15) {
        Some(v) => v & !15,
        None => return usize::MAX,
    };
    if aligned >= 512 && (aligned / 16).is_multiple_of(2) {
        aligned.saturating_add(16)
    } else {
        aligned
    }
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

/// EXACTLY sum-preserving separable box spread (task #67 C2b): each source
/// value spreads uniformly over its `(2r+1)²` window *clipped to bounds*,
/// normalized **per source** — the transpose-normalized counterpart of a box
/// blur, i.e. the allocation operator for attribution mass whose signals live
/// on a blur window.
///
/// Boundary convention (documented per the C2b gate): windows are CLIPPED at
/// the image edge (no mirror); each 1-D pass computes `y_i = x_i / len_i`
/// (`len_i` = in-bounds window length of source `i`) followed by a clipped
/// unnormalized box sum, so `Σ out == Σ x` to f64 rounding **exactly** —
/// every source's mass lands fully in bounds (a mirrored convention re-enters
/// mass at edges and does not preserve column sums). Interior pixels (≥ `r`
/// from every edge) match a normalized `(2r+1)²` box blur bit-for-bit in
/// exact arithmetic. O(N) via prefix sums per row + a sliding row-window for
/// the vertical pass. `tmp` must be `width` long; `plane` is modified in
/// place.
#[cfg_attr(
    not(any(all(feature = "custom-profiles", feature = "feature-regime-v2"), test)),
    allow(dead_code)
)] // f64 attribution-density pass B (needs BOTH features since the cluster
// was custom-profiles-gated, matching af4417f8) + tests
pub(crate) fn box_spread_sum_preserving(
    plane: &mut [f64],
    width: usize,
    height: usize,
    r: usize,
    tmp: &mut Vec<f64>,
) {
    if r == 0 || width == 0 || height == 0 {
        return;
    }
    // ── Horizontal pass (per row): out_j = Σ_{i∈[j−r, j+r]∩bounds} x_i/len_i.
    tmp.clear();
    tmp.resize(width + 1, 0.0);
    for row in plane.chunks_mut(width) {
        // Prefix sums of the per-source-normalized row.
        let mut run = 0.0f64;
        for (x, v) in row.iter().enumerate() {
            let len = ((x + r).min(width - 1) - x.saturating_sub(r) + 1) as f64;
            run += v / len;
            tmp[x + 1] = run;
        }
        for (j, out) in row.iter_mut().enumerate() {
            let lo = j.saturating_sub(r);
            let hi = (j + r).min(width - 1);
            *out = tmp[hi + 1] - tmp[lo];
        }
    }
    // ── Vertical pass: normalize each SOURCE row by its in-bounds window
    //    length, then slide a row-window sum down the image.
    for y in 0..height {
        let len = ((y + r).min(height - 1) - y.saturating_sub(r) + 1) as f64;
        let inv = 1.0 / len;
        for v in &mut plane[y * width..(y + 1) * width] {
            *v *= inv;
        }
    }
    // Sliding window of normalized rows: out_row(j) = Σ rows [j−r, j+r]∩bounds.
    tmp.clear();
    tmp.resize(width, 0.0);
    let mut out = vec![0.0f64; width * height];
    // Initialize window sum for j = 0: rows [0, r].
    for y in 0..=r.min(height - 1) {
        for (t, v) in tmp.iter_mut().zip(&plane[y * width..(y + 1) * width]) {
            *t += *v;
        }
    }
    out[..width].copy_from_slice(tmp);
    for j in 1..height {
        let add = j + r;
        if add < height {
            for (t, v) in tmp.iter_mut().zip(&plane[add * width..(add + 1) * width]) {
                *t += *v;
            }
        }
        if j > r {
            let rem = j - r - 1;
            for (t, v) in tmp.iter_mut().zip(&plane[rem * width..(rem + 1) * width]) {
                *t -= *v;
            }
        }
        out[j * width..(j + 1) * width].copy_from_slice(tmp);
    }
    plane.copy_from_slice(&out);
}

/// Row band height for the parallel phases of [`box_spread_merge_f32`]:
/// coarse enough that a rayon job is tens of microseconds (per-row jobs
/// drown in fork-join overhead), fine enough to load-balance.
#[cfg(feature = "threads")]
const SPREAD_ROW_BAND: usize = 64;

/// Call-site policy cutoff for [`box_spread_merge_f32`]'s `parallel`
/// switch: engage rayon only at/above this element count. Measured
/// crossover 2026-08-01 (28-core WSL2 host, `spread_microbench`, against
/// the fused serial V+merge pass): 0.18x @341k, 0.45x @1.35M, 0.95x
/// @5.3M, **1.36x @16.8M** — the fork-join barriers cost ~0.5-1 ms
/// against low-ms serial work, and the serial path got faster than the
/// banded one below ~8M elements. Every 576²/1152² compare therefore
/// takes the serial path; 4K-class compares' scale 0 engages rayon.
#[cfg_attr(not(any(feature = "custom-profiles", test)), allow(dead_code))] // attribution parallel-spread crossover + its bitwise gate test
pub(crate) const SPREAD_PARALLEL_MIN_N: usize = 8_388_608;

/// f32 spread-and-merge twin of [`box_spread_sum_preserving`] for the
/// fused attribution paths (C3a/#70): same clipped-window
/// per-source-normalized convention, f32 storage, fused with the
/// window→identity merge — `target[i] += spread(win)[i]` — so the spread
/// result never round-trips through a full plane store/reload.
/// `win_plane` is clobbered (used as the H-pass intermediate). Sum
/// delivered into `target` is preserved to f32 rounding — the fused
/// path's documented precision class. Merging is value-exact vs the old
/// two-step (spread in place, then add): an f32 store/load round-trip
/// between producing the spread value and adding it cannot change bits.
///
/// **task #70 lever 1 (parallel spread)**: `parallel` enables a rayon
/// split whose output is **BITWISE identical** to the serial path for any
/// thread/band count — every partition owns its accumulation chains:
///
/// - Pass 1 (horizontal prefix-window + source-row normalization, fused):
///   each row's prefix-sum chain and its `1/len_row` scale are row-local,
///   so row-band parallel with per-band scratch preserves every element's
///   op order.
/// - Pass 2 (vertical slide): the running window sum is an independent
///   per-COLUMN add/sub chain (`acc[x]` only ever combines with values
///   from column `x`, in fixed row order), so column-band partitioning
///   replays exactly the serial chain per element. Bands write band-local
///   column strips into `out_scratch`.
/// - Pass 3 (merge-gather): `target[row][x] += strip value` — row-band
///   parallel, one f32 add per element, disjoint target rows.
///
/// The serial path runs the same loops with one band each, so
/// parallel==serial bitwise (gated by
/// `box_spread_merge_f32_parallel_matches_serial_bitwise`). `tmp` and
/// `out_scratch` are caller-owned so per-iteration callers (the stale
/// single-pass session) pay no per-call allocation.
pub fn box_spread_merge_f32(
    win_plane: &mut [f32],
    target: &mut [f32],
    width: usize,
    height: usize,
    r: usize,
    tmp: &mut Vec<f32>,
    out_scratch: &mut Vec<f32>,
    parallel: bool,
) {
    if width == 0 || height == 0 {
        return;
    }
    let n = width * height;
    debug_assert!(win_plane.len() >= n && target.len() >= n);
    // `parallel` is the mechanism switch only — the WHEN-it-pays policy
    // (plane-size cutoff) lives at the call sites, because the output is
    // bitwise-invariant to the branch taken. Measured 2026-08-01 (28-core
    // WSL2 host): rayon loses below ~4M elements (0.19x @341k, 0.54x
    // @1.35M) and wins above (1.15x @5.3M, 1.58x @16.8M) — three
    // sub-millisecond fork-join barriers dominate small planes.
    #[cfg(not(feature = "threads"))]
    let parallel = {
        // Without rayon the banded path (and its scratch) never runs.
        let _ = (parallel, &mut *out_scratch);
        false
    };
    if r == 0 {
        // Radius-0 spread is the identity: plain merge.
        let merge = |dst: &mut [f32], src: &[f32]| {
            for (d, s) in dst.iter_mut().zip(src) {
                *d += *s;
            }
        };
        if parallel {
            #[cfg(feature = "threads")]
            {
                use rayon::prelude::*;
                target[..n]
                    .par_chunks_mut(SPREAD_ROW_BAND * width)
                    .zip(win_plane[..n].par_chunks(SPREAD_ROW_BAND * width))
                    .for_each(|(d, s)| merge(d, s));
            }
        } else {
            merge(&mut target[..n], &win_plane[..n]);
        }
        return;
    }

    // ── Pass 1: horizontal clipped prefix-window + vertical source-row
    //    normalization, fused per row:
    //    h(y,j) = (Σ_{i∈[j−r,j+r]∩bounds} x_i/lenH_i) · 1/lenV_y.
    //    The 3-segment split (left edge / constant-`len` interior / right
    //    edge) changes no operand or op order — the interior windowed
    //    subtract is a plain slice zip so LLVM can vectorize it.
    let h_row = |y: usize, row: &mut [f32], tmp: &mut Vec<f32>| {
        // `tmp[1..=width]` is fully overwritten before any read and
        // `tmp[0]` is re-anchored explicitly — no full re-zero per row.
        if tmp.len() != width + 1 {
            tmp.clear();
            tmp.resize(width + 1, 0.0);
        }
        tmp[0] = 0.0;
        let mut run = 0.0f32;
        if width > 2 * r {
            for (x, v) in row[..r].iter().enumerate() {
                run += *v / (x + r + 1) as f32;
                tmp[x + 1] = run;
            }
            let ilen = (2 * r + 1) as f32;
            for (t, v) in tmp[r + 1..width - r + 1].iter_mut().zip(&row[r..width - r]) {
                run += *v / ilen;
                *t = run;
            }
            for (x, v) in (width - r..width).zip(&row[width - r..]) {
                run += *v / (width + r - x) as f32;
                tmp[x + 1] = run;
            }
        } else {
            for (x, v) in row.iter().enumerate() {
                let len = ((x + r).min(width - 1) - x.saturating_sub(r) + 1) as f32;
                run += *v / len;
                tmp[x + 1] = run;
            }
        }
        let vlen = ((y + r).min(height - 1) - y.saturating_sub(r) + 1) as f32;
        let vinv = 1.0 / vlen;
        if width > 2 * r {
            for (j, out) in row[..r].iter_mut().enumerate() {
                *out = (tmp[j + r + 1] - tmp[0]) * vinv;
            }
            let (mid, lo_side) = (&tmp[2 * r + 1..width + 1], &tmp[..width - 2 * r]);
            for ((out, hi), lo) in row[r..width - r].iter_mut().zip(mid).zip(lo_side) {
                *out = (*hi - *lo) * vinv;
            }
            let top = tmp[width];
            for (j, out) in (width - r..width).zip(row[width - r..].iter_mut()) {
                *out = (top - tmp[j - r]) * vinv;
            }
        } else {
            for (j, out) in row.iter_mut().enumerate() {
                let lo = j.saturating_sub(r);
                let hi = (j + r).min(width - 1);
                *out = (tmp[hi + 1] - tmp[lo]) * vinv;
            }
        }
    };
    if parallel {
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            win_plane[..n]
                .par_chunks_mut(SPREAD_ROW_BAND * width)
                .enumerate()
                .for_each_init(Vec::new, |tmp, (band, rows)| {
                    let y0 = band * SPREAD_ROW_BAND;
                    for (dy, row) in rows.chunks_mut(width).enumerate() {
                        h_row(y0 + dy, row, tmp);
                    }
                });
        }
    } else {
        for (y, row) in win_plane[..n].chunks_mut(width).enumerate() {
            h_row(y, row, tmp);
        }
    }

    // ── Pass 2 (+3): vertical slide over normalized rows, as independent
    //    per-column running chains, merged into `target`.
    //
    //    SERIAL (#70 lever 2 endpoint — the in-strip-scatter payoff in its
    //    bitwise-safe form): the slide merges DIRECTLY into `target`
    //    (`target_row(j) += acc`) in one fused pass — no band scratch, no
    //    gather. The in-place hazard the scratch existed for died when the
    //    merge target became a separate plane.
    //
    //    PARALLEL: column bands write band-local strips into
    //    `out_scratch`, then a row-band merge-gather adds them into
    //    `target` (safe Rust cannot hand disjoint column ranges of the
    //    same rows to different workers). Bitwise-equal to the serial
    //    fused pass: identical per-column chain order, and the strip
    //    copy round-trip cannot change the single merge add per element.
    let win_ro: &[f32] = win_plane;
    if parallel {
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            let workers = rayon::current_num_threads().max(1);
            let bw = (width.div_ceil(workers).max(64) + 15) & !15;
            // (x0, band_width, base offset into out_scratch) per band.
            let mut layout: Vec<(usize, usize, usize)> = Vec::new();
            let mut x0 = 0usize;
            let mut base = 0usize;
            while x0 < width {
                let b = bw.min(width - x0);
                layout.push((x0, b, base));
                x0 += b;
                base += b * height;
            }
            // Ensure capacity WITHOUT re-zeroing: every element of the
            // band strips is written (row 0 and each j copy the full
            // accumulator row) before the merge-gather reads it, so stale
            // contents are never observable.
            if out_scratch.len() < n {
                out_scratch.resize(n, 0.0);
            }
            // Per-band slide: exactly the serial per-column chain over the
            // band's columns; writes are band-local (disjoint regions).
            let v_slide = |x0: usize, b: usize, out_band: &mut [f32], acc: &mut Vec<f32>| {
                acc.clear();
                acc.resize(b, 0.0);
                for y in 0..=r.min(height - 1) {
                    let src = &win_ro[y * width + x0..y * width + x0 + b];
                    for (t, v) in acc.iter_mut().zip(src) {
                        *t += *v;
                    }
                }
                out_band[..b].copy_from_slice(acc);
                for j in 1..height {
                    let add = j + r;
                    if add < height {
                        let src = &win_ro[add * width + x0..add * width + x0 + b];
                        for (t, v) in acc.iter_mut().zip(src) {
                            *t += *v;
                        }
                    }
                    if j > r {
                        let rem = j - r - 1;
                        let src = &win_ro[rem * width + x0..rem * width + x0 + b];
                        for (t, v) in acc.iter_mut().zip(src) {
                            *t -= *v;
                        }
                    }
                    out_band[j * b..(j + 1) * b].copy_from_slice(acc);
                }
            };
            let mut regions: Vec<(usize, usize, &mut [f32])> = Vec::with_capacity(layout.len());
            let mut rest = &mut out_scratch[..n];
            for &(x0, b, _) in &layout {
                let (head, tail) = rest.split_at_mut(b * height);
                regions.push((x0, b, head));
                rest = tail;
            }
            regions
                .into_par_iter()
                .for_each_init(Vec::new, |acc, (x0, b, region)| v_slide(x0, b, region, acc));
            // Merge-gather the band strips into `target` (one f32 add per
            // element; strips read-only, target rows disjoint per job).
            let scratch: &[f32] = out_scratch;
            target[..n]
                .par_chunks_mut(SPREAD_ROW_BAND * width)
                .enumerate()
                .for_each(|(band, rows)| {
                    for (dy, row) in rows.chunks_mut(width).enumerate() {
                        let j = band * SPREAD_ROW_BAND + dy;
                        for &(x0, b, base) in &layout {
                            let src = &scratch[base + j * b..base + (j + 1) * b];
                            for (d, s) in row[x0..x0 + b].iter_mut().zip(src) {
                                *d += *s;
                            }
                        }
                    }
                });
        }
    } else {
        // Fused V+merge single pass.
        tmp.clear();
        tmp.resize(width, 0.0);
        for y in 0..=r.min(height - 1) {
            for (t, v) in tmp.iter_mut().zip(&win_ro[y * width..(y + 1) * width]) {
                *t += *v;
            }
        }
        for (d, t) in target[..width].iter_mut().zip(tmp.iter()) {
            *d += *t;
        }
        for j in 1..height {
            let add = j + r;
            if add < height {
                for (t, v) in tmp.iter_mut().zip(&win_ro[add * width..(add + 1) * width]) {
                    *t += *v;
                }
            }
            if j > r {
                let rem = j - r - 1;
                for (t, v) in tmp.iter_mut().zip(&win_ro[rem * width..(rem + 1) * width]) {
                    *t -= *v;
                }
            }
            for (d, t) in target[j * width..(j + 1) * width]
                .iter_mut()
                .zip(tmp.iter())
            {
                *d += *t;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `box_spread_sum_preserving` must conserve total mass EXACTLY (to f64
    /// rounding) on arbitrary signed planes including edge-heavy mass, and
    /// match a normalized box blur in the deep interior.
    #[test]
    fn box_spread_preserves_sums_exactly() {
        for &(w, h, r) in &[
            (37usize, 23usize, 5usize),
            (64, 64, 5),
            (8, 128, 3),
            (11, 1, 5),
        ] {
            let mut seed = 0x1234_5678_9ABC_DEF0u64;
            let mut next = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            };
            let mut plane: Vec<f64> = (0..w * h).map(|_| next()).collect();
            // Pile extra mass on the borders to stress the clipped windows.
            for x in 0..w {
                plane[x] += 3.0;
                plane[(h - 1) * w + x] -= 2.0;
            }
            let before: f64 = plane.iter().sum();
            let mass: f64 = plane.iter().map(|v| v.abs()).sum();
            let mut tmp = Vec::new();
            box_spread_sum_preserving(&mut plane, w, h, r, &mut tmp);
            let after: f64 = plane.iter().sum();
            assert!(
                (after - before).abs() <= 1e-12 * mass.max(1.0),
                "({w}x{h}, r={r}): sum {before} -> {after}"
            );
        }
        // Interior equivalence with a normalized (2r+1)^2 box blur: an
        // impulse far from every edge spreads to a uniform window.
        let (w, h, r) = (32usize, 32usize, 5usize);
        let mut plane = vec![0.0f64; w * h];
        plane[16 * w + 16] = 121.0;
        let mut tmp = Vec::new();
        box_spread_sum_preserving(&mut plane, w, h, r, &mut tmp);
        for dy in -(r as isize)..=(r as isize) {
            for dx in -(r as isize)..=(r as isize) {
                let i = ((16 + dy) as usize) * w + (16 + dx) as usize;
                assert!(
                    (plane[i] - 1.0).abs() < 1e-12,
                    "interior impulse spread: got {} at ({dx},{dy})",
                    plane[i]
                );
            }
        }
    }

    /// #70 lever 1 HARD GATE: the parallel spread+merge must be
    /// **BITWISE** identical to the serial path — the row-band H pass,
    /// the column-banded V slide, and the row-band merge-gather preserve
    /// every output element's accumulation order by construction, for ANY
    /// thread/band count. Exercised across several pool sizes (⇒ several
    /// band layouts) via local rayon pools.
    #[cfg(feature = "threads")]
    #[test]
    fn box_spread_merge_f32_parallel_matches_serial_bitwise() {
        // Sizes at/above the parallel cutoff (n ≥ 32768, height ≥ 64),
        // including a padded-width-like shape, a non-multiple-of-16 width,
        // a narrow-width clamp case, and the r == 0 identity-merge path.
        for &(w, h, r) in &[
            (592usize, 576usize, 5usize),
            (331, 200, 5),
            (296, 288, 3),
            (48, 1000, 7),
            (592, 576, 0),
        ] {
            let mut seed = 0x0DDB_1A5E_5BAD_5EEDu64;
            let mut next = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 11) as f32 / (1u64 << 53) as f32 - 2.4e-4
            };
            let mut base: Vec<f32> = (0..w * h).map(|_| next() * 3.0).collect();
            // Edge mass to stress the clipped windows.
            for x in 0..w {
                base[x] += 2.0;
                base[(h - 1) * w + x] -= 1.5;
            }
            // Non-trivial merge target (the id plane in production).
            let target0: Vec<f32> = (0..w * h).map(|_| next()).collect();
            let mut serial_win = base.clone();
            let mut serial_tgt = target0.clone();
            let mut tmp = Vec::new();
            let mut scratch = Vec::new();
            box_spread_merge_f32(
                &mut serial_win,
                &mut serial_tgt,
                w,
                h,
                r,
                &mut tmp,
                &mut scratch,
                false,
            );
            // f32-class sanity: merged mass = target mass + win mass.
            let sb: f64 = base.iter().map(|&v| v as f64).sum::<f64>()
                + target0.iter().map(|&v| v as f64).sum::<f64>();
            let sa: f64 = serial_tgt.iter().map(|&v| v as f64).sum();
            let mass: f64 = base.iter().map(|&v| v.abs() as f64).sum::<f64>()
                + target0.iter().map(|&v| v.abs() as f64).sum::<f64>();
            assert!(
                (sa - sb).abs() <= 1e-3 * mass.max(1.0),
                "({w}x{h}, r={r}): f32 spread+merge sum {sb} -> {sa}"
            );
            for threads in [1usize, 2, 3, 7, 28] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut par_win = base.clone();
                let mut par_tgt = target0.clone();
                pool.install(|| {
                    let mut tmp = Vec::new();
                    let mut scratch = Vec::new();
                    box_spread_merge_f32(
                        &mut par_win,
                        &mut par_tgt,
                        w,
                        h,
                        r,
                        &mut tmp,
                        &mut scratch,
                        true,
                    );
                });
                for (i, (a, b)) in par_tgt.iter().zip(serial_tgt.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "({w}x{h}, r={r}, threads={threads}) px {i}: parallel {a} != serial {b}"
                    );
                }
            }
        }
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

    /// `fused_blur_h_ssim3`'s three outputs must be BIT-IDENTICAL to the
    /// 4-output kernel's — the accumulator chains are independent, so
    /// removing the mu1 chain cannot change them. Guards the
    /// cached-moments fast path's numeric contract at the blur level.
    #[test]
    fn ssim3_matches_ssim4_bitwise() {
        for &(w, h) in &[(64usize, 64usize), (97, 149), (200, 137)] {
            let mut state = 0xABCD_EF01u32;
            let mut next_val = |i: usize| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) * 0.8 + (i % w) as f32 * 1e-3
            };
            let src: Vec<f32> = (0..w * h).map(&mut next_val).collect();
            let dst: Vec<f32> = (0..w * h).map(&mut next_val).collect();
            let n = w * h;
            let (mut m1a, mut m2a, mut sqa, mut s12a) = (
                vec![0.0f32; n],
                vec![0.0f32; n],
                vec![0.0f32; n],
                vec![0.0f32; n],
            );
            fused_blur_h_ssim(&src, &dst, &mut m1a, &mut m2a, &mut sqa, &mut s12a, w, h, 5);
            let (mut m1b, mut m2b, mut sqb, mut s12b) = (
                vec![0.0f32; n],
                vec![0.0f32; n],
                vec![0.0f32; n],
                vec![0.0f32; n],
            );
            fused_blur_h_ssim3(&src, &dst, &mut m1b, &mut m2b, &mut sqb, &mut s12b, w, h, 5);
            for i in 0..n {
                assert!(
                    m2a[i].to_bits() == m2b[i].to_bits()
                        && sqa[i].to_bits() == sqb[i].to_bits()
                        && s12a[i].to_bits() == s12b[i].to_bits(),
                    "{w}x{h}: ssim3 diverged from ssim4 at {i}"
                );
            }
        }
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
}
