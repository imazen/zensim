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
        [v4, v3, neon, wasm128, scalar]
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
            let add_idx = if add_raw < height {
                add_raw
            } else {
                2 * (height - 1) - add_raw
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
            let add_idx = if add_raw < height {
                add_raw
            } else {
                2 * (height - 1) - add_raw
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
            let add_idx = if add_raw < height {
                add_raw
            } else {
                2 * (height - 1) - add_raw
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
            let add_idx = if add_raw < height {
                add_raw
            } else {
                2 * (height - 1) - add_raw
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
            let add_idx = if add_raw < height {
                add_raw
            } else {
                2 * (height - 1) - add_raw
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
            let add_idx = if add_raw < height {
                add_raw
            } else {
                2 * (height - 1) - add_raw
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
            let add_idx = if add_raw < height {
                add_raw
            } else {
                2 * (height - 1) - add_raw
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
        [v4, v3, neon, wasm128, scalar]
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
        [v4, v3, neon, wasm128, scalar]
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
        [v4, v3, neon, wasm128, scalar]
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
        [v4, v3, neon, wasm128, scalar]
    );
}

/// Out-of-place 2× downscale: read from `src` (`src_w × src_h`), write to
/// `dst` (`new_w × new_h` where `new_w = src_w / 2`, `new_h = src_h / 2`).
///
/// Compared to [`downscale_2x_inplace`] this avoids reading and writing the
/// same buffer — useful when callers want to keep the source data alive
/// (e.g. multi-scale pyramid construction with all levels owned).
pub fn downscale_2x_into(
    src: &[f32],
    src_w: usize,
    dst: &mut [f32],
    new_w: usize,
    new_h: usize,
) {
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

// Tier-natural SIMD widths: AVX-512 wants f32x16, AVX2 wants f32x8 native.
// We'd ideally express this as two `#[magetypes]` blocks (v4/v4x f32x16 and
// v3/neon/wasm128 f32x8), but the magetypes resolver auto-appends a `_scalar`
// variant to any block that doesn't list `scalar` or `default`, which makes
// both blocks emit `_scalar` and collide. There's no `-scalar` / `no_scalar`
// flag in 0.9.22. Tracking issue: imazen/archmage (this comment can be
// removed once magetypes adds opt-out).
//
// Workaround (Pattern C from the magetypes README): one #[magetypes] block
// for v4/v4x/neon/wasm128/scalar with f32x16 body, plus a standalone
// `#[arcane]` for v3 with the hand-tuned f32x8 body. `incant!` resolves
// suffixes uniformly. Same per-tier-natural-width perf as 4 hand-tuned
// `#[arcane]` variants, but only the v3 path is hand-written.

#[magetypes(define(f32x16), v4, v4x, neon, wasm128, scalar)]
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

#[cfg(target_arch = "x86_64")]
#[arcane]
fn downscale_2x_into_inner_v3(
    token: archmage::X64V3Token,
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
    let aligned = (width + 15) & !15;
    if aligned >= 512 && (aligned / 16).is_multiple_of(2) {
        aligned + 16
    } else {
        aligned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: 1-pass box blur (H then V) on a plane.
    fn blur_1pass(input: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
        let n = width * height;
        let mut temp = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];
        box_blur_h(input, &mut temp, width, height, radius);
        box_blur_v_from_copy(&temp, &mut output, width, height, radius);
        output
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
