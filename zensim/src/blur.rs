//! Fast box blur cascade: 3-pass box filter approximates Gaussian blur.
//!
//! Unlike recursive Gaussian IIR (used in ssimulacra2, ~60-70% of runtime),
//! box blur is O(1) per pixel using running sums, regardless of radius.
//! Three passes of box blur converge to Gaussian (central limit theorem).
#![allow(clippy::assign_op_pattern, clippy::needless_range_loop)]

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x16;

/// Blur into pre-allocated output buffer. Uses temp as scratch.
/// 3-pass cascade approximates Gaussian (piecewise quadratic).
pub fn box_blur_3pass_into(
    input: &[f32],
    output: &mut [f32],
    temp: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    // Each pass: horizontal into one buffer, vertical into another.
    // Pass 1: input →(h)→ temp →(v)→ output
    box_blur_h(input, temp, width, height, radius);
    box_blur_v_from_copy(temp, output, width, height, radius);
    // Pass 2: output →(h)→ temp →(v)→ output
    box_blur_h(output, temp, width, height, radius);
    box_blur_v_from_copy(temp, output, width, height, radius);
    // Pass 3: output →(h)→ temp →(v)→ output
    box_blur_h(output, temp, width, height, radius);
    box_blur_v_from_copy(temp, output, width, height, radius);
}

/// 1-pass blur: rectangular kernel, 50% fewer operations than 2-pass.
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

/// 2-pass blur: triangular kernel, 33% fewer operations than 3-pass.
/// Use with radius+1 to approximate same effective width as 3-pass with radius.
pub fn box_blur_2pass_into(
    input: &[f32],
    output: &mut [f32],
    temp: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    box_blur_h(input, temp, width, height, radius);
    box_blur_v_from_copy(temp, output, width, height, radius);
    box_blur_h(output, temp, width, height, radius);
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
        [v4, v3]
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

            let add_idx = (y + r + 1).min(height - 1);
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
            let add_idx = (y + r + 1).min(height - 1);
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
            let add_idx = (y + r + 1).min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);
            sum += src[add_idx * width + x] - src[rem_idx * width + x];
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

            let add_idx = (y + r + 1).min(height - 1);
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
            let add_idx = (y + r + 1).min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);
            sum += src[add_idx * width + x] - src[rem_idx * width + x];
        }
    }
}

fn box_blur_v_copy_inner_scalar(
    _token: archmage::ScalarToken,
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv = 1.0 / diam as f32;
    let r = radius;

    for x in 0..width {
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
            let add_idx = (y + r + 1).min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(height - 1);
            sum += src[add_idx * width + x] - src[rem_idx * width + x];
        }
    }
}

/// Horizontal box blur using running sum. O(1) per pixel.
fn box_blur_h(input: &[f32], output: &mut [f32], width: usize, height: usize, radius: usize) {
    incant!(
        box_blur_h_inner(input, output, width, height, radius),
        [v4, v3]
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

            let add_idx = (x + r + 1).min(width - 1);
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
            let add_idx = (x + r + 1).min(width - 1);
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
            let add_idx = (x + r + 1).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum += inp[add_idx] - inp[rem_idx];
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

            let add_idx = (x + r + 1).min(width - 1);
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
            let add_idx = (x + r + 1).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum += inp[add_idx] - inp[rem_idx];
        }
    }
}

fn box_blur_h_inner_scalar(
    _token: archmage::ScalarToken,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv = 1.0 / diam as f32;
    let r = radius;

    for row in 0..height {
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
            let add_idx = (x + r + 1).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 {
                rem_i.unsigned_abs()
            } else {
                rem_i as usize
            };
            let rem_idx = rem_idx.min(width - 1);
            sum += inp[add_idx] - inp[rem_idx];
        }
    }
}

/// Fused horizontal blur for SSIM: computes blur(src), blur(dst), blur(src²+dst²), blur(src*dst)
/// in a single pass. Reads each pixel of src/dst exactly once, eliminating 3 extra H-passes
/// and 2 element-wise ops (sq_sum_into, mul_into).
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
        [v4, v3]
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
            sum_sq = sum_sq + sv * sv + dv * dv;
            sum_prod = sum_prod + sv * dv;
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

            let add_idx = (x + r + 1).min(width - 1);
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
            sum_sq = sum_sq + sa * sa + da * da - sr * sr - dr * dr;
            sum_prod = sum_prod + sa * da - sr * dr;
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
            sum_sq = sum_sq + sv * sv + dv * dv;
            sum_prod = sum_prod + sv * dv;
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

            let add_idx = (x + r + 1).min(width - 1);
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
            sum_sq = sum_sq + sa * sa + da * da - sr * sr - dr * dr;
            sum_prod = sum_prod + sa * da - sr * dr;
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
            sum_sq += s * s + d * d;
            sum_prod += s * d;
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;
            out_sigma_sq[row_off + x] = sum_sq * inv;
            out_sigma12[row_off + x] = sum_prod * inv;

            let add_idx = (x + r + 1).min(width - 1);
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
            sum_s += sa - sr;
            sum_d += da - dr;
            sum_sq += sa * sa + da * da - sr * sr - dr * dr;
            sum_prod += sa * da - sr * dr;
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
            sum_sq = sum_sq + sv * sv + dv * dv;
            sum_prod = sum_prod + sv * dv;
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

            let add_idx = (x + r + 1).min(width - 1);
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
            sum_sq = sum_sq + sa * sa + da * da - sr * sr - dr * dr;
            sum_prod = sum_prod + sa * da - sr * dr;
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
            sum_sq += s * s + d * d;
            sum_prod += s * d;
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;
            out_sigma_sq[row_off + x] = sum_sq * inv;
            out_sigma12[row_off + x] = sum_prod * inv;

            let add_idx = (x + r + 1).min(width - 1);
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
            sum_s += sa - sr;
            sum_d += da - dr;
            sum_sq += sa * sa + da * da - sr * sr - dr * dr;
            sum_prod += sa * da - sr * dr;
        }
    }
}

/// Scalar fallback for fused SSIM horizontal blur.
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_inner_scalar(
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
) {
    let diam = 2 * radius + 1;
    let inv = 1.0 / diam as f32;
    let r = radius;

    for row in 0..height {
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
            sum_sq += s * s + d * d;
            sum_prod += s * d;
        }

        for x in 0..width {
            out_mu1[row_off + x] = sum_s * inv;
            out_mu2[row_off + x] = sum_d * inv;
            out_sigma_sq[row_off + x] = sum_sq * inv;
            out_sigma12[row_off + x] = sum_prod * inv;

            let add_idx = (x + r + 1).min(width - 1);
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
            sum_s += sa - sr;
            sum_d += da - dr;
            sum_sq += sa * sa + da * da - sr * sr - dr * dr;
            sum_prod += sa * da - sr * dr;
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
    incant!(downscale_2x_inner(plane, width, new_w, new_h), [v4, v3]);
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

fn downscale_2x_inner_scalar(
    _token: archmage::ScalarToken,
    plane: &mut [f32],
    width: usize,
    new_w: usize,
    new_h: usize,
) {
    for y in 0..new_h {
        let row0 = y * 2 * width;
        let row1 = row0 + width;
        let out_row = y * new_w;
        for x in 0..new_w {
            let sx = x * 2;
            plane[out_row + x] =
                (plane[row0 + sx] + plane[row0 + sx + 1] + plane[row1 + sx] + plane[row1 + sx + 1])
                    * 0.25;
        }
    }
}

/// Pad plane width to `padded_width` by replicating the rightmost pixel value.
/// Operates in-place, processing rows bottom-to-top to avoid overwriting source data.
pub fn pad_plane_width(plane: &mut Vec<f32>, width: usize, height: usize, padded_width: usize) {
    if padded_width == width {
        return;
    }
    debug_assert!(padded_width > width);
    plane.resize(padded_width * height, 0.0);

    // Process bottom-to-top so we never read a position that was already overwritten.
    for y in (0..height).rev() {
        let src_start = y * width;
        let dst_start = y * padded_width;

        // Fill padding columns with rightmost pixel value
        let edge_val = plane[src_start + width - 1];
        for x in (width..padded_width).rev() {
            plane[dst_start + x] = edge_val;
        }

        // Shift row data to padded position (right-to-left for overlap safety)
        if dst_start != src_start {
            for x in (0..width).rev() {
                plane[dst_start + x] = plane[src_start + x];
            }
        }
    }
}
