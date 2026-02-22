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
fn box_blur_v_from_copy(src: &[f32], dst: &mut [f32], width: usize, height: usize, radius: usize) {
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

/// Downscale in-place: writes to beginning of buffer, truncates.
/// Safe because output index < source index for all elements.
pub fn downscale_2x_inplace(plane: &mut Vec<f32>, width: usize, height: usize) -> (usize, usize) {
    let new_w = width / 2;
    let new_h = height / 2;

    // In-place: output index y*new_w+x < source index (2y)*width+(2x) for all y,x.
    // Proof: 2y*width + 2x = 2*(y*width + x) >= 2*(y*new_w + x) > y*new_w + x.
    for y in 0..new_h {
        let sy = y * 2;
        for x in 0..new_w {
            let sx = x * 2;
            let a = plane[sy * width + sx];
            let b = plane[sy * width + sx + 1];
            let c = plane[(sy + 1) * width + sx];
            let d = plane[(sy + 1) * width + sx + 1];
            plane[y * new_w + x] = (a + b + c + d) * 0.25;
        }
    }
    plane.truncate(new_w * new_h);
    (new_w, new_h)
}
