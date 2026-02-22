//! Fast box blur cascade: 3-pass box filter approximates Gaussian blur.
//!
//! Unlike recursive Gaussian IIR (used in ssimulacra2, ~60-70% of runtime),
//! box blur is O(1) per pixel using running sums, regardless of radius.
//! Three passes of box blur converge to Gaussian (central limit theorem).

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;

/// Blur a single plane in-place using 3-pass box blur cascade.
#[allow(dead_code)]
pub fn box_blur_3pass(plane: &mut [f32], width: usize, height: usize, radius: usize) {
    let n = plane.len();
    let mut output = vec![0.0f32; n];
    let mut temp = vec![0.0f32; n];
    box_blur_3pass_into(plane, &mut output, &mut temp, width, height, radius);
    plane.copy_from_slice(&output);
}

/// Blur into pre-allocated output buffer. Uses temp as scratch.
/// temp2 is an additional scratch buffer for the vertical pass.
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

/// Vertical box blur: read from `src`, write to `dst`.
fn box_blur_v_from_copy(src: &[f32], dst: &mut [f32], width: usize, height: usize, radius: usize) {
    incant!(box_blur_v_copy_inner(src, dst, width, height, radius), [v3]);
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
            let idx = if i <= r { (r - i).min(height - 1) } else { (i - r).min(height - 1) };
            let base = idx * width + col_base;
            sum = sum + f32x8::from_array(token, src[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            let base = y * width + col_base;
            dst[base..base + 8].copy_from_slice(&(sum * inv_v).to_array());

            let add_idx = (y + r + 1).min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 { rem_i.unsigned_abs() } else { rem_i as usize };
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
            let idx = if i <= r { (r - i).min(height - 1) } else { (i - r).min(height - 1) };
            sum += src[idx * width + x];
        }

        for y in 0..height {
            dst[y * width + x] = sum * inv;
            let add_idx = (y + r + 1).min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 { rem_i.unsigned_abs() } else { rem_i as usize };
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
            let idx = if i <= r { (r - i).min(height - 1) } else { (i - r).min(height - 1) };
            sum += src[idx * width + x];
        }

        for y in 0..height {
            dst[y * width + x] = sum * inv;
            let add_idx = (y + r + 1).min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 { rem_i.unsigned_abs() } else { rem_i as usize };
            let rem_idx = rem_idx.min(height - 1);
            sum += src[add_idx * width + x] - src[rem_idx * width + x];
        }
    }
}

/// Horizontal box blur using running sum. O(1) per pixel.
fn box_blur_h(input: &[f32], output: &mut [f32], width: usize, height: usize, radius: usize) {
    let diam = 2 * radius + 1;
    let inv = 1.0 / diam as f32;
    let r = radius;

    for row in 0..height {
        let row_off = row * width;
        let inp = &input[row_off..row_off + width];
        let out = &mut output[row_off..row_off + width];

        // Initialize running sum with mirrored padding
        let mut sum = inp[0] * (r as f32 + 1.0);
        for i in 1..=r.min(width - 1) {
            sum += inp[i];
        }
        // Mirror the right side of the initial window
        for i in 1..=r {
            if i < width {
                sum += inp[i];
            } else {
                sum += inp[width - 1];
            }
        }
        // Actually, simpler mirror: left edge mirrors first pixel
        // Reset and redo properly
        sum = 0.0;
        for i in 0..diam {
            let idx = if i <= r {
                // Left mirror: clamp to 0
                let mirror_i = r as isize - i as isize;
                mirror_i.unsigned_abs().min(width - 1)
            } else {
                (i - r).min(width - 1)
            };
            sum += inp[idx];
        }

        for x in 0..width {
            out[x] = sum * inv;
            // Add right pixel, remove left pixel
            let add_idx = (x + r + 1).min(width - 1);
            let rem_i = x as isize - r as isize;
            let rem_idx = if rem_i < 0 { (-rem_i) as usize } else { rem_i as usize };
            let rem_idx = rem_idx.min(width - 1);
            sum += inp[add_idx] - inp[rem_idx];
        }
    }
}

/// Old in-place vertical blur — replaced by box_blur_v_copy_inner.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)]
fn box_blur_v_inner_v3(
    token: archmage::X64V3Token,
    data: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv = f32x8::splat(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    // Process 8 columns at a time
    for cg in 0..col_groups {
        let col_base = cg * 8;

        // Read column values into a local buffer for cache efficiency
        let mut col_buf: Vec<[f32; 8]> = Vec::with_capacity(height);
        for y in 0..height {
            let base = y * width + col_base;
            let chunk: [f32; 8] = data[base..base + 8].try_into().unwrap();
            col_buf.push(chunk);
        }

        // Initialize running sums
        let mut sum = f32x8::zero(token);
        for i in 0..diam {
            let idx = if i <= r {
                let mirror_i = r as isize - i as isize;
                mirror_i.unsigned_abs().min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            sum = sum + f32x8::from_array(token, col_buf[idx]);
        }

        // Process each row
        for y in 0..height {
            let result = sum * inv;
            let base = y * width + col_base;
            data[base..base + 8].copy_from_slice(&result.to_array());

            let add_idx = (y + r + 1).min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 { (-rem_i) as usize } else { rem_i as usize };
            let rem_idx = rem_idx.min(height - 1);
            sum = sum + f32x8::from_array(token, col_buf[add_idx])
                - f32x8::from_array(token, col_buf[rem_idx]);
        }
    }

    // Scalar remainder columns
    for x in (col_groups * 8)..width {
        let mut col_buf: Vec<f32> = (0..height).map(|y| data[y * width + x]).collect();

        let diam = 2 * radius + 1;
        let inv = 1.0 / diam as f32;
        let r = radius;

        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                let mirror_i = r as isize - i as isize;
                mirror_i.unsigned_abs().min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            sum += col_buf[idx];
        }

        for y in 0..height {
            let old = col_buf[y];
            col_buf[y] = sum * inv;
            let _ = old;

            let add_idx = (y + r + 1).min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 { (-rem_i) as usize } else { rem_i as usize };
            let rem_idx = rem_idx.min(height - 1);
            // Use original values (before we wrote results)
            sum += data[add_idx * width + x] - data[rem_idx * width + x];
        }

        // Write back
        for y in 0..height {
            data[y * width + x] = col_buf[y];
        }
    }
}

fn box_blur_v_inner_scalar(
    _token: archmage::ScalarToken,
    data: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let diam = 2 * radius + 1;
    let inv = 1.0 / diam as f32;
    let r = radius;

    for x in 0..width {
        let col_buf: Vec<f32> = (0..height).map(|y| data[y * width + x]).collect();

        let mut sum = 0.0f32;
        for i in 0..diam {
            let idx = if i <= r {
                let mirror_i = r as isize - i as isize;
                mirror_i.unsigned_abs().min(height - 1)
            } else {
                (i - r).min(height - 1)
            };
            sum += col_buf[idx];
        }

        for y in 0..height {
            data[y * width + x] = sum * inv;

            let add_idx = (y + r + 1).min(height - 1);
            let rem_i = y as isize - r as isize;
            let rem_idx = if rem_i < 0 { (-rem_i) as usize } else { rem_i as usize };
            let rem_idx = rem_idx.min(height - 1);
            sum += col_buf[add_idx] - col_buf[rem_idx];
        }
    }
}

/// Downscale a plane by 2x in each dimension (average of 2x2 blocks).
pub fn downscale_2x(plane: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let new_w = width / 2;
    let new_h = height / 2;
    let mut out = vec![0.0f32; new_w * new_h];

    incant!(downscale_2x_inner(plane, &mut out, width, new_w, new_h), [v3]);

    (out, new_w, new_h)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn downscale_2x_inner_v3(
    token: archmage::X64V3Token,
    input: &[f32],
    output: &mut [f32],
    in_width: usize,
    out_width: usize,
    out_height: usize,
) {
    let quarter = f32x8::splat(token, 0.25);

    for y in 0..out_height {
        let sy = y * 2;
        let chunks = out_width / 8;

        for chunk in 0..chunks {
            let ox = chunk * 8;
            let sx = ox * 2;

            // Load 16 pixels from two rows, interleave-sum
            let mut a_arr = [0.0f32; 8];
            let mut b_arr = [0.0f32; 8];
            let mut c_arr = [0.0f32; 8];
            let mut d_arr = [0.0f32; 8];
            for i in 0..8 {
                a_arr[i] = input[sy * in_width + sx + i * 2];
                b_arr[i] = input[sy * in_width + sx + i * 2 + 1];
                c_arr[i] = input[(sy + 1) * in_width + sx + i * 2];
                d_arr[i] = input[(sy + 1) * in_width + sx + i * 2 + 1];
            }
            let a = f32x8::from_array(token, a_arr);
            let b = f32x8::from_array(token, b_arr);
            let c = f32x8::from_array(token, c_arr);
            let d = f32x8::from_array(token, d_arr);
            let sum = (a + b + c + d) * quarter;
            output[y * out_width + ox..][..8].copy_from_slice(&sum.to_array());
        }

        // Scalar remainder
        for x in (chunks * 8)..out_width {
            let sx = x * 2;
            let a = input[sy * in_width + sx];
            let b = input[sy * in_width + sx + 1];
            let c = input[(sy + 1) * in_width + sx];
            let d = input[(sy + 1) * in_width + sx + 1];
            output[y * out_width + x] = (a + b + c + d) * 0.25;
        }
    }
}

fn downscale_2x_inner_scalar(
    _token: archmage::ScalarToken,
    input: &[f32],
    output: &mut [f32],
    in_width: usize,
    out_width: usize,
    out_height: usize,
) {
    for y in 0..out_height {
        let sy = y * 2;
        for x in 0..out_width {
            let sx = x * 2;
            let a = input[sy * in_width + sx];
            let b = input[sy * in_width + sx + 1];
            let c = input[(sy + 1) * in_width + sx];
            let d = input[(sy + 1) * in_width + sx + 1];
            output[y * out_width + x] = (a + b + c + d) * 0.25;
        }
    }
}
