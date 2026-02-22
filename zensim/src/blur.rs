//! Fast box blur cascade: 3-pass box filter approximates Gaussian blur.
//!
//! Unlike recursive Gaussian IIR (used in ssimulacra2, ~60-70% of runtime),
//! box blur is O(1) per pixel using running sums, regardless of radius.
//! Three passes of box blur converge to Gaussian (central limit theorem).
//!
//! For ssimulacra2-equivalent sigma ≈ 1.5, we use radius 2 (5-pixel kernel).
//! Three passes of radius-2 box filter ≈ Gaussian with sigma ≈ sqrt(3 * (2*2+1)/12) ≈ 1.94

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;

/// Blur a single plane in-place using 3-pass box blur cascade.
/// `radius` controls the box filter half-width. Total kernel width = 2*radius + 1.
pub fn box_blur_3pass(plane: &mut [f32], width: usize, height: usize, radius: usize) {
    let mut temp = vec![0.0f32; plane.len()];
    // 3 passes: plane → temp → plane → temp, then copy back
    box_blur_single_pass(plane, &mut temp, width, height, radius);
    box_blur_single_pass(&temp, plane, width, height, radius);
    box_blur_single_pass(plane, &mut temp, width, height, radius);
    plane.copy_from_slice(&temp);
}

/// Blur into pre-allocated output buffer.
pub fn box_blur_3pass_into(
    input: &[f32],
    output: &mut [f32],
    temp: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    box_blur_single_pass(input, output, width, height, radius);
    box_blur_single_pass(output, temp, width, height, radius);
    box_blur_single_pass(temp, output, width, height, radius);
}

/// Single separable box blur pass (horizontal then vertical).
fn box_blur_single_pass(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    // Horizontal pass: input → output
    box_blur_horizontal(input, output, width, height, radius);

    // Vertical pass: output → output (in-place via temp column)
    // We need a temp buffer for the vertical pass to avoid aliasing
    let mut col_buf = vec![0.0f32; height];
    for x in 0..width {
        // Extract column
        for y in 0..height {
            col_buf[y] = output[y * width + x];
        }
        // Blur column
        let diam = 2 * radius + 1;
        let inv = 1.0 / diam as f32;
        let mut sum = 0.0f32;

        // Initialize sum for first pixel
        for i in 0..=radius.min(height - 1) {
            sum += col_buf[i];
        }
        // Mirror padding for top edge
        for i in 1..=radius {
            sum += col_buf[i.min(height - 1)];
        }

        for y in 0..height {
            output[y * width + x] = sum * inv;
            // Slide window
            let add_idx = y + radius + 1;
            let rem_idx = (y as isize) - (radius as isize);
            let add_val = col_buf[add_idx.min(height - 1)];
            let rem_val = col_buf[(rem_idx.max(0)) as usize];
            sum += add_val - rem_val;
        }
    }
}

/// Horizontal box blur using running sum. O(1) per pixel.
fn box_blur_horizontal(input: &[f32], output: &mut [f32], width: usize, height: usize, radius: usize) {
    incant!(box_blur_h_inner(input, output, width, height, radius), [v3]);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn box_blur_h_inner_v3(
    _token: archmage::X64V3Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    // Horizontal blur is inherently serial per row (running sum),
    // but we can process multiple rows simultaneously via SIMD.
    // Process 8 rows at a time.
    let diam = 2 * radius + 1;
    let inv = 1.0 / diam as f32;
    let row_groups = height / 8;

    for group in 0..row_groups {
        let base_y = group * 8;
        let mut sums = [0.0f32; 8];

        // Initialize sums with mirrored padding
        for dy in 0..8 {
            let row = base_y + dy;
            let row_start = row * width;
            sums[dy] = 0.0;
            for i in 0..=radius.min(width - 1) {
                sums[dy] += input[row_start + i];
            }
            for i in 1..=radius {
                sums[dy] += input[row_start + i.min(width - 1)];
            }
        }

        // Slide across columns
        for x in 0..width {
            for dy in 0..8 {
                let row = base_y + dy;
                output[row * width + x] = sums[dy] * inv;

                let add_idx = (x + radius + 1).min(width - 1);
                let rem_idx = (x as isize - radius as isize).max(0) as usize;
                let row_start = row * width;
                sums[dy] += input[row_start + add_idx] - input[row_start + rem_idx];
            }
        }
    }

    // Remainder rows
    for row in (row_groups * 8)..height {
        let row_start = row * width;
        let diam = 2 * radius + 1;
        let inv = 1.0 / diam as f32;
        let mut sum = 0.0f32;

        for i in 0..=radius.min(width - 1) {
            sum += input[row_start + i];
        }
        for i in 1..=radius {
            sum += input[row_start + i.min(width - 1)];
        }

        for x in 0..width {
            output[row * width + x] = sum * inv;
            let add_idx = (x + radius + 1).min(width - 1);
            let rem_idx = (x as isize - radius as isize).max(0) as usize;
            sum += input[row_start + add_idx] - input[row_start + rem_idx];
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

    for row in 0..height {
        let row_start = row * width;
        let mut sum = 0.0f32;

        for i in 0..=radius.min(width - 1) {
            sum += input[row_start + i];
        }
        for i in 1..=radius {
            sum += input[row_start + i.min(width - 1)];
        }

        for x in 0..width {
            output[row * width + x] = sum * inv;
            let add_idx = (x + radius + 1).min(width - 1);
            let rem_idx = (x as isize - radius as isize).max(0) as usize;
            sum += input[row_start + add_idx] - input[row_start + rem_idx];
        }
    }
}

/// Downscale a plane by 2x in each dimension (average of 2x2 blocks).
pub fn downscale_2x(plane: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let new_w = width / 2;
    let new_h = height / 2;
    let mut out = vec![0.0f32; new_w * new_h];

    for y in 0..new_h {
        for x in 0..new_w {
            let sx = x * 2;
            let sy = y * 2;
            let a = plane[sy * width + sx];
            let b = plane[sy * width + sx + 1];
            let c = plane[(sy + 1) * width + sx];
            let d = plane[(sy + 1) * width + sx + 1];
            out[y * new_w + x] = (a + b + c + d) * 0.25;
        }
    }

    (out, new_w, new_h)
}
