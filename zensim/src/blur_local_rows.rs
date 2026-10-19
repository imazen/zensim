use super::*;

// Row windows of the existing canonical radius-five vertical kernel.
// Original full-plane kernels stay unchanged; cached pair-tree nodes and
// scalar tails retain exactly the same operands and rounding.

pub(crate) fn box_blur_v_local_rows(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    window: core::ops::Range<usize>,
) {
    assert!(window.start <= window.end && window.end <= height);
    assert_eq!(output.len(), width * window.len());
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(token) = <archmage::X64V4xToken as archmage::SimdToken>::summon() {
        box_blur_v_local11_rows_wide_v4x(token, input, output, width, height, 5, window);
        return;
    }
    incant!(
        box_blur_v_local11_rows(input, output, width, height, 5, window),
        [v4x, v4, v3, scalar]
    );
}

fn box_blur_v_local11_rows_scalar(
    _token: archmage::ScalarToken,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    window: core::ops::Range<usize>,
) {
    debug_assert_eq!(radius, 5);
    if width == 0 || height == 0 || window.is_empty() {
        return;
    }
    for (out_y, y) in window.clone().enumerate() {
        let rows: [&[f32]; 11] = std::array::from_fn(|k| {
            let start = crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * width;
            &input[start..start + width]
        });
        for x in 0..width {
            let taps: [f32; 11] = std::array::from_fn(|k| rows[k][x]);
            output[out_y * width + x] = local_sum_taps(&taps) * (1.0 / 11.0);
        }
    }
}

#[magetypes(define(f32x8), v4x, v4, v3, -scalar)]
fn box_blur_v_local11_rows(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    window: core::ops::Range<usize>,
) {
    debug_assert_eq!(radius, 5);
    if width == 0 || height == 0 || window.is_empty() {
        return;
    }
    let inv = f32x8::splat_t(token, 1.0 / 11.0);
    // Cache the frozen tree's pair and quadruple nodes, never an output
    // recurrence. Each node has the same two operands and rounding as before.
    // Eight vector columns bound scratch and keep adjacent rows in one tile.
    let blocks = width / 8;
    for tile in (0..blocks).step_by(8) {
        let count = (blocks - tile).min(8);
        let mut pairs = [[f32x8::zero_t(token); 8]; 16];
        let mut quads = pairs;
        for k in 0..9 {
            let a = crate::featcanon::tap_mirror(window.start as isize + k as isize - 5, height)
                * width;
            let b = crate::featcanon::tap_mirror(window.start as isize + k as isize - 4, height)
                * width;
            for c in 0..count {
                let x = (tile + c) * 8;
                pairs[k][c] = f32x8::load_t(token, input[a + x..a + x + 8].try_into().unwrap())
                    + f32x8::load_t(token, input[b + x..b + x + 8].try_into().unwrap());
            }
        }
        for k in 0..7 {
            for c in 0..count {
                quads[k][c] = pairs[k][c] + pairs[k + 2][c];
            }
        }
        for (out_y, y) in window.clone().enumerate() {
            let a = crate::featcanon::tap_mirror(y as isize + 4, height) * width;
            let b = crate::featcanon::tap_mirror(y as isize + 5, height) * width;
            for c in 0..count {
                let x = (tile + c) * 8;
                let last = f32x8::load_t(token, input[b + x..b + x + 8].try_into().unwrap());
                pairs[(out_y + 9) & 15][c] =
                    f32x8::load_t(token, input[a + x..a + x + 8].try_into().unwrap()) + last;
                quads[(out_y + 7) & 15][c] =
                    pairs[(out_y + 7) & 15][c] + pairs[(out_y + 9) & 15][c];
                let sum = ((quads[out_y & 15][c] + quads[(out_y + 4) & 15][c])
                    + pairs[(out_y + 8) & 15][c])
                    + last;
                (sum * inv).store(
                    (&mut output[out_y * width + x..out_y * width + x + 8])
                        .try_into()
                        .unwrap(),
                );
            }
        }
    }
    // The incomplete vector uses the unchanged per-output scalar tree.
    if blocks * 8 == width {
        return;
    }
    for (out_y, y) in window.clone().enumerate() {
        let rows: [usize; 11] = std::array::from_fn(|k| {
            crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * width
        });
        for x in blocks * 8..width {
            let taps: [f32; 11] = std::array::from_fn(|k| input[rows[k] + x]);
            output[out_y * width + x] = local_sum_taps(&taps) * (1.0 / 11.0);
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[magetypes(define(f32x16), v4x, -v4, -v3, -scalar)]
fn box_blur_v_local11_rows_wide(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    window: core::ops::Range<usize>,
) {
    debug_assert_eq!(radius, 5);
    if width == 0 || height == 0 || window.is_empty() {
        return;
    }
    let inv = f32x16::splat_t(token, 1.0 / 11.0);
    // Cache the frozen tree's pair and quadruple nodes, never an output
    // recurrence. Each node has the same two operands and rounding as before.
    // Eight vector columns bound scratch and keep adjacent rows in one tile.
    let blocks = width / 16;
    for tile in (0..blocks).step_by(8) {
        let count = (blocks - tile).min(8);
        let mut pairs = [[f32x16::zero_t(token); 8]; 16];
        let mut quads = pairs;
        for k in 0..9 {
            let a = crate::featcanon::tap_mirror(window.start as isize + k as isize - 5, height)
                * width;
            let b = crate::featcanon::tap_mirror(window.start as isize + k as isize - 4, height)
                * width;
            for c in 0..count {
                let x = (tile + c) * 16;
                pairs[k][c] = f32x16::load_t(token, input[a + x..a + x + 16].try_into().unwrap())
                    + f32x16::load_t(token, input[b + x..b + x + 16].try_into().unwrap());
            }
        }
        for k in 0..7 {
            for c in 0..count {
                quads[k][c] = pairs[k][c] + pairs[k + 2][c];
            }
        }
        for (out_y, y) in window.clone().enumerate() {
            let a = crate::featcanon::tap_mirror(y as isize + 4, height) * width;
            let b = crate::featcanon::tap_mirror(y as isize + 5, height) * width;
            for c in 0..count {
                let x = (tile + c) * 16;
                let last = f32x16::load_t(token, input[b + x..b + x + 16].try_into().unwrap());
                pairs[(out_y + 9) & 15][c] =
                    f32x16::load_t(token, input[a + x..a + x + 16].try_into().unwrap()) + last;
                quads[(out_y + 7) & 15][c] =
                    pairs[(out_y + 7) & 15][c] + pairs[(out_y + 9) & 15][c];
                let sum = ((quads[out_y & 15][c] + quads[(out_y + 4) & 15][c])
                    + pairs[(out_y + 8) & 15][c])
                    + last;
                (sum * inv).store(
                    (&mut output[out_y * width + x..out_y * width + x + 16])
                        .try_into()
                        .unwrap(),
                );
            }
        }
    }
    // The incomplete vector uses the unchanged per-output scalar tree.
    if blocks * 16 == width {
        return;
    }
    for (out_y, y) in window.clone().enumerate() {
        let rows: [usize; 11] = std::array::from_fn(|k| {
            crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * width
        });
        for x in blocks * 16..width {
            let taps: [f32; 11] = std::array::from_fn(|k| input[rows[k] + x]);
            output[out_y * width + x] = local_sum_taps(&taps) * (1.0 / 11.0);
        }
    }
}

#[cfg(test)]
mod local_row_window_tests {
    use super::*;

    #[test]
    fn requested_rows_match_the_original_full_plane_bits() {
        let mut state = 0x294a_8bd7_u32;
        for width in [1, 7, 8, 15, 16, 17, 63, 64, 97] {
            for height in [1, 7, 64, 151] {
                let input: Vec<f32> = (0..width * height)
                    .map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        (state as i32 as f32) / (i32::MAX as f32)
                    })
                    .collect();
                let mut full = vec![0.0; width * height];
                box_blur_v_local(&input, &mut full, width, height, 5);
                for (start, end) in [
                    (0, height),
                    (0, 1),
                    (height / 3, (2 * height / 3).max(height / 3)),
                    (height - 1, height),
                    (height, height),
                ] {
                    let expected: Vec<_> = full[start * width..end * width]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    let mut out = vec![0.0; (end - start) * width];
                    box_blur_v_local_rows(&input, &mut out, width, height, start..end);
                    assert_eq!(
                        expected,
                        out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "dispatch {width}x{height}/{start}..{end}"
                    );
                    box_blur_v_local11_rows_scalar(
                        archmage::ScalarToken,
                        &input,
                        &mut out,
                        width,
                        height,
                        5,
                        start..end,
                    );
                    assert_eq!(
                        expected,
                        out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "scalar {width}x{height}/{start}..{end}"
                    );
                }
            }
        }
    }
}
