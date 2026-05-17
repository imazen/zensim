//! Equivalence tests: zensim's sRGB math vs linear-srgb crate.
//!
//! Verifies that switching to linear-srgb won't change metric output.

/// Reproduce zensim's IEC textbook sRGB linearization (from color.rs)
fn zensim_srgb_u8_to_linear(v: u8) -> f32 {
    let s = v as f64 / 255.0;
    let linear = if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    };
    linear as f32
}

fn zensim_srgb_u16_to_linear(v: u16) -> f32 {
    let s = v as f64 / 65535.0;
    let linear = if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    };
    linear as f32
}

#[test]
fn u8_lut_equivalence_exhaustive() {
    let mut max_diff: f32 = 0.0;
    let mut worst = 0u8;
    for i in 0..=255u8 {
        let zensim_val = zensim_srgb_u8_to_linear(i);
        let lsrgb_val = linear_srgb::default::srgb_u8_to_linear(i);
        let diff = (zensim_val - lsrgb_val).abs();
        if diff > max_diff {
            max_diff = diff;
            worst = i;
        }
    }
    eprintln!(
        "u8 max diff: {max_diff:.2e} at u8={worst} (zensim={:.10}, linear-srgb={:.10})",
        zensim_srgb_u8_to_linear(worst),
        linear_srgb::default::srgb_u8_to_linear(worst),
    );
    assert!(max_diff < 1e-5, "u8 max diff {max_diff:.2e} too large");
}

#[test]
fn u16_exhaustive_equivalence() {
    let mut max_diff: f64 = 0.0;
    let mut worst = 0u16;
    for i in 0..=65535u16 {
        let zensim_val = zensim_srgb_u16_to_linear(i) as f64;
        let lsrgb_val = linear_srgb::default::srgb_u16_to_linear(i) as f64;
        let diff = (zensim_val - lsrgb_val).abs();
        if diff > max_diff {
            max_diff = diff;
            worst = i;
        }
    }
    eprintln!(
        "u16 max diff: {max_diff:.2e} at u16={worst} (zensim={:.10}, linear-srgb={:.10})",
        zensim_srgb_u16_to_linear(worst),
        linear_srgb::default::srgb_u16_to_linear(worst),
    );
    assert!(max_diff < 1e-4, "u16 max diff {max_diff:.2e} too large");
}

#[test]
fn linear_srgb_has_slice_apis() {
    // Verify linear-srgb provides the batch APIs we'd use
    let srgb_bytes: Vec<u8> = (0..=255).collect();
    let mut linear = vec![0.0f32; 256];
    linear_srgb::default::srgb_u8_to_linear_slice(&srgb_bytes, &mut linear);
    assert!((linear[0] - 0.0).abs() < 1e-7);
    assert!((linear[255] - 1.0).abs() < 1e-6);

    // u16 slice
    let u16_vals: Vec<u16> = vec![0, 32768, 65535];
    let mut linear16 = vec![0.0f32; 3];
    linear_srgb::default::srgb_u16_to_linear_slice(&u16_vals, &mut linear16);
    assert!((linear16[0] - 0.0).abs() < 1e-7);
    assert!((linear16[2] - 1.0).abs() < 1e-6);

    // RGBA slice (alpha preserved)
    let rgba_bytes: Vec<u8> = vec![128, 128, 128, 200];
    let mut rgba_linear = vec![0.0f32; 4];
    linear_srgb::default::srgb_u8_to_linear_rgba_slice(&rgba_bytes, &mut rgba_linear);
    // Alpha should be passed through as a/255, not sRGB-decoded
    assert!((rgba_linear[3] - 200.0 / 255.0).abs() < 1e-5);
}
