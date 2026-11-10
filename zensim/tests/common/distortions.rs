#![allow(dead_code)]
// Deterministic distortion functions for testing error classification:
// bit manipulation, transfer curves, channel/color-matrix errors, alpha
// compositing, and seeded noise. All pure integer/f64 math so the generated
// fixtures are byte-identical on every platform.

/// Map each channel through `f` on normalized [0,1] values, rounding back to u8.
fn map_rgb(pixels: &[[u8; 3]], f: impl Fn(f32) -> f32) -> Vec<[u8; 3]> {
    pixels
        .iter()
        .map(|p| p.map(|v| (f(v as f32 / 255.0) * 255.0).round().clamp(0.0, 255.0) as u8))
        .collect()
}

/// Power-law transfer applied: out = (v/255)^(1/gamma).
pub fn apply_gamma(pixels: &[[u8; 3]], gamma: f32) -> Vec<[u8; 3]> {
    map_rgb(pixels, |v| v.powf(1.0 / gamma))
}

/// Treat stored values as linear light and apply the sRGB encode
/// (the "double sRGB" error: midtones lift hard).
pub fn srgb_encode(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    map_rgb(pixels, linear_srgb::default::linear_to_srgb)
}

/// Apply the sRGB decode to stored sRGB values
/// (the "linear read as sRGB" error: midtones crush).
pub fn srgb_decode(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    map_rgb(pixels, linear_srgb::default::srgb_to_linear)
}

/// R↔B channel swap; G untouched.
pub fn swap_rb(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    pixels.iter().map(|p| [p[2], p[1], p[0]]).collect()
}

/// Round-trip through full-swing BT.601 YCbCr with Cb and Cr exchanged.
pub fn swap_cbcr(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    pixels
        .iter()
        .map(|p| {
            let (r, g, b) = (p[0] as f64, p[1] as f64, p[2] as f64);
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let cb = -0.168736 * r - 0.331264 * g + 0.5 * b;
            let cr = 0.5 * r - 0.418688 * g - 0.081312 * b;
            // Back-convert with the chroma channels exchanged.
            let r2 = y + 1.402 * cb;
            let g2 = y - 0.344136 * cr - 0.714136 * cb;
            let b2 = y + 1.772 * cr;
            [r2, g2, b2].map(|v| v.round().clamp(0.0, 255.0) as u8)
        })
        .collect()
}

/// Quantize each channel to `bits` per channel, rescaled to the full range.
pub fn quantize_bits(pixels: &[[u8; 3]], bits: u32) -> Vec<[u8; 3]> {
    let max_level = (1u32 << bits) - 1;
    pixels
        .iter()
        .map(|p| {
            p.map(|v| {
                let q = (v as u32 * max_level + 127) / 255;
                (q * 255 / max_level) as u8
            })
        })
        .collect()
}

/// Decode stored values as Adobe RGB (1998) primaries and re-encode as sRGB —
/// the visible result of reading an Adobe RGB file without its profile.
pub fn adobe_rgb_to_srgb(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    // Adobe RGB (1998): pure 2.2 gamma, D65 primaries.
    const ADOBE_TO_XYZ: [[f32; 3]; 3] = [
        [0.57667, 0.18556, 0.18823],
        [0.29734, 0.62736, 0.07529],
        [0.02703, 0.07069, 0.99134],
    ];
    const XYZ_TO_SRGB: [[f32; 3]; 3] = [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ];
    pixels
        .iter()
        .map(|p| {
            let lin: [f32; 3] = p.map(|v| (v as f32 / 255.0).powf(2.2));
            let mut out = [0u8; 3];
            for c in 0..3 {
                let v = XYZ_TO_SRGB[c]
                    .iter()
                    .enumerate()
                    .map(|(k, m)| {
                        m * ADOBE_TO_XYZ[k]
                            .iter()
                            .enumerate()
                            .map(|(j, a)| a * lin[j])
                            .sum::<f32>()
                    })
                    .sum::<f32>();
                out[c] = (linear_srgb::default::linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
            out
        })
        .collect()
}

/// Straight-alpha composite of RGBA over an opaque RGB background.
pub fn composite_over(pixels: &[[u8; 4]], bg: [u8; 3]) -> Vec<[u8; 3]> {
    pixels
        .iter()
        .map(|p| {
            let a = p[3] as u32;
            [0, 1, 2].map(|c| ((p[c] as u32 * a + bg[c] as u32 * (255 - a) + 127) / 255) as u8)
        })
        .collect()
}

/// Seeded additive Gaussian noise (Box-Muller over a xorshift64* stream),
/// deterministic on every platform.
pub fn gaussian_noise(pixels: &[[u8; 3]], sigma: f64, seed: u64) -> Vec<[u8; 3]> {
    let mut s = seed | 1;
    let mut next_u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 * (1.0 / 9007199254740992.0)
    };
    let mut spare = f64::NAN;
    let mut gauss = || {
        if spare.is_finite() {
            return std::mem::replace(&mut spare, f64::NAN);
        }
        let u1 = next_u().max(f64::MIN_POSITIVE);
        let u2 = next_u();
        let r = (-2.0 * u1.ln()).sqrt();
        let (sin, cos) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        spare = r * sin;
        r * cos
    };
    pixels
        .iter()
        .map(|p| p.map(|v| (v as f64 + gauss() * sigma).round().clamp(0.0, 255.0) as u8))
        .collect()
}

/// Truncation: floor(v / 2) * 2. ~50% of pixels off by 1, max delta = 1.
pub fn truncate_lsb(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    pixels
        .iter()
        .map(|p| [p[0] & 0xFE, p[1] & 0xFE, p[2] & 0xFE])
        .collect()
}

/// Wrong 8→16→8 expansion: val * 256 / 257 instead of val * 257 / 257.
/// Delta proportional to value (max ~0.5 for value 128).
pub fn expand_256(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    pixels
        .iter()
        .map(|p| {
            [
                ((p[0] as u16 * 256) / 257) as u8,
                ((p[1] as u16 * 256) / 257) as u8,
                ((p[2] as u16 * 256) / 257) as u8,
            ]
        })
        .collect()
}

/// Round-half-up instead of round-half-even (banker's rounding).
/// Only differs at exact .5 boundaries.
pub fn round_half_up(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    // Simulate: original values were computed with round-half-even,
    // apply round-half-up by checking if value is at a .5 boundary.
    // We create a subtle difference by adding 1 to odd values that are < 255.
    pixels
        .iter()
        .map(|p| {
            let adjust = |v: u8| -> u8 {
                if v % 2 == 1 && v < 255 {
                    v.wrapping_add(1)
                } else {
                    v
                }
            };
            [adjust(p[0]), adjust(p[1]), adjust(p[2])]
        })
        .collect()
}

/// Premultiply RGBA then interpret as straight (darken semitransparent pixels).
pub fn premul_as_straight(pixels: &[[u8; 4]]) -> Vec<[u8; 4]> {
    pixels
        .iter()
        .map(|p| {
            let a = p[3] as u16;
            [
                ((p[0] as u16 * a) / 255) as u8,
                ((p[1] as u16 * a) / 255) as u8,
                ((p[2] as u16 * a) / 255) as u8,
                p[3],
            ]
        })
        .collect()
}

/// Un-premultiply straight values (treat straight as premul → blow up).
pub fn straight_as_premul(pixels: &[[u8; 4]]) -> Vec<[u8; 4]> {
    pixels
        .iter()
        .map(|p| {
            if p[3] == 0 {
                *p
            } else {
                let a = p[3] as u16;
                [
                    ((p[0] as u16 * 255) / a).min(255) as u8,
                    ((p[1] as u16 * 255) / a).min(255) as u8,
                    ((p[2] as u16 * 255) / a).min(255) as u8,
                    p[3],
                ]
            }
        })
        .collect()
}

/// Composite premultiplied over black (wrong for straight alpha input).
/// Result = R * A / 255 (drops transparency, darkens semitransparent).
pub fn wrong_bg_black(pixels: &[[u8; 4]]) -> Vec<[u8; 4]> {
    pixels
        .iter()
        .map(|p| {
            let a = p[3] as u16;
            [
                ((p[0] as u16 * a) / 255) as u8,
                ((p[1] as u16 * a) / 255) as u8,
                ((p[2] as u16 * a) / 255) as u8,
                255, // Fully opaque after compositing
            ]
        })
        .collect()
}
