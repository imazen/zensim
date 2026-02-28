/// Bit-manipulation distortion functions for testing error classification.
/// These produce errors that ImageMagick can't express: truncation, premul math, rounding.

/// Truncation: floor(v / 2) * 2. ~50% of pixels off by 1, max delta = 1.
pub fn truncate_lsb(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    pixels.iter().map(|p| [p[0] & 0xFE, p[1] & 0xFE, p[2] & 0xFE]).collect()
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
