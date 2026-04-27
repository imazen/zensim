//! Color type, named constants, and CSS-style constructors.

/// RGBA color (straight alpha — the `image` crate's storage convention).
pub type Color = [u8; 4];

pub const WHITE: Color = [255, 255, 255, 255];
pub const BLACK: Color = [0, 0, 0, 255];
pub const TRANSPARENT: Color = [0, 0, 0, 0];

/// Build an opaque RGB color.
pub const fn rgb(r: u8, g: u8, b: u8) -> Color {
    [r, g, b, 255]
}

/// Build an RGBA color.
pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    [r, g, b, a]
}

/// Parse a CSS-style hex color: `#RGB`, `#RRGGBB`, or `#RRGGBBAA`. The
/// leading `#` is optional. Panics at compile time (or runtime, if not
/// in a const context) on malformed input.
pub const fn hex(s: &str) -> Color {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let start = if len > 0 && bytes[0] == b'#' { 1 } else { 0 };
    let n = len - start;

    if n == 3 {
        let r = hex1(bytes[start]) * 17;
        let g = hex1(bytes[start + 1]) * 17;
        let b = hex1(bytes[start + 2]) * 17;
        [r, g, b, 255]
    } else if n == 6 {
        [
            hex2(bytes[start], bytes[start + 1]),
            hex2(bytes[start + 2], bytes[start + 3]),
            hex2(bytes[start + 4], bytes[start + 5]),
            255,
        ]
    } else if n == 8 {
        [
            hex2(bytes[start], bytes[start + 1]),
            hex2(bytes[start + 2], bytes[start + 3]),
            hex2(bytes[start + 4], bytes[start + 5]),
            hex2(bytes[start + 6], bytes[start + 7]),
        ]
    } else {
        panic!("invalid hex color literal — expected #RGB, #RRGGBB, or #RRGGBBAA")
    }
}

const fn hex2(hi: u8, lo: u8) -> u8 {
    hex1(hi) * 16 + hex1(lo)
}

const fn hex1(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => panic!("invalid hex digit"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_3_digit() {
        assert_eq!(hex("#abc"), [0xaa, 0xbb, 0xcc, 255]);
    }
    #[test]
    fn hex_6_digit() {
        assert_eq!(hex("#1a2b3c"), [0x1a, 0x2b, 0x3c, 255]);
    }
    #[test]
    fn hex_8_digit() {
        assert_eq!(hex("#11223344"), [0x11, 0x22, 0x33, 0x44]);
    }
    #[test]
    fn hex_no_hash() {
        assert_eq!(hex("ff0080"), [0xff, 0x00, 0x80, 255]);
    }
    #[test]
    fn rgb_helper_is_opaque() {
        assert_eq!(rgb(10, 20, 30), [10, 20, 30, 255]);
    }
}
