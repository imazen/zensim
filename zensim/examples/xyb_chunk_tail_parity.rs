//! Diagnostic for the CLAUDE.md Known Bug "bulk sRGB→XYB gives different bits
//! for the same colour in the vector chunks and in the scalar remainder"
//! (2026-09-25).
//!
//! For every sRGB8 colour it converts nine copies with the conversion owner,
//! `srgb_to_positive_xyb_planar_into`. The first eight go through a vector
//! chunk and the ninth through the scalar remainder. It prints how many colours
//! give different bits, per XYB channel, on the tier this process dispatches
//! to. A consistent conversion prints 0.
//!
//! ```sh
//! cargo run --release -p zensim --example xyb_chunk_tail_parity
//! ```
use zensim::__bench_stages::srgb_to_positive_xyb_planar_into;

fn main() {
    let mut differing = [0u64; 3];
    let mut any_channel = 0u64;
    let mut max_ulp = 0u32;
    let (mut x, mut y, mut b) = ([0f32; 9], [0f32; 9], [0f32; 9]);
    for r in 0..=255u8 {
        for g in 0..=255u8 {
            for blue in 0..=255u8 {
                let pixels = [[r, g, blue]; 9];
                srgb_to_positive_xyb_planar_into(&pixels, &mut x, &mut y, &mut b);
                let mut hit = false;
                for (channel, plane) in [&x, &y, &b].into_iter().enumerate() {
                    // Positive XYB, so the bit distance is the ULP distance.
                    let ulp = plane[0].to_bits().abs_diff(plane[8].to_bits());
                    if ulp != 0 {
                        differing[channel] += 1;
                        max_ulp = max_ulp.max(ulp);
                        hit = true;
                    }
                }
                any_channel += u64::from(hit);
            }
        }
    }
    let total = 1u64 << 24;
    println!(
        "colours={total} differing={any_channel} ({:.3}%) per-channel X/Y/B={differing:?} max_ulp={max_ulp}",
        100.0 * any_channel as f64 / total as f64
    );
}
