//! Quick probe: generate 200 random distortion pairs spanning distance ~0.5 to ~50,
//! emit zensim scores. Run twice (cbrt_midp build vs cbrt_lowp build) and diff.
//!
//! Pairs cover a wide content × distortion grid:
//!   - 4 sizes (128, 256, 384, 512)
//!   - 50 seeds per size
//!   - distortion = saturating_add(noise(0..MAX), src) where MAX varies 4..96
//!     to cover near-identical → heavy-distortion range.

use zensim::{RgbSlice, Zensim, ZensimProfile};

fn main() {
    let z = Zensim::new(ZensimProfile::latest());
    let sizes = [(128usize, 128usize), (256, 256), (384, 256), (512, 384)];
    let mut count = 0;
    for &(w, h) in &sizes {
        let n = w * h;
        for seed in 0u32..50 {
            // Source: deterministic per-pixel pattern from seed.
            let mut s: u32 = seed.wrapping_mul(2654435761).wrapping_add(0x9E37);
            let src: Vec<[u8; 3]> = (0..n)
                .map(|_| {
                    s = s.wrapping_mul(2654435761).wrapping_add(1);
                    let r = (s >> 8) as u8;
                    s = s.wrapping_mul(2654435761).wrapping_add(1);
                    let g = (s >> 8) as u8;
                    s = s.wrapping_mul(2654435761).wrapping_add(1);
                    let b = (s >> 8) as u8;
                    [r, g, b]
                })
                .collect();

            // Vary distortion magnitude: 4, 8, 16, 32, 64, 96.
            for mag in [4u32, 8, 16, 32, 64, 96] {
                let mut t: u32 = (seed ^ mag).wrapping_mul(2654435761);
                let dst: Vec<[u8; 3]> = src
                    .iter()
                    .map(|&[r, g, b]| {
                        t = t.wrapping_mul(2654435761).wrapping_add(1);
                        let dr = ((t >> 8) % mag) as u8;
                        t = t.wrapping_mul(2654435761).wrapping_add(1);
                        let dg = ((t >> 8) % mag) as u8;
                        t = t.wrapping_mul(2654435761).wrapping_add(1);
                        let db = ((t >> 8) % mag) as u8;
                        [r.saturating_add(dr), g.saturating_add(dg), b.saturating_add(db)]
                    })
                    .collect();

                let s_view = RgbSlice::new(&src, w, h);
                let d_view = RgbSlice::new(&dst, w, h);
                let r = z.compute(&s_view, &d_view).unwrap();
                println!(
                    "{w} {h} {seed} {mag} {:.10} {:.10}",
                    r.score(),
                    r.raw_distance()
                );
                count += 1;
                if count >= 200 {
                    return;
                }
            }
        }
    }
}
