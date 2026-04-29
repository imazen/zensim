//! Correctness check for the σ-elimination streaming SSIM kernel.
//!
//! Build with `--features zensim/zwe-streaming-ssim` to test the streaming
//! variant, without it for baseline. Run both and pipe through diff
//! externally; the program just emits scores.

use zensim::{RgbSlice, Zensim, ZensimProfile};

fn make_test_images(width: usize, height: usize, seed: u32) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = width * height;
    let mut s: u32 = seed.wrapping_mul(2654435761);
    let mut rng = || {
        s = s.wrapping_mul(2654435761).wrapping_add(1);
        (s >> 8) as u8
    };
    let src: Vec<[u8; 3]> = (0..n).map(|_| [rng(), rng(), rng()]).collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| {
            [
                r.wrapping_add(rng() / 8),
                g.wrapping_add(rng() / 8),
                b.wrapping_add(rng() / 8),
            ]
        })
        .collect();
    (src, dst)
}

fn run(label: &str, w: usize, h: usize, seed: u32) {
    let (src, dst) = make_test_images(w, h, seed);
    let z = Zensim::new(ZensimProfile::latest());
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    let r = z.compute(&s, &d).unwrap();
    let feats = r.features();
    let n = feats.len();
    let head: Vec<String> = feats.iter().take(8).map(|f| format!("{:.10}", f)).collect();
    let tail: Vec<String> = feats
        .iter()
        .skip(n - 4)
        .map(|f| format!("{:.10}", f))
        .collect();
    println!(
        "{label} {w}x{h} seed={seed}  score={:.10}  dist={:.10}",
        r.score(),
        r.raw_distance()
    );
    println!("  feats[0..8]:  {}", head.join(" "));
    println!("  feats[{}..]:  {}", n - 4, tail.join(" "));
}

fn main() {
    for &(label, w, h) in &[
        ("tiny", 64, 64),
        ("small", 128, 96),
        ("med", 320, 240),
        ("medplus", 800, 600),
        ("hd", 1280, 720),
    ] {
        for seed in [1u32, 17, 999] {
            run(label, w, h, seed);
        }
    }
}
