// Standalone bench: run compute_with_ref_streaming_strips on a synthetic pair
use zensim::{Zensim, ZensimProfile};
use zensim::source::RgbSlice;

fn make_pair(w: usize, h: usize, seed: u32) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut s = seed.wrapping_mul(2654435761);
    let mut next = || { s = s.wrapping_mul(2654435761).wrapping_add(1); (s >> 16) as u8 };
    let n = w * h;
    let mut src = Vec::with_capacity(n);
    let mut dst = Vec::with_capacity(n);
    for _ in 0..n {
        let r = next(); let g = next(); let b = next();
        src.push([r, g, b]);
        dst.push([r.wrapping_add(3), g.wrapping_add(3), b.wrapping_add(3)]);
    }
    (src, dst)
}

fn main() {
    let z = Zensim::new(ZensimProfile::PreviewV0_2);
    let (w, h) = (1920, 1080);
    let (src, dst) = make_pair(w, h, 42);
    let src_img = RgbSlice::new(&src, w, h);
    let dst_img = RgbSlice::new(&dst, w, h);

    // Repeat to amortize startup
    for _ in 0..20 {
        let _ = z.compute_streaming_strips(&src_img, &dst_img, 256, 128).unwrap();
    }
}
