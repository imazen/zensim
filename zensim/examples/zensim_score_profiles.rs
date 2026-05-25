//! Compare zensim score across multiple profiles for one (ref, dist) pair.
//!
//! Usage: zensim_score_profiles ref.png dist.png
use std::env;
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} ref.png dist.png", args[0]);
        std::process::exit(1);
    }
    let img1 = image::open(&args[1]).unwrap().to_rgb8();
    let img2 = image::open(&args[2]).unwrap().to_rgb8();
    let w = img1.width() as usize;
    let h = img1.height() as usize;
    let src: Vec<[u8; 3]> = img1.pixels().map(|p| p.0).collect();
    let dst: Vec<[u8; 3]> = img2.pixels().map(|p| p.0).collect();
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    for (name, p) in [
        ("v0_3 (latest)", ZensimProfile::PreviewV0_3),
        ("v0_5_balanced", ZensimProfile::PreviewV0_5Balanced),
        ("v0_5_compression", ZensimProfile::PreviewV0_5Compression),
        ("v0_5_tuner", ZensimProfile::PreviewV0_5Tuner),
    ] {
        let z = Zensim::new(p);
        let r = z.compute(&s, &d).unwrap();
        println!("{name}: score={:.4}", r.score());
    }
}
