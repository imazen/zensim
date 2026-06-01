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
        ("v0_3 (latest)", ZensimProfile::A),
        (
            "v0_5_balanced",
            zensim_experimental::preview_v0_5_balanced(),
        ),
        (
            "v0_5_compression",
            zensim_experimental::preview_v0_5_compression(),
        ),
        ("v0_5_tuner", zensim_experimental::preview_v0_5_tuner()),
    ] {
        let z = Zensim::new(p);
        let r = z.compute(&s, &d).unwrap();
        println!("{name}: score={:.4}", r.score());
    }
}
