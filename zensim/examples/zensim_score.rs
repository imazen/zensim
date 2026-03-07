use std::env;
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} original.png distorted.png", args[0]);
        std::process::exit(1);
    }

    let img1 = image::open(&args[1]).unwrap().to_rgb8();
    let img2 = image::open(&args[2]).unwrap().to_rgb8();

    let w = img1.width() as usize;
    let h = img1.height() as usize;

    let src: Vec<[u8; 3]> = img1.pixels().map(|p| p.0).collect();
    let dst: Vec<[u8; 3]> = img2.pixels().map(|p| p.0).collect();

    let z = Zensim::new(ZensimProfile::latest());
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    let result = z.compute(&s, &d).unwrap();
    println!("{:.8}", result.score());
}
