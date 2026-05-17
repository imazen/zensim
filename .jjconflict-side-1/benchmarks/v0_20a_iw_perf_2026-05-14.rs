use std::time::Instant;
use zensim::{ZensimConfig, compute_zensim_with_config};

fn make_cfg(extended: bool, iw: bool) -> ZensimConfig {
    let mut c = ZensimConfig::default();
    c.compute_all_features = true;
    c.extended_features = extended;
    c.compute_iw_features = iw;
    c.iw_strength = 4.0;
    c.extended_masking_strength = 4.0;
    c
}

fn run_one(name: &str, src: &[[u8; 3]], dst: &[[u8; 3]], w: usize, h: usize, cfg: ZensimConfig, iters: usize) {
    for _ in 0..3 {
        let r = compute_zensim_with_config(src, dst, w, h, cfg.clone()).unwrap();
        std::hint::black_box(r);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let r = compute_zensim_with_config(std::hint::black_box(src), std::hint::black_box(dst), w, h, cfg.clone()).unwrap();
        std::hint::black_box(r);
    }
    let per_call_us = t0.elapsed().as_micros() as f64 / iters as f64;
    let nf = compute_zensim_with_config(src, dst, w, h, cfg).unwrap().features().len();
    println!("{:>32}  {:7.0} us/call  ({} features, {} iters)", name, per_call_us, nf, iters);
}

fn main() {
    let width: usize = 512;
    let height: usize = 512;
    let n = width * height;
    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = (i % width) as u8;
            let y = (i / width) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| [r.saturating_add(10), g.saturating_add(5), b])
        .collect();

    let iters = 50;
    let cfgs = [
        ("OFF (basic+peaks, 228)",      make_cfg(false, false)),
        ("extended only (300, masked)", make_cfg(true,  false)),
        ("iw only (300, IW)",           make_cfg(false, true)),
        ("BOTH (372, masked+IW)",       make_cfg(true,  true)),
    ];
    println!("Interleaved 3-round timing (512x512, 50 iters per cell):");
    for round in 0..3 {
        println!("--- round {} ---", round + 1);
        for (name, cfg) in &cfgs {
            run_one(name, &src, &dst, width, height, cfg.clone(), iters);
        }
    }
}
