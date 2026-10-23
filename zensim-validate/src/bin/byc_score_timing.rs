//! Per-call scoring time of one bake on a fixed synthetic pair (by-construction perf prototype).
//!
//! Exploration record: `benchmarks/byconstruction_2026-10-22.md`. Builds a deterministic
//! textured reference of the requested size and a distorted copy (small noise), warms up, then
//! times `BakeScorer::compute` and prints one JSON line: median and quartiles in milliseconds,
//! plus the score bits (to confirm both arms score the same pair). The caller alternates two arms
//! in paired rounds (two builds, or one build under two `ZENSIM_RING_FORM` values).
//!
//! Usage: byc_score_timing --bake PATH --size N [--iters 30] [--warmup 5]
use std::time::Instant;

use zensim::{BakeScorer, RgbSlice};
use zensim_validate::bake_runtime::post_mode_params;

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut bake, mut size, mut iters, mut warmup) = (None, 1024usize, 30usize, 5usize);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bake = args.next(),
            "--size" => size = args.next().and_then(|v| v.parse().ok()).expect("--size N"),
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).expect("--iters N"),
            "--warmup" => {
                warmup = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--warmup N")
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let bake = bake.expect("--bake PATH");
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut src = Vec::with_capacity(size * size);
    let mut dst = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = (x as f64, y as f64);
            let base = [
                128.0
                    + 60.0 * (fx * 0.031).sin() * (fy * 0.017).cos()
                    + 30.0 * (fx * 0.23 + fy * 0.11).sin(),
                120.0 + 50.0 * (fy * 0.027).sin() + 25.0 * (fx * 0.19).cos(),
                110.0 + 40.0 * ((fx + fy) * 0.021).sin() + 20.0 * (fx * fy * 0.0007).cos(),
            ];
            let px = base.map(|v| v.clamp(0.0, 255.0) as u8);
            let noisy = base.map(|v| (v + 6.0 * (rnd() - 0.5)).clamp(0.0, 255.0) as u8);
            src.push(px);
            dst.push(noisy);
        }
    }
    let model = zenpredict::Model::from_bytes(&std::fs::read(&bake).expect("read bake"))
        .expect("parse bake");
    let params = post_mode_params("raw").expect("raw post");
    let mut scorer = BakeScorer::new(&model)
        .expect("score metadata")
        .with_score_disposition(&params)
        .expect("disposition");
    let (s, d) = (
        RgbSlice::new(&src, size, size),
        RgbSlice::new(&dst, size, size),
    );
    let mut score = 0.0;
    for _ in 0..warmup {
        score = scorer.compute(&s, &d, None).expect("score").score();
    }
    let mut ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let r = scorer.compute(&s, &d, None).expect("score");
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        score = r.score();
    }
    ms.sort_by(f64::total_cmp);
    let q = |p: f64| ms[((ms.len() - 1) as f64 * p).round() as usize];
    println!(
        "{{\"size\":{size},\"iters\":{iters},\"median_ms\":{:.4},\"p25_ms\":{:.4},\"p75_ms\":{:.4},\"score\":{score:.6},\"ring_form\":\"{}\"}}",
        q(0.5),
        q(0.25),
        q(0.75),
        std::env::var("ZENSIM_RING_FORM").unwrap_or_else(|_| "raw".into())
    );
}
