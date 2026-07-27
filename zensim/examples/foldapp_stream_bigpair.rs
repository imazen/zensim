// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! Large-synthetic-pair memory harness for the STREAMING folded+append
//! walk (G-RAM evidence for `benchmarks/streaming_foldapp_*`): generates
//! a deterministic W×H pair (the `tests/streaming_strips.rs` `make_pair`
//! pattern) and runs `compute_folded720_append_features_streaming` once.
//! Run under heaptrack to record peak heap at sizes where the
//! materialized path would allocate gigabytes:
//!
//! ```sh
//! cargo build --release -p zensim --features feature-regime-v2 \
//!   --example foldapp_stream_bigpair
//! heaptrack target/release/examples/foldapp_stream_bigpair 8000 10000
//! # ZENSIM_BIGPAIR_MODE=materialized compares the pre-C5 path (careful:
//! # allocates O(image) — multiple GB at 80 MP).
//! ```

use zensim::feature_v2::{V2NewFeatureToggles, V2Scratch};
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Deterministic procedural content — `tests/streaming_strips.rs`'s
/// `make_pair` shape (mode-5 hash variant), cheap to generate.
fn make_pair(w: usize, h: usize, seed: u32) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let mut src = vec![[0u8, 0, 0]; n];
    let mut dst = vec![[0u8, 0, 0]; n];
    let h_fn = |a: u32, b: u32| -> u8 {
        let mut x = a
            .wrapping_mul(0x6C8E_9CF7)
            .wrapping_add(b.wrapping_mul(0x9E37_79B9))
            .wrapping_add(seed);
        x ^= x >> 16;
        x = x.wrapping_mul(0x85EB_CA6B);
        x ^= x >> 13;
        (x & 0xFF) as u8
    };
    for y in 0..h {
        for x in 0..w {
            let r = h_fn(x as u32 / 4, y as u32 / 4);
            let g = h_fn(x as u32 / 2, y as u32 / 8);
            let b = h_fn(x as u32 / 8, y as u32 / 4);
            src[y * w + x] = [r, g, b];
            dst[y * w + x] = [
                r.saturating_add(2),
                g.saturating_sub(1),
                b.saturating_add(3),
            ];
        }
    }
    (src, dst)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let w: usize = args.first().map(|s| s.parse().unwrap()).unwrap_or(8000);
    let h: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(10000);
    eprintln!("generating {w}x{h} ({:.1} MP) pair...", (w * h) as f64 / 1e6);
    let t0 = std::time::Instant::now();
    let (src, dst) = make_pair(w, h, 1);
    eprintln!("generated in {:.2}s", t0.elapsed().as_secs_f64());

    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    let s_img = RgbSlice::new(&src, w, h);
    let d_img = RgbSlice::new(&dst, w, h);
    let mut scratch = V2Scratch::new();
    let mode = std::env::var("ZENSIM_BIGPAIR_MODE").unwrap_or_else(|_| "streaming".into());
    let t1 = std::time::Instant::now();
    let r = match mode.as_str() {
        "materialized" => z
            .compute_folded720_append_features(&s_img, &d_img)
            .expect("materialized foldapp"),
        _ => z
            .compute_folded720_append_features_streaming(
                &s_img,
                &d_img,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .expect("streaming foldapp"),
    };
    eprintln!(
        "{mode} foldapp: {:.2}s, {} features, f0={:.6} f923={:.6}",
        t1.elapsed().as_secs_f64(),
        r.features().len(),
        r.features()[0],
        r.features()[923],
    );
}
