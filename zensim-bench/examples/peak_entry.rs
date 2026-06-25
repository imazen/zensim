//! Single-op peak-memory driver for heaptrack: isolates ONE zensim call per
//! process so `heaptrack <bin> <entry> <w> <h>` reports a clean process peak.
//!
//! Resolves the contradiction between commit 4bb5febf ("CPU zensim 372-feature
//! extraction is the per-image memory hog that OOMs large frames") and the code
//! (`compute_extended_features` funnels through the SAME streaming path as the
//! score — `metric.rs:2200` → `streaming::compute_zensim_streaming`). The
//! existing heaptrack TSV (refresh_2026-05-28) only measured the 228 `compute`
//! score, never the 372/WithIw feature path. This driver measures both.
//!
//! Replicates the sweep's exact feature call (zenmetrics
//! `crates/zenmetrics-cli/src/metrics/zensim.rs:55`): StridedBytes + Srgb8Rgb +
//! `ZensimProfile::latest_preview()` + `compute_extended_features`.
//!
//! Build:  cargo build --release --manifest-path zensim-bench/Cargo.toml --example peak_entry
//! Run:    heaptrack -o /tmp/ht ./target/release/examples/peak_entry <score|ext372|strip> <w> <h>
//!
//! entries:
//!   score  — `Zensim::compute`                       (228 streaming score; the cpu_profile baseline)
//!   ext372 — `Zensim::compute_extended_features`     (372/WithIw; the sweep --feature-output path)
//!   strip  — `Zensim::compute_streaming_strips_default` (228 score via strip aggregation; the >16MP path)

use zensim::{PixelFormat, StridedBytes, Zensim, ZensimProfile};

fn build_rgb(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let n = w * h;
    let mut src = vec![0u8; n * 3];
    let mut dst = vec![0u8; n * 3];
    for i in 0..n {
        let x = ((i % w) * 255 / w) as u8;
        let y = ((i / w) * 255 / h) as u8;
        let b = x.wrapping_add(y);
        src[3 * i] = x;
        src[3 * i + 1] = y;
        src[3 * i + 2] = b;
        // Small, non-uniform perturbation so the pair isn't byte-identical
        // (which would short-circuit before any pyramid work).
        dst[3 * i] = x.saturating_add(8);
        dst[3 * i + 1] = y.saturating_add(4);
        dst[3 * i + 2] = b.saturating_add(2);
    }
    (src, dst)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: peak_entry <score|ext372|strip> <w> <h>");
        std::process::exit(2);
    }
    let entry = args[1].as_str();
    let w: usize = args[2].parse().expect("w");
    let h: usize = args[3].parse().expect("h");
    let stride = w * 3;

    let (src, dst) = build_rgb(w, h);
    let z = Zensim::new(ZensimProfile::latest_preview());
    let s = StridedBytes::try_new(&src, w, h, stride, PixelFormat::Srgb8Rgb).expect("src slice");
    let d = StridedBytes::try_new(&dst, w, h, stride, PixelFormat::Srgb8Rgb).expect("dst slice");

    let (score, nfeat) = match entry {
        "score" => {
            let r = z.compute(&s, &d).expect("compute");
            (r.score(), 0usize)
        }
        "ext372" => {
            let r = z.compute_extended_features(&s, &d).expect("compute_extended_features");
            let sc = r.score();
            let f = r.into_features();
            (sc, f.len())
        }
        "strip" => {
            let r = z
                .compute_streaming_strips_default(&s, &d)
                .expect("compute_streaming_strips_default");
            (r.score(), 0usize)
        }
        other => {
            eprintln!("unknown entry: {other}");
            std::process::exit(2);
        }
    };

    println!("entry={entry} {w}x{h} score={score:.6} nfeat={nfeat}");
}
