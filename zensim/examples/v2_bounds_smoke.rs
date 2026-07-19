//! v2 "bounded" feature extraction — real-pair bounds smoke test.
//!
//! Iteration-1 acceptance check (`docs/FEATURE_V2_SPEC_2026-07-18.md`
//! §validation plan item 9): run `compute_v2_features` on real reference /
//! codec-output pairs and report per-block min/max, proving the documented
//! bounds hold on actual content (not just the synthetic adversarial
//! fixtures in `feature_v2.rs`'s unit tests).
//!
//! ```sh
//! cargo run --release -p zensim --features feature-regime-v2 \
//!   --example v2_bounds_smoke -- <ref1.png> <dist1.jpg> [<ref2.png> <dist2.jpg> ...]
//! ```

#[path = "support/zen_io.rs"]
mod zen_io;

use zensim::feature_v2::{FEATURES_PER_CHANNEL_V2_TOTAL, idx};
use zensim::{RgbSlice, Zensim, ZensimProfile};

struct Block {
    name: &'static str,
    offsets: &'static [usize],
    documented_bound: (f64, f64),
}

const BLOCKS: &[Block] = &[
    Block {
        name: "basic (bounded)",
        offsets: &[
            idx::SSIM_MEAN,
            idx::SSIM_DEV2,
            idx::SSIM_DEV4,
            idx::ART,
            idx::DET,
            idx::MSE,
            idx::HF_GAIN,
            idx::HF_LOSS,
            idx::HF_MAG_LOSS,
        ],
        documented_bound: (0.0, 2.0),
    },
    Block {
        name: "soft-peak",
        offsets: &[idx::SSIM_SOFT_PEAK, idx::ART_SOFT_PEAK, idx::DET_SOFT_PEAK],
        documented_bound: (0.0, 2.0),
    },
    Block {
        name: "masked",
        offsets: &[
            idx::MASKED_SSIM,
            idx::MASKED_ART,
            idx::MASKED_DET,
            idx::MASKED_MSE,
        ],
        documented_bound: (0.0, 2.0),
    },
    Block {
        name: "iw",
        offsets: &[idx::IW_SSIM, idx::IW_ART, idx::IW_DET, idx::IW_MSE],
        documented_bound: (0.0, 2.0),
    },
    Block {
        name: "pjnd",
        offsets: &[idx::PJND_TRANSDUCER, idx::PJND_FRAGILITY],
        documented_bound: (0.0, 1.0),
    },
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || !args.len().is_multiple_of(2) {
        eprintln!("usage: v2_bounds_smoke <ref1> <dist1> [<ref2> <dist2> ...]");
        std::process::exit(2);
    }

    let z = Zensim::new(ZensimProfile::codec_target());
    let mut any_oob = false;

    for pair in args.chunks(2) {
        let (ref_path, dist_path) = (&pair[0], &pair[1]);
        let (r_pixels, rw, rh) = zen_io::decode_rgb8(std::path::Path::new(ref_path));
        let (d_pixels, dw, dh) = zen_io::decode_rgb8(std::path::Path::new(dist_path));
        assert_eq!((rw, rh), (dw, dh), "pair dimension mismatch");
        let (w, h) = (rw, rh);

        let source = RgbSlice::new(&r_pixels, w, h);
        let distorted = RgbSlice::new(&d_pixels, w, h);

        let t0 = std::time::Instant::now();
        let result = z
            .compute_v2_features(&source, &distorted)
            .expect("v2 compute");
        let elapsed = t0.elapsed();

        println!(
            "\n=== {} vs {} ({}x{}, {} scales, {:.1} ms) ===",
            ref_path,
            dist_path,
            w,
            h,
            result.n_scales(),
            elapsed.as_secs_f64() * 1000.0
        );
        println!(
            "{:<18} {:>10} {:>12} {:>12} {:>10} {:>10}",
            "block", "n_values", "min", "max", "doc_lo", "doc_hi"
        );

        let features = result.features();
        for block in BLOCKS {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            let mut n = 0usize;
            for scale in 0..result.n_scales() {
                for ch in 0..3 {
                    let base = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                        + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                    for &off in block.offsets {
                        let v = features[base + off];
                        lo = lo.min(v);
                        hi = hi.max(v);
                        n += 1;
                    }
                }
            }
            let (doc_lo, doc_hi) = block.documented_bound;
            let oob = lo < doc_lo - 1e-6 || hi > doc_hi + 1e-6;
            if oob {
                any_oob = true;
            }
            println!(
                "{:<18} {:>10} {:>12.6} {:>12.6} {:>10.2} {:>10.2}{}",
                block.name,
                n,
                lo,
                hi,
                doc_lo,
                doc_hi,
                if oob { "  <-- OUT OF BOUNDS" } else { "" }
            );
        }
    }

    if any_oob {
        eprintln!("\nFAIL: at least one block exceeded its documented bound.");
        std::process::exit(1);
    }
    println!("\nOK: every block stayed within its documented bound on every pair.");
}
