//! Score one (ref, dist) pair against a NAMED zensim profile.
//!
//! Usage: zensim_score_named PROFILE_NAME ref.png dist.png
//!
//! PROFILE_NAME ∈ {
//!   v0_2, v0_3, v0_4, v0_5,
//!   v0_5_balanced, v0_5_compression, v0_5_ensemble, v0_5_tuner,
//!   latest
//! }
//!
//! Used by cross_codec_consistency.py to binary-search the q value
//! achieving a target zensim score under each shipping profile.

use std::env;
use std::process::ExitCode;
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} PROFILE_NAME ref.png dist.png", args[0]);
        eprintln!(
            "  PROFILE_NAME ∈ {{v0_2, v0_3, v0_4, v0_5, v0_5_balanced, v0_5_compression, v0_5_ensemble, v0_5_tuner, latest}}"
        );
        return ExitCode::FAILURE;
    }
    let profile = match args[1].as_str() {
        "v0_2" => ZensimProfile::PreviewV0_2,
        "v0_3" => ZensimProfile::PreviewV0_3,
        "v0_4" => ZensimProfile::PreviewV0_4,
        "v0_5" => ZensimProfile::PreviewV0_5,
        "v0_5_balanced" => ZensimProfile::PreviewV0_5Balanced,
        "v0_5_compression" => ZensimProfile::PreviewV0_5Compression,
        "v0_5_ensemble" => ZensimProfile::PreviewV0_5Ensemble,
        "v0_5_tuner" => ZensimProfile::PreviewV0_5Tuner,
        "latest" => ZensimProfile::latest(),
        other => {
            eprintln!("unknown profile: {other}");
            return ExitCode::FAILURE;
        }
    };
    let img1 = image::open(&args[2]).expect("open ref");
    let img2 = image::open(&args[3]).expect("open dist");
    let img1 = img1.to_rgb8();
    let img2 = img2.to_rgb8();
    let w = img1.width() as usize;
    let h = img1.height() as usize;
    if img2.width() as usize != w || img2.height() as usize != h {
        eprintln!(
            "dimension mismatch: ref {}x{} vs dist {}x{}",
            w,
            h,
            img2.width(),
            img2.height()
        );
        return ExitCode::FAILURE;
    }
    let src: Vec<[u8; 3]> = img1.pixels().map(|p| p.0).collect();
    let dst: Vec<[u8; 3]> = img2.pixels().map(|p| p.0).collect();
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    let z = Zensim::new(profile);
    match z.compute(&s, &d) {
        Ok(r) => {
            println!("{:.6}", r.score());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("zensim error: {e:?}");
            ExitCode::FAILURE
        }
    }
}
