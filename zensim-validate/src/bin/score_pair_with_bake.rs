//! Score one (ref, dist) PNG pair against an arbitrary ZNPR v3 bake.
//!
//! Usage:
//!   score_pair_with_bake --bake PATH [--bake-post raw|clamp|mapped[:A,B]] \
//!                        --ref REF.png --dist DIST.png
//!
//! Prints one float on stdout — the scored zensim value with the
//! specified post-processing applied. Used by cross_codec_consistency.py
//! to binary-search for the q value that matches a target zensim score.

use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::Model;
use zensim::{BakeScorer, RgbSlice};
use zensim_validate::bake_runtime::post_mode_params;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut bake: Option<PathBuf> = None;
    let mut bake_post: String = "clamp".to_string();
    let mut ref_path: Option<PathBuf> = None;
    let mut dist_path: Option<PathBuf> = None;
    let mut dump_features_to: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bake = Some(args.next().expect("--bake VALUE").into()),
            "--bake-post" => bake_post = args.next().expect("--bake-post VALUE"),
            "--ref" => ref_path = Some(args.next().expect("--ref VALUE").into()),
            "--dist" => dist_path = Some(args.next().expect("--dist VALUE").into()),
            "--dump-features-to" => {
                dump_features_to = Some(args.next().expect("--dump-features-to VALUE").into());
            }
            other => {
                eprintln!("unknown arg: {other}");
                return ExitCode::FAILURE;
            }
        }
    }
    let bake = bake.expect("--bake REQUIRED");
    let ref_path = ref_path.expect("--ref REQUIRED");
    let dist_path = dist_path.expect("--dist REQUIRED");

    // Load images.
    let src = image::open(&ref_path).expect("open ref").to_rgb8();
    let dst = image::open(&dist_path).expect("open dist").to_rgb8();
    let w = src.width() as usize;
    let h = src.height() as usize;
    let src_pixels: Vec<[u8; 3]> = src.pixels().map(|p| p.0).collect();
    let dst_pixels: Vec<[u8; 3]> = dst.pixels().map(|p| p.0).collect();

    let model = Model::from_bytes(&std::fs::read(&bake).expect("read bake")).expect("parse bake");
    let params = post_mode_params(&bake_post).expect("invalid bake-post");
    let mut scorer = BakeScorer::new(&model)
        .expect("invalid score metadata")
        .with_score_disposition(&params)
        .expect("invalid score disposition");
    let result = scorer
        .compute(
            &RgbSlice::new(&src_pixels, w, h),
            &RgbSlice::new(&dst_pixels, dst.width() as usize, dst.height() as usize),
            None,
        )
        .expect("candidate pixel scoring");
    let features = result.features();

    // EVAL-ACCEL bit-exact verification helper: optionally dump the
    // computed feature vector in the predict_features_with_bake wire
    // format so the two binaries can be cross-checked on the exact
    // same numeric input.
    if let Some(path) = &dump_features_to {
        let n_features = features.len();
        let mut buf = Vec::with_capacity(8 + n_features * 4);
        buf.extend_from_slice(&(n_features as u32).to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_rows
        for &v in features {
            buf.extend_from_slice(&(v as f32).to_le_bytes());
        }
        std::fs::write(path, &buf).expect("write --dump-features-to");
        eprintln!(
            "dumped {} features ({} bytes) to {}",
            n_features,
            buf.len(),
            path.display()
        );
    }

    let score = result.score();
    println!("{:.6}", score);
    ExitCode::SUCCESS
}
