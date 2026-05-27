//! Localized-defect scorer (task #33): tile a (ref, dist) pair, score zensim
//! per tile, report the WORST tile alongside the global score.
//!
//! The corruption-corpus gate proved a GLOBAL perceptual metric cannot rank a
//! localized 8×8 structural defect below an honest q20 encode (the defect is
//! globally negligible). Tile-min pooling fixes that: a localized corruption
//! craters ONE tile → `min(tile)` reflects it; an honest q20 degrades all
//! tiles moderately → `min ≈ global`. So `min(tile)` ranks localized
//! corruption below honest q20 — the regression-test gate.
//!
//! Usage:
//!   score_tiles_with_bake --bake B --ref R.png --dist D.png \
//!       [--tile 64] [--overlap 0.5] [--bake-post clamp]
//! Prints TSV: `global<TAB>min<TAB>p2<TAB>p5<TAB>median<TAB>n_tiles`.

use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::{Model, Predictor};
use zensim::{ZensimConfig, compute_zensim_with_config};
use zensim_validate::bake_runtime::{
    extract_hybrid_head, extract_per_sample_alpha_head, extract_tanh_output_head_scale,
    score_with_bake_alloc,
};

fn cfg() -> ZensimConfig {
    let mut c = ZensimConfig::default();
    c.extended_features = true;
    c.compute_iw_features = true;
    c
}

fn apply_post(raw: f64, mode: &str) -> f64 {
    if raw.is_nan() {
        return f64::NAN;
    }
    match mode {
        "raw" | "extrapolate" => raw,
        "clamp" => raw.clamp(0.0, 100.0),
        _ => raw.clamp(0.0, 100.0),
    }
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (mut bake, mut ref_path, mut dist_path) = (None, None, None);
    let mut tile: usize = 64;
    let mut overlap: f64 = 0.5;
    let mut bake_post = "clamp".to_string();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bake = Some(PathBuf::from(args.next().unwrap())),
            "--ref" => ref_path = Some(PathBuf::from(args.next().unwrap())),
            "--dist" => dist_path = Some(PathBuf::from(args.next().unwrap())),
            "--tile" => tile = args.next().unwrap().parse().unwrap(),
            "--overlap" => overlap = args.next().unwrap().parse().unwrap(),
            "--bake-post" => bake_post = args.next().unwrap(),
            other => {
                eprintln!("unknown arg: {other}");
                return ExitCode::FAILURE;
            }
        }
    }
    let (bake, ref_path, dist_path) = (bake.unwrap(), ref_path.unwrap(), dist_path.unwrap());

    let src = image::open(&ref_path).expect("open ref").to_rgb8();
    let dst = image::open(&dist_path).expect("open dist").to_rgb8();
    let w = src.width() as usize;
    let h = src.height() as usize;
    let sp: Vec<[u8; 3]> = src.pixels().map(|p| p.0).collect();
    let dp: Vec<[u8; 3]> = dst.pixels().map(|p| p.0).collect();

    let bake_bytes = std::fs::read(&bake).expect("read bake");
    let model = Model::from_bytes(&bake_bytes).expect("parse bake");
    let n_inputs = model.n_inputs();
    let has_tx = model.has_nontrivial_feature_transforms();
    let psa = extract_per_sample_alpha_head(&model);
    let hyb = extract_hybrid_head(&model);
    let pin = extract_tanh_output_head_scale(&model);
    let spline = zensim_validate::output_calibration_spline::extract(&model);
    let mut predictor = Predictor::new(&model);

    let mut score_region = |s: &[[u8; 3]], d: &[[u8; 3]], rw: usize, rh: usize| -> Option<f64> {
        let res = compute_zensim_with_config(s, d, rw, rh, cfg()).ok()?;
        let feats = res.features().to_vec();
        let raw = score_with_bake_alloc(
            &mut predictor, has_tx, psa.as_ref(), hyb.as_ref(), pin, spline.as_ref(), n_inputs,
            &feats,
        );
        Some(apply_post(raw, &bake_post))
    };

    // Global score (whole image).
    let global = score_region(&sp, &dp, w, h).unwrap_or(f64::NAN);

    // Tile grid. zensim's 4-scale features need a minimum size; 32 px floor.
    let min_tile = 32usize;
    let stride = ((tile as f64) * (1.0 - overlap)).max(1.0) as usize;
    let mut tiles: Vec<f64> = Vec::new();
    let mut y0 = 0usize;
    loop {
        let th = tile.min(h.saturating_sub(y0));
        if th >= min_tile {
            let mut x0 = 0usize;
            loop {
                let tw = tile.min(w.saturating_sub(x0));
                if tw >= min_tile {
                    let mut ts = Vec::with_capacity(tw * th);
                    let mut td = Vec::with_capacity(tw * th);
                    for y in 0..th {
                        let row = (y0 + y) * w + x0;
                        ts.extend_from_slice(&sp[row..row + tw]);
                        td.extend_from_slice(&dp[row..row + tw]);
                    }
                    if let Some(s) = score_region(&ts, &td, tw, th) {
                        tiles.push(s);
                    }
                }
                if x0 + tile >= w {
                    break;
                }
                x0 += stride;
            }
        }
        if y0 + tile >= h {
            break;
        }
        y0 += stride;
    }

    tiles.sort_by(|a, b| a.total_cmp(b));
    let n = tiles.len();
    let pct = |p: f64| -> f64 {
        if n == 0 {
            return f64::NAN;
        }
        tiles[(((p / 100.0) * n as f64) as usize).min(n - 1)]
    };
    let min = tiles.first().copied().unwrap_or(f64::NAN);
    println!(
        "{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}",
        global,
        min,
        pct(2.0),
        pct(5.0),
        pct(50.0),
        n
    );
    ExitCode::SUCCESS
}
