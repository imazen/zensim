//! IW-overhead measurement test (executable as a test for convenience).
//!
//! Times four configurations on the same image at multiple resolutions, using
//! **round-robin interleaving** of the four configs per sample. This kills
//! thermal / turbo bias that hits non-interleaved back-to-back timing.
//!
//! Configs:
//! - Basic (228 features)
//! - Extended (300 features, masked block on)
//! - IW-only (228 + 72 = 300 features w/ IW pool only)
//! - Both (372 features, masked + IW)
//!
//! Reports paired median ratio per config vs Basic, with MAD. Run with:
//!
//! ```
//! cargo test --release --features training,threads -p zensim --test iw_perf_baseline \
//!     -- --nocapture --test-threads=1
//! ```
//!
//! Hard-asserts WithIw is within 4x of Basic — a fence to catch
//! catastrophic regressions, NOT a goal. Print output is the actual measurement.

#![cfg(feature = "training")]

use std::time::Instant;
use zensim::{ZensimConfig, compute_zensim_with_config};

fn make_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let mut src = vec![[0u8; 3]; n];
    let mut dst = vec![[0u8; 3]; n];
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w) as u8;
            let g = ((y * 255) / h) as u8;
            let b = ((x + y) * 127 / (w + h)) as u8;
            src[y * w + x] = [r, g, b];
            dst[y * w + x] = [
                r.saturating_add(3),
                g.saturating_sub(2),
                b.saturating_add(1),
            ];
        }
    }
    (src, dst)
}

fn run_one(src: &[[u8; 3]], dst: &[[u8; 3]], w: usize, h: usize, extended: bool, iw: bool) -> f64 {
    let mut config = ZensimConfig::default();
    config.extended_features = extended;
    config.compute_iw_features = iw;
    config.compute_all_features = extended || iw;
    let t0 = Instant::now();
    let _ = std::hint::black_box(compute_zensim_with_config(src, dst, w, h, config));
    t0.elapsed().as_secs_f64()
}

/// Paired interleaved sampling: K outer rounds × 4 configs per round.
/// Returns Vec<[f64; 4]> = per-round (basic, ext, iw, both).
fn paired_round_robin(
    src: &[[u8; 3]],
    dst: &[[u8; 3]],
    w: usize,
    h: usize,
    rounds: usize,
) -> Vec<[f64; 4]> {
    // Warmup all configs to fault in pages + warm the dispatch tables.
    for _ in 0..2 {
        let _ = run_one(src, dst, w, h, false, false);
        let _ = run_one(src, dst, w, h, true,  false);
        let _ = run_one(src, dst, w, h, false, true);
        let _ = run_one(src, dst, w, h, true,  true);
    }
    let mut out = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t_basic = run_one(src, dst, w, h, false, false);
        let t_ext   = run_one(src, dst, w, h, true,  false);
        let t_iw    = run_one(src, dst, w, h, false, true);
        let t_both  = run_one(src, dst, w, h, true,  true);
        out.push([t_basic, t_ext, t_iw, t_both]);
    }
    out
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 0 { (xs[n/2 - 1] + xs[n/2]) / 2.0 } else { xs[n/2] }
}

fn mad(xs: &[f64], med: f64) -> f64 {
    let mut deviations: Vec<f64> = xs.iter().map(|x| (x - med).abs()).collect();
    median(&mut deviations) * 1.4826
}

fn run_geometry(name: &str, w: usize, h: usize, rounds: usize) {
    let (src, dst) = make_pair(w, h);
    let samples = paired_round_robin(&src, &dst, w, h, rounds);

    // Paired per-round ratios = robust to whole-run thermal drift.
    let mut ratios_ext:  Vec<f64> = samples.iter().map(|s| s[1] / s[0]).collect();
    let mut ratios_iw:   Vec<f64> = samples.iter().map(|s| s[2] / s[0]).collect();
    let mut ratios_both: Vec<f64> = samples.iter().map(|s| s[3] / s[0]).collect();

    let med_ext  = median(&mut ratios_ext);
    let med_iw   = median(&mut ratios_iw);
    let med_both = median(&mut ratios_both);
    let mad_ext  = mad(&ratios_ext, med_ext);
    let mad_iw   = mad(&ratios_iw, med_iw);
    let mad_both = mad(&ratios_both, med_both);

    let mut basics: Vec<f64> = samples.iter().map(|s| s[0]).collect();
    let med_basic = median(&mut basics);

    eprintln!(
        "{name:>12} basic={mb:>7.3}ms  ext={me:.3}±{xe:.3}x  iw={mi:.3}±{xi:.3}x  both={mb2:.3}±{xb:.3}x",
        mb = med_basic * 1000.0, me = med_ext, xe = mad_ext,
        mi = med_iw, xi = mad_iw, mb2 = med_both, xb = mad_both,
    );

    assert!(med_both < 4.0, "WithIw>4x Basic at {name}: median {med_both:.2}x");
}

#[test]
fn iw_overhead_report() {
    eprintln!("\nIW overhead report — paired round-robin median (1.4826*MAD)");
    eprintln!("====================================================================");
    run_geometry("256x256",   256,  256, 50);
    run_geometry("512x512",   512,  512, 30);
    run_geometry("1024x1024",1024, 1024, 15);
    run_geometry("2048x1024",2048, 1024, 10);
    eprintln!("====================================================================");
}
