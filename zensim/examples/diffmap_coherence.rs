//! Diffmap↔scalar coherence diagnostic — the #1 closed-loop blocker.
//!
//! For a COHERENT metric the pooled (mean) diffmap is monotone in the scalar
//! distance `100 − score`: the per-pixel error map, summed, IS the total error
//! the scalar reports (up to the pooling constant). So
//! `SROCC(mean(diffmap), 100 − score)` across a distortion sweep should be ≈ 1.
//!
//! A LOW SROCC exposes the incoherence documented in
//! `benchmarks/diffmap_coherence_2026-07-16.md`: the diffmap is weighted by the
//! stale `WEIGHTS_PREVIEW_V0_2` SSIM vector ("unused on the MLP path"), while the
//! scalar comes from the 372-feature model — so the diffmap points at pixels the
//! shipped scalar does not actually weight, and the closed loop (spend bits where
//! the diffmap is hot to raise the scalar) fights itself.
//!
//! This is the falsification gate for the coherent-diffmap fix: run it before
//! (expect low) and after (expect ≈ 1).
//!
//! ```sh
//! cargo run --release -p zensim --example diffmap_coherence -- <ref.png> <dist1> <dist2> ...
//! ```
//! Feed a reference plus its quality-ladder encodes (jpg/webp/… — `image` decodes
//! them). All distorted images must match the reference dimensions.

use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: diffmap_coherence <ref.png> <dist1> <dist2> ...  (>=2 distorted)");
        std::process::exit(2);
    }
    let ref_img = image::open(&args[0]).expect("open ref").to_rgb8();
    let (w, h) = (ref_img.width() as usize, ref_img.height() as usize);
    let rpx: Vec<[u8; 3]> = ref_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

    // `latest_preview()` is the shipped default profile (B) — the one whose
    // closed loop must work. The diffmap uses `Trained` weighting (the default).
    let z = Zensim::new(ZensimProfile::latest_preview());

    let mut scores = Vec::new();
    let mut pooled = Vec::new();
    for dp in &args[1..] {
        let d = match image::open(dp) {
            Ok(d) => d.to_rgb8(),
            Err(e) => {
                eprintln!("  skip {dp}: {e}");
                continue;
            }
        };
        if d.width() as usize != w || d.height() as usize != h {
            eprintln!(
                "  skip {dp}: size {}x{} != ref {w}x{h}",
                d.width(),
                d.height()
            );
            continue;
        }
        let dpx: Vec<[u8; 3]> = d.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
        let dm = z
            .compute_with_diffmap(
                &RgbSlice::new(&rpx, w, h),
                &RgbSlice::new(&dpx, w, h),
                DiffmapWeighting::default(),
            )
            .expect("diffmap");
        let score = dm.score();
        let map = dm.diffmap();
        let mean = map.iter().map(|&v| v as f64).sum::<f64>() / map.len().max(1) as f64;
        println!(
            "  {}: score={score:8.3}  pooled_diffmap={mean:.6}",
            dp.rsplit('/').next().unwrap_or(dp)
        );
        scores.push(score);
        pooled.push(mean);
    }
    if scores.len() < 2 {
        eprintln!("need >=2 scored distorted images");
        std::process::exit(1);
    }
    // Coherence: rank agreement of pooled diffmap with the scalar distance.
    let dist: Vec<f64> = scores.iter().map(|s| 100.0 - s).collect();
    let srocc = spearman(&pooled, &dist);
    // Also report Pearson on the raw pooled↔distance (dial-honesty: a coherent
    // diffmap pools LINEARLY to the distance, not just monotonically).
    let plcc = pearson(&pooled, &dist);
    println!(
        "\ncoherence  SROCC(pooled_diffmap, 100-score) = {srocc:+.4}   PLCC = {plcc:+.4}   (n={})",
        scores.len()
    );
    println!(
        "  ~1.0 = coherent (diffmap tracks the scalar the closed loop targets); \
         low/negative = INCOHERENT (diffmap ≠ scalar model)"
    );
}

fn rank(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; v.len()];
    for (k, &i) in idx.iter().enumerate() {
        r[i] = k as f64;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt() + 1e-12)
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    pearson(&rank(a), &rank(b))
}
