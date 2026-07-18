//! SPATIAL diffmap↔scalar coherence — the test the pooled diagnostic can't do.
//!
//! The closed loop spends bits per-BLOCK guided by the diffmap, to raise the
//! SCALAR toward a target. So the diffmap must predict WHERE refining raises the
//! scalar. This measures that directly: for each block, copy the reference pixels
//! into the distorted image there (the limit of "spend unlimited bits on this
//! block"), rescore, and record ΔS = score_refined − score_base. A coherent
//! diffmap has `diffmap_block_sum` rank-agreeing with ΔS — it IS the scalar's
//! spatial gradient. We also report SSE-per-block (the codec's PSNR default) as
//! the bar the diffmap must beat.
//!
//! `SROCC(diffmap_block, ΔS)` ≈ 1 → the diffmap points exactly where the scalar
//! rewards bits. If SSE correlates with ΔS as well as the diffmap does, the
//! diffmap adds nothing over the codec default. If the diffmap's SROCC is LOW,
//! it points at the wrong blocks — the incoherence the closed loop can't tolerate.
//!
//! ```sh
//! cargo run --release -p zensim --example diffmap_block_coherence -- <ref> <dist> [--block 32]
//! ```

use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: diffmap_block_coherence <ref> <dist> [--block N]");
        std::process::exit(2);
    }
    let mut block = 32usize;
    let mut weighting = DiffmapWeighting::default(); // Trained (V0_2 weights)
    let mut i = 2;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--block" => block = args[i + 1].parse().unwrap(),
            "--weighting" => {
                weighting = match args[i + 1].as_str() {
                    "balanced" => DiffmapWeighting::Balanced,
                    "trained" => DiffmapWeighting::Trained,
                    other => {
                        eprintln!("unknown weighting {other}");
                        std::process::exit(2);
                    }
                }
            }
            _ => {}
        }
        i += 2;
    }
    let r = image::open(&args[0]).expect("open ref").to_rgb8();
    let d = image::open(&args[1]).expect("open dist").to_rgb8();
    let (w, h) = (r.width() as usize, r.height() as usize);
    assert_eq!((d.width() as usize, d.height() as usize), (w, h), "size mismatch");
    let rpx: Vec<[u8; 3]> = r.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dpx: Vec<[u8; 3]> = d.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

    let z = Zensim::new(ZensimProfile::latest_preview());
    let base = z
        .compute_with_diffmap(
            &RgbSlice::new(&rpx, w, h),
            &RgbSlice::new(&dpx, w, h),
            weighting,
        )
        .expect("diffmap");
    let base_score = base.score();
    let diff = base.diffmap().to_vec();

    let bx = w.div_ceil(block);
    let by = h.div_ceil(block);
    let nblocks = bx * by;
    let mut dmap_block = vec![0f64; nblocks];
    let mut sse_block = vec![0f64; nblocks];
    for y in 0..h {
        for x in 0..w {
            let b = (y / block) * bx + (x / block);
            let p = y * w + x;
            dmap_block[b] += diff[p] as f64;
            let e: f64 = (0..3)
                .map(|c| {
                    let dv = rpx[p][c] as f64 - dpx[p][c] as f64;
                    dv * dv
                })
                .sum();
            sse_block[b] += e;
        }
    }

    // Ground truth per block = how much refining THIS block raises each candidate scalar.
    //   delta_s      : the FULL (non-additive) zensim scalar — the current metric.
    //   delta_pooled : the ADDITIVE scalar = pooled diffmap (Σ per-pixel weighted signal).
    //                  Its spatial gradient IS the diffmap by construction, so this measures
    //                  the exact-diffmap ceiling the design's additive core targets.
    // The gap between the two = the cost of the current scalar's NON-additivity.
    let pooled_before: f64 = diff.iter().map(|&x| x as f64).sum();
    let mut delta_s = vec![0f64; nblocks];
    let mut delta_pooled = vec![0f64; nblocks];
    let mut scratch = dpx.clone();
    for by_i in 0..by {
        for bx_i in 0..bx {
            let b = by_i * bx + bx_i;
            let (x0, y0) = (bx_i * block, by_i * block);
            let (x1, y1) = ((x0 + block).min(w), (y0 + block).min(h));
            for y in y0..y1 {
                for x in x0..x1 {
                    scratch[y * w + x] = rpx[y * w + x];
                }
            }
            let refined = z
                .compute_with_diffmap(
                    &RgbSlice::new(&rpx, w, h),
                    &RgbSlice::new(&scratch, w, h),
                    weighting,
                )
                .expect("diffmap");
            delta_s[b] = refined.score() - base_score;
            let pooled_after: f64 = refined.diffmap().iter().map(|&x| x as f64).sum();
            // refining a block REDUCES pooled error; negate so higher = more-improved,
            // matching dmap_block's polarity (high = high error = high improvement potential).
            delta_pooled[b] = -(pooled_after - pooled_before);
            for y in y0..y1 {
                for x in x0..x1 {
                    scratch[y * w + x] = dpx[y * w + x];
                }
            }
        }
    }

    let srocc_full = spearman(&dmap_block, &delta_s);
    let srocc_add = spearman(&dmap_block, &delta_pooled);
    let srocc_sse = spearman(&sse_block, &delta_s);
    println!(
        "spatial coherence ({} blocks, {block}px)  base_score={base_score:.2}",
        nblocks
    );
    println!("  additive-scalar target:");
    println!("    SROCC(diffmap_block, Δ additive-scalar) = {srocc_add:+.4}   (exact-gradient ceiling)");
    println!("  current (non-additive) scalar:");
    println!("    SROCC(diffmap_block, ΔS_full)           = {srocc_full:+.4}   PLCC = {:+.4}", pearson(&dmap_block, &delta_s));
    println!("    SROCC(SSE_block,     ΔS_full)           = {srocc_sse:+.4}   (codec PSNR default — the bar)");
    println!(
        "  => additive core buys +{:.4} spatial coherence ({:.4} → {:.4}); non-additivity is the {:.0}% gap the design removes",
        srocc_add - srocc_full,
        srocc_full,
        srocc_add,
        (srocc_add - srocc_full) * 100.0
    );
}

fn rank(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; v.len()];
    for (k, &ix) in idx.iter().enumerate() {
        r[ix] = k as f64;
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
