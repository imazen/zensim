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
    let mut i = 2;
    while i + 1 < args.len() {
        if args[i] == "--block" {
            block = args[i + 1].parse().unwrap();
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
            DiffmapWeighting::default(),
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

    // Ground truth: ΔS per block = how much refining THIS block raises the scalar.
    // Refine into a scratch copy, score, restore. One rescore per block.
    let mut delta_s = vec![0f64; nblocks];
    let mut scratch = dpx.clone();
    for by_i in 0..by {
        for bx_i in 0..bx {
            let b = by_i * bx + bx_i;
            let (x0, y0) = (bx_i * block, by_i * block);
            let (x1, y1) = ((x0 + block).min(w), (y0 + block).min(h));
            // copy ref → scratch over the block
            for y in y0..y1 {
                for x in x0..x1 {
                    scratch[y * w + x] = rpx[y * w + x];
                }
            }
            let s = z
                .compute(&RgbSlice::new(&rpx, w, h), &RgbSlice::new(&scratch, w, h))
                .expect("compute")
                .score();
            delta_s[b] = s - base_score;
            // restore
            for y in y0..y1 {
                for x in x0..x1 {
                    scratch[y * w + x] = dpx[y * w + x];
                }
            }
        }
    }

    let srocc_dm = spearman(&dmap_block, &delta_s);
    let srocc_sse = spearman(&sse_block, &delta_s);
    let plcc_dm = pearson(&dmap_block, &delta_s);
    println!(
        "spatial coherence ({} blocks, {block}px)  base_score={base_score:.2}",
        nblocks
    );
    println!("  SROCC(diffmap_block, ΔS) = {srocc_dm:+.4}   PLCC = {plcc_dm:+.4}");
    println!("  SROCC(SSE_block,     ΔS) = {srocc_sse:+.4}   (codec PSNR default — the bar)");
    let verdict = if srocc_dm >= srocc_sse && srocc_dm > 0.9 {
        "COHERENT — diffmap is the scalar's spatial gradient AND beats SSE"
    } else if srocc_dm > 0.9 {
        "coherent but SSE ties/beats it — diffmap adds little spatially"
    } else {
        "INCOHERENT — diffmap points at the wrong blocks for the scalar"
    };
    println!("  => {verdict}");
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
