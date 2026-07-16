//! RD block-selection eval: does the zensim **diffmap** pick better blocks to
//! spend bits on than the **SSE** a codec's rate-distortion loop defaults to?
//!
//! # The question
//!
//! A codec allocating a fixed bit budget must choose WHICH regions to encode
//! more precisely. The classic RD loop ranks blocks by squared error (SSE /
//! PSNR-optimal). A perceptual diffmap should rank them by *perceptual*
//! importance instead — spending bits where distortion is visible, saving them
//! where it is not. This measures whether zensim's diffmap is that better
//! ranker.
//!
//! # The simulation (a clean upper-bound proxy for "spend more bits here")
//!
//! Given a reference `R` and a distorted encode `D`, tile into blocks and, for
//! a budget fraction `f`, "refine" the top-`f` blocks — copy `R`'s pixels back
//! into `D` there (the limit of spending unlimited bits on those blocks).
//! Three selectors choose which blocks:
//!   * **sse**   — highest Σ(R−D)² per block (the codec default).
//!   * **zensim**— highest Σ zensim-diffmap per block (the candidate).
//!   * **random**— control.
//! Every strategy refines the SAME NUMBER of blocks, so they sit at the same
//! rate. The winner is whichever refined image an INDEPENDENT perceptual judge
//! (butteraugli / ssim2 via zenmetrics — a different metric family) scores best.
//! If `zensim` beats `sse`, the diffmap drives better RD decisions.
//!
//! Writes the refined variants + a `manifest.tsv` (ref, variant, strategy,
//! n_blocks) for the judge step. It does NOT score them itself — scoring is the
//! independent judge's job, kept separate so the eval can't grade its own work.
//!
//! ```sh
//! cargo run --release -p zensim --example rd_block_selection -- \
//!     <ref.png> <dist.png> <out-dir> [--block 32] [--frac 0.25]
//! ```

use zensim::DiffmapWeighting;
use zensim::RgbSlice;
use zensim::{Zensim, ZensimProfile};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!(
            "usage: rd_block_selection <ref.png> <dist.png> <out-dir> [--block N] [--frac F]"
        );
        std::process::exit(2);
    }
    let (ref_path, dist_path, out_dir) = (&args[0], &args[1], &args[2]);
    let mut block = 32usize;
    let mut frac = 0.25f64;
    let mut i = 3;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--block" => block = args[i + 1].parse().unwrap(),
            "--frac" => frac = args[i + 1].parse().unwrap(),
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }

    let r = image::open(ref_path).expect("open ref").to_rgb8();
    let d = image::open(dist_path).expect("open dist").to_rgb8();
    let (w, h) = (r.width() as usize, r.height() as usize);
    assert_eq!(
        (d.width() as usize, d.height() as usize),
        (w, h),
        "size mismatch"
    );

    let rpx: Vec<[u8; 3]> = r.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dpx: Vec<[u8; 3]> = d.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

    // zensim diffmap (per-pixel perceptual error).
    let z = Zensim::new(ZensimProfile::latest());
    let dm = z
        .compute_with_diffmap(
            &RgbSlice::new(&rpx, w, h),
            &RgbSlice::new(&dpx, w, h),
            DiffmapWeighting::default(),
        )
        .expect("diffmap");
    let diff = dm.diffmap();

    // Per-block SSE and zensim-diffmap sums.
    let bx = w.div_ceil(block);
    let by = h.div_ceil(block);
    let nblocks = bx * by;
    let mut sse = vec![0f64; nblocks];
    let mut zsum = vec![0f64; nblocks];
    for y in 0..h {
        for x in 0..w {
            let b = (y / block) * bx + (x / block);
            let p = y * w + x;
            let e: f64 = (0..3)
                .map(|c| {
                    let dv = rpx[p][c] as f64 - dpx[p][c] as f64;
                    dv * dv
                })
                .sum();
            sse[b] += e;
            zsum[b] += diff[p] as f64;
        }
    }

    let k = ((nblocks as f64) * frac).round().max(1.0) as usize;
    // Deterministic "random" control: strided pick (no rng dep, reproducible).
    let top = |scores: &[f64]| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..nblocks).collect();
        idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        idx.truncate(k);
        idx
    };
    let sse_sel = top(&sse);
    let zen_sel = top(&zsum);
    let rand_sel: Vec<usize> = (0..nblocks).step_by(nblocks.div_ceil(k)).take(k).collect();

    std::fs::create_dir_all(out_dir).unwrap();
    let refine = |sel: &[usize], name: &str| {
        let selset: std::collections::HashSet<usize> = sel.iter().copied().collect();
        let mut out = image::RgbImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let b = (y / block) * bx + (x / block);
                let p = y * w + x;
                // Refined blocks get the reference back (bits well spent);
                // the rest keep the distortion.
                let px = if selset.contains(&b) { rpx[p] } else { dpx[p] };
                out.put_pixel(x as u32, y as u32, image::Rgb(px));
            }
        }
        let path = format!("{out_dir}/{name}.png");
        out.save(&path).unwrap();
        path
    };

    let stem = std::path::Path::new(dist_path)
        .file_stem()
        .unwrap()
        .to_string_lossy();
    let variants = [
        ("sse", &sse_sel),
        ("zensim", &zen_sel),
        ("random", &rand_sel),
    ];
    // manifest for the independent judge
    let mut man = String::from("ref\tvariant\tstrategy\tn_blocks_refined\ttotal_blocks\tfrac\n");
    for (strat, sel) in variants {
        let vp = refine(sel, &format!("{stem}__{strat}"));
        man.push_str(&format!(
            "{ref_path}\t{vp}\t{strat}\t{}\t{nblocks}\t{frac}\n",
            sel.len()
        ));
    }
    // also copy the unrefined distorted as the frac=0 baseline reference row
    man.push_str(&format!("{ref_path}\t{dist_path}\tnone\t0\t{nblocks}\t0\n"));
    let manp = format!("{out_dir}/manifest_{stem}.tsv");
    std::fs::write(&manp, man).unwrap();

    // overlap between the two selectors — how differently do they choose?
    let sse_set: std::collections::HashSet<usize> = sse_sel.iter().copied().collect();
    let overlap = zen_sel.iter().filter(|b| sse_set.contains(b)).count();
    println!(
        "{stem}: {nblocks} blocks, refine top {k} ({:.0}%). \
         sse∩zensim = {overlap}/{k} ({:.0}% agree). manifest {manp}",
        frac * 100.0,
        100.0 * overlap as f64 / k as f64
    );
    eprintln!(
        "score the 3 variants + baseline with an INDEPENDENT judge (zenmetrics \
         butteraugli/ssim2); lower distortion / higher score for `zensim` than `sse` \
         at the same n_blocks => the diffmap is the better RD block-selector."
    );
}
