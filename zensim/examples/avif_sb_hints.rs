//! A-Y4 (campaign appendix Y): the AVIF per-superblock steering signal —
//! the committed half of the rect-query steering interface.
//!
//! Pipeline: (ref, dist) pair → folded-944 features → the bake's FD
//! gradient at those features (`score_features_fd_gradient_with_profile`,
//! the L-Y1 batched entry) → full attribution density
//! (`compute_attribution_density_full`) → per-block density MEANS on the
//! AV1 superblock grid (`AttributionResult::block_sums` / clipped areas)
//! → TSV for the encoder-side consumer.
//!
//! `query_rect` semantics make each block's value the first-order
//! prediction of the scalar-score gain from re-encoding that block at
//! reference quality — POSITIVE = perceptually damaged = wants bits.
//! The weight→`sb_q_scale` POLICY (normalization, exponent, clamp) is
//! deliberately the consumer's (zenavif `FrameHints::sb_q_scale` takes
//! quantizer-scale factors); this tool emits the mechanism-side signal
//! only.
//!
//! ```sh
//! cargo run --release -p zensim --features custom-profiles,feature-regime-v2 \
//!     --example avif_sb_hints -- <ref.png> <dist.png> <bake.bin> <out.tsv> [--block 64]
//! ```
//!
//! TSV format: `# width height block cols rows` header line, then `rows`
//! lines of `cols` tab-separated per-block mean densities.

use std::sync::OnceLock;
use zensim::profile::ProfileParams;
use zensim::{RgbSlice, Zensim, ZensimProfile};

static BAKE_BYTES: OnceLock<Vec<u8>> = OnceLock::new();

fn bake_bytes() -> &'static [u8] {
    BAKE_BYTES
        .get()
        .expect("bake bytes loaded before profile use")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        eprintln!("usage: avif_sb_hints <ref.png> <dist.png> <bake.bin> <out.tsv> [--block 64]");
        std::process::exit(2);
    }
    let (ref_path, dist_path, bake_path, out_path) = (&args[0], &args[1], &args[2], &args[3]);
    let mut block = 64usize;
    let mut i = 4;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--block" => block = args[i + 1].parse().expect("--block N"),
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    assert!(block > 0);

    let r = image::open(ref_path).expect("open ref").to_rgb8();
    let d = image::open(dist_path).expect("open dist").to_rgb8();
    let (w, h) = (r.width() as usize, r.height() as usize);
    assert_eq!(
        (d.width() as usize, d.height() as usize),
        (w, h),
        "pair size mismatch"
    );
    let rpx: Vec<[u8; 3]> = r.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let dpx: Vec<[u8; 3]> = d.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

    BAKE_BYTES
        .set(std::fs::read(bake_path).expect("read bake"))
        .expect("bake set once");

    // The jxl rd-bake mount shape (extended features for the attribution
    // basic block; serial — deterministic signal).
    let params = ProfileParams::builder()
        .mlp(bake_bytes)
        .skip_score_mapping(true)
        .extrapolate_score(true)
        .extended_features(true)
        .compute_iw_features(true)
        .build();
    let params: &'static ProfileParams = Box::leak(Box::new(params));
    let profile = ZensimProfile::Custom {
        params,
        name: "avif-sb-hints",
    };
    let z = Zensim::new(profile).with_parallel(false);

    let rs = RgbSlice::new(&rpx, w, h);
    let ds = RgbSlice::new(&dpx, w, h);

    // 1) Folded-944 features of the pair (the bake's true input regime).
    let toggles = zensim::feature_v2::V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        ..Default::default()
    };
    let mut scratch = zensim::feature_v2::V2Scratch::new();
    let v2 = z
        .compute_folded720_features_streaming(&rs, &ds, toggles, &mut scratch)
        .expect("folded-944 extraction");
    let feats = v2.features();
    let score =
        zensim::score_features_with_profile(profile, feats, w as u32, h as u32).expect("score");

    // 2) The bake's own gradient at this pair's features (batched FD).
    let s = zensim::score_features_fd_gradient_with_profile(profile, feats, w as u32, h as u32)
        .expect("FD gradient");
    let n_nonzero = s.iter().filter(|&&g| g != 0.0).count();
    assert!(
        n_nonzero > 0,
        "gradient identically zero — probe never engaged"
    );

    // 3) Full attribution density under that gradient.
    let attr = z
        .compute_attribution_density_full(&rs, &ds, &s)
        .expect("attribution density");

    // 4) Per-block MEANS on the superblock grid (edge blocks use their
    //    clipped area).
    let cols = w.div_ceil(block);
    let rows = h.div_ceil(block);
    let sums = attr.block_sums(block);
    assert_eq!(sums.len(), cols * rows);
    let mut out = String::new();
    out.push_str(&format!("# {w} {h} {block} {cols} {rows}\n"));
    for by in 0..rows {
        let mut line = String::new();
        for bx in 0..cols {
            let bw = ((bx + 1) * block).min(w) - bx * block;
            let bh = ((by + 1) * block).min(h) - by * block;
            let mean = sums[by * cols + bx] / (bw * bh) as f64;
            if bx > 0 {
                line.push('\t');
            }
            line.push_str(&format!("{mean:.9e}"));
        }
        line.push('\n');
        out.push_str(&line);
    }
    std::fs::write(out_path, out).expect("write tsv");
    eprintln!(
        "avif_sb_hints: {w}x{h} block={block} grid={cols}x{rows} score={score:.4} \
         grad_nonzero={n_nonzero}/{} -> {out_path}",
        s.len()
    );
}
