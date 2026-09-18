//! Render zensim's spatial diffmap as a viewable colour heatmap.
//!
//! This is the first *viewable* consumer of [`zensim::DiffmapResult`]: every
//! prior diffmap tool in the repo either consumed the `f32` map numerically
//! (block-coherence instruments, RD block selection) or dumped a
//! self-normalised grayscale PNG behind an env var. Neither answers the
//! question a human asks first — "where does the metric think the damage is,
//! and is it hotter at q20 than at q90?"
//!
//! ```sh
//! cargo run --release -p zensim --example diffmap_heatmap -- \
//!   ref.png dist.jpg /mnt/v/output/zensim/demos/foo_q50 [--profile b|d] [--scale-max 0.35]
//! ```
//!
//! Writes `<out_prefix>_heat.png` and `<out_prefix>_overlay.png`, and prints
//! one JSON object on stdout for a gallery driver to collect.
//!
//! # Resolution
//!
//! **No upsampling happens here.** `DiffmapResult::diffmap()` is already
//! `width × height` at the compared images' own resolution (coarse pyramid
//! scales are upsampled and blended *inside* `zensim`), so the heatmap is a
//! 1:1 pixel-for-pixel render of the map. It is nonetheless *spatially
//! smooth*: scale 0 reflects an 11×11 neighbourhood and scales 1-3 reflect
//! 22×22 / 44×44 / 88×88, and with the trained blend scale 0 carries only
//! ~6% of the weight — so expect broad blobs, not per-pixel speckle.
//!
//! # Normalisation
//!
//! The default is a FIXED ABSOLUTE scale (`--scale-max`, default
//! [`DEFAULT_SCALE_MAX`]) shared across every image, which is the whole point:
//! a q20 encode must *look* hotter than a q90 encode of the same source. A
//! per-image percentile normalisation (what the existing `ZENSIM_JBU_DUMP`
//! grayscale dump does) is the right choice for inspecting one pair's
//! structure and the wrong one for a gallery — it makes every row equally hot.
//! The printed `p50` / `p99` / `max` let a driver sanity-check the choice.

#[path = "support/zen_io.rs"]
mod zen_io;

use zensim::{DiffmapOptions, RgbSlice, Zensim, ZensimProfile};

/// Diffmap value rendered as the top of the colour ramp.
///
/// MEASURED, not taken from the docs. `DiffmapResult`'s own doc comment puts
/// "most values on typical photo pairs" in `[0, 0.3]` and calls `> 0.5`
/// severe — that is **an order of magnitude too high for the default
/// `Trained` weighting**. Across 6 imazen-26 TRAIN sources (photo, nature,
/// grayscale patent scan, chart, web screenshot, AI clipart), zenjpeg 4:2:0
/// at q20/q50/q80, 900px long edge, profile `B`:
///
/// | | p50 | p99 | max |
/// |---|---|---|---|
/// | q20, worst source (photo) | 0.011 | 0.069 | **0.084** |
/// | q20, flat-content source (clipart) | 0.000 | 0.009 | 0.044 |
/// | q80, worst source | 0.003 | 0.014 | 0.022 |
///
/// So 0.06 puts a q20 photo's hot regions at the bright end of the ramp
/// (clipping only the top ~1% of pixels), leaves q80 visibly dim, and keeps
/// the whole q-ladder ordered by brightness. A 0.35 ceiling renders every
/// one of those images near-black.
const DEFAULT_SCALE_MAX: f32 = 0.06;

/// 10-stop `inferno` (matplotlib) sampled at `t = 0.0, 1/9, …, 1.0`,
/// interpolated linearly in sRGB. Inferno is monotone in lightness, so "cool"
/// reads as dark/no-error and "hot" as bright without a legend — which a
/// rainbow map cannot do. Inline so this example adds no dependency.
const INFERNO: [[u8; 3]; 10] = [
    [0x00, 0x00, 0x04],
    [0x1b, 0x0c, 0x41],
    [0x4a, 0x0c, 0x6b],
    [0x78, 0x1c, 0x6d],
    [0xa5, 0x2c, 0x60],
    [0xcf, 0x44, 0x46],
    [0xed, 0x69, 0x25],
    [0xfb, 0x9b, 0x06],
    [0xf7, 0xd1, 0x3d],
    [0xfc, 0xff, 0xa4],
];

fn inferno(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0) * (INFERNO.len() - 1) as f32;
    let i = (t as usize).min(INFERNO.len() - 2);
    let f = t - i as f32;
    let (a, b) = (INFERNO[i], INFERNO[i + 1]);
    [0, 1, 2].map(|c| (a[c] as f32 + (b[c] as f32 - a[c] as f32) * f).round() as u8)
}

struct Args {
    reference: String,
    distorted: String,
    out_prefix: String,
    profile: String,
    scale_max: f32,
    edge_mse: bool,
    hf: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Vec<String> = Vec::new();
    let mut profile = "b".to_string();
    let mut scale_max = DEFAULT_SCALE_MAX;
    let (mut edge_mse, mut hf) = (false, false);
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--profile" => {
                i += 1;
                profile = argv.get(i).cloned().unwrap_or_else(|| usage("--profile"));
            }
            "--scale-max" => {
                i += 1;
                scale_max = argv
                    .get(i)
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or_else(|| usage("--scale-max"));
            }
            "--edge-mse" => edge_mse = true,
            "--hf" => hf = true,
            other if other.starts_with("--") => usage(other),
            _ => positional.push(argv[i].clone()),
        }
        i += 1;
    }
    if positional.len() != 3 {
        usage("expected <ref> <dist> <out_prefix>");
    }
    if !(scale_max.is_finite() && scale_max > 0.0) {
        usage("--scale-max must be a positive finite number");
    }
    Args {
        reference: positional[0].clone(),
        distorted: positional[1].clone(),
        out_prefix: positional[2].clone(),
        profile,
        scale_max,
        edge_mse,
        hf,
    }
}

fn usage(what: &str) -> ! {
    eprintln!("bad argument: {what}");
    eprintln!(
        "usage: diffmap_heatmap <ref.png> <dist.png> <out_prefix> \
         [--profile b|d] [--scale-max <f>] [--edge-mse] [--hf]"
    );
    std::process::exit(2);
}

/// Value at the given fraction of a sorted slice. `sorted` must be non-empty.
fn percentile(sorted: &[f32], q: f64) -> f32 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let args = parse_args();

    let profile = match args.profile.as_str() {
        "b" | "B" => ZensimProfile::B,
        "d" | "D" => ZensimProfile::D,
        other => usage(other),
    };

    let (src, sw, sh) = zen_io::decode_rgb8(std::path::Path::new(&args.reference));
    let (dst, dw, dh) = zen_io::decode_rgb8(std::path::Path::new(&args.distorted));
    if (sw, sh) != (dw, dh) {
        eprintln!("dimension mismatch: reference {sw}x{sh} vs distorted {dw}x{dh}");
        std::process::exit(2);
    }
    if sw == 0 || sh == 0 {
        eprintln!("zero-sized image");
        std::process::exit(2);
    }

    let options = DiffmapOptions {
        include_edge_mse: args.edge_mse,
        include_hf: args.hf,
        ..Default::default()
    };
    let z = Zensim::new(profile);
    let result = z
        .compute_with_diffmap(
            &RgbSlice::new(&src, sw, sh),
            &RgbSlice::new(&dst, dw, dh),
            options,
        )
        .expect("zensim compute_with_diffmap");

    let score = result.score();
    // `width`/`height` come back from the result rather than being assumed
    // equal to the input: the sub-64px path reflect-pads internally and trims
    // back, and a demo must not index past what it was handed.
    let (w, h) = (result.width(), result.height());
    let map = result.diffmap();
    assert_eq!(map.len(), w * h, "diffmap is not width*height");

    let mut sorted: Vec<f32> = map.iter().map(|v| v.abs()).collect();
    sorted.sort_by(f32::total_cmp);
    let (p50, p99, p995, vmax) = (
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.99),
        percentile(&sorted, 0.995),
        *sorted.last().expect("non-empty map"),
    );

    // Heatmap: absolute scale, so brightness is comparable across images.
    let heat: Vec<[u8; 3]> = map
        .iter()
        .map(|&v| inferno(v.abs() / args.scale_max))
        .collect();

    // Overlay: the distorted image desaturated and dimmed, with the heat
    // colour laid over it proportionally to error, so the artefact and the
    // content it sits on are visible at once.
    let overlay: Vec<[u8; 3]> = (0..w * h)
        .map(|i| {
            let px = dst[i];
            // Rec.709 luma on the sRGB-encoded bytes. Not colour-managed —
            // this is a backdrop, not a measurement.
            let luma = 0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32;
            let base = (luma * 0.55 + 40.0).clamp(0.0, 255.0);
            let t = (map[i].abs() / args.scale_max).clamp(0.0, 1.0);
            // `t^0.6` opens the low end so faint-but-real error still tints;
            // capped below 1 so the underlying content never fully vanishes.
            let alpha = t.powf(0.6) * 0.9;
            let hc = inferno(t);
            [0, 1, 2].map(|c| (base * (1.0 - alpha) + hc[c] as f32 * alpha).round() as u8)
        })
        .collect();

    let heat_path = format!("{}_heat.png", args.out_prefix);
    let overlay_path = format!("{}_overlay.png", args.out_prefix);
    std::fs::write(&heat_path, zen_io::encode_png_rgb8(&heat, w, h)).expect("write heat png");
    std::fs::write(&overlay_path, zen_io::encode_png_rgb8(&overlay, w, h))
        .expect("write overlay png");

    let json = serde_json::json!({
        "reference": args.reference,
        "distorted": args.distorted,
        "profile": profile.name(),
        "score": score,
        "width": w,
        "height": h,
        "scale_max": args.scale_max,
        "include_edge_mse": args.edge_mse,
        "include_hf": args.hf,
        "map_p50": p50,
        "map_p99": p99,
        "map_p995": p995,
        "map_max": vmax,
        "heat_png": heat_path,
        "overlay_png": overlay_path,
    });
    println!("{json}");
}
