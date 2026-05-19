//! Demo matrix: run the target-search on 3 images × 3 codecs × 4 targets
//! and print a Markdown-friendly results table.
//!
//! `cargo run --release --example demo_matrix -p zensim-target`

use std::path::PathBuf;

use image::ImageReader;
use zensim::ZensimProfile;
use zensim_target::{CodecKind, TargetSpec, target_search};

const IMAGES: &[(&str, &str, &str)] = &[
    (
        "photo",
        "kadid I12",
        "/home/lilith/work/codec-eval/codec-corpus/kadid10k/I12.png",
    ),
    (
        "screen",
        "gb82-sc gui",
        "/home/lilith/work/codec-eval/codec-corpus/gb82-sc/gui.png",
    ),
    (
        "line-art",
        "kadid I50",
        "/home/lilith/work/codec-eval/codec-corpus/kadid10k/I50.png",
    ),
];

const CODECS: &[(&str, CodecKind)] = &[
    ("zenjpeg", CodecKind::Jpeg),
    ("zenwebp", CodecKind::Webp),
    ("zenavif", CodecKind::Avif),
];

const TARGETS: &[f32] = &[30.0, 50.0, 70.0, 90.0];

/// Pick the profile from the `ZENSIM_TARGET_PROFILE` env var. Defaults
/// to `v0_3` (legacy production-grade fallback per the 2026-05-18
/// demo). Accepts: `v0_2`, `v0_3`, `balanced`, `compression`,
/// `ensemble`. The 2026-05-19 affine-calibration fix makes the v0_5
/// variants usable; this knob lets the demo verify the new default.
fn profile_from_env() -> ZensimProfile {
    let raw = std::env::var("ZENSIM_TARGET_PROFILE").unwrap_or_else(|_| "v0_3".to_string());
    match raw.to_ascii_lowercase().as_str() {
        "v0_2" | "v02" | "preview-v0.2" => ZensimProfile::PreviewV0_2,
        "v0_3" | "v03" | "preview-v0.3" | "default" => ZensimProfile::PreviewV0_3,
        "balanced" | "v0_5_balanced" => ZensimProfile::balanced(),
        "compression" | "v0_5_compression" => ZensimProfile::compression(),
        "ensemble" | "v0_5_ensemble" => ZensimProfile::ensemble(),
        other => {
            eprintln!("unknown ZENSIM_TARGET_PROFILE={other}; falling back to v0_3");
            ZensimProfile::PreviewV0_3
        }
    }
}

fn main() -> anyhow::Result<()> {
    let profile = profile_from_env();
    eprintln!("demo_matrix: profile = {profile:?}");
    // Header
    println!(
        "| image (class) | codec | target | achieved | Δ | knob | bytes | iters | converged |"
    );
    println!("|:---|:---|---:|---:|---:|---:|---:|---:|:---:|");

    let mut total = 0u32;
    let mut hits = 0u32;
    let mut by_codec: std::collections::HashMap<&str, (u32, u32)> =
        std::collections::HashMap::new();

    for (class, label, path) in IMAGES {
        let path = PathBuf::from(path);
        let img = match ImageReader::open(&path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping {label}: {e}");
                continue;
            }
        };
        let img = img.with_guessed_format()?.decode()?;
        let rgb8 = img.to_rgb8();
        let w = rgb8.width();
        let h = rgb8.height();
        let rgb = rgb8.into_raw();
        eprintln!("--- {label} ({class}, {w}x{h}) ---");

        for (codec_name, codec) in CODECS {
            for &target in TARGETS {
                total += 1;
                let spec = TargetSpec {
                    target,
                    tolerance: 1.5,
                    max_iterations: 8,
                    profile,
                };
                match target_search(&rgb, w, h, *codec, spec) {
                    Ok(r) => {
                        let delta = r.achieved_score - target;
                        let ok = r.converged;
                        if ok {
                            hits += 1;
                        }
                        let entry = by_codec.entry(*codec_name).or_insert((0, 0));
                        entry.0 += 1;
                        if ok {
                            entry.1 += 1;
                        }
                        println!(
                            "| {label} ({class}) | {codec_name} | {target:.0} | {:.2} | {:+.2} | {:.2} | {} | {} | {} |",
                            r.achieved_score,
                            delta,
                            r.final_knob,
                            r.encoded.len(),
                            r.iterations,
                            if ok { "yes" } else { "no" },
                        );
                    }
                    Err(e) => {
                        println!(
                            "| {label} ({class}) | {codec_name} | {target:.0} | err | - | - | - | - | {} |",
                            format!("error: {e}").chars().take(20).collect::<String>()
                        );
                    }
                }
            }
        }
    }

    println!();
    println!(
        "**Summary:** {hits}/{total} cells converged within tolerance ({:.0}%).",
        100.0 * hits as f64 / total as f64
    );
    let mut codec_keys: Vec<_> = by_codec.keys().copied().collect();
    codec_keys.sort();
    for k in codec_keys {
        let (n, ok) = by_codec[k];
        println!("* `{k}`: {ok}/{n} ({:.0}%)", 100.0 * ok as f64 / n as f64);
    }

    Ok(())
}
