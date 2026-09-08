//! zensim-target CLI — pick encode parameters that hit a target zensim score.
//!
//! Usage:
//!   zensim-target <input.png> --target 70 --codec zenjpeg \
//!       [--profile balanced|compression|ensemble] \
//!       [--tolerance 1.0] [--max-iterations 8] \
//!       [--output encoded.<ext>]
//!
//! Prints a per-probe trace and a one-line summary with the achieved score,
//! the chosen knob, and the encoded byte count.
#![allow(deprecated)]
// exercises the deprecated `ZensimProfile::A` (shipped behind the default-on `deprecated-profiles` feature)

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use image::ImageReader;
use zensim::ZensimProfile;
use zensim_target::{CodecKind, TargetSpec, target_search};

#[derive(Parser, Debug)]
#[command(name = "zensim-target", about, version)]
struct Cli {
    /// Input image (any format readable by the `image` crate: png, jpeg,
    /// webp, gif, ...).
    input: PathBuf,

    /// Desired finite zensim score; negative targets are allowed.
    #[arg(short, long, allow_hyphen_values = true, default_value_t = 70.0)]
    target: f32,

    /// Codec to use: zenjpeg | zenwebp | zenavif | zenjxl | zenpng.
    #[arg(short, long, default_value = "zenjpeg")]
    codec: String,

    /// Scoring profile. Defaults to the library's codec-target profile.
    /// Historical tuners remain available by explicit name, such as tuner-v4.
    #[arg(long, default_value = "codec-target")]
    profile: String,

    /// Convergence tolerance — search stops when `|achieved - target| <= tolerance`.
    #[arg(long, default_value_t = 1.0)]
    tolerance: f32,

    /// Maximum iterations of the binary search.
    #[arg(long, default_value_t = 8)]
    max_iterations: u32,

    /// Optional output path for the encoded bytes (defaults to no write).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Quiet — emit only the final summary line, no per-probe trace.
    #[arg(short, long, default_value_t = false)]
    quiet: bool,
}

fn parse_profile(s: &str) -> Result<ZensimProfile> {
    // The named experimental "trail" variants now live in the unpublished
    // `zensim-experimental` crate as free functions returning bit-identical
    // `ZensimProfile::Custom` values. This CLI enumerates every trail by name
    // for evaluation, so it depends on that crate. `codec_target` /
    // `latest_preview` resolve to the canonical shipped production profile.
    match s.to_ascii_lowercase().as_str() {
        // The historical linear `PreviewV0_2` was removed as a built-in
        // `zensim` variant; it is reconstructed bit-identically as a
        // `Custom` profile in `zensim-experimental`. (Linear `PreviewV0_1`
        // was dropped entirely — its weights were removed.)
        "v0_2" | "v02" | "preview-v0.2" => Ok(zensim_experimental::preview_v0_2()),
        // "v0.3" was the never-published deprecated alias for `A`; keep the
        // CLI string working but resolve it to the canonical `A`.
        "a" | "v0_3" | "v03" | "preview-v0.3" => Ok(ZensimProfile::A),
        "codec-target" | "codec_target" | "default" => Ok(ZensimProfile::codec_target()),
        "latest" | "latest-preview" | "latest_preview" => Ok(ZensimProfile::latest_preview()),
        "balanced" | "v0_5_balanced" | "preview-v0.5-balanced" => {
            Ok(zensim_experimental::preview_v0_5_balanced())
        }
        "compression" | "v0_5_compression" | "preview-v0.5-compression" => {
            Ok(zensim_experimental::preview_v0_5_compression())
        }
        "ensemble" | "v0_5_ensemble" | "preview-v0.5-ensemble" => {
            Ok(zensim_experimental::preview_v0_5_ensemble())
        }
        "tuner" | "v0_5_tuner" | "preview-v0.5-tuner" => {
            Ok(zensim_experimental::preview_v0_5_tuner())
        }
        "tuner-v2" | "tuner_v2" | "v0_5_tuner_v2" | "preview-v0.5-tuner-v2" => {
            Ok(zensim_experimental::preview_v0_5_tuner_v2())
        }
        "tuner-v3" | "tuner_v3" | "v0_5_tuner_v3" | "preview-v0.5-tuner-v3" => {
            Ok(zensim_experimental::preview_v0_5_tuner_v3())
        }
        "tuner-v4" | "tuner_v4" | "v0_5_tuner_v4" | "preview-v0.5-tuner-v4" => {
            Ok(zensim_experimental::preview_v0_5_tuner_v4())
        }
        "balanced-v2" | "balanced_v2" | "v0_5_balanced_v2" | "preview-v0.5-balanced-v2" => {
            Ok(zensim_experimental::preview_v0_5_balanced_v2())
        }
        "balanced-v3" | "balanced_v3" | "v0_5_balanced_v3" | "preview-v0.5-balanced-v3" => {
            Ok(zensim_experimental::preview_v0_5_balanced_v3())
        }
        "compression-v2"
        | "compression_v2"
        | "v0_5_compression_v2"
        | "preview-v0.5-compression-v2" => Ok(zensim_experimental::preview_v0_5_compression_v2()),
        "compression-v3"
        | "compression_v3"
        | "v0_5_compression_v3"
        | "preview-v0.5-compression-v3" => Ok(zensim_experimental::preview_v0_5_compression_v3()),
        other => bail!(
            "unknown profile '{other}'; expected v0_2 | v0_3 | codec-target | balanced | compression | ensemble | tuner | tuner-v2 | tuner-v3 | tuner-v4 | balanced-v2 | balanced-v3 | compression-v2 | compression-v3"
        ),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let codec = CodecKind::parse(&cli.codec)?;
    let profile = parse_profile(&cli.profile)?;

    // Load reference image as packed RGB8 via the `image` crate.
    let img = ImageReader::open(&cli.input)
        .with_context(|| format!("opening {}", cli.input.display()))?
        .with_guessed_format()
        .with_context(|| format!("guessing format of {}", cli.input.display()))?
        .decode()
        .with_context(|| format!("decoding {}", cli.input.display()))?;
    let rgb_img = img.to_rgb8();
    let width = rgb_img.width();
    let height = rgb_img.height();
    let rgb = rgb_img.into_raw();

    let spec = TargetSpec {
        target: cli.target,
        tolerance: cli.tolerance,
        max_iterations: cli.max_iterations,
        profile,
    };

    if !cli.quiet {
        eprintln!(
            "zensim-target: {}  {}x{}  codec={:?}  target={:.1}  tol=±{:.2}  profile={}",
            cli.input.display(),
            width,
            height,
            codec,
            cli.target,
            cli.tolerance,
            cli.profile,
        );
    }

    let result = target_search(&rgb, width, height, codec, spec)?;

    if !cli.quiet {
        eprintln!(
            "{:>4} {:>10} {:>10} {:>10}",
            "iter", "knob", "achieved", "bytes",
        );
        for p in &result.probes {
            eprintln!(
                "{:>4} {:>10.3} {:>10.3} {:>10}",
                p.iteration, p.knob, p.achieved_score, p.byte_count
            );
        }
    }

    println!(
        "codec={:?}  target={:.1}  achieved={:.3}  knob={:.3}  bytes={}  iters={}  converged={}",
        result.codec,
        result.target,
        result.achieved_score,
        result.final_knob,
        result.encoded.len(),
        result.iterations,
        result.converged,
    );

    if let Some(path) = cli.output {
        std::fs::write(&path, &result.encoded)
            .with_context(|| format!("writing encoded bytes to {}", path.display()))?;
        if !cli.quiet {
            eprintln!("wrote {} bytes to {}", result.encoded.len(), path.display());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_follow_the_library_and_keep_explicit_legacy_selection() {
        let cli = Cli::try_parse_from(["zensim-target", "input.png"]).unwrap();
        let library_default = TargetSpec::default().profile;
        assert_eq!(parse_profile(&cli.profile).unwrap(), library_default);
        assert_eq!(parse_profile("default").unwrap(), library_default);
        assert_eq!(
            parse_profile("tuner-v4").unwrap(),
            zensim_experimental::preview_v0_5_tuner_v4()
        );
    }
}
