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

    /// Desired zensim score in `0..=100`.
    #[arg(short, long, default_value_t = 70.0)]
    target: f32,

    /// Codec to use: zenjpeg | zenwebp | zenavif | zenjxl | zenpng.
    #[arg(short, long, default_value = "zenjpeg")]
    codec: String,

    /// Zensim profile: v0_2 | v0_3 | balanced (v0.5) | compression (v0.5) |
    /// ensemble (v0.5) | tuner (v0.5) | tuner-v2 (v0.5) | tuner-v3 (v0.5) |
    /// tuner-v4 (v0.5).
    /// Default is `tuner-v4` (PreviewV0_5TunerV4, EXP-CROSS-CODEC-V10
    /// ship, 2026-05-20) — the V9 dial reallocated: lossless = 100,
    /// JND = 80, JOD = 50, q=0 worst-codec floor = 0, pathological < 0
    /// (unclamped linear extrapolation). The wider perceptibility band
    /// (50 score units between JOD and JND vs V3's 30) gives the dial
    /// more resolution where compression product decisions live; the
    /// unclamped extrapolation lets the dial signal "broken / unreasonable"
    /// instead of collapsing to a tie at 0 for worst-case codec output.
    /// Use `tuner-v3` for the V9 JND=60 / JOD=30 dial; `tuner-v2` for
    /// the prior tuner ship; `tuner` for the V_24 baseline; `v0_3` for
    /// the legacy default. The `balanced` / `compression` / `ensemble`
    /// ranking profiles are available for end-to-end evaluation but are
    /// NOT calibrated for quality-dial use — they produce non-monotonic
    /// scores in the target search loop.
    #[arg(long, default_value = "tuner-v4")]
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
        "v0_2" | "v02" | "preview-v0.2" => Ok(ZensimProfile::PreviewV0_2),
        "v0_3" | "v03" | "preview-v0.3" => Ok(ZensimProfile::PreviewV0_3),
        "codec-target" | "codec_target" => Ok(ZensimProfile::codec_target()),
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
        "tuner-v4" | "tuner_v4" | "v0_5_tuner_v4" | "preview-v0.5-tuner-v4" | "default" => {
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
