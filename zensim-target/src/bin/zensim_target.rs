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
    /// ensemble (v0.5) | tuner (v0.5) | tuner-v2 (v0.5) | tuner-v3 (v0.5).
    /// Default is `tuner-v3` (PreviewV0_5TunerV3, EXP-CROSS-CODEC-V9
    /// ship, 2026-05-20) — passes every Tuner-trail gate apples-to-apples
    /// vs V2 AND adds **clean user-facing dial semantics**: typing "score
    /// 60" lands at JND (PJND anchor) exactly, "score 30" lands at JOD
    /// exactly, "score 0" hits the worst-codec q=5 floor, "score 100"
    /// hits near-lossless. V2's JND was 63 and dial range was [10, 90];
    /// V3 extends to full [0, 100] via 8-band anchor + post-network
    /// monotone PCHIP spline calibration. Use `tuner-v2` for the prior
    /// tuner ship if back-compat scores are required; `tuner` for the
    /// V_24 baseline; `v0_3` for the legacy default. The `balanced` /
    /// `compression` / `ensemble` ranking profiles are available for
    /// end-to-end evaluation but are NOT calibrated for quality-dial
    /// use — they produce non-monotonic scores in the target search loop.
    #[arg(long, default_value = "tuner-v3")]
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
    match s.to_ascii_lowercase().as_str() {
        "v0_2" | "v02" | "preview-v0.2" => Ok(ZensimProfile::PreviewV0_2),
        "v0_3" | "v03" | "preview-v0.3" => Ok(ZensimProfile::PreviewV0_3),
        "balanced" | "v0_5_balanced" | "preview-v0.5-balanced" => Ok(ZensimProfile::balanced()),
        "compression" | "v0_5_compression" | "preview-v0.5-compression" => {
            Ok(ZensimProfile::compression())
        }
        "ensemble" | "v0_5_ensemble" | "preview-v0.5-ensemble" => Ok(ZensimProfile::ensemble()),
        "tuner" | "v0_5_tuner" | "preview-v0.5-tuner" => Ok(ZensimProfile::tuner()),
        "tuner-v2" | "tuner_v2" | "v0_5_tuner_v2" | "preview-v0.5-tuner-v2" => {
            Ok(ZensimProfile::PreviewV0_5TunerV2)
        }
        "tuner-v3" | "tuner_v3" | "v0_5_tuner_v3" | "preview-v0.5-tuner-v3" | "default" => {
            Ok(ZensimProfile::PreviewV0_5TunerV3)
        }
        other => bail!(
            "unknown profile '{other}'; expected v0_2 | v0_3 | balanced | compression | ensemble | tuner | tuner-v2 | tuner-v3"
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
