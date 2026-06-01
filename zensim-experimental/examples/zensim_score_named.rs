//! Score one (ref, dist) pair against a NAMED zensim profile.
//!
//! Usage: `zensim_score_named PROFILE_NAME ref.png dist.png [--codec NAME] [--per-codec-calibration on|off]`
//!
//! `PROFILE_NAME` ∈ {
//!   v0_2, v0_3, v0_4, v0_5,
//!   v0_5_balanced, v0_5_compression, v0_5_ensemble, v0_5_tuner,
//!   latest
//! }
//!
//! Optional flags:
//! - `--codec NAME` — codec the distorted image was produced by
//!   (`jpeg`, `webp`, `avif`, `jxl`, `png`). Triggers per-codec
//!   score calibration when paired with `--per-codec-calibration on`
//!   (default ON for `v0_5_tuner`, OFF for legacy profiles).
//! - `--per-codec-calibration on|off` — explicitly enable / disable
//!   per-codec calibration. When ON and a `--codec` is supplied, the
//!   profile's raw output is rescaled per-codec so that "score=63"
//!   means the empirical PJND across all codecs. See
//!   `zensim::codec_calibration` for the math + provenance.
//!
//! Used by `cross_codec_consistency.py` to binary-search the q value
//! achieving a target zensim score under each shipping profile.

use std::env;
use std::process::ExitCode;
use zensim::{CodecCalibration, RgbSlice, Zensim, ZensimProfile};

fn print_usage(arg0: &str) {
    eprintln!(
        "Usage: {arg0} PROFILE_NAME ref.png dist.png [--codec NAME] [--per-codec-calibration on|off]"
    );
    eprintln!(
        "  PROFILE_NAME ∈ {{v0_2, v0_3, v0_4, v0_5, v0_5_balanced, v0_5_compression, v0_5_ensemble, v0_5_tuner, latest}}"
    );
    eprintln!("  --codec NAME — codec used for the distorted image (jpeg/webp/avif/jxl/png).");
    eprintln!(
        "  --per-codec-calibration on|off — explicitly toggle calibration (default ON for v0_5_tuner)."
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        print_usage(&args[0]);
        return ExitCode::FAILURE;
    }

    let profile = match args[1].as_str() {
        "v0_2" => ZensimProfile::PreviewV0_2,
        "v0_3" => ZensimProfile::A,
        "v0_4" => zensim_experimental::preview_v0_4(),
        "v0_5" => zensim_experimental::preview_v0_5(),
        "v0_5_balanced" => zensim_experimental::preview_v0_5_balanced(),
        "v0_5_compression" => zensim_experimental::preview_v0_5_compression(),
        "v0_5_ensemble" => zensim_experimental::preview_v0_5_ensemble(),
        "v0_5_tuner" => zensim_experimental::preview_v0_5_tuner(),
        "latest" => ZensimProfile::A,
        other => {
            eprintln!("unknown profile: {other}");
            print_usage(&args[0]);
            return ExitCode::FAILURE;
        }
    };
    let ref_path = &args[2];
    let dist_path = &args[3];

    // Parse optional flags
    let mut codec_name: Option<String> = None;
    let mut calibration_flag: Option<bool> = None; // None => default
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--codec" => {
                if i + 1 >= args.len() {
                    eprintln!("--codec requires a NAME");
                    return ExitCode::FAILURE;
                }
                codec_name = Some(args[i + 1].clone());
                i += 2;
            }
            "--per-codec-calibration" => {
                if i + 1 >= args.len() {
                    eprintln!("--per-codec-calibration requires on|off");
                    return ExitCode::FAILURE;
                }
                calibration_flag = match args[i + 1].as_str() {
                    "on" | "true" | "1" => Some(true),
                    "off" | "false" | "0" => Some(false),
                    other => {
                        eprintln!("--per-codec-calibration must be on|off, got: {other}");
                        return ExitCode::FAILURE;
                    }
                };
                i += 2;
            }
            other => {
                eprintln!("unknown argument: {other}");
                print_usage(&args[0]);
                return ExitCode::FAILURE;
            }
        }
    }

    // Default calibration policy: ON for PreviewV0_5Tuner only, OFF elsewhere.
    let calibration_enabled =
        calibration_flag.unwrap_or(profile == zensim_experimental::preview_v0_5_tuner());

    let img1 = image::open(ref_path).expect("open ref");
    let img2 = image::open(dist_path).expect("open dist");
    let img1 = img1.to_rgb8();
    let img2 = img2.to_rgb8();
    let w = img1.width() as usize;
    let h = img1.height() as usize;
    if img2.width() as usize != w || img2.height() as usize != h {
        eprintln!(
            "dimension mismatch: ref {}x{} vs dist {}x{}",
            w,
            h,
            img2.width(),
            img2.height()
        );
        return ExitCode::FAILURE;
    }
    let src: Vec<[u8; 3]> = img1.pixels().map(|p| p.0).collect();
    let dst: Vec<[u8; 3]> = img2.pixels().map(|p| p.0).collect();
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    let z = Zensim::new(profile);
    let raw = match z.compute(&s, &d) {
        Ok(r) => r.score(),
        Err(e) => {
            eprintln!("zensim error: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    // Apply per-codec calibration when enabled AND the profile is the
    // Tuner. Other profiles either are not dial-honest (Balanced /
    // Compression / Ensemble — see CLAUDE.md "tied-rate dead zone")
    // or pre-date the per-codec fit; passing the flag with them is a
    // no-op so caller code can stay uniform.
    let calibrated = if calibration_enabled && profile == zensim_experimental::preview_v0_5_tuner()
    {
        if let Some(codec) = &codec_name {
            let cal = CodecCalibration::PREVIEW_V0_5_TUNER;
            match cal.lookup(codec) {
                Some(affine) => affine.apply(raw as f32) as f64,
                None => {
                    eprintln!("warning: unknown codec name `{codec}` — no calibration applied");
                    raw
                }
            }
        } else {
            // Calibration ON but no codec given: still return raw,
            // so callers can opt in by passing --codec.
            raw
        }
    } else {
        raw
    };

    println!("{calibrated:.6}");
    ExitCode::SUCCESS
}
