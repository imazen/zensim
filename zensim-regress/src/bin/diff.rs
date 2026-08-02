//! `zensim-diff` — ad-hoc image diffing CLI (issue #14).
//!
//! A thin wrapper over the `zensim_regress::diff_image` primitives, so the
//! structural/pixel diff + montage machinery is usable on any two PNGs
//! without writing a one-off Rust binary:
//!
//! ```bash
//! zensim-diff expected.png actual.png                    # 4-panel montage → diff.png
//! zensim-diff a.png b.png --mode structural -o s.png     # high-pass residual diff
//! zensim-diff a.png b.png --mode spatial --grid 8x8      # per-region stats (text)
//! zensim-diff a.png b.png --score --json                 # + zensim score, as JSON
//! ```
//!
//! Modes:
//! - `montage` (default): labeled 2×2 grid `[expected | actual | pixel diff |
//!   structural diff]` with annotation text and spatial heatmap
//!   (`MontageOptions::render` — tiny inputs pixelate-upscale per
//!   `--min-panel`).
//! - `pixel`: amplified per-channel absolute diff (`generate_diff_image`).
//! - `structural`: high-pass residual diff with cyan/orange = structure
//!   missing/added semantics (`generate_structural_diff`).
//! - `spatial`: per-region grid stats (`spatial_analysis`) as text or JSON.
//!
//! Manual arg parsing, no new dependencies (matches `regress-report` style).

use std::process::ExitCode;

use zensim_regress::Bitmap;
use zensim_regress::diff_image::{
    AnnotationText, MontageOptions, generate_diff_image, generate_structural_diff, spatial_analysis,
};

const USAGE: &str = "\
zensim-diff — ad-hoc image diff (zensim-regress)

USAGE:
    zensim-diff <expected.png> <actual.png> [OPTIONS]

OPTIONS:
    -o, --output <path>   Output PNG path (default: diff.png; unused for --mode spatial)
    --mode <M>            montage | pixel | structural | spatial   (default: montage)
    --amp <N>             Diff amplification factor 1-255 (default: 10)
    --blur <N>            Structural-diff blur radius (default: 3)
    --gap <N>             Montage gap in pixels (default: 6)
    --min-panel <N>       Pixelate-upscale threshold for small images (default: 256; 0 disables)
    --label <S>           Title text on the montage annotation strip
    --grid <C>x<R>        Spatial grid subdivision (default: 8x8)
    --score               Also compute the zensim score (codec_target profile)
    --json                Emit spatial stats / score as JSON on stdout
    -h, --help            Show this help
";

struct Args {
    expected: String,
    actual: String,
    output: String,
    mode: String,
    amp: u8,
    blur: u32,
    gap: u32,
    min_panel: u32,
    label: Option<String>,
    grid: (u32, u32),
    score: bool,
    json: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut positional: Vec<String> = Vec::new();
    let mut a = Args {
        expected: String::new(),
        actual: String::new(),
        output: "diff.png".to_string(),
        mode: "montage".to_string(),
        amp: 10,
        blur: 3,
        gap: 6,
        min_panel: 256,
        label: None,
        grid: (8, 8),
        score: false,
        json: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let need = |i: usize, argv: &[String], flag: &str| -> Result<String, String> {
        argv.get(i)
            .cloned()
            .ok_or_else(|| format!("{flag} requires a value"))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => return Err(String::new()),
            "-o" | "--output" => {
                i += 1;
                a.output = need(i, &argv, "--output")?;
            }
            "--mode" => {
                i += 1;
                a.mode = need(i, &argv, "--mode")?;
                if !matches!(
                    a.mode.as_str(),
                    "montage" | "pixel" | "structural" | "spatial"
                ) {
                    return Err(format!("unknown --mode '{}'", a.mode));
                }
            }
            "--amp" => {
                i += 1;
                a.amp = need(i, &argv, "--amp")?
                    .parse()
                    .map_err(|e| format!("--amp: {e}"))?;
            }
            "--blur" => {
                i += 1;
                a.blur = need(i, &argv, "--blur")?
                    .parse()
                    .map_err(|e| format!("--blur: {e}"))?;
            }
            "--gap" => {
                i += 1;
                a.gap = need(i, &argv, "--gap")?
                    .parse()
                    .map_err(|e| format!("--gap: {e}"))?;
            }
            "--min-panel" => {
                i += 1;
                a.min_panel = need(i, &argv, "--min-panel")?
                    .parse()
                    .map_err(|e| format!("--min-panel: {e}"))?;
            }
            "--label" => {
                i += 1;
                a.label = Some(need(i, &argv, "--label")?);
            }
            "--grid" => {
                i += 1;
                let v = need(i, &argv, "--grid")?;
                let (c, r) = v
                    .split_once('x')
                    .ok_or_else(|| format!("--grid must be COLSxROWS, got '{v}'"))?;
                a.grid = (
                    c.parse().map_err(|e| format!("--grid cols: {e}"))?,
                    r.parse().map_err(|e| format!("--grid rows: {e}"))?,
                );
                if a.grid.0 == 0 || a.grid.1 == 0 {
                    return Err("--grid dimensions must be > 0".into());
                }
            }
            "--score" => a.score = true,
            "--json" => a.json = true,
            other if other.starts_with('-') => {
                return Err(format!("unknown option '{other}'"));
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    if positional.len() != 2 {
        return Err(format!(
            "expected exactly 2 image paths, got {}",
            positional.len()
        ));
    }
    a.actual = positional.pop().unwrap();
    a.expected = positional.pop().unwrap();
    Ok(a)
}

/// Score the pair with the shipped default profile (`codec_target`).
fn zensim_score(expected: &Bitmap, actual: &Bitmap) -> Result<f64, String> {
    use zensim::{RgbaSlice, Zensim, ZensimProfile};
    let (w, h) = expected.dimensions();
    let e: &[[u8; 4]] = bytemuck::cast_slice(expected.as_raw());
    let a: &[[u8; 4]] = bytemuck::cast_slice(actual.as_raw());
    let z = Zensim::new(ZensimProfile::codec_target());
    let exp = RgbaSlice::try_new(e, w as usize, h as usize).map_err(|e| format!("{e}"))?;
    let act = RgbaSlice::try_new(a, w as usize, h as usize).map_err(|e| format!("{e}"))?;
    z.compute(&exp, &act)
        .map(|r| r.score())
        .map_err(|e| format!("zensim compute failed: {e}"))
}

fn file_stem(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            if msg.is_empty() {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {msg}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    let expected = match Bitmap::open(&args.expected) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read '{}': {e:?}", args.expected);
            return ExitCode::FAILURE;
        }
    };
    let actual = match Bitmap::open(&args.actual) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read '{}': {e:?}", args.actual);
            return ExitCode::FAILURE;
        }
    };

    // pixel / structural / spatial need equal dimensions (montage renders a
    // shared-canvas comparison for mismatched dims).
    if args.mode != "montage" && expected.dimensions() != actual.dimensions() {
        eprintln!(
            "error: dimension mismatch ({}x{} vs {}x{}) — only --mode montage supports it",
            expected.width(),
            expected.height(),
            actual.width(),
            actual.height()
        );
        return ExitCode::FAILURE;
    }

    // Score is computable for equal dims only.
    let score: Option<f64> = if args.score {
        if expected.dimensions() != actual.dimensions() {
            eprintln!("note: --score skipped (dimension mismatch)");
            None
        } else {
            match zensim_score(&expected, &actual) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    } else {
        None
    };

    match args.mode.as_str() {
        "pixel" => {
            let img = generate_diff_image(&expected, &actual, args.amp);
            if let Err(e) = img.save(&args.output) {
                eprintln!("error: cannot write '{}': {e:?}", args.output);
                return ExitCode::FAILURE;
            }
            eprintln!("wrote {}", args.output);
        }
        "structural" => {
            let img = generate_structural_diff(&expected, &actual, args.blur, args.amp);
            if let Err(e) = img.save(&args.output) {
                eprintln!("error: cannot write '{}': {e:?}", args.output);
                return ExitCode::FAILURE;
            }
            eprintln!("wrote {}", args.output);
        }
        "spatial" => {
            let sa = spatial_analysis(
                expected.as_raw(),
                actual.as_raw(),
                expected.width(),
                expected.height(),
                args.grid.0,
                args.grid.1,
            );
            if args.json {
                let mut out = String::from("{");
                if let Some(s) = score {
                    out.push_str(&format!("\"zensim_score\":{s:.4},"));
                }
                out.push_str(&format!(
                    "\"cols\":{},\"rows\":{},\"regions\":[",
                    sa.cols, sa.rows
                ));
                for (i, r) in sa.regions.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&format!(
                        "{{\"col\":{},\"row\":{},\"pixels_differing\":{:.6},\"avg_delta\":{:.4},\"max_delta\":{},\"expected_variance\":{:.4},\"actual_variance\":{:.4}}}",
                        r.col,
                        r.row,
                        r.pixels_differing,
                        r.avg_delta,
                        r.max_delta,
                        r.expected_variance,
                        r.actual_variance
                    ));
                }
                out.push_str("]}");
                println!("{out}");
            } else {
                if let Some(s) = score {
                    println!("zensim_score: {s:.4}");
                }
                println!("{sa}");
            }
            // No image output in spatial mode.
            return ExitCode::SUCCESS;
        }
        _ /* montage */ => {
            let mut primary_lines: Vec<(String, [u8; 4])> = Vec::new();
            if let Some(s) = score {
                primary_lines.push((format!("zensim: {s:.2}"), [170, 170, 170, 255]));
            }
            let mut annotation = AnnotationText::empty();
            annotation.title = Some(args.label.clone().unwrap_or_else(|| {
                format!(
                    "{} vs {}",
                    file_stem(&args.expected),
                    file_stem(&args.actual)
                )
            }));
            annotation.primary_lines = primary_lines;
            let mut opts = MontageOptions::default();
            opts.amplification = args.amp;
            opts.gap = args.gap;
            opts.min_panel_size = args.min_panel;
            opts.expected_label = Some(file_stem(&args.expected));
            opts.actual_label = Some(file_stem(&args.actual));
            let img = opts.render(&expected, &actual, &annotation);
            if let Err(e) = img.save(&args.output) {
                eprintln!("error: cannot write '{}': {e:?}", args.output);
                return ExitCode::FAILURE;
            }
            eprintln!("wrote {}", args.output);
        }
    }

    if let Some(s) = score {
        if args.json && args.mode != "spatial" {
            println!(
                "{{\"zensim_score\":{s:.4},\"output\":\"{}\"}}",
                json_escape(&args.output)
            );
        } else if args.mode != "spatial" {
            println!("zensim_score: {s:.4}");
        }
    }
    ExitCode::SUCCESS
}
