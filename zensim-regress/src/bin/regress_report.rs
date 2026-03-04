//! Generate a standalone HTML regression report from test manifest files.
//!
//! Usage:
//!
//! ```bash
//! # From a manifest directory (nextest-friendly):
//! regress-report --manifest-dir .image-cache/manifests/ \
//!                --diffs-dir .image-cache/diffs/ \
//!                --output regression-report.html
//!
//! # From a single combined TSV:
//! regress-report --manifest test-manifest.tsv \
//!                --diffs-dir .image-cache/diffs/ \
//!                --output regression-report.html
//!
//! # With a platform name:
//! regress-report --manifest-dir .image-cache/manifests/ \
//!                --diffs-dir .image-cache/diffs/ \
//!                --platform linux-x64 \
//!                --output regression-report.html
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let mut manifest_dir: Option<PathBuf> = None;
    let mut manifest_file: Option<PathBuf> = None;
    let mut diffs_dir: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut platform_name: Option<String> = None;
    let mut combine_output: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--manifest-dir" => {
                i += 1;
                manifest_dir = Some(PathBuf::from(&args[i]));
            }
            "--manifest" => {
                i += 1;
                manifest_file = Some(PathBuf::from(&args[i]));
            }
            "--diffs-dir" => {
                i += 1;
                diffs_dir = Some(PathBuf::from(&args[i]));
            }
            "--output" | "-o" => {
                i += 1;
                output = Some(PathBuf::from(&args[i]));
            }
            "--platform" => {
                i += 1;
                platform_name = Some(args[i].clone());
            }
            "--combine-output" => {
                i += 1;
                combine_output = Some(PathBuf::from(&args[i]));
            }
            "--help" | "-h" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_usage();
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    // Determine manifest source
    let manifest_path = if let Some(dir) = &manifest_dir {
        if !dir.is_dir() {
            eprintln!("Error: manifest directory does not exist: {}", dir.display());
            return ExitCode::FAILURE;
        }

        // Optionally combine into a single TSV
        if let Some(ref combined) = combine_output {
            match zensim_regress::manifest::combine_manifest_dir(dir, combined) {
                Ok(n) => eprintln!("Combined {n} manifest entries → {}", combined.display()),
                Err(e) => {
                    eprintln!("Error combining manifests: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }

        dir.clone()
    } else if let Some(file) = &manifest_file {
        if !file.is_file() {
            eprintln!("Error: manifest file does not exist: {}", file.display());
            return ExitCode::FAILURE;
        }
        file.clone()
    } else {
        // Try environment variables
        if let Ok(dir) = std::env::var("REGRESS_MANIFEST_DIR") {
            let dir = PathBuf::from(dir);
            if dir.is_dir() {
                dir
            } else {
                eprintln!("Error: REGRESS_MANIFEST_DIR does not exist: {}", dir.display());
                return ExitCode::FAILURE;
            }
        } else if let Ok(file) = std::env::var("REGRESS_MANIFEST_PATH") {
            PathBuf::from(file)
        } else {
            eprintln!("Error: no manifest source specified.");
            eprintln!("Use --manifest-dir, --manifest, or set REGRESS_MANIFEST_DIR.");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    let output = output.unwrap_or_else(|| PathBuf::from("regression-report.html"));
    let name = platform_name.unwrap_or_else(|| "local".to_string());

    let platform = zensim_regress::report::Platform {
        name,
        manifest_path,
        diffs_dir,
    };

    match zensim_regress::report::generate_merged_report(&[platform]) {
        Ok(html) => {
            if let Err(e) = std::fs::write(&output, &html) {
                eprintln!("Error writing report: {e}");
                return ExitCode::FAILURE;
            }
            eprintln!(
                "Wrote regression report ({} bytes) → {}",
                html.len(),
                output.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error generating report: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    eprintln!(
        "\
Usage: regress-report [OPTIONS]

Options:
  --manifest-dir <DIR>     Directory of per-process manifest TSV files
  --manifest <FILE>        Single combined manifest TSV file
  --diffs-dir <DIR>        Directory containing diff PNG images
  --output, -o <FILE>      Output HTML file (default: regression-report.html)
  --platform <NAME>        Platform name for the report header (default: local)
  --combine-output <FILE>  Also write combined TSV to this path
  -h, --help               Print this help

Environment variables (used as fallback):
  REGRESS_MANIFEST_DIR     Manifest directory path
  REGRESS_MANIFEST_PATH    Manifest file path"
    );
}
