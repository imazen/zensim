//! Generate a standalone HTML regression report from test manifest files.
//!
//! Usage:
//!
//! ```bash
//! # Single platform (backward compatible):
//! regress-report --manifest-dir .image-cache/manifests/ \
//!                --diffs-dir .image-cache/diffs/ \
//!                --output regression-report.html
//!
//! # Multiple platforms:
//! regress-report \
//!     --platform linux-x64 --manifest-dir /path/to/linux/manifests --diffs-dir /path/to/linux/diffs \
//!     --platform osx-arm64 --manifest-dir /path/to/osx/manifests --diffs-dir /path/to/osx/diffs \
//!     -o merged-report.html
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

/// Accumulates arguments for one platform before being finalized.
struct PlatformBuilder {
    name: String,
    manifest_path: Option<PathBuf>,
    diffs_dir: Option<PathBuf>,
}

impl PlatformBuilder {
    fn new(name: String) -> Self {
        Self {
            name,
            manifest_path: None,
            diffs_dir: None,
        }
    }

    fn finalize(self) -> Result<zensim_regress::report::Platform, String> {
        let manifest_path = self.manifest_path.ok_or_else(|| {
            format!(
                "Platform '{}' has no --manifest-dir or --manifest",
                self.name
            )
        })?;
        Ok(zensim_regress::report::Platform {
            name: self.name,
            manifest_path,
            diffs_dir: self.diffs_dir,
        })
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let mut builders: Vec<PlatformBuilder> = Vec::new();
    let mut output: Option<PathBuf> = None;
    let mut combine_output: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--platform" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --platform requires a value");
                    return ExitCode::FAILURE;
                }
                builders.push(PlatformBuilder::new(args[i].clone()));
            }
            "--manifest-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --manifest-dir requires a value");
                    return ExitCode::FAILURE;
                }
                let path = PathBuf::from(&args[i]);
                current_builder(&mut builders).manifest_path = Some(path);
            }
            "--manifest" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --manifest requires a value");
                    return ExitCode::FAILURE;
                }
                let path = PathBuf::from(&args[i]);
                current_builder(&mut builders).manifest_path = Some(path);
            }
            "--diffs-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --diffs-dir requires a value");
                    return ExitCode::FAILURE;
                }
                let path = PathBuf::from(&args[i]);
                current_builder(&mut builders).diffs_dir = Some(path);
            }
            "--output" | "-o" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --output requires a value");
                    return ExitCode::FAILURE;
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "--combine-output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --combine-output requires a value");
                    return ExitCode::FAILURE;
                }
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

    // If no explicit --platform was given, try env vars as fallback
    if builders.is_empty() {
        let mut fallback = PlatformBuilder::new("local".to_string());
        if let Ok(dir) = std::env::var("REGRESS_MANIFEST_DIR") {
            fallback.manifest_path = Some(PathBuf::from(dir));
        } else if let Ok(file) = std::env::var("REGRESS_MANIFEST_PATH") {
            fallback.manifest_path = Some(PathBuf::from(file));
        }
        if fallback.manifest_path.is_some() {
            builders.push(fallback);
        }
    }

    if builders.is_empty() {
        eprintln!("Error: no manifest source specified.");
        eprintln!("Use --platform with --manifest-dir, or set REGRESS_MANIFEST_DIR.");
        print_usage();
        return ExitCode::FAILURE;
    }

    // Finalize all platform builders
    let mut platforms = Vec::new();
    for builder in builders {
        match builder.finalize() {
            Ok(p) => platforms.push(p),
            Err(e) => {
                eprintln!("Error: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    // Validate paths
    for p in &platforms {
        if p.manifest_path.is_dir() {
            // directory mode — ok
        } else if p.manifest_path.is_file() {
            // file mode — ok
        } else {
            eprintln!(
                "Error: manifest path does not exist for platform '{}': {}",
                p.name,
                p.manifest_path.display()
            );
            return ExitCode::FAILURE;
        }
    }

    // Optionally combine the first platform's manifest dir
    if let Some(ref combined) = combine_output
        && let Some(p) = platforms.first()
        && p.manifest_path.is_dir()
    {
        match zensim_regress::manifest::combine_manifest_dir(&p.manifest_path, combined) {
            Ok(n) => {
                eprintln!("Combined {n} manifest entries → {}", combined.display());
            }
            Err(e) => {
                eprintln!("Error combining manifests: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let output = output.unwrap_or_else(|| PathBuf::from("regression-report.html"));

    eprintln!(
        "Generating report for {} platform(s): {}",
        platforms.len(),
        platforms
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    match zensim_regress::report::generate_merged_report(&platforms) {
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

/// Get the current platform builder, creating a default "local" one if none exists.
fn current_builder(builders: &mut Vec<PlatformBuilder>) -> &mut PlatformBuilder {
    if builders.is_empty() {
        builders.push(PlatformBuilder::new("local".to_string()));
    }
    builders.last_mut().unwrap()
}

fn print_usage() {
    eprintln!(
        "\
Usage: regress-report [OPTIONS]

Options:
  --platform <NAME>        Start a new platform (default: local)
  --manifest-dir <DIR>     Directory of per-process manifest TSV files
  --manifest <FILE>        Single combined manifest TSV file
  --diffs-dir <DIR>        Directory containing diff PNG images
  --output, -o <FILE>      Output HTML file (default: regression-report.html)
  --combine-output <FILE>  Also write combined TSV to this path
  -h, --help               Print this help

Multiple platforms:
  Each --platform starts a new platform entry. The --manifest-dir,
  --manifest, and --diffs-dir that follow apply to that platform.

  regress-report \\
    --platform linux-x64 --manifest-dir /path/to/linux/manifests \\
    --platform osx-arm64 --manifest-dir /path/to/osx/manifests \\
    -o merged-report.html

Environment variables (used as fallback when no --platform given):
  REGRESS_MANIFEST_DIR     Manifest directory path
  REGRESS_MANIFEST_PATH    Manifest file path"
    );
}
