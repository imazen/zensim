# zensim-regress

Deterministic visual regression testing with chain-of-trust evidence for output differences.

Your image processing code produces slightly different pixels on x86_64 vs aarch64. Or after a dependency update. Or because you changed a rounding mode. `zensim-regress` tracks which outputs are acceptable, records forensic evidence of *how* they differ, and fails CI when something actually breaks.

## Quick start

The primary API is `ChecksumManager`. It handles hashing, comparison, tolerance, and persistence.

```rust
use zensim_regress::checksums::{ChecksumManager, CheckResult};

let mgr = ChecksumManager::new("tests/checksums");
let pixels: &[u8] = &actual_rgba_output;
let result = mgr.check_pixels("resize_bicubic", pixels, width, height).unwrap();

match &result {
    CheckResult::Match { .. } => {} // hash matched — pass
    CheckResult::WithinTolerance { report, .. } => {
        println!("within tolerance: {report}");
    }
    CheckResult::Failed { report, .. } => {
        panic!("regression: {result}");
    }
    CheckResult::NoBaseline { .. } => {
        println!("first run — run with UPDATE_CHECKSUMS=1");
    }
}
```

## Workflow modes

`ChecksumManager` reads the `UPDATE_CHECKSUMS` environment variable:

| Variable | Effect |
|----------|--------|
| *(none)* | **Normal mode** — fail on mismatch |
| `UPDATE_CHECKSUMS=1` | Auto-accept results within tolerance, create baselines on first run |

Typical workflow:

1. First run: set `UPDATE_CHECKSUMS=1`. The manager creates `.checksums` files and saves reference images.
2. Commit the `.checksums` files and reference PNGs.
3. CI runs in normal mode. Exact hash matches pass instantly. Mismatches trigger zensim comparison against the reference image.
4. When you intentionally change output, run with `UPDATE_CHECKSUMS=1` again to accept the new checksums.

For testing, use `ChecksumManager::with_modes(dir, true)` to enable update mode programmatically.

## Architecture handling

Different CPUs produce different rounding for the same floating-point operations. The tolerance system handles this:

```text
off-by-one [aarch64 max-delta:2]
```

The manager detects the current architecture tag (e.g., `x86_64-avx2`, `aarch64`) and applies matching overrides. Each checksum entry records which architectures produced it, so you can track platform-specific outputs without false failures.

Override keys match as prefixes: `x86_64` matches `x86_64-avx2`, `x86_64-avx512`, etc.

## `.checksums` file format

Each test module gets a `.checksums` file — a line-oriented log format:

```text
# zensim-regress checksums v1

[resize_bicubic_200x200] off-by-one
= sea:a1b2c3d4e5f6789a x86_64-avx2 "initial baseline"
~ sea:b2c3d4e5f6789a0b aarch64 vs sea:a1b2c3d4e5f6789a (zensim:99.80 (dissim 0.002), 0.3% pixels +/-1, category:rounding, balanced)
```

Key concepts:
- **`=`** entries are human-verified baselines
- **`~`** entries are auto-accepted within tolerance, with forensic diff evidence
- **`x`** entries are retired (wrong/superseded)
- **`vs`** links record which baseline a new checksum was compared against (chain of trust)
- **Section headers** specify the tolerance shorthand

## Remote storage

For large reference images that shouldn't live in git, configure S3/R2 remote storage:

| Variable | Purpose |
|----------|---------|
| `REGRESS_REFERENCE_URL` | Base URL for downloading references |
| `REGRESS_UPLOAD_PREFIX` | Upload destination prefix |
| `UPLOAD_REFERENCES` | Set to `1` to enable uploads |

```rust
let mgr = ChecksumManager::new("tests/checksums")
    .with_remote_storage_from_env();
```

Downloads are cached locally in `{checksums_dir}/.remote-cache/`.

## Image diffing

Generate amplified difference images and comparison montages:

```rust
use zensim_regress::diff_image::*;

// From RgbaImage
let diff = generate_diff_image(&expected, &actual, 10); // 10x amplification
let montage = create_comparison_montage(&expected, &actual, 10, 2);

// From raw RGBA bytes
let diff = generate_diff_image_raw(&exp_bytes, &act_bytes, w, h, 10);
let montage = create_comparison_montage_raw(&exp_bytes, &act_bytes, w, h, 10, 2);
```

Save montages automatically on mismatch:

```rust
let mgr = ChecksumManager::new("tests/checksums")
    .with_diff_output("test-artifacts/diffs");
```

For terminal display, the `display` module renders images as sixel sequences (requires a sixel-capable terminal like foot, WezTerm, or mintty).

## Generators

Deterministic test image generators:

| Function | Description |
|----------|-------------|
| `gradient(w, h)` | Smooth RGB gradient |
| `checkerboard(w, h, size)` | Alternating colored blocks |
| `mandelbrot(w, h)` | Mandelbrot set with smooth coloring |
| `value_noise(w, h, seed)` | Deterministic value noise |
| `color_blocks(w, h)` | Grid of distinct solid colors |
| `solid(w, h, r, g, b, a)` | Uniform solid color |
| `off_by_n(base, n, seed)` | Perturb base image by +/-n per channel |

## Distortions

Deterministic pixel-level distortions for testing tolerance boundaries:

`uniform_shift`, `round_half_up`, `truncate_lsb`, `invert`, `channel_swap_rb`, `premultiply_alpha`, `straight_as_premultiplied`, `expand_to_256_levels`

## Module index

| Module | Description |
|--------|-------------|
| `checksums` | `ChecksumManager`, `ChecksumsFile`, `ChecksumEntry`, `CheckResult` |
| `testing` | `RegressionTolerance`, `RegressionReport`, `check_regression` |
| `tolerance` | `ToleranceSpec`, `ToleranceOverride` for config-driven tolerances |
| `diff_summary` | Human-readable diff formatting and tolerance shorthand parsing |
| `hasher` | `ChecksumHasher` trait, `SeaHasher` (64-bit non-crypto) |
| `arch` | Architecture detection and tag matching |
| `diff_image` | Amplified diff images and comparison montages |
| `display` | Sixel terminal rendering |
| `generators` | Synthetic test image generators |
| `distortions` | Deterministic pixel distortions |
| `remote` | S3/R2 reference image storage config |
| `fetch` | HTTP fetcher for remote reference downloads |
| `upload` | Shell-based file uploader |
| `manifest` | TSV manifest writer for CI result aggregation |
| `petname` | Memorable names from hashes (e.g., `sea:a1b2...` → `keen-fox`) |
| `lock` | Advisory file locking for parallel test safety |
| `report` | HTML report generation from manifest data |
| `error` | `RegressError` error type |
