# zensim-regress

Deterministic visual regression testing with chain-of-trust evidence for output differences.

Your image processing code produces slightly different pixels on x86_64 vs aarch64. Or after a dependency update. Or because you changed a rounding mode. `zensim-regress` tracks which outputs are acceptable, records forensic evidence of *how* they differ, and fails CI when something actually breaks.

## Quick start

The primary API is `ChecksumManager`. It handles hashing, comparison, tolerance, and TOML persistence.

```rust
use zensim_regress::manager::{ChecksumManager, CheckResult};

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

`ChecksumManager` reads two environment variables to control behavior:

| Variable | Effect |
|----------|--------|
| *(none)* | **Normal mode** — fail on mismatch |
| `UPDATE_CHECKSUMS=1` | Auto-accept results within tolerance, create baselines on first run |
| `REPLACE_CHECKSUMS=1` | Wipe all entries, set current output as new baseline |

Typical workflow:

1. First run: set `UPDATE_CHECKSUMS=1`. The manager creates TOML files and saves reference images.
2. Commit the `.toml` files and reference PNGs.
3. CI runs in normal mode. Exact hash matches pass instantly. Mismatches trigger zensim comparison against the reference image.
4. When you intentionally change output, run with `UPDATE_CHECKSUMS=1` again to accept the new checksums.

You can also force modes programmatically with `.with_update_mode_update()` or `.with_update_mode_replace()`.

## Architecture handling

Different CPUs produce different rounding for the same floating-point operations. The tolerance system handles this:

```toml
[tolerance]
max_delta = 1
min_similarity = 99.0

[tolerance.overrides.aarch64]
max_delta = 2
```

The manager detects the current architecture tag (e.g., `x86_64-avx2`, `aarch64`) and applies matching overrides. Each checksum entry records which architectures produced it, so you can track platform-specific outputs without false failures.

Override keys match as prefixes: `x86_64` matches `x86_64-avx2`, `x86_64-avx512`, etc.

## TOML file format

Each test gets a `.toml` file in the checksum directory:

```toml
name = "resize_bicubic_200x200"

[tolerance]
max_delta = 1
min_similarity = 95.0
max_alpha_delta = 0

[[checksum]]
id = "sea:a1b2c3d4e5f6789a"
confidence = 10
commit = "1540445a"
arch = ["x86_64-avx2"]
reason = "initial baseline"

[[checksum]]
id = "sea:b2c3d4e5f6789a0b"
confidence = 10
commit = "2e3f4a5b"
arch = ["aarch64"]
reason = "auto-accepted within tolerance"

[checksum.diff]
vs = "sea:a1b2c3d4e5f6789a"
score = 99.8
max_channel_delta = [1, 0, 0]

[meta]
pixel_format = "Rgba8"
```

Key fields:
- **confidence**: 10 = active, 0 = retired/wrong
- **diff.vs**: which checksum this was compared against (chain-of-trust)
- **diff.score**: zensim similarity score at the time of acceptance
- **meta**: arbitrary key-value pairs for consumer-specific data

## Remote storage

For large reference images that shouldn't live in git, configure S3/R2 remote storage:

| Variable | Purpose |
|----------|---------|
| `REGRESS_REFERENCE_URL` | Base URL for downloading references (e.g., `https://r2.example.com/refs`) |
| `REGRESS_UPLOAD_PREFIX` | Upload URL prefix (e.g., `https://r2.example.com/refs`) |
| `UPLOAD_REFERENCES` | Set to `1` to enable uploads |

```rust
let mgr = ChecksumManager::new("tests/checksums")
    .with_remote_storage_from_env();
```

The manager downloads references on cache miss and uploads new ones on accept. Downloads are cached locally in `{checksum_dir}/.remote-cache/`.

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

`ChecksumManager` can auto-save montages on mismatch as CI artifacts:

```rust
let mgr = ChecksumManager::new("tests/checksums")
    .with_diff_output("test-artifacts/diffs");
```

For terminal display, the `display` module renders images as sixel sequences (requires a sixel-capable terminal like foot, WezTerm, or mintty).

## Generators

Test image generators for synthetic regression tests:

| Function | Description |
|----------|-------------|
| `gradient(w, h)` | Smooth RGB gradient across width and height |
| `checkerboard(w, h, size)` | Alternating colored blocks |
| `mandelbrot(w, h)` | Mandelbrot set with smooth coloring |
| `value_noise(w, h, seed)` | Deterministic value noise |
| `color_blocks(w, h)` | Grid of distinct solid colors |
| `solid(w, h, r, g, b, a)` | Uniform solid color |
| `off_by_n(base, n, seed)` | Perturb base image by ±n per channel |

## Distortions

Deterministic pixel-level distortions for testing tolerance boundaries:

`uniform_shift`, `round_half_up`, `truncate_lsb`, `invert`, `channel_swap_rb`, `premultiply_alpha`, `straight_as_premultiplied`, `expand_to_256_levels`

## Module index

| Module | Description |
|--------|-------------|
| `manager` | `ChecksumManager` — primary API for check/accept/reject workflows |
| `checksum_file` | TOML serialization, `TestChecksumFile`, `ChecksumEntry`, tolerance specs |
| `testing` | `RegressionTolerance`, `RegressionReport`, `check_regression` |
| `hasher` | `ChecksumHasher` trait, `SeaHasher` (default 64-bit non-crypto hasher) |
| `arch` | Architecture detection and tag matching |
| `diff_image` | Amplified diff images and comparison montages |
| `display` | Sixel terminal rendering |
| `generators` | Synthetic test image generators |
| `distortions` | Deterministic pixel distortions |
| `remote` | S3/R2 reference image upload/download |
| `fetch` | HTTP fetcher for remote reference downloads |
| `upload` | Shell-based file uploader |
| `error` | `RegressError` error type |
