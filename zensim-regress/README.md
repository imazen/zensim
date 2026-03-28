# zensim-regress

Deterministic visual regression testing with chain-of-trust evidence for output differences.

Your image processing code produces slightly different pixels on x86_64 vs aarch64. Or after a dependency update. Or because you changed a rounding mode. `zensim-regress` tracks which outputs are acceptable, records forensic evidence of *how* they differ, and fails CI when something actually breaks.

## Adding your first test

Add the dependency:

```toml
[dev-dependencies]
zensim-regress = "0.2"
```

Write a test that checks pixel output against a known-good baseline:

```rust
use zensim_regress::checksums::{ChecksumManager, CheckResult};

#[test]
fn test_resize_output() {
    let mgr = ChecksumManager::new("tests/checksums".as_ref());

    // Your image processing code produces RGBA pixels
    let (pixels, width, height) = my_resize_function(input, 200, 200);

    let result = mgr.check_pixels(
        "resize",           // module — groups tests into one .checksums file
        "bicubic",          // test name
        "200x200",          // detail — distinguishes variants within a test
        &pixels, width, height,
        None,               // tolerance (None = exact match)
    ).unwrap();

    assert!(result.passed(), "{result}");
}
```

The first time you run this, there's no baseline — `result.passed()` returns false and the `Display` impl tells you what to do:

```text
NO BASELINE (first run)
  Suggested line: = sunny-crab-a4839401fa:sea  x86_64-avx2  @773c807  new-baseline
```

Run with `UPDATE_CHECKSUMS=1` to create the baseline automatically:

```bash
UPDATE_CHECKSUMS=1 cargo test test_resize_output
```

This creates `tests/checksums/resize.checksums` with the hash entry. Commit that file. From now on, the test passes instantly when the hash matches.

## Oracle testing (recommended for per-pixel operations)

For operations with a scalar definition (gamma, blend, color convert, resize kernel), oracle testing verifies correctness without golden files: apply the operation to the full image, then apply a scalar reference to individual pixels, and compare results at sampled coordinates.

Two layers are available:

1. **Standalone** — pure numeric comparison, no baselines needed:

```rust
use zensim_regress::oracle::*;
use zensim_regress::generators;

#[test]
fn test_gamma_oracle() {
    let input = generators::gradient(256, 256);
    let gamma = 2.2;

    let report = oracle_check_u8(
        &input, 256, 256, 4,
        |buf, w, h| apply_gamma(buf, w, h, gamma),       // image operation
        |px| px.iter().map(|&v| v.powf(gamma)).collect(), // scalar reference
        &default_test_coords(256, 256),
        OracleTolerance::AbsEpsilon(1.0 / 255.0),
    );
    assert!(report.passed, "{report}");
}
```

2. **Tracked** — scalar oracle *plus* full-image checksum tracking, diff generation, and remote reference storage via `ChecksumManager`:

```rust
use zensim_regress::oracle::*;
use zensim_regress::checksums::ChecksumManager;
use zensim_regress::tolerance::ToleranceSpec;

#[test]
fn test_gamma_tracked() {
    let mgr = ChecksumManager::new("tests/checksums".as_ref())
        .with_diff_output("test-artifacts/diffs")
        .with_manifest_from_env();

    let input = generators::gradient(256, 256);

    let report = oracle_check_tracked(
        &mgr, "color", "gamma", "2.2_gradient",
        &input, 256, 256,
        |buf, w, h| apply_gamma(buf, w, h, 2.2),
        |px| px.iter().map(|&v| v.powf(2.2)).collect(),
        &default_test_coords(256, 256),
        OracleTolerance::AbsEpsilon(1.0 / 255.0),
        Some(&ToleranceSpec::off_by_one()),
    ).unwrap();
    assert!(report.passed, "{report}");
}
```

The tracked variant catches regressions in edge handling, padding, and multi-pixel dependencies that scalar sampling alone would miss — the full output image is compared against stored baselines and remote references.

### Writing the scalar reference

The `scalar_op` is your **ground truth** — an independent implementation of the operation's definition. It must NOT be a `_scalar` variant from `#[autoversion]` (that's the code under test, subject to compiler auto-vectorization and FP reassociation). Write a separate reference, ideally in f64 for maximum precision:

```rust
// GOOD: independent reference
|px| vec![px[0].powf(1.0 / 2.2), px[1].powf(1.0 / 2.2), px[2].powf(1.0 / 2.2), px[3]]

// BAD: calling the autoversion scalar variant — this IS the code under test
|px| { let v = linear_to_srgb_scalar(ScalarToken, px[0] as f32); vec![v as f64, ...] }
```

f64 is not mandatory — what matters is independence from the code under test. But f64 avoids ambiguity: if the image op works in f32 and the reference in f64, any delta is the image code's rounding, not the reference's.

### The image_op calls your dispatcher

When your code uses archmage, the `image_op` should call the public dispatcher (the function without a token parameter or tier suffix). This tests the real code path:

```rust
// Your library has: #[autoversion] pub fn apply_curve(data: &mut [f32]) { ... }
// Generated: apply_curve_v3, apply_curve_neon, apply_curve_scalar, plus dispatcher

oracle_check_f32(
    &input, w, h, 3,
    |buf, w, h| { let mut out = buf.to_vec(); apply_curve(&mut out); out },
    |px| reference_curve_f64(px),  // pure f64 math
    &default_test_coords(w, h),
    OracleTolerance::AbsEpsilon(1e-5),
);
```

If the function takes an explicit token parameter (inner `#[arcane]` functions), summon the token in the closure or wrap it in a dispatcher. See the `oracle` module docs for details.

**Use oracle testing in:** zenresize, zenfilters, zenpixels-convert, linear-srgb, zenblend — any crate with per-pixel operations that have a scalar definition.

## SIMD consistency testing (recommended for archmage users)

For crates using archmage (`#[arcane]`, `#[autoversion]`, `#[rite]`), SIMD consistency testing runs the same operation under every available SIMD tier and verifies all produce equivalent output.

```toml
[dev-dependencies]
zensim-regress = { version = "0.2", features = ["archmage"] }
```

```rust
use zensim_regress::simd::*;
use zensim_regress::RegressionTolerance;
use archmage::testing::CompileTimePolicy;

#[test]
fn resize_simd_consistency() {
    let input = load_test_image();

    let report = check_simd_consistency(
        || {
            let output = resize(&input, 256, 256, Filter::Lanczos3);
            (output.to_rgba8(), 256, 256)
        },
        &RegressionTolerance::off_by_one(),
        CompileTimePolicy::Warn,
    ).unwrap();

    assert!(report.all_passed, "{report}");
}
```

This wraps `archmage::testing::for_each_token_permutation()` — it disables SIMD tokens in every valid combination (respecting the cascade hierarchy), runs your operation each time, and compares outputs against the highest-tier result using zensim-regress tolerances.

Catches: vectorization bugs, accumulator ordering differences, NaN handling divergence, and any case where the SIMD path produces different results from scalar.

### Call the dispatcher, not the tier variant

The operation closure must call your **public dispatcher** — the function that dispatches internally via `incant!` or `#[autoversion]`. Do not call a specific tier variant:

```rust
// GOOD: dispatcher falls back as tokens are disabled
|| { let out = my_resize(&input, 256, 256); (out.to_rgba8(), 256, 256) }

// BAD: always calls V3 regardless of which tokens are disabled
|| { let t = X64V3Token::summon().unwrap(); (my_resize_v3(t, &input, 256, 256), 256, 256) }
```

`for_each_token_permutation` disables tokens at the process level — `summon()` returns `None` for disabled tokens, so dispatchers naturally fall back to lower tiers.

For functions that take an explicit token (`#[arcane]` inner functions), tokens are `Copy` — summon outside and capture, or use `incant!` in the closure. See the `simd` module docs for patterns.

### Skipping crypto permutations

On x86, crypto tokens (PCLMUL, AES) are independent from compute tiers. Image processing code doesn't use them, so toggling them combinatorially just multiplies test count. Use `CryptoGrouping::Skip`:

```rust
use zensim_regress::simd::{check_simd_consistency_opts, CryptoGrouping};

let report = check_simd_consistency_opts(
    || { /* ... */ },
    &RegressionTolerance::off_by_one(),
    CompileTimePolicy::Warn,
    CryptoGrouping::Skip,  // only permute compute tiers
).unwrap();
```

`Clump` (default) tests crypto as a single on/off group, separate from compute tiers. `Skip` excludes crypto from permutations entirely. `Combinatorial` is the full cross-product (use when your code actually uses crypto instructions).

### Combining with oracle testing

Oracle and SIMD consistency are complementary — oracle verifies *correctness* (matches the math), SIMD consistency verifies *equivalence* (all tiers match each other). A bug where all tiers produce the same wrong answer passes SIMD consistency but fails oracle. Use both.

**CI integration:** For full permutation coverage, compile with `testable_dispatch` on archmage and use `CompileTimePolicy::Fail`:

```toml
[dev-dependencies]
archmage = { version = "0.9", features = ["testable_dispatch"] }
```

Without `testable_dispatch`, tokens compiled with `-Ctarget-cpu=native` can't be disabled — you'll get warnings and reduced coverage. In CI (without `-Ctarget-cpu`), all tokens are testable by default.

**Use SIMD consistency testing in:** any crate using archmage for SIMD dispatch — zenresize, zenfilters, zenpixels-convert, linear-srgb, zenjpeg, zenwebp, zenpng, zenjxl-decoder, fast-ssim2, zensim.

## What happens on mismatch

When the output changes — different platform, updated dependency, code change — the manager compares the new output against the reference image using zensim (a perceptual similarity metric). There are four outcomes:

| Result | Meaning | Action |
|--------|---------|--------|
| `Match` | Hash matches a known entry | Pass. Nothing to do. |
| `WithinTolerance` | Hash differs, but perceptual diff is within tolerance | Pass. Auto-accepted in UPDATE mode. |
| `NoBaseline` | No `.checksums` entry exists | Fail. Run with `UPDATE_CHECKSUMS=1`. |
| `Failed` | Perceptual diff exceeds tolerance | Fail. Investigate the regression. |

`CheckResult` implements `Display` with forensic detail — zensim score, per-channel max delta, percentage of pixels affected, error classification, and a suggested `.checksums` line you can paste in manually if you prefer not to use UPDATE mode.

## Tolerances

By default, tests require an exact hash match. When you need to accept platform-specific rounding differences, pass a `ToleranceSpec`:

```rust
use zensim_regress::tolerance::ToleranceSpec;

// Accept off-by-one rounding differences (common across architectures)
let tol = ToleranceSpec::off_by_one();

let result = mgr.check_pixels(
    "resize", "bicubic", "200x200",
    &pixels, width, height,
    Some(&tol),
).unwrap();
```

`off_by_one()` allows: per-channel delta up to 1, any number of pixels affected, zensim score >= 85 (very permissive on perceptual similarity since off-by-one is imperceptible).

For custom tolerances, build a `ToleranceSpec` directly:

```rust
let tol = ToleranceSpec {
    max_delta: 2,              // max per-channel difference (0-255)
    min_similarity: 95.0,      // minimum zensim score (0-100, 100 = identical)
    max_pixels_different: 0.5, // fraction of pixels that may differ (0.0-1.0)
    max_alpha_delta: 0,        // alpha channel tolerance
    ignore_alpha: false,
    overrides: Default::default(),
};
```

### Architecture-specific overrides

Some platforms produce larger deltas. Override tolerance for specific architectures:

```rust
use zensim_regress::tolerance::{ToleranceSpec, ToleranceOverride};

let mut tol = ToleranceSpec::off_by_one();
tol.overrides.insert("aarch64".to_string(), ToleranceOverride {
    max_delta: Some(3),        // aarch64 gets wider delta allowance
    min_similarity: None,      // inherit from base
    max_pixels_different: None,
    max_alpha_delta: None,
});
```

Override keys match as prefixes: `x86_64` matches `x86_64-avx2`, `x86_64-avx512`, etc. The architecture tag is detected automatically (`detect_arch_tag()` returns values like `x86_64-avx512`, `x86_64-avx2`, `aarch64`).

### Tolerance shorthand

Tolerances serialize to a compact string format used in `.checksums` files:

| Shorthand | Meaning |
|-----------|---------|
| `identical` | Exact match (delta 0, score 100) |
| `off-by-one` | Delta 1, score >= 85, any pixels |
| `max-delta:2 zensim:95 (dissim 0.05)` | Custom delta + perceptual threshold |
| `off-by-one [aarch64 max-delta:3]` | Base tolerance with arch override |

The `(dissim 0.05)` annotation is informational — `dissimilarity = (100 - score) / 100`, so `zensim:95` and `dissim 0.05` say the same thing. The score form is canonical.

## .checksums file format

Each module gets a `.checksums` file. It's a line-oriented append log, human-readable and merge-friendly:

```text
# zensim-regress checksums v1

## resize_bicubic 200x200
tolerance off-by-one
= sunny-crab-a4839401fa:sea  x86_64-avx2  @773c807  human-verified
~ tidy-frog-b2c3d40e1a:sea  aarch64  @773c807  auto-accepted vs sunny-crab-a4839401fa:sea (zensim:99.87 (dissim 0.0013), 2.1% pixels ±1, max-delta:[1,1,0], category:rounding, balanced)
```

**Entry prefixes:**
- `=` — human-verified baseline (trust anchor, never auto-pruned)
- `~` — auto-accepted within tolerance, with forensic diff evidence and a `vs` link to the baseline it was compared against (chain of trust)
- `x` — retired entry (superseded or known-wrong, kept as history)

**Section headers** (`## test_name detail_name`) group entries. The `tolerance` line beneath sets the default tolerance for that section.

**Memorable names** like `sunny-crab-a4839401fa:sea` are deterministic — derived from the hash bytes. Easier to reference in conversation than raw hex. The `:sea` suffix indicates the hash algorithm (SeaHash, 64-bit non-cryptographic).

## Diff images and montages

Enable automatic diff image generation on mismatch:

```rust
let mgr = ChecksumManager::new("tests/checksums".as_ref())
    .with_diff_output("test-artifacts/diffs");
```

On `Failed` results, the manager saves a side-by-side montage to `test-artifacts/diffs/{module}/{test}_{detail}.png`: expected | amplified diff | actual, with a 2px border. Amplification is 10x by default — `min(10, 255 / max_delta)` — so off-by-one errors become visible.

`CheckResult::Failed` includes a `montage_path` field pointing to the saved image.

You can also generate diff images directly:

```rust
use zensim_regress::diff_image::*;

let diff = generate_diff_image(&expected, &actual, 10);
let montage = create_comparison_montage(&expected, &actual, 10, 2);

// Raw RGBA byte variants
let diff = generate_diff_image_raw(&exp_bytes, &act_bytes, w, h, 10);
let montage = create_comparison_montage_raw(&exp_bytes, &act_bytes, w, h, 10, 2);
```

For sixel-capable terminals (foot, WezTerm, mintty), the `display` module renders images inline.

## CI integration

### Multi-platform setup

The main value of `zensim-regress` shows up in CI across multiple platforms. A typical GitHub Actions workflow:

```yaml
test:
  strategy:
    matrix:
      os: [ubuntu-latest, windows-latest, macos-latest, ubuntu-24.04-arm, windows-11-arm]
  runs-on: ${{ matrix.os }}
  env:
    REGRESS_MANIFEST_PATH: test-manifest.tsv
  steps:
    - uses: actions/checkout@v4
    - run: cargo test --workspace
    - uses: actions/upload-artifact@v4
      with:
        name: test-manifest-${{ matrix.os }}
        path: test-manifest.tsv
        if-no-files-found: ignore
      if: always()
```

### Manifest files

Set `REGRESS_MANIFEST_PATH` to log every check result to a TSV file:

```rust
let mgr = ChecksumManager::new("tests/checksums".as_ref())
    .with_manifest_from_env();
```

The manifest records test name, status (match/novel/accepted/failed), actual and baseline hashes, zensim dissimilarity, tolerance, and diff summary. One row per check.

For parallel test runners like `cargo-nextest` (which run tests in separate processes), use `REGRESS_MANIFEST_DIR` instead — each process writes its own file, and `combine_manifest_dir()` merges them afterwards.

### HTML reports

After collecting manifests from all platforms, generate a cross-platform HTML report:

```rust
use zensim_regress::report::*;

let entries = parse_manifest("test-manifest.tsv".as_ref()).unwrap();
let platforms = vec![("ubuntu-latest", entries.as_slice())];
let html = generate_html_report(&platforms, &Default::default());
std::fs::write("report.html", html).unwrap();
```

The report shows pass/fail status per test per platform, zensim scores, recommended tolerance lines, and embedded diff images (if you pass a `diffs_dirs` map pointing to uploaded artifacts).

## When a test fails

A `Failed` result means the output changed beyond tolerance. The `Display` output gives you everything you need:

```text
FAIL: zensim 87.23 (dissim 0.1277), max-delta:[12,8,3], 34.2% pixels differ
  category: perceptual, high confidence
  Montage: test-artifacts/diffs/resize/bicubic_200x200.png
  Suggested line: ~ tidy-frog-b2c3d40e1a:sea  x86_64-avx2  @abc1234  ...
```

**What to check:**

1. **Look at the montage.** The amplified diff shows exactly where and how much the output changed. Off-by-one rounding is invisible in the diff; real regressions are obvious.

2. **Check the category.** `rounding` and `balanced` are usually benign platform differences. `perceptual` or `color_shift` with high confidence means something visually changed.

3. **Check the score.** Scores above 95 are typically acceptable platform variance. Below 90, something probably broke. Below 80, something definitely broke.

4. **Decide:**
   - **Accept it** — run with `UPDATE_CHECKSUMS=1`, or paste the suggested line into the `.checksums` file.
   - **Investigate** — the delta, category, and montage tell you where to look in your code.
   - **Tighten tolerance** — use `shrink_tolerance()` to ratchet down after observing actual values across platforms.

### Ratcheting tolerances

Start with generous tolerances, then tighten based on what you observe:

```rust
use zensim_regress::{RegressionTolerance, RegressionReport, shrink_tolerance};

let floor = RegressionTolerance::exact();
let tightened = shrink_tolerance(&current_tolerance, &report, &floor);
```

`shrink_tolerance` takes the observed max deltas and score from a passing report and produces a tolerance that's tight enough to catch new regressions but loose enough to pass the current output. Ratchet down across CI runs until your tolerances reflect real platform variance, not guesswork.

## Standalone comparison (no checksums)

If you don't need persistent checksum files — you have two images and want to compare them directly:

```rust
use zensim::{Zensim, ZensimProfile};
use zensim_regress::{RegressionTolerance, check_regression};

let z = Zensim::new(ZensimProfile::latest());
let tolerance = RegressionTolerance::off_by_one();

let report = check_regression(&z, &expected_img, &actual_img, &tolerance).unwrap();
assert!(report.passed(), "regression: score {:.1}, category {:?}",
    report.score(), report.category());
```

`RegressionReport` gives you: `score()`, `category()`, `confidence()`, `max_channel_delta()`, `pixels_differing()`, `pixels_failing()`, `rounding_bias()`, histograms, and more.

## Input formats

`check_pixels` takes flat `&[u8]` RGBA (width × height × 4 bytes). `check_image` takes anything implementing zensim's `ImageSource` trait — `RgbSlice`, `RgbaSlice`, `imgref::ImgRef`, `StridedBytes` (BGRA, 16-bit, linear float, etc.). `check_file` loads from disk and hashes the file bytes.

For hash-only checks (no pixel comparison, no tolerance), `check_hash` takes a pre-computed hash string like `"sea:a4839401fabae99c"`.

## Remote reference storage

For large reference images that shouldn't live in git, configure S3/R2 storage:

```rust
let mgr = ChecksumManager::new("tests/checksums".as_ref())
    .with_remote_storage_from_env();
```

| Variable | Purpose |
|----------|---------|
| `REGRESS_REFERENCE_URL` | Base URL for downloading references |
| `REGRESS_UPLOAD_PREFIX` | Upload destination (e.g., `s3://bucket/refs` or `r2:bucket/refs`) |
| `UPLOAD_REFERENCES` | Set to `1` to enable uploads |

Downloads are cached locally in `{checksums_dir}/.remote-cache/`. The manager fetches reference images on demand for pixel comparison when a hash mismatch occurs.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `UPDATE_CHECKSUMS` | unset | `1` or `true` to auto-accept results and create baselines |
| `REGRESS_MANIFEST_PATH` | unset | TSV file path for logging check results |
| `REGRESS_MANIFEST_DIR` | unset | Directory for per-process manifest files (nextest) |
| `REGRESS_REFERENCE_URL` | unset | Base URL for remote reference downloads |
| `REGRESS_UPLOAD_PREFIX` | unset | Remote upload destination prefix |
| `UPLOAD_REFERENCES` | unset | `1` or `true` to enable reference uploads |

## Test image generators

Deterministic generators for synthetic test inputs:

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
| `oracle` | Pixel oracle testing: scalar reference vs whole-image comparison |
| `simd` | SIMD consistency testing via archmage token permutations (feature: `archmage`) |
| `checksums` | `ChecksumManager`, `ChecksumsFile`, `ChecksumEntry`, `CheckResult` |
| `testing` | `RegressionTolerance`, `RegressionReport`, `check_regression` |
| `tolerance` | `ToleranceSpec`, `ToleranceOverride` for config-driven tolerances |
| `diff_summary` | Tolerance shorthand formatting and parsing |
| `diff_image` | Amplified diff images and comparison montages |
| `display` | Sixel terminal rendering |
| `generators` | Synthetic test image generators |
| `distortions` | Deterministic pixel distortions |
| `hasher` | `ChecksumHasher` trait, `SeaHasher` (64-bit non-crypto) |
| `arch` | Architecture detection and tag matching |
| `petname` | Memorable names from hashes (e.g., `sea:a1b2...` → `sunny-crab`) |
| `manifest` | TSV manifest writer for CI result aggregation |
| `report` | HTML report generation from manifest data |
| `remote` | S3/R2 reference image storage config |
| `fetch` | HTTP fetcher for remote reference downloads |
| `upload` | Shell-based file uploader |
| `lock` | Advisory file locking for parallel test safety |
| `error` | `RegressError` error type |
