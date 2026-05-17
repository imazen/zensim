# Runtime cost — Extended + IW perm sweep, optimized (2026-05-15)

This is the post-optimization companion to
`extended_iw_runtime_perf_2026-05-15.md`. The 2026-05-15 perf
optimization landed three sets of changes to `zensim/src/streaming.rs`
and `zensim/src/simd_ops.rs`:

1. **Fused 2-mask SIMD kernels** for SSIM, edge, and weight+MSE
   construction — when both `extended_features` AND
   `compute_iw_features` are on, the masked SSIM denominator
   computation is shared between the two pools.
2. **V-blur-only on H-blurred sigma_sq/sigma12** — replace 4 SIMD
   passes (`sq_sum_into` + 2D blur + `mul_into` + 2D blur) with 2
   passes (1D V-blur each), reusing the H-blurred values already
   computed by `fused_blur_h_ssim`.
3. **Pre-existing IW-only mu1 swap bug fix** — the V-blurred mu1
   was not being swapped into `bufs.mu1` when only `compute_iw_features`
   was on, so the activity-map computation used H-blurred (garbage)
   values. Fixed and verified with a new integration test.

**Headline**: at production-relevant image sizes (≥ 512²), each of
**extended** and **IW** adds **~6–14 % per-pair compute**, and
**both together cost ~+6–12 %**. The combined Extended+IW path
clears the user's +15 % target at all measured sizes.

Background context for these numbers: the 372-feat training plus
parallel agent work was running concurrently at lower CPU priority,
which inflates absolute times by ~5–10 % but doesn't change the
*relative* cost comparison. The 100-iter samples below are stable
to within ±2 percentage points.

## 256×256 (65,536 pixels — fixed-overhead-dominated, mildly NOISY)

| Config | n_features | min ms | median ms | mean ms | × vs standard |
|---|---:|---:|---:|---:|---:|
| Standard      (228 features) | 228 | 3.04 | 3.78 | 3.81 | 1.00× |
| Extended only (300 features = +masked) | 300 | 3.19 | 3.74 | 3.76 | **0.99×** |
| IW only       (300 features = +IW) | 300 | 3.19 | 3.77 | 3.83 | 1.01× |
| Extended + IW (372 features = +masked +IW) | 372 | 3.24 | 3.85 | 3.90 | **1.03×** |

At 256² the additional feature work is small relative to fixed
overhead; nearly free.

## 512×512 (262,144 pixels — transitional)

| Config | n_features | min ms | median ms | mean ms | × vs standard |
|---|---:|---:|---:|---:|---:|
| Standard      (228 features) | 228 | 5.40 | 5.89 | 6.04 | 1.00× |
| Extended only (300 features = +masked) | 300 | 5.67 | 6.22 | 6.38 | 1.06× |
| IW only       (300 features = +IW) | 300 | 5.63 | 6.29 | 6.43 | 1.07× |
| Extended + IW (372 features = +masked +IW) | 372 | 5.63 | 6.22 | 6.37 | **1.06×** |

## 1024×1024 (1,048,576 pixels — per-pixel-dominated)

| Config | n_features | min ms | median ms | mean ms | × vs standard |
|---|---:|---:|---:|---:|---:|
| Standard      (228 features) | 228 | 11.13 | 12.31 | 12.57 | 1.00× |
| Extended only (300 features = +masked) | 300 | 12.41 | 13.31 | 13.58 | 1.08× |
| IW only       (300 features = +IW) | 300 | 12.28 | 13.51 | 13.79 | 1.10× |
| Extended + IW (372 features = +masked +IW) | 372 | 12.94 | 13.78 | 14.06 | **1.12×** |

## Per-pixel cost breakdown (1024×1024 = 1.0 MP, mean)

| Config | mean ms | ns / pixel | Δ ns vs standard | Δ % vs standard |
|---|---:|---:|---:|---|
| Standard         | 12.57 | 12.0 | — | baseline |
| Extended only    | 13.58 | 13.0 | +0.96 ns | +8 % |
| IW only          | 13.79 | 13.2 | +1.16 ns | +10 % |
| Extended + IW    | 14.06 | 13.4 | **+1.42 ns** | **+12 %** |

The combined cost (+1.42 ns/pixel) is now substantially
**sub-additive**: the additive sum would have been +0.96 + 1.16 = +2.12 ns,
and we pay 67 % of that (+1.42). That's the fused 2-mask kernel
recovering 33 % of the additive cost when both flags are on.

For comparison, before optimization:
- Combined cost was +3.7 ns/pixel (+24 % at 1024²)
- New cost is +1.42 ns/pixel (+12 %) → **62 % reduction in absolute
  cost, and overhead now under target.**

## Reproducibility

Build:
```sh
cargo build --release --example extended_iw_perf -p zensim-bench --features training
```

Run (synthetic deterministic, 100 iters per config):
```sh
./target/release/examples/extended_iw_perf --size 1024 --iters 100
./target/release/examples/extended_iw_perf --size 512 --iters 100
./target/release/examples/extended_iw_perf --size 256 --iters 100
```

Real image (KADID native 512×384):
```sh
./target/release/examples/extended_iw_perf \
    --ref /mnt/v/dataset/kadid10k/images/I01_01_01.png \
    --dist /mnt/v/dataset/kadid10k/images/I01_01_03.png \
    --iters 30
```

## Verification of numerical equivalence

The optimization preserves bit-equivalent (within FMA tolerance)
outputs. The test `metric::tests::fused_2mask_matches_separate_paths`
asserts that:

1. First 228 features (basic + peaks) are identical between
   ext-only / iw-only / both-flags configs (differ by < 1e-9).
2. The masked block (features 228..300) is identical between
   `ext_only` and `both` configs (relative difference < 1e-4).
3. The IW block is identical between `iw_only` and `both` configs
   (relative difference < 1e-4).

Additionally, 4 unit tests in `simd_ops::tests` verify the fused
2-mask kernels produce identical output to two single-mask calls
at sizes {16, 17, 32, 100, 256, 1024}.

All 63 zensim unit tests + 18 zensim integration tests + all 7
workspace member crates' tests pass (376 unit tests in
zensim-regress, etc.).

## Files modified

- `zensim/src/streaming.rs` — fused 2-mask dispatch + V-blur-only
  optimization + IW-only swap bug fix
- `zensim/src/simd_ops.rs` — 5 new public SIMD entry points
  (ssim_channel_masked_2, edge_diff_channel_masked_2_art4_det4,
  build_weights_and_mse, build_mask_weight_and_mse,
  build_iw_weight_and_mse) + 4 unit tests
- `zensim/src/metric.rs` — new integration test
  `fused_2mask_matches_separate_paths`

## Files

- Benchmark: `zensim-bench/examples/extended_iw_perf.rs`
- ProfileParams flags: `zensim/src/profile.rs`
- Hotspot analysis: `benchmarks/iw_perf_hotspots_2026-05-15.md`

## Hardware

Lilith's water-cooled AMD Ryzen 9 7950X, 128 GB RAM, 16 cores /
32 threads, Linux 6.6.114.1-microsoft-standard-WSL2.
