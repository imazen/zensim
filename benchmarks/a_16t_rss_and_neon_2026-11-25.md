# `ZensimProfile::A` 16-thread memory/L3 optimization and NEON/WASM128 kernel parity (2026-11-25)

## Summary

Four structural optimizations in `zensim/src/{blur.rs,feature_v2_kernels.rs,feature_v2.rs}` cut 16-thread 12 MP (`4000×3000`) total-process peak RSS by **42–45%** (`584 MiB → 329–341 MiB`), accelerate 16-thread `1024²` scoring by **1.89×** (`22.27 ms → 11.81 ms` on Intel Core Ultra 7 265K AVX2) by keeping horizontal scratch inside L2/L3 cache, and enable missing ARM64 NEON and WebAssembly SIMD128 tiers on the revision-7 `local11` and `l64` kernels — while preserving **100% bit-exact scores** across x86-64 tiers and thread counts.

## Changes

1. **ARM64 NEON & Wasm SIMD128 dispatch (`blur.rs`, `feature_v2_kernels.rs`):**
   - Enabled `neon` on `fused_blur_h_local11` and `fused_blur_hv_local11`.
   - Enabled `neon` and `wasm128` on `activity_local11`, `box_blur_h_local11`, `box_blur_v_local11`, and `blockiness_row_l64`, eliminating scalar fallbacks on Apple Silicon / ARMv8 and browser Wasm targets (`3.06×–3.44×` faster than `scalar` on Apple M6 at `1T`).
2. **256-column L1/L2 horizontal-blur scratch (`feature_v2.rs`):**
   - Shrunk `Rev5StripJob`'s five horizontal moment scratch buffers (`mu1_h..s12_h`) from full image width to `fused_panel_hbuf_len(width)` (`256 + 2 * radius` `f32`s), saving `~104 MiB` across queued strip jobs at `w = 4000` and eliminating cross-core L3 thrashing at `1024²`.
3. **Eliminated dead streaming buffers under `block_walk` (`feature_v2.rs`):**
   - Skipped allocating `StripPlaneProducer` (`~41 MiB`), `scratch.strips` (`~14 MiB`), and per-scale `FoldPoolScratch`es when `plan.can_block_walk()` (`ZensimProfile::A`) is active.
4. **Bounded wide-frame strip queue (`feature_v2.rs`):**
   - Capped `ahead_chunks = num_threads.min(4)` and `scratch_cap = (128 MiB / per_job_bytes).clamp(6, 12)` when `w0 > 1600`, bounding peak concurrent strip-job allocations without starving worker threads.

## Bit-exact parity verification

- **Split-parity gate (`report.split_parity`):** `135,625 / 135,625` normalized contract rows matched the frozen research golden (`0` diffs; output SHA-256 `3ab036b8559e819a54feb5795ea6d498a3b2a09aaa34c41312bebc0365a98eb1`).
- **Thread & tier determinism:** `0` score-bit mismatches across `1T`, `8T`, `12T`, and `16T` on all 5 benchmark geometries (`64²`, `256²`, `840×944`, `1024²`, `4000×3000`) on AMD Ryzen 9 9950X3D (`v4x` and forced `v3`), Intel Core Ultra 7 265K (`v3`), and Apple M6 (`neon`).
- **Hardware NEON test suite:** `cargo test --release -p zensim --lib` (`98 / 98`) and `--test profiles` (`8 / 8`) passed natively on Apple M6 (`aarch64-apple-darwin`).

## Multi-platform resident RGB8 latency and peak RSS

Protocol: resident RGB8 sRGB frames in memory (decode/I/O excluded), `lto = "thin"`, `codegen-units = 1`, no `-C target-cpu=native` (binary SHA-256 `dfbb17fce6ffcc7ce0427e9040d79905e6fa20cf779a8e404c61017c2333c60b` on x86-64 Linux), 20 deterministic pairs × 32 interleaved `zenbench` rounds (`640` calls/cell; `96` calls/cell at `4000×3000` on Ryzen 9 9950X3D). Peak RSS is measured in a fresh OS process per `(tier, threads, size)` scoring 20 pairs sequentially via GNU `time -v` (`Maximum resident set size`) on Linux and `/usr/bin/time -l` on macOS (BSD peak memory footprint noted alongside RSS).

### 1. AMD Ryzen 9 9950X3D (CCD0 8C/16T pinned, 96 MiB L3, Linux x86-64)

| Tier | Threads | `64²` p50 / p95 (ms) | `256²` p50 / p95 (ms) | `840×944` p50 / p95 (ms) | `1024²` p50 / p95 (ms) | `4000×3000` (12 MP) p50 / p95 (ms) | OLS fit (`α + β·MP`) | `1024²` RSS | `12 MP` RSS |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `v4x` (AVX-512) | 1 | 0.39 / 0.41 | 1.78 / 1.87 | 16.68 / 18.54 | 19.97 / 22.46 | 190.32 / 194.07 | `2.14 + 15.70 ms/MP` | 32,544 KiB (31.8 MiB) | 178,192 KiB (174.0 MiB) |
| `v4x` (AVX-512) | 16 | 0.39 / 0.41 | 0.75 / 0.93 | 3.46 / 4.03 | 4.03 / 4.87 | 75.67 / 77.42 | `−1.28 + 6.40 ms/MP` | 115,988 KiB (113.3 MiB) | 337,180 KiB (329.3 MiB) |
| `v3` (AVX2 forced) | 1 | 0.40 / 0.41 | 2.28 / 2.38 | 22.70 / 24.73 | 28.85 / 30.09 | 291.27 / 293.45 | `1.97 + 24.13 ms/MP` | 32,636 KiB (31.9 MiB) | 178,036 KiB (173.9 MiB) |
| `v3` (AVX2 forced) | 16 | 0.40 / 0.41 | 0.84 / 1.03 | 4.04 / 4.76 | 4.87 / 6.01 | 81.57 / 85.18 | `−1.08 + 6.88 ms/MP` | 120,068 KiB (117.3 MiB) | 349,104 KiB (340.9 MiB) |

### 2. Intel Core Ultra 7 265K (8P + 12E Arrow Lake, Linux x86-64 `v3` AVX2)

| Threads / topology | `64²` p50 / p95 (ms) | `256²` p50 / p95 (ms) | `840×944` p50 / p95 (ms) | `1024²` p50 / p95 (ms) | `4000×3000` (12 MP) p50 / p95 (ms) | OLS fit (`α + β·MP`) | `1024²` RSS | `12 MP` RSS |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 (P-core pinned) | 0.49 / 0.55 | 2.81 / 3.08 | 28.19 / 31.42 | 35.84 / 36.46 | 382.08 / 434.89 | `1.63 + 31.72 ms/MP` | 32,844 KiB (32.1 MiB) | 178,300 KiB (174.1 MiB) |
| 8 (8 P-cores pinned) | 0.48 / 0.54 | 1.42 / 1.77 | 7.74 / 8.51 | 10.18 / 11.01 | 183.67 / 188.86 | `−3.34 + 15.56 ms/MP` | 93,520 KiB (91.3 MiB) | 344,092 KiB (336.0 MiB) |
| 16 (8P + 8E) | 0.48 / 0.69 | 1.84 / 4.09 | 8.98 / 11.92 | 11.81 / 13.82 | 167.44 / 172.44 | `−1.33 + 14.05 ms/MP` | 115,836 KiB (113.1 MiB) | 338,808 KiB (330.9 MiB) |
| *Pre-optimization 16T baseline* | *0.49* | *2.65* | *—* | *22.27* | *188.80* | *—* | *159,920 KiB (156.2 MiB)* | *586,620 KiB (572.9 MiB)* |

### 3. Apple M6 (2 Super + 4 Performance + 6 Efficiency = 12 cores, macOS `aarch64` `neon`)

| Tier | Threads | `64²` p50 / p95 (ms) | `256²` p50 / p95 (ms) | `840×944` p50 / p95 (ms) | `1024²` p50 / p95 (ms) | `4000×3000` (12 MP) p50 / p95 (ms) | OLS fit (`α + β·MP`) | `1024²` RSS (footprint) | `12 MP` RSS (footprint) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `neon` | 1 | 1.06 / 1.10 | 7.15 / 8.08 | 30.17 / 31.17 | 38.48 / 39.45 | 467.41 / 472.19 | `0.73 + 38.86 ms/MP` | 46,304 KiB (40.6 MiB) | 215,760 KiB (206.2 MiB) |
| `neon` | 12 | 1.07 / 1.12 | 2.58 / 2.86 | 6.71 / 10.23 | 8.52 / 11.90 | 122.23 / 125.20 | `−0.49 + 10.21 ms/MP` | 147,856 KiB (139.8 MiB) | 336,864 KiB (324.5 MiB) |
| `scalar` (forced) | 1 | — | — | — | 132.33 (desc) | 1,430.98 (desc) | — | — | — |

## `run-heavy` cgroup resource receipts

- AMD Ryzen 9 9950X3D suite (`mem-cap=24G`, `jobs=28/32`): `rc=0 4014s | peak-RSS 0.33GiB | min-avail 35159MiB | peak-load 8.40`
- Intel Core Ultra 7 265K suite (`mem-cap=16G`, `jobs=16/20`): `rc=0 5203s | peak-RSS 1.71GiB | min-avail 12391MiB | peak-load 13.49`
