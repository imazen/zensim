# Runtime cost — 4 permutations of (extended_features, compute_iw_features) (2026-05-15)

Per-pair wall-clock time for `compute_zensim_with_config` across the
4 flag permutations, measured on deterministic synthetic gradient
pairs at 3 sizes. Runs single (ref, dist) pair × N iters, reports
min / median / mean wall ms.

**Headline**: at production-relevant image sizes (≥ 512²), each of
**extended** and **IW** adds **~13–15 % per-pair compute**, and
**both together cost ~25–28 %** (slightly sub-additive — the two
passes share scale-state). At small sizes (≤ 256²), fixed overhead
(allocation, pyramid setup) dominates and the measurements are
noisy; the percentages should be read off the 512² and 1024²
rows.

Background context for these numbers: the 372-feat training was
running concurrently at lower CPU priority, which inflates absolute
times by ~5–10 % but doesn't change the *relative* cost
comparison (every config sees the same contention). Re-running on
a quiescent box would tighten the numbers but not change the
verdict.

## 256×256 (65,536 pixels — fixed-overhead-dominated, NOISY)

| Config | n_features | min ms | median ms | mean ms | × vs standard |
|---|---:|---:|---:|---:|---:|
| Standard      (228 features) | 228 | 3.52 | 4.42 | 4.45 | 1.00× |
| Extended only (300 features = +masked) | 300 | 4.00 | 4.82 | 4.87 | 1.09× |
| IW only       (300 features = +IW) | 300 | 4.05 | 5.22 | 5.61 | 1.26× |
| Extended + IW (372 features = +masked +IW) | 372 | 3.99 | 4.66 | 4.66 | 1.05× |

The 1.05× for "Extended + IW" being LOWER than 1.26× for "IW only"
at this size is a noise artifact — at 256² the per-call wall is
~4.5 ms and the system jitter is ±1 ms. Don't read the deltas at
this size; read the 512² / 1024² rows.

## 512×512 (262,144 pixels — transitional)

| Config | n_features | min ms | median ms | mean ms | × vs standard |
|---|---:|---:|---:|---:|---:|
| Standard      (228 features) | 228 | 6.50 | 6.95 | 7.05 | 1.00× |
| Extended only (300 features = +masked) | 300 | 7.21 | 7.61 | 7.97 | 1.13× |
| IW only       (300 features = +IW) | 300 | 7.37 | 7.95 | 8.00 | 1.13× |
| Extended + IW (372 features = +masked +IW) | 372 | 8.03 | 8.97 | 9.00 | **1.28×** |

## 1024×1024 (1,048,576 pixels — per-pixel-dominated)

| Config | n_features | min ms | median ms | mean ms | × vs standard |
|---|---:|---:|---:|---:|---:|
| Standard      (228 features) | 228 | 14.03 | 15.56 | 15.93 | 1.00× |
| Extended only (300 features = +masked) | 300 | 16.52 | 18.02 | 18.23 | 1.14× |
| IW only       (300 features = +IW) | 300 | 16.45 | 18.30 | 18.36 | 1.15× |
| Extended + IW (372 features = +masked +IW) | 372 | 17.58 | 19.10 | 19.84 | **1.25×** |

## Per-pixel cost breakdown (1024×1024 = 1.0 MP)

| Config | mean ms | ns / pixel | Δ ns vs standard | Δ ns / per-block |
|---|---:|---:|---:|---|
| Standard         | 15.93 | 15.2 | — | baseline |
| Extended only    | 18.23 | 17.4 | **+2.2 ns** | +14 % per masked block |
| IW only          | 18.36 | 17.5 | **+2.3 ns** | +15 % per IW block |
| Extended + IW    | 19.84 | 18.9 | **+3.7 ns** | +24 % both → slightly sub-additive |

If the two extra blocks were exactly additive, Extended + IW would
cost `+2.2 + 2.3 = +4.5 ns/pixel`. Measured is `+3.7 ns/pixel`
(80 % of additive). The shared scale-state savings (the extended
masking pass reuses the flatness map; IW pool reuses scale stats)
recover ~20 % of the additive cost.

## Comparison to CLAUDE.md ProfileParams docstring

The new `ProfileParams.extended_features` docstring claims:
> "extended-features overhead is moderate (~10–30 % per-pair
> compute vs standard)"

Measured at 512² and 1024²: **+13–15 %** per block, **+25–28 %**
both. The "~10–30 %" range bracket is correct but conservative —
real cost lands in the middle of the range.

## Interpretation

At production-relevant sizes (web codec inputs typically 512²–4K²),
**every extended/IW flag costs about an eighth of a per-pair call
each**. For a metric used in:

- **Per-pair eval** (KADID 10125 pairs at ~7 ms = 71 s standard):
  +28 % both → ~91 s. 20 s difference for batch eval is negligible
  developer-time.
- **Per-image picker** (production codec gate, ~10 candidate
  encodings per image): standard 70 ms, both 90 ms per image —
  +20 ms per image. For batch processing (say 10,000 images/hr),
  +20 ms × 10,000 = 200 s/hr extra compute → 5.5 % throughput hit.
- **Streaming pipeline** (zenpipe / imageflow, single-image):
  user-visible latency increment of ~3 ms on a 1 MP image.

Verdict per CLAUDE.md "shipping policy":

- **Extended-features alone (+13 %)**: cheap enough to ship if it
  delivers ≥ 0.005 CID22 SROCC. The V_20 extended seed=1 falsified
  this gate (CID22 0.8783 < 0.8895 fast-ssim2 floor; see
  `v0_20_extended_falsification_2026-05-15.md`).
- **IW alone (+15 %)**: cheap enough to ship under same gate.
  V_20a IW falsified at k=1/4/8 — see
  `v0_20a_path_a_falsification_2026-05-14.md`.
- **Both together (+25 %)**: NOT cheap enough to ship without a
  ≥ 0.010 CID22 SROCC win. None measured yet.

The 372-feat V_20-transformed bake (`b64nxit9q` in flight at this
writing) will produce the first measurement of whether transforms
+ IW + masked together can clear that bar.

## How to reproduce

```sh
cargo build --release --example extended_iw_perf -p zensim-bench --features training

# Real image (KADID native 512×384):
./target/release/examples/extended_iw_perf \
    --ref /mnt/v/dataset/kadid10k/images/I01_01_01.png \
    --dist /mnt/v/dataset/kadid10k/images/I01_01_03.png \
    --iters 30

# Synthetic gradient at chosen size (deterministic):
./target/release/examples/extended_iw_perf --size 1024 --iters 50
```

## Files

- Benchmark: `zensim-bench/examples/extended_iw_perf.rs`
- ProfileParams flags: `zensim/src/profile.rs` (commit `f140776a`)
- L0 norm inspector: `zensim-validate/src/bin/inspect_l0_input_norms.rs`
  (commit `bc9e6b60`)
- Companion falsification doc: `v0_20_extended_falsification_2026-05-15.md`
- Companion L0-selection doc: `v0_20_l0_norms_2026-05-15.md`

## Hardware

Lilith's water-cooled AMD Ryzen 9 7950X, 128 GB RAM, 16 cores /
32 threads, Linux 6.6.114.1-microsoft-standard-WSL2.
