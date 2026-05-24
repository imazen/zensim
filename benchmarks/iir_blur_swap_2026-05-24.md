# IIR Gaussian blur swap — hardest-test experiment (2026-05-24)

## Question

Would replacing zensim's box-blur 1-pass path with a Charalampidis 2016 IIR
Gaussian (the trick that won ~16% fewer instructions in butteraugli) speed up
zensim if we're willing to break score compatibility?

## Answer: no — IIR is *slower per pixel* than box blur

Wall-clock at 1024² extended+iw (the case the experiment was sized for):

| variant   | min ms | mean ms | vs box |
|---        |---:    |---:     |---:    |
| box (baseline) | 13.56 | 15.27 | 1.00× |
| IIR            | 13.94 | 14.98 | ≈ noise floor |

Callgrind at 512² (3 iters, extended+iw):

| variant | total instr | vs box |
|---      |---:         |---:    |
| box     | 4.585 B     | 1.000× |
| IIR     | 4.795 B     | **1.046×** (worse) |

Function-level delta (IIR build vs box build):
- `box_blur_h_inner_v3`: 769M → 384M  (−385M, swap removed half)
- `box_blur_v_copy_inner_v3`: 286M → 189M  (−97M)
- `blur_iir::horizontal_pass_v3`: **+558M**
- `blur_iir::vertical_pass_inner_v3`: **+148M**
- Net swap cost: **+224M instructions**

## Why the result inverts butteraugli's

Butteraugli replaced FIR Gaussian (O(σ) per pixel — kernel-width-dependent,
~15-tap kernel at σ=7). Charalampidis IIR's flat 6-FMA per-pixel cost easily
beats FIR there.

zensim was already running **box blur with running sums** — O(1) per pixel,
just 2 adds + 1 mul. Charalampidis IIR has 12 FMAs per pixel (3 parallel
2-pole sections, 4 FMAs each). It's ~4× heavier per pixel than what it
replaced. Both algorithms are O(1); IIR has a higher constant.

The win IIR offers ("escape FIR's σ-scaling") doesn't exist here because
zensim already escaped it via box blur.

## Feature-level score divergence (real images, KADID Q01/Q02 + Q01/Q04)

20 pairs across 10 reference images. Per-feature deltas:

|        | RMS Δ% | max Δ% |
|---     |---:    |---:    |
| feat 250 (masked block) | 0.47% | 0.97% |
| feat 300 (IW block)     | 0.04% | 0.07% |
| feat 350 (IW tail)      | 0.18% | 0.41% |

Sub-1% per-feature divergence. The IIR port is numerically correct (DC
preserved within 5e-3, impulse sum within 0.02 — see `blur_iir::tests`).
The MLP score wasn't compared (would require V0_4 weights and a heavier
harness), but feature-level divergence at this magnitude is in retrain
tolerance, not numerical-corruption tolerance.

## What this rules out

- IIR Gaussian as a wall-clock win on top of zensim's existing box-blur pipeline.
- Any future "swap box blur for X faster O(1) primitive" — box blur **is** the
  floor for separable O(1) smoothing on f32 planes.

## What the data still leaves open (NOT tested here)

The swap targets only `box_blur_1pass_into` — used for the activity-map blur
(line 1299) and the separate-blur fallback path (line 1459+). The dominant
hot spots in the production pipeline are the *fused* H+V blur kernels:

- `fused_blur_h_ssim_inner`: 23.2% of instructions
- `fused_vblur_ssim_inner`:  15.7% of instructions

These fuse box blur with sq_sum, mul, and feature-accumulation in a single
SIMD pass. Replacing the inner blur with IIR would force de-fusion, which the
2026-05-15 hotspot work explicitly showed costs 4-13 pp at 1024². So even if
IIR's per-pixel cost were lower than box, the de-fusion penalty would erase it.

## Recommendation

**Drop the IIR-blur direction.** The compat-breaking blur swap is the wrong
intervention for zensim because the algorithm zensim already uses is the
floor. Productive compat-breaking directions remain:

1. Smaller blur radius / fewer pyramid scales — algorithmic redesign,
   requires retraining V0_4 (or successor). True 25-40% blur-work reduction.
2. f16/bf16 storage on plane buffers — bandwidth savings on H-pass loads;
   rounding-noise compat break. Worth trying if Zen 4's f16 ALU is good
   enough; the V-pass already has state in registers so it won't help there.
3. Inline the masked-block accumulation INTO `fused_vblur_features_ssim`
   (already-scoped TODO from `iw_perf_hotspots_2026-05-15.md`, est. 4-6 pp
   at 1024², **no** compat break).

## Reproduction

```
# Build both
cargo build --release --example score_divergence -p zensim-bench --features training
cp target/release/examples/score_divergence /tmp/score_divergence_box
cargo build --release --example score_divergence -p zensim-bench --features training,zensim/iir-blur
cp target/release/examples/score_divergence /tmp/score_divergence_iir

# Build perf example
cargo build --release --example extended_iw_perf -p zensim-bench --features training
cp target/release/examples/extended_iw_perf /tmp/extended_iw_perf_box
cargo build --release --example extended_iw_perf -p zensim-bench --features training,zensim/iir-blur
cp target/release/examples/extended_iw_perf /tmp/extended_iw_perf_iir

# Run wall-clock
for sz in 256 512 1024; do
  /tmp/extended_iw_perf_box --size $sz --iters 60
  /tmp/extended_iw_perf_iir --size $sz --iters 60
done

# Run callgrind (3 iters is enough at 512²)
valgrind --tool=callgrind --cache-sim=no --branch-sim=no \
  --callgrind-out-file=/tmp/box.out  /tmp/extended_iw_perf_box --size 512 --iters 3
valgrind --tool=callgrind --cache-sim=no --branch-sim=no \
  --callgrind-out-file=/tmp/iir.out  /tmp/extended_iw_perf_iir --size 512 --iters 3
callgrind_annotate --threshold=85 --auto=no /tmp/box.out
callgrind_annotate --threshold=85 --auto=no /tmp/iir.out
```

## Files touched

- `zensim/src/blur_iir.rs` — new (port of butteraugli's `src/blur_iir.rs`
  adapted to flat-slice convention; box-radius → σ via σ = √(r(r+1)/3))
- `zensim/src/blur.rs` — `box_blur_1pass_into` dispatches to IIR under
  `iir-blur` feature
- `zensim/src/lib.rs` — `mod blur_iir` gated on feature
- `zensim/Cargo.toml` — new `iir-blur` feature flag
- `zensim-bench/examples/score_divergence.rs` — small divergence harness

The branch and worktree are kept until the user decides to drop them.
