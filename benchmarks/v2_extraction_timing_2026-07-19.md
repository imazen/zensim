# v2 / v1 / extended extraction timing — direct measurement (2026-07-19)

> **PARTIALLY SUPERSEDED (2026-07-21):** the per-pair costs below predate
> the ref-reuse + kernel campaign. Current numbers (v2c ≈ 30 ms/pair
> amortized at ~1 MP, 2.33× v2 / 1.55× ext-720):
> `benchmarks/v2_ref_reuse_perf_2026-07-21.md`. The decomposition METHOD
> and the v1c figure here remain valid.

Resolves the doc conflict (spec §A.12 "AFTER" table claimed v2/v1 ≈ 4.35× at 1MP;
the min-max/blend session memory said ~2.3×). Measured directly on the CURRENT
post-phase-6 binary, real images, single-thread — supersedes both.

## Method

`v2_ab_extract` gained `ZENSIM_AB_MODE={none|v1|v2|ext}` (decode-only / v1-372 /
v2-348 / extended-720) — one binary, one decode path, only the compute set varies.
`RAYON_NUM_THREADS=1`, load-gated (`run-heavy --jobs 1`), 100 aic3 pairs (~1 MP:
1192×832 etc.). Internally consistent: decode+v1c+v2c = 16.77s ≈ ext 16.90s (0.8%).

## Result (s / 100 pairs, single-thread, ~1 MP)

| mode | wall s/100 | ms/pair | isolated component |
|---|--:|--:|---|
| none (decode) | 3.36 | 33.6 | decode = 33.6 ms |
| v1 (372) | 8.94 | 89.4 | v1 compute = **55.8 ms** |
| v2 (348) | 11.19 | 111.9 | v2 compute = **78.3 ms** |
| ext (720) | 16.90 | 169.0 | v1c + v2c = 134 ms |

## Ratios (the answer)

- **v2 compute / v1 compute = 1.40×** at ~1 MP (78.3 / 55.8). NOT 4.4×.
  On the full aic3 set (mixed, up to 5 MP) it rises to ~1.7× (v1c 108.5s / v2c
  186s over 600 pairs) — v2's per-pixel cost grows faster with size, consistent
  with the spec's size-dependence, but the current binary is far faster than the
  spec's stale AFTER numbers at every size measured.
- **extended-720 / v2-348 = 1.51× wall** (169 / 112). Extended adds v1's whole
  pipeline, but v1 is the cheap half (56 ms) and decode (34 ms) is shared, so the
  double-pass costs only ~50% more than v2 alone.
- **extended-720 / v1-372 = 1.89× wall.**

## Production implication

The 720 double-pass is a RESEARCH artifact (compute both feature sets on identical
pixels for a fair append-only comparison). Production computes ONE set. Once the
surviving feature set is settled (mask the deprecated ones), the production cost is
~v2-class: **~1.4–1.7× v1 compute**, and less if the masked-out features are
skipped in the kernel. Decode dominates at small sizes either way — a
content-addressed sidecar that computes the final set once amortizes it.

## The gradient sqrt is ALREADY SIMD (2026-07-19, asm-verified)

The +50% `gradient_features` cost is NOT a scalar-sqrt bottleneck. `objdump -d` of
the built binary shows **16 `vsqrtps`** (AVX 8-wide hardware SIMD sqrt) in the
gradient kernel, in `ymm5`/`ymm6` pairs (grad_src_mag + grad_dst_mag), vs only 5
scalar `sqrtss` (the per-channel-scale `dev2 = (m2/n).sqrt()`, negligible). The
phase-4 magetypes work already vectorized it: `gradient_block_kernel` dispatches
via `incant![v4x,v4,v3,neon,wasm128,scalar]` → `gradient_block_kernel_generic<T:
F32x8Backend>` → `V8::sqrt` → `_mm256_sqrt_ps` (x86_v3 backend) / `_mm512_sqrt_ps`
(v4x). The +50% is inherent: two high-latency HW sqrts per pixel + strided
neighbor-load setup for the gradient stencil. The only remaining micro-opt is
`x·rsqrt_approx(x)` + a Newton step (magetypes exposes `rsqrt_approx`), which is
marginal on Zen4 (vsqrtps ~3-6cyc there) and risks the ≤5e-4 v2 numeric policy —
not taken. **"SIMD the sqrt" needs no work; it's done.**

## Doc correction

`docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12 "AFTER" table (4.35× @1MP) is stale vs
the current binary. Flagged there; this file is the current measurement of record.
