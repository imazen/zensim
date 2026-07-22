# v2 reference-pyramid reuse + kernel perf campaign — 2026-07-21

Goal (user directive): add a prepared-reference API to v2 and double
extraction performance, dropping to safe intrinsics where warranted and
experimenting with orderings + register assignment.

**Result: v2-348 extraction 2.33× end-to-end on real pairs; ext-720 1.55×
(bounded by the byte-frozen v1 block). Feature output byte-identical
except pool features on the v4x tier at max 1.1e-7 relative drift
(policy 5e-4; shipped cross-arch envelope 9.1e-6).**

Commits: `99cc8fb3` (prepared-ref API + pad-skip + scratch), `fdd6514a`
(cached ref moments + ref-grouped driver), `1e48e7c8` (SIMD pools on v4x +
inline(always) cliff), `1640f424` (3-output fused H). Host: 7950X (Zen4,
AVX-512), WSL2, single-thread measurements serialized (no concurrent heavy
jobs; earlier contaminated readings discarded and rerun).

## Scoreboard — 100 aic3 pairs (~1–5 MP), v2-348 mode, RAYON_NUM_THREADS=1

| step | wall s | ms/pair | cumulative |
|---|--:|--:|--:|
| old binary (318502e0), per-pair flow | 11.87 | 118.7 | 1.00× |
| + ref-grouped + `V2PreparedReference` + pad-skip + scratch reuse | 8.08 | 80.8 | 1.47× |
| + cached ref moments (mu1+activity, bit-exact strip replay) | 7.18 | 71.8 | 1.65× |
| + SIMD weighted pools on v4x (16-accumulator layout) | 5.26 | 52.6 | 2.26× |
| + 3-output fused H (mu1 chain compiled out) | **5.09** | **50.9** | **2.33×** |

ext-720 (v1-372 ++ v2-348, the fleet backfill workload): old 17.90 s →
new 11.52 s = **1.55×** single-thread. v1c (~56 ms/pair) is byte-frozen and
untouched — it now dominates ext. 600-pair 8-thread driver wall (ext):
54.2 → 47.7 s (intra-group parallelism + LPT; CPU-seconds under
multi-thread contention are NOT a per-pair metric — rayon spin +
bandwidth contention inflate them ~3.7× vs single-thread truth).

## Numeric safety

- Every step verified against the old binary's CSVs on the same pairs.
  Steps 1–2: **byte-identical** in all modes (ext/v1/v2 × grouped/ungrouped
  × moments on/off). Bit-exactness of the moments cache is BY CONSTRUCTION
  (strip-walk replay: same halo gather, same sliding chains, same fns) and
  test-gated (`prepared_ref_with_moments_bit_identical_to_pair_path`).
- Step 3 (pool SIMD, v4x only): 37.9 % of feature cells change, max rel
  **9.6e-8** (v2-only CSV) / 1.1e-7 (ext-720); non-pool sums bit-equal
  (`pool_simd_drift_within_policy`). 16-register tiers keep the scalar
  path → their output is unchanged, so cross-arch merge behavior is the
  same class as the already-shipped 9.1e-6 envelope (318502e0).
- Step 4: mu2/ssq/s12 chains independent of mu1 → bit-identical
  (`ssim3_matches_ssim4_bitwise`).
- v1 golden bytes: green throughout (exact `to_bits` gate, 2 fixtures).

## Where the time went (perf, moments-cached v2 path, before→after pools)

| symbol | before | after pools+ssim3 |
|---|--:|--:|
| dense_block_kernel (v4x) | 41.4 % | ~14 % |
| fused_blur_h_ssim (v4x) | 22.7 % | ~31 % (now #1) |
| box_blur_v / memmove / gathers | ~17 % | ~21 % |
| zenpng decode (dist) | ~9 % | ~13 % |
| gradient kernel | 3.9 % | ~6 % |
| XYB convert + downscale | ~9 % | ~9 % |

## The inline(always) cliff (root-cause find, load-bearing)

Adding the POOL_SIMD variant pushed `dense_block_kernel_generic` past
LLVM's inline threshold; `#[inline]` stopped fusing it into the
`#[arcane]` target_feature region and **every V8 operator compiled to a
call** into non-inlined `core::arch` shims (`__mm256_add_ps` alone = 26 %
of runtime; 5.3× whole-run regression, 38.2 s vs 7.2 s). `#[inline(always)]`
is the entire fix and is now documented in-source at both bodies. Any
future `*_generic` kernel called from an arcane entry MUST be
`#[inline(always)]`, and a standalone `dense_block_kernel_generic`-style
symbol in a profile is the smoking gun. Plausibly the same cliff
contaminated the original §A.14 "22 accumulators = register pressure"
measurement — worth re-testing the single-pass 16-acc layout on AVX2
with inline(always) before trusting the old conclusion.

## Register-assignment result (the §A.14 revision)

The 11 weighted pools need only **16** lane accumulators, not 22: the
mask family shares one Σw, the IW family shares one Σw, and both weights
derive from ONE `saturate(act)` division (4 vector divisions per 8 px
replace 40 scalar f64 divisions). 13 core + 16 pool accumulators fit
AVX-512VL's 32 registers → single-pass fully-SIMD dense kernel on v4x
(disjoint-tier magetypes blocks; AVX2/NEON keep the scalar-pool body).

## API landed (all `feature-regime-v2`-gated)

- `Zensim::prepare_v2_reference` / `prepare_v2_reference_with_moments`
  → `feature_v2::V2PreparedReference` (per-scale XYB planes; optionally
  per-scale/channel mu1+activity planes, ~2× prepare cost/memory).
- `Zensim::compute_v2_features_with_ref` /
  `compute_v2_features_with_ref_and_scratch` (+ `feature_v2::V2Scratch`,
  one per worker thread → zero steady-state allocation).
- Pair path = prepare + with_ref composition (single scale-walk owner).
- `v2_ab_extract`: ref-grouped by default (`ZENSIM_AB_GROUPED=0` legacy
  flow), LPT group ordering, intra-group parallel variants,
  `ZENSIM_AB_MOMENTS=0` opt-out, nested-rayon fix for the v1 block.

## zenmetrics handoff (fleet wiring — sibling repo, not touched here)

`zenmetrics/crates/zenmetrics-cli/src/metrics/zensim.rs:267-270` documents
that the ctx path rebuilds the v2 ref pyramid per variant because "no
precomputed-ref API" existed. It exists now: hold a
`V2PreparedReference` (prefer `_with_moments`; watch worker RSS — ~5.3×
w·h f32 without moments, ~2× that with) in `ZensimRefCtx` next to the v1
`PrecomputedReference`, call `compute_v2_features_with_ref_and_scratch`
per variant with a per-worker `V2Scratch`. Also: the per-cell metric job
(`jobexec.rs:1128-1214`) re-decodes the reference per cell and uses the
no-ctx path — worth a ref-ctx LRU keyed on ref sha.

## Next levers, ranked (measured shares, not implemented)

1. **fused_blur_h_ssim transpose restructure** (~31 % of v2c): the
   row-transposed sliding window does ~128 scalar memory ops per column
   step (64 gather loads + 64 stores across 4 planes × 16 rows).
   16×16 in-register transpose tiles (AVX-512 shuffles via archmage)
   convert both sides to contiguous vector ops — bit-exact by
   construction (same per-lane chains, different data movement).
   Est. −10–15 % total.
2. **Rolling halo gather**: successive strips re-copy 2×HALO_P=40 of 148
   rows (~27 % of gather traffic; memmove ≈ 7 %).
3. **Gradient src-mag cache** in `V2PreparedReference` (gradient ≈ 6 %,
   ~45 % of it is ref-side stencil+sqrt; bit-exact fill-by-replay).
4. **AVX2 pool revisit**: row-fission two-pass (16-acc layout, L1-resident
   d/art/det/mse row buffers) — and FIRST re-test the plain single-pass
   with inline(always) given the cliff suspicion.
5. Dist-side PNG decode ≈ 13 % (zenpng inflate+paeth) — out of scope here.

## Repro

```sh
cargo build --release -p zensim --example v2_ab_extract \
  --features feature-regime-v2,threads,training
ZENSIM_AB_MODE=v2 ZENSIM_AB_GROUPED=1 ZENSIM_AB_MOMENTS=1 \
  RAYON_NUM_THREADS=1 target/release/examples/v2_ab_extract \
  <pairs.tsv> out.csv
cargo bench -p zensim --features training,feature-regime-v2 \
  --bench v2_speed_baseline   # v2_with_ref[_moments]_1thread groups
```
Pairs: first 100 of `/mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv`.
NOTE: the bench's city.png fixture shows v2c ≈ 150 ms @1024² vs ≈ 84 ms on
aic3 photo content at the same pixel count — fixture-content effect
(verified not a regression: old and new binaries both ~11.8 s on the same
100 pairs pre-optimization). Use the bench for RELATIVE comparisons.
