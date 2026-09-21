# LANE_SIMD — DONE (rev final, 2026-09-21)

Scope executed: audit every hot generic-over-token kernel in zensim for the
call-per-op codegen defect found during the geometry lane; fix what is broken;
measure under the speed-matrix protocol with bit-exactness proven first.

## Audit verdict

The defect is confined to the **DVIFM serving pumps** —
`zensim/src/dvifm.rs` (f64) and `zensim/src/dvifm_int.rs` (i16). Featured-tier
entries were thin shells delegating to featureless generic monomorphs
(`level_push_row::<V4xToken>` et al.): 0 zmm/ymm, every vector op a call into
a shared `__tf` stub. Helper-monomorph census: **56 before → 4 after** (the 4
survivors are 3 correct scalar-tier `gradient_block_kernel` closures + 1
external `magetypes` `convert_f16` v3 helper — not zensim defects).

Already fine (featured entry materializes real vector code; calls are cold
`slice_index_fail`/`OnceLock`/alloc stubs): `blur::downscale_2x_into_inner`,
all `blur::*` arcane kernels, `color::{srgb,linear}_to_positive_xyb_planar`,
`color::pu_xyb_rows` (v3 is its top tier — no v4 body exists), all
`fused::*` kernels (generic accumulates DO inline), all `feature_v2` block
kernels (dense, gradient, append, csfw, era2), all `simd_ops::*` inline
families, `streaming::diffmap_accum_*`, `attribution::combine_basic_and_l8`,
`ssim_form` helpers (inlined into fused callers). `metric.rs` has no
generic-over-token fns.

## Root cause + fix (commit porqknnm)

`#[inline(always)]` alone could not fold `level_push_row`/`level_flush` —
rustc never inlines recursive fns, and the `l→l+1` cascade is recursive.
Fix: converted the bounded cascade (`DVIFM_LEVELS = 5`) to an explicit LIFO
worklist `PumpWork::{Push,Flush}` driven by `pump_drive`, children pushed in
reverse so pop order = the exact recursive DFS order (bit-identical), plus
`#[inline(always)]` on the whole helper chain, both f64 and i16. No public
API change, no op reorder.

After-fix disasm (v4x): `dvifm_push_rows_entry` 207→9834 ins, 0→597 zmm;
`dvifm_finish_entry` 128→9905, 10→606 zmm; i16 mirrors 232→8714 (529 zmm),
131→8738 (538 zmm). v3 entries materialize ~724–848 ymm. Scalar entries
remain scalar islands (correct).

## Correctness gates (all green, proven before the speed claim)

- `cargo test -p zensim --release dvifm`: 46/46 (incl.
  `int16_strip_size_bit_identical`, `simd_tier_parity`)
- `cargo test -p zensim --release`: 430/430 lib tests, 0 failures
- `v1_golden_bytes` (training): 5/5 (synthetic, real, nontight, fold-backed,
  same-class bit-exact determinism)
- Score bits `to_bits()` before vs after: profiles B/D/C × sizes
  256/1024/2048 identical at `ZENSIM_FORMULA_REV=1` **and** `=3` (Rev3
  ensemble path byte-identical)
- DVIFM serving feature vectors (f64 + i16), FNV-1a-64 of `to_bits()`:
  identical both revs, all sizes

## Measured speedup (30 paired interleaved invocations per thread-count;
run-heavy quiet box; t1 pinned CPU 0, t8 CPUs 0–7; no target-cpu=native)

1-thread: `dvifm_f64stream` 0.556/0.543/0.556 and `dvifm_i16stream`
0.604/0.531/0.524 (256²/1024²/2048²) — **44–48% faster**, tight dispersion
(pair-MAD ≤ 0.005 at ≥1024²).
8-thread batch-of-8: `dvifm_i16stream_batch8` 0.610/0.539/0.526 —
**~44–47% faster** (256² shows wider CI [0.40,1.20] — small-image
dispersion, median still 0.61).
Parity arms: `profile_B` 0.992–1.002, `profile_D` 0.995–1.000 t1;
batch8 0.986–1.037 — all pair-CIs straddle 1.0. Kernel arms `k_*`
0.948–1.039, CIs straddle 1.0, and those kernels' code is byte-identical
before/after (dispersion/layout luck).

## Deliverables

- Report: `benchmarks/simd_inlining_audit_2026-09-21.md`
- Machine audit + results: `benchmarks/simd_inlining_audit_2026-09-21.json`
- Bench harness: `zensim-bench/examples/simd_audit.rs`
- Before binary + raw paired outputs: `/home/lilith/tmp/devin/simd-audit/`
- Commit: `porqknnm` (jj, on top of geometry commit `lkzlwlmx`; no push)
- Log: `/home/lilith/tmp/devin/lane_simd.log`

## Caveats

- `just clippy` fails on 8 pre-existing lints in the geometry lane's
  `dvifm/geom.rs`/`research.rs` — zero in this commit's files (documented,
  not silently rewritten).
- Speed claim is at `dvifm_serving_run` granularity — the production-
  reachable surface; the row helpers have no separate public arms.
- `pu_xyb_rows` tops out at v3 (no v4/v4x arcane body) — noted, not a defect.
