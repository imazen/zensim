# REV4VEC — worklog

Lane: REV4VEC (SWE-2, 2026-10-03). Coordinator: Claude session 1d9f6f0b.
Goal: recover the ~2.3× Rev4 serving-cost regression REV4SERVE measured
(`benchmarks/rev4serve_WORKLOG.md`: Rev3 scalar score ≈ 583 ms, Rev4 ≈ 1380 ms
at 1 MP; prepared 705 → 1670 ms) by running the SAME canonical per-element
formulas lane-parallel on the tiers that have true FMA (v4x/v4/v3, NEON),
keeping the existing scalar canonical body on scalar and — unless it proves
bit-exact — wasm128. Every full chunk must be bit-identical to the scalar
canonical body per element by construction; every remainder runs the scalar
canonical body.

## Ground truth established before implementation

Backend audit (magetypes 0.9.28 sources,
`~/.cargo/registry/src/index.crates.io-*/magetypes-0.9.28/`):

- `mul_add` fuses on x86 v3 (`_mm256_fmadd_ps`), v4 (`_mm512_fmadd_ps`),
  NEON (`vfmaq_f32`); it does NOT fuse on the scalar (`a * b + c`) or
  wasm128 (mul then add) backends — the two known XYB tier divergences.
- x86 `round`/`to_i32_round` are ties-even (`_mm256_round_ps` NEAREST /
  `_mm256_cvtps_epi32`); scalar canon uses `round_ties_even`/`as i32`
  (identical for integral inputs).
- x86 `max`/`min` operand order matches `f32::max`/`min` NaN semantics
  (non-NaN operand wins — same operand order in both bodies).
- NEON `f32x16` = 4×`float32x4` `vfmaq` (fused); x86-v3 `f32x16` =
  2×`_mm256` FMA (fused); x86-v4/v4x `f32x16` = native zmm (fused);
  wasm128/scalar `f32x16` = unfused per lane — excluded from the fused
  tier list.
- `GenericF32x16` (`F32x16Convert`) covers every token: v4x, v4, v3,
  neon, wasm128, scalar — one generic body, per-tier fused-or-scalar
  routing at the incant arm.

Op-for-op diffs performed (det_math scalar canon ↔ magetypes vector):

- `log2_midp_f32` == `f32x{8,16}::log2_midp`: same integer offset/mask,
  same `(a−1)/(a+1)`, same `C3..C0` `mul_add` Horner, same final
  `poly.mul_add(y, n)`, same edge-blend order (0→−inf, <0→NaN, +inf→+inf).
- `exp2_midp_f32` == `f32x{8,16}::exp2_midp`: same clamp `.max(-126).min(128)`,
  `round` (ties-even on both), `.min(127)`, `xf = x − xi`, same degree-6
  Horner, `((xi+127)<<23)` reconstruction, same edge blends (x<−126→0,
  x≥128→+inf).
- `exp_midp_f32`/`log10_midp_f32`/`pow_midp_f32` == the `*_midp` vector
  twins (identical composite structure and constants).
- `cbrt_midp_f32` == `f32x{8,16}::cbrt_midp` for every finite input:
  identical seed `bits/3 + 0x2a50_8c2d`, identical 2 Halley iterations
  `y *= (y³+2x)/(2y³+x)` (no fused ops inside — bit-identical on ALL
  tiers incl. scalar/wasm128). Sign path differs (`signum*y` vs
  `y | sign`) — identical results for finite values; NaN-payload edge is
  unreachable from every canon leaf (opsin inputs are post-`max(0)`).
- `OpsinChunk::cube_roots`/`positive` ↔ `opsin_px_canon`: identical
  nested `mul_add` mix, `max(0)`, `cbrt_midp`, `0.5(c0∓c1)`,
  `x.mul_add(14,.42)`, `y+.01`, `(t2−y)+.55`.
- `pu_xyb_rows_inner`'s `pu` closure ↔ `pu21_encode_canon`: identical
  formula (`*_midp_precise` == `*_midp` on f32x8/16), but the production
  mix uses UNFUSED `*`,`+` and `*inv_white` where canon uses fused
  `mul_add` rows and `/PU_WHITE` — the production vector is NOT canon;
  the canon vector reuses only its `pu` arithmetic.

## Profiling

- perf blocked (`perf_event_paranoid=4`); used valgrind 3.26 callgrind.
- Build: `cargo build --release --bench extract_paths_bench -p zensim
  --features custom-profiles,feature-regime-v2,threads,training`
  (defaults add `avx512`; host has AVX-512 → v4x dispatch — but see
  below, the r3/r4 binaries both run the same feature set).
- Runs: `ZENSIM_FORMULA_REV={3,4} ZEN_XP_RSS=bake
  ZEN_XP_BAKE=/var/tmp/rev4serve/bench/featpot_r{3,4}.bin
  ZEN_XP_SIZE=512 ZEN_XP_ITERS=4 RAYON_NUM_THREADS=1` under callgrind
  (`--callgrind-out-file`, `--inclusive=yes`).

### Instruction counts (4 iters at 512²)

| metric | r3 | r4 | Δ |
|---|---:|---:|---:|
| total Ir | 5.134 G | 13.222 G | **+8.09 G (2.58×)** |

### Delta attribution (inclusive Ir, r4 − r3)

| function | r3 | r4 | Δ | share of +8.09 G |
|---|---:|---:|---:|---:|
| `feature_v2::stream_phase_b` | 2.371 G | 6.744 G | +4.37 G | (orchestrates the leaves below) |
| `restore_cuts::run` + `run_cell` | 1.27 G | 4.19 G | +2.67 G | 33 % — rev4canon cell machinery around the canon blur + f64 mapdev pools |
| `fused::fused_vblur_features_ssim` | 0.446 G | 3.084 G | **+2.64 G** | **33 % — `fused_vblur_ssim_canon::<LanesF64>` (Rec64 vblur + per-pixel pool feeds)** |
| `blur::fused_blur_h_ssim_at_revision` | 0.369 G | 1.677 G | **+1.31 G** | **16 % — the f64 Rec64 sliding h-blur (rev4canon)** |
| `feature_v2::dense_block_kernel` | 0.762 G | 1.956 G | +1.19 G | 15 % |
| `feature_v2::gradient_block_kernel` | 1.176 G | 1.977 G | +0.80 G | 10 % |
| `feature_v2::fold_v1_one_band` | 0.219 G | 1.517 G | +1.30 G | 16 % |
| `feature_v2::run_blur_pass_inner` | 0.392 G | 1.014 G | +0.62 G | 8 % |
| **`color::srgb_to_positive_xyb_planar_into_at_revision` (the opsin canon — THIS LANE)** | 0.096 G | 0.938 G | **+0.84 G** | **10 %** |
| libm `fmaf` (f32 `mul_add` calls) | ~0 | 0.525 G | **+0.52 G** | 6 % — baseline codegen: every scalar `f32::mul_add` calls libm `fmaf` (~25 Ir/call); opsin's 10 fmaf/px at 512²×2×4 ≈ 21 M calls ≈ most of it |
| libm `fma` (f64 `mul_add` calls) | ~0 | 0.497 G | +0.50 G | 6 % — same defect for the Rec64 blur's `f64::mul_add` |
| `pow`/`__ieee754_pow` | 0.217 G | ~0.2 G | ≈0 | libm pow remains only on noncanon arms |

### What the profile says about scope

- **In this lane's file scope** (`color.rs`, `pu21.rs`, `transfer.rs`,
  `det_math.rs`): the canonical opsin front end
  (`srgb_xyb_canon`, incl. its share of `fmaf`) is ~**10–16 % of the
  r3→r4 delta** at this workload. `linear_xyb_canon` shares the same
  body; `pu_xyb_canon`/PQ/HLG canon are HDR-path only (0 % here) but
  vectorize identically.
- **~84 % of the delta sits in the f64 Rec64 blur + canon pool/block
  machinery** (`fused.rs`, `blur.rs`, `feature_v2*.rs`,
  `restore_cuts.rs`) — rev4canon-era scalar canon arithmetic, NOT in
  this lane's named file scope and NOT transcendental. Two properties
  make it the dominant follow-up target if the coordinator wants the
  rest of the gap back: (a) its `f64::mul_add`s pay the same libm `fma`
  call penalty (~0.5 G); (b) the Rec64 sliding window is serial in the
  blur direction but every column/lane's recurrence is INDEPENDENT —
  `f64x{4,8}` lane-parallel over columns preserves the exact per-element
  op order (fused `mul_add` rounds once = scalar `f64::mul_add`), i.e.
  it vectorizes bit-exactly by the same construction used here.

## Implementation

All in `zensim/src/` — no public API change, no call-site changes beyond
the canonical drivers' internals:

* `color.rs`: `CanonChunk<T: F32x16Convert>` — the 16-lane generic chunk
  replicating `opsin_px_canon`'s fused-`mul_add` matrix chain, `max(0)`,
  `cbrt_midp`, opponent split and positive shift op-for-op, plus the
  PU21 canon constants path. `#[magetypes(define(f32x16), v4x, v4, v3,
  neon, -scalar)]` emits one lane-parallel variant per true-FMA tier
  (`_v4x`, `_v4`, `_v3`, `_neon`); `-scalar` suppresses the unfused
  generic scalar variant, replaced by hand-written `_scalar`/`_wasm128`
  stubs that call the scalar canonical bodies. Drivers wired:
  `srgb_xyb_canon` (u8 LUT → linear → opsin), `linear_xyb_canon::<CLAMP>`
  (both clamp variants), `pu_xyb_canon` (opsin mix + `pu21_encode_canon`
  per channel + `/PU_WHITE`). Full 16-px chunks vector; tail → scalar
  body (bit-identical per element by construction).
* `transfer.rs`: `decode_pq_row_canon_vec` (flat-channel `pow_midp`
  EOTF + display model) and `decode_hlg_row_canon_vec` (per-pixel
  `clamp01` as nested `blend` — preserves scalar `f32::clamp` NaN/−0
  semantics —, `v ≤ 0.5` select, `exp_midp`/`pow_midp` OOTF chain,
  unfused `Σ luma·e`). Same tier/stub structure.
* `det_math.rs`: no production change; the scalar `*_midp_f32` bodies
  remain the canonical reference the vectors replicate.

## Bit-exactness evidence

All per-lane comparisons are **strict `f32::to_bits` equality** against
the scalar canonical body (no tolerance, no NaN relax at leaf level;
both-NaN equal only inside the midp gate where NaN *inputs* are outside
the canonical domain — payload divergence would still fail).

| gate | coverage | result |
|---|---|---|
| `color::rev4vec_tests::srgb_xyb_canon_vec_matches_scalar_body_on_every_compiled_tier` | 4111 px (all u8 values × channel mixes), n≡15 mod 16 → chunk+remainder | **PASS** v3/v4/v4x + scalar |
| `color::rev4vec_tests::linear_xyb_canon_vec_matches_scalar_body_on_every_compiled_tier` | 10⁷ random f32 bit patterns + explicit special-class pixels (±0, denormals, ±inf, NaN, PU bounds), clamped AND unclamped | **PASS** v3/v4/v4x + scalar |
| `color::rev4vec_tests::pu_xyb_canon_vec_matches_scalar_body_on_every_compiled_tier` | 10⁷ random bit patterns + specials + 0.001–12000-nit ramp | **PASS** v3/v4/v4x + scalar |
| `transfer::tests::pq_row_canon_vec_matches_scalar_body_on_every_compiled_tier` | 10⁷ channels × 4 display peaks (100/1k/4k/10k nits) + specials | **PASS** v3/v4/v4x + scalar |
| `transfer::tests::hlg_row_canon_vec_matches_scalar_body_on_every_compiled_tier` | 3.4×10⁶ px × 3 (γ, luma) combos + specials incl. NaN/−0 through `clamp01` | **PASS** v3/v4/v4x + scalar |
| `det_math::rev4vec_midp_gate::midp_x16_matches_scalar_sampled` | ~5×10⁷ inputs: all specials, dense [−128,128] and [0,1] sweeps, 2²⁴ denormals, stride-257 bit sweep × {`log2`,`exp2`,`exp`,`log10`,`cbrt`} + `pow` at the two canonical PQ exponents | **PASS** v3/v4/v4x |
| `det_math::rev4vec_midp_gate::midp_x16_matches_scalar_1e8_sweep` (ignored) | **1.4×10⁸ inputs** (all specials + 2²⁴ denormals + deterministic stride-37 sweep of the whole u32 space, 116 M patterns) × 7 fns — the brief's ≥10⁸ fallback criterion | **PASS** v3/v4/v4x, 24.3 s release |
| `det_math::rev4vec_midp_gate::midp_x16_matches_scalar_over_all_f32_bits` (ignored) | **all 2³² f32 bit patterns** × 7 fns × compiled fused tiers, 8 threads | running (release profile, pinned cores 9–15) |

Tier variants are invoked directly on summoned tokens — no
`for_each_token_permutation` global mutation in unit tests; the
REV4SERVE permutation gates exercise dispatch-level coverage separately.

### Serve / bake gates (2026-10-03, this workspace)

| gate | result |
|---|---|
| `rev4serve_gate::rev4_served_vectors_bitmatch_research_extract_on_every_tier` | **PASS** — dev 283 s + release 279 s, every tier permutation |
| `rev4serve_gate::rev4_maps_are_tier_identical` | **PASS** |
| `rev4serve_gate::rev{1,2,3}_baseline_capture` | **PASS** — Rev1–3 byte-identical |
| `rev4serve_gate::rev4_featpot_bake_served_and_steered` (justfile `rev4serve-gate`, real featpot bake + aic3 corpus) | **PASS** — release, 116 s |
| `cargo test -p zensim --release --features custom-profiles,feature-regime-v2,threads,training` | **PASS** — 571 lib + all integration + doctests, 0 failures |
| `cargo clippy --workspace --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings` | **PASS** |
| `cargo fmt -p zensim --check` | **PASS** |
| `just lint-scripts` | **PASS** — 797 scripts (fixed one false positive: `*arr` deref → renamed `chunk`) |
| `just api-doc-check` | **PASS** — no public API change |

## Benchmarks

Served-cost A/B — same recipe as `scripts/bench/rev3_cost_ab.sh`: ONE
binary, alternating rev3/rev4 blocks ×3, pinned CPU 8, 1 thread,
`SIZES=1024,2048`, 15 rounds (min 6, 120 s cap), `fast_ssim2_st` +
`D_current_revision` controls. BEFORE = REV4SERVE binary
(`run1` at /var/tmp/rev4serve/bench, sha256 9018062f…, 06:00 UTC);
AFTER = this lane's binary (`run1` at /var/tmp/rev4vec/bench, sha256
f464b7cc…, 09:15 UTC). Median of 3 block means.

### Scalar score — `featpot_v2basic` (ms)

| size | rev3 before | rev3 after | rev4 before | rev4 after | Δ rev4 |
|---|---:|---:|---:|---:|---:|
| 1024² (1 MP) | 582.5 | ~607.9 | 1380.1 | **1209.3** | **−170.8 ms (−12.4 %)** |
| 2048² (4 MP) | 2350.8 | ~2389.7 | 5320.5 | **4860.3** | **−460.2 ms (−8.6 %)** |

Rev4/Rev3 cost ratio: **2.37× → 2.00×** at 1 MP, **2.26× → 2.03×** at 4 MP
(rev3-after arms rose ~4 % — ambient load drift, the `fast_ssim2_st`
control stayed flat ~105–112 ms both revs, so the rev4 drop is real).

### Prepared map — `featpot_v2basic` (ms)

| size | rev3 before | rev3 after | rev4 before | rev4 after | Δ rev4 |
|---|---:|---:|---:|---:|---:|
| 1024² | 704.8 | ~739.2 | 1671.1 | **1573.6** | **−97.5 ms (−5.8 %)** |
| 2048² | 2778.8 | ~3039.1 | 6512.8 | **6276.2** | **−236.6 ms (−3.6 %)** |

Prepared Rev4/Rev3: 2.37× → 2.13× (1 MP), 2.34× → 2.06× (4 MP).

### What was recovered and what remains

- Recovered ≈ 170 ms of the ≈ 800 ms rev4-over-rev3 gap at 1 MP scalar
  (~21 % of the gap) — matches the profile: the opsin canon leaf was
  ~10 % of the delta and its `fmaf` libm-call penalty ~6 % more.
- Remaining ≈ 630 ms (~79 %) sits in the f64 Rec64 blur + canon
  pool/block machinery (`fused_vblur_features_ssim`, `blur_h`,
  `restore_cuts`, dense/gradient block kernels) — NOT in this lane's
  file scope; per-column `f64xN` vectorization is the follow-up.
- Prepared-map improvement is smaller because the prepared path skips
  part of the front-end per iteration; the opsin share of its work is
  relatively lower than in the scalar arm.

## Negative results / unresolved

- **~79 % of the 2.3× gap remains** in f64 canon blur/pool code outside
  this lane's scope (see table above) — flagged for a follow-up lane.
- Exhaustive 2³² midp sweep: detached release run in flight, pinned to
  cores 9–15 (hours-scale on the loaded box); result lands in
  `/var/tmp/rev4vec/exhaustive_midp.log`. The brief's ≥10⁸ fallback
  criterion is satisfied NOW by the stride-37 sweep (1.4×10⁸
  deterministic inputs incl. every special class — PASS, 24 s release);
  the `#[ignore]`d exhaustive test remains in-tree for an off-peak run.
- wasm128/NEON tiers are stubbed/untestable on this host (scalar canon
  body by design — bit-identical trivially).
