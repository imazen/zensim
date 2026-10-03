# KERNOPT2 worklog — make the Rev4 horizontal f64 window fast on AVX-512, bit-exact

Lane: SWE-2 (devin-swe2-kernopt2), 2026-10-03. Workspace:
`zensim--rev4vec` jj workspace on `main@origin` 63838a2b (anon `@`).
Local bookmark `quarantine/swe2/kernopt2` (never pushed). Scratch
`/var/tmp/kernopt2`, `CARGO_TARGET_DIR=/var/tmp/kernopt/target`
(shares the predecessor lane's warm target dir). Bakes:
`/var/tmp/kernopt/bakes/{v2basic,by_v2fy}.r{3,4}.bin` (dense bakes,
KERNOPT's, sha-verified there).

## Native-tier profile split (BEFORE any code change)

`perf record -F 2999 -g` on `extract_paths_bench` (`ZEN_XP_RSS=bake`,
`ZENSIM_FORMULA_REV`, `RAYON_NUM_THREADS=1`, `taskset -c 8`, **no tier cap —
native v4x/AVX-512 dispatch**, perf_event_paranoid=1 works on this box).
Data: `/var/tmp/kernopt2/perf/*.data`. Bench binary sha256:
`90165a3b261fb947835d5b52d8c724a3e6b6492da129f6218d58d20973123343`.

### v2basic r4 scalar 1 MP (12 iters, 3565 samples)

| symbol | self % |
|---|---:|
| `blur::fused_blur_h_ssim_rec64_rows_v4x` | **47.98** |
| `feature_v2::dense_block_kernel_canon64_vec_v3` (v4x hw → v3 body, 63838a2b fix) | 12.18 |
| `blur::box_blur_v_copy_inner_v4x` (f32, both revs) | 10.72 |
| `fused::fused_vblur_ssim_canon64_vec_v4x` | 10.28 |
| `feature_v2::gradient_block_kernel_canon64_vec_v4x<fff>` | 4.54 |
| `blur::box_blur_h_inner_v4x` (f32) | 3.13 |
| `color::srgb_xyb_canon_vec_v4x` | 2.66 |
| `MeanOffsetRows::add_strip` | 1.59 |
| `__memmove_avx512` | 1.01 |

### v2basic r4 prepared 1 MP (8 iters, 3974 samples)

| symbol | self % |
|---|---:|
| `blur::fused_blur_h_ssim_rec64_rows_v4x` | **40.43** |
| `dense_block_kernel_canon64_vec_v3` | 6.77 |
| `box_blur_v_copy_inner_v4x` | 6.72 |
| `fused_vblur_ssim_canon64_vec_v4x` | 6.13 |
| `attr_pass_b_main_entry_v4x` | 5.77 |
| `__memmove_avx512` | 5.47 |
| `gradient_block_kernel_canon64_vec_v4x<fff>` | 2.70 |
| `srgb_xyb_canon_vec_v4x` | 2.56 |

### v2basic r3 scalar 1 MP (anchor, 24 iters, 3389 samples)

box_blur_v 25.0, vblur_ssim_inner 15.3, fused_blur_h_ssim_inner 11.7,
dense era2 11.7, srgb_to_xyb 6.2, box_blur_h 6.0, gradient 5.0, memmove 4.8.

### perf annotate of `fused_blur_h_ssim_rec64_rows_v4x`

LLVM already auto-vectorised the four per-step column gathers into
`vgatherqps` (4/step ≈ 15.8 % of kernel samples: 3.86 + 4.02 + 3.21 + 4.72).
The rest is dominated by the scalar extract-and-store tail: per x step the
four `(sum*inv).to_array()` arrays become ~32 `vcvtsd2ss`/`vmovss` plus
`vshufpd`/`vextractf128`/`vextractf32x4` lane extraction and `leaq` address
arithmetic — a ~200-instruction scalar tail per 8 rows. The f64x8 FMA work
itself is <1 % of samples.

Plan (per coordinator task, all bit-exact by construction — same per-row op
sequence `+add −rem`, same `mul_add` chains, same `(sum*inv_v64) as f32`):

1. Interior column blocks (8 x-steps/block): load the add and remove
   columns as 8×8 f32 row loads + in-register 8×8 transpose, widen each
   column to f64 once (`vcvtps2pd`) — replaces both gathers and the
   scalar converts; rem side needs NO ring (cols are contiguous for x≥r).
2. Store side: stage the four narrowed `vcvtpd2ps` columns per step,
   8×8-transpose at block end → 8 aligned vector stores per plane.
3. Interleave two 8-row groups (16 rows/pass) so the 4-FMA `sum_sq`
   latency chains overlap.
4. Head (`x<r`, mirrored rem) / tail (mirrored add, partial) keep the
   current scalar-gather body verbatim — identical values.

Tier coverage: v4x and v4 share one `#[rite(v4)]` body (all intrinsics are
AVX-512F ⊂ v4); v3 gets the same algorithm on `__m256d` pairs;
scalar/wasm128/neon keep the scalar row body (unchanged).

## Implementation (2026-10-03/04)

### `rec64_rows8x8` — 8x8-transposed row-block kernel (blur.rs)

Replaced the per-x scalar gather/convert/store row loop with an 8-column
block body. Per block: 4 x 8x8 f32 loads (sadd/dadd/srem/drem) widened to
f64 in registers (`vcvtps2pd`), 8x8 transposes in registers, the per-row
`+add -rem`/`mul_add` recurrence on f64x8 registers interleaved across G=2
row groups, results narrowed (`vcvtpd2ps`), staged and transposed back to
8 contiguous stores per output plane. Head (`x < r` mirrored add) and tail
(mirrored rem, partial block) keep the scalar gather path verbatim.
`std::array::from_fn` builds the load arrays — no dead zero-init pass.

- `#[rite(v4, import_intrinsics)] rec64_rows8x8<G>` — one const-generic
  AVX-512F body serving v4x + v4 (all ops AVX-512F); non-avx512 builds get
  a cfg'd fallback; AVX2 (`_v3`) gets the same algorithm on `__m256d`
  pairs; neon/wasm128/scalar keep the scalar row body.
- Result (v2basic r4 scalar 1 MP): ~180 ms/iter -> ~68-70 ms/iter whole
  cell; Rev4/Rev3 from ~2.97x to ~1.35x on native AVX-512.
- perf split after rewrite (native v4x, r4 scalar 1 MP):
  rec64_rows8x8 20.58 % (was ~48 %), dense canon64 v3 18.45 %,
  vblur_ssim 15.52 %, gradient 6.41 %, box_blur_v_copy 4.42 %,
  srgb_xyb 4.37 %. No gathers in the rec64 hot loop; 4 gather insns
  remain in head/tail/init edge paths only.
- Note: `srgb_xyb_canon_vec_v4x` contains one zmm `vgatherdps` at ~1.6 %
  of the cell — below the 3 % keep bar; noted only.

### Add-column history ring — measured, rejected

Rolled the 3 previous blocks' add columns in a ring to feed the remove
side for `diam <= 16` (production r=5 -> diam=11). Fixed two indexing
bugs (history layout is `[k-3|k-2|k-1]` -> flat index `24+j-diam`;
`diam <= 8` needs a 4th `cur` slot since the rem source is inside the
current block). Bit-exact after fix; A/B on the Rev4 scalar 1 MP cell:
ring 68.9-69.2, no-ring 68.6-69.2 ms/iter — noise, <3 %. Removed; kept
the simpler always-load path.

### `std::array::from_fn` load arrays — kept, not a win

Replaced `[[0.;8];8]` + overwrite with `from_fn` (removes the dead
`vmovdqa64` zero-init pass that was ~10-15 % of rec64 kernel samples).
Whole-cell timing a wash (69.2-70.1 vs 69.4-70.6). Kept as strictly less
dead code; not counted as a measured win.

## Dense canon64 v4x — the gather/scatter is the accumulate loop itself

Dispatch previously capped Canon64 dense at v3 (63838a2b) because the
generic body gather-mangled (~4.3x). Two failed shapes, then the fix:

1. Flat `[LanesF64; 29]` pool + `iter_mut().zip()` lane loop: still
   ~108-112 ms/iter vs v3's ~73. Disasm: LLVM materialised per-lane
   pointer vectors (`vmovupd rsp->zmm`) for the pool RMW ->
   `vgatherqpd`+`vaddpd`+`vscatterqpd` per pool, ~76 % of the v4x fn's
   samples. Vectorising ACROSS lanes of a memory RMW is exactly what
   gather/scatter encodes.
2. `rev4_dense_chunk8` `#[inline(never)]` helper + `black_box(8)` trip
   count: r4 hook block fully scalar (0 gathers inside it) — but the 48
   gathers were in the hot accumulate, not the r4 path. Helper kept
   anyway (documents + pins the scalar shape, still 0 gathers).
3. **Fix:** express each pool update as `f64x8` repr ops (the
   `vwin_slide64x8!` pattern from fused.rs):
   `(f64x8::load(token, &pools[j].0) + f64x8::from_array(token, a))
    .store(&mut pools[j].0)` with `a = from_fn(|l| bufs[j][l] as f64)`.
   One vector load/add/store per pool — no scalar per-lane RMW for LLVM
   to vectorise into gather/scatter. `define(f32x8, f64x8)`; dispatch
   re-enables `[v4x, v4, v3, neon, wasm128, scalar]` for Canon64.
   v4x body: 1512 insns, **0 gather/scatter**, 289 zmm ops.
   Timing (interleaved, Rev4 scalar 1 MP, 5 iters/run): NEW 64-67,
   v3-dispatch 68-71 (r1 cold outliers 84-87 both) ms/iter -> ~5 % of
   the cell, over the 3 % bar. Kernel share 18.45 % -> 13.30 %.
   Bit-exact: each pool lane is one IEEE f64+f64 add, same order as the
   scalar `add(l, v)`; `dense_canon64_vec_matches_scalar_on_every_
   compiled_tier` extended with v4/v4x arms — passes (incl. r4 hook
   cases 0/1).

## Remaining native split (Rev4 scalar 1 MP, after dense fix)

box_blur_v_copy 18.2, rec64_rows8x8 17.3, vblur_ssim 16.9, dense 13.3,
gradient 6.6, srgb_xyb 4.9, memset 4.9, box_blur_h 4.7. No single change
left that plausibly saves >= 3 % of a cell without a numeric revision —
stopping per lane rule.

## Gates (2026-10-04, AFTER = FINAL build incl. cfg_attr dead-code fix)

- `cargo fmt -p zensim --check` clean (fmt pass applied to my hunks).
- `just clippy` rc=0 (-D warnings, workspace all-targets all-features).
- `just lint-scripts`: 803 scripts, all runnable.
- `just api-doc-check`: public_api_surface_docs_are_current ok.
- `just rev4serve-gate`: rev4_featpot_bake_served_and_steered ok.
- Release suite `cargo test -p zensim --release --features
  custom-profiles,feature-regime-v2,threads,training`: 3 consecutive
  green runs (580 lib tests + all integration bins incl.
  rev4_maps_are_tier_identical, rev4_served_vectors_bitmatch_research_
  extract_on_every_tier, rec64_rows_vec_matches_scalar_body_on_every_
  compiled_tier, dense_canon64_vec_matches_scalar_on_every_compiled_
  tier with new v4/v4x arms).
- `cargo test -p zensim --release --all-features --lib`: 615 passed
  x3 runs, 0 failures.
- Feature-permutation sweep (26 cells, clippy -D warnings +
  test --no-run, /var/tmp/kernopt2/target-perm): FAILS=0.
- `cargo check -p zensim --tests` x aarch64-unknown-linux-gnu,
  i686-unknown-linux-gnu, wasm32-unknown-unknown: clean; the only new
  warning (dead rev4_dense_chunk8 off-x86) fixed via cfg_attr.
- Steering panels byte-identical BEFORE(63838a2b)/AFTER, both dense
  bakes: owner 48 cases x2 revs 0 differing; broad-96 x4 arms 384 rows
  x2 revs 0 differing (/var/tmp/kernopt2/gate).
- Final timing: see table below.

### Before/after wall time — `extract_paths_bench`, bake arm

ST, `taskset -c 8 nice -n 19`, `RAYON_NUM_THREADS=1`, **native tier (v4x)**,
interleaved BEFORE/AFTER, 3 rounds per arm, median of per-invocation ms.
BEFORE = build on parent `63838a2b` (sha256 `90165a3b…`,
`/var/tmp/kernopt2/bench/extract_paths_bench_BEFORE2`); AFTER = this tree
(`c7bb4515…`, `extract_paths_bench_FINAL`). Load ~4 (one unrelated
nice-19 exhaustive test on cores 9-15; core 8 dedicated).

| cell | 1 MP before | 1 MP after | ratio | 4 MP before | 4 MP after | ratio |
|---|---:|---:|---:|---:|---:|---:|
| v2basic r4 scalar | 112.0 | 65.1 | **1.72×** | 499.8 | 268.9 | **1.86×** |
| v2basic r4 prepared | 221.6 | 147.3 | **1.50×** | 891.2 | 573.1 | **1.56×** |
| by_v2fy r4 scalar | 53.5 | 36.2 | **1.48×** | 258.9 | 151.5 | **1.71×** |
| by_v2fy r4 prepared | 109.9 | 84.9 | **1.29×** | 480.4 | 331.8 | **1.45×** |
| v2basic r3 scalar (anchor) | 51.1 | 51.6 | 0.99 | 218.1 | 219.3 | 0.99 |
| v2basic r3 prepared (anchor) | 138.9 | 136.3 | 1.02 | 516.4 | 519.3 | 0.99 |
| by_v2fy r3 scalar (anchor) | 29.9 | 29.8 | 1.00 | 126.1 | 127.1 | 0.99 |
| by_v2fy r3 prepared (anchor) | 78.6 | 77.8 | 1.01 | 303.6 | 300.4 | 1.01 |

Anchors within ±1 %; every AFTER round beat every BEFORE round in every
r4 cell.

**Rev4/Rev3 ratio per cell (AFTER):**

| cell | 1 MP | 4 MP |
|---|---:|---:|
| v2basic scalar | 1.26× | 1.23× |
| v2basic prepared | 1.08× | 1.10× |
| by_v2fy scalar | 1.21× | 1.19× |
| by_v2fy prepared | 1.09× | 1.10× |

At the lane's parent commit (63838a2b, dense already on v3) r4/r3 scalar
was 2.19×/2.29× (v2basic 1 MP/4 MP); it is now ~1.2–1.26×. Against the
pre-63838a2b number in the brief (~2.97×) the scalar cell improved ~2.4×
end to end.

### Remaining native split (FINAL binary, r4 scalar 1 MP, 25 iters)

box_blur_v_copy 18.8, vblur_ssim canon64 17.4, rec64_rows8x8 16.1,
dense canon64 v4x 13.1, gradient 7.3, memset 5.6, box_blur_h 4.7,
srgb_xyb 4.5 (contains one zmm vgatherdps, ~1.6 % of cell — below bar).
