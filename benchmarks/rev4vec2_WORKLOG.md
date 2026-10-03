# REV4VEC2 worklog — vectorise the f64 canonical blur/pool kernels

Lane: SWE-2 (devin). Started 2026-10-03. Change: `quarantine/swe2/rev4vec2`
(first rev `57de2c1f`, rebased onto `a7a3168d` = E15 as `76c26048`).
Parent at start: `05680070` (REV4VEC landed). Scope per coordinator task:
vectorise Rev4's canonical f64 Rec64 blur + pool/block machinery
lane-parallel, bit-exact (`to_bits`) vs the scalar canonical bodies, on the
fused-FMA tiers; scalar and wasm128 keep the scalar body. Rev1–Rev3
untouched.

## Kernel-cost ranking (input to cost ordering)

Callgrind on a real Rev4 bake, `extract_paths_bench`, ST, worktree build of
2026-10-03 04:43–05:02 (`/var/tmp/rev4vec2/cg/{scalar,prepared}_{1024,2048}.out`):

| kernel (inclusive) | 1MP scalar | 4MP scalar | 4MP prepared |
|---|---:|---:|---:|
| `restore_cuts::run_cell` | 27.3% | 27.2% | 21.5% |
| `fused_vblur_features_ssim` | 25.2% | 25.1% | 30.0% |
| `gradient_block_kernel` | 16.0% | 15.9% | 12.6% |
| `dense_block_kernel` | 16.0% | 15.9% | 12.6% |
| `fused_blur_h_ssim_at_revision` | 13.7% | 13.6% | 16.5% |
| `VWin::slide` | 6.1% | 6.1% | 7.2% |
| libm `fma`/`fmaf`/`fma_with_fma` | 6.9% | ~4% | — |
| totals | 36.66 G Ir | 98.23 G Ir | 124.12 G Ir |

`run_cell` is inclusive of the blur kernels it calls (mapdev/z1max side
passes share the same Rec64 win); its exclusive own-loop is the map-row
eval + WelfordVar pushes. Ordering matches the coordinator's ~79%-in-f64
estimate; dense+gradient alone are ~32% at both sizes.

## What changed (all inside Rev4 canonical arms; Rev1–Rev3 paths unmodified)

- `zensim/src/blur.rs` — `fused_blur_h_ssim_canon` Rec64 arm: new
  `#[magetypes]` `fused_blur_h_ssim_canon64_v8` processes 8 independent rows
  in `f64x8` lanes (rows are independent; the serial x-recurrence is kept
  per-lane, same init/add/sub order as the scalar row helper
  `blur_h_ssim_rec64_row` — extracted verbatim so vector tail + scalar
  siblings share the one body). Tail `height % 8` rows call the same helper.
- `zensim/src/fused.rs` — `fused_vblur_ssim_canon`: new
  `fused_vblur_ssim_canon64_v8` on `f32x8` formula eval + `f64x8` Rec64
  window state via `VWin::slide64x8` (one slide call per 8-pixel chunk,
  lane-local state, same `(sum + add) - remove` order); pool accumulation
  via `LanesF64::add_chunk` (= per-element `add`, widening per element);
  `width % 8` tail calls the extracted scalar `vblur_ssim_elem` verbatim.
  Same for `fused_vblur_edge_canon64_v8` (edge stats + scalar `scatter`
  increments; per-lane `at32` f64 window state). `VWin` gained
  `f64_states()` + `slide64x8`; `slide`'s F64 arm and `at32`'s F64 arm got
  empty-plane guards for `sq`/`s12`/`act` — see BUG note below.
- `zensim/src/feature_v2/restore_cuts.rs` — `run_cell` Canon64 arm: new
  `restore_cuts_row_work_v8` computes the map-row eval (`mapdev`/`z1max`
  per-pixel formula, unfused ops kept unfused) and the per-lane
  `WelfordVar` pushes (x mod 8 substream → f64x8 lane; `change/n` stays a
  true division; `m2 += change*(x-mean)` stays unfused mul+add). Tail
  `width % 8` and non-Canon64 modes unchanged.
- `zensim/src/feature_v2.rs` — `dense_block_kernel_canon` and
  `gradient_block_kernel_canon`: per-element work extracted verbatim into
  `dense_elem_canon` / `gradient_interior_elem_canon` (scalar siblings
  call the same helpers); new `#[magetypes]` `..._canon64_v8` bodies run
  the element math in `f32x8`/`f64x8` lanes + `LanesF64::add_chunk`,
  column-parallel like REV4VEC's f32 bodies. All 6 const-generic combos
  of the gradient kernel tier-dispatch the same way (`go64!` arm).
- Dispatch is inside the existing `Mode::Canon64` arms only, via
  `archmage::incant!` on v3/v4/v4x (and NEON where compiled) — scalar and
  wasm128 keep the scalar canonical body because their `mul_add` is
  unfused and must stay bit-identical.

## Bit-exactness gates (all PASS, run on the rev4vec change)

Added `to_bits` parity tests, each run under
`archmage::testing::lock_token_testing()` across every compiled tier,
randomised + odd/tiny sizes + tail widths + special-value injections:

- `blur::tests::rec64_rows_vec_matches_scalar_body_on_every_compiled_tier`
- `fused::tests::rev4vec2::vblur_ssim_canon64_vec_matches_scalar_on_every_compiled_tier`
- `fused::tests::rev4vec2::vblur_edge_canon64_vec_matches_scalar_on_every_compiled_tier`
- `restore_cuts::tests` row vector parity (mapdev+z1max+Welford lanes)
- `feature_v2::tests` dense + gradient canon64 vector parity

Plus the pre-existing revision-level gates (all PASS):
`rev4_served_vectors_bitmatch_research_extract_on_every_tier`,
`rev4_maps_are_tier_identical`, `rev4_identity_and_segments_across_all_tiers`,
restore-cuts prefix-identity/invariance suite, gmsbank tier consistency.

## BUG found + fixed (latent scalar-canonical panic, pre-existing)

`VWin::slide`'s F64 arm indexed `p.sq`/`p.s12` unconditionally while the
sibling f32 `slide32` early-returns on empty planes; edge-only canonical
paths pass empty `sq`/`s12`/`act`, so the scalar canonical edge body had a
latent `index out of bounds` panic (exposed by the new parity test; the
vector body intercepted the route in release production). Fix:
`slide`'s F64 arm and `at32`'s F64 arm now early-out/`0.0` on empty
planes, symmetric with the existing `act` guard and the `Rec` arm
conventions. Same fix semantics on every tier (the scalar body itself
changed — it was unreachable-but-panicking before; where reachable its
outputs are unchanged).

## Serving / baseline gates (all PASS)

- `just rev4serve-gate` real featpot bake:
  `rev4_featpot_bake_served_and_steered` — ok (≈77 s).
- `rev1_baseline_capture`, `rev2_baseline_capture`, `rev3_baseline_capture`
  — byte-identical.
- `rev4_maps_are_tier_identical`, serving steering gates — ok.

## Mechanical gates (all PASS)

- `cargo clippy --workspace --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings` — clean (fixed `assign_op_pattern`, `manual_is_multiple_of`, `unnecessary_lazy_evaluations`, `useless_vec` in new code).
- `cargo fmt -p zensim --check` — FMT-OK.
- `just lint-scripts` — 800 scripts checked.
- `just api-doc-check` — `public_api_surface_docs_are_current` ok (no public API change; new items are `pub(crate)`).
- Full release suite
  `cargo test -p zensim --release --features custom-profiles,feature-regime-v2,threads,training`
  — PASS, rc=0, 894 s (all lib tests + every test file + doctests; the
  only skips are the documented `#[ignore]`/feature-gated ones).

## Before/after benchmark

Paired-interleave A/B, `/var/tmp/rev4vec2/bench/paired1/` —
script `/var/tmp/rev4vec2/run_paired.sh`: pinned CPU 8, `nice -n19`,
`ionice -c3`, `RAYON_NUM_THREADS=1`, sizes 1024+2048, 15 rounds
(min 6, wall cap 120 s), 3 blocks per (mode × rev × binary), BASE and NEW
alternating inside each cell (drift lands on both arms of a pair).
BASE = `05680070` built in the throwaway workspace
(`/var/tmp/rev4vec2-base-target/...`, sha256 recorded in
`paired1/binary_base.sha256`); NEW = the rev4vec2 worktree build
(`paired1/binary_new.sha256`). Rev3 arms + `fast_ssim2_st` are the
revision-independent drift anchors.

Machine conditions: concurrent work present — a runaway `#[ignore]`d
exhaustive test of this same lane (`zensim-… midp_x16_matches_scalar…`,
affinity 9-15, does not overlap CPU 8), the steercheck lane's builds, and
the featpot lane's `zensim_mlp_train` pack (affined to 0-7,16-23 and
24-31; CPU 8 free of them; most exited by the end of the run — see
`paired1/run.meta` start/end competitor dumps). zenbench marks the runs
`unreliable` under the ambient load; the paired design below plus the
drift anchors bound that noise.

Binaries measured (sha256):
- BASE `05680070`: `56fdac4abf9a912a70b81308fafec3501200b384c3ac1d83bb01e7e29c8327e2`
- NEW rev4vec2: `ccd206cecdf94527d15f69f55f61c98b948861c198ec2b874692baec5eaf83ad`

## RESULTS — featpot_v2basic served cost (median of 3 block means, ms)

| arm | size | rev4 BASE | rev4 NEW | speedup | rev4/rev3 BASE | rev4/rev3 NEW |
|---|---|---:|---:|---:|---:|---:|
| scalar score | 1024 (1 MP) | 1265.4 | **1016.7** | **1.245×** | 2.06× | **1.67×** |
| scalar score | 2048 (4 MP) | 5081.2 | **3923.8** | **1.295×** | 2.03× | **1.60×** |
| prepared map | 1024 (1 MP) | 1545.2 | **1226.1** | **1.260×** | 2.02× | **1.67×** |
| prepared map | 2048 (4 MP) | 6145.7 | **4932.1** | **1.246×** | 2.12× | **1.68×** |

Per-block (ms, BASE vs NEW — every rev4 cell has ZERO overlap):
- scalar 1024: [1210.0, 1265.4, 1275.2] vs [978.4, 1016.7, 1021.9]
- scalar 2048: [4968.7, 5081.2, 5101.1] vs [3813.5, 3923.8, 3976.7]
- prepared 1024: [1528.0, 1545.2, 1587.0] vs [1215.6, 1226.1, 1232.7]
- prepared 2048: [6142.0, 6145.7, 6201.1] vs [4927.6, 4932.1, 4956.8]

Drift anchors (base/new ratio, noise floor):
- `featpot_v2basic` rev3 (untouched code): 1.012 / 1.019 / 1.040 / 0.989
  (scalar 1MP, scalar 4MP, prepared 1MP, prepared 4MP) — ±4 %.
- `fast_ssim2_st` (revision-independent): 0.951–1.018.
- `D_current_revision`: 0.986–1.051.

Every rev4 gain (1.25–1.30×) is ~6–30× the anchor noise and every new
block beats every base block in all four cells — the speedup is real,
not drift. The remaining rev4/rev3 gap ~1.6–1.7× is now in parts this
lane did not touch (streaming `process_scale_bands`, `dvifm`,
`append_block_kernel`, scalar `VWin::slide` remainder inside
restore-cuts windows, residual libm `fma` in scalar tails, and the
prepared session's second v1 walk — the latter owned by the concurrent
steercheck lane, deliberately not touched here).

Cumulative vs the pre-REV4VEC numbers recorded in `REV4VEC_DONE.md`
(same rig, same bake): scalar 1 MP 1380 → 1209 → **1017 ms**
(2.37× → 2.00× → **1.67×** over rev3); prepared 4 MP 6590 → 6276 →
**4932 ms** (2.34× → 2.06× → **1.68×**).
