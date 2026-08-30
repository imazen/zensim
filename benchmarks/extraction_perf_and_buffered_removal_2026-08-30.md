# Extraction perf, and whether the buffered path can go — 2026-08-30

Four questions, answered with measurements:

1. What is the fastest feature-extraction path for each 944 regime — pools
   zeroed (`folded720append2`) and pools live (`folded720append2pools`)?
2. Is the BUFFERED mode removable without hurting the multithreaded paths?
3. Is buffered slower than streaming?
4. Can we have both maximum perf and clean code?

Short answers, each expanded below with numbers: **(1)** the streaming fold is
the only path for either 944 regime, and it is now 5.6 % / 6.2 % cheaper in
instructions than it was this morning; **(2) NO — four independent blockers,
none of them perf**; **(3)** the question is mis-posed for the fold (it has
been streaming-only since the C5 switchover) and for v1 the two are the same
walk with different plane residency; **(4)** yes, and the same change bought
both — the biggest win of the day came from *deleting* redundant work, not
from adding a specialisation.

Terminology, because "buffered" and "streaming" are overloaded in this repo:

| name used here | what it is | entry |
|---|---|---|
| **BUFFERED v1** | whole-image XYB pyramids materialised for both sides, then band-processed per scale | `metric.rs:3145` `compute_with_config_inner` → `streaming.rs:862` `compute_multiscale_stats_streaming` |
| **v1 strip** | the same walk with an O(strip) plane budget | `streaming.rs:3484` / `:3629` `compute_multiscale_stats_streaming_strips*` |
| **the fold** | the 720/924/944 streaming walk; no prepared reference, no materialised pyramid | `feature_v2.rs:6292` `foldapp_streaming_walk` |

Note that `compute_multiscale_stats_streaming` LIVES in `streaming.rs` and is
still the buffered path — "streaming" there refers to band processing inside a
scale, not to plane residency. This naming has misled at least one prior
session; the table above is the disambiguation.

---

## 1. Consumer audit of the buffered path

Read-only sweep of the zensim workspace plus every `~/work/zen/` repo that
depends on it. The headline finding is structural, and it reframes the whole
question:

> **`ZensimV2Result` has no `score()`** (`feature_v2.rs:1082-1110`). The fold
> is an EXTRACTOR. Every scoring entry point — `compute`, `compute_with_ref`,
> `compute_with_ref_and_diffmap`, `classify`, `compute_pu_linear*`,
> `compute_streaming_strips*` — runs the buffered pipeline. The buffered path
> is not a legacy extractor kept around beside the fold; it is the shipped
> metric.

### In-repo consumers (abridged to the load-bearing rows)

| consumer | entry | reachability | fold-substitutable? |
|---|---|---|---|
| `metric.rs:1366` `compute`, `:1402` `compute_with_codec_hint` | buffered core → score | **PRODUCTION** | **No** — no score in the fold |
| `metric.rs:1466` `compute_extended_features` | buffered core, 300/372 | **PRODUCTION** | No — bit-exact only where `simd_padded_width(w) == w` |
| `metric.rs:2021/2271` `compute_with_ref{,_into}`, `:1991` `precompute_reference` | `streaming.rs:3785` | **PRODUCTION** | No — the fold has no ref-cached form |
| `metric.rs:2127/2208` `compute_streaming_strips*` | `streaming.rs:3484/:3629` | **PRODUCTION** (>16 MP path) | No |
| `metric.rs:2390/2459/2525` `compute_pu_linear*` | `streaming.rs:930/:975` | **PRODUCTION** | Partly — the fold's HDR entries exist only at 924/944 |
| `diffmap.rs:758/:925` `compute_with_ref_and_diffmap*` | `streaming.rs:3806/:3838` | **PRODUCTION** | No |
| `attribution.rs:1137..1266` `compute_attribution_density*` | `basic_canvas_trimmed:1378` → `PrecomputedReference` | **PRODUCTION** | **No** — the basic canvas is buffered-native |
| `attribution.rs:3210/:3234/:3551/:3577` `compute_with_ref_score_and_attribution*` | `streaming.rs:4062/:4156` | **PRODUCTION** | No |
| `zensim/examples/v2_ab_extract.rs:320/:374` | `compute_zensim_with_config` | **PRODUCTION fleet extractor** | Only its `fold*` modes (already implemented at `:406-421`) |
| `zensim-bench/examples/extract_features_372col{,_omni}.rs` | `compute_zensim_with_config` | **PRODUCTION dataset extractor** | Only at exact widths |
| `zensim-validate/src/main.rs:865/:1753`, `scale_invariance.rs:339` | `compute_zensim_with_ref_and_config` | **PRODUCTION corpus builder** | No — no ref-cached fold form |
| `zensim-target/src/lib.rs:236/:330`, `zensim-regress` (published crate) | `.compute()` / `.classify()` | **PRODUCTION** | No |
| `bake_verdict`, `zensim_mlp_train`, `bake_dial_refit`, `panel`, `freeze_check` | — | — | **n/a — parquet-only, fully decoupled from the extractor** |

`extract_features_372col` is a cargo EXAMPLE, not a symbol
(`zensim-bench/examples/extract_features_372col.rs:195`).

### Fleet routing (`zenmetrics jobexec`)

`zenmetrics-cli/src/jobexec.rs:126-142` maps the metric string to a regime;
`metrics/zensim.rs:220-227` early-returns the four folded regimes into
`extract_features_folded_streaming`, and `:305-315` maps them to
`V1PoolsMode::{Off, Off, Carriers, Full}` — so `zensim-foldapp2pools` is the
`Full` route and `zensim-foldapp2` the `Off` route, both on the fold.
**Everything else is still buffered in production**: `--metric zensim`
`score-pairs --feature-output` (`zensim.rs:108`), `sweep/run.rs:1319`, the
`V2Ab` route (`zensim.rs:259` + `:266`), the ref-ctx variant (`zensim.rs:461`),
HDR (`hdr.rs:1211`), and `zenfleet-vastai`'s inline worker
(`worker/inline.rs:607`). `with-iw` (372) is the CLI default and is hard-coded
in seven live sweep scripts.

### Cross-repo

`jxl-encoder`'s production loop (`vardct/zensim_loop.rs:953/1198/1212/1329`)
consumes the 372-class attribution/diffmap arms. `zensim-cpu-gpu-bench` exists
to compare against the buffered 372. `zenmetrics/crates/zensim-gpu` duplicates
CPU extraction in CubeCL for regimes 228/300/372 ONLY and uses
`compute_extended_features` as its correctness oracle in four test files —
deleting buffered removes the GPU kernel's only reference. ~20 further repos
(zenjpeg, zenwebp, zenavif, zenpipe, coefficient, imageflow, …) consume
`.compute()` / diffmap and therefore run the buffered walk internally.

### Width divergence — MEASURED, and much larger than "a tolerance"

The gate is `folded720_v1_pools_match_v1_path` (`feature_v2.rs:11121`,
`#[cfg(feature = "training")]` — **the plain
`--features custom-profiles,feature-regime-v2` invocation compiles it out
silently**). It asserts `to_bits()` equality only where
`simd_padded_width(w) == w`; elsewhere it asserts finiteness and *prints* the
divergence. Running it and reading the print:

```
pool parity  96x64:  BIT-EXACT
pool parity  64x300: BIT-EXACT
pool parity 208x144: BIT-EXACT
pool parity 127x93:  padded-width class (v1 pads 127->128), max rel 1.739e-1
pool parity 200x150: padded-width class (v1 pads 200->208), max rel 8.155e-1
```

**17.4 % at 127 px and 81.6 % at 200 px.** That is not a rounding question,
and the divergent class has no tolerance assertion at all today.

Worse for practical purposes, `simd_padded_width` (`blur.rs:3748`) is
`(w+15) & !15`, **plus another 16 when that is ≥ 512 and an even multiple of
16** (an anti-aliasing stride trick). So essentially every production width is
in the divergent class:

| width | 64 | 96 | 127 | 200 | 208 | 256 | 512 | 576 | 768 | 1024 | 1152 | 2304 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 walks | 64 | 96 | 128 | 208 | 208 | 256 | **528** | **592** | **784** | **1040** | **1168** | **2320** |

The buffered path computes its pools over those padded columns
(mirror-padded, `streaming.rs:3185`); the fold walks `w`. Two consequences:

* a fold row and a v1 row are **not** substitutable at any of the common
  sizes, so re-pointing the 372 consumers at the fold would change values;
* in the perf tables below the buffered arm is doing **+2.8 % / +1.4 % /
  +0.7 %** more column work at 576 / 1152 / 2304, which is stated wherever an
  absolute buffered-vs-fold comparison appears.

### Multithreading — the two paths parallelise on different axes

| | BUFFERED | FOLD |
|---|---|---|
| unit | **band per strip** | **channel** |
| site | `streaming.rs:2641-2647` `band_aux.into_par_iter().map_init(...)` | `feature_v2.rs:6401-6408` + `:6436` `par_iter_mut()` |
| degree | `layout_h.div_ceil(STRIP_INNER)` — **scales with image height** | **fixed 3** |
| producer | — | `feature_v2_stream.rs` has **zero rayon** (serial rolling walk) |

This is the sharpest perf answer in the audit and it is structural, not
incidental: **the buffered path's parallelism grows with the image; the fold's
is capped at 3.** Any claim that the fold can replace buffered at high thread
counts has to clear that, and the thread sweep below is where it is tested.
(zenmetrics sidesteps it by calling the fold with `.with_parallel(false)` and
parallelising across pairs instead — which is the right shape for a fleet, and
is not available to a single-pair product caller.)

---

## 2. What was shipped: the rem-ring

### The observation

`cargo flamegraph`/`perf` are unavailable here (`perf_event_paranoid = 4`
under WSL2 refuses even user-space CPU events), so profiling is callgrind on a
`--no-default-features` build without `avx512` — valgrind cannot execute
AVX-512, per the workspace rule. Instruction counts are deterministic, so they
are unaffected by the shared box.

**Read the instruction counts as a STRUCTURE finding, not as production
timing.** valgrind cannot execute AVX-512, so the profiled binary is built
`--no-default-features` (no `avx512`) and the hot symbols it reports are the
`__arcane_*_v3` (SSE4.2) monomorphisations. Two consequences: the *relative*
size of kernels can shift at the `v4`/`v4x` tiers the product actually
dispatches to, and `dense_block_kernel`'s `POOL_SIMD` path is v4x-only so it
is OFF in this profile entirely. What transfers cleanly is the *reason* a
kernel is expensive — the gather structure below is identical in all four
SIMD variants — and the wall-clock table in §4, which is measured on the real
(avx512-enabled) build, is the ground truth for how much it is worth.

Profiling the 944 fold at 576² split by `V1PoolsMode`, the live-pools delta
was **+84.4 M Ir**, and it was not spread evenly:

| kernel | pools Off | pools Full | delta | share of the pool cost |
|---|---:|---:|---:|---:|
| `box_blur_h_inner_v3` | 63,831,285 | 103,327,722 | **+39,496,437** | **47 %** |
| `ssim_channel_inline_both_inner_v3` | 0 | 11,985,990 | +11,985,990 | 14 % |
| `box_blur_v_copy_inner_v3` | 44,895,045 | 54,204,960 | +9,309,915 | 11 % |
| `fused_vblur_ssim_inner_v3` | 61,564,959 | 64,869,759 | +3,304,800 | 4 % |
| everything else | | | ~+20 M | 24 % |

Nearly half the price of the live-pools regime is one H-blur, and the
activity block's H half alone outweighs its V half **4.2 : 1** despite doing
the same amount of blurring. The reason is in the codegen: a horizontal
sliding-sum is sequential in x, so the kernels vectorise **across rows** —
each x-step assembles a vector from 16 (or 8) *strided scalar loads*, and does
it twice, for the add-side column and the remove-side column, then scatters 16
scalar stores. That is **3 scalar memory ops per output element**, where the
vertical blur reads and writes contiguous vectors.

### The identity

For every `x >= diam`:

* `add_idx(x - diam) = (x - diam) + r + 1 = x - r`, unmirrored because
  `x - r < width` always, and unclamped because `x - r <= width - 1`;
* `rem_idx(x) = x - r`, unmirrored because `x >= diam > r`, same clamp.

They are **the same column**. The remove-side gather has been re-reading bytes
the add-side gather already loaded `diam` steps earlier, for every horizontal
blur in the crate. Keeping the last `diam` add-vectors in a stack ring
replaces it with one contiguous load: **3 memory ops per element → 2** for the
single-plane kernels, and **4 gathers → 2** for the two-plane ones.

This is bit-exact **by construction, not by tolerance**: same memory, same f32
values, same `(sum + add) - rem` evaluation order. Nothing is recomputed or
reassociated. `ring_pos` is a counter, never `x % diam` (an integer division
in the hot loop would have eaten the win); it resets per row-group, the first
`diam` steps of each group still gather explicitly so the mirrored init region
is untouched, and `radius > 16` falls back to the two-gather form — making
`H_RING_CAP` a perf bound, never a correctness one.

Applied to all four H-blur families, 24 gather sites across the `v4`, `v4x`,
`v3` and `magetypes`-generic variants (including the two `row_off` closure
forms the generic ssim/mu variants use).

### MEASURED (callgrind Ir, 576², serial, v3 tier, minus the constant 27,549,000 the bench harness spends building the image)

| arm | baseline Ir | after Ir | delta | % |
|---|---:|---:|---:|---:|
| `buf_v1_372` (BUFFERED v1, 372) | 365,966,196 | 335,621,513 | −30,344,683 | **−8.29 %** |
| `fold944_off` (`folded720append2`) | 485,994,568 | 458,918,753 | −27,075,815 | **−5.57 %** |
| `fold944_full` (`folded720append2pools`) | 570,380,120 | 534,913,840 | −35,466,280 | **−6.22 %** |

Per kernel:

| kernel | arm | before | after | % |
|---|---|---:|---:|---:|
| `fused_blur_h_ssim_inner_v3` | buffered | 87,070,506 | 72,966,483 | −16.2 % |
| `fused_blur_h_ssim_inner_v3` | both folds | 77,633,121 | 64,508,874 | −16.9 % |
| `box_blur_h_into_abs_diff_inner_v3` | buffered | 47,713,545 | 40,088,604 | −16.0 % |
| `box_blur_h_inner_v3` | buffered | 40,583,637 | 31,930,440 | −21.3 % |
| `box_blur_h_inner_v3` | fold, pools Off | 63,831,285 | 49,896,194 | −21.8 % |
| `box_blur_h_inner_v3` | fold, pools Full | 103,327,722 | 80,978,858 | −21.6 % |

Every other kernel in the profile is unchanged **to the instruction**
(`dense_block_kernel_entry_v3` 125,098,677 on both sides, `fused_vblur_ssim`,
`box_blur_v_copy`, `srgb_to_positive_xyb_planar`, `downscale_2x_into`), which
is the profile confirming the change is isolated to what it edits.

**The BUFFERED path gains more than the fold** (−8.3 % vs −5.6 / −6.2 %),
because it is the more H-blur-dominated of the two: three H kernels are 44.5 %
of its instructions against roughly 25 % of the fold walk. That is worth
stating plainly, because the lever was found while profiling the fold.

### Gates

Three new reference tests in `blur.rs`, each comparing `to_bits()` against a
scalar per-row reference that copies the kernels' evaluation order exactly —
including `(sum + add) - rem` (**not** `sum += add - rem`) and the nested
`mul_add` chains for `sigma_sq` / `sigma12` — over nine geometries straddling
the 16-row group, the 8-row remainder and the scalar/masked tail, and every
`width` vs `diam` relation including `width <= diam` (ring never engages) and
`width == diam` (the boundary), at radius 1 / 2 / 5 / 8:

* `box_blur_h_ring_matches_regathered_reference`
* `abs_diff_h_ring_matches_regathered_reference`
* `fused_h_ring_matches_regathered_reference`

**Negative control RUN, not asserted**: changing `x >= diam` to
`x >= diam - 1` fails all three (`14214.558` vs `7862.271`). A test that
cannot fail is worse than no test.

Full suite `--features custom-profiles,feature-regime-v2,threads,training`:
**333 passed, 0 failed** — including `v1_golden_bytes`,
`v1_same_class_determinism_bitexact`, `folded720_v1_pools_match_v1_path` and
`folded720_v1_basic_matches_v1_path`.

### A pre-existing wart found while writing the reference — documented, NOT fixed

`fused_blur_h_mu_inner_{v4,v4x,v3}` still carry a **scalar remainder** for the
last `height % 8` rows which accumulates `sum += add - rem` = `sum + (add -
rem)`, while their vector bodies evaluate `(sum + add) - rem`. f32 addition is
not associative, so those tail rows differ from the vector rows in the last
ulp or two — MEASURED at `2528.7349` vs `2528.7344` (7×8, r=1). It is proven
pre-existing: the identical assertion fails identically on the pre-ring
kernels. `fused_blur_h_ssim` already fixed exactly this wart in its generic
variant by masking the tail into a vector group (see the comment above its
`run_group`); the `mu` family was never converted.

Converting it would **move v1's shipped bytes**, so it belongs to the
golden-gate policy and a deliberate decision, not a perf drive-by. Recorded
here and in the test's own doc comment. Note the production band shape (42
rows = 32 + 2×5 overlap) *does* hit this tail.

---

## 3. Levers considered and rejected, with the reason

| lever (source: `pools_full_extraction_2026-08-30.md` ranking) | verdict |
|---|---|
| **#1 `box_blur_h_of_abs_diff`** — fuse `|src − mu1_h|` into the activity H-blur's load sites so `act_raw` is never materialised | **NOT SHIPPED, and the ranking is now stale.** It was ranked first on the assumption that removing a band-sized round-trip is free. It is not: the kernel it lands in is *gather-bound*, and taking `|a − b|` at the load sites means gathering **two** strided planes where the ring now gathers one — re-adding exactly the cost the ring just removed, to save a contiguous pass. The trade is +1 strided gather against −3 contiguous ops per element, which is genuinely uncertain and must be measured, not reasoned about. It should be re-ranked *after* the ring, not before it. |
| **#2 inner-rows-only V-blur write** in `box_blur_1pass_into`'s V half | **REJECTED ON VALUE, measured.** The band is `V1_BAND_ROWS + 2·V1_BAND_OVERLAP` = 42 rows of which 32 are read, so ~24 % of that pass's stores are dead. But the activity V-blur is only 9,309,915 Ir of the pool block (`box_blur_v_copy` 44.9 M → 54.2 M), so 24 % of its stores is ≈ 2.2 M Ir ≈ **0.4 % of the walk** — for a new kernel variant and a new correctness surface. |
| **#3 per-band ±overlap activity recompute** | Not removable — v1 mirror-clamps the activity blur at its own strip edges, and reproducing that boundary behaviour is what makes the fold v1-exact. (Unchanged from the prior lane's finding.) |
| **art-L4 weighted sums inside the fused V-blur kernel** | Still unshippable bit-exact, for the reason the prior lane established: the fused loop is column-group-major and folds into `f64` in that order, while `simd_ops::*` is row-major; moving the math changes the `f64` summation order and `f64` addition is not associative. Unchanged. |

The one genuinely new candidate this profile surfaces is
`dense_block_kernel_entry` — **125,098,677 Ir, the single largest kernel in
both fold arms (22–26 %)**, identical in both, i.e. pure v2/append cost that
neither regime escapes. It was **not** attacked here, and on reading it that
is the right call rather than a punt: it has already had two rounds of
targeted work (the §A.14 register-pressure fix that scalarised the 22
weighted-pool accumulators, and the 2026-07-21 `POOL_SIMD` re-vectorisation
that got them back to 16 lanes for the v4x tier only), plus an
`#[inline(always)]` that is load-bearing to the tune of a measured **5.3×**
whole-extraction regression when LLVM stopped honouring the hint. It is
tuned, it is fragile, and — because `POOL_SIMD` is v4x-gated — the 125 M
figure above is its *un*-POOL_SIMD form, so the number to beat has not even
been measured on the production tier. Anyone picking it up should start by
re-profiling at v4x, not from this row.


---

## 4. Head-to-head wall clock

`zensim/benches/extract_paths_bench.rs`, four arms **interleaved inside one
process** so the machine's noise is common-mode, zenbench's paired bootstrap
CI, `min_rounds(25)` / 600 s group budget (the floor the pools lane raised for
exactly this reason), `--features custom-profiles,feature-regime-v2,threads,
training`, no `-C target-cpu=native`, everything under
`run-heavy --mem 16G --jobs 8`.

**Load honesty.** The gate for these runs was "primary `.workongoing` absent
AND 1-min load < 8". **It never opened** — over 45 minutes of bounded polling
the box carried 6–20 load from three other lanes' builds and S3 syncs, and a
marker was continuously present (v1-width lane, then drift lane). The runs
below were therefore taken **NOT load-clean**, under `nice -n19 ionice -c3`
so they yield to those lanes rather than steal from them. What makes them
usable anyway is that the quantity of interest is the *paired ratio measured
inside one interleaved process*, not the absolute ms — the same argument the
pools lane recorded on 2026-08-30. Absolute ms in these tables are inflated by
an unknown amount and **must not be compared against numbers from any other
run**; the `vs base` columns and the within-table differences are the results.
One further mitigation found in passing: **zenbench takes a box-wide exclusive
lock** (`/tmp/zenbench/zenbench.lock`), so no *other* benchmark can overlap
these — only builds can.

### Serial (`RAYON_NUM_THREADS=1`), 20 / 20 / 19 usable rounds

| size | `buf_v1_228` | `buf_v1_372` | `fold944_off` | `fold944_full` |
|---|---:|---:|---:|---:|
| 576² | **8.49 ms** (base) | 12.77 ms `[+44.0 – +51.6 %]` | 16.33 ms `[+86.8 – +98.4 %]` | 20.21 ms `[+128.7 – +147.7 %]` |
| 1152² | **38.00 ms** (base) | 54.22 ms `[+40.3 – +49.2 %]` | 71.37 ms `[+78.5 – +97.3 %]` | 81.71 ms `[+99.5 – +119.4 %]` |
| 2304² | **166.2 ms** (base) | 231.4 ms `[+33.2 – +39.6 %]` | 338.3 ms `[+94.3 – +114.0 %]` | 379.4 ms `[+116.1 – +129.0 %]` |

The absolute columns are **not** a like-for-like comparison and should not be
read as one: `fold944_*` computes 944 features, `buf_v1_372` computes 372.
The honest same-feature-set comparison is the marginal one below.

### α + β·pixels — and why the fit is the finding

| arm | α (ms) | β (ms/MP) | ms/MP at 576² | at 1152² | at 2304² |
|---|---:|---:|---:|---:|---:|
| `buf_v1_228` | −3.04 | 31.8 | 25.6 | 28.6 | 31.3 |
| `buf_v1_372` | −2.94 | 44.1 | 38.5 | 40.9 | 43.6 |
| `fold944_off` | −9.81 | 65.4 | 49.2 | 53.8 | 63.7 |
| `fold944_full` | −8.91 | 72.9 | 60.9 | 61.6 | 71.5 |

**Every intercept is negative, which means the linear model is wrong, not that
there is a fixed-cost saving.** A negative α is the least-squares line bending
to accommodate a *convex* curve: per-pixel cost RISES with size in all four
arms (25.6 → 31.3 ms/MP buffered, 49.2 → 63.7 ms/MP for the zeroed fold), the
signature of a memory-bound walk whose working set outgrows L3 between 1152²
and 2304². Reporting a single "ms/MP" for any of these arms would be wrong at
both ends. The fold is the more size-sensitive of the two families (+29 %
ms/MP across the range vs +22 % buffered), consistent with it carrying more
live planes per strip.

### The like-for-like number: marginal cost of the SAME 216 v1 pool features

`buf_v1_372 − buf_v1_228` and `fold944_full − fold944_off` both price exactly
v1's pool block, each inside its own family, both differences taken between
arms interleaved in the same process:

| | 576² | 1152² | 2304² | α (ms) | β (ms/MP) |
|---|---:|---:|---:|---:|---:|
| BUFFERED marginal | 4.28 ms | 16.22 ms | 65.20 ms | +0.10 | **12.26** |
| FOLD marginal | 3.88 ms | 10.34 ms | 41.10 ms | +0.91 | **7.55** |
| fold vs buffered | **−9.3 %** | **−36.3 %** | **−37.0 %** | | **−38 %** |

**The streaming fold computes v1's own pool features ~38 % cheaper per pixel
than v1's buffered path computes them**, paying a larger fixed cost (0.91 vs
0.10 ms) that it recovers by 576². Both marginal fits have sensible positive
intercepts, unlike the totals above — differencing cancels the convex
plane-residency term, which is a good sign the two arms really do differ only
by the pool work.

The mechanism is visible in the profile and is not mysterious. For the same
216 features the fold does strictly less work:

* **activity map** — the fold already holds the H-blurred source as `mu1_h`
  (the shared fused H-pass plane), so it needs one `abs_diff_into`; the
  buffered path calls `box_blur_h_into_abs_diff`, which *recomputes* the whole
  H-blur (47.7 M Ir before the rem-ring, 40.1 M after);
* **sigma planes** — the fold takes `sigma1_sq` / `sigma12` from the fused
  V-blur kernel's `store_sigma` side-output (the pools lane's 2026-08-30
  change); the buffered path runs its own V sweeps.

That is a genuinely useful result for the two 944 regimes: the live-pools
regime is not paying a penalty for being folded — it is the *cheaper* place to
compute those features. What it does not do is make the fold substitutable for
v1 (§1: different values at padded widths, and no score).

---

## 5. Is BUFFERED removable? — **NO.** Four blockers, none of them perf

The brief's condition for removal was: the fold matches buffered's outputs
within the golden policy on the full golden set, AND is not slower on any
(size × threads) cell, AND every consumer can be re-pointed. **All three
fail**, and the first and third fail on structure rather than on numbers.

**Blocker 1 — the fold has no score.** `ZensimV2Result` exposes no `score()`
(`feature_v2.rs:1082-1110`). `Zensim::compute` and every sibling scoring entry
run the buffered walk. That surface is published (`zensim-regress` is a
released crate) and consumed by zenjpeg, zenwebp, zenavif, zenpipe,
coefficient, imageflow, `zensim-target` and ~15 more. Removing buffered
without first giving the fold a scoring path deletes the metric.

**Blocker 2 — the pool values are not the same numbers at production widths.**
`simd_padded_width` pads every width ≥ 512 that is an even multiple of 16 by a
further 16 columns, so 512/576/768/1024/1152/2304 all diverge, and the
measured divergence is **17.4 % at 127 px and 81.6 % at 200 px** — orders of
magnitude outside the golden policy's `max(1e-6 abs, 1e-5·scale)`. Re-pointing
any 372 consumer at the fold changes its answers. (Which of the two is
*correct* is a separate and real question — the buffered path pools over
mirror-padded columns that are not in the image — but that is a v1
byte-stability decision for the width lane and the user, not a perf lane's to
take.)

**Blocker 3 — no ref-cached fold form.** `precompute_reference` /
`compute_zensim_with_ref_and_config` amortise one reference across N distorted
candidates. That is the shape every encoder loop and the zensim-validate
corpus builder uses, and the fold has no equivalent — zenmetrics'
`zensim.rs:422-431` hard-errors for folded regimes on the ref-ctx path.

**Blocker 4 — attribution's basic canvas is buffered-native.**
`attribution.rs:1378` `basic_canvas_trimmed` builds a `PrecomputedReference`;
`compute_attribution_density*` and the fused
`compute_with_ref_score_and_attribution*` are all on it, and jxl-encoder's
production loop consumes the 372-class arms.

Two secondary findings that matter to the decision:

* **`zensim-gpu`'s only correctness oracle is `compute_extended_features`**
  (four test files in `zenmetrics/crates/zensim-gpu`). Deleting buffered
  removes the CPU reference its CubeCL kernels are validated against.
* **Multithreading is the one place the fold is structurally weaker.** The
  buffered path parallelises **band-per-strip**, degree
  `layout_h.div_ceil(STRIP_INNER)` — it grows with image height. The fold
  parallelises **per channel**, degree **fixed at 3**, over a strip producer
  (`feature_v2_stream.rs`) that contains no rayon at all. So the answer to
  "can the fold's rayon parallelism match buffered's at 8 and 28 threads" is
  *no, by construction*: past 3 threads the fold has nothing left to hand out.
  The measured thread sweep is §6.

### The smallest change that would unblock it

Not "make the fold faster" — perf is not what is in the way. In dependency
order:

1. **Give the fold a ref-cached entry and a scoring path** (blockers 1 + 3).
   Both are additive API, not a rewrite.
2. **Decide the width question deliberately** (blocker 2): either the fold
   adopts v1's `simd_padded_width` padding so the pool block is bit-exact at
   every width, or v1 stops pooling over pad columns — the second changes
   every stored 372 table and needs the golden-gate policy plus user sign-off.
   Until one of those lands, "bit-exact at SIMD-exact widths" is a caveat, not
   a substitution licence.
3. Only then re-point consumers, and only then delete.

**What IS deletable today**, found by the audit and left for a follow-up
because it is unrelated to this lane's measurements: five `simd_ops` kernels
(`ssim_channel_masked`, `ssim_channel_masked_2`, `edge_diff_channel_masked`,
`edge_diff_channel_masked_2_art4_det4`, `build_iw_weight_and_mse`) reachable
only from their own `#[cfg(test)]` module.

### Corollary: "is buffered slower than streaming?"

The question is mis-posed for the fold and needs splitting:

* **For the 720/924/944 regimes there is no buffered path to be slower than.**
  They have been streaming-only since the C5 switchover (2026-07-26), which
  was taken as a user-approved ~1.33× batch-CPU trade for a 4.7× memory
  reduction and one code path. Nothing to remove; the decision is already made
  and is not being revisited here.
* **For v1 the two are the same walk** with different plane residency
  (`compute_multiscale_stats_streaming` materialises whole-image pyramids;
  `compute_multiscale_stats_streaming_strips*` keeps an O(strip) budget), and
  the strip form exists for the >16 MP case where the buffered form does not
  fit. Both are live and neither is redundant.
* **Where the two families overlap — the 216 v1 pool features — the streaming
  fold is the faster one**, by 38 % per pixel (§4). That is the answer to the
  spirit of the question, and it is the opposite of the intuition that a
  general streaming walk must pay for its generality.

---

## 6. Thread sweep — the fold's serial advantage inverts under threads

Same bench, `RAYON_NUM_THREADS` set per process, everything else identical.

| arm | 576² 1T | 8T | ×  | 1152² 1T | 8T | × | 2304² 1T | 8T | × |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `buf_v1_228` | 8.5 | 2.9 | **2.93×** | 38.0 | 19.4 | **1.96×** | 166.2 | 44.6 | **3.73×** |
| `buf_v1_372` | 12.8 | 4.0 | **3.19×** | 54.2 | 25.1 | **2.16×** | 231.4 | 55.2 | **4.19×** |
| `fold944_off` | 16.3 | 12.4 | **1.32×** | 71.4 | 65.8 | **1.08×** | 338.3 | 244.6 | **1.38×** |
| `fold944_full` | 20.2 | 13.6 | **1.49×** | 81.7 | 74.4 | **1.10×** | 379.4 | 279.6 | **1.36×** |

**The buffered path takes 2–4× from eight threads; the fold takes 1.1–1.5×.**
That is the audit's structural prediction landing exactly: buffered
parallelises band-per-strip with a degree that grows with image height, while
the fold parallelises per channel with a **degree fixed at 3**, on top of a
strip producer that has no rayon in it at all. Past three threads the fold has
nothing left to hand out, and what remains — the serial rolling-plane walk —
becomes the floor.

Read alongside §4 this reverses the like-for-like verdict:

| marginal cost of the SAME 216 pool features | 576² | 1152² | 2304² |
|---|---:|---:|---:|
| serial — fold ÷ buffered | **0.91×** | **0.64×** | **0.63×** |
| 8 threads — fold ÷ buffered | **1.09×** | **1.51×** | **3.30×** |

**The fold computes v1's pool block more cheaply only while single-threaded.**
Its advantage is per-pixel work removed (§4); the buffered path's advantage is
that it can spread that same work over as many bands as the image has strips.
By 2304² at eight threads the fold pays **3.3×** what buffered pays for the
identical 216 features. Any decision that trades one path for the other has to
name its thread budget first — which is precisely why zenmetrics runs the fold
with `.with_parallel(false)` and parallelises across *pairs* instead. That is
the right shape for a fleet, and it is unavailable to a single-pair product
caller.

**Measurement quality, stated plainly.** zenbench flagged CV between 24 % and
90 % and drift in both directions on these runs — the box carried 6–22 load
throughout from other lanes. The effect sizes above (2–4× vs 1.1–1.5×;
0.63× → 3.30×) are one to two orders of magnitude larger than that noise, so
the *direction and rough magnitude* are safe. Individual cells are not: the
`buf_v1_228` 1152² 1.96× against 2.93×/3.73× at its neighbours is almost
certainly a noise artefact of that particular cell (its base CI spans
12.6–27.3 ms), not a real scaling dip at 1152². **Do not quote a single cell
from this table**; quote the pattern.

### 28 threads

| arm | 576² | vs base | 1152² | vs base |
|---|---:|---|---:|---|
| `buf_v1_228` | **2.3 ms** | base | **6.4 ms** | base |
| `buf_v1_372` | **2.8 ms** | +20.1 – +31.8 % | **8.4 ms** | +23.4 – +32.6 % |
| `fold944_off` | 10.3 ms | +334 – +368 % | 35.8 ms | +425 – +490 % |
| `fold944_full` | 11.0 ms | +349 – +398 % | 40.4 ms | +495 – +556 % |

`buf_v1_372` at 576²/28T measures **2.8 ms**, which independently reproduces
the 2.75 ms figure quoted with the subset mandate — a useful cross-check that
this harness and that one are measuring the same thing.

1T → 28T scaling: `buf_v1_372` **4.6× / 6.5×**, `fold944_full` **1.84× /
2.02×**. The gap is the 3-way channel cap.

### Peak RSS — the streaming design's payoff, and where it starts

`/usr/bin/time -v`, one arm per process (`ZEN_XP_RSS`), serial, 8 iterations.
The harness holds both input images (2 × w² × 3 B = 7.96 MB at 1152²,
31.85 MB at 2304²) in every arm; the "working set" row subtracts that constant.

| arm | 1152² peak | working set | 2304² peak | working set | 1152²→2304² |
|---|---:|---:|---:|---:|---:|
| `buf_v1_228` | 49.3 MB | 41.4 MB | 223.8 MB | 192.0 MB | **4.54×** |
| `buf_v1_372` | 48.7 MB | 40.8 MB | 203.2 MB | 171.4 MB | **4.17×** |
| `fold944_off` | 62.7 MB | 54.7 MB | 136.6 MB | 104.8 MB | **2.18×** |
| `fold944_full` | 68.1 MB | 60.2 MB | 146.5 MB | 114.6 MB | **2.15×** |

**The fold costs MORE memory at 1152² and much LESS at 2304²**, and the reason
is the shapes: buffered holds whole-image pyramids so it scales with area
(4.2–4.5× for 4× the pixels), while the fold holds O(strip × width) rolling
planes and scales closer to width (2.2×). The crossover sits between the two
sizes — at 2304² the fold's working set is **0.6×** buffered's, at 1152² it is
**1.4×**. So the C5 switchover's "4.7× memory reduction" is a large-image
property, and at small sizes the fold is the heavier of the two. Anything
choosing a path on memory grounds needs to know which side of ~1.5 MP it is
on.

---

## 7. The 372-as-a-subset-of-944 mandate — findings

The decision arrived mid-lane: v1-372 becomes a subset MODE of the single-pass
944 fold, buffered goes away once gates pass, and the gate is bit-exactness
across width classes and thread counts — with an explicit instruction to STOP
and report a class with numbers rather than ship a silent divergence. Here is
what is now known, ending in a named blocker.

### 7.1 The divergence is ENTIRELY v1's mirror-padded columns — CONSTRUCTIVE RESULT

The independent corroboration that 60 % of tbig rows differ between the fold's
`f0..371` and buffered v1-372 is consistent with everything measured here, and
the mechanism is now pinned to one line each:

* v1 starts its scale walk at `w = simd_padded_width(width)`
  (`streaming.rs:871`);
* the pad columns are filled by reflect-101 mirroring
  (`streaming.rs:3185` `mirror_pad_columns`, `period = 2*(width-1)`,
  `src_x = m if m < width else period - m`);
* so v1's pools and means include columns that are not in the image, and the
  fold's do not.

**The fix follows directly and is cheap: mirror-pad the RGB input by the same
rule and run the fold on the padded image.** sRGB→XYB is per-pixel, so
pad-then-convert and convert-then-pad are the same values. New gate
`v1_padded_width_divergence_is_column_padding` (`feature_v2.rs`) measures it
over 20 geometries:

| class | geometries | result |
|---|---|---|
| tight (`simd_padded_width(w) == w`) | 96×64, 208×144, 592×80, 128×93 | **0/372 slots differ** |
| even, non-tight | 200×150, 200×151, 576×96, 1152×72, 100×96 | **0/372 differ** |
| odd, non-tight | 127×64, 127×96, 127×128, 129×96, 201×96, 255×96, 577×80 | **0/372 differ** |
| **residual** | 126×93, 127×93, 255×93 | 64 / 50 / 37 slots differ, worst rel **8.920e-7 / 1.098e-6 / 2.287e-7** |

**17 of 20 geometries are BIT-IDENTICAL across all 372 slots**, including
200×150 — which un-padded was the 81.6 % divergence — and 576/1152, the
production classes. Crucially this needs **no change to the 944 regime's
pooling**, so the shipped 944-era tables stay reproducible: the pre-pad is
scoped to the 372-subset request, exactly the scoping the mandate requires.

### 7.2 The residual is a HEIGHT interaction, not a width class — NAMED BLOCKER

The three remaining cells are all at **h = 93**, and only when the width is
non-tight. The isolation is unambiguous:

* the same widths are **exact** at h = 64, 96 and 128 (127×64, 127×96,
  127×128 all 0/372);
* the same height is **exact** at the tight width 128 (128×93, 0/372);
* so it is an interaction between the pad columns and the pyramid's row-group
  tiling at that height (93 → 46 → 23 → 11 rows), not a property of any width.

Magnitudes are ≤ 1.098e-6 relative — inside the golden policy's `1e-5·scale`,
above its `1e-6` absolute floor — but **not bit-exact**, and the mandate's
gate is bit-exactness. **Per that gate this is the STOP-and-report class.** It
is not root-caused to a line yet; the next step is to bisect it by scale (the
11-row deepest level is the suspect) rather than to widen a tolerance.

### 7.3 Block-skipping is required, and the naive form is measured

The mandate requires a 372-only request to skip the v2-348/append work, and
requires that it not regress against today's buffered 372. The naive
alternative — compute all 944 and project — is measured here and is
disqualified:

| | 576² | 1152² |
|---|---:|---:|
| `buf_v1_372` (today, 28T) | 2.8 ms | 8.4 ms |
| `fold944_full` → project (28T) | 11.0 ms | 40.4 ms |
| **penalty** | **3.9×** | **4.8×** |

(At 8T the same ratio is 3.4× / 3.0×; at 1T, 1.6× / 1.5×. The penalty grows
with thread count because of §6 — projecting inherits the fold's 3-way cap.)

So block-skipping is not an optimisation, it is the feature. **It is not
implemented**: `foldapp_streaming_walk` hard-codes `let fold_v1 = true;`
(`feature_v2.rs:6303`) and the v2-348 block is unconditional — there is no
toggle today that computes v1's blocks without also computing v2's. Adding one
is tractable (the toggles struct and the phase-A/phase-B split are already the
right shape) but it is a real change to the walk, and its perf must be
measured on the same paired bench, not assumed.

### 7.4 What this lane did NOT do, and why

**The subset mode is not implemented and buffered is not removed.** Two
reasons, in order:

1. **The gate says stop.** §7.2 is a width×height class that is not bit-exact,
   and the instruction was explicit that such a class is reported with
   numbers rather than shipped over. Implementing the mode and re-pointing
   consumers before that class is resolved would bake a silent 1e-6 divergence
   into the eval roots — precisely the failure the gate exists to prevent.
2. The remaining work is genuinely large and each piece has its own gate:
   block-skipping (§7.3) + the pre-pad wiring (§7.1) + a fold scoring path and
   a ref-cached entry (§5 blockers 1 and 3) + attribution's buffered-native
   basic canvas (§5 blocker 4) + re-pointing ~25 consumers + the eval-root
   re-extraction equality run on cid22val / kon504 / tid.

**The acceptance-gate re-extraction was NOT run.** It requires the subset mode
to exist. What can be said now is that it *would* have failed before §7.1 (the
60 % tbig figure) and that §7.1 removes that failure mode for 17 of 20
geometries.

### Recommended order

1. Root-cause and fix the h = 93 interaction (§7.2) — smallest, and it is the
   gate.
2. Land block-skipping behind a toggle (§7.3); measure on `extract_paths_bench`.
3. Wire the pre-pad into the 372-subset request only (§7.1), leaving the 944
   regime's pooling untouched.
4. Run the eval-root re-extraction equality gate on cid22val + kon504 + tid.
5. Only then: fold scoring path + ref-cached entry, re-point consumers, delete.

**Public API note (for approval, nothing added yet):** steps 2–3 need one new
toggle/mode on the existing `V2NewFeatureToggles` surface, and step 5 needs a
scoring method and a ref-cached entry on the fold. No public API was added or
changed by this lane.

---

## 8. "Make the 372 segment of 944 correct" — the fork, measured

A second directive arrived: do **not** scope the buffered-equal semantics to a
372-subset flag; fix the fold's v1 blocks so they equal buffered v1 wherever
the 944-class regimes emit them, ship the corrected semantics under a new
regime/era tag, and accept the re-extraction cost.

Executing that hits a fork that the measurements above settle, and it needs a
decision before any code moves.

### 8.1 First, a correction to the premise

The framing supplied with the directive was that "the fold pools at the
SIMD-padded width where buffered v1 pools at unpadded semantics (or vice
versa; measure, don't assume)". **Measured: it is vice versa.**

* **BUFFERED v1 pools over PADDED columns.** It starts its walk at
  `w = simd_padded_width(width)` (`streaming.rs:871`) and mirror-fills the
  extra columns (`streaming.rs:3185`), so its means and pools include up to 16
  columns that are not in the image.
* **The fold pools over the image only.**

So on the plain reading of the word, **the fold's v1 blocks are already the
"unpadded semantics" ones, and buffered is the path carrying the artifact.**
The operative gate as stated — "corrected fold `f0..371` bit-identical to
buffered v1 at HEAD" — therefore asks the fold to *reproduce v1's phantom
columns*, not to drop them. Worth naming explicitly, because "correct" and
"bit-identical to buffered" point in opposite directions here.

### 8.2 The blast radius of the pre-pad — why it cannot just be switched on

§7.1's pre-pad makes `f0..371` bit-exact. It also makes the fold compute the
v2/append blocks **of the padded image**. Measured by the same gate (tight
widths as the control):

| width class | v1 block `f0..372` | v2/append block `f372..944` |
|---|---|---|
| tight (96×64, 208×144, 592×80, 128×93) | 0/372 differ | **0/552 differ** |
| non-tight (all 15 others) | 0/372 (17 cells) or ≤1.1e-6 (3 cells) | **505–508 of 552 differ**, worst rel **up to 36.4×** |

Worst movers: f487 rel **3.644e1** (129×96), f603 **2.166e1** (126×93), f866
**2.017e0**, f882 **1.350e0**, f432 **1.370e0**. These are not last-ulp
drifts; they are different features. The tight-width control (0/552) proves
the movement is caused by the pad columns and nothing else — that invariant is
now asserted by the gate, in both directions.

**So the pre-pad fixes 372 slots and destroys the comparability of 552.**
"Fix the 372 segment" and "leave the 944 regime alone" cannot both be had by
padding the walk.

### 8.3 The three options, with their measured cost

| | what changes | blast radius | perf cost |
|---|---|---|---|
| **A. Pre-pad the whole walk** | fold `f0..371` becomes buffered-exact; `f372..943` becomes features-of-the-padded-image | **92 % of the v2/append block moves, up to 36×.** Every 944 table re-extracts **and every 944-trained model is invalidated** — the inputs changed meaning, not just their values | ~free (one wider walk; +2.8 % columns at 576²) |
| **B. Two plane sets** | v1 blocks over padded planes, v2 blocks over unpadded | `f0..371` buffered-exact, `f372..943` **unchanged** — no 944 table or model touched | **the expensive one**: a second pyramid + H/V blur chain, directly opposing the perf mandate. Not yet measured; the H-blur families are 25–44 % of the walk (§2), so a naive second set is a large regression |
| **C. Stop v1 pooling over pad columns** | buffered v1 changes to match the fold | `f0..371` correct-by-construction everywhere, **nothing in 944 moves at all** — but **every stored 372 table, `v1_golden_bytes`, and the shipped metric's output change** | free |

**Option A is the one the directive's letter selects and it is the most
expensive by far** — it invalidates the 944 models, not just their tables. The
directive authorises re-extraction; it does not obviously authorise
re-training every 944-class model, which is what a 36× shift in 92 % of the v2
features amounts to. **This needs an explicit decision and is not a call this
lane should make silently**, which is exactly why it is written down here with
numbers instead of implemented.

Option C is the one that matches the word "correct", costs nothing in the
944 regime, and is the only one where the fold needs no change at all — its
price is v1's byte-stability, which is the golden-gate policy's territory.

Option B is the only one that satisfies both "372 equals buffered" and "944
untouched", and its cost is unmeasured. **If a decision is wanted quickly, the
cheapest next measurement is B's**: instrument a second plane set behind a
flag and price it on `extract_paths_bench`.

### 8.4 Status of the rest of the directive

* **`f0..371` correctness fix** — mechanism found, measured, gated
  (§7.1/§7.2); implementation blocked on the §8.3 decision plus the h = 93
  residual (§7.2).
* **New regime/era tag + annotations-registry marking of the shipped 944-era
  tables** (ext924/ext944 legs, tbig_924/944, r1b-pools944, svt/aom harvest
  features, kadis-924, eval instruments) — **not done**; it should follow the
  decision, since which tables become prior-era depends on which option ships
  (A invalidates all of them, B and C invalidate none).
* **Eval-root re-extraction equality on cid22val / kon504 / tid** — **not
  run**; needs the subset mode to exist.
* **"Really optimize with and without the 372 top half"** — the rem-ring
  shipped (§2: −8.29 % buffered, −5.57 % zeroed-944, −6.22 % live-944 in
  instructions) and the ranked lever list is executed and adjudicated (§3),
  including the re-ranking of lever #1 that this profile forces. **The
  profile-driven work beyond that list is not done**, and it should be
  re-based on the CORRECTED semantics rather than on today's — otherwise every
  bit-exactness gate it is measured against is pinned to bytes that are about
  to change.

---

## 9. Option C — measured, and it is a WIN not a cost

User decision: stop v1 pooling phantom columns; the fold's unpadded semantics
are the truth; buffered is fixed to match. Measured before implementing.

### 9.1 C makes fold == buffered BIT-EXACT, and dissolves the h=93 residual

The experiment switch `ZEN_C_NOPAD` makes `simd_padded_width` return the real
width, so no phantom columns exist. Re-running the existing parity gate:

| geometry | baseline (padded) | under C |
|---|---|---|
| 96×64, 64×300, 208×144 (tight) | BIT-EXACT | BIT-EXACT |
| 127×93 | max rel **1.739e-1** | **0.000e0** |
| 200×150 | max rel **8.155e-1** | **0.000e0** |

**Under C the fold and the buffered path agree bit-for-bit at every width
tested, h=93 included.** So the §7.2 h = 93 residual was an artifact of the
option-A pre-pad workaround — a property of padding the *input*, not of the
two walks. **Under C it does not exist**, and item 3 is resolved by deletion
rather than by a fix.

Full suite under C: **223 passed, 1 failed** — the one failure is
`v1_padded_width_divergence_is_column_padding`, this lane's own
characterisation gate, which asserts the *padded* behaviour by construction
and cannot hold once padding is gone. `v1_golden_bytes`, `size_invariance`,
`streaming_strips` and `v1_feature_width_pure_function` all pass under C.

### 9.2 …but the goldens pass for the wrong reason

`v1_golden_bytes` uses **64×64** fixtures, and `simd_padded_width(64) == 64`.
The golden set is entirely in the TIGHT class, so it is **structurally blind
to the defect C fixes** — it would have passed no matter how wrong the padded
class was. Any C rollout needs a golden fixture at a non-tight width
(e.g. 200×150 or 576×96); without one the golden gate gives false confidence
on exactly this axis.

### 9.3 The cost of exactness — it is negative

Ask: "what is the perf difference of padded vs exact above the must-pad
threshold?" Measured on the buffered v1-372 path (callgrind Ir, serial, v3
tier, minus the 27,549,000 harness constant):

| width | padded Ir | exact Ir | delta | % |
|---|---:|---:|---:|---:|
| 576 (pads → 592) | 335,626,055 | 305,345,180 | −30,280,875 | **−9.02 %** |
| **592 (TIGHT — control)** | 326,245,903 | 326,257,116 | +11,213 | **+0.00 %** |
| 1152 (pads → 1168) | 1,386,215,532 | 1,284,044,571 | −102,170,961 | **−7.37 %** |

**Exactness is 7–9 % CHEAPER at non-tight widths and free at tight ones.** The
tight-width control landing at +0.003 % is the measurement validating itself:
where there is nothing to exclude, C is exactly a no-op.

The reason is simple once stated: the phantom columns were not free to compute.
`simd_padded_width` adds 16 columns to every width ≥ 512 that is an even
multiple of 16 (the anti-alias stride trick), and every blur, every pool and
every downscale then ran over them. Removing them removes ~2.8 % of the
columns at 576 — and yields ~9 %, so the padded width was also landing on a
worse row-group tail than the exact one.

**This inverts one premise of the brief.** The instruction was that
lane-alignment padding of BUFFERS should stay and only pooling should change.
Measured, the alignment padding is not paying for itself on this walk: it
costs 7–9 % and buys correctness problems. The cheapest *and* most correct
implementation of C is simply not to pad. **Not flipped here** — the default
is untouched pending the era rollout; this is the number the decision needs.

### 9.4 Era discipline — what would change, and what would not

* **The fold's 944 outputs are UNCHANGED by construction under C.** The fold
  never padded; C only removes padding from the buffered path. This is
  assertable and should be asserted before rollout (a before/after
  bit-identity gate on the 944 regimes) — noting that the pre-pad blast-radius
  gate (§8.2) already proves the converse direction, that *adding* padding
  moves 92 % of the v2 block.
* **372 artifacts that become prior-era** — every table extracted through
  `compute_zensim_with_config` / `compute_zensim_with_ref_and_config` at a
  **non-tight** width. Tight-width rows are bit-identical under C and do not
  change era. Since `simd_padded_width` bumps every width ≥ 512 that is an
  even multiple of 16, and 576/1152/2304 are all in that class, the practical
  answer is "most of them" — but it is a per-row property of the width, not a
  per-table one, so the annotation should record the predicate rather than a
  list.
* **Not flipped, listed for the user**: the `eval_roots` default, and whether
  C ships as (a) no padding at all (fastest, simplest, measured above),
  (b) padded buffers with pool-width exclusion (matches the brief's letter;
  needs a `(width, stride)` split through the pooling kernels; strictly slower
  than (a) since the pad columns still get blurred), or (c) a config flag with
  the old behaviour retained for prior-era reproduction.

### 9.5 Fusion vs fission — the register-pressure steer, measured

Prompted by "multiple passes per row can be faster depending on register
spills; do NOT assume maximal fusion wins", the registered lever #1
(`box_blur_h_of_abs_diff` — fold the activity's abs-diff into the H-blur's
load sites so `act_raw` is never materialised) was implemented, gated
bit-exact, measured, and **REVERTED**:

| arm | two-pass (shipped) | fused | delta |
|---|---:|---:|---:|
| `fold944_full` | 562,449,826 | 568,313,440 | **+5,850,600 (+1.04 %)** |
| `fold372_only` | 276,789,555 | 282,338,738 | **+5,561,565 (+2.01 %)** |

Fusion LOSES. Post-rem-ring the H-blur gathers one strided column per x-step;
folding in the abs-diff makes it gather **two** planes, and the contiguous
`abs_diff_into` pass it removes is cheap per element by comparison. The
comparison is lane-width-fair (both arms are 8-wide at the profiled v3 tier),
so this attributes to fusion itself, not to the kernel's width. In production
it would be worse still: the two-pass path has hand-written 16-wide v4/v4x
variants, while a `magetypes`-generic fused kernel is 8-wide everywhere.
Kernel and its bit-exactness test were deleted rather than parked.

**Spill audit of the kernels this lane shipped** (`objdump`, xmm/ymm/zmm ↔
`(%rsp)` traffic inside the function body):

| kernel | spill stores | spill loads |
|---|---:|---:|
| `box_blur_h_inner_v4x` (rem-ring) | 1 | 2 |
| `fused_blur_h_ssim_inner_v4x` (rem-ring) | 4 | 2 |
| `fused_vblur_ssim_inner_v4x` (carries `store_sigma`) | 4 | **28** |

The rem-ring is spill-clean — the stack ring did not cost register pressure.
`fused_vblur_ssim` carries 28 spill reloads and is the standing
fission-vs-fusion candidate: it is the most-fused kernel in the walk and the
only one showing real spill traffic. Splitting it was NOT attempted here and
is the registered next experiment.

### 9.6 Where the 944-full → 944-zeroed gap actually lives

The priority steer asks for this gap crushed. Attributed (callgrind, 576²,
serial, v3; gap = **75,958,119 Ir, +16.55 %** over zeroed):

| kernel | zeroed | full | delta | of gap |
|---|---:|---:|---:|---:|
| `box_blur_h_inner` | 49,896,194 | 80,978,858 | +31,082,664 | **40.9 %** |
| `ssim_channel_inline_both` | 0 | 11,985,990 | +11,985,990 | 15.8 % |
| `edge_diff_channel_inline_both` | 0 | 9,589,905 | +9,589,905 | 12.6 % |
| `box_blur_v_copy` | 44,895,045 | 54,204,960 | +9,309,915 | 12.3 % |
| `build_inline_mse` | 0 | 4,134,045 | +4,134,045 | 5.4 % |
| `fused_vblur_ssim` (`store_sigma`) | 61,564,959 | 64,869,759 | +3,304,800 | 4.4 % |
| `abs_diff_into` | 2,941,191 | 6,143,526 | +3,202,335 | 4.2 % |
| memset + memcpy (scratch growth) | 13,791,991 | 17,076,945 | +3,284,954 | 4.3 % |

Two groups: the **activity chain** (H-blur + V-blur + abs-diff) is
**43.6 M = 57.4 %** of the gap, and the **pool math proper** (SSIM / edge /
MSE kernels) is 25.7 M = 33.8 %.

The activity chain looks close to irreducible at fixed semantics:
`blur(|src − blur_h(src)|)` is a blur of a non-linear function of a blur, so
the two passes cannot be algebraically merged; the fused-load form is measured
above and loses; and the ±overlap recompute per band is what makes the fold
v1-exact (v1 mirror-clamps the activity at its own strip edges), so it is
semantic, not waste. The remaining named levers are the `fused_vblur_ssim`
fission experiment (§9.5) and the inner-rows-only V-blur write, which was
priced at ≈0.4 % of the walk and rejected on value.

**So the honest answer to "which world are we in": the pool block costs
+16.55 % over zeroed and that is close to its intrinsic price.** With
`fold372_only` at **276,789,555 Ir = 0.49× of 944-full and 0.74× of today's
buffered v1-372**, a caller that only wants v1's 372 saves roughly half by
having a dedicated path — which is why the block-skipping mode was still worth
shipping even though 944-full could not be brought down to 944-zeroed.

---

## 10. Block-skipping, MT, and the final picture

### 10.1 Block-skipping (`v1_only`) — 53 % of the walk removed

A 372-only request had been paying for every v2-era block. `V2NewFeatureToggles
::v1_only` now skips dense-348, gradient, append, append2/BANDVIS, CSFW and
blockiness — **and their upstream work in phase A**, which is where most of the
win is: the four whole-window `box_blur_v_from_copy` sweeps producing the
V-blurred `mu1/mu2/ssq/s12` strips, plus the v2 activity chain. Nothing v1
needs reads any of it (`fold_v1_basic_bands` takes the H-blurred planes and
computes its own activity), so this is real compute removal.

| arm | Ir (576², serial, v3) | vs `buf_v1_372` | vs `fold944_full` |
|---|---:|---:|---:|
| `fold372_only` | 249,228,173 | **0.743×** | **0.466×** |
| `buf_v1_372` | 335,620,797 | 1.000× | — |
| `fold944_full` | 534,893,298 | 1.594× | 1.000× |

**The 372-only fold is 25.7 % cheaper in instructions than today's buffered
v1-372**, and block-skipping removes 53.4 % of the 944 walk. The naive
compute-944-then-project it replaces was 3.9–4.8× buffered in wall at 28T.

Pure compute-skipping: `folded_v1_only_matches_full_walk` asserts the emitted
slots are bit-identical to the same request with the v2 blocks on, over 5
geometries × 3 `V1PoolsMode` values × serial and rayon, and that the skipped
range is FINITE (never NaN from finalising an accumulator nothing wrote). One
subtlety the gate caught: the emitted vector must keep the caller's WIDTH and
REGIME — deriving them from the compute flags handed back a 720-wide vector for
a 944 request.

### 10.2 MT — the 3-thread ceiling, and lifting it

The fold parallelised only across its 3 channels, and a thread sweep showed
that is exactly where it stopped (576², wall, 40–60 iters/point):

| threads | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 944-full **before** | 17.50 | 10.50 | **7.75** | 8.50 | 8.75 | 7.75 | 9.25 | — |
| 944-off **before** | 14.25 | 9.75 | **6.25** | 8.00 | 7.00 | 7.75 | 7.50 | — |
| 944-full **after** | 17.17 | 10.50 | 8.67 | 7.67 | 7.33 | **6.67** | 7.00 | 7.33 |
| 944-off **after** | 13.83 | 8.67 | 6.83 | **5.50** | 5.83 | 5.83 | 6.33 | 7.17 |
| 372-only **after** | 10.00 | 6.17 | 6.00 | 5.17 | 4.33 | 4.17 | **3.67** | 5.33 |

Before: 2.26× at 3 threads and **negative returns past it** — rayon
oversubscription with no work left to hand out. Not a diffuse serial fraction;
a hard structural cap.

After adding band parallelism inside the channel (4 bands per strip at
`STRIP_ROWS` 128 / `V1_BAND_ROWS` 32): **944-full best 7.75 → 6.67 ms (−14 %,
2.57× at 8T), 944-off 6.25 → 5.50 (−12 %), 372-only 3.67 ms at 12T (2.72×)**,
and the past-3-threads regression is gone in all three modes.

**Bit-exactness is the load-bearing part**: each band accumulates into its own
zero-initialised `V1BasicSums` and the merge runs **sequentially in band
order**, so the f64 addition sequence is `((0 + b0) + b1) + …` — exactly the
in-place serial loop's. An unordered or tree reduction would not be equivalent;
f64 addition is not associative. Both paths use the same local-then-merge
shape, so serial and parallel are identical by construction.

**The first attempt measured a LOSS and the reason generalises**:
`map_init(FoldPoolScratch::default)` re-allocates ~580 KB of band buffers per
worker per strip per channel, which made 944-full *worse* at 3T (7.75 → 10.00).
One persistent scratch slot per band position fixed it. Parallelism that
allocates in the hot path is not parallelism.

Still short of buffered-class scaling (buffered reaches 4.6× at 28T). The
remaining cap is the **serial `StripPlaneProducer`** — the next axis, not
attempted here.

### 10.3 Peak RSS per mode — the low-memory hook

`/usr/bin/time -v`, 8 threads, one arm per process. Working set subtracts the
harness's two input images (7.96 MB at 1152², 31.85 MB at 2304²).

| arm | 1152² peak | 2304² peak | 1152²→2304² | 2304² working set | vs `buf_v1_372` |
|---|---:|---:|---:|---:|---:|
| `buf_v1_228` | 56.9 MB | 216.5 MB | 3.81× | 184.6 MB | 1.12× |
| `buf_v1_372` | 55.5 MB | 196.2 MB | 3.53× | 164.4 MB | 1.00× |
| `fold372_only` | 62.8 MB | 135.5 MB | **2.16×** | 103.6 MB | **0.63×** |
| `fold944_off` | 61.3 MB | 134.4 MB | **2.19×** | 102.5 MB | **0.62×** |
| `fold944_full` | 76.3 MB | 163.6 MB | **2.14×** | 131.8 MB | **0.80×** |

The shapes are the story: buffered holds whole-image pyramids and scales with
AREA (3.5–3.8× for 4× the pixels); the fold holds O(strip × width) rolling
planes and scales closer to WIDTH (2.1–2.2×). **At 2304² the fold's working set
is 0.62–0.80× buffered's, and the gap widens with size** — that is the
low-memory hook, and it is why the fold is the right shape for large images
even where its wall clock is not yet.

**The MT win was not free in memory**, as promised: `fold944_full`'s 2304²
working set went 114.6 → 131.8 MB (+15 %) because `pool_scratch` is now four
band-slot scratches per channel instead of one. Below ~1.5 MP the fold is still
the *heavier* path (62.8 vs 55.5 MB at 1152²); the crossover is real but it is
a large-image property, not a universal one.

### 10.4 Final paired numbers, 8 threads

`extract_paths_bench`, five arms interleaved in one process, 20 rounds each,
`RAYON_NUM_THREADS=8`. Box load 4–8 during this run (the quietest conditions
this lane got; still not certified clean).

| arm | 576² | vs base | 1152² | vs base |
|---|---:|---|---:|---|
| `buf_v1_228` | **2.2 ms** | base | **6.9 ms** | base |
| `buf_v1_372` | **3.0 ms** | +31.5 – +42.1 % | **9.3 ms** | +30.9 – +38.8 % |
| `fold372_only` | 6.6 ms | +176.8 – +217.3 % | 22.6 ms | +208.2 – +243.2 % |
| `fold944_off` | 9.4 ms | +297.3 – +349.0 % | 31.9 ms | +335.0 – +385.0 % |
| `fold944_full` | 9.4 ms | +301.9 – +349.9 % | 35.0 ms | +394.2 – +422.9 % |

Two results worth separating:

* **The 944-full → 944-zeroed marginal collapses under threads.** At 576² the
  two arms are **identical (9.4 vs 9.4 ms)**; at 1152² the gap is **+9.7 %**,
  against **+16.55 % in serial instructions**. Band parallelism absorbs the
  pool work, because the pool block is exactly the part that now has a second
  parallel axis. **This is the answer to "which world are we in": at ≥ 8
  threads 944-full is at or near 944-zeroed, so 944-full's overhead does NOT
  on its own justify a separate 372-only path.**
* **But `fold372_only` is still 30 % below `fold944_full`** (6.6 vs 9.4;
  22.6 vs 35.0), so the mode earns its keep for callers that genuinely only
  want v1's 372 — the dataset extractors, the GPU oracle, `zensim-target`,
  the published `zensim-regress`. It is one boolean on an existing struct with
  a bit-identity gate, not a third pipeline.

**The "372-only at or under buffered at matched thread counts" bar: MET at 1
thread, NOT met at 8.** Serial the 372-only fold is 0.743× buffered v1-372 in
instructions (and 10.0 vs ~12.8 ms wall); at 8 threads it is **2.2–2.4×**
buffered. The cause is not the fold's per-pixel work — it is that buffered
parallelises band-per-strip with a degree that grows with image height while
the fold now tops out at 3 channels × 4 bands, over a **serial
`StripPlaneProducer`**. Until the producer parallelises, buffered wins at high
thread counts on wall clock no matter how much per-pixel work the fold drops.
That is the single named blocker for retiring buffered on perf grounds.

---

## 11. Mode taxonomy collapsed: 944-full is the only product mode (2026-08-30)

User decision, superseding §10's three-mode framing: **944 with all pools live
is the ONLY product mode.** The zeroed regime and the `v1_only` block-skipping
boolean are not product paths.

* **`V2NewFeatureToggles::v1_only` is `#[doc(hidden)]`** and documented as
  test/bench instrumentation — the control arm for
  `folded_v1_only_matches_full_walk` and for pricing what the v2-era blocks
  cost inside the one product walk. **It could not be made `pub(crate)`**:
  the struct is constructed by external crates with `..Default::default()`,
  and functional record update requires every field to be visible, so a
  private field breaks every out-of-crate constructor — zenmetrics included,
  which is a scope fence. `#[doc(hidden)]` is as far as this goes without a
  builder-style redesign of the struct. **Listed for approval.**
* **`V1PoolsMode` stays fully public.** The fleet constructs it and the
  in-flight aom/svt harvests run at `foldapp2` / `carriers`; touching it would
  break their internal consistency. Canonicalising `zensim-foldapp2pools` as
  the datagen default is a registered fleet-lane task, not this lane's.
* The `fold372_only` bench arm is removed; `toggles_off` survives in
  `extract_paths_bench` **as the measurement control that prices the pool
  block**, not as a shippable configuration.

### 11.1 Row-parallel H-blur — MEASURED NEUTRAL, reverted

With 944-full as the only target, the next MT axis tried was splitting phase
A's `fused_blur_h_ssim` across row bands. It is provably bit-exact — a
horizontal box blur is an independent running-sum recurrence per row, and
`box_blur_h_ring_matches_regathered_reference` already proves a scalar
per-row reference matches the vector kernels bit-for-bit, so band boundaries
can only change which lane-width tier a row lands in, never a value. (The
VERTICAL blur has no such property: its recurrence walks the whole column, so
restarting it mid-plane re-inits the running sum from a different sequence of
adds. It can only be split by column, which needs a kernel signature change.)

Implemented, gated, measured, **reverted**:

| threads | 576² ON | 576² OFF | 1152² ON | 1152² OFF |
|---|---:|---:|---:|---:|
| 4 | 7.33 | 7.50 | 34.00 | 34.00 |
| 6 | 7.00 | 7.17 | 31.00 | 31.50 |
| 8 | 7.00 | 7.83 | 34.00 | 33.50 |
| 12 | 8.17 | 7.17 | 31.50 | 35.00 |

Every delta is inside the noise of this instrument. The reason is structural:
phase A already runs inside the 3-way channel fan-out and the fold hook is
already band-parallel, so a third nesting level of rayon adds fork/join
overhead against plane sizes that are already saturating the threads
available. **Complexity for no measured gain does not ship** — the wrapper and
its wiring were deleted, not left behind a flag.

That makes three levers this lane implemented and then rejected on
measurement (activity-fusion §9.5, `map_init` band scratch §10.2, row-parallel
blur here), against three that shipped (rem-ring, block-skipping,
band-parallel fold hook). The rejections are the measurements working.

### 11.2 Where 944-full's remaining MT ceiling is

944-full now reaches **2.40–2.45×** (576² 17.17 → 7.00 ms; 1152² 74.50 →
31.00 ms), against buffered's ~5.8× at 1152². The gap is NOT the producer,
which is only ~8 % of the walk by instruction count
(`srgb_to_positive_xyb_planar` 17.0 M + `downscale_2x_into` 15.5 M + memcpy
8.6 M of 535 M). It is two things the fold cannot currently split:

1. **`dense_block_kernel` is 23 % of the walk and gets 3-way parallelism
   only.** It accumulates row-by-row into a running f64 `DenseAccum`, so
   splitting rows into chunks and merging gives `((0+r0)+r1) + ((0+r2)+r3)`
   where the serial path gives `(((0+r0)+r1)+r2)+r3` — **not bit-exact**,
   f64 addition is not associative. The fold hook could be band-parallelised
   precisely because each band returns a self-contained value that is then
   merged in order; `dense` would need the same restructuring to qualify.
2. **Channel load imbalance.** `append`, BANDVIS and CSFW are Y-only, so the
   3-way fan-out is bounded by channel 1's time, not the mean. More threads do
   not help a fan-out whose critical path is one task.

So the honest next step for 944-full MT is **restructuring `dense_block_kernel`
to produce per-band self-contained accumulators** (the fold-hook pattern),
not parallelising the producer. Recorded here rather than attempted, because
it is a kernel rewrite with a bit-exactness gate and wants its own stage.
