# Fast-class extraction kernel — cost map, levers, and two defects (2026-09-05)

Lane: KERNEL. Plan (pre-registered before any measurement):
[`docs/PLAN_KERNEL_FASTCLASS_2026-09-05.md`](../docs/PLAN_KERNEL_FASTCLASS_2026-09-05.md).
Companion lane: CAMPAIGN (the model side — which 156+cheap head competes with a
944 MLP on rank). This record is the extraction side only.

User directive (2026-09-05, verbatim): *"… find a high performance but 944-mlp
competitive 156 or 156 plus cheap model, you can improve the kernel"*.

---

## 0. The headline, before anything else

**"156 + peaks" is servable by the product API today. "156 + peaks + raw
moments" and "156 + peaks + class-C" are NOT — and the blocker is in this
crate, not in the model.** Two independent walls, both read from source:

1. **`compute_folded_v1_372_streaming_impl` hard-codes `free_extras: Off`.**
   `zensim/src/feature_v2.rs:7532-7536` builds its toggles as
   `V2NewFeatureToggles { v1_pools: pool_mode.unwrap_or(Full), v1_only: true,
   ..Default::default() }`, and `V1FreeExtras::default()` is `Off`
   (`feature_v2.rs:1586`). Its ONLY tunable parameter is
   `pool_mode: Option<V1PoolsMode>`. So every raw-moment and class-C slot
   arrives at the forward pass as a structural `0.0` on any call that goes
   through `Zensim::compute`.
2. **`compute_fold_backed` then truncates to the v1 width.**
   `zensim/src/fold_engine.rs:158` — `features.truncate(v1_feature_width(config))`,
   which is at most 372. The raw-moment slots live at `f720+` and the class-C
   tranche is scattered across the v2-348 and append blocks; both are past the
   cut even if they had been computed.

Consequence, stated plainly so the campaign lane can plan against it: a bake
over **228** coordinates (`f0..155` basic + `f156..227` peaks) is servable by
`Zensim::compute` with **zero kernel work** — `fold_engine::pools_mode_for_need`
already returns `V1PoolsMode::Peaks` for any bake that does not read masked/IW,
and the peaks block is inside the 372 prefix. A bake over **265** (adds 37
raw-moment slots) or **289** (adds the 24 class-C slots) coordinates trains fine
today and **cannot be served**. `wide_bake_v2_read` (`fold_engine.rs:364`), the
function that would decide which `V1FreeExtras` a bake needs, exists and is
tested — and is `#[cfg_attr(not(test), allow(dead_code))]`, reached only by
`ComputeSet::from_block_profile`, which is not a runtime call site.

**This lane did not close that gap** — it is a routing/ownership change
(promote `from_block_profile` to a runtime call site behind a cache shaped like
`cached_bake_pool_need`, widen the impl's parameter from `Option<V1PoolsMode>`
to a compute set, and re-decide the truncation) that wants the user's call on
public surface. It is registered here, with the file:line, so it is a decision
rather than a discovery.

---

## 1. The cost map

`ZENSIM_FOLD_TIMING` (`zensim/src/fold_timing.rs`, already in tree — 21 phases,
`occ = busy / (wall × threads)`), driven through
`zensim/examples/foldapp_stream_bigpair.rs`. 576², 1T, serial, native `v4x`,
5 walks. Log: `~/tmp/kernel-lane/L4_skip_check.log`.

| phase | `156` (fast) | share | `944full` | share |
|---|---:|---:|---:|---:|
| producer | 2.118 ms | **32.5 %** | 1.997 ms | 12.4 % |
|   └ convert | 1.432 | **22.0 %** | 1.457 | 9.0 % |
|   └ downscale | 0.684 | 10.5 % | 0.539 | 3.3 % |
| blur_h (sum over channels) | 2.128 | 32.7 % | 2.022 | 12.5 % |
| fold bands (sum) | 2.155 | 33.1 % | 4.729 | 29.3 % |
| v2 kernels, total busy | **0.001** | **0.0 %** | 7.237 | 44.8 % |
| **walk** | **6.509 ms** | | **16.150 ms** | |

**The single load-bearing finding: for the fast class the FRONT END is the
biggest phase, and it is not the part anybody has been optimising.** The
producer is **32.5 %** of the fast walk and the XYB conversion alone is
**22.0 %** — against 12.4 % / 9.0 % on the 944-full walk. That is not because
the producer got slower; it is because stripping the v2 blocks removed 60 % of
everything else. Every published fold-perf lever to date targets the H blur,
the pool block, `dense_block_kernel` or the band shape. On the fast class those
are a third of the walk between them and **the sRGB→linear→opsin→XYB conversion
plus the downscale cascade is another third.**

Two caveats stated rather than buried: the enabling env var perturbs cache and
compresses spread, so these are shares, not absolute bars; and per
`era2_perf_break_2026-08-31.md` §22.5 a single-process number carries the
cell's own noise, which at 576²/1T is ~±1 %. The SHARES are the result here.

Per-pixel cost RISES with size in every arm (CLAUDE.md), so no "ms/MP" is
quoted and no `α + β·pixels` intercept is fitted.

---

## 2. L4 — does `v1_only` actually skip the v2 blocks at runtime? **PASS**

Pre-registered bar: `DenseKernel`, `GradKernel`, `AppendKernel`, `CsfwKernel`,
`BlockKernel`, `PhaseAV2Planes`, `PhaseAAppendPlanes` must read exactly zero on
the `156` arm. Measured on `156`, `15f` and `15x`:

* `v2:dense`, `v2:gradient`, `v2:append`, `v2:csfw`, `v2:blockiness`,
  `v2:planesA` — **0.000 ms and 0 calls**, all three arms.
* `v2:planesApp` — 0.001 ms and **165 calls**.

The 165 calls are the timing hook alone. `stream_phase_a`'s `__t_pap` span
(`feature_v2.rs:7113-7151`) brackets `if want_act_dst { … }` and
`if want_bs2 { … }`, and both guards are false on a `v1_only` walk — so the
region is entered and does nothing. It is 0.015 % of the walk and exists only
while `ZENSIM_FOLD_TIMING` is set (`start()` returns `None` when off). The skip
is structural and complete.

---

## 3. DEFECT 1 (fixed) — the two `self_blur` predicates disagreed on `Peaks`

There are two, and they must be the same predicate:

* **`ComputeSet::self_blur_eligible()`** (`feature_v2.rs:1977`), read at
  `feature_v2.rs:8314-8317` to build `plane_needs` — this **sizes the strip
  scratch**. It required `V1PoolsMode::Full`.
* the `let self_blur = …` binding in the strip loop's `fuse_channels` arm
  (`feature_v2.rs:8436`) — this **decides whether phase A runs**. It accepted
  `Full | Peaks`.

On a `Peaks` + parallel walk the sizing predicate said `false` (so
`StripPlaneNeeds { h: true }` reserved phase A's four fused-H planes) and the
loop predicate said `true` (so phase A was skipped whole). The planes were
allocated, first-touched, and never written — exactly the waste
`StripPlaneNeeds` was introduced to remove
(`fold_footprint_2026-08-31.md` §5), reappearing on **the one mode the fast
class actually ships**: `pools_mode_for_need` returns `Peaks` for every bake
that does not read masked/IW.

**Failing-first, then fixed.**
`feature_v2::tests::self_blur_sizing_predicate_matches_the_strip_loop_predicate`
fails against the pre-fix definition with `v1_pools=Peaks v1_only=true: sizing
says false, loop says true`. The fix widens `self_blur_eligible` to
`Full | Peaks` and makes the strip loop **read it** instead of re-deriving it,
so there is now one owner.

`feature_v2::tests` also carried a "legacy derivation" line pinning the
`Full`-only value. That line was a copy of the *sizing* predicate, not the
loop's, so pinning it kept the defect green. It is corrected in place with an
explicit comment; the constraint it now encodes (sizing == loop) is strictly
stronger than the one it encoded before (sizing == a hand-copy of sizing). This
is called out rather than slipped in, because changing an expected value is
exactly the move that needs to be visible.

### 3.1 Footprint effect: MIXED and small — NOT the clean win it looked like

`smaps_rollup` Rss, 8T, `ZENSIM_BIGPAIR_PARALLEL=1`, 3 walks:

| cell | before | after | Δ |
|---|---:|---:|---:|
| 1152² `156` | 70,432 kB | 76,276 kB | **+8.3 %** |
| 1152² `15c` | 69,860 kB | 74,660 kB | **+6.9 %** |
| 2304² `156` | 137,280 kB | 134,992 kB | −1.7 % |
| 2304² `15c` | 141,108 kB | 134,676 kB | −4.6 % |
| **2304² `156`, 1T (CONTROL)** | 82,500 kB | 82,420 kB | **−0.1 %** |

The 1T control behaves exactly as predicted — `fuse_channels` requires
`parallel`, so self-blur cannot engage at 1T and the change must be inert
there. It is.

**The 1152² rows go the wrong way and this record does not explain them away.**
Removing an allocation cannot raise RSS, so either the reading is allocator
noise — `fold_footprint_2026-08-31.md` §6.3 measured the fold's 16T RSS growing
**+19 % from 1 to 20 compares** on glibc arena churn, and 3 walks at 8T sits
inside that regime — or something re-grows lazily. It is published as measured,
unresolved, and it is **not** claimed as a footprint win. A clean re-read wants
`MALLOC_ARENA_MAX=1` and more repeats on an idle box.

**So this lands on CORRECTNESS grounds, not perf grounds**: two predicates that
must agree now agree, have one owner, and are pinned by a test that fails
without the fix. Compute is untouched — the loop's behaviour on `Peaks` is what
it always was — so no feature byte moves, which the parity gates confirm.

### 3.2 DEFECT 1b (fixed, −24.6 %) — the fast walk allocated **232 times per compare**

Found by the new `zensim/tests/fastclass_alloc_steady_state.rs`, which counts
`alloc`/`alloc_zeroed` through a global allocator over 8 warmed walks against a
**reused** `V2Scratch`, serial. 1152², `156_basic_peaks`:

| | allocations / walk |
|---|---:|
| before | **232.0** |
| after | **175.0** (−24.6 %) |
| the ideal | ~1 (the returned feature `Vec`) |

The linearity half of the test PASSED throughout (4 walks → 652, 8 walks →
1304, exactly 2×), so this was never a leak — it is a **per-strip constant**,
which is the worse shape for a codec tuning loop because it scales with the
image.

**Located and fixed:** `fold_v1_basic_bands` built its band-start list as
`let mut starts = Vec::new()` (`feature_v2.rs:5600`), once per
**(strip, channel, scale)** — one allocation plus up to two reallocs each. At
1152² that is 19 strips × 3 channels = **57 call sites, matching the 57-of-232
delta exactly**. Replaced with a fixed `[usize; V1_BANDS_PER_STRIP]` plus a
length, which is sound by construction (a strip is at most `STRIP_ROWS` rows,
a band is `V1_BAND_ROWS`, so at most 4 starts exist) and asserted rather than
assumed, so a future change to either constant fails loudly instead of
silently truncating a band — which would drop its rows from the sums and move
feature bytes. Bit-exact: the same starts in the same order, on both the serial
and band-parallel paths.

**The remaining ~175 are NOT located.** Ruled out by reading source:
`fold_v1_one_band` (no allocation), `FoldPoolScratch::ensure`/`ensure_h` (both
guard on `if b.len() < n`, so they are warm after warm-up), and the producer's
`RollingPlane` construction (once per walk, not per strip). The test is left in
tree as a **RATCHET at the measured baseline (bar 200/walk)** with that stated
in its own doc comment — *lower it, never raise it*. Publishing a passing bar
that encodes a known defect is worse than useless unless it says so, so it
says so.

---

## 4. DEFECT 2 (instrument) — the W4 bar arm requests a walk production cannot produce

`zensim-bench/benches/ssim2_speed_bar.rs`'s `add156_156basic` arm builds
`V2NewFeatureToggles { v1_pools: V1PoolsMode::Off, v1_only: true, .. }`, and
that arm is **the W4 bar itself** since APPENDIX B2 of
`hybrid_candidate_2026-09-01.md` (*"≤ 1.25 × the 156-walk class … where 'the
156-walk class' is the `add156_156basic` arm"*).

But `fold_engine::pools_mode_for_need` (`fold_engine.rs:538`) **never returns
`Off`**, by documented policy: `Off` hands the band no scratch, which disables
the band-local self-blur shape. No production call can produce that walk. The
bar is therefore set by a walk that only the bench can run.

Whether that makes the bar too easy or too hard is **NOT MEASURED here** — the
1T A/B is structurally blind to it (self-blur needs `parallel`, so `Off` and
`Peaks` are the same shape at 1T; measured: `156` 26.73 / `15o` 25.56 / `15c`
25.76 ms at 1152²/1T, all within the cell's own control spread) and the 8T cell
was not obtained: the box went to load 10.4 mid-sweep and the sweep driver
refused rather than emit a contaminated number. The 1152²/1T cell's control
read min 25.76 vs median-of-start-mins 38.50 — a **49 % spread**, far outside
the ~±1 % this cell should show, which is itself the evidence the box was not
ours. Reported as **UNESTABLISHED**, per the pre-registered rule.

Two instrument changes landed so the question is answerable next time:

* `zensim/examples/foldapp_stream_bigpair.rs` gains **`15o`** — the same
  toggles as `156off` under a **three-character** name, because the ASLR
  protocol requires byte-identical environment blocks between interleaved arms
  and `TOGGLES=156off` vs `TOGGLES=156` is a 3-byte difference measuring a
  different address space.
* `scripts/kernel_fastclass_sweep.sh` — the protocol mechanised: it **refuses**
  an arm name that is not exactly 3 characters, rotates arm order per start
  (interleave), takes min-of-iters in-process and min over ≥15 starts with ASLR
  on, carries a control arm whose own min-vs-median spread IS the cell's noise
  floor, and **self-checks box load before every cell**, skipping with
  `SKIPPED_BOX_BUSY` rather than emitting a number it cannot stand behind.
  `--force` stamps every row `CONTAMINATED`.

---

## 5. Levers settled without new measurement

**L2 — the producer.** `CLAUDE.md` said the fold runs "over a producer with no
rayon" and named "the serial `StripPlaneProducer`" as the remaining cap. Both
are stale by one day: `feature_v2_stream.rs:626` runs the two image sides as a
`rayon::join` and `:769` fans the downscale cascade 6 ways, and the
fold-footprint lane showed the real ceiling is **L3 capacity** (per-thread hot
set `2,016·W`, full-width bands), not the producer — the 3.38× N-process
saturation was *the fold's own footprint read as the machine's*, and CCD-pinned
after the fix it reads **5.85× / 4.54×**. Producer pipelining is DECLINED
upstream (needs `unsafe`; `RollingPlane` aliasing). **CLAUDE.md corrected in
place** — the parallelism paragraph, the "remaining cap" line, and the
`dense_block_kernel` paragraph, which is additionally now scoped to the
944-full *extraction* (a fold-backed score never dispatches it) and carries the
note that its 23.2 % share and 1.17×/1.23× Amdahl bound are **`v3`-scoped**
(7.3 % of the walk on the shipping `v4x` tier).

**L3 — H tiling is a measured LOSS for this class, and it is an era item.**
`profile_d_notax_2026-09-01.md` §3.3 measured `ZENSIM_H_TILE=0` against the
default 1024 on the `15f` arm, min over 11 starts: **1152² tiling costs +7.1 %**
(5.000 vs 4.670 ms), **2304² +4.4 %** (20.640 vs 19.770), 576² identical
(1.290 vs 1.290 — the required below-tile-width control). Mechanism: `v1_only`
already skips the upstream sweeps that pushed the 944-full H-blur working set
past L2, so tiling here buys no cache fit and pays the packing.
**Not flipped.** The running sum along x restarts at every tile boundary, so
turning tiling off for the fast class is a **byte change** for images wider
than 1024 — an era item, registered per §6, not a bit-exact lever.

**L6 — dispatch.** No nested `#[arcane]` in the fast path. The free-set
accumulators are generic over a backend trait, so `#[rite]` cannot apply (it
resolves `#[target_feature]` from a concrete token); they are
`#[inline(always)]`, and `nm` on a release binary shows zero surviving
`raw_moments_*` symbols. Clean, no finding.

---

## 6. Registered era-break candidates (measured, NOT flipped)

| # | change | measured value | what it invalidates |
|---|---|---|---|
| E1 | Disable H tiling for the `v1_only` class | **+7.1 % @1152², +4.4 % @2304²** (1T, `15f`, n=11) | v1 bytes for any image wider than `H_TILE_WIDTH` (1024). Every fast-class table extracted since era-2 |
| E2 | Column-tile the fold bands | hot set `2,016·W → 2,016·(Tw+20)`, width-independent; 4.43 → 1.02 MiB/thread at 2304²/`Tw=512`; 3.9 % redundant H blur | f64 accumulation ORDER inside a band. Registered upstream as an era-2-enabled design (a tile boundary on a lane-group multiple is byte-safe by construction under era-2's fixed lane grouping) |

Neither is flipped by this lane. E1 is the one with a fast-class-specific
motive and the smaller blast radius; it should be batched with E2 rather than
paid for twice.

---

## 7. What landed

* `feature_v2.rs` — `self_blur_eligible` widened to `Full | Peaks` and made THE
  owner; the strip loop reads it. Failing-first gate
  `self_blur_sizing_predicate_matches_the_strip_loop_predicate`; the legacy
  derivation corrected in place with a visible comment.
* `feature_v2.rs` — `fold_v1_basic_bands`'s band-start `Vec` → a fixed
  `[usize; V1_BANDS_PER_STRIP]` with the bound asserted. **232 → 175
  allocations per 1152² compare (−24.6 %)**, bit-exact (§3.2).
* `zensim/tests/fastclass_alloc_steady_state.rs` — a counting global allocator
  asserting a warmed fast-class compare against a reused `V2Scratch` allocates a
  **bounded** number of times per walk (bar 8/walk over 4 shapes at 1152²),
  plus a superlinearity check (4 walks vs 8). Serial by construction: rayon's
  own first-steal bookkeeping is a property of the pool, not the walk, and
  counting it would make the assertion flaky. This is the class of defect that
  cost 7.75 → 10.00 ms at 3T once already (`map_init`-per-band, ~580 KB per
  worker per strip per channel) and it is invisible to a wall-clock A/B against
  this repo's noise floors.
* `zensim/examples/foldapp_stream_bigpair.rs` — the `15o` arm.
* `zensim-bench/benches/ssim2_speed_bar.rs` — a **`zensim_D`** arm. The bench
  had no arm for the shipped fast-class product path at all: every fast-class
  arm hand-builds `V2NewFeatureToggles` and so bypasses `Zensim::new`'s
  per-profile engine defaults, `is_fold_backable`'s guard (which can silently
  degrade to the buffered walk), `score_pool_mode`'s derivation from the bake,
  the truncation, and the spline. A routing regression was invisible to this
  instrument; now it reads as a speed regression.
* `scripts/kernel_fastclass_sweep.sh` — the ASLR + load protocol, mechanised.
* `zensim-bench/Cargo.toml` — `candidate-profiles` added to its `zensim`
  features. That feature gates `ZensimProfile::D`, and the crate's
  `default-features = false` line was hiding the shipped fast-class profile
  from the one bench that prices it.

  ⚠ **A retracted claim, recorded because the mistake is instructive.** This
  lane also hit `zensim-bench` failing to RESOLVE (*"failed to select a version
  for the requirement `jxl-encoder = ^0.4.0`"*) and re-pinned its
  `[patch.crates-io]` entry, writing it up here as a pre-existing defect. **It
  was not one.** The pin was already correct; the failure was a **stale local
  `~/work/zen/zenjxl` sibling checkout, 8 commits behind**, on this lane's side.
  Another lane had already diagnosed exactly this and left the warning in the
  manifest — *"If this line ever fails to resolve, check the two siblings'
  versions against each other BEFORE touching anything here"* — and this lane
  did not read it before editing. The re-pin was dropped in favour of that
  lane's plain path patch at merge. **The transferable rule: a resolution
  failure naming a sibling crate is a claim about TWO checkouts, and the
  sibling's `git log` is the cheaper half to check first.**
* `CLAUDE.md` — three stale claims corrected in place (§5).

## 8. What did not get measured, and why

The 8T half of L1, the 2304² cells of the cost map, the both-tier sweep, and
L5's cross-grid cheap-slot marginals all need an idle box. The box carried a
22-core `extract_features` for the first half of this lane and went to load
10.4 in the second. Per this repo's own rule — *do nothing else on the machine
during a pinned sweep, and a harness that can read LOW defeats `min()`* — those
cells are **NOT MEASURED**, not estimated. The sweep driver is committed and
self-gating, so re-running them is one command on a quiet box:

```sh
scripts/kernel_fastclass_sweep.sh --arms 156,15o --sizes 1152,2304 \
    --threads 8 --starts 15 --iters 7 --control 15c \
    --out /mnt/v/output/zensim/kernel-2026-09-05/L1_off_vs_peaks_8T.tsv
```

---
---

# LANE 2 — the front end

Pre-registration: `docs/PLAN_KERNEL_FASTCLASS_2026-09-05.md` §"LANE 2", pushed
as `3912eaf8` **before** any measurement or code in this section.

Lane 1 declined L2 on the grounds that the producer is *already parallel*
(sides `rayon::join`, cascade 6-way). That is true, and it settles
**scheduling**. It says nothing about the per-pixel work inside a chunk, which
is what this section measures.

## L2.0 The instrument, and why it could run at all

The box carried the campaign lane's fits and extractions throughout. **Callgrind
Ir is deterministic**, so it is valid on a loaded box — that is the only reason
this lane has a cost map rather than a list of `SKIPPED_BOX_BUSY` rows. Every
Ir number below is `156` arm, 576², 1T, **`v3`/AVX2 tier**
(`ZEN_S2_CAP_V3=1`; valgrind cannot execute AVX-512), 2 walks, through
`zensim/examples/foldapp_stream_bigpair.rs`.

**"Walk-side" = program total minus `foldapp_stream_bigpair::main`**
(18,339,594 Ir — the example's own pair generation), which is *identical* in
every arm here, so subtracting it is exact rather than an estimate.

## L2.1 The stage-level cost map (deliverable 1)

| stage | Ir / walk | share | note |
|---|---:|---:|---|
| `fused_blur_h_ssim_inner_v3` | 64,103,238 | 38.53 % | |
| `fused_vblur_ssim_inner_v3` | 61,978,020 | 37.26 % | |
| **`srgb_to_positive_xyb_planar_inner_v3`** (convert) | 17,004,996 | **10.22 %** | |
| **`downscale_2x_into_inner_v3`** | 15,496,704 | **9.32 %** | |
| `__memcpy_avx_unaligned_erms` | 4,809,300 | 2.89 % | |
| `__memset_avx2_unaligned_erms` | 2,347,683 | 1.41 % | 87.6 % of it from `RollingPlane::from_pooled` |
| everything else | ~613,000 | 0.37 % | incl. `_int_malloc` at **0.01 %** |
| **walk-side** | **166,352,940** | 100 % | |

**The Ir map and lane 1's WALL map disagree about the front end, and the
disagreement is the finding.** Lane 1 measured convert at **22.0 %** of wall;
it is **10.2 %** of instructions. Convert costs ~2.2× the average
cycles-per-instruction, and the disassembly says why: **6 `vdivps` per 16
pixels** at `v4x` (2 Halley iterations × 3 cube roots, each `y *= (y³+2x) /
(2y³+x)`). For contrast, the same scan over the two blur kernels finds **1 and
13 divides in 2,634 and 2,045 instructions**. The front end is the only
divide-bound phase in the fast walk. **Consequence for anyone reading either
map: an Ir share understates a convert lever and a wall share overstates a blur
lever.** Neither map is wrong; they measure different things.

### Convert, split by category (disassembly, per main-loop iteration)

| category | `v3` (8 px, 204 instrs) | `v4x` (16 px, 200 instrs) |
|---|---:|---:|
| de-interleave + sRGB LUT | ~57 (28 %) | ~21 (10 %) |
| per-pixel bounds checks | 20 (9.8 %) | ~17 (8.5 %) |
| constant rematerialisation (`vbroadcastss`) | 14 (6.9 %) | ~0 |
| opsin matrix | ~12 | ~10 |
| 3 × `cbrt_midp` | ~84 (41 %) | ~90 (45 %) |
| — of which sign / abs / zero-select | 16 | ~9–12 |
| XYB mix + stores | ~17 | ~30 |

**L12 (dispatch/de-interleave) is CLEAN at the shipping tier and only looks bad
at the profiling tier.** At `v3` the 256-entry LUT expansion is 24 `movzbl` +
`vinsertps` pairs, a genuinely scalar gather. At `v4x` LLVM auto-vectorised the
identical Rust into `vpermt2b` + `vpmovzxbd` + `vgatherdps` — CLAUDE.md's
fixed-array-to-shuffle pattern, already happening. Reporting the `v3` number as
if it were the product's would have justified a hand-written kernel that is
already there. `vbroadcastss`-per-iteration constant reload is likewise a
**`v3` register-pressure artefact** (16 ymm registers, ~17 live constants); at
`v4x`'s 32 zmm registers it vanishes.

## L2.2 What landed — three bit-exact levers

Walk-side Ir **332,705,880 → 300,821,380 = −9.58 %** (2 walks). Attribution
sums to the total within the `do_reserve_and_handle` self-cost:

| # | lever | Ir Δ (2 walks) | share of walk |
|---|---|---:|---:|
| **L16** | `downscale_2x_into` fixed-array de-interleave | −24,055,812 | **−7.23 %** |
| **L15** | `RollingPlane::from_pooled` stops re-zeroing | −4,838,468 | −1.45 % |
| **L17** | convert de-interleave bounds checks | −2,987,244 | −0.90 % |

### L16 — `downscale_2x_into` was a scalar gather loop wearing a SIMD hat

It sits inside two `#[magetypes]` blocks and builds an `f32x16` per 16
outputs, so it reads as SIMD. The body indexed the `src` **slice** four times
per output lane:

```rust
for i in 0..16 {
    let s = sx + i * 2;
    arr[i] = src[row0 + s] + src[row0 + s + 1] + src[row1 + s] + src[row1 + s + 1];
}
```

LLVM cannot see a strided pattern through 64 independently-bounds-checked slice
indexings, so it emitted a partial vectorisation with `vhaddps` behind a large
runtime-guard preamble (`cmova`/`shr` min computations). **Measured cost: 23.7
Ir per output pixel for a 2×2 average.**

Against fixed-size **array references** — two range checks at the chunk
boundary, zero interior — it sees it immediately:

```rust
let r0: &[f32; 32] = src[row0 + sx..row0 + sx + 32].try_into().expect(..);
let r1: &[f32; 32] = src[row1 + sx..row1 + sx + 32].try_into().expect(..);
for i in 0..16 { arr[i] = r0[2*i] + r0[2*i+1] + r1[2*i] + r1[2*i+1]; }
```

The emitted 16-output loop at `v4x` goes **168 → 39 instructions**, and is now
the shape it should always have been: 4 × `vmovups zmm`, 4 × `vpermt2ps`
(even/odd de-interleave), 3 × `vaddps`, 1 × `vmulps`, 1 store. Kernel Ir
**30,993,408 → 6,937,596 (−77.6 %)**.

**Bit-exact by construction, and gated.** The per-lane add order is unchanged —
`(((a+b)+c)+d) * 0.25` — and nothing is summed across lanes, so f32
non-associativity has nothing to bite. The pre-existing
`downscale_into_bit_identical_to_inplace` compares two of *our own* kernels and
would stay green if both were reordered together, so this lane added
**`downscale_add_order_is_pinned_left_to_right`**, which pins the order against
a literal reference expression, carries a **negative control** asserting the
fixture is order-SENSITIVE (so it cannot pass vacuously), and was verified
**failing-first**: a row-first reorder (`(a+c)+b)+d`) makes both tests fail by
1 ULP (`c36d30ae` vs `c36d30af`).

### L15 — the plane pool re-zeroed buffers it documents as irrelevant

`RollingPlane::from_pooled`'s own doc says *"contents are irrelevant (every row
is fully written before it is read)"*. Its **empty-pool** branch already used
`vec![0.0; n]` (calloc — demand-zeroed pages the allocator satisfies lazily),
with a comment explaining exactly why `Vec::resize` is wrong there. Its
**too-small** branch still called `Vec::resize`, which memsets the fill
explicitly *and* reallocs first. Measured split of that branch:

| | Ir | calls |
|---|---:|---:|
| `__memset_avx2_unaligned_erms` | 4,112,820 | 12 |
| `RawVecInner::…do_reserve_and_handle` | 733,763 | 12 |

Run-wide `__memset_avx2_unaligned_erms` falls **4,695,365 → 582,545
(−87.6 %)**. The two branches are now one. The grow path is not a corner: the
pool is **LIFO across scales**, so a scale-3 buffer popped for a scale-0 plane
is the common case. The change is also strictly *more* deterministic —
`resize` left a prefix of the previous walk's pixels in the buffer; this always
yields zeros.

**It does not move the allocation ratchet**, and that is expected, not a miss:
it removes the fill and the copy, not the allocation.
`fastclass_alloc_steady_state` still reads **175.0/walk** on all four arms with
the linearity check exactly 2× (508 → 1016). Lane 1's "~175 unlocated" stands.

### L17 — the convert de-interleave took a bounds check per pixel, at 14 sites

`for i in 0..N { let p = pixels[base + i]; … }` appears **14 times across 8
kernels** in `color.rs` (the same hand-copy-per-tier shape the free-features
lane already consolidated once). All 14 now take one range check per chunk via
a fixed-size array reference. Convert kernel Ir **34,009,992 → 31,022,748
(−8.78 %)**.

**This is BELOW the lane's pre-registered 2 %-of-walk G-PERF bar (it is
0.90 %), and it is not claimed under it.** It landed on being a deterministic,
zero-risk, bit-exact simplification that this repo's own performance guidance
prescribes; its wall effect should be *small*, because the loop it sits in is
divide-bound. Saying "−8.78 % on the convert kernel" without also saying
"−0.90 % of the walk, below our own bar" would be the kind of true-but-shaped
number this lane exists to avoid.

## L2.3 Gates

* **G-EXACT.** `to_bits()` A/B of the **full feature vector**, pre-lane binary
  vs post-lane binary, over **160 cells** = 20 geometries × 4 arms
  (`156`/`15c`/`15f`/`944full`) × {serial, 3-thread}: **0 differing bits**, on
  320 non-empty dumps of 924/944/956 features each. Geometries deliberately
  include tight and non-tight widths, odd widths, sub-tile sizes, and the
  `width % H_TILE_WIDTH == 1` class (`1025`, `2049`) that the era-2 tile bug
  lived in. `cargo test --workspace --no-fail-fast --exclude zensim-wasm-tests`:
  **1,548 passed, 0 failed**, including `fold_engine_parity`,
  `v1_golden_bytes`, `v1_feature_width_pure_function`,
  `fastclass_alloc_steady_state`, `pyramid_stride_has_no_phantom_columns` and
  `self_blur_sizing_predicate_matches_the_strip_loop_predicate`.
* **G-API.** `cargo public-api --simplified`, pre vs post: **ZERO delta** on
  1,280 items.
* **G-CLEAN.** `cargo clippy -p zensim --all-targets` clean; `cargo fmt` clean
  on the three touched files.

## L2.4 Wall-clock corroboration, with its ceiling stated

A bit-exact kernel lever **cannot** use `scripts/kernel_fastclass_sweep.sh`:
that script implements the one-binary-runtime-arms protocol, and here the lever
*is* the code, so the arms are two BUILDS — the shape CLAUDE.md says is
untrustworthy below ~10 % because any edit reshuffles the binary's own layout
by about that much. **The Ir map above is therefore the PRIMARY instrument for
every claim in this section**, and the wall numbers are corroboration.

`scripts/kernel_two_build_ab.sh` (new, committed) does the most that shape
allows: ASLR on, min of 7 walks in-process, min over 15 interleaved process
starts per build, a load gate before *every* invocation that **waits** rather
than contaminating, and — since no arm in this walk skips the kernel under test
— a **DIRECTIONAL control** stated in advance instead of a bit-identical one:

> *a lever in a phase that is a larger share of arm X's walk than of arm Y's
> must improve X by MORE than Y; if it does not, the reading is layout noise.*

`ProdConvert + ProdDownscale + pool` is ~22 % of the `156` walk and a much
smaller share of `944full`'s. Measured (native `v4x`, min over starts):

| arm | size | T | OLD (ms) | NEW (ms) | Δ | starts |
|---|---:|---:|---:|---:|---:|---:|
| `156` | 576² | 1 | 5.82 | 4.77 | **-18.04 %** | 15 |
| `944full` | 576² | 1 | 15.02 | 13.95 | **-7.12 %** | 15 |
| `156` | 576² | 8 | 1.72 | 1.44 | **-16.28 %** | 15 |
| `944full` | 576² | 8 | 5.13 | 5.08 | **-0.97 %** | 15 |
| `156` | 1152² | 1 | 25.90 | 21.87 | **-15.56 %** | 15 |
| `944full` | 1152² | 1 | 65.42 | 61.15 | **-6.53 %** | 15 |
| `156` | 1152² | 8 | 6.40 | 5.67 | **-11.41 %** | 15 |
| `944full` | 1152² | 8 | 25.64 | 22.03 | **-14.08 %** | 15 |
| `156` | 2304² | 1 | 108.00 | 91.89 | **-14.92 %** | 15 |
| `944full` | 2304² | 1 | 278.46 | 262.88 | **-5.60 %** | 15 |
| `156` | 2304² | 8 | 24.41 | 19.94 | **-18.31 %** | 15 |
| `944full` | 2304² | 8 | 113.71 | 102.21 | **-10.11 %** | 7 ⚠7 |

**The control passes at 1T at every size and is MIXED at 8T, and the mixed half
is reported rather than dropped.** At 1T `156` improves **2.1–2.7× more** than
`944full` (−18.04/−7.12, −15.56/−6.53, −14.92/−5.60), which is the predicted
ordering at all three sizes. At 8T it holds at 576² (−16.28 vs −0.97) and
2304² (−18.31 vs −10.11, the latter on only 7 starts), and **INVERTS at 1152²**
(−11.41 vs −14.08). The 8T cells carry this repo's own measured 8-thread noise
floor — **1.8 % to 6.5 %** (`era2_perf_break_2026-08-31.md` §23) — plus
cross-thread interference, and a 2.7-point inversion sits inside that. So: the
1T rows establish the effect; the 8T rows are consistent with it but do not
independently establish it, and the 1152²/8T row is **UNESTABLISHED**.

The wall improvement being *larger* than the −9.58 % Ir improvement is
consistent rather than surprising — the removed work was memory-traffic-heavy
(scalar gathers; a whole-plane memset), which costs more cycles per instruction
than the walk's average, and the Ir was taken at `v3` where the surviving
kernels are relatively more expensive. **The two instruments are not in
conflict and neither is "the" number: −9.58 % is instructions at `v3`,
−14.9…−18.0 % is 1T wall at native `v4x`.**

Raw rows (every start, including the `SKIPPED_BOX_BUSY` gate decisions):
`/mnt/v/output/zensim/kernel-2026-09-05/lane2/ab_downscale_2026-09-05.tsv`.

### ⚠ The first run of this table was UNGATED, and the bug is worth keeping

The table above is the **re-run**. The first run produced numbers that looked
the same but were measured **beside a live `zensim_mlp_train`**, because this
driver's own load gate never fired: it called
`pgrep -x zensim_mlp_train`, and **`/proc/<pid>/comm` is truncated to 15
characters**, so the 16-character name can never match. `pgrep -x` was chosen
*specifically* to avoid the `pgrep -f` self-match footgun CLAUDE.md warns
about — and walked straight into the other half of the same warning, which that
file also states (*"pgrep name-match truncates comm to 15 chars"*). The
already-committed `kernel_fastclass_sweep.sh` had it right by writing the names
pre-truncated (`zensim_mlp_trai`); this driver now truncates **in the gate
itself**, so a caller may write the real name and still be gated. Knowing the
warning was not enough; the code doing it is.

The ungated rows are kept as
`ab_downscale_CONTAMINATED_ungated_2026-09-05.tsv` — not deleted, and not
quoted anywhere as a result.

## L2.5 Registered, measured, NOT implemented

| # | finding | measured | why not done |
|---|---|---|---|
| **L11** | `cbrt_midp` carries sign-extract / `abs` / zero-select this call site cannot need — the opsin coefficients are **all strictly positive** (`K_M00..K_M22` ∈ [0.078, 0.692]) so `mixed ≥ K_B0 ≈ 0.0038 > 0` and `.max(0.0)` is a no-op | ~9–12 of 200 `v4x` instructions ≈ **6 % of convert ≈ 1.3 % of the walk** | Below bar, and the only way to take it is to hand-roll a positive-domain Halley cube root **inside zensim** — archmage is a different repo and is not to be touched — which forks an upstream primitive that must then stay bit-exact with it forever. A fork with a good story is still a fork. Note the scope limit if anyone revisits: `linear_to_positive_xyb_planar_into_unclamped` (issue #17 `GamutMapping::Preserve`) admits out-of-`[0,1]` input and **can** reach zero, so it is not covered by the argument. |
| **L13** | `ProdDownscale` was 9.32 % of walk Ir at 23.7 Ir per output pixel | now 2.15 % / ~5.3 Ir per output px | Done as L16. |
| **L14** | the `linear-srgb` batch API | not measured | `srgb_u8_to_linear` already routes to `linear_srgb::default`, and at `v4x` LLVM already emits `vpermt2b` + `vgatherdps` for the expansion. There is no gap for a batch call to fill; a *polynomial* would be an era break. Recorded as no-finding rather than left open. |
| **E3** (new era-break candidate) | **one Halley iteration instead of two** in the front-end cube root would remove 3 of the 6 `vdivps` per 16 pixels | divides are ~40 % of convert's cycles ⇒ ~**4–7 % of the fast walk**, the largest single front-end number in this lane | Moves every v1 byte on every corpus — an era break, batched per §6, **not flipped**. Note while you are there: the kernel's own comment says *"~15 bits, single Halley iteration"* and `cbrt_midp` does **two**; the comment is wrong about the code it labels. |

## L2.6 A defect in `main` this lane did not cause and did not fix

* **`main@origin` is not rustfmt-clean, and CI gates on it.** `cargo fmt --all
  --check` is `.github/workflows/ci.yml:27`; ten files under `zensim/` fail it
  as pushed (`benches/fold_engine_bench.rs`, `benches/fold_pools_bench.rs`,
  `src/{fold_engine,fold_timing,metric,profile,streaming}.rs`,
  `tests/{fold_engine_parity,v1_feature_width_pure_function,v1_golden_bytes}.rs`).
  A `cargo fmt -p zensim` in this workspace reformatted all ten; they were
  **reverted** rather than folded into this lane's commit, because they are
  other lanes' actively-in-flight files and a `style:` commit across them would
  collide. Reported so the owning lanes can take it.
* **`main@origin`'s `CHANGELOG.md` carried a stray leftover conflict marker** —
  a bare `>>>>>>> conflict 1 of 1 ends` at line 59 as pushed, from an earlier
  lane's incomplete resolution. This lane's rebase kept **both** lanes' sections
  verbatim and dropped that one line.
