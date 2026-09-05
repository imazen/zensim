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
