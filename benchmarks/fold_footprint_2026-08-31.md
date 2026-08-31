# The fold's footprint — where every byte goes, and why it was ever heavier than buffered

**The question this lane exists to answer, in the user's words:** *why does the
streaming fold ever use MORE memory than the whole-image buffered path?*
[`fold_mt_scaling_2026-08-31.md`](fold_mt_scaling_2026-08-31.md) §5.2 measured
the fold's working set at **1.32× buffered at 1152²/1T and 1.38× at 16T**, but
**0.75×/0.85× at 2304²** — and recorded the shape (buffered scales with AREA,
the fold closer to WIDTH) without decomposing it. On paper a rolling-window walk
cannot lose to one that materialises whole-image pyramids. It did, below ~3 MP.

**The answer in one sentence: buffered sizes its band scratch by the WORKER
COUNT and the fold sized its by the FAN-OUT SHAPE.** Buffered's `ScaleBuffers`
comes from a rayon `map_init`, so it exists once per worker — one buffer at one
thread. The fold pre-allocated a `FoldPoolScratch` per *(channel × band slot)* =
**12 band buffers regardless of the pool**, plus **three full 14-plane
`ScratchV2Strip` sets of which a `v1_only` score writes two planes**. At one
thread that is 12 band buffers against buffered's 1, on identical work.

**What shipped:** three byte-neutral fixes, each gated by the existing
bit-identity suite. The fold's working set drops **55–57 % at 1 thread** and
**9–19 % at 8/16 threads**, and the RSS crossover moves from **~3.2 MP down to
~0.5 MP at 1 thread**, and from ~2.2 MP to **~1.4 MP at 16 threads**. Buffered is the control and
moved ≤ 1.5 %. Zero bytes moved: `fold_engine_parity` (22 geometries × rayon
pools 1/2/3/8/16, both engines, `to_bits()` on score + every feature +
`mean_offset`) passes, and the two new thread-derived quantities are swept BY
that suite.

Predecessors: [`fold_mt_scaling_2026-08-31.md`](fold_mt_scaling_2026-08-31.md)
(§2.3 `ADVANCE_ROWS`, §2.5 self-blur bands, §5.2 RSS, §6's rejected
`scale_capacity_rows` change), [`fold_engine_2026-08-31.md`](fold_engine_2026-08-31.md),
[`extraction_perf_and_buffered_removal_2026-08-30.md`](extraction_perf_and_buffered_removal_2026-08-30.md) §6.

---

## 1. Instruments and conditions

| what | how |
|---|---|
| peak RSS | `/usr/bin/time -v`, ONE arm per process, `ZEN_FE_RSS` mode of `zensim/benches/fold_engine_bench.rs` (the bench owns the workload; `fold_footprint_2026-08-31/rss_sweep.sh` only drives it) |
| working set | peak RSS − the two input images (`6·W·H` bytes), the column §5.2 introduced |
| per-term truth | `heaptrack` + `heaptrack_print` PEAK MEMORY CONSUMERS, which attributes **requested** bytes to a call site — the allocation side, exact |
| pool-block share | two new RSS control arms, `poolctl_full` / `poolctl_off`: the fold's scoring walk shape (`v1_only`, SDR, one fresh `V2Scratch` per call, exactly as `Zensim::compute` builds one) differing ONLY in `V1PoolsMode`. `poolctl_off` emits structural zeros in f156..371 and is a measurement control, never shippable. The control is faithful by measurement, not by argument: `poolctl_full` and `score_fold` come back at **83.14 vs 83.17 MB** peak heap before the fix and **29.78 vs 29.81 MB** after |
| model check | `fold_footprint_2026-08-31/model.py`, every term derived from source; the ONLY fitted quantity is a per-arm process baseline `P0` taken from the 128×128 row and then held fixed |

Units are **KiB** (`/usr/bin/time -v`'s own unit) and **MB decimal** where a
heaptrack figure is quoted, because heaptrack prints decimal. §5.2's table is in
decimal MB; ratios are unit-free and comparable across both.

**Load conditions.** The `rss_before` sweep ran 05:30–05:36 UTC at box load
**0.49–1.52**; `rss_after` ran 05:48–05:50 at load **3.4–5.7** (an unrelated
lane). Peak RSS is determined by what the program allocates and touches, not by
scheduling, and the buffered control arm — measured in both sweeps, unchanged
code — reproduces to **≤ 1.5 %** across that 4× load difference, which is the
evidence that the load did not reach these numbers. The speed A/B in §7 is a
different matter and states its own conditions.

---

## 2. The buffered model — and the pyramid series that is not there

```
R_buf(W,H,T) = P0
             + 24·W·H                                    XYB planes
             + max( min(T, ⌈H/32⌉) · 1176·W ,             ScaleBuffers
                    min(T, ⌈H/64⌉) ·  192·W )             conversion rgb_buf
```

* **`24·W·H` — and there is NO `1 + 1/4 + 1/16 + 1/64` series.** `convert_source_to_xyb`
  (`streaming.rs:1104`) allocates `vec![0.0f32; W*H]` × 3 channels × 2 sides, and the
  pyramid cascade runs `blur::downscale_2x_inplace` (`blur.rs:3852`), which downscales
  a plane **over its own prefix** and then `Vec::truncate`s — and `truncate` never
  releases capacity. All four scales share the same six allocations, pinned at
  scale-0 capacity, with only one scale's data live at a time. Confirmed by
  heaptrack: **31.85 MB over 18 calls** at 1152² (= 3 iterations × 6 planes), which
  is `24·W·H` to the byte. The 85/64 series belongs to `PrecomputedReference`
  (`streaming.rs:2884`), which owns every level — the `ref_*` arms, not this one.
* **Widths are NOT SIMD-padded.** `blur::pyramid_plane_stride(w) == w` since option C
  (`blur.rs:4198`), so every `n` is `W·H` exactly and `mirror_offsets` is empty.
* **`ScaleBuffers` = 7 `Vec<f32>` × `(STRIP_INNER + 2·overlap) = 42` rows × `w_s`**
  (`pool.rs:9`, sized at `streaming.rs:2446`). The band layout is geometry-only —
  one band per 32-row strip, deliberately not thread-derived — and the fan-out is
  `map_init` per rayon worker (`streaming.rs:2677`), so **concurrently live =
  min(T, ⌈h_s/32⌉)**, each `1176·w_s` bytes, freed at the end of every scale.
  heaptrack at 1152²: **1.38 MB at 1T, 21.70 MB at 16T** against the model's 1.35 /
  21.68 MB (`1176·W` and `16 × 1176·W`).
* **`max`, not sum**: the conversion completes before the first `ScaleBuffers`
  exists, so the two phases are disjoint in time and the allocator reuses the pages.
  Summing them over-predicts 16T by 9–14 %; taking the max lands within 3.6 %.
* Not on this path at all: `V2Scratch` / `ScratchV2Strip` (fold-only), the phase-6
  whole-image bypass (`STRIP_BYPASS_HEIGHT = 0`, permanently dead), the attribution
  retention planes, and any whole-image `mu1`/`activity`.

**Measured vs predicted, `score_buffered`** (`P0` = 4,016 KiB fitted on 128²):

| size | 1T | 8T | 16T |
|---|---:|---:|---:|
| 512² | −4.4 % | +6.4 % | +9.4 % |
| 768² | −0.6 % | +0.3 % | +3.6 % |
| 1152² | +0.6 % | +1.4 % | +2.8 % |
| 1536² | +0.6 % | +0.6 % | +1.8 % |
| 2048² | +0.2 % | +0.9 % | +1.0 % |
| 2304² | +0.3 % | +1.0 % | +1.5 % |
| 3072² | +0.1 % | +0.4 % | +0.6 % |

`min(T, ⌈H/32⌉)` is an UPPER bound on live folders — rayon creates one per
sequential leaf, not per thread — which is why the error is positive and largest
where the band count is smallest (512² has 16 bands for 16 threads).

---

## 3. The fold model

```
R_fold(W,H,T) = P0
              + 24 · Σ_{s=0..3} (W≫s) · cap_s              rolling pyramid windows
                   cap_s = min( 128 + 20 + max(A≫s, 2) + 32 , (H≫s) + 20 )
              +  2 · 3 · 148 · W · 4                        ScratchV2Strip, WRITTEN planes
              +  3 · slots · 10 · 42 · W · 4                FoldPoolScratch
              + min(T, 2A/64) · 64 · W · 3                  conversion rgb_buf
```

with `A` the producer advance and `slots` the band-scratch count per channel.
Reading the constants: `24 = 2 sides × 3 channels × 4 B`; `cap_s`'s
`128 = STRIP_ROWS`, `20 = 2·HALO_P`, `32` = `scale_capacity_rows`'s slack;
`2 · 3 · 148 · 4` = 2 written planes × 3 channels × `(STRIP_ROWS + 2·HALO_P)`
rows × 4 B; `3 · slots · 10 · 42 · 4` = 3 channels × slots × (6 pool + 4
band-local H) planes × `(V1_BAND_ROWS + 2·V1_BAND_OVERLAP)` rows × 4 B; the
conversion chunk is `CONVERT_CHUNK_ROWS` rows of 3-byte RGB. Every one is read
from source; none is fitted. The buffered model likewise takes its band term at
**scale 0**, which dominates — deeper scales have a quarter of the rows and half
the width, and run sequentially after it.

Before this lane: `A = 256` always; `slots = V1_BANDS_PER_STRIP = 4` always; the
strip term was **14** planes allocated (2 written); and slot 0's ten planes were
`Vec`-doubled to **74** rows for a 42-row band.

### 3.1 The exact decomposition, heaptrack, 1152²/1T, BEFORE

| consumer | calls | peak | closed form | ✓ |
|---|---:|---:|---|---|
| `ScratchV2Strip::new` | 126 | **28.64 MB** | `14·3·148·W·4 = 24,864·W` | exact |
| pool `ensure`/`ensure_h` (`RawVecInner::finish_grow`) | 603 | **27.65 MB** | `3·(3·42 + 74)·10·W·4 = 24,000·W` | exact |
| `RollingPlane::from_pooled` | 72 | **18.57 MB** | `24·Σ W_s·cap_s` | exact |
| the bench's two input images | 2 | 7.96 MB | `6·W·H` | exact |
| `convert_…_chunked` `rgb_buf` | 108 | 221 KB | `min(T,8)·192·W` | — |
| `MeanOffsetRows` + accums | 66 | 40 KB | `O(H)` | — |
| **peak heap** | | **83.17 MB** | | |
| *buffered, same cell* | | **41.30 MB** | 31.85 + 1.38 + 7.96 | |

126 = 14 planes × 3 channels × 3 iterations; 72 = 2 sides × 3 channels × 4 scales
× 3 iterations. The `27.65 MB` is `3 channels × [3 slots at 42·W + 1 slot at
74·W] × 10 planes × 4 B` **to the byte** — which is how the `Vec` growth slack was
found: slot 0 always takes the top-clamped first band of a strip (37 rows), then
an interior band (42), and `Vec::resize` reserves `max(2·cap, need)`.

### 3.2 Where the excess was, priced

At 1152² the fold allocated **75.21 MB** of working memory (peak heap − input)
against buffered's **33.34 MB** — 2.26×. Three items account for all of it:

1. **12 of 14 `ScratchV2Strip` planes are never written by a score.** A
   `v1_only` + `V1PoolsMode::Full` request runs self-blur bands, so
   `stream_phase_a` is skipped WHOLE (§2.5 of the MT note) and the only planes
   touched are `src_wide`/`dst_wide`. **21,312·W bytes untouched** — 24.6 MB at
   1152², 49.1 MB at 2304².
2. **The pool block is 12 band buffers where buffered has one per worker.**
   `3 ch × 4 slots × 10 planes × 42 rows` = `20,160·W` bytes needed and `24,000·W`
   allocated. Buffered's directly comparable term is `1,176·W` **per live worker**.
   At 1 thread that is **20.4×**.
3. **`ADVANCE_ROWS = 256` is a parallel-degree knob priced in capacity.** Between
   64 and 256 it costs `24 · 1.328 · (A−64) · W` = **6,120·W bytes**, height-
   independent — 6.9 MB at 1152², 13.7 MB at 2304² — for a producer fan-out no
   small pool can spend.

**Is the untouched allocation free?** Under stock glibc, `vec![0.0; n]` lowers to
`alloc_zeroed`; a plane that size is `mmap`ed and its pages are demand-zero, so
not faulting them costs address space rather than RSS — mostly. MEASURED at
1152²/1T: the fold allocates **75.21 MB** of working memory and its working RSS
is **55.17 MB**, a 20.0 MB gap against 24.55 MB of untouched plane, so about
four fifths of it genuinely stays out of RSS and the rest does not. And that
much is an ALLOCATOR POLICY, not a property of the program; it was tested rather
than assumed:
with **`MALLOC_ARENA_MAX=1`** the pre-fix binary's `score_fold` peak at 1152²/1T
goes **61,952 → 71,768 KiB (+15.8 %)**, because the allocation now comes from the
main heap and `calloc` must memset it. Not asking for the planes is the only way
not to depend on the allocator's mood — and after §5.1 the same probe costs
**31,412 → 31,784 KiB (+1.2 %)**. The exposure is gone because the allocation is.

---

## 4. The pool block, answered with numbers

**Before the fix the f156–371 pool block (peaks / masked / IW) is 33–34 % of the
fold walk's whole working set, at every size, at 1 thread — and 39–43 % at 16
threads.** It is the single largest term, it scales with WIDTH, and it is
independent of HEIGHT. That is the headline: buffered's dominant term is area, so
below the size where area overtakes width, a term like this decides the
comparison. `poolctl_full − poolctl_off`, working set:

| size | 1T before | 16T before | 1T after | 16T after |
|---|---:|---:|---:|---:|
| 512² | 11,296 KiB (32.4 %) | 15,768 KiB (39.8 %) | **−996 KiB** | 10,644 KiB (40.7 %) |
| 768² | 16,504 (32.6 %) | 25,836 (43.0 %) | **−1,580** | 15,684 (40.9 %) |
| 1152² | 25,136 (33.4 %) | 37,708 (42.7 %) | **−2,348** | 21,892 (38.1 %) |
| 1536² | 33,080 (33.4 %) | 48,760 (41.9 %) | **−2,604** | 29,976 (39.4 %) |
| 2048² | 44,228 (33.9 %) | 62,048 (40.8 %) | **−3,500** | 37,152 (37.5 %) |
| 2304² | 49,476 (33.9 %) | 73,668 (42.2 %) | **−4,124** | 40,920 (37.4 %) |
| 3072² | 67,036 (34.5 %) | 86,828 (39.4 %) | **−7,516** | 58,524 (39.2 %) |

Before the fix the delta is **21.5–22.1 KiB per unit of width** at 1T (flat
across a 6× range of widths — the term really is linear in W and blind to H)
against the `24,000·W` bytes = 23.4 KiB/W the allocation model predicts, i.e.
**92–94 % resident**; the missing 6–8 % is the untouched tail of slot 0's
doubled buffers.

**The negative column is real and is the point.** After the fix, at one thread,
turning the pools ON makes the walk **smaller**: `poolctl_full` runs self-blur
bands, so it pays `3 ch × 1 slot × 10 × 42 · W · 4 = 5,040·W` bytes of pool and
skips phase A entirely, while `poolctl_off` must run phase A and therefore pays
the four strip-wide H planes, `4 × 3 × 148 · W · 4 = 7,104·W`. Predicted delta
**−2,064 B/W**; measured **−2,087 B/W at 1152²** (1.1 % off). The pool block went
from the fold's largest liability to cheaper than the phase it replaces.

At 16 threads all four slots are justified — four bands of one channel really can
run at once — so the pool stays `20,160·W` and remains the largest term (37–41 %).
**That is the honest floor:** the band buffer is `10 planes × 42 rows × width`
because `V1_BAND_ROWS = 32` is v1's numerics contract (the f32 sliding V-blur
re-initialises at every band's buffer top, so pooled sums depend on the tiling)
and the four H planes plus six V/activity planes are what
`fused_vblur_features_ssim` reads and writes. Below `min(bands, threads)` copies
of it there is nothing left to remove without changing either the kernel or the
tiling, and the tiling is not ours to change.

---

## 5. What shipped

All three are byte-neutral **by construction**, and all three are swept by
`fold_engine_parity::both_engines_are_bit_identical_across_rayon_pool_sizes`,
which compares the fold at 1T against the fold at 2/3/8/16T bit-identically over
22 geometries. Because two of the fixes derive a quantity FROM the pool size,
that pre-existing test is now also a direct sweep of them.

### 5.1 `StripPlaneNeeds` — size the strip scratch to the planes the walk writes

`ScratchV2Strip::new_for(max_n, needs)` + `V2Scratch::ensure_for`. The set has two
groups: `h` (the four fused-H planes phase A produces) and `v2` (everything below
`run_blur_pass_inner`'s `want_v2` early return, plus the σ-split and dst-activity
buffers that reuse its temps). A `v1_only` + `Full` score asks for neither, i.e.
**2 planes of 14**. The `MALLOC_ARENA_MAX=1` probe closes on this: **+15.8 % peak
RSS before the fix, +1.2 % after** (1152²/1T). Both the size and the set are grow-only — the union of every
request so far — so a driver alternating a score with a 944 extraction converges
to ALL and never thrashes, exactly like the existing `sized_for`.

`fuse_channels` / `self_blur` were hoisted out of the strip loop to compute the
set (they were strip-independent already; the loop now reads the hoisted values,
so there is one definition of each).

*Value: −21,312·W bytes allocated (−24.6 MB at 1152², −49.1 MB at 2304²).*

### 5.2 `band_slots_for` — size the band scratch to the concurrency, like buffered

`slots = min(V1_BANDS_PER_STRIP, rayon::current_num_threads())` when
band-parallel, else 1. A band cannot run without a thread to run it on, and rayon
nests, so that is exactly the count that can ever be simultaneously live inside
one channel task.

`fold_v1_basic_bands`'s parallel arm becomes `starts.par_chunks(per).zip(slots.par_iter_mut())`
with `per = ⌈bands/slots⌉`. With at least as many slots as bands — every pool of
4+ threads — `per` is 1 and it is exactly the old one-band-per-slot zip. **Order
is the whole bit-exactness argument:** `par_chunks` and `par_iter_mut` are both
INDEXED, so `collect` yields chunks in band order and each chunk yields its bands
in band order; flattening reproduces the serial band sequence, and the merge below
is the same left-to-right `f64` fold either way. (The serial arm already indexed
`p[i.min(len−1)]` and needed no change.)

*Value: 12 band buffers → 3 at 1 thread, −15,120·W bytes; unchanged at ≥ 4.*

### 5.3 `advance_rows_for` — `ADVANCE_ROWS` becomes a ceiling

The constant exists to keep the conversion fan-out fed and nothing else: the two
sides run concurrently and each fans out at `CONVERT_CHUNK_ROWS`, so one
`produce()` offers `2A/64` tasks and `32·threads` rows is the smallest advance
that offers one per thread. The advance is now
`clamp(⌈32·T⌉₆₄, 64, 256) ∧ ⌈H⌉₆₄` — **64** at 1–2 threads, **128** at 3–4,
**192** at 5–6, **256** at ≥ 7, i.e. **unchanged at the 8- and 16-thread
configurations the MT lane's headline was measured on**.

Chunk HEIGHT is semantics (`convert_chunk_rows_is_semantics_not_a_knob` measures
one ULP at 97×51); chunk COUNT is not, because every boundary still lands on a
global multiple of 64 — which is why this rounds UP to that lattice and asserts
it at construction. The new
`feature_v2_stream::tests::producer_windows_are_advance_invariant` pins the
producer directly: 4 geometries × advances 64/128/192/256/512, every emitted
strip's wide window on both sides and all 3 channels compared by `to_bits()`.
(Emission ORDER across scales legitimately differs — that is what the advance
changes — so the test compares the sorted set of `(scale, y0, side, channel)`
windows, which is the invariant that matters: each scale owns its own accumulator
and strips within a scale stay strictly ascending.)

*Value: −6,120·W bytes at 1–2 threads, −4,080·W at 3–4, −2,040·W at 5–6;
nothing at ≥ 7. (Unclamped; `cap_s`'s `(H≫s) + 20` cap trims it at deep scales
for short images.)*

### 5.4 `FoldPoolScratch::ensure` sized to the maximum band

`ensure(band_cap_n)` with `band_cap_n = min(42, height)·width` instead of this
band's own `h_local·width`, so slot 0 stops being doubled to 74 rows for a
42-row band. Everything downstream already sliced `[..band_n]`.

*Value: −3,840·W bytes at every thread count.*

---

## 6. Result — the crossover moved

Working set (peak RSS − input), KiB. `ratio` is `score_fold ÷ score_buffered`;
`score_buffered` is the untouched control and is shown to move ≤ 1.5 %.

| size | T | buf before | buf after | fold before | fold after | fold Δ | ratio before | **ratio after** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512² | 1 | 11,244 | 10,872 | 28,244 | **12,516** | −55.7 % | 2.512 | **1.151** |
| 512² | 8 | 13,964 | 13,772 | 30,172 | 25,284 | −16.2 % | 2.161 | 1.836 |
| 512² | 16 | 17,888 | 17,656 | 33,900 | 25,736 | −24.1 % | 1.895 | 1.458 |
| 768² | 1 | 18,840 | 18,392 | 38,660 | **17,304** | −55.2 % | 2.052 | **0.941** |
| 768² | 8 | 24,824 | 24,100 | 43,204 | 37,384 | −13.5 % | 1.740 | 1.551 |
| 768² | 16 | 30,844 | 29,940 | 46,060 | 36,904 | −19.9 % | 1.493 | 1.233 |
| 1152² | 1 | 36,228 | 36,096 | 53,880 | **23,188** | −57.0 % | 1.487 | **0.642** |
| 1152² | 8 | 45,088 | 44,796 | 64,624 | 53,400 | −17.4 % | 1.433 | 1.192 |
| 1152² | 16 | 54,760 | 54,700 | 64,668 | **55,208** | −14.6 % | 1.181 | **1.009** |
| 1536² | 1 | 60,732 | 61,288 | 69,436 | 30,512 | −56.1 % | 1.143 | 0.498 |
| 1536² | 8 | 73,008 | 72,552 | 79,596 | 72,428 | −9.0 % | 1.090 | 0.998 |
| 1536² | 16 | 86,008 | 86,516 | 82,368 | 72,020 | −12.6 % | 0.958 | 0.832 |
| 2048² | 1 | 104,488 | 104,420 | 88,356 | 39,200 | −55.6 % | 0.846 | 0.375 |
| 2048² | 8 | 120,088 | 119,616 | 105,980 | 94,440 | −10.9 % | 0.883 | 0.790 |
| 2048² | 16 | 138,524 | 138,760 | 115,492 | 93,096 | −19.4 % | 0.834 | 0.671 |
| 2304² | 1 | 130,708 | 130,980 | 99,520 | 43,832 | −56.0 % | 0.761 | 0.335 |
| 2304² | 8 | 148,072 | 148,532 | 108,816 | 106,848 | −1.8 % | 0.735 | 0.719 |
| 2304² | 16 | 168,264 | 169,324 | 116,724 | 104,748 | −10.3 % | 0.694 | 0.619 |
| 3072² | 1 | 228,480 | 228,280 | 129,336 | 57,084 | −55.9 % | 0.566 | 0.250 |
| 3072² | 8 | 252,352 | 252,164 | 157,956 | 137,872 | −12.7 % | 0.626 | 0.547 |
| 3072² | 16 | 279,940 | 280,520 | 163,556 | 131,872 | −19.4 % | 0.584 | 0.470 |

**Crossover** (the size at which the fold becomes the lighter path):

| threads | before | after |
|---|---|---|
| 1 | between 1536² and 2048², interpolated **~3.2 MP** | between 512² and 768², **~0.5 MP** |
| 8 | between 1536² and 2048², **~3.2 MP** | at 1536², **~2.35 MP** (ratio 0.998) |
| 16 | between 1152² and 1536², **~2.2 MP** | at 1152², **~1.4 MP** (ratio 1.009) |

The bracketing sizes are measured; the MP figure inside each bracket is a linear
interpolation of the ratio in AREA and is quoted to one decimal for that reason.
(The landing commit message rounds the 1T pair as 2.7 → 0.6 MP; these are the
interpolated values and supersede it.)

**Per-term predicted vs measured, AFTER** — heaptrack at 1152², decimal MB, the
same decomposition §3.1 gives for the before state:

| term | 1T measured | 1T closed form | 16T measured | 16T closed form |
|---|---:|---|---:|---|
| `RollingPlane::from_pooled` | **11.60** | `24·Σ W_s·cap_s`, A=64 → 11.60 | **18.57** | A=256 → 18.57 |
| pool `ensure` (`finish_grow`) | **5.81** | `3·1·10·42·W·4` → 5.81 | **23.22** | `3·4·10·42·W·4` → 23.22 |
| `ScratchV2Strip::new_for` | **4.09** | `2·3·148·W·4` → 4.09 | **4.09** | → 4.09 |
| `rgb_buf` | 0.22 | `min(T, 2A/64)·192·W` → 0.22 | 1.77 | → 1.77 |
| input images | 7.96 | `6·W·H` → 7.96 | 7.96 | → 7.96 |
| **peak heap** | **29.81** | Σ = 29.68 | **55.86** | Σ = 55.61 |
| *buffered, same cell* | **41.30** | 31.85 + 1.38 + 7.96 | **61.73** | 31.85 + 21.70 + 7.96 |

Every fold term matches its closed form to **< 0.1 %** (the 0.2–0.4 % gap in
the peak-heap row is the accumulators, `MeanOffsetRows` and rayon/crossbeam
bookkeeping the table does not itemise), and buffered's `ScaleBuffers` at 16T is
21.70 measured against 21.68 predicted. So on the
allocation side the fold is now the lighter path at 1152² at BOTH thread counts
— 29.81 vs 41.30 MB at 1T and 55.86 vs 61.73 at 16T.

### 6.1 Model accuracy, `score_fold`, AFTER (`P0` = 3,824 KiB from the 128² row)

| size | 1T | 8T | 16T |
|---|---:|---:|---:|
| 512² | +3.8 % | −5.5 % | −7.1 % |
| 768² | +3.2 % | −7.5 % | −6.3 % |
| 1152² | +8.0 % | −5.7 % | −8.8 % |
| 1536² | +5.6 % | −8.8 % | −8.2 % |
| 2048² | +6.3 % | −8.0 % | −6.7 % |
| 2304² | +5.9 % | −9.0 % | −7.2 % |
| 3072² | +6.2 % | −6.9 % | −2.7 % |

Every cell inside ±10 %, with a sign that is itself informative: **positive at
1T** (a fraction of the strip scratch's pages genuinely never fault) and
**negative at 8/16T** by a term the model deliberately does not carry — §6.3.

### 6.2 The crossover PREDICTED from the model, then measured

The model is closed-form, so the crossover is a root, not an observation:
solve `fold_terms(W,W,T) = buf_terms(W,W,T)` for the square family (`P0` cancels
— same process either way). `model.py crossover`:

| threads | before (predicted) | after (predicted) |
|---|---:|---:|
| 1 | W = 1835, **3.37 MP** | W = 730, **0.53 MP** |
| 2 | 1794, 3.22 MP | 902, 0.81 MP |
| 4 | 1712, 2.93 MP | 1331, 1.77 MP |
| 8 | 1548, 2.40 MP | 1335, 1.78 MP |
| 16 | 1150, 1.32 MP | 937, 0.88 MP |

Against the measurement: **at 1 thread the prediction is within 5 %** — 3.37 vs
~3.2 MP before, 0.53 vs ~0.5 MP after. At 8 and 16 threads it under-predicts by
0.6–0.9 MP, in exactly the direction and roughly the magnitude of the term §6.3
isolates: the fold's measured multi-thread RSS runs 5–9 % ABOVE model while
buffered's runs 1–4 % below, and a ~10 % error in the ratio moves the crossing
point by about that much. **The model's only systematic failure is a term that
is not a program allocation**, and it fails only where that term exists.

One shape worth naming: the predicted crossover is NOT monotone in threads. It
improves from 1 → 2 threads (0.53 → 0.81 MP is a worsening in MP terms, i.e. the
fold needs a bigger image to win) and worsens again to 4, because that is where
`band_slots_for` steps 1 → 2 → 4 and `advance_rows_for` steps 64 → 128; then it
improves through 16 as buffered's per-worker `ScaleBuffers` grows past the fold's
now-fixed 12. The 4–8 thread band is where the fold's footprint is structurally
worst, and §8's cross-channel pool is exactly the lever that band wants.

### 6.3 The one term the model does not predict, isolated

The fold's 8/16-thread residual is **allocator churn across per-worker arenas,
not a program allocation**, and it was isolated rather than fitted. Peak RSS at
1152² on the POST-fix binary, varying only the number of compares in the loop:

| iterations | fold 1T | fold 16T | buffered 1T | buffered 16T |
|---|---:|---:|---:|---:|
| 1 | 31,632 | **54,508** | 44,028 | 62,236 |
| 2 | 31,332 | 60,132 | 43,652 | 63,560 |
| 5 | 31,380 | 61,972 | 44,508 | 63,416 |
| 20 | 31,876 | **64,804** | 48,856 | 63,056 |

The fold's 16T figure grows **+19 % from one compare to twenty** while its 1T
figure is flat to 0.8 %. `Zensim::compute` builds a fresh `V2Scratch` per call
(`metric.rs:3336`), and the pool buffers inside it are grown *inside band tasks*,
so the arena that served compare *k* is not the one compare *k+1* asks — glibc
retains both. On the PRE-fix binary `MALLOC_ARENA_MAX=1` cuts the fold's 16T−1T
delta from 9,800 to 4,756 KiB, i.e. **arenas are about half of it**.

Against the ITERS=1 column the model's 16T error at 1152² **flips sign, to
+7.8 %** (50,358 predicted against a 46,732 KiB working set) — so the churn is
the whole of the −8.8 % at ITERS=5 and then some, and what is left over is the
same positive residual the 1T column shows. It is a single-compare model, which
is what it should be; the ITERS=5 table is the LOOP figure, and comparable to
§5.2's ITERS=20 predecessor.

Buffered has the mirror-image effect at **1T** (44,028 → 48,856 over 20 compares):
`streaming.rs:357`'s background `dealloc_planes` thread joins the *previous*
handle before spawning, so one compare's `24·W·H` can still be resident during the
next one's conversion.

**Fix direction, not taken here:** the ref-loop entry already keeps its
`V2Scratch` alive in `ZensimScratch` (the MT lane's private-field fix), and
`refinto_fold` therefore does not churn. Giving the plain `compute` the same
amortisation needs either state on `Zensim` (which is `Sync` and shared) or a
thread-local, and that is an API/ownership decision, not a footprint one.

---

## 7. Speed — the footprint cut is worth 13–27 % of wall clock at one thread

`fold_engine_bench` under zenbench, the same bench/arms/generator/box the MT
lane's §5 used. zenbench interleaves arms WITHIN one process, so the before and
after builds cannot share a group; the two binaries are alternated instead
(`speed_ab.sh`), one process per (binary, thread count). Raw outputs are
committed beside this note under `fold_footprint_2026-08-31/speed/`.

**Sanity check on the pairing:** the BEFORE binary reproduces the MT lane's §5
table on every arm it shares — 1152²/1T `score_buffered` 48.0 vs 48.59,
`score_fold` 50.5 vs 50.39; 2304²/1T 204.2 vs 210.00 and 201.6 vs 201.95;
2304²/16T 34.8 vs 32.25 and **54.4 vs 54.47**. The box has not moved under us.

### 7.1 One thread — the clean measurement

| arm | 1152² before | after | Δ | 2304² before | after | Δ |
|---|---:|---:|---:|---:|---:|---:|
| `score_buffered` *(control)* | 48.0 | 47.7 | −0.6 % | 204.2 | 200.5 | −1.8 % |
| **`score_fold`** | 50.5 | **37.1** | **−26.5 %** | 201.6 | **175.1** | **−13.1 %** |
| `feat_buffered` *(control)* | 47.6 | 47.8 | +0.4 % | 202.1 | 198.3 | −1.9 % |
| `feat_fold` | 48.0 | 37.3 | −22.3 % | 199.1 | 173.1 | −13.1 % |
| `ref_buffered` *(control)* | 40.8 | 40.7 | −0.2 % | 181.4 | 174.6 | −3.7 % |
| `ref_fold` | 45.9 | 33.8 | −26.4 % | 187.6 | 161.9 | −13.7 % |
| `refinto_buffered` *(control)* | 41.4 | 40.9 | −1.2 % | 179.5 | 173.1 | −3.6 % |
| `refinto_fold` | 44.0 | 34.1 | −22.5 % | 185.9 | 161.8 | −13.0 % |
| `fused_buffered` *(control)* | 66.8 | 66.3 | −0.7 % | 274.2 | 271.2 | −1.1 % |
| `split_fold` | 138.8 | 124.7 | −10.2 % | 556.3 | 556.4 | +0.0 % |

Every buffered arm is untouched code and moves ≤ 3.7 %; every fold arm that
takes the scoring walk moves **13–27 %**. **A memory change bought a quarter of
the serial wall clock**, which is the shape of a cache effect and not of a
compute one — nothing in these fixes removes work; `advance_rows_for` and the
band chunking both *add* a little (four times as many `produce()` calls at 1
thread, and 4 bands through 1 scratch instead of 4).

**`score_fold` is now FASTER than `score_buffered` serially: 37.1 vs 47.7 ms at
1152² (0.78×) and 175.1 vs 200.5 at 2304² (0.87×)**, against the MT lane's
recorded 1.03×/0.96×. The fold's serial position changed from parity to a clear
win, on a byte-identical result.

`split_fold` at 2304² is the one arm that does not move, and it is informative
rather than anomalous: the standalone attribution map does **not** go through
`foldapp_streaming_walk` (no `StripPlaneProducer` reaches `attribution.rs`), so
it cannot have changed — and it is large enough at 2304² (≈ 370 ms of the 556)
to evict everything the fold-score half just gained. At 1152², where both halves
still fit, the arm moves −10.2 %, consistent with `ref_fold`'s −12.1 ms. **The
gain is cache residency, and it is only there while something else is not
holding the cache.**

### 7.2 Sixteen threads — read the ratio, not the level

| arm | 1152² before | after | 2304² before | after |
|---|---:|---:|---:|---:|
| `score_buffered` *(control)* | 9.4 | 9.0 | 34.8 | **38.8** |
| `score_fold` | 13.9 | **12.0** | 54.4 | 57.6 |
| `feat_buffered` *(control)* | 8.1 | 7.9 | 28.5 | 32.1 |
| `feat_fold` | 13.9 | 11.9 | 53.1 | 57.9 |
| `ref_buffered` *(control)* | 7.2 | 6.8 | 27.5 | 29.7 |
| `ref_fold` | 14.3 | 11.7 | 53.7 | 57.7 |
| `refinto_fold` | 12.9 | 12.2 | 50.9 | 53.9 |

**The 2304²/16T "after" run is contaminated and must not be read as a level.**
EVERY arm in it is slower than its "before" counterpart, the untouched buffered
controls by **+8 % to +24 %**, and its `mad` triples (±6.3–47 ms against
±1.5–7.4). A concurrent lane's bench had been queued on zenbench's exclusive
lock across that window. What survives contamination is the RATIO inside one
process:

| size | `score_fold ÷ score_buffered` before | after |
|---|---:|---:|
| 1152² / 16T | 1.48 | **1.33** |
| 2304² / 16T | 1.56 | **1.49** |

At 1152² the ratio improves by 10 %; at 2304² by 4 %, which is inside that run's
noise. **That split is exactly what §9 predicts** and is the strongest
independent evidence for it: the fixes cut the walk's TOTAL footprint but not
its per-thread HOT SET, so they help wherever the reduced total changes what
stays resident (1 thread everywhere, 16 threads at 1152²) and do essentially
nothing at 16 threads on 2304², where `8 × 4.43 MiB = 35.4 MiB` still overflows
CCD1's 32 MiB L3 either way.

**Load conditions.** The four round-1 runs ran back to back 05:50–07:20 UTC
under zenbench's exclusive lock. Box load was 0.2–1.5 for the two 1-thread runs
and the 16-thread BEFORE run; the 16-thread AFTER run overlapped a queued
sibling lane and is flagged above. A second round was cancelled rather than run
against that contention.

---

## 8. What did not move, and why

* **The rolling window's 180-row floor.** `cap_s ≥ STRIP_ROWS(128) + 2·HALO_P(20)
  + 32` before the advance is added at all, so on a 512-row image the scale-0
  window is **244 rows (48 %) at 1 thread and 436 (85 %) at 8**, where buffered
  holds `24 B/px` outright — a rolling window over a short image is barely a
  window. That, plus four justified band slots at 8 threads, is why 512²/8T is
  still 1.84×. Cutting `STRIP_ROWS` is a numerics change, not a footprint one.
* **The band buffer's shape.** `10 planes × 42 rows` is `V1_BAND_ROWS = 32` plus
  v1's `overlap = 5` on both sides, and v1's band tiling is part of its numerics
  contract. The four H planes and six V/activity planes are exactly what
  `fused_blur_h_ssim` produces and `fused_vblur_features_ssim` consumes.
* **`scale_capacity_rows`'s `+32` slack.** Worth `1,020·W` bytes (1.1 MB at
  1152²). The C1 test asserts the budget is never exceeded, but that is a
  one-sided check; tightening it needs the actual `max_held_rows` distribution
  swept over geometries first. Not attempted — small, and the cheap version of it
  is a silent capacity-fallback `resize` in the hot path.
* **Sharing band slots ACROSS channels — the next lever, priced but not taken.**
  The fold parallelises over (channel × band) and so keeps `3 × min(4, T)` band
  buffers; buffered parallelises over bands alone and keeps `min(T, bands)`. At
  8 threads that is 12 buffers of 10 planes against buffered's 8 of 7 — 120
  plane-instances to 56 — and only 8 of the 12 can run. A pool of `min(12, T)`
  shared across channels would take the fold to 80, worth **−7.4 MB at 1152²/8T
  (ratio 1.192 → ~1.02)** and −3.4 MB at 512²/8T (1.836 → ~1.60), and **nothing
  at 16 threads**, where 12 < T already. The route that would work is
  `rayon::current_thread_index()` into a `Vec<Mutex<FoldPoolScratch>>` — one
  slot per worker, so the mutex is uncontended by construction, and no
  `unsafe` (the crate forbids it). It is not shipped here because it puts a
  lock in the band kernel's path for a win confined to one thread count, and
  because the MT lane already measured the naive form
  (`map_init(FoldPoolScratch::default)`, which re-allocates ~580 KB per worker
  per strip per channel) as a net LOSS — the persistent-slot version has to be
  measured on its own, not assumed from this arithmetic.
* **The MT note's §6 rejected `scale_capacity_rows` doubling** stays rejected, and this lane
  strengthens the case: it would have bought ~2 ms for ~20 MB at 2304², against
  which the advance is now *smaller* on the pools that could not use the degree.

---

## 9. The per-thread cache budget — the target the ratio was a proxy for

A footprint RATIO is not the quantity that decides thread scaling. The
predecessor measured the real one: **N independent single-threaded processes
saturate this box at 3.5× for the fold against 10.9× for buffered**, from the
same serial speed
([`fold_mt_scaling_2026-08-31.md`](fold_mt_scaling_2026-08-31.md) §4). That is a
statement about **per-thread HOT SET against L3**, not about total footprint, and
it is what the rest of this section prices.

### 9.1 The box is not the box the docs say

MEASURED (`lscpu`, `/sys/devices/system/cpu/cpu*/cache/index3/`), because this
matters and the workspace's Environment note says "AMD Ryzen 9 7950X":

| | |
|---|---|
| CPU | **AMD Ryzen 9 9950X3D**, 16 cores / 32 threads, 1 socket, 1 NUMA node |
| L1d | 48 KiB per core |
| L2 | **1 MiB per core**, shared by its 2 SMT siblings (`shared_cpu_list` 0,16) |
| L3 | **ASYMMETRIC — 128 MiB total in 2 instances**: CCD0 = cpus 0-7 + 16-23 → **96 MiB** (3D V-Cache); CCD1 = cpus 8-15 + 24-31 → **32 MiB** |

**`getconf LEVEL3_CACHE_SIZE` returns 33554432 (32 MiB) and is wrong for half
the machine** — it reports one instance. Read
`/sys/devices/system/cpu/cpuN/cache/index3/{size,shared_cpu_list}` instead. Any
budget derived from a flat "64 MiB / 16 threads" is doubly wrong here: the total
is 128 MiB, and it is not shared.

**Per-thread L3 budget**, assuming Linux spreads a rayon pool across physical
cores (8 per CCD at 16 threads):

| pool | CCD0 threads | CCD0 budget | CCD1 threads | **CCD1 budget (binding)** |
|---|---:|---:|---:|---:|
| 8 | 4 | 24 MiB | 4 | **8 MiB** |
| 16 | 8 | 12 MiB | 8 | **4 MiB** |
| 32 | 16 | 6 MiB | 16 | **2 MiB** |

The asymmetry is itself a hazard for any scaling claim on this box: half the
threads have **3× the cache of the other half**, so a hot set between 4 and
12 MiB per thread scales on CCD0 and thrashes on CCD1 — which is exactly the
shape a "saturates around 3.5×" measurement takes.

### 9.2 Every term as bytes PER THREAD

The hot set is what one band task touches while it runs, not the walk's total.
At scale 0, width `W`, a fold band task (`v1_only` + `Full`, self-blur) touches
**12 planes over `V1_BAND_ROWS + 2·V1_BAND_OVERLAP = 42` rows**: the 2 raw
windows it reads plus the 10 it writes (4 band-local H, `act_raw`, `act`,
`mu1_v`, `mu2_v`, `ssq_v`, `s12_v`) — `2,016·W` bytes. Buffered's band task
touches its 7 `ScaleBuffers` planes plus the same 2 raw windows — `1,512·W`.

| W | fold / thread | buffered / thread | fold × 8 (one CCD) | buffered × 8 |
|---|---:|---:|---:|---:|
| 1152 | 2.21 MiB | 1.66 MiB | 17.7 MiB | 13.3 MiB |
| 2304 | 4.43 MiB | 3.32 MiB | **35.4 MiB** | **26.6 MiB** |
| 4096 | 7.88 MiB | 5.91 MiB | 63.0 MiB | 47.2 MiB |
| 8192 | 15.75 MiB | 11.81 MiB | 126.0 MiB | 94.5 MiB |

**At 2304² — the size the ceiling was measured at — 8 threads on CCD1 need
35.4 MiB for the fold and 26.6 MiB for buffered against a 32 MiB L3. The
threshold falls exactly between them.** That is an arithmetic coincidence
precise enough to be worth testing rather than asserting, and §9.4 is the test.

**This lane's shipped fixes do NOT move this number**, and saying so is the
point: a band task touches ONE pool slot whether the walk keeps 3 of them or 12,
so `band_slots_for` changes the walk's total footprint and its cross-strip
reuse, not its per-task hot set. `advance_rows_for` shrinks a STREAMED term
(each rolling row is written once and read a few times) rather than a resident
one. What they bought is real and large (§6, §7) — it is just a different
quantity from the one that bounds thread scaling.

### 9.3 Column tiling is the only lever that reaches the budget — and it is not this lane's to pull

Every band today is **FULL WIDTH**, so per-thread bytes are `2,016·W`: linear in
image width, with no bound. No amount of pooling, slot-counting or window
trimming changes that, because it is the shape of the task, not of the
allocation. Splitting a band into column tiles of width `Tw` makes the per-thread
hot set **independent of image width**, which is the only property that reaches
an L3 budget at 4K and beyond.

**Halo cost, derived from the kernel chain, not guessed.** The band's chain is
`src → box_blur_h(r=5) → mu1_h → abs_diff → act_raw → box_blur_1pass(r=5) → act`,
and `box_blur_1pass_into` is `box_blur_h` **then** `box_blur_v_from_copy`
(`blur.rs:29`) — so there are **two chained H passes**, and a tile needs
`2 × BLUR_RADIUS = 10` extra columns per side. Buffer width `Tw + 20`;
redundant work `20/Tw`:

| `Tw` | redundant H work | hot set / thread (`2,016·(Tw+20)`) | fits CCD1 @16T (4 MiB)? | fits L2 (1 MiB/core)? |
|---:|---:|---:|---|---|
| 128 | 15.6 % | 0.28 MiB | yes | yes |
| 256 | 7.8 % | 0.53 MiB | yes | yes |
| 512 | 3.9 % | 1.02 MiB | yes | marginal |
| 1024 | 2.0 % | 2.01 MiB | yes | no |
| 2048 | 1.0 % | 3.98 MiB | just | no |
| 4096 | 0.5 % | 7.91 MiB | **no** | no |

There is a real optimum between redundant compute and locality somewhere around
`Tw` 256–1024 and it is a MEASUREMENT, not a derivation — the table gives the
two costs, not the answer. What the table does settle is that a fixed `Tw ≤ 2048`
holds the per-thread hot set under CCD1's 16-thread budget **at every image
width**, which nothing else on the table does.

Two further gains come free with it: the fan-out widens from `3 channels × 4
bands = 12` tasks per strip to `12 × ⌈W/Tw⌉` (24 at 2304² with `Tw = 1024`, 48
at 4096²), which is the other half of "scale with cores"; and the same shape
applies to buffered's `ScaleBuffers` bands, which have the identical
full-width problem at `1,512·W`.

**Why this lane must not implement it.** Column tiles change the ORDER of the
`f64` pooled accumulation inside a band — today a band's sums run whole rows
left-to-right and merge per band; tiled, each tile produces partials that must be
combined, and `f64` addition is not associative. **That moves bytes**, which is
this lane's hard constraint. The concurrent era-2 lane's fixed virtual-lane
grouping with a fixed band-order merge is precisely the property that would make
a column split byte-safe **by construction** — a tile boundary on a lane-group
multiple leaves every lane's addition sequence untouched. So this is recorded as
an **era-2-enabled design with a predicted gain**, to be bundled into that break
rather than paid for twice:

> **Predicted:** per-thread hot set `2,016·W → 2,016·(Tw+20)`, width-independent;
> at 2304²/16T that is 4.43 → 1.02 MiB per thread at `Tw = 512`, taking 8-thread
> CCD1 occupancy from **35.4 MiB (over a 32 MiB L3) to 8.2 MiB**, at a measured
> **3.9 % redundant H-blur cost** and a 2× wider fan-out. The saturation
> experiment in §9.4 is what says whether that converts into thread scaling.

### 9.4 The experiment that decides it

`fold_footprint_2026-08-31/ccd_saturation.sh` runs the predecessor's
N-independent-process saturation test **pinned per CCD** (`taskset -c 0-7` vs
`8-15`). If the fold's ceiling is an L3-per-thread effect it must saturate
EARLIER on CCD1's 32 MiB than on CCD0's 96 MiB; if the two CCDs behave the same,
the ceiling is DRAM bandwidth and column tiling buys far less than §9.3
predicts. Results below.

---

## 10. Reproduction

```sh
cargo build --release --bench fold_engine_bench -p zensim \
    --features feature-regime-v2,threads,custom-profiles
FE_BIN=<the bench binary> benchmarks/fold_footprint_2026-08-31/rss_sweep.sh out.tsv
benchmarks/fold_footprint_2026-08-31/model.py out.tsv after      # per-cell error
benchmarks/fold_footprint_2026-08-31/model.py crossover           # predicted crossover
BEFORE_BIN=… AFTER_BIN=… benchmarks/fold_footprint_2026-08-31/speed_ab.sh speed 1 16
```

Gates: `cargo test --release -p zensim --features feature-regime-v2,threads,custom-profiles,training`
(232 lib + 11 `fold_engine_parity` + every integration test, all green), plus a
`--no-default-features --features feature-regime-v2` build for the non-threaded
arms of both new functions.
