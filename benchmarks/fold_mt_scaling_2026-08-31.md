# Fold thread scaling — profile, six levers, and the ceiling that is not ours

**Lane:** get fold-backed scoring to buffered-class thread scaling **without
changing a single output byte.** Predecessors:
[`fold_engine_2026-08-31.md`](fold_engine_2026-08-31.md) §10 (the fold now
backs `score()`/`classify`/diffmap/attribution bit-identically, at serial
parity 1.03× but 2.3–3.3× behind under threads) and
[`extraction_perf_and_buffered_removal_2026-08-30.md`](extraction_perf_and_buffered_removal_2026-08-30.md)
§5/§6/§10.2/§11.1/§14.

**Result in one line, `fold_engine_bench` means:** at 2304² the fold-backed
score's thread scaling went **1.95× → 3.71×** (1→16T) and its ratio to buffered
**3.25× → 1.69×** (108.6 → **54.47 ms**, 2.0× faster); at 8 threads
**2.54× → 1.46×** (113.1 → **59.67 ms**). At 1152²: **1.94× → 3.49×** and
**2.93× → 1.61×** (26.4 → **14.45 ms**), 8T **2.30× → 1.37×**. Serial parity is
preserved (**0.96× / 1.04×**). Every lever is byte-neutral and gated.

The lane then STOPS on a measured ceiling that is a property of the fold's
memory behaviour, not of its schedule: N *independent single-threaded
processes* running the same walk saturate this box at **3.5×**, where the
buffered walk reaches **10.9×** — from the same serial speed. **The threaded
implementation now sits at 94 % of that bound at 8 threads and 108 % at 16**,
i.e. one process using sixteen threads is as fast per compare as sixteen
processes sharing nothing. There is no scheduling left to find.

---

## 1. Profile first — and the premise had to be re-derived, not inherited

The predecessors' shares are **serial instruction** shares (callgrind). A
serial profile structurally cannot see a thread idling at a barrier, so it
cannot name a parallel critical path. This lane added
`zensim/src/fold_timing.rs`: an env-gated (`ZENSIM_FOLD_TIMING=<N>`) per-phase
accumulator that records, for every phase of the strip loop, both the **wall**
span (its share of the critical path) and the summed **busy** time of the tasks
inside it — so `busy / (wall × threads)` is a measured occupancy. Every hook is
a timestamp plus a relaxed atomic add; disabled it is one resolved-`OnceLock`
load. It cannot touch a feature byte.

Driver: the existing `fold_engine_bench`'s `ZEN_FE_RSS` single-arm loop (the
bench is the owner; no new harness).

### 1.1 The measured parallel critical path, 2304², 16 threads, BEFORE the lane

| phase | wall ms/walk | % of walk | busy ms | occupancy |
|---|---:|---:|---:|---:|
| producer (`next_strip`) | 29.18 | 29.0 % | — | serial |
|   · scale-0 XYB convert | 19.33 | 19.2 % | | |
|   · downscale cascade | 9.82 | 9.7 % | | |
| phase A (3-way over channels) | 35.86 | 35.6 % | 90.25 | **0.157** |
| phase B (3-way, bands nested) | 28.78 | 28.6 % | 81.14 | 0.176 |
| unaccounted (scratch alloc, epilogue, bake) | 7.0 | 6.9 % | | |
| **walk** | **100.77** | | | |

### 1.2 The top three stalls, and the correction they force

1. **Phase A at degree 2.5 of 16** — 35.6 % of the wall, occupancy 0.157. It
   is `fused_blur_h_ssim` over the strip's wide window, parallelised only
   across the 3 channels.
2. **The producer, serial** — 29.0 %. Its inner conversion fan-out chunks at 64
   rows and only ever received `ADVANCE_ROWS` = 128, i.e. **two chunks**; the
   downscale cascade had no rayon at all.
3. **Phase B at effective degree 9.6** — 28.6 %; 3 channels × 4 bands = 12
   tasks is its hard cap, and `V1_BAND_ROWS` is semantics.

**`dense_block_kernel` does not appear.** The predecessor named it as the
23 %-of-walk, era-locked MT ceiling (§14.1/§14.2, Amdahl bound 1.17×@8T). That
is true of the 944-full *extraction*. A fold-backed **score** asks for
`v1_only + V1PoolsMode::Full`, and `stream_phase_b` gates the whole dense +
gradient + append + blockiness block behind `v2_blocks` — so a scoring walk
never dispatches it. The era decision the predecessor put the MT ceiling behind
is **not on this lane's critical path at all**.

The same correction applies to §11.1's "row-parallel H-blur — MEASURED NEUTRAL,
reverted". That measurement was taken in the 944-full walk, where phase A sits
behind dense and the Y-only blocks. Here phase A **is** the largest single
phase. Re-deriving the share, rather than trusting the recorded verdict, is
what made lever 1 available.

---

## 2. The levers, each byte-neutral by construction

### 2.1 Row-band-parallel H blur (`fused_blur_h_ssim_banded`)

A horizontal box blur is an independent running-sum recurrence per row, and
the kernels transpose a group of rows into lanes and run one group at a time
plus a partial tail: **16 rows on `v4`/`v4x`**, **8 on `v3`/`neon`/`wasm128`/
`scalar`**. Bands start at multiples of the coarser of those (16), which is a
multiple of the other, so on EVERY tier a band's internal grouping is a
**sub-sequence of the whole-plane call's** — same group size, same offset
within it — and only the last band can hold the partial group the whole-plane
call's tail would have run.

(The per-tier group sizes were read from `blur.rs` rather than assumed: an
earlier draft of this note said "8" for all tiers, which is wrong for the two
tiers this box actually runs. The shipped band size satisfies both either way.)

Gate: `phase_a_blur_bands_are_bit_exact` — 7 geometries × 6 band sizes,
`to_bits` equality on all four output planes. Three of the six (8, 12, 24) are
NOT multiples of the row group, so they test the per-row independence directly
rather than the alignment argument; they are recorded, not relied on.

`H_BLUR_BAND_ROWS = 16` — the smallest multiple of the row group, which is
also what measurement wanted (§3).

### 2.2 Producer: two sides concurrent + a 6-way downscale cascade

The two sides read disjoint sources and write disjoint plane sets, so
`rayon::join` over them is a pure schedule change (HDR keeps the serial loop —
it shares one `hdr_row` scratch). Within one pyramid scale the six
(side, channel) downscales are likewise disjoint; scales stay **sequential**
because scale `s` consumes rows scale `s−1` produced in the same call.

### 2.3 `ADVANCE_ROWS` 128 → 256 — and the chunk height that is NOT a knob

The direct way to raise the producer's degree is to lower the conversion's
parallel chunk height. **That moves bytes**, and the gate caught it: at 97×51
into a 104-wide destination, chunk height 1 moves plane 0 index 200 from
`0.7426108` to `0.7426113` — one ULP. The mechanism is the per-chunk
`srgb_to_positive_xyb_planar_into` call: which elements land in the SIMD body
versus the scalar tail depends on the buffer length, and the two disagree in
the last bit.

The streaming and materialised conversions are byte-identical only because
every producer chunk boundary lands on a **global multiple of 64 rows**, which
is where the whole-image call cuts. So the lane raised the row COUNT instead —
`ADVANCE_ROWS` 128 → 256 doubles the chunk count without moving a boundary, and
carries a `const` assert that it stays a multiple of 64.

Gate: `convert_chunk_rows_is_semantics_not_a_knob` pins both halves — 64
reproduces the default exactly on both arms, and at least one other height must
**not**, so if the per-pixel kernels ever became length-invariant the
constraint (and this section) can be lifted deliberately rather than by
accident.

### 2.4 The fused per-channel fan-out

The split fan-out (phase A for all 3 channels → barrier → phase B for all 3)
exists because Y's `append`/BANDVIS cross-transducer reads X's and B's phase-A
activity. A `v1_only` scoring request has `append_on == false`, so `cross` is
`None` and `refy` empty on every channel: **the only cross-channel edge in the
walk is absent**, and the barrier buys nothing while holding three channels'
~9 MB of phase-A buffers live across it. Fused when
`!append_on && retention.is_none()`; the 944-full extraction satisfies neither
and keeps the split, unchanged.

### 2.5 Self-blur bands — the structural one

With `v1_only`, phase A's **only** output is the four H-blurred planes over the
strip's 148-row wide window, and the v1 bands are their only consumer. Writing
them in one set of tasks and reading them back in another sends four planes
through L3/DRAM per strip.

`FoldHSource::SelfBlur` gives each band its own H blur over exactly the
`[b0 − 5, b1 + 5)` rows it consumes, into its own `FoldPoolScratch`. Phase A is
then skipped whole; only its wide-window gather remains
(`stream_gather_windows`).

**This is the shape `streaming::process_channel_strip` has always had** —
`fused_blur_h_ssim` into band-private buffers, then `fused_vblur_features_ssim`
reading them in the *same* task — which is why buffered's tasks are
self-contained and the fold's were not.

It costs redundant blur at the band seams: 42 rows blurred per 32 consumed
against 148 per 128, i.e. **+40 % blur compute** — and it is faster anyway,
serially included.

Gate: `fold_self_blur_matches_precomputed_h` — every `V1BasicSums` field, 5
geometries (clamped top band, clamped bottom band, partial last band, 2304
wide, sub-strip plane) × serial/rayon, with an explicit NaN guard so the
comparison cannot be blinded.

### 2.6 `vec![0.0; n]` instead of `Vec::resize` for a fresh rolling plane

The "unaccounted" row in §1.1 — everything inside `foldapp_streaming_walk` that
is not the strip loop — held steady at **7.0–7.5 ms** through every lever above,
which is the signature of a fixed serial cost rather than anything the schedule
touches. It is the producer's rolling-plane allocation:
`RollingPlane::from_pooled` grows a buffer with `Vec::resize(need, 0.0)`, and
on the FIRST walk of a fresh `V2Scratch` the pool is empty, so that is a
reserve-plus-fill of ~32 MB at 2304² on every `Zensim::compute` call.

`vec![0.0; need]` lowers to `alloc_zeroed` (the `IsZero` specialisation covers
`f32`), which the kernel satisfies with demand-zero pages instead of a memset.
Every element is written before it is read either way — the zeroing was never
load-bearing — so this is the same allocation asked for in the form the
allocator can satisfy lazily. Only the empty-pool case changes; a recycled
buffer still takes the `resize` path and, being already large enough, does
nothing at all.

This is not a thread-scaling lever and it is not on the parallel critical path.
It is here because the profile put a number on a cost nobody had looked at.

### 2.7 The `mean_offset` side channel — the whole of the "unaccounted" row

§1.1's phase table left **7.0–7.5 ms** unaccounted at every thread count, and it
stayed there through all of stages 1–3 — the signature of a fixed serial cost.
It survived scratch reuse (`refinto_fold` and `score_fold` both reported
`unaccounted 7.1`), which ruled out allocation. It is
`MeanOffsetRows::add_strip_channel`: a full pass over the scale-0 planes for
all three channels — **127 MB read per walk at 2304²** — running between
`next_strip` and the channel fan-out, at degree 1, and never inside any span
the profile timed.

It parallelises for free. `rows[y][ch]` is **assigned**, not accumulated into,
by a pure function of that one row of `src`/`dst`; the only ordered arithmetic
is the left-to-right `f64` sum WITHIN a row, and that is never split.
`finish()` then reduces `rows` in its own fixed 64-row-chunk order regardless.
So a row-band fan-out cannot move a bit — and the three channels moved into one
pass so a band owns whole `[f64; 3]` elements rather than aliasing lanes of
them.

MEASURED at 2304², whole-loop wall over 40 compares:

| arm | 1T | 8T | 16T |
|---|---:|---:|---:|
| `refinto_fold` before | 185.5 | 61.5 | 53.8 |
| `refinto_fold` after | **182.3** | **50.8** | **51.0** |
| `score_fold` before | 210.0 | 65.3 | 54.0 |
| `score_fold` after | **203.8** | **57.8** | 55.8 |

The mean-offset phase itself goes **7.00 → 2.38 ms (8T) / 2.32 ms (16T)**, and
the profile now accounts for **100.0 %** of the walk (`unaccounted 0.013 ms`)
instead of 86 %. The 8-thread gain is the larger one, and §4 says why: 8T was
the configuration with ~18 % of scheduling headroom left, and this was it.

Gate: `mean_offset_row_bands_are_bit_exact` — serial vs parallel on every
`rows[y][ch]` AND on `finish()`, 5 geometries including heights that straddle
the band size and a multi-strip walk. `assert_result_bit_identical` in
`fold_engine_parity` independently compares `mean_offset` bit-exactly at rayon
pools 1/2/3/8/16 over 22 geometries.

### 2.8 Candidates the profile ruled out — and the bound that rules them out

The brief listed more candidates than shipped here. Each was priced against
§4's measured bound rather than against intuition, and the bound is what
retires them:

* **A flat `(channel, band)` fan-out instead of the nested one.** The nested
  shape (3 channels, bands inside) tops out at 12 tasks per strip at scale 0.
  Flattening would give rayon one 12-element iterator instead of 3 × 4 — a
  better shape in principle. It cannot pay: §4 shows the threaded run is
  already within a few percent of what N *independent processes* achieve, and
  no arrangement of tasks inside one process can beat what N processes with
  nothing shared between them do.
* **Pipelining the producer** (produce strip N+1 while consuming N). The
  producer is still 22–24 % of the 16-thread wall and fully serial, so this
  looks like the biggest remaining item — and it is the one the predecessor
  named as "the next axis, not attempted here" (§10.2). Two reasons it is not
  attempted here either. **(a)** It is not implementable without `unsafe`: the
  producer appends rows to a `RollingPlane` while consumers hold `&[f32]` into
  the same `Vec`, and the crate is `#![forbid(unsafe_code)]`. Making it safe
  means restructuring `RollingPlane` into per-row or per-band owned buffers —
  a redesign of the fold's memory shape, which is its product advantage.
  **(b)** More decisively, the bound already covers it: the producer's work is
  *inside* the N-process ceiling, so perfectly overlapping it cannot take the
  walk below `1 / T_N`. The total headroom left for ANY scheduling change is
  the gap in §4's last table — a few percent — not the 22 % the producer's
  wall share suggests.
* **Widening the band decomposition** (more, smaller bands). `V1_BAND_ROWS` is
  the v1 accumulation grouping: bands merge in band order into one running
  `f64`, so a different band height is a different sum. Semantics, not a knob.
  `STRIP_ROWS` is more subtle — the v1 band sequence is invariant to it (bands
  stay 32-row aligned and merge in ascending order either way) — but it is
  global, and the v2-era accumulators DO fold per strip, so raising it for the
  944 walk would move 944 bytes. A `v1_only`-specific strip height is possible
  and is left registered, not built: with self-blur bands its only effect is
  task count, and task count is not what binds (see the first bullet).
* **Per-thread scratch pooling for the plain `compute` entry.** The ref-loop
  entry is fixed (§7) and §2.6 removes the memset half of the cost; what is
  left is ~5 ms of page-fault commit, which the `score_fold` − `refinto_fold`
  gap in §2.7's table prices directly (203.8 vs 182.3 at 1T). `Zensim::compute` still allocates its `V2Scratch` per
  call (the ~7 ms "unaccounted" row in §1.1) — but so does the buffered walk
  allocate its pyramids per call, so the comparison in §5 is like-for-like, and
  removing it needs interior mutability on a `&self` method. Registered.

---

## 3. The progression, 2304², `score_fold` walk time

`ZENSIM_FOLD_TIMING`, 12 walks per point, under `run-heavy` (nice −19),
box load 0.2–2.5 throughout. This is the exploratory instrument — run-to-run
spread on this box is a few percent, so read the column, not the last digit.
§5 is the paired/interleaved zenbench table.

**Each row is a matched pair with the row below it** — measured immediately
before and after that change, same binary options, same box state. Rows are NOT
all from one session, and the instrument itself grew a band-level busy counter
between the `fused` and `ADVANCE_ROWS` rows (the `ADVANCE_ROWS` change first
measured 66.4 / 60.7 on the older instrument and 65.5 / 59.0 on the newer one;
the table quotes the newer, which is the pair the self-blur row is against).
Do not subtract non-adjacent rows.

| step | 1T | 8T | 16T |
|---|---:|---:|---:|
| main at lane start | 215.6 | 98.5 | 100.8 |
| + H-blur row bands (2.1) | — | 80.2 | 82.5 |
| + producer two-sided + 6-way cascade (2.2) | — | 68.8 | 69.2 |
| + fused per-channel fan-out (2.4) | — | 69.1 | 66.5 |
| + `ADVANCE_ROWS` 256 (2.3) | 214.3 | 65.5 | 59.0 |
| + self-blur bands (2.5) | 203.7 | 54.0 | 59.0 |
| + parallel `mean_offset` + calloc (2.6, 2.7) | **199.9** | **53.1** | **50.0** |

Two observations worth keeping:

* **Phase-A busy FELL** when it was split — 90.2 → 46.2 ms at 16T. A 16-row
  band's four output planes fit L2, so the split is cheaper in total work, not
  merely spread wider. A lever that reduces work while adding parallelism is
  rare enough to name.
* **Self-blur is faster at 1 thread too** (214.3 → 203.7, −5 %), despite +40 %
  blur compute. Locality beat arithmetic.

Other sizes, self-blur A/B only, same instrument, 30 walks per point:

| size | 8T before self-blur | 8T after | 16T before | 16T after |
|---|---:|---:|---:|---:|
| 576² | 3.67 | **2.87** | 2.94 | **2.85** |
| 1152² | 13.20 | **12.01** | 14.59 | **12.44** |
| 2304² | 65.5 | **54.0** | 59.0 | 59.8 |

Better at every cell except 2304²/16T, where it is a wash inside this
instrument's spread. (The 2304²/16T self-blur row reads 59.0 → 59.8 in the pair
it was measured against and 59.8 → 50.0 once §2.7 landed on top; the §5 zenbench
table is what the two changes are worth together.)

**And the last row is where the profile paid off twice.** The `unaccounted`
column held 7.0–7.5 ms flat through every row above it — the instrument could
not see inside it, and that constancy under four different schedule changes is
what identified it as a serial pass rather than anything parallel. §2.7 names
it; after that change the profile accounts for **100.0 %** of the walk
(`unaccounted 0.013 ms`).

---

## 4. The ceiling — measured, and it is not the schedule

The obvious next question after §3 is "why not 16×?", and the honest way to
answer it is to remove the scheduler from the experiment entirely: run **N
independent single-threaded processes**, each doing the whole walk, and read
aggregate throughput. No shared rayon pool, no fan-out, no barriers. Scratch
reuse is on (`refinto_*`) so per-call allocation is not the confound.

2304², walks/s, box otherwise quiet:

| N processes | `refinto_fold` | ×  | `refinto_buffered` | × |
|---|---:|---:|---:|---:|
| 1 | 5.42 | 1.00 | 5.78 | 1.00 |
| 4 | 16.09 | 2.97 | 21.39 | 3.70 |
| 8 | 19.19 | **3.54** | 38.76 | 6.71 |
| 16 | 18.32 | 3.38 | 62.95 | **10.89** |

**The fold's WORK saturates a shared machine resource at ~8 concurrent
instances. Buffered's does not — from the same serial speed.** Whatever that
resource is, sixteen copies of the fold with nothing shared between them cannot
exceed 3.6×, so no scheduling change inside one process can either.

**Reading the ceiling as a latency bound:** if the machine can complete `T_N`
walks/s with N independent copies, then one walk spread over N threads cannot
beat `1 / T_N` seconds however it is scheduled — the resource is the same. So
the fold's best possible latency here is **52.1 ms at 8 threads** and
**55.4 ms at 16**.

Against that bound, measured on the SAME arm and the SAME binary
(`refinto_fold`, scratch reused, whole-loop wall over 40 compares, so the
ceiling test and the threaded run price identical work):

| | achieved (threaded) | bound (N processes) | of bound |
|---|---:|---:|---:|
| 1 thread | 180.0 ms | 184.4 ms | 102 % |
| 8 threads | 55.3 ms | 52.1 ms | **94 %** |
| 16 threads | 50.5 ms | 54.6 ms | **108 %** |

**The implementation is AT the bound at both thread counts** — one process
using 8 or 16 threads now completes a compare as fast as 8 or 16 processes with
nothing shared between them manage per walk. There is no scheduling left to
find.

The 8-thread row is the one this bound earned its keep on. Measured BEFORE
§2.7 it read **61.5 ms against a 50.5 ms bound — 82 %** — and that 18 % was the
pointer that sent the lane back to the profile and found the serial
`mean_offset` pass. A bound is only useful if you act on the gap; this is the
gap it found, and closing it is §2.7.

The rest of the gap to buffered is a property of **what the fold computes and
how it touches memory** — buffered's own bound is 16.5 ms/walk at n=16, three
times better — not of how the fold is scheduled. That is the honest end of a
thread-scaling lane: the remaining lever class is traffic reduction (§2.5 was
one, and it moved the walk without moving the bound), not parallelism.

Two candidate mechanisms, neither isolated here and both recorded rather than
claimed: (a) the fold's rolling planes are an L3-sized hot set that degrades
under sharing where buffered's whole-image pyramids already stream from DRAM at
higher arithmetic intensity per byte; (b) the producer's `RollingPlane`
compaction `copy_within` fires on nearly every `produce()` call. (b) was tested
directly — see §6.

---

## 5. Final table — `fold_engine_bench`, paired and interleaved

`zensim/benches/fold_engine_bench.rs` under zenbench: all arms interleaved in
ONE process so shared-box noise cancels, `min_rounds 25 / max_rounds 200 /
max_wall 600 s` per group, one process per thread count
(`RAYON_NUM_THREADS`). The **before** column is the predecessor's §10 table,
which is the same bench, the same arms and the same deterministic generator on
the same box.

**Load conditions, stated because they matter:** the three runs were serialised
by zenbench's exclusive lock and ran back to back, unattended, 03:22→04:24 UTC,
with box load 0.2–2.5 (one unrelated single-core process, no other lane
benching). `cv` is 5–18 % on every cell reported here — a few percent of
run-to-run spread — so quote the ratios and the scaling factors, not the third
digit of a single cell. Raw zenbench outputs are committed beside this note
under `fold_mt_2026-08-31/` (`final_{1,8,16}t.txt`), with the ceiling/RSS
measurements in `postbench_2304.log`.

All three runs are on the SHIPPED code (`457ec709`, §2.7 included). §5.1
additionally reports an independent earlier run at 16 threads on the stage-3
commit `4fb56e04`, which reproduces the same conclusion from a different
build — a useful check that the headline is not one lucky run.

| size | arm | 1T | 8T | 16T | 1→8 | 1→16 |
|---|---|---:|---:|---:|---:|---:|
| 1152² | `score_buffered` | **48.59** | **10.73** | **8.97** | 4.53× | 5.42× |
| 1152² | `score_fold` | **50.39** | **14.74** | **14.45** | 3.42× | 3.49× |
| 1152² | `feat_buffered` | 48.90 | 9.51 | 7.56 | 5.14× | 6.47× |
| 1152² | `feat_fold` | 48.18 | 14.22 | 14.08 | 3.39× | 3.42× |
| 1152² | `ref_buffered` | 41.06 | 8.48 | 6.48 | 4.84× | 6.34× |
| 1152² | `ref_fold` | 45.93 | 14.35 | 14.17 | 3.20× | 3.24× |
| 1152² | `refinto_buffered` | 42.32 | 10.00 | 7.87 | 4.23× | 5.38× |
| 1152² | `refinto_fold` | 43.96 | 13.29 | 13.04 | 3.31× | 3.37× |
| 1152² | `fused_buffered` | 66.55 | 26.62 | 24.16 | 2.50× | 2.75× |
| 1152² | `split_fold` | 144.54 | 71.80 | 71.75 | 2.01× | 2.01× |
| 2304² | `score_buffered` | **210.00** | **40.86** | **32.25** | 5.14× | 6.51× |
| 2304² | `score_fold` | **201.95** | **59.67** | **54.47** | 3.38× | 3.71× |
| 2304² | `feat_buffered` | 206.34 | 36.72 | 27.22 | 5.62× | 7.58× |
| 2304² | `feat_fold` | 199.20 | 60.65 | 54.42 | 3.28× | 3.66× |
| 2304² | `ref_buffered` | 182.53 | 33.97 | 27.78 | 5.37× | 6.57× |
| 2304² | `ref_fold` | 190.37 | 59.42 | 52.92 | 3.20× | 3.60× |
| 2304² | `refinto_buffered` | 180.41 | 38.42 | 30.32 | 4.70× | 5.95× |
| 2304² | `refinto_fold` | 187.41 | 57.94 | 50.20 | 3.23× | 3.73× |
| 2304² | `fused_buffered` | 279.86 | 103.00 | 98.63 | 2.72× | 2.84× |
| 2304² | `split_fold` | 556.15 | 277.54 | 269.27 | 2.00× | 2.07× |

**The ratio this lane exists to move** — `score_fold ÷ score_buffered`, against the predecessor's §10 numbers on the same bench, arms, generator and box:

| size / threads | before | after | buffered scaling | fold scaling |
|---|---:|---:|---:|---:|
| 1152² / 1T | 1.03× | **1.04×** | 1.00× → 1.00× | 1.00× → **1.00×** |
| 1152² / 8T | 2.30× | **1.37×** | 4.60× → 4.53× | 2.06× → **3.42×** |
| 1152² / 16T | 2.93× | **1.61×** | 5.51× → 5.42× | 1.94× → **3.49×** |
| 2304² / 1T | 1.03× | **0.96×** | 1.00× → 1.00× | 1.00× → **1.00×** |
| 2304² / 8T | 2.54× | **1.46×** | 4.61× → 5.14× | 1.87× → **3.38×** |
| 2304² / 16T | 3.25× | **1.69×** | 6.15× → 6.51× | 1.95× → **3.71×** |

One asymmetry worth naming rather than glossing: on buffered,
`score_buffered − feat_buffered` is 7.5 ms at 2304²/16T, while on the fold the
same difference is 0.4 ms. Both arms compute the same 372 features and the
shared scoring tail is the same code, so that gap is a property of the two
buffered ENTRIES (`compute` vs `compute_extended_features`), not of this lane —
it is present unchanged in the predecessor's table (33.4 vs 28.1). It is
flagged here because it makes `feat_fold ÷ feat_buffered` look worse than
`score_fold ÷ score_buffered` for reasons that have nothing to do with the fold.

### 5.1 Cross-check: the same bench on the stage-3 commit (`4fb56e04`), 16 threads

Recorded because it is a second, independently-run confirmation of the headline
from a build that predates §2.6. Means, `cv` 8–17 % on every cell:

| arm | 1152² | 2304² |
|---|---:|---:|
| `score_buffered` | 9.15 | 35.28 |
| `score_fold` | **16.99** | **64.60** |
| `feat_buffered` | 7.80 | 28.34 |
| `feat_fold` | 16.72 | 62.68 |
| `ref_buffered` | 6.88 | 27.04 |
| `ref_fold` | 16.58 | 61.31 |
| `refinto_buffered` | 8.29 | 33.04 |
| `refinto_fold` | 16.04 | 60.97 |
| `fused_buffered` | 24.28 | 97.88 |
| `split_fold` | 72.70 | 283.27 |

`score_fold ÷ score_buffered` = **1.86× at 1152², 1.83× at 2304²**, against the
predecessor's **2.93× / 3.25×** on the same bench and box.

### 5.2 Peak RSS per config

`/usr/bin/time -v`, one arm per process, 20 compares each. The loop holds both
input images (7.96 MB at 1152², 31.85 MB at 2304²) in every arm.

The **working set** column subtracts that constant, which is what the
predecessor's §10.3 table reports — a peak-RSS ratio and a working-set ratio
are not the same number, and quoting one against the other is the mistake this
column exists to avoid.

| size | threads | `score_buffered` peak / w.set | `score_fold` peak / w.set | fold ÷ buffered (w.set) |
|---|---|---:|---:|---:|
| 1152² | 1 | 47.6 / 39.6 MB | 60.2 / 52.2 MB | 1.32× |
| 1152² | 16 | 60.8 / 52.8 MB | 80.8 / 72.8 MB | 1.38× |
| 2304² | 1 | 159.6 / 127.8 MB | 127.6 / 95.8 MB | **0.75×** |
| 2304² | 16 | 195.4 / 163.6 MB | 171.1 / 139.3 MB | **0.85×** |

The predecessor's crossover holds and this lane did not move it: buffered
scales with AREA (whole-image pyramids), the fold closer to WIDTH (rolling
planes), so the fold is the heavier path below ~1.5 MP and the lighter one
above it. `ADVANCE_ROWS` 256 (§2.3) and the band-local H planes (§2.5) both add
to the fold's footprint; against that, self-blur stops touching the strip-wide
H planes at all, so those pages are never faulted in.

Net at 2304² the fold's working set is **0.75× serially and 0.85× at 16
threads**. The predecessor's §10.3 measured **0.80×** for the 944 walk (and
0.62–0.63× for the narrower modes) at 8 threads, so serially this lane is
slightly AHEAD of that and at 16 threads slightly behind — **the lane's memory
cost is real and it has eaten part of the large-image margin, without crossing
it.** At 1152² the fold was already the heavier path (1.13× in the
predecessor's §6 numbers) and is now **1.32–1.38×**. If the fold's memory shape
is being defended as a product property, that 1152² row is the number to watch,
and it is why §6's `scale_capacity_rows` change — which would have added ~20 MB
more for ~2 ms — was rejected rather than spent.

---

## 6. Measured negatives, recorded rather than shipped

* **Lowering the conversion's parallel chunk height** — moves bytes (§2.3). The
  gate is committed so the next attempt is stopped in seconds instead of
  shipping a silent divergence.
* **Doubling `scale_capacity_rows`** so the rolling planes stop compacting on
  nearly every `produce()` call: producer 13.0 → 11.1 ms, walk **unchanged** at
  16T (49.89 vs 49.89 in a matched pair) and 1.5 ms better at 8T, for ~20 MB
  more RSS at 2304². The fold's memory shape is one of its product properties;
  it is not worth 2 ms. Not shipped.

---

## 7. The ref-loop gap the predecessor left open — closed

`fold_engine_2026-08-31.md` §10.4 measured the fold's ref-cache saving at
7–9 % against buffered's 14–20 % and named the likely cause. Confirmed and
fixed: `compute_fold_backed_with_ref` built a fresh `V2Scratch` per compare
(three `ScratchV2Strip` sets, 15 planes each — ~61 MB at 2304 wide — re-paying
the first-touch page-fault commit every call), **and `compute_with_ref_into`,
the one entry that exists to amortise work across compares, did not route to
the fold at all.**

Both fixed. The scratch lives in a new **private** field on the existing public
`ZensimScratch` — no new public type and no new public method, so the additive
surface this needed is zero. `compute_with_ref_into` now routes through the
shared `compare_against_ref_into` with the same reflect-pad rule
`compute_with_ref` applies, and out-of-domain requests still degrade to
buffered byte-for-byte.

Gate: `fold_ref_scratch_reuse_is_bit_identical` — ONE scratch reused across all
18 geometries × 4 candidates × serial/rayon, `to_bits` equal to both a
fresh-allocation `compute_with_ref` and a plain `compute`.

New bench arms `refinto_buffered` / `refinto_fold` price it.

---

## 8. Gates

Every commit in this lane ran, and all passed:

* `fold_engine_parity` — 11 tests, including
  `both_engines_are_bit_identical_across_rayon_pool_sizes` **widened** from 4
  hand-picked shapes to all 18 `CELLS` + 4 large ones × pools 1/2/3/8/16. Every
  lever here is a parallel-arm schedule change, so a band boundary interacting
  with a geometry is the failure mode, and only that sweep can see it.
* `v1_golden_bytes` — 5 tests, including `fold_backed_fixtures_match_golden`.
* `phase_a_blur_bands_are_bit_exact`, `fold_self_blur_matches_precomputed_h`,
  `convert_chunk_rows_is_semantics_not_a_knob`, `fold_ref_scratch_reuse_is_bit_identical`
  — the new gates.
* `mean_offset_row_bands_are_bit_exact` — serial vs parallel on every
  `rows[y][ch]` and on `finish()`.
* `cargo test --release -p zensim --features custom-profiles,feature-regime-v2,threads,training,classification` — **385/385**.
* `cargo clippy … --all-targets` — clean; `--no-default-features --features threads` builds.

**No byte moved at any point in this lane.** Where a lever would have moved
one, it was measured, reverted, and turned into a test.
