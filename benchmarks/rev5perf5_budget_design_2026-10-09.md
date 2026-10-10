# Rev5 queue byte accounting (2026-10-09)

The ordered Rev5 strip queue now limits live slots by a private byte budget,
thread count and the existing 16-slot ceiling:
`min(thread_count, 16, floor(budget / per_job_bytes))`.
Checked size arithmetic refuses
batching when no whole job fits, including overflow; the original route handles
that request. Kernels and producer-order merges are unchanged.

Per-job accounting includes eleven float planes at the maximum strip size,
the job slot, every fixed-size result vector and the pool scratch object.
The budget bounds owned queue payload. Allocator bookkeeping, input pixels,
parent streaming scratch and total process RSS are outside this accounting.
Before reuse, excess slots and planes cached for wider inputs are released.
The slot vector reserves exactly the admitted slot count. Growing a job drops
its old planes before allocating replacements.

The allocation test sums actual vector capacities after strided, partial-strip
and changing-size requests, including Off and Peaks attribution. Explicit
budgets cover zero-byte fallback through 256 MiB; exact-fit and overflow cases
are separate assertions. Native 8/16/32-thread outputs match serial feature bits.
All-feature tests, formatting, clippy and no-default-feature checks pass at the
initial 128 MiB candidate. Existing ignored tests and dependency warnings remain.

The 128 MiB default is provisional: speed not yet measured. COSTCMP compares frozen 64/128/256 MiB builds
against a fresh uncapped build at 1024×1024, 4096×4096 and 8192×4096, with
8/16/32 threads on v4x. Each candidate requires strict 384/384 parity. Timing
requires 32 common admitted rounds under the existing 64-round ceiling and
quiet gate; peak RSS comes from separate fresh processes and time -v.
Recipes are in benchmarks/rev5perf5.just; raw evidence stays outside git.

Qualification: all three frozen budget builds pass 384/384 strict score and
420-feature checks against the uncapped control, with 576/576 frozen
comparisons overall. Each passes native 8/16/32-thread invariance. All 36
requested measurement-grid preflight records also agree bit for bit, including
8192×4096. The extended report owner successfully replays the existing
REV5PERF4 evidence. On 2026-10-09 the coordinator instructed fresh-process RSS
under shared load, keeping all timing admission rules unchanged. All 36 RSS
observations are complete; timing remains queued. The measured table and actual
Rust per-job accounting are in
[the RSS record](rev5perf5_rss_2026-10-09.md). The selected source is exactly the
qualified 128 MiB snapshot, with no kernel or merge changes. The RSS-only report
contains no timing medians or confidence intervals.

## Slot counts by width, and the open speed risk below three slots

Computed from the accounting above (`per_job = 6,512 × width + 37,640` bytes,
scale-0 width after sampling); not a measurement:

| budget | ≥16 slots | ≥8 slots | ≥3 slots | ≥2 slots | ≥1 slot (else original route) |
|---|---|---|---|---|---|
| 64 MiB | width ≤ 638 | ≤ 1,282 | ≤ 3,429 | ≤ 5,146 | ≤ 10,299 |
| 128 MiB | ≤ 1,282 | ≤ 2,570 | ≤ 6,864 | ≤ 10,299 | ≤ 20,605 |
| 256 MiB | ≤ 2,570 | ≤ 5,146 | ≤ 13,734 | ≤ 20,605 | ≤ 41,215 |

Widths here are the scale-0 width the walk uses. Inputs narrower than 64 px
are reflect-padded to 64 first, so their per-job size uses 64.

**The floor (`REV5_MIN_JOB_SLOTS = 3`).** When fewer than three jobs fit, the walk
takes the original route, exactly as when none fit. Live slots are
`min(threads, 16, floor(budget / per_job))`, or 0 (original route) when that is
below 3. At 128 MiB, widths up to 6,864 queue with 3–16 slots, and widths from
6,865 (8192×4096 included) take the original route. Both routes pass strict
parity, so the floor changes scheduling, not bits.

**Rationale, corrected after review.** The first rationale said a 1–2-slot queue
runs fewer channel jobs at once than the original route's three. That
undercounts the original route: it runs the three channels in parallel
(`fuse_channels`) **and** each channel's v1 bands in parallel (`band_parallel`).
That is up to 3 × 4 band tasks per strip, while a queued job runs its bands
serially. A 3- or 4-slot queue is therefore not obviously faster than the
original route either. A diagnostic on a loaded box (review, 2026-10-09) pointed
the other way at 3 slots. The floor's value is decided by the timing below,
not by this argument.

**Floor grid (timing).** `rev5perf5-budget-timing` times six arms on 15 cells:
- **Arms:** uncapped; the frozen 64/128/256 MiB builds; and the 128 MiB source
  built with floor 3 (shipped) and with floor 17. Floor 17 never queues,
  because the queue has at most 16 slots, so it is the original route.
- **Inputs:** 1024², 4096², 4608×4096, 6144×4096 and 8192×4096 at 8, 16 and 32
  threads. At 128 MiB the queue has 8–16, 5, 4, 3 and 2 slots there.
- **Answer per cell:** frozen 128 MiB (the queue at that slot count) minus floor
  17 (the original route), paired on the same admitted rounds. Floor 3 minus
  floor 17 shows the shipped behaviour directly.
- **Gate:** unchanged (load1 < 2, no foreign build or training, the first 32 clean
  rounds from at most 64).
- **Inventory check:** the v3 inventory admits each floor build only if removing
  its exact floor hunk gives back the frozen 128 MiB source.
- **RSS:** fresh-process RSS of both floor arms covers all 15 cells
  (`rev5perf5-floor-rss`, quiet-gated). Records carry the worker's rayon pool size.

The frozen 64/128/256 MiB binaries measured for RSS predate the floor. Their 1-
and 2-slot cells (64 MiB at 4096² and 8192×4096, 128 MiB at 8192×4096) show
the queued route's memory, not the route the source takes now.
