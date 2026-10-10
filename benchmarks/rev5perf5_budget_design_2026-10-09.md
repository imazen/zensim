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

The original route (`fuse_channels`) runs a strip's three channels in
parallel. The queued route runs at most `slots` channel jobs at once, so with
one or two slots it has less v2 parallelism than the route it replaces. At
128 MiB that is widths 6,865–20,605, including the 8192×4096 grid geometry
(2 slots); at 64 MiB it is 3,430–10,299, including 4096² and 8192×4096
(2 and 1 slots). This follows from reading the code and is **not measured**.
The 4096² and 8192×4096 timing cells decide it; if they show a loss, the
candidate fix is to fall back to the original route below three slots, which
keeps bits by construction (both routes already pass strict parity).
