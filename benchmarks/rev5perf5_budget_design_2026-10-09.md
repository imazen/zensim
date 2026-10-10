# Rev5 queue byte accounting (2026-10-09)

The ordered Rev5 strip queue now limits live slots by a private byte budget,
thread count and the existing 16-slot ceiling. Checked size arithmetic refuses
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

The default is provisional until COSTCMP measures frozen 64/128/256 MiB builds
against a fresh uncapped build at 1024×1024, 4096×4096 and 8192×4096, with
8/16/32 threads on v4x. Each candidate requires strict 384/384 parity. Timing
requires 32 common admitted rounds under the existing 64-round ceiling and
quiet gate; peak RSS comes from separate fresh processes and time -v.
Recipes are in benchmarks/rev5perf5.just; raw evidence stays outside git.
