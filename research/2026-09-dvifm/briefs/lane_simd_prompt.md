# LANE `simd` — codebase-wide SIMD inlining audit (this may be worth more than the whole DVIFM programme)

`~/tmp/devin/LANE_PREAMBLE.md` binds you. Your own fair-cost investigation found the generic SIMD helper chain was
call-per-op with ZERO inline zmm in the production path, and `#[inline(always)]` on that chain took 1024 squared
from 52.4 ms to 32.4 ms — a 38 percent speedup with bit-exact parity. Find out how far that reaches.

## Audit
Every hot generic-over-token kernel in zensim, not just dvifm: `zensim/src/blur.rs` (downscale_2x_inner and the
blur kernels), `zensim/src/feature_v2.rs` and its fused walk kernels, `zensim/src/metric.rs` pixel kernels,
`zensim/src/feature_v2_stream.rs`, and any `#[rite]` or generic helper reached from an `#[arcane]` / `incant!`
entry. For each: disassemble the FEATURED tier entry as you did, count zmm/ymm occurrences, look for call-per-op
into the generic backend, and report a table of kernel by tier by inlined-or-not.

## Fix and measure
Where a kernel is not inlining, apply the same narrowest fix and measure: zenbench interleaved, at least 30 paired
rounds, 1 thread and 8, at 256/1024/2048 squared, before and after, with dispersion. This is a PRODUCTION speedup
claim, so hold it to the speed-matrix protocol: quiet box, pinned threads, no target-cpu=native, competing
processes recorded.

## Correctness gate — non-negotiable
Prove bit-exactness of every affected feature before claiming any speedup: to_bits equality on the golden and
parity fixtures and the existing byte-stability gates. Profiles B, D and the Rev3 ensembles must produce
byte-identical scores. If any output moves, STOP and report rather than shipping it.

## Deliverable
`benchmarks/simd_inlining_audit_2026-09-21.md` plus `.json`: the audit table, measured before/after per kernel and
per profile, and a clear statement of which kernels were already fine. Terminal file
`~/tmp/devin/LANE_SIMD_DONE.md`, progress `~/tmp/devin/lane_simd.log`. Same rules: heavy lock, jj only, no push,
never relax a test.
