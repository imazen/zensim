# All-944-live extraction (`V1PoolsMode::Full`) — perf, 2026-08-30

**What this measures.** The streaming folded-944 walk with v1's pool block
(`f156..372`) emitted LIVE — `Off` (the production default, structural zeros)
vs `Carriers` (the ten `fused944native` slots) vs `Full` (all 216) — on one
textured pair at 576² and 1152², serial, via the committed paired/interleaved
zenbench instrument `zensim/benches/fold_pools_bench.rs`.

**Why the Full path matters.** The 944 regime zeroes `f156..371` by policy, so
any model that wants v1's pool slots had to fuse them in from a separate
720/372-width table — and v1 pool features DIVERGE across widths. `Full` is the
one-pass, one-width extraction that removes the fusion; it is now reachable
from the fleet as `zenmetrics jobexec --metric zensim-foldapp2pools`
(zenmetrics `905ae73d`, regime tag `folded720append2pools`).

Run:
```
cargo bench --bench fold_pools_bench -p zensim \
  --features custom-profiles,feature-regime-v2
```
(no `--release` flag — `cargo bench` already builds the bench profile; no
`-C target-cpu=native`; every run under `~/work/zen/scripts/run-heavy
--mem 16G --jobs 8`.)

## MEASURED — before vs after

Both runs: the same committed instrument, the same `run-heavy --mem 16G
--jobs 8` caps, serial (`with_parallel(false)`), **20 usable rounds per arm**
in each group. The comparable quantity across two runs is the **paired
`vs base` ratio measured inside one interleaved process**, not the absolute
ms — the box was shared with three other agents and its load differed between
the runs (baseline `pools_zeroed` 23.6 ms, after 16.6 ms at 576²). zenbench's
paired bootstrap is what separates the shared machine noise from the arm
delta, which is why the 1152² `vs base` intervals are tight even where CV
reads 69-77 %.

`pools_carriers10` is the **untouched control** — `Carriers` stores no sigma,
so this lever cannot move it. It moved within its interval at both sizes,
which is the instrument saying it works.

| size | arm | BEFORE (95 % CI vs `pools_zeroed`) | AFTER (95 % CI vs `pools_zeroed`) |
|---|---|---|---|
| 576²  | `pools_zeroed`     | 23.61 ms (base)   | 16.62 ms (base)   |
| 576²  | `pools_carriers10` | +9.6 % – +21.6 %  | +11.6 % – +15.9 % |
| 576²  | **`pools_full216`**| **+18.0 % – +25.3 %** | **+16.1 % – +18.9 %** |
| 1152² | `pools_zeroed`     | 80.01 ms (base)   | 74.31 ms (base)   |
| 1152² | `pools_carriers10` | +10.4 % – +14.1 % | +12.9 % – +14.4 % |
| 1152² | **`pools_full216`**| **+21.9 % – +29.2 %** | **+17.7 % – +20.2 %** |

- **At 1152² the two `pools_full216` intervals are DISJOINT** (before ≥ 21.9 %,
  after ≤ 20.2 %) — the clean result. At 576² they touch only at their edges
  (before ≥ 18.0 %, after ≤ 18.9 %), with the after-interval entirely below the
  before-midpoint.
- **The Full-specific cost — `full` measured against `carriers`, both arms
  inside the same process, so machine load cancels exactly** — is where the
  lever shows up most directly:

  | size | BEFORE `full`/`carriers` | AFTER `full`/`carriers` |
  |---|---|---|
  | 576²  | 27.59 / 26.18 = **+5.4 %** | 19.53 / 18.60 = **+5.0 %** |
  | 1152² | 100.93 / 90.93 = **+11.0 %** | 88.95 / 85.12 = **+4.5 %** |

  The 1152² number is the one to quote: the marginal cost of emitting all 216
  pool slots instead of only the ten carriers **more than halved, +11.0 % →
  +4.5 %**. That is the two removed band-sized V-blur sweeps, and it scales
  with band area, which is why 1152² separates it and 576² (where the fixed
  per-band costs are a larger share) barely does.

**Correctness, same build:** `cargo test --release -p zensim --features
custom-profiles,feature-regime-v2,threads,training` — **322 passed, 0 failed**,
including `folded720_v1_pools_match_v1_path` (the `to_bits()` comparison of all
216 pool slots against the buffered v1 372 path). NOTE for whoever runs this
next: that gate is `#[cfg(feature = "training")]`, so the plain
`--features custom-profiles,feature-regime-v2` invocation **silently compiles
it out** — 206 lib tests pass and the gate never runs. Always include
`training` when the pool block is in scope.

## The lever: the sigma planes come out of the fused V-blur kernel

`fold_v1_basic_bands`'s `Full` arm used to run, per band per (scale, channel),
**two extra `box_blur_v_from_copy` sweeps** over the whole band buffer to
produce v1's `sigma1_sq` / `sigma12` for the masked+IW SSIM kernel:

```rust
box_blur_v_from_copy(&ssq_h[span], &mut ps.ssq_v[..band_n], width, h_local, R);
box_blur_v_from_copy(&s12_h[span], &mut ps.s12_v[..band_n], width, h_local, R);
```

But `fused::fused_vblur_features_ssim` — which the same arm already calls on
the same band, with the same inputs and the same radius — **already carries
those two V-blurred planes in registers** (`let ssq = sum_sq * inv_v; let s12 =
sum_s12 * inv_v;`), it just threw them away. It had `store_mu` / `store_sd`
side-outputs but no sigma one. Added `ssq_out` / `s12_out` / `store_sigma`
(the exact shape of `store_mu`, in all four SIMD variants × their 16-lane,
8-lane and scalar sections), and the `Full` arm now takes the side-output and
runs no V-blur of its own. Two band-sized read+write sweeps per (band, scale,
channel) removed, and the side-output writes only the INNER rows — the only
rows the masked/IW SSIM kernel reads — where the old sweeps wrote all
`h_local` rows (a 32-row band carries ±5 rows of overlap, so ~24% of those
writes were never read).

**No extra buffer.** The activity blur borrows `ps.ssq_v` as its scratch temp,
so the call order flipped: activity FIRST, fused kernel SECOND (which then
overwrites `ssq_v`'s inner rows with the real sigma). The reorder is
sum-neutral — the `sums` fields the activity block feeds (masked / IW) are
disjoint from the ones `accumulate` writes (basic).

**Why the values are bit-identical, not merely close.** Each column's V-blur is
an independent scalar recurrence (`sum += src[add] − src[rem]`, then
`sum * (1.0 / diam)`), so lane width cannot change a value and only the index
sequence can. Init (`mirror_idx`) and `rem_idx` are written identically in both
kernels; the one textual difference is the bottom-edge `add_idx` fold
(`|2·(h−1) − add_raw|` here vs `saturating_sub` there), which can only diverge
when `h < r + 2` (= 7 at `BLUR_RADIUS` 5). The folded walk reflect-pads to a
64px floor and runs 4 pyramid scales, so the smallest plane it ever V-blurs is
8 rows — the divergent branch is unreachable on this path. This is asserted,
not assumed: `folded720_v1_pools_match_v1_path` compares every one of v1's 216
pool slots to the buffered v1 path with `to_bits()` at SIMD-exact widths.

## What was NOT done, and why (so the next session doesn't re-derive it)

The registered next lever was phrased as "weighted art-L4 sums inside the fused
V-blur kernel, so the masked/IW pools stop paying a separate sweep". The
sigma-side of that shipped (above). The **weighted-sum side cannot ship without
breaking the bit-exact gate**, and the reason is arithmetic, not effort:

- `fused_vblur_ssim_inner` iterates **column-group-major** (`for cg { for y }`)
  and folds each 16-lane `reduce_add()` into an `f64` accumulator in that order.
- `simd_ops::{edge_diff_channel_inline_both, ssim_channel_inline_both,
  build_inline_mse}` iterate the flat inner slice **row-major** in 16-lane
  chunks and accumulate in *that* order.
- Both v1 and the fold call the same `simd_ops::*` functions, which is exactly
  why the fold's pool slots are bit-identical to v1's today. Moving that math
  into the fused loop keeps the per-lane values identical but changes the `f64`
  summation ORDER, and `f64` addition is not associative — so the pooled sums
  would drift in the last ulps and `folded720_v1_pools_match_v1_path` would
  fail. Relaxing that gate is not on the table: it is the only thing that makes
  a live-pool 944 row substitutable for a v1 372 row.

Everything the fused loop already computes (`ed`, `a4`, `dl4`) is therefore
recomputed by `edge_diff_channel_inline_both` on purpose. The remaining
bit-exact levers, in rough order of expected value:

1. **`box_blur_h_of_abs_diff`** — the activity block is `abs_diff_into(src,
   mu1_h → act_raw)` then `box_blur_1pass_into(act_raw → act)`, i.e. a
   band-sized intermediate written and immediately re-read. A H-blur kernel
   that takes `|a − b|` at its load sites removes that round-trip
   (~2 band-sized ops of ~7 in the activity block). Bit-exact by construction
   — same values, never materialized. Costs a new SIMD kernel in all four
   variants, because `box_blur_h_inner_*` is **row-group-major** (it vectorizes
   ACROSS 16 rows, gathering one column at a time), so the cheap trick of
   calling the existing kernel one row at a time falls into its scalar
   remainder and is slower, not faster.
2. **Inner-rows-only V-blur write** in `box_blur_1pass_into`'s V half: the band
   is `V1_BAND_ROWS + 2·V1_BAND_OVERLAP` rows but only the inner
   `V1_BAND_ROWS` are read, so ~24% of that pass's stores are dead.
3. The per-band ±overlap recompute of the activity itself is **not** removable:
   v1 mirror-clamps the activity blur at its own strip edges, and reproducing
   that boundary behaviour is what makes the fold v1-exact.

