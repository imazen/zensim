# Blur radius, plane locality, and branch mispredictions — three questions, measured

**The user's three questions, in their own terms:** *(a) can the blur radius be
smaller? (b) is there a more LOCAL solution for the six shared planes? (c) are
trailing/edge pixels causing branch mispredictions?*

All three are answered with measurement. Two of them (a and c) are answered
here from scratch; the third (b) had its *specified prototype* falsified by two
other lanes while this one was running, so what this note adds instead is the
coupling nobody had priced — and then measures it:

> **The halo closure that decides every locality experiment is
> `±2·BLUR_RADIUS` in rows and `±BLUR_RADIUS` in columns, so (a) and (b) are
> the same knob seen twice.** Every row-shape falsification in this repo is a
> statement about *radius 5* — and at radius 2 the sign of one of them flips
> (§4.4), with the winning cell running **−7.6 % of wall at −37.6 % of peak
> RSS** against the shipped configuration.

The radius is not just a cost; it is the currency every tiling and banding
shape spends.

Predecessors, all read before measuring:
[`v2_block_cost_2026-08-31.md`](v2_block_cost_2026-08-31.md) (§2.2 the H-plane
shape row, §5 the four falsified levers, §7 the retarget),
[`fold_footprint_2026-08-31.md`](fold_footprint_2026-08-31.md) (the closed-form
footprint model, the 9950X3D asymmetric-L3 facts),
[`era2_perf_break_2026-08-31.md`](era2_perf_break_2026-08-31.md) (§14.4 pass
structure is not semantics, §21.1 the registered utility-preservation bar,
§22 band-local phase A FALSIFIED, §22.5 the ASLR measurement protocol,
§23 column tiling).

---

## 0. The three answers in one paragraph each

**(c) Branch mispredictions are NEGLIGIBLE, and the row-tail hypothesis is
falsified.** Nobody had measured branch behaviour at all; this note does, with
hardware counters. Across the 944 walk, the 372 walk and the buffered control,
at 576²/1152²/2304² and at widths that are and are not multiples of 8 and 16,
the misprediction rate at one thread is **0.015–0.050 %** and the whole
misprediction budget — every miss, charged 20 cycles, with no overlap credit —
is **0.14–0.50 % of cycles**. Moving from a 16-aligned width (2304) to the
worst tail class (2303, seven lanes of tail) costs **+0.06 % of cycles**.
The rate is size-correlated, but *inversely* — it FALLS as the image grows
(0.043 % at 576² to 0.028 % at 2304²) — and the only regime anywhere near
1 % is **576²/16T at 1.2–1.65 % of cycles**, where the whole walk is 5 ms and
**48.6 % of the misses are `crossbeam_epoch`**, i.e. the rayon runtime, not a
kernel. There is no fix worth making, and the negative is worth as much as a
positive would have been: it removes a whole class of speculative work from
era-2's list. §1.

**(a) Yes, and the axis is real but SMALL — and the one radius that clears the
quality bar is the one that buys almost no time.** Almost none of the win comes
from the blur arithmetic, because the box blur is a running sum and therefore
**O(1) per pixel at any radius**. What radius buys is the **halo**:
`HALO_P = 2·BLUR_RADIUS`, so the strip's wide window is `128 + 4·R` rows, and
dropping R from 5 to 2 removes 12 of 148 rows from every plane pass. Under the
full era-2 estimator, hardened with **two independent builds per radius**
because a single build's layout is worth as much as the effect: **radius 4 is
+0.68 % / −0.17 % (2304², 1T/16T) — inside the floor — while radius 3 and 2 buy
−4.4 % to −7.1 %.** Peak RSS is the layout-immune column and it is monotone
everywhere: **−1.35 / −2.90 / −4.12 %** at 2304²/1T. §2.

**QUALITY is the other axis, and it is where the answer turns.** Against era-2
§21.1's registered utility-preservation bar (no corpus loses more
than 0.005 SROCC, composite does not fall), scored on 20,516 pairs re-extracted
at each radius: **radius 4 PASSES for the shipped 944 flagship (Profile C) —
worst corpus −0.0007, composite +0.0038** — and fails for the other three
models on exactly one corpus each while their composites also rise. Every
model, at every radius, moves the same way: it **gains** on the human-MOS codec
corpora (cid22, aic3/aic4) and hugely on **KonJND**, the near-threshold anchor
(Profile C +0.089 at radius 2), and **loses** on TID and KADID, whose
distortion sets are blur and noise at scales a wide support is built to see. A
smaller radius is a redistribution toward the axes this metric is weakest on,
not a degradation. §3.

**(b) The local shape the hand-off specified was falsified — twice, by other
lanes, while this one was running** — and the axis that won is the *other* one.
Band-local phase A is measured-and-falsified at every band height
(era-2 §22); the rolling row window's load-bearing component (row-major
running-sum V blur) was measured at +9 % and reverted (v2-block L3). Column
tiling *did* win at 1T (1.23×@5 MP, 1.85×@21 MP) and is behind
`ZENSIM_H_TILE` / `ZENSIM_A_TILE`, both DEFAULT-OFF because era-2 then measured
its sign inverting at 8/16 threads below ~4000 px. This note does not re-run a falsified experiment. What it
contributes is the **coupling those falsifications did not price**: the halo
closure is linear in the radius on both axes, so both row-shape falsifications
are statements about *radius 5*. MEASURED (§4.4), on the one row-shape knob
that is still a live constant: at radius 5 a 32-row strip is **+12.0 %** against
the shipped 128 — reproducing the v2-block lane's falsification — and at radius
2 the same cell is **−4.7 %**. **The sign flips.** The best cell measured,
`radius 2 × STRIP_ROWS 32`, runs 2304²/1T in **301.8 ms against the shipped
configuration's 326.5 (−7.6 %) at 61.0 MB peak RSS against 97.6 (−37.6 %)** —
which is the locality prize the hand-off wanted, reached through the radius
rather than through a new plane shape. It is not shippable as it stands (radius
2 fails the quality bar; `STRIP_ROWS` is not byte-neutral), and that is the
point: **the plane pipeline CAN be more local, and what unlocks it is (a).**
§4.4.1 reconciles this with era-2 §24.3, which reached the opposite conclusion
from the same closure — the break-even is one ratio away, and era-2's own
1.25× per-unit figure predicts both the +12.5 % it measured at radius 5 and the
−5.9 % measured here at radius 2.

---

## 1. (c) Branch behaviour — the first hardware-counter measurement of these kernels

### 1.1 Instrument, and the one thing it cannot do

`perf_event_paranoid` was **4** on this box, which blocks every event. It was
lowered to **1** for the duration of this lane (user-space counters on
self-owned processes) and **restored to 4 afterwards**. `/home/lilith/bin/perf`
is a stale hand-built binary linked against a `libpython3.10` that is no longer
installed; **`/usr/bin/perf` (the packaged `linux-perf` 7.0.12) is the working
one** and is what every number below used.

Workload: `zensim/benches/extract_paths_bench`'s single-arm `ZEN_XP_RSS` loop,
which this lane extended with **`ZEN_XP_W` / `ZEN_XP_H`** — it previously took
only a square `ZEN_XP_SIZE`, so the width class the question is about was not
reachable at all. Arms: `fold944_full` (the product walk), `fold372_full`
(v1-only), `buf_v1_372` (the buffered control). Each cell subtracts an
`ITERS=0` process baseline, so process start-up, the `test_pair` generator and
teardown are removed rather than assumed small.

**What it cannot do:** precise (PEBS-class) sampling. `branch-misses:upp` and
`:uppp` both come back *"PMU Hardware doesn't support sampling/overflow-
interrupts"* on this virtualised host, so `perf record` attribution carries
skid — the recorded IP lands a few instructions past the branch. Aggregate
`perf stat` counts are exact; the per-symbol split in §1.3 is a
skid-limited attribution and is labelled as one.

### 1.2 The counters — rate, and the cycle budget it can possibly cost

Full matrix: [`blur_radius_locality_branches_2026-08-31/branch_matrix.tsv`](blur_radius_locality_branches_2026-08-31/branch_matrix.tsv)
(3 arms × 3 square sizes × 4 width classes × {1,16}T, with a per-arm process
baseline and the 1-minute load at each cell).

**One thread.** `branch-misses / branches`, and the cycle penalty of *every*
miss at 20 cycles as a fraction of the cell's cycles:

| arm | 576² | 1152² | 2304² | 2296×2304 | 2303×2304 | 2297×2304 |
|---|---:|---:|---:|---:|---:|---:|
| `fold944_full` rate | 0.043 % | 0.037 % | **0.028 %** | 0.030 % | 0.032 % | 0.030 % |
| `fold944_full` cycle cost | 0.44 % | 0.33 % | **0.18 %** | 0.23 % | 0.25 % | 0.24 % |
| `fold372_full` rate | 0.050 % | 0.032 % | 0.019 % | 0.020 % | 0.020 % | 0.017 % |
| `fold372_full` cycle cost | 0.50 % | 0.29 % | 0.15 % | 0.17 % | 0.16 % | 0.14 % |
| `buf_v1_372` rate | 0.042 % | 0.022 % | 0.015 % | 0.018 % | 0.021 % | 0.021 % |
| `buf_v1_372` cycle cost | 0.44 % | 0.21 % | 0.14 % | 0.17 % | 0.19 % | 0.19 % |

IPC at 1T is **2.5–3.9**. A front-end-bound loop does not run at IPC 3.

**Width class — the actual question.** 2304 is a multiple of 16 (and of 8);
2296 is a multiple of 8 and not of 16; 2303 ≡ 7 (mod 8), the widest possible
scalar tail; 2297 ≡ 1 (mod 8), the narrowest. On `fold944_full` at 1T the
worst tail class costs **+0.061 percentage points of cycles** over the aligned
width (0.184 → 0.245 %). At 16T the four classes are **within each other's
noise** (0.58 / 0.66 / 0.66 / 0.69 %). At 256² vs 257² the miss *count* moves
by 7 % of a number that is already 0.07 % of branches.

**Sixteen threads.** Rates rise to 0.15–0.29 % and the cycle budget to
0.25–1.65 %, worst at 576²/16T where the walk is 5 ms and there is almost no
work per thread. That growth is **not** the kernels — see §1.3.

**Small images, where edge handling is maximal.** A 64×64 pair runs a pyramid
whose deepest scale is 8×8, so the reflect-clamped rows are a large fraction of
every plane. [`branch_small.tsv`](blur_radius_locality_branches_2026-08-31/branch_small.tsv):
rate falls monotonically 0.160 % (64²) → 0.141 (96²) → 0.111 (128²) → 0.090
(192²) → 0.072 (256²) → 0.061 (384²), and the odd-width twins move it by
+0.004 pp (129² 0.115 vs 128² 0.111) and +0.005 pp (257² 0.077 vs 256² 0.072).
(That sweep does **not** subtract a process baseline; it does not need to —
the baseline is ~29 M branches against 3–8 G in every cell.) At 64² the
20-cycle budget is **1.66 % of cycles**, the highest anywhere in this note, and
§1.3 shows where it goes.

### 1.3 Where the remaining misses are (skid-limited attribution)

`perf record -e branch-misses:u -c 2000`, `fold944_full`, 1T. Full output:
[`branch_miss_attribution.txt`](blur_radius_locality_branches_2026-08-31/branch_miss_attribution.txt).
Each record collects 600–800 samples, so a share here carries a few percentage
points of sampling noise on top of the skid — read the ORDERING and the top
entry, not the third decimal (the `dense` row's 6.2 → 14.6 % between two
adjacent width classes is that noise, not a width effect).

| symbol | 2304×2304 | 2303×2304 | 64×64 |
|---|---:|---:|---:|
| `box_blur_v_copy_inner_v4x` | **37.5 %** | **34.8 %** | 14.0 % |
| `stream_phase_b` | 9.0 % | 8.0 % | 2.4 % |
| `downscale_2x_into_inner_v4x` | 7.4 % | 6.6 % | 7.3 % |
| `dense_block_kernel_entry_v4x` | 6.2 % | 14.6 % | — |
| `crossbeam_epoch::…::with_handle` | 4.2 % | 2.8 % | **28.6 %** |
| `box_blur_h_inner_v4x` | 4.1 % | 2.4 % | 7.4 % |

The third candidate site the question named — **band / strip boundary halo
handling** — is `stream_phase_b` plus `foldapp_streaming_walk`, and it is
**9.0 % of 1T misses** at 2304², i.e. **0.017 % of cycles**. It is a real
entry in the table and it is two orders of magnitude too small to act on.

Three readings, and the second is the one that answers the question:

1. **The hypothesis was right about the location.** The single largest
   misprediction source *is* the V blur, whose per-row body computes the
   reflect-mirror `add_idx` / `rem_idx`. So blur edge clamping is exactly where
   the misses live — it is simply that all of them together are 0.18 % of
   cycles.
2. **It was wrong about the mechanism.** If row tails drove mispredicts, the
   *misaligned* width would concentrate them in the tail-handling kernels. It
   does the opposite: the V blur's share **falls** 37.5 → 34.8 % at width 2303
   while the total rises only 0.028 → 0.032 %. **And the compiler has already
   made the clamping branchless.** Disassembling the hot loop, the whole
   reflect-mirror index computation — `add_idx`, `rem_idx`, both `.min(height-1)`
   clamps — is `cmovae` / `cmovb` / `cmovs`, with no conditional jump anywhere
   in it. The only jumps in the loop body are two never-taken slice-length
   guards (`ja` into the function's cold tail) and the back-edge; the 33.6 %
   sample sitting on a `vsubps` load is skid, not a branch. There is no
   branchy edge handling left to make branchless.
3. **The 16-thread and tiny-image growth is the parallel runtime, not the
   kernels — measured directly, not inferred.** At **2304²/16T** (a separate
   record, CCD1-pinned so it did not disturb the concurrent cost sweep on
   CCD0) `crossbeam_epoch::…::with_handle` alone is **48.6 %** of all branch
   misses, `Global::try_advance` adds 4.8 %, and `rayon_core`'s
   `sleep`/`find_work`/`wake_specific_thread` another 5.3 % — **~58.6 % of the
   16-thread misprediction budget is the scheduler**, against the V blur's
   11.4 %. The same shape appears at the other end: at 64×64/1T
   `crossbeam_epoch` is 28.6 % against the V blur's 14.0 %, because per call
   the walk is a few hundred microseconds and the runtime is a fixed cost.
   Both elevated regimes — many threads, and tiny images — are the same
   effect, and neither is edge handling.

### 1.4 Disposition — closed, with nothing shipped

**Nothing was changed, and that is the result.** Branchless edge handling,
masked tails, table-driven clamping and loop peeling for the first/last radius
columns were all on the table; the measurement retires all four. The ceiling on
the entire class is **0.5 % at one thread** and it is **0.06 %** for the
specific row-tail hypothesis. That is below the ASLR noise floor this repo has
just measured at 10 % for a single-process 2304² number (era-2 §22.5) — i.e.
the fix is two orders of magnitude smaller than the measurement's own floor,
so it could not be *shown* to work even if it were made.

Two by-products are worth keeping:
* `ZEN_XP_W`/`ZEN_XP_H` in the bench, so the width class is reachable at all.
* The fact that `/usr/bin/perf` works on this box once `perf_event_paranoid` is
  lowered. Hardware counters were previously assumed unavailable here; every
  prior profile in this repo is instruction counts (callgrind, which cannot
  even execute the `v4x` path) or wall clock. `perf stat` sees the shipping
  tier and costs nothing.

### 1.5 Re-verified on current `main`

The matrix in §1.2 was taken on `bf750a26`; era-2 landed three column-tiling
commits during this lane's run. The tile knobs default to OFF, so the shipped
path is the same code, but the binary is not the same binary — so the headline
cells were re-measured on a current-`main` build
([`branch_recheck.tsv`](blur_radius_locality_branches_2026-08-31/branch_recheck.tsv)),
through `foldapp_stream_bigpair` (which takes `W H` directly, so the width
classes stay reachable).

| geometry | T | branch-miss rate | miss penalty @20c / cycles | IPC |
|---|---|---:|---:|---:|
| 576×576 | 1 | 0.0327 % | 0.348 % | 4.11 |
| 576×576 | 16 | 0.2660 % | 1.985 % | 2.57 |
| 1152×1152 | 1 | 0.0323 % | 0.333 % | 3.99 |
| 1152×1152 | 16 | 0.1732 % | 1.187 % | 2.47 |
| 2304×2304 | 1 | 0.0255 % | 0.195 % | 2.95 |
| 2304×2304 | 16 | 0.1115 % | 0.512 % | 1.71 |
| 2296×2304 | 1 | 0.0282 % | 0.296 % | 4.06 |
| 2296×2304 | 16 | 0.1151 % | 0.770 % | 2.48 |
| 2303×2304 | 1 | 0.0301 % | 0.272 % | 3.50 |
| 2303×2304 | 16 | 0.1138 % | 0.650 % | 2.12 |
| 2297×2304 | 1 | 0.0287 % | 0.300 % | 4.05 |
| 2297×2304 | 16 | 0.1121 % | 0.740 % | 2.45 |

(These cells are **not** baseline-subtracted — `foldapp_stream_bigpair`
generates its pair inside the process. The generator is a scalar loop over
`W·H` pixels, i.e. a few million branches against 3–10 **billion** in every
cell, so it moves no digit shown.)

**Read the RATE column, not the cycle column, in this table.** These are
single-process cells, so the ASLR lottery is in the *denominator*: the 2304²/1T
cell drew a slow layout (16.35 G cycles for the same nine walks the 2296 cell
did in 11.89 G, IPC 2.95 vs 4.06), which flatters its cycle percentage and
penalises the others. **Branches and branch-misses are work-determined and the
lottery cannot touch them**, so `branch-misses / branches` is the honest
column here.

**Same conclusion, on a different binary and a different harness.** 1T rates
are **0.0255–0.0327 %** — a 0.007-percentage-point spread across every size and
every width class — and the width classes still do not order the way a tail
hypothesis needs: **2296, a multiple of 8, comes back at 0.0282 %, between
2303's 0.0301 % and 2304's 0.0255 %, with 2297 at 0.0287 %.** If the scalar
tail were the mechanism, the mod-8 classes would separate from the others.
They do not. The 16T column reproduces the runtime effect too
(1.99 % at 576²/16T, falling to 0.51 % at 2304²/16T as there is more work per
thread to amortise the scheduler against).

---

## 2. (a) Radius — the COST axis

### 2.1 What radius actually controls, read from source

`ZensimConfig::blur_radius` (`metric.rs:170`, default 5) and the v2 walk's
`BLUR_RADIUS` (`feature_v2.rs:575`) are the same quantity in two places; the
fold additionally *gates* on it (`fold_engine.rs:89` — `blur_radius != 5` falls
back to buffered), and `V1_BAND_OVERLAP` (`feature_v2.rs:4554`) is a third
hard-coded 5 that is the v1 band's blur overlap. A coherent radius change is
those four plus the nine `blur_radius: 5` profile constants; the buffered v1
path is already fully radius-parametric (`streaming.rs:2400`,
`overlap = passes * r`).

**The blur itself is O(1) per pixel at any radius.** Both the H and V kernels
are running sums (`sum = sum + src[add] − src[rem]`, `blur.rs`
`box_blur_v_copy_inner_v4x` and siblings), so a wider kernel costs the same per
output pixel. That is the prediction the sweep tests, and it is why the answer
is not "radius 2 is 2.5× cheaper than radius 5". Radius buys exactly three
things:

1. **The halo.** `HALO_P = 2 · BLUR_RADIUS` (`feature_v2.rs:611`; the closure is
   `2R` because `activity = blur(|src − blur(src)|)` chains two blurs), so the
   strip's wide window is `STRIP_ROWS + 2·HALO_P = 128 + 4R` rows. **Every**
   plane pass in phase A runs over that window.

   | R | `HALO_P` | wide window | row redundancy | plane rows vs R=5 |
   |---:|---:|---:|---:|---:|
   | 5 | 10 | 148 | 1.156× | 1.000 |
   | 4 | 8 | 144 | 1.125× | 0.973 |
   | 3 | 6 | 140 | 1.094× | 0.946 |
   | 2 | 4 | 136 | 1.063× | 0.919 |

2. **The running-sum prologue.** Each column group primes its sum with `diam =
   2R+1` loads before the row loop: 11 of 148 rows at R=5, 5 of 136 at R=2.
3. **The working set**, which is the same `128 + 4R` rows times width times the
   ~13 live planes.

### 2.2 Method — and why it had to be this careful

A radius change is a **compile-time constant**, so the arms are different
binaries, and era-2 §22.5 measured that "a before/after across two BUILDS
cannot be trusted at all below ~10 %" because any edit reshuffles the binary's
layout by about that much, on top of a **10.1 % ASLR-driven bimodality** in the
944 walk at 2304². The full sound estimator was used: **one arm per process,
`min` of 11 walks inside the process (kills interference), `min` over 15
process starts with ASLR on (kills layout)**, CCD-pinned (`taskset -c 0` at 1T,
`-c 0-7,16-23` at 16T), byte-identical environment blocks across arms (the
radius is compile-time, so the env is literally the same string in every cell).
Instrument: `zensim/examples/foldapp_stream_bigpair`, era-2's own.

Raw: [`radius_cost.tsv`](blur_radius_locality_branches_2026-08-31/radius_cost.tsv)
(4 radii × 3 sizes × {1,16}T × 15 starts = 360 cells), analysis
[`radius_cost_analysis.txt`](blur_radius_locality_branches_2026-08-31/radius_cost_analysis.txt).
The estimator's own honesty column is in that file: the spread of `min_ms`
across the 15 starts is **1–5 % at 576²/1152² and 7–16 % at 2304²**, which
reproduces era-2's bimodality finding independently and is exactly why the
headline is a min-over-starts and not a mean.

### 2.3 The cost table — round 1, ONE build per radius

**Superseded by §2.4; kept because the control that supersedes it is the point.**
`min`-of-`min` ms, 944-full walk, and Δ against radius 5:

| size | T | R=5 | R=4 | R=3 | R=2 |
|---|---|---:|---:|---:|---:|
| 576² | 1 | 15.68 | 15.47 (−1.3 %) | 15.37 (−2.0 %) | 15.12 (−3.6 %) |
| 576² | 16 | 5.15 | 5.10 (−1.0 %) | 4.94 (−4.1 %) | 4.85 (−5.8 %) |
| 1152² | 1 | 64.42 | 61.88 (−3.9 %) | 62.11 (−3.6 %) | 61.24 (−4.9 %) |
| 1152² | 16 | 26.45 | 19.99 (−24.4 %) | 19.35 (−26.8 %) | 19.44 (−26.5 %) |
| **2304²** | **1** | **346.41** | **330.61 (−4.6 %)** | **320.96 (−7.4 %)** | **307.56 (−11.2 %)** |
| 2304² | 16 | 106.01 | 101.37 (−4.4 %) | 98.12 (−7.4 %) | 96.78 (−8.7 %) |

Peak RSS from `smaps_rollup`, same runs:

| size | T | R=5 | R=4 | R=3 | R=2 |
|---|---|---:|---:|---:|---:|
| 576² | 1 | 20.6 MB | 20.3 (−1.6 %) | 19.9 (−3.3 %) | 19.9 (−3.6 %) |
| 1152² | 16 | 86.8 MB | 85.7 (−1.2 %) | 83.8 (−3.4 %) | 81.6 (−6.0 %) |
| 2304² | 1 | 97.6 MB | 96.4 (−1.3 %) | 94.9 (−2.8 %) | 93.7 (−4.1 %) |
| 2304² | 16 | 183.2 MB | 178.4 (−2.6 %) | 175.8 (−4.0 %) | 173.7 (−5.2 %) |

The 2304²/1T column is monotone and reaches **−11.2 %** at radius 2, against a
halo model that predicts −8.1 % of plane rows. **That reading did not survive
its own control** — §2.4 measures the cross-build layout floor at that cell at
4.67 %, so a single-build-per-radius comparison is not evidence there. What
does survive is the working-set column, which is allocation-determined and
therefore layout-immune, and the monotone direction across four independently
built arms.

**The 1152²/16T row is called out as suspect, not as a result.** R=4/3/2 agree
at 19.35–19.99 ms and only R=5 sits at 26.45. Three arms agreeing and one
outlying is the signature era-2 §22.5 warns about (cross-build layout), not a
32 % cliff. §2.4 measures the control for exactly this.

### 2.4 The cross-build layout control — and how much it moves the headline

Because a radius arm is a different binary, §2.3's deltas are exactly the
comparison era-2 §22.5 says cannot be trusted below ~10 %. So the floor was
**measured** rather than assumed: a second radius-5 binary (`r5b`) was built
whose ONLY source difference from `r5` is a dead 24-byte `const` — identical
semantics, identical output, different code layout — and run through the same
15-start estimator on the same cells.

| cell | R=5 | R=5b | layout floor |
|---|---:|---:|---:|
| 576²/1T | 15.68 | 15.86 | +1.15 % |
| 576²/16T | 5.15 | 5.17 | +0.39 % |
| 1152²/1T | 64.42 | 63.20 | −1.89 % |
| **1152²/16T** | 26.45 | 19.83 | **−25.03 %** |
| **2304²/1T** | 346.41 | 330.22 | **−4.67 %** |
| 2304²/16T | 106.01 | 104.75 | −1.19 % |

**The control earned its place twice.** It confirms §2.3's suspect row — the
1152²/16T "R=5 is 24 % slower" cell is a property of the `r5` *binary*, not of
radius 5, and a second radius-5 build reproduces the other three arms' 19.4–20.0
ms exactly. And at 2304²/1T it puts the layout floor at **4.67 %**, which is the
same order as the radius effect being measured. A single before/after pair at
that cell is therefore not evidence, however many ASLR starts it has.

So the sweep was **re-run with two layouts per radius, all eight binaries built
from ONE source tree**, with the arms interleaved *within* each of 15 rounds so
drift hits every arm equally, and the estimator becomes `min` over
{2 layouts × 15 ASLR starts × 11 in-process walks}:

| size | T | n | R=5 | R=4 | R=3 | R=2 |
|---|---|---:|---:|---:|---:|---:|
| 576² | 1 | 30 | 15.68 (+0.00 %) | 15.34 (-2.17 %) | 15.37 (-1.98 %) | 15.15 (-3.38 %) |
| 576² | 16 | 30 | 4.97 (+0.00 %) | 4.92 (-1.01 %) | 4.85 (-2.41 %) | 4.85 (-2.41 %) |
| 1152² | 1 | 30 | 63.10 (+0.00 %) | 61.86 (-1.97 %) | 62.34 (-1.20 %) | 61.60 (-2.38 %) |
| 1152² | 16 | 30 | 19.90 (+0.00 %) | 19.66 (-1.21 %) | 19.55 (-1.76 %) | 19.15 (-3.77 %) |
| 2304² | 1 | 30 | 330.75 (+0.00 %) | 333.00 (+0.68 %) | 312.45 (-5.53 %) | 315.16 (-4.71 %) |
| 2304² | 16 | 30 | 102.98 (+0.00 %) | 102.81 (-0.17 %) | 98.43 (-4.42 %) | 95.63 (-7.14 %) |

Peak RSS (MB) on the same runs:

| size | T | R=5 | R=4 | R=3 | R=2 |
|---|---|---:|---:|---:|---:|
| 576² | 1 | 20.6 (+0.00 %) | 20.3 (-1.29 %) | 19.9 (-3.09 %) | 19.9 (-3.43 %) |
| 576² | 16 | 44.1 (+0.00 %) | 43.9 (-0.42 %) | 42.5 (-3.62 %) | 41.5 (-5.94 %) |
| 1152² | 1 | 40.2 (+0.00 %) | 39.6 (-1.49 %) | 40.0 (-0.52 %) | 39.4 (-2.06 %) |
| 1152² | 16 | 87.0 (+0.00 %) | 84.9 (-2.35 %) | 83.1 (-4.43 %) | 82.6 (-5.05 %) |
| 2304² | 1 | 97.6 (+0.00 %) | 96.3 (-1.35 %) | 94.8 (-2.90 %) | 93.6 (-4.12 %) |
| 2304² | 16 | 181.9 (+0.00 %) | 179.1 (-1.51 %) | 175.7 (-3.40 %) | 173.1 (-4.85 %) |

Three things to read out of it, in order of confidence:

1. **Peak RSS is the layout-immune column, and it is monotone on all six
   cells.** Resident set is determined by what the program allocates and
   touches, not by where the mapping lands. `−1.35 / −2.90 / −4.12 %` at
   2304²/1T (and `−1.51 / −3.40 / −4.85 %` at 2304²/16T) is the mechanism
   (`128 + 4R` rows of every plane) showing up in a quantity the layout lottery
   cannot reach — and it sits just under the `−2.7 / −5.4 / −8.1 %` the halo
   model predicts for *plane rows*, which is right, because planes are most of
   the resident set and not all of it.
2. **2304²/16T is the cleanest wall signal, and it is monotone**: `−0.17 /
   −4.42 / −7.14 %`. Sixteen threads average over sixteen placements inside one
   process, which damps the single-mapping lottery that dominates the 1T cell.
3. **576² and 1152² are monotone too** (`−2.17 / −1.98 / −3.38 %` and
   `−1.97 / −1.20 / −2.38 %` at 1T), and small — which is the expected shape,
   because the halo is a fixed 12 rows out of 148 whatever the width, so its
   share of a walk grows with how plane-bound that walk is, and the small sizes
   are less plane-bound (v2-block §2.1: the block is 32 % plane work at 576²
   against 64 % at 2304²).
4. **2304²/1T is STILL not resolvable below ~5 %, even at 30 draws per arm.**
   `+0.68 / −5.53 / −4.71 %` is not monotone: radius 2 comes back *slower* than
   radius 3 while allocating 1.2 % less and doing measurably less work. Two
   layouts × 15 ASLR starts was not enough to separate a ~4 % effect from a
   ~5 % floor at that one cell, and saying so is the result. **Do not quote a
   2304²/1T single-cell radius delta as a level.**

**And the honest headline that follows:** the radius axis is worth **4–7 % of
wall at radius 3–2**, and **radius 4 is inside the floor** — its measurable
gain is the RSS, not the clock.
### 2.5 The mechanism, phase by phase — and an internal control for the layout lottery

`ZENSIM_FOLD_TIMING` decomposes the walk into the same phases the v2-block lane
introduced. Run per radius at 2304²/1T, `min` over 7 process starts
([`phase_radius2.tsv`](blur_radius_locality_branches_2026-08-31/phase_radius2.tsv)):

| phase | R=5 | R=4 | R=3 | R=2 | Δ R2 vs R5 |
|---|---:|---:|---:|---:|---:|
| `producer` (convert + downscale) | 30.76 | 30.75 | 30.65 | 30.66 | **−0.3 %** |
| `v2:dense` | 26.15 | 26.16 | 26.17 | 26.17 | **+0.1 %** |
| `v2:gradient` | 18.05 | 18.05 | 18.05 | 18.01 | **−0.2 %** |
| `v2:append` | 15.44 | 15.48 | 15.45 | 15.49 | **+0.3 %** |
| `v2:blockiness` | 4.59 | 4.60 | 4.59 | 4.59 | **−0.0 %** |
| `fold(sum)` (v1 band replay) | 77.92 | 77.03 | 78.21 | 78.18 | **+0.3 %** |
| `blur_h(sum)` | 104.90 | 104.58 | 97.03 | 90.49 | **−13.7 %** |
| `v2:planesApp` (the `bs2` chain) | 18.16 | 17.36 | 16.34 | 15.49 | **−14.7 %** |
| `v2:planesA` (4 × V-blur + activity) | 38.24 | 36.08 | 31.55 | 29.19 | **−23.7 %** |

(Levels here are not §2.4's levels — `ZENSIM_FOLD_TIMING` is on, which adds a
timestamp and an atomic per span, and these are separate runs. The phases sum
to 335.0 ms at R=5 against §2.4's 330.75, i.e. ~1.3 % of instrument. Read the
**columns against each other**, which is what the table is for.)

**The split is total and it is exactly where §2.1 said it would be.** Every
feature kernel and the producer are radius-invariant to **≤ 0.3 %**; the three
plane passes carry the entire effect. `planesA` moves more than the halo alone
(−8.1 % of rows) because it also drops more than half its running-sum prologue
(`diam` = 11 → 5 loads per column group) and shrinks a working set that was at
the DRAM ceiling. `fold(sum)` is flat despite `V1_BAND_OVERLAP = R` taking its
band buffer 42 → 36 rows, because in the 944 arm the fold reads phase A's
precomputed H planes and its overlap rows are cheap V-blur, not the fused
kernel's work.

**And those invariant rows are an internal control the whole-walk numbers do
not have.** They come from **four independently built binaries** — the exact
cross-build comparison §2.4 says is worth ~5 % on the whole walk — and
`v2:dense` returns 26.15 / 26.16 / 26.17 / 26.17. So the layout lottery is not
a uniform ±5 % smeared over everything: **it lands on the plane passes, whose
~13 buffers can conflict, and essentially not at all on the pointwise kernels,
which stream.** That is a useful thing for any future perf lane on this code to
know, and it is why the §2.4 RSS column and this table agree while the
whole-walk 1T column wobbles.

---

## 3. (a) Radius — the QUALITY axis, against the registered bar

### 3.1 Why this is a redefinition, and which bar applies

Changing the radius changes what every feature *means*, so it is a feature
redefinition and falls under era-2 §21.1's bar, **registered on 2026-08-31
before any candidate existed**:

> PASS iff **no corpus loses more than 0.005 SROCC** and the **product
> composite does not fall**, on shipped B + the 944 roster + the two W-LIN
> winners, over the **same pairs**.

### 3.2 Method — and the control that makes it a controlled comparison

Every corpus was **re-extracted at each radius** with the owner tools, over
byte-identical pairs TSVs: `zensim/examples/v2_ab_extract`
(`ZENSIM_AB_MODE=foldapp2pools`) → `scripts/canonical_corpus/
promote_ext944_canonical.py` (`EXT944_MODE=folded720append2pools`) → four
feature roots `/mnt/v/zen/zensim-training/blurradius-r{5,4,3,2}-2026-08-31/`,
scored by `bake_verdict --regime 944 --full-json`. No statistic is hand-rolled
anywhere in the chain. Row counts are identical across all four roots
(cid22 4292, kadid 10125, tid 3000, csiq 866, live 779, aic3 600, konjnd 504,
aic4 300, sdr25 50 — 20,516 pairs per radius), so the only thing that varies
between two cells of a row is the radius.

The `foldapp2pools` regime (f156-371 LIVE) serves **both** model classes from
one root, which is what lets the 372-input B and the 944-input models be read
on exactly the same pixels.

**The control is the radius-5 row itself, and it is BIT-IDENTICAL to the
canonical root.** Radius 5 was re-extracted through this same chain and then
compared cell-by-cell on `to_bits()` against
`/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/` — the canonical
`folded720append2pools` root:

| leg | rows | bit-identical cells | max abs diff |
|---|---:|---|---:|
| ext_cid22val | 4,292 | 4,051,648 / 4,051,648 (100 %) | 0 |
| ext_kadid | 10,125 | 9,558,000 / 9,558,000 (100 %) | 0 |
| ext_tid | 3,000 | 2,832,000 / 2,832,000 (100 %) | 0 |
| ext_csiq | 866 | 817,504 / 817,504 (100 %) | 0 |
| ext_live | 779 | 735,376 / 735,376 (100 %) | 0 |
| ext_konjnd_jpeg_val | 504 | 475,776 / 475,776 (100 %) | 0 |
| ext_aic3 | 600 | 566,400 / 566,400 (100 %) | 0 |
| ext_aic4 | 300 | 283,200 / 283,200 (100 %) | 0 |
| ext_sdr25 | 50 | 47,200 / 47,200 (100 %) | 0 |
| **TOTAL** | **20,516** | **19,367,104 / 19,367,104 (100.0000 %)** | **0** |

([`r5_byte_identity.txt`](blur_radius_locality_branches_2026-08-31/r5_byte_identity.txt).)
Two things follow. First, the radius-4/3/2 deltas are attributable to the
radius and to nothing else in the extraction chain — there is no pipeline
difference to confound them. Second, and incidentally: the canonical root's
`_MANIFEST.json` names `build_commit ced6f52a`, which is **155 commits behind**
the tree this extraction ran on — a span containing era-2 stage A and stage B
(the `V8` accumulator rewrite), both column-tiling commits, the whole
fold-footprint sizing rework and the v2-block timing instrument. So this is an
**independent byte-neutrality check on all 155 of them, over 19.4 M feature
cells of real corpus data** — a far broader sample than the unit gates, and it
agrees with them.

And the SROCCs follow from the bytes: every model's radius-5 verdict reproduces
its published values **exactly**: W-LIN 7b g0.20 comes back cid22 0.8588 / konjnd 0.5118 / csiq
0.8794 / live 0.8129 / kadid 0.7218 / tid 0.7767 / aic3 0.7444 — all seven
identical to `feature_cost_frontier_2026-08-31.md` §5 to four decimals — and B
comes back cid22 0.8821, its recorded runtime value. So the r4/r3/r2 deltas are
attributable to the radius and to nothing else in the pipeline.

Several corpora carry a distortion-oriented target and so have a canonically
NEGATIVE `srocc_signed` (aic4, kadid, konjnd, sdr25). Deltas below are in
**magnitude**; no sign flipped in any cell.

**These deltas carry no sampling noise.** The bar's 0.005 was sized to be
"loose enough not to trip on reseeding noise" (era-2 §21.1), but there is no
reseeding here: the same frozen weights are read on the same pairs, and the
only thing that changes is the feature extraction. Re-running any cell
reproduces it exactly. So a −0.0007 and a −0.0073 are both exact quantities,
and the bar is being applied to a deterministic measurement rather than to a
draw from a distribution.

### 3.3 The result

Full table: [`radius_quality_analysis.txt`](blur_radius_locality_branches_2026-08-31/radius_quality_analysis.txt).
Δ|SROCC| against each model's own radius-5 row:

| model | corpus | R=4 | R=3 | R=2 |
|---|---|---:|---:|---:|
| **C944** (Profile C, shipped 944) | cid22 | +0.0019 | +0.0020 | −0.0050 |
| | konjnd | **+0.0152** | **+0.0359** | **+0.0890** |
| | live | +0.0005 | +0.0010 | −0.0009 |
| | csiq | +0.0023 | +0.0040 | +0.0009 |
| | kadid | **−0.0007** | **−0.0059** | −0.0221 |
| | tid | +0.0006 | −0.0020 | −0.0114 |
| | *composite* | *+0.0038* | *+0.0070* | *+0.0090* |
| | **BAR** | **PASS** | FAIL (kadid) | FAIL |
| **B** (shipped default, 372) | cid22 | +0.0030 | +0.0062 | +0.0095 |
| | konjnd | +0.0114 | +0.0184 | +0.0187 |
| | live | **−0.0125** | −0.0317 | −0.0556 |
| | kadid | −0.0097 | −0.0259 | −0.0530 |
| | csiq | −0.0086 | −0.0236 | −0.0520 |
| | *composite* | *+0.0040* | *+0.0073* | *+0.0098* |
| | **BAR** | FAIL (live) | FAIL | FAIL |
| **W-LIN 7b g0.20** | cid22 | +0.0030 | +0.0085 | +0.0137 |
| | live | **+0.0214** | **+0.0498** | **+0.0787** |
| | aic4 | +0.0051 | +0.0119 | +0.0238 |
| | tid | **−0.0073** | −0.0175 | −0.0280 |
| | *composite* | *+0.0034* | *+0.0080* | *+0.0129* |
| | **BAR** | FAIL (tid) | FAIL | FAIL |
| **W-LIN 7b g0.25** | *composite* | *+0.0039* | *+0.0093* | *+0.0150* |
| | **BAR** | FAIL (tid −0.0073) | FAIL | FAIL |

**One cell passes the whole bar: Profile C at radius 4**, worst corpus −0.0007,
composite +0.0038 — and §2.4 prices that cell at **+0.68 % / −0.17 % of wall**
(2304², 1T/16T; inside the layout floor either way) and **−1.35 % peak RSS**.
The cell that clears the quality bar is very nearly free in both directions.

### 3.4 What the pattern says, which matters more than the pass/fail

**A smaller radius is not a degradation. It is a REDISTRIBUTION, and it moves
quality toward the axes the product is weakest on.** Every model, at every
radius, moves the same way:

* **Gains, monotonically, on the human-MOS codec corpora**: cid22 (+0.010 to
  +0.016 for the linear pair at R=2), aic3, aic4, and — by far the largest
  single move in the whole table — **KonJND**, the near-threshold /
  visually-lossless anchor, where Profile C goes **0.5006 → 0.5896 (+0.089)**
  going from radius 5 to 2. CLAUDE.md names HF/near-lossless as "the metric's
  WEAK ZONE"; a narrower blur support is measurably better there, which is the
  physically obvious result stated for the first time with numbers.
* **Loses, monotonically, on the synthetic-distortion corpora**: TID and KADID
  for every model, CSIQ for B. Those corpora's distortion sets are dominated by
  blur, noise and contrast manipulations at scales a wide support is built to
  see. They are also the two corpora this repo's own CLAUDE.md flags as **100 %
  train==val pair-overlap — "integrity guards, not ranking signal"**.
* **LIVE splits by model class**, which is the one genuinely surprising row:
  the 372-input linear B loses 0.056 at R=2 while the 944-input W-LIN 7b linear
  pair *gains* 0.079. Same corpus, same pixels, opposite sign — so this is
  about which features a model leans on, not about the corpus.

**The models were all TRAINED at radius 5.** Every number above is an
old-weights-on-new-features read, which is the strict form of the test (it
cannot flatter the change) but also its main limitation: the losses sit exactly
where a retrain would be expected to recover most, because the weights are
fitted to a support width that no longer exists. **A retrain at radius 4 and 3
is REGISTERED here and deliberately NOT launched** — it is a training decision,
not a measurement, and this lane's job was to price the axis.

### 3.5 What is NOT measured (bar clause 3)

**The dial gates are NOT MEASURED for the radius axis, and must not be read
from these verdicts.** `bake_verdict`'s dial and corruption panels score a
*stored* 944-feature grid (`dial_grid`/`corruption_grid` parquets extracted at
radius 5), so those blocks are byte-identical across all four of my radius
roots — verified: Profile C reports `mono_pct 0.9931943853679286`,
`dynamic_range 66.8107534847681`, `tied_pct 0.03764355593364526` and
`corruption.pass_q10 0.07738095238095238` at **every** radius, to the last
digit. That is the grid being radius-blind, not the dial being radius-stable.
Clause 3 of the bar ("the dial gates, wherever a redefinition could bend
monotonicity") therefore stands **OPEN**: it needs the 4,817-cell dial grid and
the corruption grid re-extracted per radius from their persisted pixels
(`build_dial944.py` already has the `DIAL944_MODE` mechanism for exactly this
kind of twin rebuild). Registered, not run.

Also not covered: `imazen26`, `nonphoto` and `hfnlproxy`, whose 944 slices come
from the bigcodec test views rather than a local pairs TSV, so they were outside
this lane's re-extraction. Nine corpora were measured; three of the twelve in
the campaign list were not, and are listed as absent rather than counted as
passes.

### 3.6 The shipping consideration nobody would hit until it bit them

`fold_engine::is_fold_backable` requires `config.blur_radius == 5`
(`fold_engine.rs:89`). If the radius were ever exposed as a **runtime** knob
rather than changed as a compile-time constant, every non-5 radius would
silently take the **buffered** path — a different walk, ~1.15× slower serially
at 2304² and with a completely different footprint curve — while still
returning a valid-looking result. Every measurement in §2 changed that gate
along with the constants, so the fold is what was measured. Any future radius
change must move all four sites together (`BLUR_RADIUS`, `V1_BAND_OVERLAP`, the
profile/`ZensimConfig` constants, and the fold gate); the patch that does it is
recorded in [`patch_radius.sh`](blur_radius_locality_branches_2026-08-31/patch_radius.sh).
### 3.7 Correctness at a non-default radius — the suite result

A radius change has to be a coherent pipeline, not just a compiling one, so the
whole `zensim` suite was run at radius 3 and, as a paired control, at radius 5,
both `--no-fail-fast` (the first radius-3 attempt stopped at the lib target and
never reached the integration gates).

**Result: radius 5 is clean and radius 3 has 7 failures — and six of the seven
are radius-5 expectations, which a redefinition MUST move.**

| radius | targets | passed | failed | ignored |
|---|---:|---:|---:|---:|
| 5 (control) | 25 | 369 | **0** | 14 |
| 3 | 25 | 362 | **7** | 14 |

| target | test | why it fails at radius 3 |
|---|---|---|
| `tests/v1_golden_bytes.rs` | `v1_real_fixture_matches_golden`, `v1_synthetic_fixture_matches_golden`, `v1_nontight_fixture_matches_golden`, `fold_backed_fixtures_match_golden` | the golden feature vectors were **captured at radius 5**. They move by design — e.g. `f0: got 1.1147e-1 vs golden 8.902e-2`. |
| `tests/cross_platform.rs` | `hardcoded_reference_scores` | the expected scores are literals captured at radius 5 (`checkerboard+blur score −109.09 vs expected −79.87`). |
| `tests/cross_platform.rs` | `pixel_format_equivalence` | compares formats against a reference score with an **absolute** per-format tolerance calibrated at radius 5; the score scale moved, the tolerance did not. |
| `src/feature_v2.rs` | `v2_append_attr_signed_integrand_directions` | the only failure not explained by a stored radius-5 value — see below. |

**What PASSES is the part that says the pipeline is coherent.** All 13
`tests/fold_engine_parity.rs` tests pass at radius 3, including
`both_engines_are_bit_identical_across_rayon_pool_sizes` and
**`folded944_is_bit_identical_across_rayon_pool_sizes`** (22 geometries ×
rayon pools 1/2/3/8/16, `to_bits()` on all 944 slots). So at radius 3 the fold
and the buffered walk still agree bit-for-bit, and the 944 walk is still
thread-count-invariant. The radius change is a *different quantity*, computed
coherently — which is exactly the distinction a redefinition has to establish.

**The one genuine failure, stated without spin.**
`v2_append_attr_signed_integrand_directions` refines a 32×32 block of a
distorted 128×128 fixture to match the source exactly, then asserts that the
attribution density's sum over that block has the same SIGN as the true finite
difference of each feature. At radius 3 the `SSIM_DEV2 s0 Y` slot disagrees:
density **−2.332e-4** against a true Δf of **+6.577e-4**. My reading — offered
as a hypothesis, not a conclusion — is that this is a linearisation-vs-finite-
perturbation disagreement on a **second-moment** slot: `dev2` is a variance
about the image mean, its first-order integrand is
`2·(ssim_p − mean)·∂ssim_p/∂·`, and setting a whole block to *exact* is a large
perturbation that can cross the mean, which the test's own comment already
allows for on magnitude ("within a factor of ~3 either way — blur bleed +
curvature") but not on sign. The perturbation's size relative to the base point
moves with the radius, so a fixture tuned at radius 5 need not hold at 3.

**No test was relaxed, skipped, or `#[ignore]`d**, and none needed to be — this
lane ships no radius change. What the result establishes is the *cost of
shipping one*: a radius change requires re-capturing the v1 goldens, the
cross-platform reference scores, and re-tuning that attribution fixture, and
that re-capture cost belongs on the ship decision's ledger.

---

## 4. (b) A more local shape — the specified prototype was falsified elsewhere; the radius is what unlocks the axis

### 4.1 What happened to the shape this lane was asked to prototype

The hand-off asked for a **rolling row window** (~130 rows × width × 4 B, halo
redundancy at the 1.156× minimum) for the six shared planes, with a predicted
−65.8 ms. While this lane was running its branch matrix, era-2 landed four
commits that settle it:

| finding | where | verdict |
|---|---|---|
| **Band-local phase A** at B = 32 / 64 / 128 | era-2 §22, `33e77f4b`, `835ac14e` | **FALSIFIED at every height** — +15.6 % at B=32, +5.5 % at B=64, and the bit-identical B=128 control at +2.5 %, i.e. +13.1 %/+3.0 % net of plumbing, monotone in halo redundancy |
| the halo closure the proxy assumed | era-2 §22.3 | the proxy compared the fold's self-blur (`±V1_BAND_OVERLAP = ±5`, 1.31×) against phase A's chain, whose closure is **`±2·BLUR_RADIUS = ±10`** because `activity` is `blur(abs(src − blur(src)))` — so a 32-row band pays **1.625×**, not 1.31× |
| the ~1.2 MiB target | era-2 §22.3 | belongs to the rolling window, not the band; a 52-row band of the 13 planes the chain keeps is **6.2 MiB** at 2304 |
| the **rolling row window** itself | era-2 §22.4 + v2-block §5 L3 | its load-bearing component — the row-major running-sum V blur that carries one accumulator per column through memory instead of a register — was **built, proved bit-identical over 21 geometries × 3 radii, measured at +9 % and reverted** |
| what *did* win | era-2 §23, `c9db838d`, `fcdadf9c` | **column** tiling: **1.226× at 5 MP, 1.849× at 21 MP**, `blur_h` 57.4 % → 13.6 % of the walk (8.5× on the item), shipped behind `ZENSIM_H_TILE` / `ZENSIM_A_TILE` |

**This lane did not rebuild a falsified prototype.** Both shapes that could
deliver the predicted −65.8 ms are now measured — the band directly at three
heights, and the rolling window through its load-bearing component — and the
prize as scoped is not there. Re-running it would have been a duplicate
implementation of a losing design, which this repo's own rule says to delete
rather than park.

### 4.2 What IS new: the halo closure is linear in the radius, on both axes

The two locality results and the radius question turn out to be **the same
arithmetic seen twice**, and nobody had written it down:

| shape | halo closure | redundancy | at R=5 | at R=3 | at R=2 |
|---|---|---|---:|---:|---:|
| strip (rows, `STRIP_ROWS`=128) | `±2R` | `(128+4R)/128` | 1.156× | 1.094× | **1.063×** |
| band, B = 64 rows | `±2R` | `(64+4R)/64` | 1.313× | 1.188× | **1.125×** |
| band, B = 32 rows | `±2R` | `(32+4R)/32` | **1.625×** | 1.375× | **1.250×** |
| column tile, 256 wide | `±R` | `(256+2R)/256` | 1.039× | 1.023× | **1.016×** |
| column tile, 128 wide | `±R` | `(128+2R)/128` | 1.078× | 1.047× | **1.031×** |

The two closures differ for a structural reason worth stating: a **row band is
the unit for the whole phase-A chain**, so the chain's two blurs compose and it
must load `±2R` extra rows (`activity = blur(|src − blur(src)|)`); a **column
tile is applied per PASS** (`column_tiled_pass` tiles one single-plane pass at
a time and the intermediate is materialised full-width between passes), so each
tiled pass pays only its own H blur's `±R`. That asymmetry is why the column
axis won and the row axis did not — and it is also why the radius matters twice
as much to the row axis as to the column one.


Three consequences, and the first two are handed to era-2:

1. **BOTH row-shape falsifications are radius-conditional, and §4.4 measures
   one of them directly.** era-2 measured band-local at +15.6 % (B=32) and +5.5 %
   (B=64) against a per-unit efficiency gain it put at 1.25×, and called the
   trade "a wash at best" at B=64. At radius 3 a 32-row band pays **1.375×**
   where it paid 1.625×, and a 64-row band pays **1.188×** where it paid
   1.313× — a 15.4 % and 9.5 % cut in exactly the term that killed it. That
   does not resurrect the design on its own, but it means the falsification is
   a statement about *radius 5*, not about band-local phase A, and it should be
   re-read that way if the radius ever moves. The band implementation was
   reverted so it cannot be re-measured here — but **`STRIP_ROWS` pays the
   identical `(S + 4R)/S` closure and is still a live constant**, so §4.4 tests
   the same term on the knob that survives.
2. **The column tile is radius-INSENSITIVE — MEASURED (§4.3) — and it
   composes with the radius.** The tile ratio is `1.229× / 1.189× / 1.203×` at
   R = 5/3/2 and the tile-width spread from 128 to 2048 is ~1.4 % at every
   radius, so era-2 §24.3's prediction holds. The two levers stack:
   335.2 ms (neither) → 272.8 (tile) → 255.8 (**both**, 1.311×), which is 98 %
   of the product of the individual ratios. **The asymmetry is the point**: the
   radius moves a 32-row strip from +12.0 % to −4.7 % and moves the tile ratio
   by 3 %, which is exactly what `±2R` in rows against `±R` in columns
   predicts. Practical consequence for era-2 §23.6 item 4, which leaves
   `TILE_WIDTH` as "a constant with provenance, to be re-derived per the
   workspace sweep rule": **the radius does not belong in that grid.** (Nor,
   on this evidence, does the tile width matter much — 128 and 2048 are within
   1.4 % at every radius, even though a 128-wide tile pays 7.8 % redundancy at
   radius 5 and a 2048-wide one pays 0.5 %.)
3. **It also explains §2.4's cost curve without any new mechanism.** The
   `(128+4R)/128` row above predicts −8.1 % of plane rows at radius 2; the
   measured peak-RSS move is −4.12 % (planes are most, not all, of the
   resident set) and the measured wall is −4.7 % at 1T and −7.1 % at 16T.
   Radius is a *locality* knob that happens to be spelled as a *feature*
   parameter.

### 4.3 Tile width × radius, measured

The tile knobs (`ZENSIM_H_TILE`, `ZENSIM_A_TILE`, both default 0 = off on
`6306303e`) were swept against the radius at 2304²/1T (7 process starts of
min-of-9 walks) plus a two-point probe at 4608²/1T (5 starts), with the tile width **zero-padded to four
bytes** (`ZENSIM_H_TILE=0128`) so the environment block is the same size in
every arm. **Both knobs are set together here**, so this prices H + activity
tiling as a pair — and era-2 removed `ZENSIM_A_TILE` shortly afterwards
(`f13c2d03`, "activity tiling was a wash"), so on current `main` only the H
half of this exists. The question being asked is the *radius interaction*,
which that does not change.

| R | size | no tile | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2 | 2304² | 307.7 | 257.4 (-16.3 %) | 255.8 (-16.9 %) | 257.4 (-16.4 %) | 258.9 (-15.9 %) | 259.7 (-15.6 %) |
| 2 | 4608² | 2236.3 | — | — | 1193.5 (-46.6 %) | — | — |
| 3 | 2304² | 311.8 | 263.2 (-15.6 %) | 262.3 (-15.9 %) | 263.8 (-15.4 %) | 264.7 (-15.1 %) | 263.9 (-15.4 %) |
| 5 | 2304² | 335.2 | 276.6 (-17.5 %) | 273.5 (-18.4 %) | 273.9 (-18.3 %) | 273.8 (-18.3 %) | 272.8 (-18.6 %) |
| 5 | 4608² | 2349.1 | — | — | 1278.9 (-45.6 %) | — | — |

| R | tile redundancy at 128 | at 256 | at 512 |
|---|---:|---:|---:|
| 5 | 1.078× | 1.039× | 1.020× |
| 3 | 1.047× | 1.023× | 1.012× |
| 2 | 1.031× | 1.016× | 1.008× |

**era-2's claim that column tiling is radius-insensitive is CONFIRMED, and the
two levers COMPOSE.** Three readings:

1. **The tile's benefit barely moves with the radius**: `1.229× / 1.189× /
   1.203×` at R = 5 / 3 / 2 at 2304², and `1.837× / 1.874×` at R = 5 / 2 at
   4608² — a 2 % difference across a radius change that more than halves the
   column halo. That is the `±R` closure showing up as almost nothing, exactly
   as era-2 §24.3 predicted, and it is the mirror image of §4.4, where the
   `±2R` row closure moved a 32-row strip from +12.0 % to −4.7 % over the same
   radius change. **Same note, two axes, opposite sensitivities, both as the
   closure says.**
2. **The tile WIDTH barely matters either** — the spread from 128 to 2048 wide
   is 2.4–3.9 ms, about 1.4 % of the walk, at every radius. Even a 128-wide
   tile, whose halo redundancy is 7.8 % at radius 5, is within 1.4 % of the
   best. Whatever the tile is buying, it is not sensitive to how much
   redundancy it pays for it.
3. **The two levers compose, near-multiplicatively.** At 2304²/1T:

| | ms | vs neither |
|---|---:|---:|
| neither | 335.2 | — |
| column tile only (best width) | 272.8 | **1.229×** |
| radius 2 only | 307.7 | **1.089×** |
| **both** | **255.8** | **1.311×** |

`1.229 × 1.089 = 1.338` against a measured `1.311`, i.e. the two capture 98 %
of their product. The same holds at 21 MP: 2349.1 (neither) → 1278.9 (tile) →
2236.3 (radius 2) → **1193.5 (both, 1.968×)**, against a predicted
`1.837 × 1.050 = 1.929×`. They are attacking different terms — the tile shrinks the H
blur's row-group working set along x, the radius shrinks the strip's halo along
y — and neither eats the other's win.


**Caveat that governs how this reads:** era-2 measured on 2026-08-31 that the
tile's sign **inverts below ~4000 px once the H blur is row-banded across
threads** — 0.891× at 8T/2304² and 0.912× at 16T/2304², against 1.251× at
1T/2304² — so the knob stays DEFAULT-OFF pending a two-threshold sweep
(era-2 §23.6). Everything here is **1T**, the regime where the tile wins, and
it is a measurement of the *radius interaction*, not a recommendation to turn
tiling on.

### 4.4 `STRIP_ROWS` × radius — the halo term, isolated

§4.2 predicts that the halo term which killed row banding is linear in the
radius. The band implementation was reverted, but `STRIP_ROWS` is still a live
constant and pays **exactly the same closure** — an `S`-row strip loads
`S + 4R` rows — so it can test the prediction directly. The v2-block lane
measured the `STRIP_ROWS` ladder at radius 5 and found 128 optimal (L2:
`128 → 378.0 ms, 64 → 421.8, 32 → 418.5, 256 → 376.1` at 2304²/1T). Here the
same ladder is run at radius 5 **and** radius 2, one build per cell, min over
11 process starts:

| `STRIP_ROWS` | halo R=5 | ms R=5 | vs S=128 | RSS MB | halo R=2 | ms R=2 | vs S=128 | RSS MB | **R2 vs R5** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1.625× | 365.6 | +12.0 % | 63.0 | 1.250× | 301.8 | -4.7 % | 61.0 | **-17.5 %** |
| 64 | 1.312× | 352.7 | +8.0 % | 77.9 | 1.125× | 309.8 | -2.1 % | 69.9 | **-12.2 %** |
| 128 | 1.156× | 326.5 | +0.0 % | 97.6 | 1.062× | 316.5 | +0.0 % | 93.6 | **-3.0 %** |

**The sign flips, and it flips exactly where the halo model says it should.**

* **At radius 5 the v2-block lane's falsification reproduces**: a 32-row strip
  is **+12.0 %** and a 64-row strip **+8.0 %** against the shipped 128. (Levels
  differ from L2's `418.5 / 421.8 / 378.0` because 155 commits of perf work have
  landed since; the *direction* is the same and that is what reproduces.)
* **At radius 2 it inverts**: a 32-row strip is **−4.7 %** and a 64-row strip
  **−2.1 %** against 128. Smaller strips stop losing.
* **The radius delta tracks the halo term across all three heights.** R2-vs-R5
  wall is **−17.5 / −12.2 / −3.0 %** at S = 32 / 64 / 128, against a halo
  redundancy that falls **−23.1 / −14.3 / −8.1 %** over the same cells —
  monotone, same ordering, same shape. Three heights × two radii is a much
  stronger statement than any single cross-build pair, and it is the direct
  evidence for §4.2's prediction.
* **The best cell in the table is `radius 2 × STRIP_ROWS 32`: 301.8 ms against
  the shipped `radius 5 × 128`'s 326.5 ms (−7.6 %), at 61.0 MB peak RSS against
  97.6 (−37.6 %).** That **is** the locality prize the hand-off was after — a
  much smaller working set that is also faster — and it is reached by changing
  the radius, not by changing the plane shape.

### 4.4.1 Reconciling with era-2 §24.3 — the same model, one ratio apart

era-2's cross-lane note (§24.3, written for this lane) reaches the opposite
conclusion from the same closure, and the difference is one arithmetic step
worth spelling out because **its own numbers predict what I measured**:

> "a 32-row band costs `(32 + 4R)/32` — 1.625× at R=5, 1.50× at R=4, 1.375× at
> R=3. Row banding does not become viable until R≈2 (1.25×), and even then it
> must beat a measured 1.25× per-unit efficiency gain, so **a radius cut does
> not rescue §22**."

The efficiency gain has to be compared against the halo **relative to the shape
it replaces**, not against the absolute redundancy — because the strip form is
not halo-free either. The two shapes are, pleasingly, the same ratio:

| | at R=5 | at R=3 | at R=2 |
|---|---:|---:|---:|
| `STRIP_ROWS` 32 vs 128 — `((32+4R)/32) ÷ ((128+4R)/128)` | 1.406× | 1.257× | **1.176×** |
| band B=32 inside a 128-row strip — `4(32+4R) ÷ (128+4R)` | 1.405× | 1.257× | **1.176×** |

So the break-even against a 1.25× per-unit gain is at **1.25 relative extra**,
which lands between R=3 (1.257×, a wash — exactly era-2's read) and R=2
(1.176×, a predicted **1.25 / 1.176 = 1.063× win, i.e. −5.9 %**).

**Measured: −4.7 %.** And the R=5 end of the same model predicts
`1.25 / 1.406 = 0.889×`, i.e. **+12.5 %** — against era-2's band measurement of
**+13.1 % net of plumbing** and this table's **+12.0 %**. Three numbers from
two lanes and two different knobs, inside 1.1 points of each other and of the
closed form.

**So §24.3's conclusion is right at R=3 and its own model flips it at R=2**, and
that is the only correction this lane makes to it. Everything else in §22–§24
— the closure being `±2R` and not `±R`, the falsification at radius 5, the
column axis being the one that pays, the 1.25× per-unit figure that makes this
arithmetic work at all — reproduces here.

**Two honesty notes on this table.** (1) These are one build per cell, `min`
over 11 ASLR starts, so the two small deltas (−4.7 % and −3.0 %) sit at the
cross-build layout floor §2.4 measured; the ≥8 % ones are well clear of it, and
the *trend* is what carries the argument. (2) `radius 2` fails the quality bar
(§3.3), so this cell is not a shippable configuration — it is a **measurement
of the mechanism**, and what it establishes is that the row-shape falsifications
in this repo are statements about radius 5.


**`STRIP_ROWS` is a COST-ONLY knob here.** It changes the accumulation grouping
(the v2 block's per-strip partials are merged by `DenseAccum::accumulate`), so
it is not byte-neutral and nothing in this subsection is a ship proposal — it
exists to measure the halo term in isolation. One second-order confound to
name: `V1_BANDS_PER_STRIP = STRIP_ROWS / V1_BAND_ROWS`, so a 32-row strip has
**one** v1 band where a 128-row strip has four. At 1T (which is all that is
measured here) that changes the band fan-out's *shape* but not its work, and it
is the same confound in both radius columns — so the **radius-to-radius**
comparison within one `STRIP_ROWS` row is clean, which is the comparison this
subsection is for.

---

## 5. The consolidated cost/quality table

The brief's table: radius → wall → working set → per-corpus SROCC delta →
PASS/FAIL against the registered bar. Cost is the two-layout estimator (§2.4);
quality is §3.3's Δ|SROCC| against each model's own radius-5 row. `worst` is
the single worst corpus for that model at that radius; the bar needs
`worst ≥ −0.005` **and** `composite ≥ 0`.

| R | 2304²/1T ms | Δ | 2304²/16T ms | Δ | peak RSS MB | Δ | C944 worst corpus | C944 composite | **BAR** | B worst | W-LIN 7b worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 5 | 330.75 | +0.00 % | 102.98 | +0.00 % | 97.6 | +0.00 % | +0.0000 | +0.0000 | **PASS** | +0.0000 | +0.0000 |
| 4 | 333.00 | +0.68 % | 102.81 | -0.17 % | 96.3 | -1.35 % | -0.0007 | +0.0038 | **PASS** | -0.0125 | -0.0073 |
| 3 | 312.45 | -5.53 % | 98.43 | -4.42 % | 94.8 | -2.90 % | -0.0059 | +0.0070 | FAIL | -0.0317 | -0.0175 |
| 2 | 315.16 | -4.71 % | 95.63 | -7.14 % | 93.6 | -4.12 % | -0.0221 | +0.0090 | FAIL | -0.0556 | -0.0280 |

**How to read it.** The cost column is small and the quality column is a trade,
so this is not a "free win" table — it is a **priced axis**. Concretely:

* **Radius 4 is the only cell that clears the bar, and it does not pay for
  itself in time.** Its wall delta (+0.68 % / −0.17 %) is inside the
  cross-build layout floor; what it measurably buys is **−1.35 % peak RSS**.
  So the case for radius 4 is not "it is faster" — it is "it is free, it
  shrinks the working set slightly, and it improves KonJND by +0.015 and CID22
  by +0.002 at a −0.0007 worst-corpus cost".
* **Radius 3 and 2 buy 4.4–7.1 % of wall (16T) and 2.9–4.1 % of RSS**, and cost
  0.006–0.056 SROCC on the corpora that lose. Both fail the bar as written.
  Both also raise every model's product composite and every model's KonJND by a
  lot.
* **The models were trained at radius 5.** Nothing here says what a
  radius-matched model would score; §3.4 registers that retrain rather than
  guessing at it.
* **The dial clause of the bar is not measured** (§3.5). A PASS in this table
  is a PASS on clauses 1 and 2 of 3.

---

## 6. Disposition

### 6.1 What shipped

**One thing, and it is an instrument.** `ZEN_XP_W` / `ZEN_XP_H` in
`zensim/benches/extract_paths_bench.rs`'s RSS/perf loop, which previously took
only a square `ZEN_XP_SIZE` and therefore could not express the width class the
branch question is about. Bench-only; the library is byte-for-byte unchanged by
this lane (`jj diff` on `zensim/src` is empty at every commit).

**Nothing else shipped, and each non-ship is a measured decision:**

* The branch-misprediction fixes were retired by measurement (§1.4), not by
  taste — the whole class is 0.5 % of cycles at one thread.
* The radius change is a **feature redefinition** and belongs to whoever owns
  the ship decision, on the table in §5, not to a perf lane. It also cannot be
  shipped as a runtime knob without moving the fold gate (§3.6).
* The rolling row window was not rebuilt (§4.1) because both shapes that could
  deliver its prize are already measured and reverted; a losing second
  implementation is a duplicate.

### 6.2 Handed to era-2, with predicted gains

1. **The radius is the locality knob, and every halo-conditioned result in the
   break is radius-conditional.** Row closure `±2R`, column closure `±R` (§4.2).
   Concretely: era-2 §22's band-local falsification (+15.6 % at B=32, +5.5 % at
   B=64) was measured against a 1.625× / 1.313× redundancy that becomes
   **1.375× / 1.188× at radius 3** and **1.250× / 1.125× at radius 2**. The
   falsification is a statement about radius 5. It stays **PREDICTED, not
   measured** for the band itself (the implementation was reverted) — but the
   *same closure* is measured directly through `STRIP_ROWS`, and §4.4 is the
   result, and **the sign flips**: at radius 5 a 32-row strip is +12.0 %
   against the shipped 128-row one (reproducing the v2-block lane's L2
   falsification), and at radius 2 it is **−4.7 %**. The radius delta tracks
   the halo redundancy across all three heights (−17.5 / −12.2 / −3.0 % wall
   against −23.1 / −14.3 / −8.1 % halo). **So "128 is at the optimum and the
   plane-footprint fix cannot come from strip height" is true at radius 5 and
   false at radius 2** — and the best cell measured, `radius 2 × STRIP_ROWS
   32`, runs 2304²/1T in **301.8 ms against the shipped configuration's 326.5
   (−7.6 %) at 61.0 MB against 97.6 (−37.6 %)**. Not shippable as it stands
   (radius 2 fails the quality bar, and `STRIP_ROWS` is not byte-neutral), but
   it is the measured answer to "can the plane pipeline be more local":
   **yes, if the radius moves.** §4.4.1 reconciles this against era-2 §24.3's
   opposite reading: the 1.25× per-unit gain has to be compared against the
   halo *relative to the shape it replaces* (1.406 / 1.257 / 1.176× at
   R = 5/3/2 — the same ratio for a `STRIP_ROWS` change and for a B=32 band
   inside a 128-row strip), which puts break-even between R=3 and R=2 and then
   predicts **+12.5 % at R=5** and **−5.9 % at R=2** against measurements of
   +12.0 / +13.1 % and −4.7 %.
2. **`TILE_WIDTH`'s derivation is radius-conditional** (§4.3). era-2 §23.6
   item 4 registers a `512-px steps 1536..6144 × {1,4,8,16} threads × both
   CCDs` sweep to turn the tile width into a constant with provenance; the
   a 128-wide tile costs 7.8 % redundancy at radius 5 and 3.1 % at radius 2 —
   but measurement says that term is not what the tile width is trading, since
   128 and 2048 are within 1.4 % of each other at every radius. **So the
   radius does NOT belong in era-2 §23.6 item 4's `TILE_WIDTH` grid**, which
   is worth knowing before that sweep is run: it is one fewer dimension.
3. **A radius-4 retrain**, if the cost is wanted. Every model measured here was
   trained at radius 5, so §3's losses are the strict (unflattering) reading;
   the corpora that lose are the ones whose distortions live at the support
   width the weights were fitted to. Registered, not launched — it is a
   training decision.
4. **A PER-SCALE radius is the obvious next cut, and it is not currently
   expressible.** `BLUR_RADIUS` is one global constant applied at every pyramid
   level, and each level is stripped at `STRIP_ROWS = 128` rows **of its own
   plane** (`feature_v2_stream.rs:569`), so every scale pays the identical
   `(128 + 4R)/128` halo — but scale 0 holds `1/1.328 = 75 %` of the pyramid's
   pixels, so **75 % of the halo saving is available from scale 0 alone**.
   §3.4's loss pattern is what makes this interesting rather than merely
   possible: the corpora that lose (TID, KADID) lose on distortion families
   that live at fine scales, which is exactly where a scale-0-only cut would
   hurt most — so the informative experiment is the *opposite* split (keep
   radius 5 at scale 0, cut the coarse levels), which costs only 25 % of the
   halo but might cost none of the rank. Making it measurable needs the radius
   threaded through the ~20 `BLUR_RADIUS` call sites as a per-scale value;
   registered, not built.
5. **`perf` works on this box.** `perf_event_paranoid` was 4 (everything
   blocked) and `/usr/bin/perf` had been shadowed by a broken hand-built binary
   on `PATH`. Lowering the sysctl to 1 gives hardware counters on the *shipping*
   `v4x` tier — which callgrind structurally cannot profile. Any future era-2
   claim about front-end, cache or TLB behaviour can now be measured rather than
   inferred from instruction counts.

### 6.3 Open, with the measured reason

* **Bar clause 3 (dial monotonicity) for the radius axis is NOT MEASURED**
  (§3.5) — `bake_verdict`'s dial and corruption panels read stored radius-5
  feature grids, verified byte-identical across all four radius roots. Needs the
  4,817-cell dial grid and the corruption grid re-extracted per radius from
  their persisted pixels.
* **Three of the twelve campaign corpora were not re-extracted** —
  `imazen26`, `nonphoto`, `hfnlproxy` come from the bigcodec test views rather
  than a local pairs TSV. Nine were measured; the three are listed as absent,
  never counted as passes.
* **The BAND-local shape at a smaller radius is predicted, not measured**
  (§6.2 item 1). `STRIP_ROWS` measures the same closure and flips sign (§4.4),
  and the two shapes' relative halo is identical to four digits (§4.4.1), so
  the prediction is well supported — but the band itself needs era-2 to rebuild
  `stream_phase_ab_banded`, which it deliberately reverted, and this note does
  not claim a measurement it did not take.
* **Nothing here is measured at more than one thread for the shape knobs.**
  §4.3 and §4.4 are 1T only. era-2 has since shown the tile's sign inverting at
  8/16 threads below ~4000 px, so a threaded `STRIP_ROWS` × radius cell could
  behave differently and is not extrapolated to.
* **The 576² cells of the two-layout sweep were added after the main run**
  and share its estimator; §2.4's `n` column reports the draws per arm for
  every cell so the two are distinguishable.
* **Precise (PEBS-class) branch sampling is unavailable on this host**
  (§1.1), so the per-symbol misprediction split carries skid. The aggregate
  counts do not, and the aggregate is what closes the question.

### 6.4 What is irreducible

* **The blur's per-pixel cost.** Both kernels are running sums; radius does not
  change the work per output pixel, only the halo, the prologue and the working
  set. There is no "cheaper blur" here to find.
* **The mispredictions that remain.** They are dominated by the V blur's
  reflect-mirror row indexing, which the compiler has already made branchless
  (`cmov`), plus loop back-edges and the parallel runtime. §1.
* **The halo itself, AT A GIVEN RADIUS.** `1.156×` is the minimum a
  strip-shaped walk can pay at radius 5 with `STRIP_ROWS = 128`, and v2-block's
  L2 measured every smaller strip height as worse there — a result §4.4
  reproduces (+12.0 % at S=32). What is *not* irreducible is the pair: at
  radius 2 the minimum moves to `STRIP_ROWS = 32` and `1.25×`, which is **−4.7 %
  of wall and −37.6 % of peak RSS** against the shipped pair. The halo is a
  floor for a fixed radius, not for the walk.

---

## 7. Method appendix

**Box.** Ryzen 9 9950X3D, 16C/32T, asymmetric L3 (CCD0 cpus 0-7,16-23 = 96 MiB;
CCD1 cpus 8-15,24-31 = 32 MiB), 60 GiB, WSL2. Tier `v4x` (AVX-512) for every
number here — no callgrind was used, because it cannot execute the `POOL_SIMD`
path that ships.

**Trees.** §1.2's branch matrix is on `bf750a26` and is re-verified on
`6306303e` in §1.5. Everything with a radius in it — §2.4, §2.5, §3, §4.3,
§4.4 — is built from **`6306303e`** (era-2's two column-tiling commits landed,
knobs default-off), one tree for all eight two-layout binaries and all six
`STRIP_ROWS` binaries. `main` moved three more times while this ran
(`f13c2d03` deleted the x-range tiling refactor and the `ZENSIM_A_TILE` knob as
a wash, `e64f9784` and `09464b84` are ledger/measurement commits); §4.3's tile
sweep therefore measures `ZENSIM_H_TILE` **and** the since-removed
`ZENSIM_A_TILE` together, on `6306303e`, and says so rather than being restated
as a current-`main` number.

**`perf_event_paranoid` was changed and restored.** It was **4** (all events
blocked); this lane set it to **1** for the duration and put it back to **4**
afterwards. `/usr/bin/perf` (packaged `linux-perf` 7.0.12) is the working
binary; `/home/lilith/bin/perf`, which comes first on `PATH`, is a stale
hand-built one that fails to load `libpython3.10.so.1.0`.

**Timing protocol.** era-2 §22.5's estimator, **plus one layer this lane had
to add**: one arm per process,
`min` of N walks inside the process, `min` over ≥15 process starts with ASLR
on, CCD-pinned, byte-identical environment blocks (radius is compile-time, so
the environment string is literally identical between arms; the tile sweep
zero-pads `ZENSIM_H_TILE=0128` etc. to a fixed byte length for the same
reason). Because a radius arm is necessarily a
different BINARY, era-2's protocol item 1 ("one binary, runtime-selected arms")
is unreachable here, so its guarantee had to be rebuilt from the outside: a
**cross-build layout control** measured the floor (§2.4), and every radius then
got **two independent builds**, making the estimator `min` over
{2 layouts × 15 ASLR starts × 11 in-process walks} with the eight binaries
interleaved inside each round. Where that still does not resolve the effect —
2304²/1T — the note says so instead of quoting the number.

**Load honesty.** The branch matrix ran at 1-minute loads of 1.4–7.9 (the
counters it reports are per-process totals, not wall times, so load moves the
*wall* column and not the *counts*; the `sec` column was not captured and is
not used anywhere). The radius cost sweep ran at loads 0.5–5. The two-layout
re-measurement in §2.4 interleaves all eight binaries **within each round**, so
drift hits every arm equally. Cells whose estimator disagreed with its control
are called out where they occur (§2.3's 1152²/16T row) rather than averaged in.

**Everything is run through the owner tools.** No statistic in this note is
hand-rolled: `bake_verdict --full-json` produces every SROCC (which routes to
`zenstats::panel`), `v2_ab_extract` + `promote_ext944_canonical.py` produce
every feature table, `foldapp_stream_bigpair` produces every timing, and
`/usr/bin/perf` produces every counter. The analysis scripts in
[`blur_radius_locality_branches_2026-08-31/`](blur_radius_locality_branches_2026-08-31/)
only tabulate.

**Artifacts.**

| file | what |
|---|---|
| `branch_matrix.tsv` + `branch_matrix.sh` | the §1.2 counter matrix and the script that made it |
| `branch_small.tsv` | the 64²–384² edge-dominated sweep |
| `branch_miss_attribution.txt` | `perf report` at 2304×2304/1T, 2303×2304/1T, 64×64/1T and 2304×2304/16T |
| `branch_recheck.tsv` | the §1.5 re-verification on a current-`main` build |
| `radius_cost.tsv` + `radius_cost_analysis.txt` | the §2.3 sweep (360 cells) and its tabulation |
| `radius_cost_control.tsv` | the R=5 vs R=5b layout control |
| `radius_cost_2layout.tsv` | the §2.4 two-layouts-per-radius re-measurement |
| `radius_quality_analysis.txt` | the §3.3 per-corpus deltas against the bar |
| `r5_byte_identity.txt` | the §3.2 `to_bits()` check against the canonical root |
| `suite_r3.log`, `suite_r5.log` | the §3.7 full-suite runs (not committed — multi-MB; the counts are in §3.7) |
| `phase_radius.txt`, `phase_radius2.tsv` | the phase decomposition (single-start and min-over-7) |
| `tile_radius_partial_r5.tsv` | the first tile pass, radius 5 only, kept for provenance |
| `tile_radius2.tsv` | the §4.3 tile × radius ladder |
| `strip_radius.tsv` | the §4.4 `STRIP_ROWS` × radius cells |
| `patch_radius.sh` | the four-site coherent radius change (§3.6) |
| `analyze_cost.py`, `analyze_quality.py` | tabulation only |

Feature roots (not committed — block storage per the >30 KB rule):
`/mnt/v/zen/zensim-training/blurradius-r{5,4,3,2}-2026-08-31/` (9 legs × 944
features × 20,516 rows each, `regime: folded720append2pools`), CSVs at
`/mnt/v/output/zensim/blurradius-2026-08-31/run-r{5,4,3,2}/`, verdicts at
`.../verdicts-r{5,4,3,2}/`.
