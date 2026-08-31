# Inside the v2-348 + append-204 block — where the time goes, and what moves it

**The user's question: "within v2-348 + append, what is so costly? how can we
get that down?"**

The fact that prompts it, from
`benchmarks/feature_cost_frontier_2026-08-31.md` §3: the block costs
**+7.4 / +38.9 / +159.3 ms at 576² / 1152² / 2304² (1T)** on top of the full
v1-372 walk — it roughly doubles extraction — and it cannot be dropped,
because the 944 MLP is the quality ceiling and needs it and ablating v2-348
costs the W-LIN 7b blend CID22 −0.745.

---

## 0. The answer in one paragraph

**It is not the v2 feature math. It is the plane pipeline that feeds it.**
Measured at the production SIMD tier, every v2 *kernel* — `dense`,
`gradient`, `append`, `blockiness` — is **flat in ns/px to within 5 % across
a 16× pixel range**, i.e. compute-bound and already at its design point. The
*plane* work that exists only to feed them — four `box_blur_v_from_copy`
sweeps, the activity chain, the `bs2` chain, and the switch from the fold's
band-local H planes to strip-wide ones — degrades **1.8–3.7×** over the same
range and is **64 % of the block at 2304²**. Production and consumption stand
at **1.84 : 1**: 128.1 ms to make the six planes, 69.8 ms to read them.

That inverts the profile everyone has been working from.
`benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md` named
`dense_block_kernel` "the single largest kernel in both fold arms
(22–26 %)" — and told the next lane to re-profile at `v4x` before picking it
up, because callgrind masks AVX-512 out of CPUID and had measured the
`v3` tier. **That re-profile is this note.** At `v4x`, dense is **13.5 % of
the block** and 7.3 % of the walk. The `POOL_SIMD` re-vectorisation already
did its job; dense is no longer the problem on the tier we ship.

**Four byte-neutral levers were tried and all four are falsified or already
taken** (§5). What shipped is the *instrument* that makes the block
attributable at `v4x` (§1) and the bit-identity gate the 944 walk did not
have (§6). What goes to era-2 is a **retargeting**: its design prices a dense
restructure at 1.17–1.23× (threads only), against a plane restructure this
note prices at **1.22–1.34× on the serial walk** (§7).

---

## 1. Why a new instrument, and what it is

`dense_block_kernel`'s `POOL_SIMD` path is **`v4x`-only by design** (32 SIMD
registers; the 16-register tiers spill, which is what the §A.14 scalar-pool
fix exists for). This box is a Ryzen 9 9950X3D — `avx512f/bw/dq/vl` present,
so `harness_active_tier()` selects `v4x` and `POOL_SIMD` is **on** in
production. Valgrind masks AVX-512 out of CPUID, so **callgrind physically
cannot profile the path that ships**; every Ir number in the predecessor doc
is the `v3` tier's scalar-pool form.

So the decomposition is done by **wall clock**, through the existing owner
for phase timing — `zensim/src/fold_timing.rs`, extended (not duplicated)
with seven v2-era phases:

| phase | wraps |
|---|---|
| `v2:dense` | `dense_block_kernel` |
| `v2:gradient` | `gradient_block_kernel` (all three const-instantiations) |
| `v2:append` | `append_block_kernel` |
| `v2:csfw` | `csfw_block_kernel` (default off; measured 0) |
| `v2:blockiness` | `blockiness_sparse_strip_wide` |
| `v2:planesA` | phase A's v2-only chain — 4 × `box_blur_v_from_copy` + `abs_diff_into` + `box_blur_1pass_into` |
| `v2:planesApp` | phase A's `bs2` chain (`square_into` + H + V) and the optional BANDVIS dst-activity twin |

Each hook is a timestamp and a relaxed atomic add behind an already-resolved
`OnceLock`, exactly like the module's existing hooks; nothing it records is
read by any kernel. `ZENSIM_FOLD_TIMING=N` enables it.

**Reproduction.** `zensim/benches/extract_paths_bench.rs`'s single-arm loop:

```sh
cargo build --release --bench extract_paths_bench -p zensim \
  --features custom-profiles,feature-regime-v2,threads,training
RAYON_NUM_THREADS=1 ZENSIM_FOLD_TIMING=10 \
  ZEN_XP_RSS=fold944_full ZEN_XP_SIZE=2304 ZEN_XP_ITERS=10 \
  <bench-binary>
```

`fold944_full` and `fold372_full` are the frontier lane's arms unchanged, so
the block being priced here is exactly the `+159.3 ms` that note reports.
Raw output: `benchmarks/v2_block_cost_2026-08-31/decomposition_2026-08-31.tsv`
(both runs, every cell, with the 1-minute load at each).

**Load honesty and reproducibility.** The matrix was run twice, ~20 minutes
apart, at 1-minute loads of 2.8–3.0 and 0.75–0.79. **At 1T every cell agrees
to ≤ 3 %, and at 1152²/2304² to ≤ 1 %** — so the 1T column, which is what
every conclusion below rests on, is solid. The 8/16T columns disagree by up
to 13 % (and at 576² by up to 108 %, where the whole walk is 5–7 ms and
thread start-up dominates); they are reported for shape, not for ratios. The
zenbench run in §7.3 waited **4 minutes** on the box-wide exclusive lock
before starting — that is another lane holding it, which is normal, not a
stall.

---

## 2. The ranked decomposition

### 2.1 Attribution — 1T, additive to ≤ 1.2 %

`fold944_full − fold372_full`, decomposed. Every row is a measured span; the
residual is what the spans do not cover. Full table:
`benchmarks/v2_block_cost_2026-08-31/attribution_1T_2026-08-31.tsv`.

| item | 576² ms | % | 1152² ms | % | **2304² ms** | **%** |
|---|---:|---:|---:|---:|---:|---:|
| **H-plane shape** (strip-wide `blur_h` − the fold's self-blur saving) | −0.46 | −7.6 | 3.13 | 9.2 | **65.83** | **32.9** |
| **`v2:planesA`** (4 × V-blur + activity chain) | 1.74 | 28.7 | 9.62 | 28.4 | **49.25** | **24.6** |
| `dense_block_kernel` | 1.60 | 26.4 | 6.58 | 19.4 | 27.10 | 13.5 |
| `append_block_kernel` | 1.24 | 20.5 | 5.14 | 15.2 | 21.32 | 10.7 |
| `gradient_block_kernel` | 1.17 | 19.3 | 4.38 | 13.0 | 16.89 | 8.4 |
| `v2:planesApp` (`bs2` chain) | 0.68 | 11.2 | 2.92 | 8.6 | 13.00 | 6.5 |
| `blockiness_sparse_strip_wide` | 0.27 | 4.5 | 1.12 | 3.3 | 4.46 | 2.2 |
| producer (the `refy` plane) | −0.24 | −4.0 | 0.55 | 1.6 | 1.48 | 0.7 |
| RESIDUAL (unattributed) | 0.07 | 1.1 | 0.41 | 1.2 | 0.76 | 0.4 |
| **TOTAL** | **6.06** | 100 | **33.84** | 100 | **200.09** | 100 |

**The block's composition inverts with size.** At 576² the four feature
kernels are **70.6 %** of it and the plane work **32.3 %** (the H-plane shape is
*negative* — the strip form is cheaper there). At 2304² the feature kernels
are **34.8 %** and the plane work **64.0 %**. Any conclusion drawn at one end
is wrong at the other, which is why §0.1 of the frontier note quotes 1T at
three sizes rather than one.

### 2.2 The H-plane shape row, explained

`fold372_full` runs `v1_only`, so phase A never runs and each fold band
blurs its own 42 rows into a band-local buffer (`FoldHSource::SelfBlur`).
`fold944_full` cannot: the v2 kernels need V-blurred planes, which need
strip-wide H planes, so phase A blurs the whole 148-row wide window and the
band reads `FoldHSource::Precomputed`. The row above is the net of the two
halves — at 2304²/1T, `blur_h` **+120.48** against the fold getting **−54.65**
cheaper.

**This is not an instruction-count effect, and callgrind proves it.** At
576²/v3, `fused_blur_h_ssim_inner` is **71,021,931 Ir** in the band-local arm
and **64,531,920 Ir** in the strip-wide arm: the strip shape executes **9.1 %
FEWER instructions** (it re-blurs a 1.156× halo where the band re-blurs
1.31×) and still costs **2.2× more wall** at 2304². Two instruments, opposite
signs — the cost is memory, not work.

Per-unit, at 2304²/1T: the band-local form runs the same kernel at **2.00
ns/px** and the strip-wide form at **4.94 ns/px**.

### 2.3 ns/px — the finding, in one table

Per-pixel cost of each span at 1T. Plane passes are divided by wide-window
pixels (the halo they actually blur), kernels by strip pixels. Basis counts
are in the TSV.

| span | 576² | 1152² | 2304² | **576²→2304²** |
|---|---:|---:|---:|---:|
| `blur_h` (strip-wide) | 1.34 | 1.77 | **4.94** | **3.69×** |
| `v2:planesA` | 1.11 | 1.56 | **2.01** | **1.81×** |
| `v2:planesApp` | 0.43 | 0.48 | 0.53 | 1.23× |
| `v2:dense` | 1.21 | 1.24 | 1.28 | **1.06×** |
| `v2:append` | 0.94 | 0.97 | 1.01 | **1.07×** |
| `v2:gradient` | 0.88 | 0.83 | 0.80 | **0.90×** |
| `v2:blockiness` | 0.17 | 0.18 | 0.18 | **1.04×** |

**Every v2 feature kernel is flat. Every plane pass is not.** The kernels did
not get slower with size; the planes stopped fitting in cache.

Byte rates make the two plane rows different problems:

| pass | bytes/px moved | 2304²/1T rate | reading |
|---|---:|---:|---|
| `v2:planesA` | 60 | **29.8 GB/s** | at the single-thread DRAM ceiling — only **fewer bytes** helps |
| `v2:planesApp` | 24 | 45.2 GB/s | partly L3-resident (its three passes hand off a hot buffer) |
| `blur_h` | 24 | **4.88 GB/s** | *not* bandwidth-bound — see below |

`blur_h` is nowhere near the bandwidth limit, so its 3.69× is a working-set
effect: the kernel transposes **16 rows** into lanes, and 16 rows × 6 planes
(src, dst, four outputs) × 2304 × 4 B = **884 KiB**, which is the 1 MiB L2 on
this part. At 1152² the same set is 442 KiB and fits with room. That is
structural to the row-group-transpose + running-sum-along-x design, and
`H_BLUR_BAND_ROWS = 16` is already the smallest legal value (it must be a
multiple of `H_BLUR_ROW_GROUP`).

### 2.4 Threads

Busy sums (not wall spans — at >1T the per-channel fan-out overlaps them),
2304², run 2:

| span | 1T | 8T | 16T | 8T/1T |
|---|---:|---:|---:|---:|
| `v2:planesA` | 49.25 | 70.50 | 63.16 | **1.43×** |
| `v2:dense` | 27.10 | 30.31 | 29.05 | 1.12× |
| `v2:gradient` | 16.89 | 19.17 | 18.63 | 1.13× |
| `v2:append` | 21.32 | 17.75 | 16.86 | 0.83× |
| `v2:blockiness` | 4.46 | 4.85 | 4.79 | 1.09× |

Total v2 CPU-time grows ~24 % going from 1 to 8 threads, and **`planesA`
alone is ~75 % of that growth** — the bandwidth-bound pass is what the
threads contend over. It is the block's scaling limiter as well as its
largest single item.

---

## 3. The structural read — which slots are byproducts, which force a sweep

Read from `zensim/src/feature_v2.rs` (`idx`, `idx_append`, `idx_append2`,
`DenseAccum`, `GradientAccum`, `stream_phase_b`), not inferred from names.

### 3.1 Slot ownership

v2-348 is 29 slots per (channel, scale) × 3 × 4:

| kernel | local `idx` | slots/ch/scale | of 348 | own sweep? |
|---|---|---:|---:|---|
| `dense_block_kernel` | 0–21, 23–24 | 24 | 288 | **yes** — one sweep over 5 planes |
| `gradient_block_kernel` | 22, 26, 27, 28 | 4 | 48 | **yes** — a 3×3 neighbourhood sweep with halo rows |
| `blockiness_sparse_strip_wide` | 25 | 1 | 12 | **yes**, but visits only the ⅛+⅛ lattice |

append-204 is 17 per (channel, scale), all from `append_block_kernel`, minus
the `APPEND_SKIP_B_SCALE0` cell. append2's 20 Y-only slots split:
`BANDVIS_GAIN`/`LOSS` ride the **gradient** sweep as a third const
instantiation; `LUMA_MEAN_REF` and `HL_BIN1`/`HL_BIN2` ride the **append**
sweep. Neither adds a pass.

### 3.2 What is genuinely free, and what is not

* **The 11 weighted-pool slots inside dense** (3 soft-peak + 4 masked + 4 IW)
  are **byproducts of a sweep that runs anyway** — they add per-pixel
  arithmetic (16 lane accumulators on `v4x`, 11 scalar `f64` accumulates per
  pixel elsewhere) but **no extra pass and no extra plane**. Dropping them
  would shorten dense's inner loop, not remove anything. This is the direct
  analogue of v1's peaks — and the difference from v1's masked/IW, which *do*
  own a pass group.
* **`sum_gms2`** in `GradientAccum` is free by the same argument and its doc
  says so ("one FMA per pixel on values already in registers"); it feeds the
  append block's `GMS_DEV2`.
* **`transducer_bank`** (2 slots) is the only in-kernel predicate — already
  hoisted, already a toggle.
* **BANDVIS is not free**: at v3, the bandvis gradient instantiation is
  **11,034,127 Ir for one channel** against **7,071,563 Ir/channel** for the
  plain one — **1.56×** — for 2 of the 20 append2 slots.
* **The append block owns a plane**, not just a sweep: `bs2 = blur(src²)` is
  the σ-split input and nothing else reads it. It is *not* recoverable from
  `ssq`, which is `blur(src² + dst²)` — the sum, accumulated in one running
  sum, so no f32-exact split exists.

### 3.3 The sentence that matters

All four v2 kernels read the **same six strip-wide planes** — `mu1`, `mu2`,
`ssq`, `s12`, `activity`, `bs2` — and at 2304²/1T:

```
producing them   128.08 ms   (H-plane shape 65.83 + planesA 49.25 + planesApp 13.00)
consuming them    69.77 ms   (dense 27.10 + append 21.32 + gradient 16.89 + blockiness 4.46)
                  ─────────
ratio              1.84 : 1
```

**So "what is so costly" is the plane pipeline, and it is costly because
producing a plane and reading it back is, at 2304², more expensive than every
formula evaluated on it put together.** No amount of kernel tuning reaches
that; it is a question about where the planes live.

---

## 4. The tier picture — v4x versus everything else

The predecessor's Ir table was the `v3` tier. Reproduced here for both arms
at 576² (`--tool=callgrind --cache-sim=no`), with the deltas that are the v2
block's `v3`-tier cost:

| kernel | `fold944_full` Ir | `fold372_full` Ir | Δ = v2 cost |
|---|---:|---:|---:|
| `dense_block_kernel_entry_v3` | 125,098,677 | 0 | **+125,098,677** |
| `box_blur_h_inner_v3` (activity + bs2 H) | 80,978,858 | 31,082,664 | +49,896,194 |
| `box_blur_v_copy_inner_v3` | 54,204,960 | 9,309,915 | +44,895,045 |
| `gradient_block_kernel_entry_v3` | 14,143,126 | 0 | +14,143,126 |
| `gradient_..._bandvis_v3` | 11,034,127 | 0 | +11,034,127 |
| `append_block_kernel_entry_nocross_v3` | 11,474,092 | 0 | +11,474,092 |
| `append_block_kernel_entry_cross_v3` | 11,344,947 | 0 | +11,344,947 |
| `fused_blur_h_ssim_inner_v3` | 64,531,920 | 71,021,931 | **−6,490,011** |
| `fused_vblur_ssim_inner_v3` | 64,869,759 | 64,869,759 | 0 (shared) |

At `v3`, dense is **47.9 %** of the block's Ir. At `v4x` it is **13.5 %** of
the block's wall. That gap is `POOL_SIMD`: one path accumulates the 11
weighted pools in 16 SIMD lanes reduced per row, the other extracts every
lane and does 11 `f64` accumulates **per pixel**.

Two consequences worth stating plainly:

1. **Every non-`v4x` tier pays roughly twice for dense**, and the v2 pool
   slots are already **allowed to differ cross-tier by policy** (the
   dispatcher deliberately emits two bodies; `pool_simd_drift_within_policy`
   is the gate, not a bit-exactness test). So giving the 16-register tiers a
   `POOL_SIMD`-equivalent path would be a *convergence* toward the bytes we
   actually extract with, not a new divergence. It is blocked on register
   pressure, not on byte policy — which is a materially different obstacle
   from the one the era-1 doc recorded, and worth re-stating in era-2's terms.
2. **Anyone citing "dense is 22–26 % of the walk" is citing the `v3`
   number.** On the shipping tier it is 7.3 % of the walk.

---

## 5. Levers tried — three falsified, three already taken

Nothing in this section shipped, and that is the result.

| # | lever | verdict |
|---|---|---|
| L1 | **Drop the rayon band split in `blur_h` at 1T** — the hypothesis that `par_chunks_mut` overhead explains the strip form's 2.2× | **FALSIFIED.** Serial whole-strip is *slower*: 121.5 vs 114.5 ms at 2304²/1T (banded wins by 6 %), 10.66 vs 10.92 at 1152². The 16-row bands are a cache win even at one thread. |
| L2 | **Shrink `STRIP_ROWS`** so the 14 strip-wide planes fit cache | **FALSIFIED.** 2304²/1T: 128 → **378.0 ms**, 64 → 421.8, 32 → 418.5, 256 → 376.1. Smaller strips re-pay the halo — the wide window is `STRIP_ROWS + 20`, so 32 rows means 1.63× redundancy against 128's 1.156×, and the redundant blur costs more than the cache buys. **128 is at the optimum; the fix cannot come from strip height.** |
| L3 | **Row-major running-sum V blur** — `box_blur_v_from_copy` walks columns with a `width·4` stride (9216 B at 2304², past a page), so a row-major form with a tile of running sums gives three sequential streams instead of three strided ones. Bit-identical by construction (`v_add_idx`/`v_rem_idx` depend only on `y`, so every column's term order is invariant) — **and it was proven so**, over 21 geometries × 3 radii covering every tile-boundary and remainder-column class. | **FALSIFIED ON SPEED.** `planesA` at 2304²/1T: column-major **47.71 ms** vs row-major 55.47 (tile 512), 53.23 (64), 52.13 (128) — **+9 % at best**. The column-major form keeps its accumulator in a *register* across all 148 rows; the row-major form must round-trip it through the tile array every row, and that costs more than the traversal saves. Implementation and gate were **reverted** rather than parked (a losing second implementation is a duplicate). |
| L4 | **Bounds-check elimination** in the v2 kernels | **NOTHING TO FIX.** Disassembled the release binary: `dense`, `gradient`, `append`, `csfw`, `blockiness`, `box_blur_v_copy`, `box_blur_h_inner`, `fused_blur_h_ssim`, `fused_vblur`, `abs_diff_into` — **zero `panic_bounds_check` sites in any of them**. The fixed-size-array pattern is already applied throughout. |
| L5 | **Skip `bs2` where the append kernel is inactive** | **ALREADY DONE.** `want_bs2 = append_cell_active(append_on, ch, scale)` at both call sites; the timing confirms it (870 append calls against 1050 dense at 2304²). |
| L6 | **Fuse `abs_diff_into` into the activity H blur** (the predecessor's rejected lever #1) | **NOT ATTEMPTED, and the case for it is now weaker.** It removes 2 of `planesA`'s 15 plane-touches (−13 % of a bandwidth-bound pass), but needs a two-input gathering H-blur — ~670 lines across four hand-written tier bodies — and L3 just showed that trading register residency or gather count for traffic loses on this hardware. The existing `box_blur_h_into_abs_diff` is **not** reusable: it computes `|src − blur_h(src)|` (v1's activity, one input, transform at the *store* site), where v2 needs `blur_h(|src − mu1|)` (two inputs, transform at the *load* site). Priced, not built. |

An honest reading of L1–L5: **the era-1 block is close to its optimum under
its own byte constraint.** The previous lanes took the available wins
(`POOL_SIMD`, the 16-row H bands, the append-cell gating, the fixed-array
pattern), and what is left is structural.

---

## 6. What shipped

1. **The decomposition instrument** — seven v2 phases in `fold_timing.rs`
   plus their hooks in `stream_phase_a` / `stream_phase_b` /
   `run_blur_pass_inner`. Byte-neutral by construction (a timestamp and an
   atomic add, gated on an already-resolved `OnceLock`). This is what makes
   the block attributable at `v4x` at all, and it is why it belongs in the
   repo rather than in a scratch patch.
2. **`folded944_is_bit_identical_across_rayon_pool_sizes`**
   (`zensim/tests/fold_engine_parity.rs`) — **a gap, now closed.**
   `both_engines_are_bit_identical_across_rayon_pool_sizes` sweeps pools for
   the *scoring* path, which runs `v1_only` and therefore exercises **not one
   v2-era kernel**; the 944 walk's only thread coverage was a single
   serial-vs-parallel pair at the ambient pool size. The new gate asserts all
   944 slots on `to_bits()` across **22 geometries × pools {1, 2, 3, 8, 16}**.
   It matters now: the v2 block's per-strip partials are merged by
   `DenseAccum::accumulate` and siblings, whose own doc records that strip
   order changes the *grouping* of the merge — so anything that re-schedules
   the phase-A/B fan-out, the H-blur bands, or the band-slot count is one
   grouping change from silently moving `f372..943`. That is precisely what
   era-2 and the column-tiling lane are about to do.

**Gates.** `cargo test --release -p zensim --features custom-profiles,
feature-regime-v2,threads,training`: **367 passed, 0 failed, 14 ignored**
before the new test, and the new test passes. `blur.rs` is byte-for-byte back
at baseline after the L3 revert (`jj diff` empty for that file).

---

## 7. Handed to era-2 — with a retarget

`benchmarks/era2_perf_break_2026-08-31.md` opens on `dense_block_kernel` as
"23.2 % of the 944-full walk" and prices fixing its parallelism at **1.17× @
8T / 1.23× @ 16T**. Both numbers are `v3`-tier and threads-only. On the
shipping tier the same kernel is **13.5 % of the block, 7.3 % of the walk**,
and the break's own bigger prize is sitting in a part of the walk its design
does not currently mention.

### 7.1 The prize, measured by proxy

Phase A's strip-wide H blur and the fold's band-local self-blur **run the
same kernel on the same data**. Measured at 2304²/1T:

| shape | halo redundancy | ns/px | end-to-end for H + band features |
|---|---:|---:|---:|
| band-local (`fold372_full`) | 1.31× | **2.00** | 134.4 ms |
| strip-wide (`fold944_full`) | 1.156× | **4.94** | 200.2 ms (`blur_h` 120.48 + fold 79.74) |

**1.49× in favour of band-local, on 13 % more blur work**, and callgrind
independently says the strip form runs 9.1 % *fewer* instructions. Putting
phase A on the band-local shape is therefore worth **−65.8 ms** of the block's
200.1 ms at 2304²/1T: block **1.49× cheaper**, whole 944 walk **367.7 → ~302
ms (1.22×)**. If `planesA` — currently pinned at the 29.8 GB/s DRAM ceiling —
gains the same factor once it is cache-resident, that is another **−27 ms**:
block **1.87×**, walk **~275 ms (1.34×)**. The first number is
measured-by-proxy; **the second is a projection and is labelled as one.**

Both are *serial* wins, so they compose with whatever thread work era-2 does,
rather than competing with it.

### 7.2 The design constraint, and the shape L2 rules out

The reason phase A cannot be band-local today is exactly era-2's subject:
`DenseAccum`/`GradientAccum`/`AppendAccum` fold **one partial per strip**, so
sub-banding the strip re-groups the f64 merge — `(0 + Σa) + Σb` where the
strip form computes one running sum. Fixed-lane accumulation makes that
grouping thread- and schedule-invariant, which is what unlocks the plane
restructure.

**L2 says the shape is not "smaller strips".** Halving `STRIP_ROWS`
*regressed* 2304²/1T by 11 % because the wide window is `STRIP_ROWS + 2·HALO_P`
and the halo tax grows faster than the cache win. The shape that works is a
**rolling row window**: compute each H row exactly once, retain ~21 rows of H
planes and ~11 of V planes, and run the kernels on rows as they mature — zero
halo redundancy, and a working set of roughly `130 rows × width × 4 B` ≈
**1.2 MiB at 2304²**, i.e. L2-resident, against the 18.2 MiB of strip-wide
planes a channel holds today.

### 7.3 One calibration note — and its correction, same day

**Superseded while this note was being written; recorded because the trade it
prices is the point.** Measured mid-session on `a25ee68e` with era-2's own
instrument (`cargo bench --bench era2_dense_ab`), the era-2 dense kernel was
**~2× slower than era-1 on `v4x`** — 223.1 vs 111.7 µs at 576×128
(+94.7…+104.6 %) and 452.2 vs 236.9 µs at 1152×128 (+89.2…+103.7 %), 4 noisy
rounds each after a 4-minute wait on the box-wide bench lock.

**That gap is now CLOSED.** era-2 stage B (`f146cbe3`, landed while this lane
was measuring) reports **98.4 → 101.8 µs at 576×128 (+4.0…+5.8 %)** and
statistical parity at 1152×128, by replacing closures with `macro_rules!` —
a 55× step, because LLVM drops the inline hint on a closure and every `V8`
operator becomes a call outside the target-feature region. Read
`benchmarks/era2_perf_break_2026-08-31.md` §20 for the current number; **do
not cite the 2× from this paragraph as live.**

The reason to keep the record is the trade it made visible, which the parity
result does not change: **the kernel era-2 spent stage B on is 13.5 % of the
block and 7.3 % of the walk at `v4x`, while the 64 % that is plane traffic is
untouched by the design as written.** Stage B was necessary — a 2× compute
deficit would have masked any cache-footprint win downstream, which is
exactly why that lane did it first — and with it closed, the plane pipeline
is now the largest thing the break can still buy. Its round-25 ordering
already puts **C (tiling)** next; §7.1 and §7.2 are the measured target for it.

### 7.4 Also worth carrying into the break

* **`POOL_SIMD` on the 16-register tiers** (§4). The obstacle is register
  pressure, not byte policy — the dispatcher already emits two bodies and the
  drift is governed by a tolerance gate, so convergence toward the `v4x`
  bytes is an improvement, not a break. era-2's two-pass split (pass A: core
  terms; pass B: pools from a row scratch) is exactly the shape that fits a
  16-register budget, so this may fall out of the work already planned.
* **The `bs2` plane as a fifth output of the fused H pass.** Today
  `square_into` + `box_blur_h` + `box_blur_v_from_copy` = 6 plane-touches for
  `blur(src²)`; folding `blur_h(src²)` into `fused_blur_h_ssim` alongside the
  existing `blur_h(src² + dst²)` would drop two of them (−33 % of
  `planesApp` ≈ **−4.4 ms** at 2304²/1T). Byte-neutral *in principle* — the
  running-sum shape is identical — but it adds an accumulator and an
  instantiation to the v1-golden-gated kernel, so it is worth doing **only
  inside a break that is already touching it**, never as a standalone 1 %
  drive-by.

---

## 8. What is irreducible, and why

* **The four V-blurred planes.** `dense` reads `mu1`, `mu2`, `ssq`, `s12`;
  `append` reads `mu1`, `mu2`, `ssq`, `bs2`, `activity`; `gradient` reads
  `activity`. No plane is unread, none is derivable from another in f32 (`ssq`
  is a *sum* accumulated in one running sum, so it does not split), and the
  V blur's own kernel is already at its optimum (L3).
* **The activity chain.** v1's activity is `|src − blur_h(src)|`; v2's is
  `|src − mu1|` over the full H-then-V blur. **They are different quantities**,
  so the fold's band-local activity cannot be reused and the two are not a
  redundancy — the one candidate the brief asked me to hunt for is
  genuinely absent here.
* **`blockiness`** at 0.18 ns/px, flat, 2.2 % — already the sparse
  lattice-only form.
* **Halo redundancy** at 1.156×, which L2 shows is already the minimum a
  strip-shaped walk can pay.

The floor for the block, holding era-1 bytes, is roughly what it costs today.
Moving it needs the plane pipeline to change shape, and that needs the era-2
accumulation contract.

---

## Artifacts

* `benchmarks/v2_block_cost_2026-08-31/decomposition_2026-08-31.tsv` — both
  runs × 3 sizes × {1, 8, 16}T × both arms, every phase, with load.
* `benchmarks/v2_block_cost_2026-08-31/attribution_1T_2026-08-31.tsv` — the
  §2.1 attribution with ns/px and basis pixel counts.
* Instrument: `zensim/src/fold_timing.rs` (`Phase::DenseKernel` … `PhaseAAppendPlanes`).
* Gate: `zensim/tests/fold_engine_parity.rs::folded944_is_bit_identical_across_rayon_pool_sizes`.

Box: Ryzen 9 9950X3D, 16C/32T, asymmetric L3 96 MiB / 32 MiB, 60 GiB, WSL2.
Tier: `v4x` (AVX-512) for every wall number; `v3` for every callgrind number,
because valgrind masks AVX-512 out of CPUID.
