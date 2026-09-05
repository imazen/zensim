# PLAN — fast-class extraction kernel (156 + cheap), 2026-09-05

**Pre-registered before any measurement or code.** Lane: KERNEL. The parallel
CAMPAIGN lane owns the MODEL side (which 156+cheap head competes with a 944
MLP on rank); this lane owns the EXTRACTION KERNEL that serves it.

User directive (2026-09-05, verbatim): *"… find a high performance but 944-mlp
competitive 156 or 156 plus cheap model, you can improve the kernel"*.

Mission constraint (memory `user_zensim_mission`, 2026-08-31, verbatim): *"we
can do an era break in order to push performance higher, remember the point of
zensim is to be extremely fast, and as good or better than ssim, and be good at
hdr."* — so a byte-moving lever is **presentable as an era-break candidate**,
never silently flipped, and era breaks are **batched**.

---

## 0. Scope

**In scope:** the cost of producing the fast class — v1 basic `f0..155`, plus
the cheap extras that ride the same walk (`V1PoolsMode::Peaks` = `f156..227`;
`V1FreeExtras::RawMoments` and `::RawMomentsPlusBoundedErr` = the 40 + 24 free
v2-layout slots), through `Zensim::compute_folded720_append_features_streaming`
and through the product entry `ZensimProfile::D` on a **default** build.

**Out of scope:** model training, rank, dial, any `bake_*` tool, the 944 pool
block's own cost (already priced, `pools_full_extraction_2026-08-30`), HDR
front-end, GPU.

**Non-negotiable:** no shipped feature byte moves in this lane without a
registered era break. A lever that is only reachable through an era break gets
**measured and registered, not flipped**.

---

## 1. What is already measured — do NOT re-derive

Sources: `benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md`,
`era2_perf_break_2026-08-31.md`, `fold_engine_2026-08-31.md`,
`fold_mt_scaling_2026-08-31.md`, `fold_footprint_2026-08-31.md`,
`profile_d_notax_2026-09-01.md`, `free_features_2026-09-01.md`,
`free_features_classC_2026-09-04.md`, `d_ship_flip_2026-09-05.md`.

Facts this plan builds on and will not re-measure:

- The **rem-ring** (H-blur redundant-gather removal) landed: −8.29 % buffered
  v1-372, −5.57 % zeroed-944 fold, −6.22 % live-pools fold at 576²/v3.
- **H-tile PACKING** landed and is worth 1.15× @5 MP / 1.73× @21 MP at 1T;
  **restricting a loop range instead bought nothing** (1.06× @2304², 0.96 ×
  @4608²). *Locality comes from LAYOUT, not loop bounds.* Do not re-attempt an
  x-range refactor.
- **Row banding of phase A LOSES** (+15.6 % / +5.5 %) — the activity closure is
  `±2·BLUR_RADIUS` = 62 % redundancy on a 32-row band.
- **Fusing the activity abs-diff into the H-blur LOSES** (+1.04 % 944-full,
  +2.01 % 372-only) — post-rem-ring the H blur gathers one strided column and
  fusing makes it two.
- **`map_init` per-band scratch LOSES** (allocating parallelism: 7.75 → 10.00 ms
  at 3T); persistent per-band scratch slots are the fix and shipped.
- **`fused_vblur_ssim` fission is RETIRED** — 27 of its 28 spill loads are
  per-column-group setup, only 1 load / 0 stores inside the innermost loop.
- **`dense_block_kernel` is the MT ceiling and ERA-LOCKED.** Band/row-partial
  merging is bit-exact only under `POOL_SIMD` (v4x) AND `width % 8 == 0` at
  every scale; neither holds generally (−2 ulps with a scalar tail, 13 ulps with
  per-pixel pools). Amdahl UPPER bound on fixing it: **1.17× @8T, 1.23× @16T**
  — against re-extracting every 944 table and re-training every 944 model.
- **Y-channel imbalance has no free fix** — X and B are the *small* channels, so
  overlapping their phase B with Y's phase A shortens nothing.
- `V1PoolsMode::Peaks` **costs the same as `Off`** to compute (the peak
  accumulators are the fused V-blur's unconditional tier).
- Raw moments: **+0.8–1.6 %**; the 944 *layout* alone is inside noise.
- **`feature-regime-v2` is default-on since 2026-09-01**, so `ZensimProfile::D`
  gets the fold on a plain `cargo add zensim` build.

## 1b. The two measurement hazards that invalidate results here

Both are measured, both bind this lane:

- **ASLR, ±10 % at 2304², bimodal (~334 / ~360 ms), not noise.** Protocol:
  ONE binary + RUNTIME arms; **identical-BYTE-LENGTH env values** (`156` /
  `15c` / `15f` / `15x` — three characters, deliberately); arms interleaved;
  min of N walks in a process; min over ≥15 process starts with ASLR ON; a
  **bit-identical control arm** carried throughout. `setarch -R` is a second
  opinion, never primary.
- **zenbench can degenerate under a tight wall budget** and report a
  spuriously *near-zero* mean for every arm at once. `min()` does not protect
  against a harness that reads LOW. Therefore: scale `ZEN_S2_WALL_S` with size
  (≥60 s at 2304²), and **validate every reading at collection time** against a
  stable reference arm's plausible floor.
- **Own-process contention is real even niced.** No concurrent `cargo`, no
  other lane's fits, during a pinned sweep. The sweep script self-checks load
  before each cell and refuses rather than producing a contaminated number.

---

## 2. H1 — the cost map (deliverable 1)

**Question:** where does the fast walk's time go, by kernel, and how does that
change with size / threads / tier?

**Instrument:** `ZENSIM_FOLD_TIMING=N` (already in tree,
`zensim/src/fold_timing.rs`, 21 phases incl. `Producer`, `ProdConvert`,
`ProdDownscale`, `AWall`/`ABusy`, `BWall`/`BBusy`, `BandBusy`, `BlurHWall`,
`BlurBandBusy`, `Between`, `MeanOffset`, `DenseKernel`, `GradKernel`,
`AppendKernel`, `PhaseAV2Planes`, `PhaseAAppendPlanes`) driven through
`zensim/examples/foldapp_stream_bigpair.rs`. `occ = busy / (wall × threads)` is
the occupancy column — a phase with `occ ≈ 1/threads` is serial.

**Grid:** arms `156` / `15c` / `15f` / `15x` / `372` / `944full`
× sizes 576² / 1152² / 2304² × threads 1 / 8 / 16 × tiers native(v4x) and
`ZEN_S2_CAP_V3=1`(v3/AVX2). Callgrind (`--no-default-features`-style v3 build)
for the **serial instruction** attribution only, with the tier caveat stated on
every number.

**Pre-registered reporting rule:** every published cell carries size, threads,
tier, arm, and the control-arm reading. No "ms/MP" without the intercept, and
per CLAUDE.md the intercept here is **negative** (per-pixel cost RISES with
size) — so the linear model is reported as *failing*, not as a fixed-cost
saving.

**Success = the map exists and is honest.** H1 has no pass/fail bar; it is the
map every lever below is judged against.

---

## 3. Levers, with pre-registered gates

**Universal admission gate (binding, all levers):**

- **G-EXACT.** `cargo test --workspace` green, and specifically
  `fold_engine_parity` (11 tests, `to_bits()` over 18 geometries × {serial,
  rayon} × pools 1/2/3/8/16), `v1_golden_bytes` (incl. the non-tight fixture),
  `folded_v1_only_matches_full_walk`, `v1_feature_width_pure_function`,
  `free_extras_are_pure_addition_to_the_v1_only_walk`, plus a **new
  before/after `to_bits()` dump comparison** over the 20-geometry set via
  `ZENSIM_BIGPAIR_DUMP` on every arm. Zero differing bits.
- **G-PERF.** Landed only if measured **≥ 2 %** at some (size, threads, tier)
  cell, with the **bit-identical control arm reading 1.000 ± the cell's own
  measured noise floor**. A cell inside the noise band is *unestablished*, not
  a result. CIs reported.
- **G-API.** `cargo public-api` ZERO delta unless the lever is an explicit,
  named API addition.
- **G-CLEAN.** clippy + fmt on touched files.

A lever failing G-PERF is **reverted and recorded as falsified**, not parked.

| # | Lever | Hypothesis | Pre-registered gate / decision rule |
|---|---|---|---|
| **L1** | **`V1PoolsMode::Off` disables the band-local self-blur** — `pools_mode_for_need` never returns `Off`, and the `self_blur` predicate requires `Full \| Peaks`, so an `Off` request falls back to phase A's four strip-wide H planes. The bench arm `add156_156basic` and bigpair `156off` both request `Off`. | The instrument is pricing a walk production never runs, and pricing it **too high**. | Measure `Off` vs `Peaks` A/B at equal env-byte-length. If `Off` is slower by >2 %, this is an **instrument defect**: fix the arms to the production mode, restate any affected published fast-class number, and tell the campaign lane. No product code change is implied. |
| **L2** | **Producer** — `Phase::Producer`'s own doc says "serial by construction today", but `feature_v2_stream.rs` has `rayon::join` (sides) + `par_iter_mut` (channels) and `ADVANCE_ROWS` was raised by the fold-MT lane. | The CLAUDE.md "producer with no rayon" claim is stale; the real residual is the strip-emission cursor, not conversion. | First **settle the fact** from `fold_mt_scaling`. Publish the `Producer` / `ProdConvert` / `ProdDownscale` occupancy from H1. Only if `Producer` `occ` is low AND its wall share ≥ 10 % at some cell does an implementation follow; otherwise record "already parallel" and **correct CLAUDE.md**. |
| **L3** | **Width dependence after H-packing** — packing shipped; what remains at 2304 / 4608 for the *fast* arms specifically (H1 measured phase-A width dependence on the 944 walk). | The fast walk's H blur may still be width-bound at 4608. | Report `BlurHWall` / `BlurBandBusy` vs width for `156`/`15f`. Implement only under G-PERF. Any tile-shape change must be **packing**, never loop-range restriction (falsified). |
| **L4** | **Does `v1_only` actually skip everything at runtime?** Source says `if v2_blocks { dense … grad … }` and `v2_blocks = !v1_only`, and phase A's v2 planes are gated. | The skip is structural and complete; BANDVIS/CSFW/append cost the fast class nothing. | **Verify, don't assume:** `DenseKernel`, `GradKernel`, `AppendKernel`, `CsfwKernel`, `PhaseAV2Planes`, `PhaseAAppendPlanes` must all read **exactly zero** ns on the `156` arm. A non-zero reading is a bug to fix. This is a correctness check with a numeric bar, not a perf lever. |
| **L5** | **Cheap-slot marginal cost** — peaks (claimed zero-marginal) and raw moments (+0.8–1.6 %) and class-C. | The claims hold at every cell in the H1 grid, not just the one they were measured at. | Publish `15c`→`15f`→`15x` marginals across the grid. **Bar: ≤ +2 % on the 156 walk for the whole cheap set at every cell.** If a cell exceeds it, find the redundant pass; if none exists, report the honest number and revise the claim. |
| **L6** | **`#[rite]` / dispatch audit** — a nested `#[arcane]` in the hot path is a perf bug (memory `feedback_archmage_dispatch`); generic SIMD helpers must be `#[inline(always)]` (a 5.3× regression from an un-inlined one is already on record here). | The fast path has no nested `#[arcane]` and no un-inlined generic SIMD helper. | Static audit + `nm` on a release build: zero `raw_moments_*` / helper symbols surviving. A finding is fixed; no finding is reported as clean. |
| **L7** | **Allocation audit** — zero allocations per compare after warm-up on the fast path. | `V2Scratch` reuse is complete. | A counting-allocator test asserting **0 allocations** on the 2nd..Nth `compute_folded720_append_features_streaming` with a reused `V2Scratch`. A non-zero count is a defect to fix. |
| **L8** | **Thread policy for the product loop** — intra-image threading vs pair-level parallelism (the zenmetrics pattern: fold `.with_parallel(false)`, parallelise across pairs). | For the fast class at product sizes, pair-level wins. | Measure aggregate throughput (pairs/s) both ways at 8T/16T. Deliverable is a **documented recommended thread policy**, not code. |

**Ranking:** L1 and L4 first (cheapest, and both are potential *defects* rather
than optimisations — a wrong number in an instrument is worse than a slow
kernel). Then L5, L7, L6 (cheap, bounded). Then L2, L3 (real work, gated on
H1). L8 last.

---

## 4. Extension point for new cheap slots (deliverable 3)

The free-features lane established the pattern: generic helper pairs
(`raw_moments_accumulate{8,16}` / `raw_moments_finish{8,16}<T: F32x{8,16}Backend>`)
plus one scalar pair, `#[inline(always)]`, replacing what had been 6 vector +
4 scalar hand-copies per tier. `#[rite]` does **not** apply — it resolves
`#[target_feature]` from a concrete token, and these are generic over a backend
trait.

**This lane's obligations:**

1. Verify the pattern still holds after any change here.
2. Add a test that a **new dummy cheap slot** costs ≤ 1 % and is emitted
   bit-exactly **in both engines** (fold and buffered).
3. Document in `docs/FEATURE_SET_IDS.md` how a new cheap slot gets its id —
   it names a SUBSET of the existing append-only numbering and never renumbers,
   so a new slot extends the compute-token vocabulary and changes `slots_hash8`.

---

## 5. Era-break policy for this lane

If a lever is only reachable by moving bytes, it is **registered, not flipped**:
what changes, by how much (max |Δ| and fraction of cells over tolerance), which
tables and which models it invalidates, and the re-extraction / re-training wave
it implies. Batched with any other pending era break. Every era break still owes
determinism, thread-invariance, declared numerical equivalence, and
rank-preservation gates. `dense_block_kernel` restructuring is the known
example and its Amdahl ceiling (1.17× @8T) is already below what its
re-extraction wave would cost — this lane does **not** propose it.

---

## 6. Deliverables

1. The cost map (H1), per kernel × size × threads × tier, with occupancy.
2. The lever table with measured deltas, CIs, and the control-arm reading.
3. Landed bit-exact levers, each with its `to_bits()` proof.
4. Registered era-break levers with their numbers — none flipped.
5. A `zensim-bench/benches/ssim2_speed_bar.rs` arm for **"D + cheap"** so the
   campaign lane's W4 is one command. Note the bench today has **no
   `ZensimProfile::D` arm at all** (its `zensim_B` arm is the buffered B walk),
   and its `add156_156basic` arm requests `V1PoolsMode::Off` — see L1.
6. Doc corrections in place (CLAUDE.md producer claim, if L2 confirms stale).
7. `cargo test --workspace` green, clippy/fmt clean on touched files,
   `cargo public-api` zero delta, CHANGELOG lines.

**Artifacts:** `/mnt/v/output/zensim/kernel-2026-09-05/`.
**Record:** `benchmarks/kernel_fastclass_2026-09-05.md`.

---

## 7. Budget

Measurement is serialized behind box idle (the campaign lane's extraction and
fits have priority; the sweep script self-checks load and refuses rather than
producing a contaminated number). Levers are implemented in ranked order and
each is landed or falsified before the next begins — no parked work.
