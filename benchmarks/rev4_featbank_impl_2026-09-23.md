# Rev4 feature bank — implementation design note (2026-09-23)

Lane: `featbank-impl` (`/home/lilith/tmp/zensim-paper/rev4/FEATBANK_IMPL_brief.md`).
Design inputs: `docs/REV4_FEATURE_BANK_PLAN_2026-09-23.md` §1.1 and §2.4–2.5.
This file is the design note; test and cost results are appended in §Results
below as they land. **No human labels are read anywhere in this lane.**

## Final ID geometry

Append-only after f985. `n_scales = 4`, channels X/Y/B as today.

| family    | token       | replication   | signals | slots | ID range      |
|-----------|-------------|---------------|---------|-------|---------------|
| C1        | `gridblk`   | PerChannel    | 8       | 96    | f986–f1081    |
| C2        | `ringbasis` | PerChannel    | 6       | 72    | f1082–f1153   |
| C3        | `tailhist`  | PerChannel    | 12      | 144   | f1154–f1297   |
| C4        | `arttype`   | PerScale      | 6       | 24    | f1298–f1321   |

New registered layout width: **1322** (`REGISTERED_LAYOUT_WIDTHS` gains one
entry). PerChannel cell order follows the existing convention:
`(scale * 3 + channel) * signals_per_cell + local`, scale-major then
channel-minor. PerScale: `scale * 6 + local` (channel `Scalar`).

Within-cell signal order:

- `gridblk` (8): `mag_bin1..mag_bin6`, `on_mean`, `onoff_ratio`
- `ringbasis` (6): `mag_bin1..mag_bin6`
- `tailhist` (12): `ssim_p95`, `ssim_p99`, `ssim_max`, `art_p95`, `art_p99`,
  `art_max`, `det_p95`, `det_p99`, `det_max`, `mse_p95`, `mse_p99`, `mse_max`
- `arttype` (6): `blur`, `noise_x`, `noise_y`, `noise_b`, `bleed_x`, `bleed_b`

Compute tokens `gridblk`, `ringbasis`, `tailhist`, `arttype` append to
`ComputeToken` at enum positions 13–16 (`dvifm` already holds bit 12), so
`ComputeParts` widens `u16`→`u32` — every existing bit position and every
existing string form is unchanged. `ALL` renders them in slot order between
`dvifm` and `moments`.

## The shared log-magnitude coordinate (C1/C2 bins, C3 histogram)

Both the triangular bins and the tail histogram need a fixed log-magnitude
axis per pixel. A per-pixel `ln()` is the literal reading, and it was priced
first: at 1024² C2 would pay ~1M scalar `f64::ln` calls (~5–8 ms) against a
+2% budget (~1.2 ms), and C1's on-grid pass would add a similar order. That
fails the budgets before the kernels do anything else.

Decision, stated here so the registry can repeat it: **the log axis is the
IEEE-754 bit pattern of the positive value**. For `v > 0`,

```
u(v) = f64::to_bits(v) as f64        // monotone in v for v > 0
```

`u` is a uniform-in-exponent coordinate (piecewise-log: exactly log2-uniform
at binade granularity, mantissa-linear inside each binade — a ≤0.086 log2
warp relative to true `log2`, far inside a bin width). It is:

- exact and transcendental-free: 1 integer op per pixel, identical on every
  tier, every thread count, every libc — no F18-class exposure;
- monotone, so hard histograms (`partition_point`) and triangular hats
  (`hat_memberships` shape) are both well-defined on it.

For C3 the histogram edges are stated in the same domain as the emitted
values: the 32 bins are the `partition_point` cells of 31 true-log interior
edges `e_k = 10^(-6 + k*6/32)` (k = 1..31) — comparisons only, no `ln` per
pixel. For C1/C2 the triangular hats are over `u`-domain centres
`u(level)` for the level sets below; "triangular, as in DVIFM's 5 bins"
holds exactly in the declared axis.

If a later lane wants true-`ln` hats instead, that is a registry revision,
not a silent fix — this is precisely what `SignalDef::revisions` exists for.

## C1 — `gridblk` (8 signals per channel-scale)

Per plane (scale `s`, channel `c`): lattice period `P = 8 >> s` on Y,
`P = 16 >> s` on X/B (the 4:2:0 chroma lattice).

- **P < 2 (Y at scale 3, P = 1):** stated definition — every column/row is
  a lattice boundary, no phase contrast exists, the cell's 8 slots emit
  **0.0**. Registered with the stated definition, per the brief's option.
- **Boundary step excess.** For each vertical boundary between columns
  `x−1` and `x` (`x ∈ 1..w`, phase `x mod P`), and each horizontal boundary
  between rows `y−1` and `y` (`y ∈ 1..h`, phase `y mod P`):
  `e = |Δ_dst| − |Δ_src|` where `Δ` is the 1-px step across the boundary,
  in absolute plane units; normalised by the local activity of the two
  neighbouring pixels (the existing activity plane's own values):
  `ẽ = e / ((act_a + act_b)/2 + C_ACTIVITY)`.
- **Phase profile (per orientation):** `Σ|ẽ|` and `Σẽ` per phase
  `φ ∈ 0..P`. `φ* = argmax` over phases, ties to the lowest phase (scan
  order, strict `>`).
- **Accumulation.** The strip kernel evaluates `ẽ` once per boundary
  position (V8 SIMD on the wide window — the shifted-load pattern the
  gradient kernel already uses), accumulates the per-phase profile, and
  stores `ẽ` into two per-plane f32 buffers (`v[y*w+x]`, `h[y*w+x]`).
  Storing rather than per-phase binning is the measured-cheap way to keep
  the emitted bins gated on the *global* winning phase: the phase is only
  known after the whole plane is walked, and a second pass over the stored
  `ẽ` at on-grid positions costs `2n/P` reads instead of membership work on
  all `2n` boundaries.
- **Emission (V+H pooled per cell):**
  - `mag_bin_k` = `Σ_on ẽ·m_k(u(|ẽ|)) / n_on` over the 6 triangular hats
    centred on `u([1e-3, 1e-2, 1e-1, 1, 10, 100])`, ends clamped (DVIFM
    `hat_memberships` shape). Signed: the bins carry the signed excess, so
    a suppression (blur) reads negative in the same bin a block reads
    positive.
  - `on_mean` = `Σ_on ẽ / n_on`.
  - `onoff_ratio` = `(Σ_on|ẽ|/n_on) / (Σ_off|ẽ|/n_off + C_BLOCK)`.
  - `n_on = 0` or `n_off = 0` (degenerate planes) → all 8 slots emit 0.
- Identity → every `ẽ ≡ 0` → all 8 slots 0. Difference-form ✓.

## C2 — `ringbasis` (6 signals per channel-scale)

Inside `gradient_block_kernel` the existing per-pixel ringing term is
already in registers: `ring = sat(|s−d|, C_RING_ERR) · sat(act, C_ACTIVITY)
· (1 − sat(g_src, C_RING_EDGE))`. With `rev4_ringbasis` on, each pixel adds
`Σ_k ring·m_k(u(ring))` into a 6-bin f64 accumulator (hats centred on
`u([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])`), scalar-scattered per lane after
`to_array()` — the bit-domain partition is ~4 ops per pixel. Emit
`mag_bin_k = acc[k] / n_px`. Identity → 0 ✓.

## C3 — `tailhist` (12 signals per channel-scale)

Per plane, the four existing per-pixel maps `{ssim d, art, det, mse}` from
the dense kernel (era-1 scalar f64 tail and f32 lanes, era-2 V8 f32 lanes —
values taken as each era computes them):

- `hist[map][32]` u32 counts over the true-log edges above — integer, so
  thread counts, strip heights and strip merges cannot perturb it.
- `max[map]` — the running f64 max of the per-pixel values (order-free:
  bitwise identical across merges by construction).
- Emitted: `p95`/`p99` = the lower edge `e_k` of the first bin whose
  cumulative count reaches `ceil(0.95n)`/`ceil(0.99n)`, **except bin 0
  which emits 0.0**. `max` = the exact map max.
- **Identity form (stated exception to the difference-form rule):** the
  `d` map's direct-form `ssim_d_local` is NOT exactly 0 on identical
  pairs — it carries the same registered floating-point-residue class as
  the parent v2 slots (`feature_invariants.rs` class 3, bar 2e-3;
  measured ≤ ~1.4e-4 here). C3 reports the map's true quantiles, so its
  identity slots are bounded by that residue, not exact-0. C1/C2/C4 are
  exact-0 on identity (asserted in `rev4_identity_emits_zero`).
- No std pooling (plan §1.1: std of these maps is a function of mean+L2).

## C4 — `arttype` (6 signals per scale, channel `Scalar`)

- `blur` = `EDGE_WIDTH_CHANGE_Y(s) × HF_LOSS_Y(s)` — product of the two
  emitted v2 slots at that scale, formed at finalize after the cross-scale
  EWC fill (identical inputs both engines → identical bits). Identity → 0
  via HF_LOSS.
- `noise_{x,y,b}` = `Σ_flat hfg / n_flat` per channel — the existing
  `hf_gain` per-pixel term (`bounded_excess(hf_dst², hf_src², C_HF)`)
  accumulated inside the dense kernel over pixels where the reference is
  flat, `act ≤ C4_FLAT_ACT` (new constant = `C_ACTIVITY` = 0.01, stated
  here). `n_flat = 0` → 0. Identity → 0.
- `bleed_{x,b}` = `bounded_excess(Σ_out g_dst, Σ_out g_src, C_GMS)` per
  chroma channel — dst/src gradient magnitudes summed only OUTSIDE the
  ±1-px (3×3) dilated luma-edge mask of the **dst** Y plane (edge iff the
  kernel's own `grad_dst_mag ≥ C_RING_EDGE`, dilation clamped to plane
  bounds). The mask is built once per strip in the strip loop (Y channel
  rows of that scale, halo-aware, `#[magetypes]` entry) and handed to the
  X/B gradient kernels. Identity → 0; a luma-only distortion leaves
  `Σ_out g_dst == Σ_out g_src` → 0.

## Wiring

- `V2NewFeatureToggles`: `rev4_gridblk`, `rev4_ringbasis`, `rev4_tailhist`,
  `rev4_arttype` — `#[doc(hidden)]`, default OFF, independent (not the
  nested layout chain: each family must be measurable alone). All are
  v2-era blocks: `ComputeSet::from_toggles` gates each on `v2_blocks`
  (`v1_only` kills them like every other v2 block).
- `ComputeSet`: four matching flags; `v2_blocks` in `Plan::derive` extends
  to `|| touches(any rev4 token)`; `at_scale`/`populated_slots`/
  `compute_parts`/`feature_set_id` follow the existing derivations.
- `LayoutBlocks` gains the four rev4 flags (each `width > block_base`), and
  `Plan::toggles()` emits them nested on `dvifm` (width 1322 → all four).
- Emit order in `foldapp_streaming_walk_impl`: `… csfw | dvifm | gridblk |
  ringbasis | tailhist | arttype`, present iff the corresponding layout
  flag — a `any-rev4` layout is always 1322 wide.
- `FeatureRegime::Folded720Rev4` appended to the enum.
- The materialized walk (`compute_channel_scale_v2` path) gets the same
  hooks so rev4-on requests agree bitwise with the streaming walk.
- `extract_features_372col --full-rev4` emits f0..1321.
- Era-1 *and* era-2 dense kernels take the C3/C4-noise accumulator (era-1
  remains reachable under `ZENSIM_ERA2_DENSE=0`).

## Numerics invariants carried

- Difference-form everywhere (identity → exact 0) for C1/C2/C4; C3
  inherits the `d` map's registered identity fp-residue (stated above).
- Row-ordered f64 accumulation in every running sum (`blockiness_sparse_rows`
  convention); integer histogram counts; order-free max.
- No `Σs²/n − mean²` anywhere; no new `powf`/`ln`/`exp` calls on the hot
  paths (the bit-log coordinate is integer math; quantile edges are
  constants).
- All new work sits behind default-OFF toggles and const/`Option` gates:
  with every flag off, no new instruction executes on the existing paths —
  the f0..f985 bit-identity gate tests that directly.
- `#[magetypes]`/`incant!` tiers `[v4x, v4, v3, neon, wasm128, scalar]` for
  the new hot loops (gridblk strip kernel, luma-edge mask); the dense/
  gradient hooks ride the existing tier-dispatched kernels.

## Registry entries

All four families get `SignalDef` rows: `form: Difference` (bleed uses
`bounded_excess` — difference-form by construction), `direction` —
`HigherIsWorse` where signed semantics hold (ratio, ringbasis bins,
tailhist quantiles, arttype all), `Unsigned` for gridblk's signed-mass bins
and `on_mean`; `tranche: None`, `placement: AllCells` (gridblk's Y-scale-3
cell is registered-with-stated-zero, not skipped); `kernel` — `V2Dense`
(tailhist, arttype noise/blur), `V2Gradient` (ringbasis, arttype bleed),
new `KernelId::Gridblk` for C1's own pass; `revisions` — one `rev4bank`
era entry marking introduction.

## Results

### Family unit tests (all in `feature_v2.rs::tests`)

- `rev4_identity_emits_zero` — C1/C2/C4 exact-0; C3 under the registered
  `d`-map residue bar (2e-3, measured ≤ ~1.4e-4).
- `rev4_streaming_materialized_parity` — streaming vs materialized
  engines, cached + uncached moments, serial + parallel: bit-identical
  rev4 cells/segments.
- `rev4_strided_equals_tight` — strided rows bit-equal tight rows.
- `rev4_thread_and_strip_invariance` — serial vs pools {1,2,8,16} on
  128+72 mixed strip heights and `H_TILE_WIDTH`-crossing widths: all 336
  slots bit-identical.
- `rev4_gridblk_phase_shift_three` — a synthetic planted 8-px lattice
  shifted 5 px recovers phase 3 (argmax exact); response within ±40% of
  unshifted. **Deviation from the implementation brief:** this does not
  shift a reference by 3 px and then encode with zenjpeg; that encoded
  phase-recovery claim remains untested.
- `rev4_gridblk_blur_noise_band` — blur-only and noise-only on/off ratio
  inside [0.5, 2.0] on all scale-0 channels. This is a broad sanity
  bound only: measured JPEG ladder ratios 1.01–1.24 lie inside it.
- Real corpus gates require explicit `just rev4-corpus-tests <root>
  <KADID INPUTS.json> <expected unsupported SafeSyn count>`; missing
  assets, malformed roles and unexpected unsupported formats fail.
  KADID admission is keyed on `INPUTS.json` `role` (`train`/`fit`). The
  unignored 16-pair generated identity tier runs in the all-features CI
  suite without external corpus mounts.
- `rev4_tailhist_quantile_semantics` — max exact vs sorted reference;
  p95/p99 within one bin; integer counts.
- `rev4_arttype_bleed_luma_vs_chroma` — bleed ≡ 0 on luma-only
  distortion; positive on synthetic chroma shift.
- `rev4_accumulators_flat_no_cancellation` — flat high-value plane: f64
  sums track `N·v` within 1e-10 rel (200k terms); exact alternating
  cancellation.

### C1 zenjpeg ladder (corpus-gated `rev4_gridblk_zenjpeg_ladder`)

zenjpeg 4:2:0, q 95→10 step 5, 4 imazen-26 TRAIN refs (lilith/,
leading-stem LSD even). The **internal, non-emitted** winning-phase
`Σ|ẽ|` (V+H) did not decrease at any rung in the reviewer's rerun;
growth factors were 2.82×/3.86×/4.16×/3.91× q95→q10. The test now
asserts strict non-decrease without its former 2% slack. No emitted
C1 slot has been shown monotone on JPEG. In particular, signed
`on_mean` changes sign or is negative, while `onoff_ratio` peaks
mid-ladder and falls at low quality. The reviewer's phase series did
not establish JPEG-lattice lock; phases `(1,1)` and `(1,0)` persisted
down the two named ladders, while the unshifted lattice would be phase
0. The [0.5,2.0] blur/noise test cannot distinguish these ratios from
its no-lattice fixtures.

### f0–f985 bit-identity matrix (2026-09-23, this workspace)

Two independent measurements, both zero-diff:

1. **In-process toggle matrix** (`rev4_featbank_parity.rs`, synthetic
   geometry): `rev4_identity_and_segments_across_all_tiers` — 8
   geometries (incl. sub-64 and non-tight strides) × 10 archmage token
   permutations → 3 effective x86 tiers (v4, v3, scalar). f0–f985
   bit-identical between rev4-on and rev4-off **within each tier**;
   rev4 segments (f986+) bit-identical across permutations of the same
   tier (0-ULP). Cross-tier drift is *measured, not asserted* (the
   era-2 contract): observed cross-tier rev4 differences are
   categorical — histogram bin-edge flips (f1167, f1293) and C1
   phase-argmax winner flips (f986 family) on values that differ in
   the last ULP upstream — same class as registered v2 tier drift.
2. **Corpus toggle identity** — production extractor
   (`extract_features_372col --full-rev4` vs `--full-986`, the shipped
   omni-decode path; path columns only, no label column read):
   144 pairs — CID22 64 + SafeSyn 64 (incl. AVIF via zenavif) + KADID
   TRAIN 16 — × {serial, MT8} × {native v4, forced v3, forced scalar}
   → **0 diffs over 144 × 986 × 6 = 851,904 compared cells** (CSV
   substrate + `compare.py` + `SHA256SUMS` archived at
   `/var/tmp/featbank-impl/evidence/gate/`; the
   forced-tier files were re-extracted to thread-suffixed names —
   see the worklog repair note). The
   in-test gate `rev4_corpus_toggle_identity` re-covers the
   PNG/JPEG-decodable subset in-repo: 250 pair-mode extractions
   (CID22 128, SafeSyn 90, KADID 32; 19 AVIF pairs skipped there,
   covered by the extractor matrix above).
3. **CID22 canonical-parquet check**: `--full-944` on CID22 rows
   0–499 vs `baseline-recovery/cid22-train944.parquet` (Sep-14):
   500 × 944 = 472,000 feature cells, **0 bit diffs**.

### Registry / census / engine parity

- `id_arithmetic_round_trips_on_every_slot`, block-base layout,
  `every_registered_layout_width_is_a_candidate` (1322 registered):
  pass.
- `servability_census::every_registered_producer_set_is_plannable` +
  `planned_pixels_match_canonical_features_at_consumed_ids`: pass —
  zero refused slots.
- `research_engine_parity.rs` (6 tests): production vs research
  bit-exact at v1, 944, and 1322 (`everything`) layouts;
  `dropping_a_family_perturbs_only_its_own_slots` covers all four
  rev4 tokens; thread-invariant research output.
- `research::tests::everything_covers_the_whole_registry` and the
  feature-set-id era separation tests: pass.
- CI-exact gates: `just clippy` clean, `just lint-scripts` clean
  (626 scripts), `cargo fmt --all --check` clean; `cargo test
  -p zensim` — 566 lib + all integration tests pass release
  all-features; 521 pass debug (debug_asserts hold).

### Cost — ST, zenbench interleaved (taskset core 8, RAYON_NUM_THREADS=1)

Contended box: loadavg ~10/32 during the run; zenbench flagged 40
noisy rounds and several drift-correlated arms — recorded, not hidden.
20 rounds/size (8 at the larger sizes), raw rounds saved to
`/var/tmp/featbank-impl/evidence/st1.zenbench`
(sha256 `8ee4e59b`; see `rev4_cost_2026-09-23.pointer.md`).
These rounds predate the C1/C3 definition revision above. The table
recomputes the recorded rounds; no post-revision cost rerun is claimed.

Medians (ms), sizes 256/1024/2048/4096²:

| arm | 256² | 1024² | 2048² | 4096² |
|---|---|---|---|---|
| fold944_full | 3.10 | 63.02 | 271.00 | 1113.96 |
| fold986_dvifm (OFF) | 5.92 | 101.87 | 422.85 | 1714.06 |
| +C1 gridblk | 6.38 | 117.25 | 472.15 | 1925.50 |
| +C2 ringbasis | 7.75 | 134.00 | 540.42 | 2153.89 |
| +C3 tailhist | 10.48 | 184.56 | 739.21 | 2974.86 |
| +C4 arttype | 6.01 | 105.53 | 428.98 | 1753.03 |
| +all four (f1322) | 13.20 | 230.53 | 925.48 | 3716.31 |

`α + β·px` fits on the marginal family cost (arm − fold986_dvifm),
from the four raw medians per arm in `st1.zenbench` (recompute with
`python3 benchmarks/rev4_featbank_st_cost_recompute.py`):

| family | α (ms) | β (ns/px) | 1024² marginal | vs fold944_full | budget | verdict |
|---|---|---|---|---|---|---|
| C1 gridblk | −0.345 | 12.6 | +12.9 ms | **+20.4 %** (raw +24.4 %) | ≤ +5 % | MISS |
| C2 ringbasis | +4.070 | 26.0 | +31.4 ms | **+49.8 %** (raw +51.0 %) | ≤ +2 % | MISS |
| C3 tailhist | +1.696 | 75.1 | +80.4 ms | **+127.6 %** (raw +131.2 %) | ≤ +8 % | MISS |
| C4 arttype | −0.601 | 2.3 | +1.8 ms | **+2.9 %** (raw +5.8 %) | ≤ +3 % | borderline: fitted value passes, raw median misses |
| all four | +1.742 | 119.3 | +126.8 ms | **+201.2 %** (raw +204.1 %) | ≤ +15 % | MISS |

The C4 ST result is borderline: the fitted value passes, the raw median
misses. C1, C2, C3 and all-four miss. C4's fit has r² = 0.9866; the other
fits have r² ≥ 0.9994. The β decomposition is consistent with
the mechanism: C3 pays ~16M `partition_point`+increment scatters per
1024²-image (4 maps × 12 cells' worth of pixels ≈ 75 ns/px); C2 pays
six hat evals on every gradient pixel (27 ns/px); C1 pays a full
extra V8 pass over every boundary plus the on-grid hat rescan
(13 ns/px); C4 is the smallest marginal (2.3 ns/px). Sum of family
βs (116.0) ≈ the all-four fit (119.3) — no
interaction term worth reporting.

### Peak memory (`/usr/bin/time -v` + heaptrack, serial)

| arm | 1024² max RSS | B/px | 4096² max RSS | B/px | 4096² heaptrack peak |
|---|---|---|---|---|---|
| fold944_full | 55.99 MB | 54.7 | 278.6 MB | 17.0 | — |
| fold986_dvifm | 53.44 MB | 52.2 | 266.6 MB | 16.3 | 275.9 MB |
| fold1322_rev4 | 86.79 MB | 84.8 | 793.0 MB | 48.4 | 815.0 MB |

heaptrack peak-heap (1024²): dvifm 50.20 MB vs rev4 84.88 MB.
The +32 B/px at 4096² (+539 MB) is dominated by C1's stored `ẽ`
planes — two full-image f32 planes per (scale, channel) cell, the
store-rescan trade this note committed to. C2/C3/C4 alone are
memory-neutral (per-pixel scalar accumulators only); arttype's
bleed mask is one f32 plane per chroma strip (O(strip), invisible
above). The C1 plane footprint is O(image), not O(strip) — an
honest structural regression vs the walk's streaming envelope,
recorded with the cost misses.

### Cost — MT8, zenbench interleaved (`RAYON_NUM_THREADS=8`)

Same protocol as ST, acquired the exclusive lock after a ~37 min queue
behind another lane's `speed-*` sweep. 49 noisy rounds flagged —
recorded. Raw rounds:
`/var/tmp/featbank-impl/evidence/mt8.zenbench`
(sha256 `3e944e39`; see `rev4_cost_2026-09-23.pointer.md`).

Medians (ms), sizes 256/1024/2048/4096²:

| arm | 256² | 1024² | 2048² | 4096² |
|---|---|---|---|---|
| fold944_full | 1.29 | 25.03 | 111.95 | 477.40 |
| fold986_dvifm (OFF) | 4.68 | 72.88 | 298.22 | 1181.60 |
| +C1 gridblk | 4.95 | 79.28 | 333.60 | 1335.13 |
| +C2 ringbasis | 5.25 | 84.36 | 343.49 | 1347.19 |
| +C3 tailhist | 6.14 | 100.95 | 409.53 | 1613.26 |
| +C4 arttype | 4.65 | 73.95 | 306.11 | 1204.26 |
| +all four (f1322) | 7.42 | 121.36 | 493.39 | 1966.95 |

`α + β·px` marginal fits (r² ≥ 0.987):

| family | α (ms) | β (ns/px) | 1024² marginal | vs fold944_full | budget | verdict |
|---|---|---|---|---|---|---|
| C1 gridblk | −2.25 | 9.3 | +7.5 ms | **+29.8 %** | ≤ +5 % | MISS |
| C2 ringbasis | +1.53 | 9.8 | +11.8 ms | **+47.2 %** | ≤ +2 % | MISS |
| C3 tailhist | +1.32 | 25.7 | +28.2 ms | **+112.8 %** | ≤ +8 % | MISS |
| C4 arttype | +0.47 | 1.3 | +1.9 ms | **+7.5 %** (median +4.3 %) | ≤ +3 % | MISS |
| all four | −0.71 | 46.8 | +48.4 ms | **+193.4 %** | ≤ +15 % | MISS |

The marginal work parallelizes (β drops ~2.6× ST→MT8: 119.3 → 46.8
ns/px all-four) but `fold944_full` speeds up too, so the budget ratios
move only modestly and every family still misses at MT8. C4 is the
nearest to its band (+4.3 % raw median marginal vs a +3 % budget).
