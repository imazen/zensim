# Attribution-density steering map — C1 (task #67) — 2026-07-29

Commit: (this commit; workspace `zensim--attrmap`, based on `87d04153`+rebase).
Machine: WSL2 dev box (7950X), single run per cell.
Pair: `city.png` vs `city_576_q50.jpg` (576², `/mnt/v/output/zensim/diffmap-coherence-2026-07-18/`;
`city_576_q50.jpg` is byte-identical to `city_q50.jpg` — the E-M9 anchor pair).
Bakes: `EM2_fold924_s99.bin` (924, the pathological coarse-MSE bake) and
`foldmlp_bigcodec_kadis_720.bin` (720, healthy).
Full logs: `~/tmp/attrmap-c1/gate_matrix_final.log` (+ the pre-fix matrix in `gate_matrix.log`).

## What shipped

- **`zensim::AttributionResult`** + **`Zensim::compute_attribution_density{,_with_ref}`**
  (`zensim/src/attribution.rs`, gated `custom-profiles`): per-pixel attribution density
  `D = Σ_k (−s_k)·d_k` over the BASIC block (f0-155) with **true integrands + absolute
  normalization** (no per-scale renorm, no mass blend), sum-preserving coarse-scale
  upsample (NN ÷ footprint area), f64-truth SAT, `query_rect` O(1) (half-open, clamped),
  `block_sums(block)` harness-layout grid, `from_density` for externally-assembled maps.
- Internal V-blur banding replicates the production 32-row band layout → per-pixel
  signals and pooled scalars are production-bit-compatible (1e-9 sum-preservation tests
  against `compute_extended_features` pass).
- Harness: `diffmap_block_coherence` now prints **M3a** (attribution block sums vs
  ΔS_bake), a perf line, and a `ZENSIM_ATTR_DIAG=1` decomposition (true-linearization
  restrictions per feature region/class + approximation-quality SROCCs).
- 6 unit tests (SAT vs naive rects; block grid vs naive partition; mean-slot sum
  preservation at 1e-9 vs production features incl. the scale-1/2 upsample path;
  p-pool `f/p` + hf `f` identities; identical-pair ~0; NaN guards). Full
  `cargo test -p zensim --features custom-profiles,feature-regime-v2,training`:
  254 passed / 0 failed.

## Integrands (verified against `finalize()` / `fused_vblur_features_ssim`)

| slots | pooling (verified) | per-pixel integrand `d_k(i)` | status |
|---|---|---|---|
| 0,3,6,9 (ssim/art/det mean, mse) | mean | `v_i/N` | **exact** (full-plane Σ = f_k) |
| 1,2 / 4,5 / 7,8 (p4/p2 pools) | `(mean v^p)^{1/p}` | `(1/p)·(v_i^p/N)·M^{(1−p)/p}` | **exact first-order** (removal-consistent; Σ = f/p) |
| 10,12 (hf energy loss/gain) | clamped ratio of means | `±((s−μ1)²−(d−μ2)²)/Σ(s−μ1)²`, clamp-gated | **exact first-order** (Σ = f on the active side) |
| 11 (hf mag loss) | clamped ratio of means | `(|s−μ1|−|d−μ2|)/Σ|s−μ1|`, clamp-gated | **exact first-order** |

Corrections to the task brief discovered by verification, both implemented exactly:
(1) hf slots 10-12 are **not** mean-pooled — they are clamped ratios of means; the exact
signed first-order integrands are used (not a mean fallback). (2) The art/det per-pixel
signal in the FEATURE is the ratio form `ed = (1+|d−μ2|)/(1+|s−μ1|)−1` (the old diffmap
fold uses squared residuals — a different signal); the density uses the feature's own form.
Deviation with reason: for p-pools the brief's literal `v·∂f/∂v` (Euler form, sums to `f`)
over-weights p-pooled slots ×p vs their true block-removal first order; the `1/p`
(removal-consistent) form is used so cross-slot mixing matches `Σ s_k Δf_k`.

**Honest approximations list**: (a) f156-371 peak/masked/iw not spatialized (0.0% |s|-mass
on both gate bakes); (b) 924-append block f720-923 not spatialized (0.5% mass on EM2 —
see verdict: decisive at 128px); (c) v2 fold-in (harness) is a mean-integrand
approximation through `compute_v2_diffmap` with weights ×`1/(w·h)` (unit conversion, exact
for even pyramid dims) — the v2 fold's non-additive families (DEV2/DEV4, soft-peaks,
fragility, transducer-bank k-variants) are excluded by that fold; (d) blur bleed: signals
are attributed wholly to their own pixel (refinement actually affects ±blur-radius
neighborhoods); (e) SIMD-padding columns' mass (576→592 here) unattributable after trim;
(f) p-root curvature beyond first order; hf clamp state assumed fixed.

**v2 fold-in unit lesson (measured)**: adding the RAW `compute_v2_diffmap` map to the
score-unit density (the brief's "same as M3 does today") swamps it by orders of magnitude
— M3a degenerated to exactly M3's value in every cell of the first matrix
(`gate_matrix.log`). The unit-correct fold-in scales the v2 gradient by `1/(w·h)` before
the fold (valid because the fold is linear in per-(scale,ch,slot) weights, replicates
coarse scales ×4^s, and its family maps pool to features by mean — the
`v2_diffmap_block_pool_matches_features` identity).

## Gate table (SROCC vs ΔS_bake unless noted)

`M3a` = attribution density + unit-correct v2 fold-in (the headline);
`M3a-basic` = f0-155 density alone; `ceiling` = TRUE linearization restricted to
basic+v2 (what a PERFECT density with the append block still blind could reach);
`attr≈lin` = SROCC(attr-basic block sums, true basic linearization) — the density's
approximation quality against its own target.

### EM2_fold924_s99 (924, pathological; M3 today ≈ 0.30 @32, negative @128)

| block | M2 | M3 (fold) | M3a-basic | M3a | ceiling (append-blind) | attr≈lin | SSE bar |
|---|---|---|---|---|---|---|---|
| 16 | +1.0000 | +0.3771 | +0.5011 | **+0.5056** | +0.6803 | 0.9202 | −0.1606 |
| 32 | +0.9999 | +0.3048 | +0.4475 | **+0.4262** | +0.5006 | 0.9637 | −0.0795 |
| 64 | +0.9999 | +0.2491 | +0.5265 | **+0.3836** | +0.4309 | 0.9857 | −0.0656 |
| 128 | +0.9992 | −0.3615 | +0.3292 | **−0.1869** | −0.0815 | 0.9923 | +0.2300 |

### foldmlp_bigcodec_kadis_720 (720, healthy; no-regression clause)

| block | M2 | M3 (fold) | M3a-basic | M3a | ceiling (=full, no append) | attr≈lin | SSE bar |
|---|---|---|---|---|---|---|---|
| 16 | +1.0000 | +0.4204 | +0.5071 | **+0.5687** | +1.0000 | 0.8873 | +0.1757 |
| 32 | +1.0000 | +0.2814 | +0.5254 | **+0.4987** | +1.0000 | 0.9156 | +0.4463 |
| 64 | +0.9999 | +0.0030 | +0.6083 | **+0.4264** | +0.9999 | 0.9595 | +0.5762 |
| 128 | +1.0000 | +0.5915 | +0.2031 | **+0.8954** | +1.0000 | 0.9823 | +0.4623 |

## Verdict (gate NOT met — reported, not relaxed)

- **TARGET M3a ≥ 0.85 at every size: NOT MET** (met in 1/8 cells: K720@128 = 0.8954).
- **M3a > M3 in 8/8 cells** (both bakes, every size); the healthy-bake no-regression
  clause is exceeded everywhere (largest: 64px 0.003 → 0.426; 128px 0.592 → 0.895).
  The 128px sign-inversion of the fold on EM2 is reduced (−0.36 → −0.19) but not cured.
- **The density is at/near its measured structural ceiling on EM2 at every size**
  (0.51/0.68, 0.43/0.50, 0.38/0.43, −0.19/−0.08): the miss is NOT integrand quality —
  `attr≈lin` is 0.89-0.99 in all 8 cells — it is **coverage**.
- **Root cause (the C1 discovery, `ZENSIM_ATTR_DIAG=1`): |s|-mass ≠ rank-variance.**
  EM2 puts 98.4% of |s|-mass on basic (MSE-dominated), yet the TRUE basic-restricted
  linearization ranks ΔS at only +0.33..+0.54 (mse class alone: ≈0); per-block ΔS is the
  small residue of large cancelling class terms, and the ranking is completed by the
  1.6%-mass v2+append tail (basic 0.44 → +v2 0.50 → +append 0.9999 @32). At 128px the
  append block ALONE carries the positive signal (append-only +0.4646; basic+v2 ceiling
  −0.0815). This corrects the E-M9 design assumption that basic coverage ⇒ M2-transfer.
- **Named follow-ups that close the remaining distance** (measured, not guessed):
  (1) an append-block (f720-923) attribution fold — mandatory for any 924-era bake, it
  is the whole 128px signal on EM2 despite 0.5% mass; (2) exact integrands for the v2
  fold's excluded non-additive families (DEV2/DEV4, soft-peaks, fragility, k-bank) —
  on K720 the v2 true-lin ranks 0.82-0.89 but the current mean-integrand v2 fold-in
  approximates it at only 0.23-0.93 (`v2attr_block vs true-lin_v2` per cell).
- SSE bar: M3a beats block-SSE in 6/8 cells (exceptions: EM2@128 where everything
  basic+v2 is structurally negative, and K720@64).

## Perf (C1 reports; C2 bar is ≤1.1×)

Attribution build 31.9-50.7 ms vs ModelSensitivity diffmap build 11.7-15.1 ms on the
576² pair → **2.4-3.4× today** (both timings include reference precompute). Known C2
levers: the attribution path is single-threaded (the fold uses rayon bands), combines in
f64, and runs its own full pyramid rather than sharing planes with a diffmap call.

## Repro

```sh
cargo build --release -p zensim --features custom-profiles,feature-regime-v2 \
  --example diffmap_block_coherence
D=/mnt/v/output/zensim/diffmap-coherence-2026-07-18
ZENSIM_ATTR_DIAG=1 target/release/examples/diffmap_block_coherence \
  $D/city.png $D/city_576_q50.jpg \
  --bake /mnt/v/output/zensim/bakes/coherent-089/em2/EM2_fold924_s99.bin --block 32
```

---

# C2a — coverage completion: exact v2 + append integrands (2026-07-29)

Same pair/bakes/host as C1. Logs: `~/tmp/attrmap-c1/gate_matrix_c2a_v2.log`
(final, post edge-width sign fix); perf re-measured uncontended.

## What shipped

- **`Zensim::compute_attribution_density_full`** (`custom-profiles` +
  `feature-regime-v2`): basic density (C1) + **exact-integrand densities for
  every v2 (f372-719) and append (f720-923) slot**, built by
  `feature_v2::compute_v2_append_attribution`. Pass A replays the materialized
  strip walk (STRIP_ROWS/HALO_P geometry, `run_blur_pass`, the `stream_phase_a`
  σ-split `bs2` chain) and runs the PRODUCTION kernels
  (dense/gradient/append/blockiness) over the replicated planes — pooled
  scalars are production-arithmetic, gated at **1e-9** against the canonical
  924 streaming extractor (`v2_append_attr_features_match_production`).
  Pass B combines exact f64 integrands per slot class and upsamples
  sum-preservingly. This REPLACES C1's unit-scaled mean-integrand
  `compute_v2_diffmap` fold-in in the harness.
- Integrand classes (all 29 v2 + 17 append slots covered):
  mean pools exact (`−v/N`); reference-weighted pools exact (`−w·v/Σw`:
  masked/iw, append luminance bins with the Bernstein mid-bin);
  **self-weighted soft-peaks** exact first-order (`w(v)(f−v)/W`, full-plane
  sum ≡ 0); **deviation pools** (SSIM_DEV2/DEV4, GMS/ART/DET_DEV2) exact
  central-moment chain rule, SIGNED; **global slots** (GLOBAL_DMEAN/CGAIN/
  CLOSS) exact whole-plane chain rule; **blockiness** lattice terms split
  50/50 across the step pair; **EDGE_WIDTH_CHANGE** exact two-scale chain
  rule (incl. the last-scale copy's weight); reference-only slots
  (PJND_FRAGILITY, GRAD_SRC_MEAN) exactly 0; structural zeros (X/B
  transducers, the (B, scale 0) append cell) exactly 0.
- Tests: production-parity 1e-9 (the strict gate); density sum identities at
  1e-5 (mean + weighted pools; **the density-sum-vs-feature identity is
  1e-5/1e-6-class by physics** — kernels pool f32-lane values, pass B
  recomputes in f64; a 1e-9 density-sum identity is only achievable for
  kernel-stored planes, i.e. C1's basic sd slots); soft-peak zero-sum;
  finite-difference direction checks on every SIGNED integrand family
  (GLOBAL_DMEAN at 5 %; dev/soft-peak/edge-width/blockiness/globals at
  sign + factor-3) — **this FD test caught a real sign bug in the
  edge-width scale-t term before landing**; full-density wiring test.
  Suite: 259 passed / 0 failed.

## Gate table (M3a = full-coverage density; targets ≥ 0.85 every cell)

### EM2_fold924_s99 (924, pathological)

| block | M2 | M3 (fold) | C1 M3a | **C2a M3a** | gate |
|---|---|---|---|---|---|
| 16 | 1.0000 | +0.377 | +0.506 | **+0.8605** | PASS |
| 32 | 0.9999 | +0.305 | +0.426 | **+0.9254** | PASS |
| 64 | 0.9999 | +0.249 | +0.384 | **+0.9748** | PASS |
| 128 | 0.9992 | −0.362 | −0.187 | **+0.9915** | PASS |

The 128 px inversion — the E-M9 pathology this program exists to fix — is
CURED: −0.36 (fold) → **+0.99**, essentially at the M2 ceiling. The append
block (0.5 % |s|-mass, the whole coarse-block signal) is what did it:
non-basic density vs non-basic true-lin = 0.84/0.91/0.96/1.00.

### foldmlp_bigcodec_kadis_720 (720, healthy)

| block | M2 | M3 (fold) | C1 M3a | **C2a M3a** | gate |
|---|---|---|---|---|---|
| 16 | 1.0000 | +0.420 | +0.569 | **+0.6850** | miss |
| 32 | 1.0000 | +0.281 | +0.499 | **+0.5330** | miss |
| 64 | 0.9999 | +0.003 | +0.426 | **+0.3828** | miss |
| 128 | 1.0000 | +0.592 | +0.895 | **+0.9023** | PASS |

## Verdict — 5/8 cells pass (EM2 4/4, K720 1/4); K720 gap decomposed

- **Coverage: COMPLETE.** Every f0-923 slot is attributed (exact, exact
  first-order, or exactly-zero-by-structure). The C1 append-blind ceiling is
  gone; for a 720 bake, basic+v2 true-lin = M2 = 1.0000.
- **True-nonlinearity: negligible** (M2 0.9992-1.0000 everywhere).
- **The K720 16-64 px miss is APPROXIMATION error, isolated to the v2
  density**: basic-density-vs-its-true-lin holds 0.89-0.98 at all sizes,
  while v2-density-vs-true-lin_v2 runs 0.61/0.40/0.18/0.89 (16/32/64/128).
  K720 carries 17.3 % v2 mass spread over ~29 slots × 12 cells; EM2's v2
  mass is 1.1 % (noise there, dominated by its append block, hence 4/4).
  Leading mechanism (hypothesis, C2b lever): **blur bleed** — the v2 signals
  are 11×11-box-blurred at each scale and the density attributes each signal
  wholly to its own pixel; at 16-64 px the bleed zone rivals the block. The
  natural C2b remedy is spreading each blurred signal's attribution over its
  blur support (sum-preserving box-spread before upsampling).
- M3a ≥ M3 (fold) in 7/8 cells; the exception is K720@64 where M3 is 0.003
  (noise floor) and M3a is 0.383.

## Perf (measure-only; C2b optimizes)

Uncontended at 576²: **full density 125-138 ms** vs C1 basic 28-31 ms vs
ModelSensitivity fold 11.3 ms. Known C2b levers: single sweep instead of the
current pass-A/pass-B double blur, sharing the reference prep with the basic
path (each currently prepares its own pyramid), SIMD for the f64 combine,
strip-level parallelism.

## Repro (C2a)

```sh
cargo test -p zensim --features custom-profiles,feature-regime-v2,training \
  --release --lib v2_append_attr
ZENSIM_ATTR_DIAG=1 target/release/examples/diffmap_block_coherence \
  $D/city.png $D/city_576_q50.jpg \
  --bake /mnt/v/output/zensim/bakes/p1kadis/foldmlp_bigcodec_kadis_720.bin --block 64
```

---

# C2b — bleed allocation (measured NEGATIVE) + perf floor (2026-07-29)

Same pair/bakes/host. Logs: `~/tmp/attrmap-c1/gate_matrix_c2b_p1.log` (50/50
residual-split variant), `gate_matrix_c2b_final.log` (final state). Tests:
260 passed / 0 failed (adds `box_spread_preserves_sums_exactly`).

## Part 1 — the adjoint/spread hypothesis, measured honestly

The proposed exact fix (blur the plane-integrand with the pipeline kernel,
`∂f/∂x = K∗(∂f/∂g)`) is exact only for signals LINEAR in a blurred plane.
Two structural findings, then the A/B:

1. **The pure adjoint is wrong for residual-form signals** (art/det/hf/mscn
   cores are `d_i − (K∗d)_i`): the adjoint of `I − K` is `I − K` (symmetric),
   whose columns sum to 0 — it allocates ZERO net mass per signal, destroying
   the removal-semantics contract (refining a signal's support kills the whole
   signal, mass 1). The implementable family is support-allocation: spread a
   signal's mass over the pixels whose refinement kills it.
2. **Sum-preserving spread operator**: `blur::box_spread_sum_preserving` —
   separable, clipped windows, per-SOURCE normalization (`Σout == Σin` to f64
   rounding, gated by `box_spread_preserves_sums_exactly`; interior ==
   normalized box blur). Boundary convention documented at the fn.

**A/B on the 8-cell gate** (allocation classes: window = ssim-d + v2
contrast/texture; residual = art/det/hf/mscn; pixel = mse/transducers/
globals/blockiness/gradient-family):

| variant | result |
|---|---|
| C2a baseline (all pixel-allocated) | EM2 .8605/.9254/.9748/.9915; K720 .6850/.5330/.3828/.9023 |
| 50/50 residual split + window spread | **REGRESSED all 8 cells** (−0.01..−0.08; EM2@16 fell to .784, below gate) |
| window-only spread (SHIPPED) | NEUTRAL (±0.003): EM2 .8602/.9225/.9722/.9900; K720 .6836/.5309/.3811/.9023 |

**Verdict: the blur-bleed hypothesis for the K720 fine-block gap is
FALSIFIED** — no allocation scheme in this family recovers it. Since C2a
already showed coverage is complete and M2 (true linearization) ≈ 1.0, the
K720 16-64 px residual is the **finite-removal / nonlinear-interaction floor
of a first-order density on a v2-heavy gradient mix** — reported as the
measured floor, not a gate relaxation. The shipped state keeps the window-only
spread (theoretically defensible for zero-pixel-support signals, measured
free) and pixel allocation for everything else. Gate state unchanged: **5/8
(EM2 4/4 — no regression from the spread; K720 128 px only).**

## Part 2 — perf: measured floor and the honest gap to ≤1.1×

Levers landed (exactness gates unchanged — pass-A production-kernel parity
still 1e-9; all sum identities/FD tests green; the final 8-cell table above
IS the final perf state):

1. **Single-sweep pipeline** in `compute_v2_append_attribution`: pass B(s)
   scheduled after pass A(s+1) (the edge-width lookahead), ping-ponged plane
   sets — kills the former second full blur+cache sweep.
2. **Channel-parallel pass A** (rayon join, per-channel scratch) + **row-banded
   parallel pass B** (64-row bands, disjoint output rows; blockiness stays
   serial — it writes the row above).
3. Same channel-parallel structure in the basic builder (measured ~neutral —
   memory-bound; kept for uniformity).

Section timing (`ZENSIM_ATTR_PERF=1`, 576², uncontended): v2app pipeline
108 → **47-51 ms** (blur+cache 26→13, kernels 6, pass-B combine 69→21).

**Final perf line (uncontended, city 576²):** full **95-98 ms** | basic
**~38 ms** | ModelSensitivity fold **11-12 ms** → ratios **~8.3×** (full) /
**~3.3×** (basic) vs the fold. C2a was 125-138 ms → **−30 %**.

**The ≤1.1× bar is NOT met, and is structurally out of reach for the
STANDALONE full density**: the fold baseline computes only the v1 pyramid +
a trivial f32 per-pixel fold, while the full density computes the v1 pipeline
PLUS the entire v2+append plane set + production kernels + exact f64
integrands — strictly more plane work than a 924-feature extraction (54 ms/MP
single-thread per the C0 gate doc). Remaining levers, with honest estimates:

- **f32 + magetypes SIMD for the pass-B combine** (~21 ms → est 5-8 ms; the
  1e-5/1e-6 identity class already tolerates f32 recompute; pass-A parity
  unaffected). Est. full ~75-80 ms.
- **Deeper band-parallel blur** inside channels (13 → est 5-6 ms).
- **Fusion into a single combined compare** — the real path to ≤1.1×: a codec
  computing the 924 scalar per iteration already produces every plane the
  density needs; fused, the marginal density cost ≈ pass-B combine + SAT
  (est 10-25 ms today, 5-10 ms with SIMD). Requires bit-unchanged golden
  gating on the existing outputs; integration-level work, out of C2b scope.

## Approximation-list delta (C2b)

- ADDED: window-supported signals spread over their blur window
  (sum-preserving, clipped-window per-source-normalized boundary convention).
- CONFIRMED (measured): residual-form + pixel-form signals stay
  pixel-allocated — the 50/50 window split regresses, the pure adjoint is
  structurally wrong (zero net mass).
- The remaining K720 fine-block gap is reclassified from "blur bleed
  (hypothesis)" to **finite-removal floor (measured)**.

---

# C3a — the FUSED compare: score + steering map from one pipeline (2026-07-29)

Tests: 262 passed / 0 failed. 8-cell gate rerun at the final fused state:
**IDENTICAL to C2b-final to 4 decimals in all 8 cells** (the standalone path
is untouched; fusion is additive). Logs: `~/tmp/attrmap-c1/gate_matrix_c3a.log`.

## What shipped

- **`Zensim::compute_with_ref_score_and_attribution(pre, dist, s)`**
  (`custom-profiles`) → `(ZensimResult, AttributionResult)` — the codec-loop
  call shape for the shipped 372-class profiles (requires a profile that
  computes the full basic set: an MLP/bake profile or `extended_features`;
  `blur_passes == 1`). One plane pipeline: the production per-scale walk
  retains each (scale, channel)'s SSIM-error + mu planes IN PLACE
  (band-aligned pre-split slices through the parallel bands — no per-band
  allocation, no concatenation), derives the basic-density coefficients from
  the scalar's own `ScaleStats` (+ two reference-side hf sums the stats
  don't carry), and combines with a row-band-parallel `#[autoversion]` f32
  kernel + f32 sum-preserving spread + f32 upsample + f64 SAT.
- Retention plumbing (`AttrScaleRetention`, additive `Option` params through
  `process_scale_bands_into_accum` / `process_strip_channel`) is `None` on
  every pre-existing caller — byte-identical behavior (suite-verified).
- `examples/fused_compare_bench.rs` — the four-number perf harness.

## Golden gates (fused vs unfused)

- **Score: BIT-identical** to `compute_with_ref_and_diffmap`'s score
  (`fused_score_bit_matches_diffmap_path` — same pipeline, same stats, same
  `apply_mlp_scoring`). The plain `compute_with_ref`/fold paths are
  untouched code (retention `None`).
- **Attribution vs the standalone f64 path**: agrees to the f32-combine
  class (`fused_matches_standalone_attribution`: per-pixel ≤ 3e-5·max|D|,
  16-px block sums ≤ 1e-4 rel) — the standalone path and its strict f64
  1e-9 identities are unchanged.

## Perf (codec-loop shape: reference precomputed once, medians of 15-21)

| size | scalar-only | fold (score+fold-map) | FUSED (score+attr map) | marginal map | marginal fold | ratio |
|---|--:|--:|--:|--:|--:|--:|
| 576² | 5.8 ms | 7.4 ms | **14.8 ms** | 9.0 ms | 1.6 ms | **5.7×** |
| 1152² | 13.3 ms | 26.2 ms | **41.9 ms** | 28.6 ms | 12.9 ms | **2.2×** |

(1152² fixture: 2×2 montage of city/dog/girl/city + IM q50 JPEG —
`~/tmp/attrmap-c1/fixtures/`, regenerable from the doc'd command.)

Levers landed inside the fusion: in-place band-sliced retention (killed the
alloc/concat churn: 1152² marginal 60.3 → 29 ms), f32 `#[autoversion]`
combine + row-band rayon + f32 spread (576² marginal 14.7 → 9.0 ms).
Context against the pre-fusion state: the standalone basic density was
36.8 ms at 576² — the fused call now delivers **score + map in 14.8 ms
total**, 2.5× less than the old map alone.

**TARGET (marginal ≤ 1.1× the fold's marginal): NOT met — measured floor
5.7× @576² / 2.2× @1152².** The fold's marginal is nearly free because its
per-pixel fold runs INSIDE the strip kernels on cache-hot bands and needs no
pooled scalars; the attribution combine structurally needs post-scale pooled
scalars (the C1 ordering problem), so it pays retention traffic + a separate
combine pass. Remaining levers, honest estimates (per-section timing
`ZENSIM_ATTR_PERF=1`: 1152² = retention ~8-11 ms + combine block ~18 ms
(≈6 ms of which is the hf pre-pass) + SAT 1.7 ms):

1. **Reference-side hf-sum caching** (`Σ(src−mu1)²`, `Σ|src−mu1|` are
   distortion-independent — computable once per `PrecomputedReference`):
   −5-6 ms @1152² → ratio ~1.8×. API-additive; next chunk.
2. **In-kernel fold for the MEAN slots** (coefficients `−s_k/N` are known
   up front; only p-pools + hf need post-scale scalars): moves the dominant
   MSE/mean mass to fold-marginal cost; the deferred slots then need only
   sd-plane retention. Est. ratio ~1.3-1.5×; deeper kernel surgery.
3. **Stale-scalar single-pass** (codec-loop iteration n uses iteration
   n−1's pooled scalars → everything folds in-kernel): the true ≤1.1×
   endpoint; changes steering semantics between iterations — needs its own
   coherence gate; C3b candidate. _[SHIPPED 2026-08-01 as
   `AttributionSession` + `compute_with_ref_score_and_attribution_stale`
   (zensim `326185e9`): the in-strip fold IS fold-marginal-class, but the
   shared spread+SAT tail leaves the measured floor at 1.84×@576² /
   1.40×@1152² — status, decomposition + next levers in
   `docs/PLAN_LOOP_STEERING_69.md` "#70 status (2026-08-01)"; loop-quality
   staleness measured free a fourth time.]_

## Shape (b) — 924-MLP class: measured, not fused (per constraint)

The 924 scalar's planes live inside `compute_folded720_append_features_streaming`
(another session's domain; "do not restructure the extractor"). Fusing would
need the same retention hooks inside that walk. Measured today at 576²: 924
scalar features ≈ the extractor call, + standalone full density 95-98 ms
(C2b). The v1-fused machinery here covers the basic block's share when a
924 loop uses a v1-class steering profile; full 924 fusion = extractor-side
retention hooks, listed as the follow-up requiring that session's sign-off.
