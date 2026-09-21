# DVIFM as a spatial-steering substrate — steer lane record

Date: 2026-09-20 · Lane: `dvifm-steer` · Status: **COMPLETE — 482 pairs,
all gates passed**

Companion data: `dvifm_steer_2026-09-20.json` (aggregate), study output at
`/mnt/v/output/zensim/dvifm-steer-2026-09-20/` (`pairs.jsonl`, `blocks.bin`,
`rects.bin`, `summary.json`).

## Question

P3 asks whether a DVIFM-derived spatial block map is a better steering
substrate than the current attribution-density map for JPEG XL spatial
guidance — i.e. whether the DVIFM block field (a) predicts where finite
edits raise the scalar score (S2), and (b) agrees with independent
perceptual-error judges about WHERE the error mass lives (S3). This lane
does NOT implement or claim a codec loop.

## S1 — the map (what was built)

The DVIFM steering field is emitted by the SAME owner that produces pooled
DVIFM features — no second map pipeline:

- `zensim/src/dvifm.rs`: `block_terms()` was factored out of `pool_block()`
  so the field's `ε_b = v_b · m_b^P` is literally the pump's own arithmetic.
  `block_field()` produces per-level `ε`, `v`, `m` vectors plus `f1_sum`
  (bitwise the pooled F1 numerator — asserted in tests). `paint_block_field()`
  paints `ε_b/25` uniformly over each full 5×5 block (partial border blocks
  dropped, exactly as pooling drops them). Block cache widened from
  `feature = "training"` to `any(training, test)`; `take_block_cache()` now
  also returns per-level `(w,h)` dims.
- `zensim/src/attribution.rs`: `AttributionResult` gains
  `query_rect_frac()` (fractional-edge, area-weighted cut-block queries —
  a rect that cuts a block receives `ε_b · covered_fraction`); derives
  `Clone, Debug`; `from_f64_canvas` stays `pub(crate)`.
- `zensim/src/feature_v2.rs`: `DvifmBlockCacheOut` carries `grid`, `dims`,
  `levels` (the records' level dims, needed to paint correctly).
- `zensim/src/research.rs`: `Request::collect_dvifm_fields(bool)` →
  `Extraction::dvifm_fields()` → `DvifmFieldMap { plane, levels }`, each
  `DvifmLevelField` carrying the painted `eps_map` (an `AttributionResult`),
  block-constant `vis`/`err` inspection planes, raw per-block vectors,
  `f1_sum`, and `query_eps`/`query_eps_scale0` rectangle queries. Gated
  `all(training, custom-profiles)`; the `training`-only request bit compiles
  in every feature combination.

Gate tests (all in `dvifm.rs`, 47/47 pass under `--features training`):

- `f1_sum` is **bitwise** the pooled `LevelSums::f1` (field cannot drift
  from the emitted feature).
- Painted full-block `query_rect` == `ε_b`; fractional half-block ==
  `ε_b/2` (cut-block rule).
- Identity pair → all-zero `ε`/`m` fields.
- Strip/streaming parity and "cache disabled unless requested" unchanged.

## S2 — finite-edit gain prediction (definition)

Rectangles are pixel-size grids at five sizes {16, 32, 64, 128, 256} px
(the prompt's size axis; a size runs only where `w,h ≥ 2·size`), capped at
50 rects/size/pair by even stride (≈250 interventions/pair max). Rects are
pixel-aligned, not lattice-aligned — predictions use the fractional
area-weighted queries; edits are pixel-exact. For each rect:

- **Actual gain** `ΔS` = score of the distorted image with the rect's
  pixels replaced by reference pixels, minus the base score — the same
  finite-intervention protocol as `diffmap_block_coherence`, scored through
  `BakeScorer::compute` on the shipped **B bake**
  (`b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin`, sha256 in
  summary.json).
- **DVIFM prediction** = `Σ_l query_eps_scale0(rect)/n_l` — the level-wise
  F1-normalized ε mass under the rect (units of pooled F1); per-level arms
  reported separately (the "by pyramid level" axis).
- **Control** = `AttributionResult::query_rect(rect)` from
  `BakeScorer::compute_with_ref_and_attribution(..., bin = 1)` — the current
  steering density map (the `SteeringSession` integrity-head gate is
  bypassed so every pair yields a map; `has_corruption_gate` is recorded).
- **Also reported**: `ScoredAttribution::refinement_gain(rect)` (the
  non-additive prediction), per-pixel SSE (the codec default bar).

Metrics per (pair, size): SROCC / Pearson / Kendall-τ vs `ΔS`,
top-quartile precision, least-squares (slope, intercept) calibration, and
**calibrated relative error** `|ŷ − Δ| / (|Δ| + 0.1·median|Δ|)` where `ŷ`
is the per-(pair, size, predictor) least-squares fit — median and p90
pooled over rects by size and by DVIFM level.

**Additivity** — up to 12 disjoint, non-edge-adjacent 64-px rect pairs per
image: `residual = ΔS(A∪B) − (ΔS_A + ΔS_B)`; a strictly local+additive
score predicts 0.

**Pyramid leak** — up to 4 of the 64-px edits per image: re-run the field
extraction and the attribution owner on the edited image, and report the
fraction of `|Δfield|` mass deposited OUTSIDE the edited rect (per DVIFM
level's own block grid; per-pixel for the attribution density). A
perfectly local map scores 0; pyramid spreading and global-context maps
leak.

## S3 — two-judge severity agreement (definition)

Per-block severity on the same lattice, ranked against:

- **fast-ssim2 0.8.2 field** — scalar replication of `ssim_map_inner` /
  `edge_diff_map_inner` over the public `Ssimulacra2Reference` scale planes
  (`new(dist)` runs the identical pipeline as `compare_with`'s distorted
  side; `σ12 = blur(img1·img2)` via public `Blur`). Per-pixel severity is
  the **sum-preserving** decomposition of the pooled score's weighted
  terms: L1 terms contribute `w·x_i/N`, L4 terms `w·x_i⁴/(N·r4³)`. Each
  scale's mass is splatted over its valid scale-0 footprint (border cells
  clipped) so `Σ density = weighted score mass` exactly. Parity gate:
  replicated `Msssim::score()` vs `Ssimulacra2Reference::compare` must be
  < 1e-2 absolute per pair (observed max in summary.json).
- **butteraugli 0.9.3** — `butteraugli(..., compute_diffmap=true)` diffmap
  summed per block.
- **Agreement set** = blocks in the top 10% of BOTH judges.

Metrics per pair: SROCC/Kendall of each candidate map vs each judge, judge
mutual SROCC, agreement-set precision@k (k = |agreement|).

## Sample

`/mnt/v/output/zensim/joint-core-v1/pairs/pairs_core.tsv` (52,963 rows;
sha256 in summary.json). SDR legs only (`hdr` excluded — the B bake and
DVIFM default plane are SDR). Deterministic stratified caps per
(leg, codec): fresh_imazen26 56/codec × 4, fresh_safesyn 18/codec × 4,
cid22 {aom 20, cld_avif 20, cld_heic 15, cld_jp2 15, cld_webp 15,
libjxl 15, mozjpeg 10, vis_avif 6}, human {images 30,
distorted_images_png 30}, konfig jnd-levels 10 → **487-cell cap, 482 pairs
done** (5 cells undersubscribed), stride-sampled over
(band, ref_basename, q, dist) sort inside each cell. Zero decode/score
failures (hard-error policy).

## Results

n = 482 pairs, 0 skipped, 0 failures; ssim2 adapter parity max 3.7e-3
(gate < 1e-2). Win counts are per-pair SROCC comparisons.

### S2 — finite-edit ΔS prediction (within-size; the honest comparison)

| size | pairs | srocc dvifm | srocc attr | srocc refgain | srocc sse | dvifm>attr |
|-----:|------:|------------:|-----------:|--------------:|----------:|-----------:|
| 16   | 482   | 0.176       | **0.558**  | 0.581         | 0.458     | 58/482     |
| 32   | 466   | 0.203       | **0.579**  | 0.620         | 0.510     | 73/466     |
| 64   | 410   | 0.208       | **0.567**  | 0.638         | 0.540     | 85/410     |
| 128  | 338   | 0.208       | **0.539**  | 0.634         | 0.562     | 93/338     |
| 256  | 184   | 0.097       | **0.553**  | 0.663         | 0.588     | 36/184     |

Calibrated relative error, median/p90 (dvifm | attr | refgain | sse):

| size | dvifm        | attr         | refgain      | sse          |
|-----:|--------------|--------------|--------------|--------------|
| 16   | 0.62 / 5.45  | 0.51 / 4.36  | 0.48 / 3.81  | 0.49 / 3.68  |
| 32   | 0.49 / 3.37  | 0.39 / 2.64  | 0.36 / 2.41  | 0.36 / 2.13  |
| 64   | 0.38 / 2.23  | 0.29 / 1.66  | 0.26 / 1.45  | 0.27 / 1.53  |
| 128  | 0.27 / 1.30  | 0.20 / 0.91  | 0.17 / 0.80  | 0.19 / 0.82  |
| 256  | 0.13 / 0.48  | 0.10 / 0.39  | 0.09 / 0.34  | 0.09 / 0.31  |

- **DVIFM loses S2 at every size** — below the attribution map, below
  `refinement_gain`, and below raw SSE. It wins only 12–28% of pairs per
  size and 150/482 pooled. Within a size (rect area held constant), ε's
  visibility×error ordering does not track which fixes raise the B score.
- Pooled-over-sizes SROCC (dvifm 0.638, attr 0.729, sse 0.713) is
  reported for completeness only — it is dominated by the area confound
  (bigger rects carry more mass and more gain; `mean ΔS` runs
  0.029 → 3.63 from 16 → 256 px) and is not evidence of discrimination.
- **By pyramid level** (calibrated relerr, median/p90): L0 0.43/2.96,
  L1 0.46/3.38, L2 0.47/3.63, L3 0.49/3.87, L4 0.49/3.98 — and within-size
  SROCC level arms: L0 0.28–0.37 vs ≤0.17 for L1 and ≈0 (or negative)
  for L3/L4 at every size. The little score-gain signal DVIFM has lives
  entirely in the finest scale — consistent with the visibility weighting
  doing its perceptual (not model-derivative) job at coarse scales.
- **Additivity (64 px disjoint pairs, n=4760):** joint/sum median 1.002,
  |rel residual| median 0.007, p90 0.125 — the B bake is nearly additive
  over disjoint fixes at this scale; attr's additive-map semantics are
  not where its S2 advantage comes from.
- **Pyramid leak (fraction of |Δfield| outside the 64 px edit):**
  DVIFM L0–L4 means 0.008 / 0.009 / 0.016 / 0.048 / 0.032 (p90 ≤ 0.14 —
  the downsample-halo fringe), **attr density mean 0.571, p90 0.795**.
  The current steering map is a global-context quantity: editing one
  rect moves most of its density change *outside* that rect. DVIFM is a
  nearly local field.

### S3 — independent two-judge severity agreement

| map      | srocc vs ssim2 | srocc vs butteraugli | vs judge-agreement set | wins vs attr |
|----------|---------------:|---------------------:|-----------------------:|--------------|
| dvifm ε  | **0.327**      | **0.338**            | 0.130                  | 355/482, 329/482 |
| attr     | 0.209          | 0.218                | 0.094                  | —            |
| judges ∗ | 0.366 (mutual) | —                    | —                      | —            |

- **DVIFM beats attr against both judges**, 0.33 vs 0.21 on each —
  ~90% of the judges' own mutual agreement (0.366) on ssim2. Per-pair
  wins: 74% (ssim2), 68% (butteraugli).
- By content group (d_ssim2/a_ssim2/d_butter/a_butter): **photo** n=410
  0.33/0.20/0.33/0.21 — DVIFM clear; **doc** n=20 0.47/0.35/0.54/0.35 —
  DVIFM clear; **lineart** n=24 0.23/0.17/0.31/0.19 — DVIFM; **screen**
  n=20 0.35/0.40/0.46/0.41 — mixed (attr edges ssim2, DVIFM edges
  butter); **ai** n=8 0.21/0.17/0.14/0.10 — DVIFM, thin.
- DVIFM level arms vs ssim2: L2 strongest (0.400); vs butter: L0
  strongest (0.390) — the pooled ε sits between its own arms, as designed.

## Verdict

**Is the DVIFM map more faithful (S2) and better aligned with
independent judges (S3) than the map we steer with today?**

Split verdict, and the split is the finding:

- **S2: no.** The attribution-density map predicts which rect fixes
  raise the B score far better at every intervention size (SROCC
  0.54–0.58 vs 0.10–0.21; relerr ~25% lower at every size). This is
  partly circular — attr/refgain derive from the model's own gradient —
  but the result stands: as a *score-gain steering* substrate DVIFM is
  strictly worse, even below raw SSE.
- **S3: yes.** DVIFM agrees with the two-judge severity truth ~55%
  better than attr does (0.33 vs 0.21 on both judges, vs the judges'
  own 0.37 mutual ceiling), and it is a nearly *local* field (≤5% leak)
  while the attr map is global-context (57% of its Δmass lands outside
  an edit). For P3's stated goal — steering that adds *independent*
  perceptual value rather than re-optimizing the metric's own gradient —
  DVIFM is the more honest substrate; attr is the better score mirror.

**Smallest change that lets a codec loop consume it:** none of it is
new machinery. The pooled-DVIFM walk already runs inside the 986-feature
extraction the scorer performs — `Request::collect_dvifm_blocks` already
retains the same block records the field is built from. The minimal
wiring is: have the steering session retain the DVIFM block cache (or
call `research::extract` with `collect_dvifm_fields(true)` once per
pair) and consume `DvifmLevelField::query_eps_scale0(x0,y0,x1,y1)` —
F1-normalized ε mass under any scale-0 rect — per candidate region.
Additive semantics are empirically safe at 64 px (joint/sum ≈ 1.00 —
only that size was additivity-tested) and the field is local enough
that rect-level consumption needs no halo bookkeeping below level 3.

## Limitations

- Interventions are rect replacements, not codec quantizer changes — this
  lane measures spatial-gain structure, not RD or bytes (no codec loop).
- `refinement_gain` and the density map are the CURRENT owner — they are
  the control, not the ceiling: both carry documented removal-based
  approximations. The integrity-head gate was bypassed (0/482 would have
  fired anyway).
- Judge fields measure different quantities (ssim2 = score-mass share,
  butteraugli = diffmap amplitude); agreement-set membership is the
  conservative two-judge reading, and the judges' own mutual agreement
  (~0.37 SROCC) bounds what "agrees with judges" can mean.
- Calibrated relative error uses a per-(pair,size) least-squares fit —
  it is an oracle-calibration view (rank quality is calibration-free);
  a deployed predictor would carry a fixed global calibration.
- The `konfig` leg's `jnd-levels` distortions are JND-fraction synthetic
  rungs, kept for breadth at 10 rows.
- `unsupported_ids` per pair lists bake feature slots the retention
  attribution marked unsupported on that pair — informational provenance,
  not skipped data (all 482 pairs have complete metric rows).
- Disk floor breached during the lane (48–67 GB < 80 GB): outputs were
  held to JSONL + f32 sidecars (~150 MB total); no full per-block caches.

## Provenance

- zensim source: jj change `wtwzoykq` on top of
  `mywlzowt` — bake/pairs sha256 in `summary.json`. The commit also
  carries the concurrent loss-lane v2 BlockRec schema (line-interleaved
  in the same files; see commit message).
- Bake: `b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin`
  (shipped `ZensimProfile::B` bytes).
- fast-ssim2 0.8.2 (crates.io; `imgref` feature), butteraugli 0.9.3
  (`avx512`), zenstats (`parallel`).
- Harness: `zensim-bench/examples/dvifm_steer_study.rs`, built with
  `--features "training zen-decode"`.
