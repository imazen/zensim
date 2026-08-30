# The fold becomes the engine — design note (started 2026-08-30)

Lane: give the streaming 944 fold everything the buffered walk still owns, so
that retiring buffered becomes a deletion rather than a rewrite. This note is
**stage 1**: the structure, the attachment points, the memory shapes, and the
complete list of items this lane intends to add. Nothing here is a measurement
claim; measurements land in the stage sections appended below as each stage is
gated.

Predecessor: `benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md`
(§5 four blockers, §15 the retirement checklist). That lane closed blocker 2
(option C, `56bbcda2`) and left blockers 1, 3, 4, 5 — **all additive API or
oracle work, none of them perf**. This lane takes those four.

Explicitly **not** this lane: multithreading (bounded ~1.2×, era-gated —
§14 of the predecessor), any change to a summation grouping, any change to a
byte of any output. Every gate below is bit-exactness against the path that
ships today.

---

## 0. Vocabulary, because two words are overloaded

| name | what it is | entry |
|---|---|---|
| **buffered** | whole-image XYB pyramids for both sides, band-processed per scale | `metric.rs:3145` `compute_with_config_inner` → `streaming.rs:862` `compute_multiscale_stats_streaming` |
| **the fold** | the streaming 720/924/944 walk; rolling planes, no materialised pyramid | `feature_v2.rs:6517` `foldapp_streaming_walk` |
| **the pyramid cache** | `PrecomputedReference` — 3 XYB planes × 4 scales for ONE image | `streaming.rs:2728` |

The third row is the reframing this note rests on and is developed in §5:
**`PrecomputedReference` is not buffered-walk state.** It is an image-side
pyramid cache with no walk in it, and both engines can consume it.

---

## 1. What is actually in the way (re-derived from source, 2026-08-30)

### 1.1 The fold already emits v1's 372 — bit-exactly

`V2NewFeatureToggles { v1_pools: V1PoolsMode::Full }` makes the folded walk
replay v1's extended strip section per v1-aligned band, so `f0..372` of the
folded vector is v1's 372-feature extraction. Under option C
(`blur::pyramid_plane_stride(w) == w`, `blur.rs:4198`) that equality is
**bit-exact at every width**, gated by
`feature_v2.rs:11380 v1_372_bit_exact_to_fold_at_every_width` over 19
geometries including the three `h = 93` cells that were the last residual
under the option-A workaround.

So the extraction half of "fold-backed scoring" is done and gated. What is
missing is everything between a feature vector and a `ZensimResult`.

### 1.2 The cheapest fold request for a v1 score is `v1_only + pools Full`

`V2NewFeatureToggles::v1_only` (`feature_v2.rs:1489`) skips every v2-era block
*and its upstream work* — the four `box_blur_v_from_copy` sweeps and the v2
activity chain — because `fold_v1_basic_bands` takes the H-blurred planes and
computes its own activity internally. It is documented as **pure
compute-skipping**: the emitted slots are bit-identical to the same request
with it off (`folded_v1_only_matches_full_walk`). The predecessor measured it
as **53 % of the walk removed** (§10.1).

Every shipped profile scores from at most 372 inputs (`PROFILE_B`:
`extended_features: true`, `compute_iw_features: true`, `num_scales: 4`, a
372-input linear bake; `PROFILE_PREVIEW_V0_*`: 228, no bake). So the fold
request that backs `score()` is exactly:

```rust
V2NewFeatureToggles { v1_pools: V1PoolsMode::Full, v1_only: true, ..default() }
```

**Registered tension, flagged for approval.** `v1_only` is `#[doc(hidden)]`
and its own doc says "TEST/BENCH INSTRUMENTATION — NOT A PRODUCT MODE … there
is no 372-only product path". That statement was made about the *datagen
extractor* modes. It is not true of `score()`: the shipped metric is a
372-input bake and always has been. Backing `score()` with the fold makes
`v1_only` a production compute request — as an internal request, not as a
public toggle. The alternative is to run the whole 944 walk and read the first
372 slots, which computes ~2× the work for a bit-identical answer. This note
takes `v1_only`, and names the change rather than sliding it in.

### 1.3 The structural blocker nobody has named yet: the feature gate

`feature-regime-v2` is **not** in zensim's default feature set
(`zensim/Cargo.toml:37`: `default = ["avx512", "imgref", "threads",
"deprecated-profiles", "candidate-profiles"]`). The whole fold —
`feature_v2`, `feature_v2_stream`, every `compute_folded720*` entry — is
behind it.

Consequence for retirement, and it is a hard one: **a default `cargo add
zensim` build contains no fold at all.** Buffered cannot be deleted while
that is true, no matter how good the parity gates are. Making
`feature-regime-v2` default-on (or unconditional) is therefore a *prerequisite*
of retirement and belongs on the checklist as its own line — it is not implied
by any of blockers 1/3/4/5. This lane's fold-backed code is gated the same way,
so a default build is byte-for-byte unaffected by everything below.

### 1.4 `num_scales` is a real constraint

The buffered walk takes `config.num_scales` from the profile. The fold is
hard-wired to `crate::NUM_SCALES` (4) — `foldapp_streaming_walk` sets
`let n_scales = crate::NUM_SCALES;` and the producer builds exactly
`NUM_SCALES` rolling planes. Every shipped profile uses 4, and
`min_pyramid_dim_for_scales(4) == MIN_PYRAMID_DIM == 64`, so the two paths'
sub-64 reflect-pad decisions coincide exactly at 4. Away from 4 they do not.

**Rule this lane adopts:** the fold-backed engine is available only when
`config.num_scales == crate::NUM_SCALES`; any other value falls back to
buffered rather than silently scoring a different pyramid. That fallback is
also the honest statement for the retirement proposal — a `training`-feature
consumer that sets `num_scales != 4` in a `ZensimConfig` keeps buffered until
the fold is generalised.

---

## 2. Fold-backed scoring — structure

### 2.1 Where the buffered path splits, and where the hooks live

Today, reading `metric.rs` top-down:

```
Zensim::compute / compute_with_codec_hint
  ├─ HDR routing → compute_pu_linear
  ├─ validate_pair, check_within_max_pixels, check_stop
  ├─ config_from_params(params, parallel)
  ├─ compute_with_config_inner ─────────────── EXTRACTION + LINEAR SCORE
  │    ├─ needs_pyramid_pad → reflect_pad_for_scales           (shared)
  │    ├─ images_byte_identical → identical_result             (shared)
  │    └─ compute_zensim_streaming_stoppable
  │         ├─ compute_multiscale_stats_streaming → (Vec<ScaleStats>, mean_offset)
  │         └─ combine_scores  ── assemble 372 ─┬─ sanitize non-finite
  │                                             ├─ Σ f_i·w_i  (i < min(228, |w|))
  │                                             ├─ ÷ n_scales
  │                                             └─ squash → score
  ├─ apply_mlp_scoring_with_codec ──────────── BAKE FORWARD + OUTPUT SPLINE
  └─ .with_profile(self.profile)
```

Only the middle box is buffered-specific. Everything above and below it is
already engine-agnostic. So the fold-backed path is:

```
compute_with_config_inner
  ├─ needs_pyramid_pad → reflect_pad_for_scales                (SHARED, unchanged)
  ├─ images_byte_identical → identical_result                  (SHARED, unchanged)
  └─ match engine
       ├─ Buffered → compute_zensim_streaming_stoppable        (unchanged)
       └─ Fold     → fold_engine::compute_fold_backed
                       ├─ foldapp walk, v1_only + pools Full   (the fold)
                       ├─ mean_offset from the scale-0 strips  (§2.3)
                       ├─ truncate 720 → 228/300/372 by config (prefix)
                       └─ score_v1_layout_features(...)        (SHARED, §2.2)
```

`apply_mlp_scoring_with_codec` — which is where the **bake forward, the
per-sample-α / hybrid head, the tanh pin, the PCHIP output spline, the
per-codec affine and the clamp/soft-clamp/extrapolate disposition all live**
(`metric.rs:3307`, `:4132`, `:4249`) — is untouched. It consumes
`ZensimResult { score, raw_distance, features, mean_offset }` and the original
`(width, height)`. **The fold attaches under it, not into it.** That is the
whole reason blocker 1 is additive: no scoring machinery moves.

`classify` (`metric.rs:2605`) calls `compute_with_params` → the same
`compute_with_config_inner`, plus `compute_delta_stats` (a pure per-pixel pass
over the two `ImageSource`s, no pyramid, no walk). So classify is fold-backed
for free the moment `compute_with_config_inner` routes.

The diffmap entries (`diffmap.rs:758/:925`) call
`streaming::compute_zensim_streaming_with_ref_and_diffmap*`, which is the
*ref-cached* buffered walk with a diffmap sink — they attach at §3, not here.

### 2.2 The one refactor: `combine_scores` splits in two

`combine_scores` (`metric.rs:4898`) does two unrelated jobs: it assembles the
372-layout vector out of `ScaleStats`, and it turns a 372-layout vector into
`(raw_distance, score)`. The second job is engine-independent and both engines
need it.

Extract, verbatim, no arithmetic change:

```rust
pub(crate) fn score_v1_layout_features(
    features: &mut [f64],     // sanitised in place, exactly as today
    weights: &[f64],
    config: &ZensimConfig,
    n_scales: usize,
) -> (f64 /*score*/, f64 /*raw_distance*/)
```

containing, in this order and no other: the non-finite→0 sanitise loop, the
`Σ features[i]·weights[i]` over `i < min(basic_total + peak_total,
weights.len())` **in ascending `i`**, the `÷ max(n_scales,1)`, and the
`bounded_squash ? bounded_score_squash : distance_to_score_mapped` branch.
`combine_scores` then calls it. Per the NO-DUPLICATE-IMPLEMENTATIONS rule this
function is the single owner of "score a v1-layout feature vector"; the fold
path calls it rather than restating the dot product.

Gate: the whole existing suite, plus `v1_golden_bytes` — the extraction is
untouched, so any drift is a refactor bug.

### 2.3 `mean_offset` — the only quantity the fold does not already produce

`ZensimResult::mean_offset()` is public and is asserted bit-exactly by
`tests/cross_platform.rs:464-471`. It is **not** an input to any score; it is
carried for consumers. The buffered path computes it in
`compute_xyb_mean_offset` (`streaming.rs:737`) as:

```
chunks = [0, 64, 128, …]                      // 64 scale-0 rows each
per chunk, per channel c:  acc = Σ_y ( Σ_{x<width} (src[y][x] − dst[y][x]) as f64 )
offset[c] = ( Σ_chunks acc[c] ) / (width·height)
```

Two properties make this reproducible from the fold, and both are worth
stating because they are what makes the gate bit-exact rather than
tolerance-based:

1. **It is already thread-invariant.** The rayon arm is
   `row_indices.into_par_iter().map(per_chunk).collect()` — an
   order-preserving collect — and the final reduction walks that `Vec` in row
   order serially. Parallelism changes nothing about the summation order.
2. **It decomposes per row.** `acc` is a plain left-to-right sum of per-row
   f64 sums, so accumulating `row_sum[y][c]` as the fold's scale-0 strips go
   past, then performing the chunk reduction at the end **in exactly the loop
   nest above**, is bit-identical — not "within an epsilon".

Cost: one `Vec<[f64; 3]>` of `height` entries (55 KB at 2304², vs the walk's
163 MB RSS) plus one f64 subtract-and-add per scale-0 pixel per channel, taken
while the rows are already in the producer's rolling plane and cache-hot. The
fold's scale-0 strips are emitted once each with `info.y0 .. info.y0 +
info.strip_h` as the interior row range, so every row is visited exactly once.

### 2.4 Truncation to the config's width

`config_from_params` derives the feature width from the profile:
`(extended, iw) = (true, true) → 372`, `(true, false) → 300`,
`(false, _) → 228`. v1's layout is `[0,156) basic · [156,228) peaks ·
[228,300) masked · [300,372) IW`, so a narrower config is a **prefix** of the
wider one and truncation is the whole conversion.

That is a claim about the buffered path, not an assumption: it requires that
buffered with `extended_features: false` produces bit-identical `f0..228` to
buffered with `extended_features: true`. Gated explicitly (§ stage-2 gates)
rather than assumed, because if it is false then the fold-backed narrow path
would be comparing against a different quantity.

### 2.5 Engine selection

```rust
#[cfg(feature = "feature-regime-v2")]
pub enum ScoringEngine { Buffered, Fold }

#[cfg(feature = "feature-regime-v2")]
impl Zensim {
    #[doc(hidden)] pub fn with_engine(self, engine: ScoringEngine) -> Self;
    #[doc(hidden)] pub fn engine(&self) -> ScoringEngine;
}
```

`Buffered` is the default and a default build cannot name `ScoringEngine` at
all. It is `#[doc(hidden)]` for the same reason `v1_only` is: the integration
tests that hold the parity gate live outside the crate and cannot reach
`pub(crate)`, and the struct-update problem that blocked `pub(crate)` on
`v1_only` applies to any constructor-side alternative. This is the "internal
constructor" the brief asks for, spelled as far toward private as the crate's
own precedent allows.

---

## 3. The ref-cached fold form

### 3.1 What the buffered M1 precompute caches

`Zensim::precompute_reference` → `PrecomputedReference::new` → the reference's
XYB pyramid, all `NUM_SCALES` levels × 3 channels, at `(padded_width, height)`
per level. Under C `padded_width == width`, so:

```
bytes = 3 planes · 4 B/f32 · w·h · (1 + 1/4 + 1/16 + 1/64) = 15.94 · w·h
```

5.3 MB at 576², 21.2 MB at 1152², 84.6 MB at 2304². `compute_with_ref` then
runs the buffered walk against it, skipping the reference decode, the sRGB→XYB
conversion (the cube roots) and the three downscale levels — **but not any
blur**, because the buffered `*_with_ref` walk re-blurs the reference side per
band.

### 3.2 The fold's ref-cached form caches the same thing

The insight that keeps the API small: **the fold's source-side needs are
exactly `PrecomputedReference`'s contents.** The producer
(`feature_v2_stream.rs:360 produce`) fills side-0 scale-0 rows via
`convert_source_to_xyb_into_slices` and cascades `downscale_2x_into` — the
same two functions `build_v2_ref_scales` uses to build the cache, in the same
order. So there is no new reference type: the fold learns to *read* the
existing one.

Producer change (the actual work in this stage):

```rust
enum RefFeed<'a, S: ImageSource> { Image(&'a S), Cached(&'a PrecomputedReference) }
```

`produce()` branches on it for side 0 only: `Image` converts as today,
`Cached` copies the row block out of the cached plane at that scale, and the
downscale cascade for side 0 is skipped entirely (the deeper levels are
already in the cache). Side 1 (distorted) is untouched. Bit-exactness is by
construction — the bytes copied in are the bytes the conversion would have
produced — and is gated against N independent `compute` calls, not argued.

Saving, tier 1: the whole reference front-end per compare. The predecessor
measured the producer at ~8 % of the walk for both sides, so the ceiling here
is ~4 %, plus the reference's share of the front-end's memory traffic. Small,
and it will be reported as measured rather than as a target.

### 3.3 Tier 2 — cached reference moments — is specified, not assumed

The bigger prize is the reference-only blur work: `mu1 = blur(src)`,
`activity = blur(|src − mu1|)`, `bs2 = blur(src²)`. `V2PreparedReference`
(`feature_v2.rs:5283`) already carries exactly this shape for the
*materialised* v2 walk — `moments: Option<Vec<[V2RefMoments; 3]>>` with
`mu1` + `activity` per channel-scale, filled by `fill_ref_moments` which
**replays the strip walk** so the cached values are bit-identical to what the
kernels would recompute.

Memory is the reason this is tier 2 and not tier 1: three more full pyramids
is `3 × 15.94 = 47.8` bytes/pixel, i.e. 254 MB at 2304² against the fold's
measured 163 MB peak RSS — it would make the streaming walk cost *more*
memory than the buffered one it is replacing, which inverts the fold's whole
justification. Tier 2 therefore ships only if measurement says the saving is
large enough to be worth an explicit opt-in, and it ships as an opt-in
(`prepare_reference_with_moments`) either way, never as the default.

### 3.4 Diffmap attaches here

`compute_with_ref_and_diffmap*` is `compute_with_ref` plus a per-scale sink.
Once the fold has a ref-cached compare, the diffmap entries route the same
way; the diffmap's own weighting maths is untouched.

---

## 4. Attribution — and a correction to how blocker 4 was framed

The predecessor recorded blocker 4 as "attribution's basic canvas is
buffered-native". Read at source, that is true of the *type* and not of the
*walk*, and the difference decides how much work retirement is:

* `attribution.rs:1378 basic_canvas_trimmed` → `basic_attr_prep` (converts the
  distorted to XYB at the cached dims) → `build_attribution_canvas` →
  `build_attribution_into_sink`.
* `build_attribution_into_sink` calls **only `crate::blur`**. It does not call
  `compute_multiscale_stats_streaming`, `process_scale_bands`, or any other
  walk function. Its buffered dependency is `&PrecomputedReference` — read for
  `.scales` — plus `convert_source_to_xyb` and `downscale_3_planes`, which are
  front-end helpers, not the walk.

So `compute_attribution_density*` **survives the deletion of the buffered walk
untouched**, provided the pyramid cache and the two front-end helpers survive.
It is a *materialised* computation by nature — it emits a full-resolution f64
canvas — so there is nothing to stream it into. The work this lane owes here
is therefore: prove that statement by construction (route the entries through
the fold-backed reference and gate the canvas bit-identical), not port a walk.

The **fused** compare is the genuinely walk-bound one.
`compute_with_ref_score_and_attribution*` (`attribution.rs:3210/:3234/:3551/
:3577`) calls `streaming::compute_zensim_streaming_with_ref_and_attr_planes` /
`…_and_attr_fold` — the buffered band walk with an `on_scale` callback that
folds attribution planes *in strip*, which is what made C3a's 14.8 ms
score+map at 576² possible. Its per-band geometry (`BAND_ROWS = 32` ≡
`streaming::STRIP_INNER`) is chosen to be bit-compatible with the buffered
extractor. Re-hosting that on the fold's own strip geometry (`STRIP_ROWS`,
`HALO_P`) is not a re-point; it is a re-derivation of the in-strip fold against
a different tiling, and it carries real bit-risk. It is scoped last, measured
against the C3a numbers, and reported honestly if it does not land.

---

## 5. `zensim-gpu`'s oracle

The GPU crate validates its CubeCL kernels against `compute_extended_features`.
Two independent things it could mean, and they have different answers:

* If the oracle wants **v1's 372 features**, it can call the fold-backed
  extraction the moment §2 lands, because that is exactly what the C gate
  proves is bit-identical.
* If the oracle wants **a score**, `score_features_with_profile` (already
  exported for precisely this purpose — "the entry point alternative feature
  backends (e.g. `zensim-gpu`) use to produce a bit-exact CPU-equivalent
  score") is already engine-independent.

Either way the change is on the zenmetrics side and is a one-line re-point
plus a feature flag, gated by the GPU crate's existing parity suite passing
unchanged. If zenmetrics is claimed by another lane, this becomes a registered
proposal rather than a commit.

---

## 6. Complete list of items this lane intends to add

**All additive. No existing public signature changes. Every item below is
`#[cfg(feature = "feature-regime-v2")]`, so a default build's public surface is
byte-for-byte unchanged.**

| # | item | kind | stage |
|---|---|---|---|
| 1 | `zensim::ScoringEngine` (`Buffered` \| `Fold`) | `#[doc(hidden)] pub enum` | 2 |
| 2 | `Zensim::with_engine(self, ScoringEngine) -> Self` | `#[doc(hidden)] pub fn` | 2 |
| 3 | `Zensim::engine(&self) -> ScoringEngine` | `#[doc(hidden)] pub fn` | 2 |
| 4 | `Zensim::precompute_reference_with_moments(&self, &impl ImageSource)` | `#[doc(hidden)] pub fn` | 3, tier 2 only |
| 5 | `PrecomputedReference::has_cached_moments(&self) -> bool` | `pub fn` | 3, tier 2 only |

Internal-only (no public surface): `metric::score_v1_layout_features`,
`fold_engine::{compute_fold_backed, fold_mean_offset}`,
`feature_v2_stream::RefFeed`, and the `engine` field on `Zensim`.

Items 4 and 5 ship **only if tier 2 measures worth its memory** (§3.3); if it
does not, the list is items 1–3 and nothing else. Any item added beyond this
table gets appended here with its reason before it is committed.

---

## 7. Gates, per stage

| stage | gate |
|---|---|
| 2 | `score()`/`compute*`/`classify` fold-backed vs buffered: `score`, `raw_distance`, all 372 `features`, and `mean_offset` **bit-identical** (`to_bits()`), on `GOLDEN_SYNTHETIC` (64×64), `GOLDEN_NONTIGHT` (200×150), `GOLDEN_REAL` (96×96) and the width matrix, × {serial, rayon} × {1, 8, 16} threads. Plus the narrow-config prefix claim of §2.4. Plus the whole existing suite. |
| 3 | N distorted vs one prepared reference == N independent `compute` calls, bit-identical, all four fields. |
| 4 | `attribution_covers_expected_slots_per_width`, the plane-sum identities and the M3a instruments unchanged and passing; canvases bit-identical between engines. |
| 5 | the GPU crate's existing parity suite passes unchanged. |

"Bit-identical" everywhere means `f64::to_bits()` equality, not the golden
policy's tolerance. The tolerance exists for cross-*environment* drift; two
paths on the same box computing the same statistic over the same pixels have
no licence to differ, and §1.1 shows they do not.

---

## 8. Stage results

### 8.1 Stage 2 — fold-backed scoring (`3dcb777c`)

Landed as designed. The additive surface is items 1–3 of §6 and nothing else;
items 4–5 (tier-2 cached moments) were **not** built, because stage 3 turned
out not to need a new reference type at all (§8.2).

**Gate — `zensim/tests/fold_engine_parity.rs`, `to_bits()` equality on
`score`, `raw_distance`, every feature, and `mean_offset`:**

| test | matrix | result |
|---|---|---|
| `compute_is_bit_identical_across_engines` | 18 geometries × {serial, rayon}, profile B (372 + bake + spline) | PASS |
| `extended_features_are_bit_identical_across_engines` | same × {B → 372, `PreviewV0_2` → 300} | PASS |
| `golden_real_fixture_is_bit_identical_across_engines` | the 96×96 real-photo golden fixture, both entries | PASS |
| `both_engines_are_bit_identical_across_rayon_pool_sizes` | 4 geometries × rayon pools 1/2/3/8/16 — cross-engine parity at every count AND each engine thread-invariant | PASS |
| `identical_pair_short_circuit_is_shared` | the `mark_identical` payload | PASS |
| `classify_is_bit_identical_across_engines` | 3 geometries, result + classification | PASS |
| `fold_falls_back_on_a_weight_skipping_profile` | `PreviewV0_2::compute()` | PASS (falls back) |
| `fold_backed_fixtures_match_golden` (`be63ff39`) | the fold vs the **pinned golden arrays** — `GOLDEN_SYNTHETIC`, `GOLDEN_NONTIGHT`, `GOLDEN_REAL` | PASS |

The last row is the one the retirement rests on and is deliberately stated
against the frozen constants rather than against the buffered path: the two
walks agreeing proves nothing if they could drift together.

Whole suite: **374 passed, 0 failed**. Clippy clean at default features, at
`feature-regime-v2` alone, and at
`training,classification,feature-regime-v2,custom-profiles`.

**Design changes vs §2, with reasons:**

* `is_fold_backable` gained a fourth condition —
  `compute_all_features || extended_features`. `streaming::active_channels`
  drops a channel whose basic+peak weights are all ≈0 unless one of those is
  set, leaving its slots at the `ScaleStats` default; the fold always computes
  all three. Same SCORE (the dropped slots carry zero weight), DIFFERENT
  features. Requiring an all-active flag keeps the parity claim about the whole
  `ZensimResult`. Every MLP-scored profile sets `compute_all_features` (it is
  `mlp_bytes.is_some()`) and every `compute_extended_features` call sets
  `extended_features`; the excluded case is the plain linear
  `PreviewV0_1`/`PreviewV0_2` `compute()`.
* A live cancellation token also falls back. Issue #48's contract is
  cooperative cancellation *inside* the walk (row-band and scale boundaries)
  and the fold has no stop hook, so routing a stoppable request to it would
  silently downgrade cancellation to the caller's before/after `check_stop`.

### 8.2 Stage 3 — the ref-cached fold form (`02a8fd35`)

**The API stayed at zero new items**, because `PrecomputedReference` is not
buffered-walk state — it is an XYB pyramid cache, and its contents are exactly
what the fold's source side needs. The producer fills scale 0 with
`convert_source_to_xyb_into_slices` and cascades `downscale_2x_into`; the cache
is built by `convert_source_to_xyb` + `downscale_2x_into`; and
`producer_windows_byte_equal_materialized` already pins the producer against
exactly that materialisation. So the fold learned to READ the existing cache
(`StripPlaneProducer::new_with_ref_feed` → `ref_planes`) instead of getting a
parallel type, and `Zensim::compute_with_ref` routes with no signature change.

`feature_v2::cached_ref_feed_usable` is the admission test:
`NUM_SCALES` levels, dims matching the fold's floor-halving recurrence, tightly
packed at each level's width. `PrecomputedReference` stops its pyramid at
`w < 8 || h < 8`, so a short cache is refused and the call falls back.

| test | matrix | result |
|---|---|---|
| `fold_ref_cache_matches_independent_computes` | 18 geometries × 4 distorted candidates × {serial, rayon}: N-vs-one-ref **bit-identical** to N independent `compute` calls, all four fields | PASS |
| `compute_with_ref_cross_engine` | features / `score` / `raw_distance` bit-identical across engines at the with-ref entry | PASS |

**A property the buffered path does not have for itself.** Buffered's
`*_with_ref` derives `mean_offset` from a strip-wise `offset_sums /
pixel_count` accumulation rather than `compute_xyb_mean_offset`'s 64-row chunk
reduction, which is why `cross_platform::mean_offset_precomputed_ref` can only
assert `< 1e-10` between buffered's OWN two entries. The fold's ref-cached form
feeds the same producer over the same planes as its direct form, so it has
nothing to round differently — hence the bit-exact gate above. Across engines
at the with-ref entry the residual is that same buffered accumulation, measured
at worst **|Δmean_offset| = 8.674e-19** over the whole matrix.

Tier 2 (cached `mu1`/`activity`/`bs2`) was **not built.** §3.3's memory
argument stands and tier 1 needed no new type, so shipping a 47.8 B/px opt-in
before anyone has asked for it would be speculative. The spec stays in §3.3.

### 8.3 Stage 4 — the attribution canvas (`30aee3d7`)

**Blocker 4 was narrower than recorded, and the correction is the deliverable.**
`build_attribution_into_sink` calls only `crate::blur` — never
`compute_multiscale_stats_streaming`, `process_scale_bands`, or any other walk
function. Its buffered dependency was a pyramid to read, so the builders now
take `&impl MultiScaleRef` and `basic_attr_prep` reads through the trait
(`scale(0)` / `num_scales()`). The only concrete-`PrecomputedReference` use
left in the canvas path is `validate_ref_match`, which is API-level validation,
not pyramid access.

| test | result |
|---|---|
| `attribution_density_is_engine_independent` — `compute_attribution_density`, `_binned(8)`, `_full`: density AND SAT block sums bit-identical across engines, 4 geometries | PASS |
| `fused_compare_splits_into_fold_score_plus_standalone_map` — the fold-backed `compute_with_ref` score is bit-identical to the fused compare's score | PASS |
| the 26 pre-existing attribution unit gates, **unchanged** — incl. `attribution_covers_expected_slots_per_width`, `sum_preservation_mean_slots`, `sum_preservation_ppool_and_hf_slots` (the plane-sum identities), `fused_score_bit_matches_diffmap_path`, `fused_matches_standalone_attribution`, `fused944_features_bitwise_and_score_match_standalone` | 26 passed, 0 failed |

M3a cannot move: it is computed FROM these densities, and they are bit-identical
between engines.

**What is still walk-bound, stated plainly:** the FUSED compare.
`compute_with_ref_score_and_attribution*` calls
`streaming::compute_zensim_streaming_with_ref_and_attr_{planes,fold}` — the
buffered band walk with an `on_scale` callback folding the map *in strip*, on a
tiling (`BAND_ROWS == streaming::STRIP_INNER`) chosen to be bit-compatible with
the buffered extractor. Re-hosting that on the fold's own strip geometry
(`STRIP_ROWS`, `HALO_P`) is a re-derivation, not a re-point, and carries real
bit-risk. **Not attempted.** It falls back to buffered under
`ScoringEngine::Fold`, and the migration path is the split, whose score half is
bit-identical (gate above) and whose map half is already characterised by
`fused_matches_standalone_attribution` (f32-combine precision, 3e-5·max_abs).

### 8.4 Stage 5 — `zensim-gpu`'s CPU oracle (zenmetrics `92bdec00`)

`crates/zensim-gpu/tests/it/cpu_oracle.rs` makes the oracle's WALK selectable:
buffered by default (byte-for-byte the previous `ZensimCpu::new(profile)`
expression) and the fold under `ZENSIM_GPU_ORACLE_ENGINE=fold`. Six oracle
constructors route through it.

Measured on this box (`cargo test -p zensim-gpu --test it
--no-default-features --features wgpu` — a **software** wgpu adapter; the dev
box has no GPU):

| run | zensim | zenmetrics | result |
|---|---|---|---|
| baseline | `5e83d94c` (pre-lane) | pristine | 102 passed, 6 failed |
| buffered oracle | `be63ff39` | the change | 103 passed, 6 failed |
| **FOLD oracle** | `be63ff39` | the change | **103 passed, 6 failed** |

The same six failure NAMES in all three, and the same CPU reference VALUES in
the failure text (`odd_769x513 … cpu=2.478096e2` identically). Two conclusions,
both load-bearing:

* the six failures are **pre-existing** on this adapter — the odd-dimension
  pyramid-height class the module doc names, the stale cubecl-pool-page read,
  and two diffmap tolerances — and are not caused by either repo's changes;
* **swapping the oracle to the fold changes nothing the suite measures.**

The 103rd test is `cpu_oracle_engines_agree`, which needs no GPU: both walks
must produce `to_bits`-identical CPU references at the geometries this suite
compares kernels against (64/128/192 squares, 320×240, the odd 320×241 and
769×513 that `odd_dim_pyramid_parity` exists for, and the sub-64 48×40
reflect-pad case) for `compute_extended_features` AND `Zensim::compute`.

**Not repointed, and it is the one oracle site that still needs buffered:**
`cpu_gpu_diffmap_parity`'s `precompute_reference_linear_planar` — the PU-linear
route, which §1 excludes.

---

## 9. REGISTERED PROPOSAL — retiring the buffered walk

**Status: proposal only. Nothing in §9 has been executed, and this lane
deletes no buffered code.** It needs the user's sign-off, and one item (§9.2)
is an era-shaped decision that is not a perf or API question at all.

### 9.1 What the four blockers look like now

| # | blocker (predecessor §15) | status after this lane |
|---|---|---|
| 1 | the fold has no `score()` | **CLOSED** (`3dcb777c`, `be63ff39`) — every SDR scoring entry runs fold-backed, bit-identical, and the fold reproduces the pinned golden bytes |
| 2 | pool values differ at production widths | CLOSED by option C (`56bbcda2`) |
| 3 | no ref-cached fold form | **CLOSED** (`02a8fd35`) — and with zero new API, because the pyramid cache serves both engines |
| 4 | attribution's basic canvas is buffered-native | **NARROWED AND CLOSED for `compute_attribution_density*`** (`30aee3d7`); the FUSED compare remains, see §9.3 |
| 5 | `zensim-gpu`'s only CPU oracle is `compute_extended_features` | **CLOSED** (zenmetrics `92bdec00`) — the oracle's walk is selectable and the swap is measured inert |
| 6 | the fold must not be slower | see §10 |
| **7** | **`feature-regime-v2` is not a default feature** | **NEW, OPEN — and it is now the gating one.** §1.3 |

### 9.2 The prerequisites, in order

1. **Make the fold reachable in a default build.** Either move
   `feature-regime-v2` into `default`, or make `feature_v2` /
   `feature_v2_stream` / `fold_engine` unconditional and keep the feature as a
   no-op alias for one release. Until this lands, deleting buffered breaks
   `cargo add zensim`. This is a public-surface decision (it un-hides
   `feature_v2`'s whole module) and needs approval on its own terms.
2. **Decide `v1_only`.** §1.2: backing `score()` with the fold makes a
   `#[doc(hidden)]` "test/bench instrumentation" toggle a production compute
   request. Either bless it (rename it to something that says what it is —
   `compute_v1_blocks_only` — and drop the "not a product mode" language), or
   accept ~2× the work per score by running the full 944 walk. Measured cost of
   the latter is the predecessor's §10.1: `v1_only` removes 53 % of the walk.
3. **Flip the engine default** to `Fold` and let the whole suite run against
   it — every gate in §8 is already engine-parametrised, so this is a one-line
   change plus a full-suite run. Then run the zenmetrics A/B (§8.4) on real GPU
   hardware and flip `cpu_oracle`'s default too.
4. **Then, and only then, delete.**

### 9.3 What is NOT covered, and therefore blocks a *complete* deletion

Each of these still routes to buffered today, by design, with a named reason:

| entry | why buffered | what closing it needs |
|---|---|---|
| `compute_pu_linear{,_planar,_extended_features}` + `precompute_reference_linear_planar` | the fold has a PU front-end (`FrontEnd::Hdr`) but no wired mean-offset path there, and these take f32 slices rather than an `ImageSource` | a PU `MeanOffsetRows` hook + a linear-planar source adapter |
| `compute_with_ref_score_and_attribution*` (the fused compare) | the map is folded IN STRIP on the buffered band tiling (`BAND_ROWS == STRIP_INNER`) — §8.3 | a re-derivation on the fold's `STRIP_ROWS`/`HALO_P` geometry, with its own bit-gate; or accept the split (score bit-identical, map to 3e-5·max_abs) |
| `compute_streaming_strips*` (> 16 MP) | no fold entry with that API | probably **dissolves rather than ports**: the fold's plane residency is O(strip) natively and its peak RSS is thread-independent (predecessor §13: 163 MB at 2304²), so the >16 MP case is what the fold is *for*. Routing these to the fold would also drop the strip path's documented "minor approximation in the strip-boundary blur context" — an improvement, but an output change, so an era decision |
| `PreviewV0_1`/`PreviewV0_2` `compute()` | weight-skipping (§8.1) | either compute all channels for those profiles (changes their feature vectors, not their scores) or keep a buffered path for them |
| any `with_stop` request | no in-walk cancellation hook in the fold | a `Stop` check at the fold's strip boundary — small, and it makes issue #48's contract engine-independent |
| any `num_scales != 4` `ZensimConfig` | the fold is hard-wired to `NUM_SCALES` | generalise the fold's scale count, or declare 4 the only supported value |
| `compute_zensim_with_config` (the `training` free function) | deliberately unrouted — it is the entry `v1_golden_bytes` measures | route it LAST; `fold_backed_fixtures_match_golden` already shows the fold reproduces the same arrays through `compute_extended_features` |

### 9.4 Deletion order, once §9.2 and §9.3 are settled

Delete outside-in, one commit per step, full suite between steps:

1. **`metric::combine_scores`** and the `ScaleStats` → feature assembly.
   Pinned by: every parity test in §8 (they compare against it), plus
   `v1_golden_bytes`. Delete only after the goldens are re-pinned to the
   fold-backed entry.
2. **`streaming::compute_zensim_streaming{,_stoppable}`,
   `compute_multiscale_stats_streaming`, `multiscale_stats_over_pu_xyb`,
   `compute_multiscale_stats_pu_linear_*`.** Pinned by `cross_platform`,
   `pu_entry`, `size_invariance`.
3. **`compute_multiscale_stats_streaming_with_ref{,_borrowed}`,
   `compute_multiscale_accums_streaming_with_ref_borrowed`,
   `compute_zensim_streaming_with_ref{,_and_diffmap{,_linear_planar}}`.**
   Pinned by `cross_platform::mean_offset_precomputed_ref` and the diffmap
   tests.
4. **`compute_multiscale_stats_streaming_strips{,_with_ref}`** — only after
   §9.3's ">16 MP dissolves" decision. Pinned by `streaming_strips` and the
   `streaming_strips_oom` feature test.
5. **`compute_zensim_streaming_with_ref_and_attr_{planes,fold}`** — only after
   the fused compare is re-hosted or the split is accepted. Pinned by
   `attribution::tests::fused_*`.
6. **The band machinery itself**: `process_scale_bands`,
   `process_scale_bands_into_accum`, `process_strip_channel`,
   `ScaleAccumulators` (`streaming.rs:386-679`, ~294 lines), `active_channels`,
   `downscale_6_planes`, `AttrScaleRetention`/`AttrBandSlices`/`AttrFoldBand`.
   This is the bulk — `streaming.rs` is 6,738 lines and most of it is here.
7. **`compute_xyb_mean_offset{,_range}`** — LAST, and read §9.5 first.

### 9.5 What must SURVIVE, and why it is not "buffered code"

Deleting these would break the fold:

* **`PrecomputedReference` + `PrecomputedReferenceView` + `MultiScaleRef`** —
  the pyramid cache. Consumed by the fold's own ref feed (§8.2) *and* by the
  attribution canvas (§8.3). These deserve to move out of `streaming.rs` into
  their own `pyramid` module as part of the deletion, precisely so that
  "delete the walk" cannot take them with it.
* **`convert_source_to_xyb{,_into,_into_slices}`,
  `convert_linear_planar_to_{,pu_}xyb_into`,
  `convert_linear_interleaved_to_pu_xyb_into`, `downscale_3_planes`,
  `blur::downscale_2x_into`** — the front end. The producer calls them.
* **`compute_delta_stats`** and the `DeltaAccum` cluster — classification is a
  pure per-pixel pass with no pyramid and no walk.
* **`upsample_row_powx_add`** — attribution uses it.
* **`compute_xyb_mean_offset`** — not because the fold calls it (it does not),
  but because it is the **definition** `MeanOffsetRows::finish` reproduces. If
  it is deleted, that reproduction loses its oracle. Keep it as a
  `#[cfg(test)]` reference with a test asserting the two agree bit-for-bit on
  materialised planes, or accept that `MeanOffsetRows` becomes the definition
  and say so in `docs/DATASET_HISTORY.md`.

### 9.6 One thing this proposal will not do

It will not claim the fold is as fast as buffered. The predecessor measured
944-full at 2.2–3.8× `buf_v1_372` (§13), and this lane's own numbers are in
§10. Retirement is being proposed **on the API and correctness blockers**,
which are now closed or named, and the perf question is a separate trade the
user has already been given the shape of: the fold buys one code path and a
4.7× memory reduction, and §14 of the predecessor bounds the remaining MT
headroom at ~1.2× behind an era decision. If the trade is not acceptable,
the right outcome is "keep both, with the fold as a gated alternative" — which
is exactly the state this lane leaves the tree in.
