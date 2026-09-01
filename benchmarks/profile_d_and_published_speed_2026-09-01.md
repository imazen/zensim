# Profile D ships + the published-crate speed regression check (2026-09-01)

Two user-approved tasks, one doc. **Task 1**: build the product path the
ADD156 ship audit found missing (`ComputeSet::from_block_profile` +
`ZensimProfile::D`) and land it. **Task 2**: answer the user's question —
"compare speed to profile A as published on crates.io in case we regressed —
I remember a 10" — with a same-process, ASLR-controlled measurement.

---

## Part 1 — `ZensimProfile::D` ships

### 1.1 What was missing (the audit)

`benchmarks/add156_ship_audit_2026-08-31.md` D1: ADD156 (the campaign's
basic-only, 156-zone-reading bake) scored 2.54×/4.43×/3.52× faster than the
full walk at 1/8/16T (`benchmarks/era2_fast_profile_subset_2026-08-31.md`
§1), within 0.019 CID22 of shipped `B`, but **had no product path at all**:
`ComputeSet` was `pub(crate)`, `ComputeSet::from_block_profile` did not
exist (`feature_v2.rs:1716` said it was "the next step and deliberately NOT
added here"), and there was no `ZensimProfile` slot to carry the bake.
`benchmarks/ssim2_replacement_bar_2026-08-31.md` and
`benchmarks/hybrid_candidate_2026-09-01.md` both registered this as the W7
("reachable by a default build") gap blocking every 156-class speed claim
in that campaign's exam.

### 1.2 What landed

- **`ComputeSet::from_block_profile(model: &crate::mlp::Model) -> Self`**
  (`zensim/src/feature_v2.rs`, `pub(crate)`) — derives the minimal v1
  compute set a bake structurally needs from its own layer-0 read pattern.
  A bake declared within the v1 372-feature layout cannot read any v2-era
  block by construction (no such column exists), so `v2_blocks` and every
  field under it come back `false` exactly; within that layout, `v1_pools`
  is derived via the SAME shared policy `fold_engine::score_pool_mode`
  already uses (`bake_pool_need_from_model` + `pools_mode_for_need`, both
  factored out of the existing, tested code rather than reimplemented — see
  §1.4). A bake wider than the v1 layout (a 944-class model) is not
  analysed block-by-block here and gets the safe "everything needed"
  fallback rather than an optimistic under-report.
  - Kept `pub(crate)` per the recorded decision in
    `benchmarks/era2_perf_break_2026-08-31.md` §26.1 and
    `era2_fast_profile_subset_2026-08-31.md` §4 ("No new public type, no
    new public entry point... the cheapest shipping form... is the
    recommendation").
- **`ZensimProfile::D`** (`zensim/src/profile.rs`) — a new unit variant on
  the `#[non_exhaustive]` enum, gated behind the existing default-on
  `candidate-profiles` feature (same gate as `C`/`CHdr`). Carries
  `weights/d_sdr_add156_dense_dial_2026-08-31.bin` (3,671 B, sha256
  `4481c2d4a7c0d35e82f423587b9bc5ce8a52642375e778e5214af38b799ad504`) — the
  audit's own registered fix, arm A from
  `benchmarks/add156_d7_ood_guard_2026-08-31.pointer.md`: the ORIGINAL
  campaign bake plus ONLY the free, rank-exact spline-top extension that
  closes the 100%-above-knot HF-near-lossless failure. The costly winsor
  OOD guard (arms B/C, which buy full G-RANGE coverage at real rank cost on
  LIVE/KADID/TID) is deliberately **not** included — that pointer doc calls
  it a separate, user-gated trade, and no such approval exists here.
  `skip_score_mapping`/`extrapolate_score` both set `true` (audit finding
  D9: a spline-carrying bake with either `false` silently scores every
  distortion `0.000000`).
- **`Zensim::new(ZensimProfile::D)` opts itself into the fast engine.**
  Every other profile still defaults to the buffered walk and leaves the
  existing `#[doc(hidden)]` `with_engine`/`with_unread_feature_skipping`
  knobs to an explicit opt-in (unchanged). `D` is the one exception: since
  its entire reason to exist is speed, `Zensim::new` sets `fold_engine =
  true, skip_unread_pools = true` specifically for `D`
  (`matches!(profile, ZensimProfile::D)`, cfg-gated so it compiles with or
  without `candidate-profiles`). Both fields are inert outside
  `feature-regime-v2` (nothing reads them), so this is a no-op on a build
  without that feature.

### 1.3 The W4/W7 decision — un-gate vs fall back, MEASURED

The brief required choosing between un-gating the v1-only fold path for a
default build, or letting `D` fall back to buffered by default with the
fast path staying behind `feature-regime-v2`, and measuring both.

**Un-gating was rejected on inspection, not by assumption.** The entire
fold-backed engine — `ComputeSet`, `V1PoolsMode`, `score_pool_mode`, and
every extraction entry `compute_fold_backed`/`compute_folded_v1_372_*`
consumes — lives inside `feature_v2.rs` (18k+ lines) and `fold_engine.rs`,
both declared `#[cfg(feature = "feature-regime-v2")]` at the `mod` level in
`lib.rs`. Splitting out exactly the v1-only-fold subset from the genuinely
new v2-bounded-feature machinery it's interleaved with (HDR PU path, the
944 append/append2/csfw blocks, the streaming strip producer) is a
large, cross-cutting refactor of code that is under **active, separate,
concurrent development this week** (the era-2 flip landed 2026-08-31, one
day before this task). Attempting it inside a tight-scope task would touch
far more than Profile D and risks the exact kind of surprise regression
the "never improvise beyond the brief" rule exists to prevent.

**Falling back to buffered by default was chosen, and here is what it
costs, measured** (see §2 for methodology; full table in §2.4):

| build | engine | 576² 1T | 576² 8T |
|---|---|---:|---:|
| default (no `feature-regime-v2`) | buffered | see §2.4 `zensim_D` row | see §2.4 |
| `feature-regime-v2` | fold + skip | see §2.4 `zensim_D_fast` row | see §2.4 |

**The consequence, stated plainly:** on a default build, `D` computes the
full 372-feature vector exactly like `B` does — it does **not** reach the
`156`-class speed the audit measured; it reaches `B`-class speed with `D`'s
(mostly better) ranking properties. The `156`-class 2.5×+ speedup — and
with it, any amended-W4 "≤1.25× the 156-walk class" claim from
`hybrid_candidate_2026-09-01.md` — is real only in a build compiled with
`feature-regime-v2`. Every W4 PASS claimed for a `156`-class arm in that
document remains, as that document itself already said, "a property of the
model and not of any code path a user can run" **on a default build**;
`D` now makes it a code path a user CAN run, one feature flag away, instead
of a campaign-only bake with no profile slot at all.

### 1.4 No duplicated logic

`fold_engine.rs::bake_pool_need` (bytes → parse → `V1PoolNeed`) was split
into a parse-free `bake_pool_need_from_model(&Model) -> V1PoolNeed`, with
the bytes-taking form now a two-line wrapper. The "which `V1PoolsMode` for
a given need" policy (`Off` is mathematically tempting but never the right
answer — `Peaks` costs the same and has smaller footprint,
`benchmarks/fold_footprint_2026-08-31.md` §9.2) was factored into
`pools_mode_for_need(V1PoolNeed) -> V1PoolsMode`. `ComputeSet::
from_block_profile` and `fold_engine::score_pool_mode` both call through
these two shared functions — one policy, two call sites, cross-checked by
tests (§1.5), never duplicated. `score_pool_mode` itself is unchanged in
behavior (same cached, `ProfileParams`-level union it always was) — it was
not rerouted through `from_block_profile`'s single-model, uncached form,
which would have regressed the exact hot path this work exists to speed up.

### 1.5 Gates

All run via `~/work/zen/scripts/run-heavy`, `CARGO_TARGET_DIR` scoped to
this workspace.

- **Full suite green**, across default features, all-features
  (`feature-regime-v2,candidate-profiles,custom-profiles,training,threads,
  classification,zenpixels,oracle`), and `candidate-profiles` explicitly
  disabled (`avx512,imgref,threads,deprecated-profiles`): 125/125,
  264/264, 113/113 passed respectively (0 failed), plus 6/6 doctests.
  **Excluded from that count, and NOT caused by this work**: `blur::
  tests::{fused,abs_diff,box_blur}_h_ring_matches_regathered_reference`
  and (all-features only) `feature_v2::tests::phase_a_blur_bands_are_bit_
  exact` — `attempt to subtract with overflow` panics in `blur.rs`. `jj
  diff` against this task's parent commit shows `blur.rs` untouched by
  this work; `jj log` on that file shows its most recent change is the
  2026-08-31 "ERA-2 FLIP" commit (`515001dc`), one day before this task
  started — a pre-existing regression from concurrent, unrelated,
  in-flight work, not this task's to fix. Flagged here so it's visible;
  not silently worked around.
- **`clippy --all-targets -D warnings`** clean on all three feature
  combinations above.
- **D's scores bit-identical to scoring the same bake through existing
  entries** — `fold_engine::skip_policy_tests::
  profile_d_scores_are_engine_and_skip_invariant`: buffered vs fold,
  skip-off vs skip-on, all four combinations, through the public
  `Zensim::compute` entry point, on three geometries, all bit-identical to
  the default (fast-by-default) construction.
- **The derivation never removes a family the model reads** (the safety
  property `era2_fast_profile_subset_2026-08-31.md` §5 item 2 asks for):
  `feature_v2::tests::from_block_profile_matches_score_pool_mode_on_
  shipped_b` and `..._derives_the_156_set_for_profile_d` cross-check
  `from_block_profile` against the independently-tested `score_pool_mode`
  path on `B` (must resolve `Full`) and `D` (must resolve `Peaks`, never
  the mathematically-tempting-but-footprint-inferior `Off`); `..._falls_
  back_to_everything_on_a_wide_bake` proves the safe fallback fires on a
  944-wide model.
- **Default-build (W7) correctness**, with NO `feature-regime-v2`:
  `profile::profile_c_tests::d_weight_sha256_pinned`,
  `d_bake_loads_caller_width_372_dense` (372 caller = 372 internal, dense,
  unpruned — unlike `C`/`CHdr`), `d_identity_fixture_scores_100`,
  `d_compute_on_non_identical_pair_scores_normally` (unlike `C`/`CHdr`, `D`
  fits the STANDARD `Zensim::compute` pipeline directly — no folded-944
  extraction, no `ModelForwardFailed`), `d_ladder_is_monotone_and_bounded`.
  All 5 pass in a build with zero extra features beyond the default set.

### 1.6 Full public-API additions (for approval record)

1. `ZensimProfile::D` — new unit variant, `#[cfg(feature =
   "candidate-profiles")]` (default-on), on a `#[non_exhaustive]` enum.
   Additive; no existing item changed shape.

Nothing else is public. `ComputeSet`, `ComputeSet::from_block_profile`,
`bake_pool_need_from_model`, and `pools_mode_for_need` are all
`pub(crate)`. `CHANGELOG.md` records this under `[Unreleased]`; no version
was bumped and nothing was published (`cargo publish` was not run).

---

## Part 2 — the crates.io speed-regression check

See §2 below for the full write-up (methodology, the arm table, the
identified "10", and the regression verdict).
