# NEIGHSTEER worklog — exact local coarse-scale refinement (SWE-2 lane, 2026-10-04)

Lane: `quarantine/swe2/neighsteer` on workspace `zensim--neighsteer` (base dc589ded main*).
Agent: devin swe2-neighsteer-lane. Scratch `/var/tmp/neighsteer/lane/`, target `/var/tmp/neighsteer/lane-target`.
Every heavy command through `taskset -c 16-23 nice -n 19 ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- ...`,
`CARGO_TARGET_DIR=/var/tmp/neighsteer/lane-target`.

## 2026-10-04 — recon complete, engine implementation

Design decisions (all verified against production source):

- `ret.planes`/`ret.cells`/`ret.mg` are written by the foldapp streaming walk's retention hooks
  (`foldapp_streaming_walk_impl` ~14173/14503) only for `compute.channel_active(scale, ch)` cells; unserved
  cells stay `AttrCellSums::default()` and unserved feature slots are 0 in both base and intervened walks,
  so the engine emits Δ = 0 for them (exact, not fabricated).
- Phase-A semantics: `run_blur_pass_inner` (`feature_v2.rs:4172`) = fused H (`fused_blur_h_ssim_at_revision`,
  f32 sliding for Rev<=3, Rec64 f64 for Rev4 via `featcanon::mode(revision)`) + V `box_blur_v_from_copy`
  (always f32 Rec in product builds — the blur-axis switch is `#[cfg(feature = "oracle")]`-only) of
  `mu1 mu2 ssq s12`, plus `act = blur(|src-mu1|)` (src-only) and `bs2` (src-only, append-only → not needed).
  `mu1`/`act` are reference-only → reused from retention; `mu2`/`ssq`/`s12` are dst-dependent → recomputed.
- Pyramid: `blur::downscale_2x_into` (blur.rs:6624): `(a+b+c+d)*0.25` left-to-right f32, floor-drop odd
  rows/cols (`new_w = w/2`).
- Per-pixel terms: `dense_terms32` (feature_v2.rs:28276) is the universal f32 evaluator at every revision
  (canon64 = same f32 terms into f64 lanes). `gradient_terms64` (:5933) is the border-path f64 evaluator;
  `blockiness_sparse` (:7260) = `bounded_excess(step_dst, step_src, C_BLOCK)` at 8-lattice positions.
- Accumulation deltas are exact f64 differences applied to the retained f64 `AttrCellSums` — the lane-reduce
  drift term is O(n·eps) ≈ 1e-9 relative, far under the 1e-6 + 1e-4|Δ| golden bound.
- Blur windows are re-summed FRESH per position in the production init tap order (i-ordered `tap_mirror`
  taps, same op sequence as `fused_blur_h_ssim_inner`/`fused_blur_h_rec64_row` init): production's sliding
  recurrence drifts O(1e-6)-relative over a row; the difference cancels in old-vs-new deltas.
- EWC: `finish_channel_scale` returns (mean gsrc, mean gdst); `features[prev+EWC] =
  1 - bsim(mg[s+1].0/(mg[s].0+C_GRAD_DECAY), mg[s+1].1/(mg[s].1+C_GRAD_DECAY), C_EDGEWIDTH)`, coarsest
  scale copies the second-coarsest (production foldapp finalize ~15111).
- `apply_transducer_luma_gate` zeroes PJND slots on ch != 1 when `transducers_luma_only`.
- Snapshot: trimmed to scales 1..=3 planes+cells+mg + scale-0 dst pyramid (candidate cascade base);
  scale-0 src and `bs2` not needed.
- Refusals (snapshot capture → None): sampling bakes, v2_blocks off, `ret.dims[0] != (src_w,src_h)`
  (reflect-padded <64), dims len != NUM_SCALES. HDR refused at the call site (prepare_steering_hdr
  already refuses v2 reads; additionally guarded).
- Integration: `ZENSIM_NEIGHBOUR_EXACT` env read per `compute_attribution_input`; when the snapshot
  captures successfully, `spatial[372+87 .. 720]` is zeroed before `attribution_from_retention_binned`
  and `refinement_gain` adds `Σ_k s_k·Δf_k` from the engine. Unset → untouched bytes.

## 2026-10-04 (cont.) — residue cone: the decisive correction

**Symptom:** real JPEG q10 pair (91×87, rect (0,0,8,8)) failed goldens at Rev3+Rev4 — six ch2
features, `HF_MAG_LOSS` worst (local −1.9656e-2 vs full −1.9767e-2). Synthetic cases passed.

**Bisect (in-crate probe test, later kept as `cone_planes_bit_exact_and_complete`):**
recomputed `mu2`/`ssq`/`s12` were bit-identical to the intervened walk's retention *inside*
`dilate(C,5)` — but the whole-plane diff showed **1,148 last-ulp diffs OUTSIDE** it (e.g.
`ssq` at x=43, y=0), all downstream of the changed rows/columns.

**Root cause — sliding-sum residue:** the production blur is a running-sum recurrence
(`sum = sum + add − rem`, `out = sum·inv`). Once a changed tap enters the window the sum's
f32 rounding residue persists *for the rest of the row/column*: H rows differ to the row
end, V columns to the strip end (strips re-init each `STRIP_ROWS=128` band). Amplified by
`HF_MAG_LOSS`'s `|hf_src|+|hf_dst|+C_HF` denominator, the residue produced ~1e-4 feature
errors — exactly the observed failures. Fresh-window blur was never the truth.

**Fix:** the diff region is the *residue cone*, not `dilate(C,5)`:
`[c.x0−5, w)` over every strip whose `2*HALO_P` halo gather touches `[c.y0,c.y1)`, from
that strip's top to its bottom. `recompute_planes` now replays full H rows (from x=0)
and full V columns (from wide-row 0) per affected strip, keeping per-cell `new != old`
as an eval mask (`planes.diff`) so `dense_delta` only visits cells with real changes
(+ C's own cells for the `dd` terms). Gradient/blockiness eval regions were already
correct (`gradient_terms64`/`blockiness_sparse` read only src/dst pixels: `dilate(C,1)`
and the 8-lattice).

**Result (probe, ch2 rect (0,0,8,8)):** `sum_d` delta now *exact* at all 3 scales;
`sum_hf_mag` to ~5e-6 cell units (was 0.21); `ws_ssim_den` ~3e-8; `sum_mse` ~1.5e-7 —
all pure lane-pool-order noise, divided by n at feature level.

**Golden gate:** all cases pass at Rev3 **and** Rev4 — synthetic textured 137×101
(1×1, 8×8 aligned/unaligned, 23×17, 16×16, 32×32, edge/clipped rects), real zenjpeg
q10 4:4:4 91×87, q30-candidate (non-reference pixels) 121×93, and a new multi-strip
141×301 case whose rects straddle the scale-1 strip boundary (row 128).

Files: `local_refine.rs` (`recompute_planes` cone rewrite + `strip_first_changed`,
`dense_delta` mask-driven eval, `Planes{diff,rx0,ry0,rw,rh}`).

## 2026-10-04 (cont.) — first cost numbers (1024×1024, best-of-5, release, lane cores)

- Snapshot heap: **54,067,200 B** (~51.6 MiB) — scale-0 src+dst (24 MB) + scales 1–3
  src/dst + 5 phase-A planes ×3 ch.
- Reference-repair query: **8×8 ≈ 6.57 ms**, **32×32 ≈ 7.44 ms** — dominated by the
  residue-cone eval (at 1 MP an 8×8 at (400,400) touches ~40 k cells at scale 1 alone).
  Exactness is the price: the cone is inherent to the streaming recurrence. Still
  ~100-1000× cheaper than a full walk per query.

## 2026-10-04 (cont.) — integration bug: sparse retention at scale 0 (found + fixed)

**Symptom:** first steering run under `ZENSIM_NEIGHBOUR_EXACT=1` scored M3f −0.17 on
`I01_10_05`/`by_v2fy`/s5101 (vs published 0.6572, oracle ~0.97). `refinement_gain` added
≈ +1.2/block while true ΔS ≈ −0.06.

**Bisect:** `kadid_case_debug` test (real KADID pair + real `by_v2fy` bake through
`prepare_steering`) showed `Σ s·Δ_engine` = +1.245 vs `Σ s·Δ_true(emitted)` = −0.076,
with ~160/261 emitted ids off by up to 400× — all in the X and B channels.
`finish_channel_scale(retained cells)` reproduced the served feature row exactly
(no BASE MISMATCH), so the base cells were right and the *deltas* were wrong.

**Root cause:** retention fills `pyr_src[scale][ch]`/`pyr_dst[scale][ch]` only where
`compute.channel_active(scale, ch)` (`FoldRetention::ensure` zero-fills, `copy_strip`
is gated). `by_v2fy`-shaped plans have `full_res_xb=false` (no scale-0 X/B ids read),
so scale-0 X/B stay **zero-filled** — but the cascade still needs those pixels to
rebuild scale-1 changed values. The engine silently cascaded zeros into every X/B
coarse plane. Y-only plans worked; the goldens passed because `full_plan()` sets
`full_res_xb=true` → `ALL_CHANNELS` walk → everything retained. The unit-walk and
production walks differed in exactly the channel mask.

**Fix:** `LocalRefineSnapshot::capture` now takes the session's `source`/`distorted`
`ImageSource`s and rebuilds scale-0 channels where `!channel_active(0, ch)` using
`crate::streaming::convert_source_to_xyb_into_slices` — the same conversion the
producer's `convert_side_scale0` runs, so values are bit-identical to what the walk
used. Once per session, only when a channel is missing.

**Verification:** `sparse_retention_scale0_channels_rebuilt` (permanent regression):
derives the by_v2fy-shaped plan (`13..26 ∪ 39..156 ∪ 401..430 ∪ 459..720`, width 720 —
width matters: `Plan::toggles()` switches append-family blocks on layout width ≥ 720+
flags, which flips `chroma_local_families`/`full_res_xb`), asserts retention really
lacks scale-0 X/B, asserts the rebuilt planes are **bit-equal** to an `ALL_CHANNELS`
walk's retained planes, and checks `deltas` vs a full plan'd recomputation on 3 rects.
`kadid_case_debug` (external-asset, env-gated, `#[ignore]`d): `bad 0` on all blocks,
`Σs·Δ_engine` ≈ `Σs·Δ_true` to ~1e-9.

**Note on `Plan::derive` layout width:** `LayoutBlocks::for_width(944)` turns every
block flag on; through `ComputeSet::from_toggles` that disables the
`allows_full_res_y_subset` optimization. Plans must be derived at the bake's real
walk width (720 for the dense bakes) to reproduce production compute sets.

## 2026-10-04 — 48-case KADID panel under `ZENSIM_NEIGHBOUR_EXACT=1`: ON the oracle

Panel: `run_jp48.sh` — I01/I21/I41/I61 × levels 03,05 × arms {by_v2fy, v2basic} × seeds
{5101,5103,5107} = 48 cases, `ZENSIM_FORMULA_REV=4 ZENSIM_PREPARED_STEERING=1
ZENSIM_NEIGHBOUR_EXACT=1`, bakes `/var/tmp/steercheck/gate3/rev4dense`, block 8,
8-way parallel on cores 16-23. Per-case JSONs `/var/tmp/neighsteer/lane/jp48/`.

Median M3f per set×level (12 cases each), vs the published map-today and the
§2 oracle "exact v2 at scales 1–3" column:

| set, level | map today | oracle 1–3 | MEASURED (env on) |
|---|---:|---:|---:|
| by_v2fy, 03 | 0.813 | 0.983 | **0.9827** |
| by_v2fy, 05 | 0.650 | 0.966 | **0.9654** |
| v2basic, 03 | 0.833 | 0.931 | **0.9315** |
| v2basic, 05 | 0.644 | 0.903 | **0.9008** |

All four cells land within ~0.001–0.003 of the per-feature oracle — the engine
reproduces the oracle's "exact v2 at scales 1–3" behavior through the real
`refinement_gain` path. Min per-case M2 across the panel: 0.9813
(by_v2fy I21_10_05 s5101, the one case that also has the lowest m3f 0.8541).
Per-case table in `jp48/*.json` (48 files, sha on request).

## 2026-10-04 — env-unset byte-identity gate: PASS (0 differing everywhere)

Baseline binary: `basesrc` checkout at dc589ded (main@origin is dc589ded + one
benchmark-doc commit — binary-equivalent). New binary: this lane's
`diffmap_block_coherence`. `ZENSIM_NEIGHBOUR_EXACT` absent from the run env in
all cases; `ZENSIM_PREPARED_STEERING=1`, `RAYON_NUM_THREADS=1`.

| panel | bakes | rev | cases | differing |
|---|---|---|---:|---:|
| owner-12 (`gate.py owner`, 4 arms × 3 seeds × 4 KADID-03, block 32) | rev4dense | 4 | 48 | **0** |
| owner-12 | rev3dense | 3 | 48 | **0** |
| broad-96 (`run_broad.py`, 4 arms × 24 TRAIN pairs × blocks 8/16/32/64) | rev4dense | 4 | 384 | **0** |
| broad-96 | rev3dense | 3 | 384 | **0** |

`gate.py broad` compares every RESULT row field (score/m2/m3f/interventions/
status/unsupported) AND the full per-case JSONs — 0 differing at both revs.
Outputs: `/var/tmp/neighsteer/lane/ident/{owner,broad}_{rev3,rev4}dense_{old,new}/`.
Old-binary sha256 `5194ca13…`, lane binary `cda65edc…` (pre-doc-comment rebuild;
the code is unchanged since).

Quick gates: `cargo fmt -p zensim --check` clean; `just lint-scripts` 807 scripts
OK; `just api-doc-check` PASS (public API snapshot unchanged); clippy
(feature set, lib+tests) clean; cross-target `cargo check -p zensim --tests
--features custom-profiles,feature-regime-v2,threads,training` finished for
aarch64-unknown-linux-gnu, i686-unknown-linux-gnu, wasm32-unknown-unknown —
only pre-existing warnings (fused.rs, ssim_form.rs, dvifm.rs; none in lane files).

## 2026-10-04 — gate completion: full suites, perm cells, serve gate

**Feature-gating bug found by the 26-cell perm check** (`feature-regime-v2` alone and
`deprecated-profiles,candidate-profiles` — which pulls `feature-regime-v2` without
`custom-profiles`): `mod local_refine` compiled but its only caller
(`compute_attribution_input`, `all(custom-profiles, feature-regime-v2)`) was cfg'd
out → the entire module was dead code under `-D warnings`, and `mod tests` called
`compute_folded944_streaming_with_retention` (custom-profiles-gated). Fix: the
module now gates identically to its caller (`lib.rs`), the `neighbour_exact` field
and its `refinement_gain` block carry the same `all(...)` cfg. Verified: both cells
pass, plus `custom-profiles` alone and `custom-profiles,feature-regime-v2`.

**Full gate board (all green):**

| gate | result |
|---|---|
| `cargo test -p zensim --release --features custom-profiles,feature-regime-v2,threads,training` | 4 runs, **0 failures** (767 tests) |
| `cargo test -p zensim --release --all-features --lib` | 3 runs, **621 pass each** |
| `just rev4serve-gate` (corpus-gated aic3 bake serve+steer) | PASS (66.6 s) |
| CI-exact `just clippy` (workspace --all-targets --all-features -D warnings) | clean |
| `cargo fmt -p zensim --check` | clean |
| `just lint-scripts` | 807 scripts ok |
| `just api-doc-check` | public API snapshot unchanged |
| 26 perm cells (`perm_check.keep.sh` → this workspace, `/var/tmp/neighsteer/target-perm`) | all ok after the gating fix |
| cross-target `cargo check --tests` | aarch64/i686-linux-gnu, wasm32 — finished; only pre-existing warnings (fused.rs, ssim_form.rs, dvifm.rs) |
| env-unset byte-identity | owner 96/96 + broad 768/768 case JSONs identical, both revs |

`local_refine` group re-run after the cfg fix: 6 passed / 2 ignored.

## 2026-10-04 — FIX ROUND: independent review of c90711d0, six defects found and fixed

Review asked for exactness "everywhere the engine serves, or refuse". All six items
fixed; two additional real bugs found while building the required tests.

### Item 1 — BLOCKER: unretained coarse channels read as data. FIXED.

`capture` now calls `rebuild_unretained_coarse` (`local_refine.rs` ~317): for every
scale `s≥1` and channel the plan did not retain (`channel_active` false — e.g.
`coarse_y_only_scales`, sparse masks, or a stale reused retention buffer), the
source AND distorted planes are rebuilt from the scale-0 planes through the
production `downscale_2x_into` cascade — same arithmetic the walk's producer ran.
Per-scale dims must equal exact floor-halving of the parent or capture returns
`None`. The chain reads `ret.mg` for the edge-width-change term too: an inactive
channel's walk-side `mg` contribution is `(0.0, 0.0)` (see item-extra below), which
is what the rebuild writes. Test: `coarse_y_only_plan_coarse_channels_rebuilt`
seeds the retention buffers with STALE data from a previous pair (not just zeros)
so a stale read fails loudly; deltas then equal a full recomputation under the
same plan. `sparse_retention_scale0_channels_rebuilt` covers the scale-0 arm.

### Items 2+3 — H tiling + fused/unfused mul_add: FIXED by deleting the copies.

`recompute_planes` no longer hand-rolls the blur. Each touched strip's mixed wide
window goes through the production `fused_blur_h_ssim_at_revision` (inheriting
`H_TILE_WIDTH=1024` column tiling, `ZENSIM_H_TILE` override, Rev3/Rev4 and
oracle/canon dispatch, and the tier's magetypes `mul_add` semantics — unfused on
scalar/wasm128) and `box_blur_v_from_copy` (inheriting per-strip re-init). A
per-cell `new != old` mask then limits term re-evaluation to cells that differ.
The whole-cone comparison asserts bit-equality of the replayed mu2/ssq/s12 vs an
intervened walk's retention — at Rev3, Rev4, AND the forced-scalar tier
(`cone_planes_bit_exact_and_complete_rev3/_rev4`, `cone_planes_bit_exact_scalar_tier`).
Audit: after the rewrite the engine keeps **no f32 arithmetic of its own** —
every f32 value comes from a production kernel call; every remaining hand-written
reduction is scalar f64, IEEE-identical on every tier (the mul_add divergence was
f32-only). `wide_rev3_scale1_over_h_tile` covers source width 2060 → scale-1
width 1030 > 1024.

### Item 4 — blockiness old/src subtraction widened in f32. FIXED.

Both the old and source steps now subtract in f64 before `.abs()`
(`(a as f64 - b as f64).abs()`), matching production (`feature_v2.rs` ~7288).
`noop_candidate_yields_exact_zero_deltas` proves a no-op candidate emits exactly
0.0 on every covered slot — before the fix, the f32-subtracted sides produced
nonzero residue.

### Item 5 — env switch enabled on presence. FIXED.

`bake.rs` now reads `std::env::var("ZENSIM_NEIGHBOUR_EXACT").as_deref() == Ok("1")`.
Presence alone, `=0`, empty, `=yes` → frozen density, byte-identical (test below).
Documented in `lib.rs` (the module comment — where the steering env vars live),
the `local_refine` module doc, and at the gate site. Test
`attribution::tests::neighbour_exact_env_gate_and_gain` re-executes itself in
child processes under unset / "" / "0" / "yes" / "1" and asserts the snapshot is
present only for exactly `"1"`; it then checks `refinement_gain` equals frozen
density + Σ s_k·Δf_true vs a full recomputation through a real
`BakeScorer::prepare_steering` session (zenjpeg-generated pair).

### Item 6 — memory: snapshot boxed. DONE.

`ScoredAttribution.neighbour_exact` is `Option<Box<LocalRefineSnapshot>>` —
8 bytes inline when None vs ~51.6 MiB if it were stored inline. Arc-sharing was
considered and rejected: `FoldRetention` is session-owned and borrowed at capture;
an `Arc` would force the retention into shared ownership upstream of its owner —
not a contained change. The per-result deep copy (measured 51.56 MiB at 1 MP,
`cost_per_query_1mp`) is documented in DONE.

### Found while testing — EWC for inactive channels (beyond the six items).

The first coarse-y-only test failed `f487` (scale-1 X `edge_width_change`):
engine 0 vs full walk +2.25e-6. Production's finalize computes EWC for EVERY
channel gated only on the per-scale `gradient` pair, not on `channel_active` —
an inactive channel still gets a real EWC built from `(0,0)` gradient means, and
the coarsest scale copies unconditionally. The engine now matches: the EWC patch
iterates all channels with `gradient_on[s] && gradient_on[s+1]` gating, `(0,0)`
grads for inactive channels — which is also exactly why item 1's rebuild writes
`(0.0, 0.0)` for inactive `mg`. Without this the plan'd walk and the engine
diverge on every adjacent-scale EWC slot of an inactive channel.

### Test board after the fix round

`cargo test -p zensim --features custom-profiles,feature-regime-v2,threads,training --lib local_refine`:
**11 passed / 2 ignored** (cost + loud-gated KADID diagnostic — now panics unless
`NEIGHSTEER_KADID_DIR`+`NEIGHSTEER_BAKE`+`ZENSIM_NEIGHBOUR_EXACT=1` are all set,
instead of skipping quietly). Plus the attribution env-gate test (1).
New/named: `cone_planes_bit_exact_and_complete_rev3/_rev4` (91×87 jpeg pair:
aligned/unaligned/edge rects; **141×301 textured: scale-1 height 150 rows spans
the 128-row strip boundary** — the TRUE multi-strip case; the prior "256×200
multi-strip" claim was wrong since strips iterate per-scale height),
`cone_planes_bit_exact_scalar_tier` (child processes, X64V2+ disabled, Rev3+Rev4),
`wide_rev3_scale1_over_h_tile` (2060×98 → s1 width 1030 > 1024),
`coarse_y_only_plan_coarse_channels_rebuilt`, `sparse_retention_scale0_channels_rebuilt`,
`noop_candidate_yields_exact_zero_deltas`, `refusals_return_none_not_zeros`,
`empty_and_full_rects`, `golden_reference_and_candidate_match_full_recompute_rev3/_rev4`
(tolerance now `4·f32::EPSILON·max(1,|base|,|full|) + 1e-6·|Δ|` — magnitude-aware,
justified in the helper: same f32 phase-A values on both sides, only f64 pool
accumulation order can differ).
**i686 (`cross test -p zensim … --target i686-unknown-linux-gnu`, QEMU): 10
passed** — real 32-bit scalar-tier coverage, stronger than the forced-token
stand-in (which is x86_64-only by construction).

### Gates rerun after the fix round — all green

| gate | result |
|---|---|
| `cargo test --release -p zensim --features custom-profiles,feature-regime-v2,threads,training --lib` | 3 runs, **581 pass each, 0 fail** |
| `cargo test -p zensim --all-features --lib` | 3 runs, **627 pass each, 0 fail** |
| `just rev4serve-gate` | PASS |
| CI-exact `just clippy` | clean (pre-existing dep future-incompat note only) |
| `cargo fmt -p zensim --check` | clean |
| `just lint-scripts` | 807 scripts ok |
| `just api-doc-check` | PASS — public API unchanged |
| 26 perm cells (`~/tmp/perm_check.neighsteer.sh`, `/var/tmp/neighsteer/target-perm`) | **FAILS=0** |
| cross-target `cargo check --tests` aarch64/i686/wasm32 | finished; only pre-existing warnings (fused.rs, ssim_form.rs, dvifm.rs) |
| `cross test` i686 local_refine | **10 pass** (see above) |

### Panels rerun on the fixed binary (`48f6a8ca…`; baseline `5194ca13…` reused)

- **48-case KADID env-on** → `/var/tmp/neighsteer/lane/jp48fix2/`: medians
  IDENTICAL to the pre-fix panel — by_v2fy 03 **0.9827** / 05 **0.9654**,
  v2basic 03 **0.9315** / 05 **0.9008** (oracle columns 0.983/0.966/0.931/0.903).
  Min M2 0.9813 (by_v2fy-5101-I21_10_05); min m3f 0.8338 (v2basic-5103-I21_10_05,
  byte-identical to its pre-fix JSON).
- **env-unset byte-identity**: owner-12 rev4+rev3 48+48 cases, broad-96
  rev4+rev3 384+384 — **0 differing** (`gate.py broad` compares RESULT rows and
  every per-case JSON; `gate_owner.py` whole JSONs). New side regenerated with
  the fixed binary; old side is the dc589ded baseline.
- **Timing** (`cost_per_query_1mp`, release, 1024×1024, best-of-5/3, single
  thread): snapshot heap **54,067,200 B (51.56 MiB)**; full fold-944 walk
  **107.1 ms**; 8×8 query **3,947.9 µs → walk/query 27.1×**; 32×32 query
  **4,882.2 µs → 21.9×**. The earlier "~100–1000× cheaper" claim is WITHDRAWN —
  measured ≈22–27× at 1 MP for a rect at (400,400); the ratio shrinks as the
  rect moves toward the image centre (the residue cone runs to row/strip ends).
