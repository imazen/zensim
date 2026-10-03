# REV4SERVE worklog — canonicalize every served Rev4 leaf

Lane: SWE-2, 2026-10-02. Workspace: `zensim--rev4serve` @ `c1294fd1`.
Bookmark: `quarantine/swe2/rev4serve` (local only). Objective per task
brief: make Rev4 the canonical, tier-independent arithmetic revision on
every *served* path (Zensim compute, BakeScorer, HDR, diffmap,
attribution, prepared steering, corruption head), not only
`research::extract`. Rev1–Rev3 must not move a single bit.

## 1. Reachable-leaf inventory (task 1)

Method: read every served/HDR/diffmap/attribution/prepared/corruption
call path from the public entries down to the tier-dispatched leaves,
starting from the three families named in the brief
(`linear_to_pu_xyb_planar_into`, the edge-only `fused_blur_h_mu` route,
`attr_pass_b_*`), then proved the list by enumerating the engines and
the per-channel dispatch rules in `streaming::process_strip_channel`.

### Engines and who reaches them

- **Fold walk** (`feature_v2::foldapp_streaming_walk` +
  `compute_folded_v1_372_streaming_impl`, `fold_engine::compute_fold_backed*`):
  already canonical at Rev4 — `fused_blur_h_ssim_banded` passes the
  computation's `revision` to `fused_blur_h_ssim_at_revision`, which
  dispatches `fused_blur_h_ssim_canon` (Rec64 f64 sliding recurrence)
  under `featcanon::mode(revision)`; `fused_vblur_features_ssim`
  dispatches `fused_vblur_ssim_canon::<LanesF64>` at `Mode::Canon64`.
  `compute_with_config_core` routes eligible configs
  (`stop.is_none() && is_fold_backable(config)`) through it today.
- **Buffered strips walk** (`compute_multiscale_stats_streaming*`,
  `compute_multiscale_stats_strips*`, `process_scale_bands[_into_accum]`
  → `process_strip_channel`): reaches the same canon kernels on the
  `need_ssim` route (`fused_blur_h_ssim_at_revision`,
  `fused_vblur_features_ssim`); the `!need_ssim` routes (edge-only,
  MSE-only) reach non-canonical leaves, listed below.
- **Diffmap** (strips engine + `diffmap_accum_*`), **attribution**
  (retention copies + `fused_basic_into_at_revision` +
  `attr_pass_b_*` + spread/upsample/bin sinks), **prepared steering**
  (`BakeScorer::prepare_steering` → `planned_features` → strips-with-ref
  or fold-with-ref + attribution), **corruption head**
  (`check_route` + the normal feature walk feeding the head).
- **PU/HDR front-end**: `hdr_source_row_to_nits` (transfer decode +
  gamut matrix) → `linear_to_pu_xyb_planar_into`, used by the fold's
  `FrontEnd::Hdr` arm, `hdr_source_to_xyb`, and the `compute_pu_linear*`
  strips entries via `convert_linear_{planar,interleaved}_to_pu_xyb_into`.

### Non-canonical leaves actually reachable at Rev4

1. **`pu_xyb_rows_inner` + `pu_xyb_pixel` tail**
   (`color::linear_to_pu_xyb_planar_into`). Vector chunks use
   `log2_midp_precise`/`exp2_midp_precise` whose `mul_add`s are fused on
   v3+/neon but *unfused* on scalar/wasm128 (same divergence class as
   the known chunk-vs-remainder XYB bug). The `n mod 8` tail calls
   `pu21::pu21_encode`, which is platform `powf` — also libc-dependent.
   The chunk and tail mix forms even differ: chunks use unfused
   `a*r + b*g + c*b + kb0`, the tail uses fused `mul_add` chains, and
   chunks multiply by `inv_white` where the tail divides by `PU_WHITE`.
   → NEW canonical owner needed (scalar midp replication + fused opsin
   mix, one body for all positions).
2. **`transfer.rs` HDR decoders**: `pq_eotf` (`powf` ×2, plus the
   `decode_pq_u16_rgba_row` LUT built from it), `hlg_inverse_oetf`
   (`exp`), `hlg_system_gamma` (`log10`), and the
   `ys.powf(gamma - 1.0)` in `decode_hlg_row_in_primaries`. All platform
   libm — different bits across libc/toolchains regardless of SIMD.
   → canonical midp arms needed for the Rev4 HDR front-end.
3. **Edge-only strips leaves**, reachable only when
   `active_channels` resolves a channel to `!need_ssim`:
   `fused_blur_h_mu` (tier-dispatched H blur), `box_blur_h_into_abs_diff`,
   `box_blur_1pass_into` activity chain, `build_inline_*`,
   `edge_diff_channel_inline_*`, and `simd_ops::sq_diff_sum` (whose v4
   variant reduces 16-wide vs 8-wide elsewhere — a real lane-grouping
   divergence). Also `sq_diff_sum` in the multi-pass fallback
   (`blur_passes != 1`, already refused at Rev3+ by `check_route`).
   → closed by admission rule, not new arithmetic: at Rev4 every active
   channel computes the SSIM route (see decisions).
4. **`attr_pass_b_*` f32 fused helpers**: audited —
   per-lane IEEE mul/add blends, no `mul_add`, no reductions;
   tier-identical by construction. `box_spread_merge_f32` /
   `box_spread_sum_preserving` are per-row/per-column independent chains
   (serial == rayon bitwise, already gated). The named family is
   precautionary, not defective — kept under the parity gate rather than
   rewritten.
5. **`check_pixel_revision`** (`metric/bake.rs:878`) refuses any bake
   when process or bake is Rev4 — blocks Rev4 bakes outright.
   Post-lift: `refuse_rev4_mix` covers the mix; a Rev4 bake in a Rev4
   process serves via the fold/plan route.

### Verified tier-identical leaves (no change; gated by parity tests)

`downscale_2x*` (lane-local adds, same order per output);
`box_blur_v_from_copy` (per-column independent `sum + add − rem`);
`box_blur_h` (per-row independent, tails aligned by the 2026-09-26 fix);
`abs_diff_rows_into`, `diffmap_accum_*`, `weighted_add`,
`upsample_row_powx_add`, `fused_combine_plane_f32` /
`combine_basic_and_l8` (elementwise);
`compute_xyb_mean_offset` (sequential f64);
`apply_gamut_matrix` (scalar per-pixel 3×3);
`box_spread_*`, `upsample_add_sum_preserving_f32`,
`BinAccum::add_scale_plane_f32` (elementwise or per-line chains);
Scale finalize + `combine_scores` (`det_math` roots, f64 scalars);
zenpredict MLP forward (shared model runtime — identical within a
process; not a revision leaf).

### Unreachable at Rev4 after the admission rule

`fused_blur_h_mu`, `box_blur_h_into_abs_diff`, `box_blur_1pass_into`,
`build_inline_*`, `edge_diff_channel_inline_*`, `ssim_signal_iw_inline`,
`sq_diff_sum`, `abs_diff_sum`, `ssim_channel_extended`,
`edge_diff_channel_extended`, the whole `blur_passes != 1` fallback.
All sit behind `!need_ssim` (or `passes != 1`, refused since Rev3), and
Rev4 strips admission makes every channel `need_ssim`.

## 2. Plan (recorded detail in `rev4serve_decisions.md`)

- New canon helpers: scalar midp `log2/exp2/exp/log10/pow` replicating
  magetypes' `*_midp_precise` formulas with always-fused `mul_add` and
  `round_ties_even` (x86 `vroundps` ties-even == scalar `roundevenf` ==
  `round_ties_even`; verified in magetypes 0.9.28 sources).
- `color`: `linear_to_pu_xyb_planar_into_at_revision` → canon body at
  Rev4; `pu21`: canon `pu21_encode` arm via the midp chain.
- `transfer`: `_at_revision` arms for `pq_eotf`, `hlg_inverse_oetf`,
  `hlg_system_gamma`, the decode rows + PQ16 LUT.
- `feature_v2_stream`: thread `revision` through `hdr_source_*`.
- `streaming`: pass `config.revision()` into the PU converters; at Rev4
  `active_channels` returns all-SSIM channel plans so no non-canonical
  leaf is reachable.
- `ssim_form`: `check_route` keeps the Rev3+ `blur_passes` gate; the
  blanket `refuse_rev4_served` is replaced by `refuse_rev4_mix` plus
  route-local checks; per-entry refusals lifted as listed.
- `metric/bake.rs`: `check_pixel_revision` drops the served refusal,
  keeps the mix refusal.

## 3. Log

- 2026-10-03 — Workspace up at `c1294fd1`; inventory above written from
  full-path source reads (streaming, fold engine, feature_v2[_stream],
  blur, fused, color, transfer, attribution, bake, ssim_form,
  magetypes-0.9.28 generated midp sources).
- 2026-10-03 (cont.) — Inventory code landed: det_math midp canon
  helpers, color PU-XYB canon body + `_at_revision` drivers, pu21 canon
  arm, transfer canon arms + revision threading, feature_v2_stream HDR
  revision plumbing, all-SSIM strips admission at Rev4, refusal sweep
  (`refuse_rev4_served` → `refuse_rev4_mix`), bake narrow-refusal fix
  (v2/basic plans are fold-servable; sampling + corruption companions
  still refuse), contract test rewrite (8/8 pass).

- 2026-10-03 — GATE FINDINGS (measured, `rev4serve_gate.rs`):
  * Every served fold/buffered entry is bit-identical to
    `research::extract` at Rev4 on all tiers (scalar/v3/v4/v4x, 10
    permutations), geometries 64x64..2048x2049 incl. odd/sub-64/4MP.
  * `compute_v2_features` (V2Bounded buffered walk) diverges from the
    fold at Rev4 on the very first basic slot (64x64: 9.7988230696e-2
    vs 9.7931999607e-2) — its v1 moments come from its own H-blurred
    planes, parity-gated but never byte-frozen vs v1. REFUSED at Rev4
    by name ("the V2Bounded buffered walk is not canonical at formula
    revision 4"); the folded entries cover the same slots canonically.
    Also refuses in `compute_v2_features_with_toggles` and both
    `*_with_ref` variants via the shared `compute_v2_features_with_ref_impl`.
  * `compute_streaming_strips` / `compute_with_ref_streaming_strips`
    (STRIP_INNER=256+margin merge) are epsilon-equivalent only — their
    own docstring says "within f64 machine epsilon". At ≤300 rows the
    f64 strip sums stay exact and coincidentally match; at 2048x2048
    the reassociation shows (29 slots, ~1 ULP each, basics+pools).
    REFUSED at Rev4 by name (`refuse_rev4_strips`); buffered/fold
    entries are the canonical equivalents.
  * Real bake gate: v2c `set:v2+basic@h32:H128` cells train on
    Rev4-declared tables but carry no `zentrain.formula_revision` stamp
    (table_admission resolved null). The gate stamps a copy in-memory
    via `zenpredict_bake::append_metadata_utf8`, which is exactly the
    declaration that key exists to carry.
  * Emit-width gap FIXED: the fold emit is a regime width (720 for
    basic+v2 reads) but the bake's identity layout is 1853 wide; the
    `truncate(keep)` could never extend. Changed to `resize(keep, 0.0)`
    in `compute_fold_backed` + `compute_attribution_input` +
    `compute_hdr`: positions past the emit bound are provably-unread
    dead inputs (caller_line_reads ends at f719), and zero is what the
    extraction emits for unpopulated identity slots.
  * `rev4_featpot_bake_served_and_steered` PASSES: 24 held-out aic3
    pairs, pixel score == `score_features(research::extract(...))` bit
    for bit, every read slot identical, `prepare_steering` serves
    (steered score == direct score bit-for-bit, map finite).

- 2026-10-03 — Rev1–Rev3 byte-identity capture harness
  (`ZENSIM_REV4SERVE_CAPTURE`) in the same gate file: 120 deterministic
  pairs × {compute, fold compute, with_ref, diffmap, pu_linear} per
  revision. Baseline run staged in jj workspace `rev4serve-base`
  (/var/tmp/rev4serve/base at c1294fd1) — same test file, so the
  capture format is literally identical on both sides.
- 2026-10-03 — Feature-matrix clippy (CI cell parity): every
  `--no-default-features` cell that fails at c1294fd1 fails with the
  IDENTICAL lint set here (featcanon machinery dead without
  feature-regime-v2 — pre-existing). My additions were clippy-clean once
  `#[allow(clippy::manual_clamp)]` was applied to the three canon
  clamp sites (`det_math::exp2_midp_f32`, `pu21_encode_canon`,
  `pu_xyb_px_exact`) — `.clamp()` is NOT a substitute: it changes NaN
  semantics vs the SIMD `.max().min()` lanes the canon bit-matches.
  `feature-regime-v2` and `oracle,custom-profiles,training` cells clean.

- 2026-10-03 — Default-features `cargo test -p zensim`: all green except
  `feature_v2::tests::rev4_tailhist_quantile_semantics` — PRE-EXISTING
  at c1294fd1 (verified on the base workspace: identical panic at
  feature_v2.rs:1383 `left: 0, right: 1240`; the test only scatters map
  0 while `finish_tailhist_cell`'s debug assert requires all four maps
  to sum to n_px). Not mine, not fixed here — flagged for follow-up.
  `bake_over`/`DVIFM_BLOCK_F32`/`to_f32` dead-code warnings also
  pre-existing at base under default features.

- 2026-10-03 — Perf gate: `fold_engine_bench` ZEN_FE_SIZES=576, base vs
  new alternating A/B/A/B on cores 8-15. The box is running a
  zensim_mlp_train fleet (load ~19) and BOTH binaries alternate between
  a ~5.4 ms mode and a ~6.0 ms mode (score_buffered): run1 base 6.0,
  run1 new 5.4, run2 base 5.4, run2 new 6.0 — ±11% bimodal contention
  noise, no detectable code regression. Expected: Rev1–3 arithmetic is
  byte-identical (G-REG), and the only additions to their paths are
  `== Rev4` integer compares plus resize-for-truncate at bake emit
  boundaries. Any true Rev1–3 cost is below the noise floor on this
  host; rerun on a quiet box if a tighter bound is wanted.

- 2026-10-03 — Test-matrix finish: `rev4serve_gate` needed
  `#[cfg(feature = "custom-profiles")]` on `first_attr` and the
  `prepare_steering` section (that method is custom-profiles +
  feature-regime-v2 gated); clippy fixes in the same file (type alias
  `GeoMap`/`RgbPair`, dropped same-type `as usize` casts). Re-ran the
  gate + contract after the edits: 8/8 contract, 5 pass + 1 ignored
  (real-bake gate passed explicitly earlier), tier-parity 356 s debug.
  All-features lib: 566 passed, 1 failed =
  `rev4_tailhist_quantile_semantics`, verified byte-identical failure at
  c1294fd1 (pre-existing debug_assert bug: the test scatters only
  map 0 while `finish_tailhist_cell` asserts all four maps sum to
  n_px; release compiles the assert out — release lib is 448/448).

- 2026-10-03 — `just clippy` needed one drive-by:
  `examples/diffmap_block_coherence.rs:446` `&number` → `number`
  (clippy::needless_borrows_for_generic_args under rust-1.99). The file
  is byte-identical at c1294fd1 and fails there too — environmental
  lint, not a Rev4 change; fixed anyway because the lane requires the
  gate green. After it, `just clippy` (workspace, --all-targets,
  --all-features, -D warnings) is fully green.

- 2026-10-03 — New tool `zensim-validate/src/bin/bake_stamp_revision.rs`:
  splices `zentrain.formula_revision` onto a bake copy via
  `zenpredict_bake::append_metadata_utf8` (weights byte-identical).
  Exists because the v2c bakes carry the revision only inside the
  `zentrain.repro` JSON blob, not the metadata key the loader reads
  (see D9). Used to mint rev3/rev4 twins of the v2+basic featpot bake
  for the served-cost A/B. 982775 -> 982807 bytes, +32 for the section.

- 2026-10-03 — Served-cost A/B, Rev3 vs Rev4 (the lane's runtime gate).
  Same weights twice: the v2+basic featpot bake stamped rev3 and rev4
  (`bake_stamp_revision`, weights byte-identical). One bench binary
  (`extract_paths_bench`, features custom-profiles,feature-regime-v2,
  threads,training), alternating revision blocks on core 8, single
  thread (`RAYON_NUM_THREADS=1`), nice -n19 + ionice -c3, 3 blocks per
  revision per mode, `ZEN_XP_ROUNDS=15`. Interleaved per
  `rev3_cost_ab.sh` discipline; `fast_ssim2_st` + `D_current_revision`
  are the revision-independent anchors. Raw rounds:
  `/var/tmp/rev4serve/bench/run1/`.

  featpot_v2basic (BakeScorer, pixel->score):
    scalar    1024²: rev3 582.5 ms  rev4 1380.1 ms   +136.9%
    scalar    2048²: rev3 2350.8 ms rev4 5320.5 ms   +126.3%
    prepared  1024²: rev3 704.8 ms  rev4 1670.0 ms   +136.9%
    prepared  2048²: rev3 2780.0 ms rev4 6590.0 ms   +137.1%
  anchors (must NOT move): fast_ssim2_st 1024² −3.1%, 2048² −1.4%;
    D_current_revision 1024² +6.5%, 2048² −3.9% — the v1-only path is
    add/mul-dominated so its canon delta is small, as expected.

  Rev4 serving costs ~2.3× Rev3 on this PU-heavy bake, and that is
  STRUCTURAL, not a defect: the canonical PU-XYB front end
  (`color::pu_xyb_canon`) and mid-precision transcendentals
  (`det_math::{log2,exp2,exp,log10,pow}_midp_f32`) are scalar
  per-element bodies — tier independence is purchased by giving up
  SIMD batching on exactly the transcendental-heavy paths. The
  alternatives (SIMD canon forms with per-tier roundings) are
  precisely what Rev4 exists to remove. If a future lane wants the
  cost back, the correct target is a VECTOR canonical midp kernel —
  same formula, lane-parallel — not a return to tier dispatch.

  Box note: zensim_mlp_train fleet ran throughout (load ~19);
  anchors moved ≤4%, blocks agree within ±3%, so the deltas are
  attributable.

## 2026-10-03 — review fixes (F1–F4) on top of the lane commit

Independent review (`~/tmp/zensim-paper/rev4/REV4SERVE_REVIEW_DONE.md`):
LAND WITH FIXES. Applied as a new commit on top of `04a85d43`, rebased
onto main `07b4cc7f` (which brought the tailhist test fix, featcanon
minimal-build cfg gates, the identical diffmap-example lint fix, and
the WASM job pin — my drive-by copy of the example fix dropped out of
the diff on rebase).

- **F1 — emit-coverage checks at every zero-extension site.** The
  `resize(plan.walk_width(), 0.0)` pattern from D8 could have masked a
  plan/walk disagreement by zero-filling slots `plan.emit` promised
  were computed. New `Plan` API: `emit_bound()` (one past the highest
  promised id), `emit_covered(len)` predicate, `check_emit_covered(len)`
  release check → `PlanError::Uncomputable`. All five sites
  (fold_engine emit, bake SDR arm, bake HDR arm, bake sampling arm,
  steering `compute_attribution_input`) now `debug_assert!(covered)`
  + release-check before resizing. Unit test
  `emit_coverage_check_refuses_ids_the_walk_did_not_materialize`
  (feature_plan.rs:1279): synthetic plan promising ids through 799
  over a 720-emitted walk → error naming `720..800`; full-width emit
  passes; the real 1853-layout/720-emit plan passes.
- **F2** — `REV4SERVE_decisions.md` → `benchmarks/rev4serve_decisions.md`;
  worklog + report references updated.
- **F3** — `just rev4serve-gate` recipe runs the corpus-gated
  `rev4_featpot_bake_served_and_steered` (release,
  `custom-profiles,feature-regime-v2,training`, `--ignored`);
  `REV4SERVE_BAKE` override documented in the justfile comment and the
  `#[ignore]` reason now names the recipe.
- **F4** — featcanon.rs doc line: oracle builds may force canonical
  PU/transfer at Rev1–3 via `ZENSIM_FEATCANON` (measurement-only).

Re-qualification after rebase+fixes: `cargo fmt --check` clean;
`python3 scripts/lint_scripts.py` 797/797; `api-doc-check` green;
debug lib `rev4_tailhist_quantile_semantics` now PASSES (main's fix);
CI clippy matrix 26 cells (`--no-default-features [--features X]
--lib -D warnings`, list from ci.yml): 25 pass; only `custom-profiles`
fails with 5 attribution.rs lattice dead-code items — identical at
main (07b4cc7f did not touch attribution.rs; every production caller
of those items is `feature-regime-v2`-gated and `custom-profiles`
does not imply it). Pre-existing baseline cell, not a lane
regression; flagged to the coordinator.
