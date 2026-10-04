# REV5 implementation worklog (SWE-2 lane)

Workspace `~/work/zen/zensim--rev5`, bookmark `quarantine/swe2/rev5`. Spec:
`benchmarks/rev5_spec_2026-10-04.md`. Brief: `~/tmp/zensim-paper/rev4/REV5_brief_v2.md`.
Coordinator's pipeline side: `benchmarks/rev5_pipeline_WORKLOG.md`.

## 2026-10-04 — Step 1: plumbing

`FormulaRevision::Rev5` registered (era `localwin`, extends the Rev4 token
list). Every `FormulaRevision` match site extended: `feature_defs` (enum +
era tokens + `localwin` era registered as `ARITHMETIC_REVISIONS` /
`DEFECT_WINDOW_DRIFT`/`REV_LOCALWIN`), `ssim_form` (`ZENSIM_FORMULA_REV=5`,
`SsimLumaForm` → Rev2 form), `feature_layout` (`"5"|"rev5"|"Rev5"` bake
metadata), `corruption_head` (revision field 5), `hf_gain_form` (Rev2 form),
`det_math` (`RootForm::NestedSqrt`, `PowForm::PureRust`), `color` (opsin →
canonical body via `featcanon::mode(Rev5) == Canon64`), `streaming` (stamp
5), `feature_v2` (`TailAccum` folded bins gated `>= Rev4`), `metric/bake`
(error text generalized to "4 and later"), `attribution` (`>= Rev4`).

**Rev5 two-family scope** (`basic + peaks + v2`, spec §1), two mechanisms:

- `feature_plan::PlanError::UnsupportedAtRev5 { extra }` + shared
  `rev5_unsupported(want, layout_width, revision)` called by
  `Plan::derive_with_layout` (process revision) and `Plan::for_bake` (the
  bake's own declared revision — the derive checks the process switch, so a
  Rev5-declared bake in a pre-5 process is still scope-checked). Refusal is
  `want ∖ (basic ∪ peaks ∪ v2)`; `missing_from`'s operand order is
  `b∖a`-on-`a.missing_from(&b)` — first draft had it backwards (caught by the
  tests below).
- `ComputeSet::rev5_scope()` — clears every flag outside the three families,
  degrades the masked/IW pool modes to `Peaks`, drops free extras. Applied
  inside `ComputeSet::from_toggles` under the Rev5 revision **and** at the
  walk's own compute resolution in `foldapp_streaming_walk_impl`
  (belt-and-suspenders for a hand-built `ComputeSet`). Rationale:
  `Plan::toggles()` sets `append_block`/`csfw_block`/`rev4_*`/… from the
  LAYOUT width, and `Plan::normalized` runs `from_toggles(probe.toggles())`
  — without the clamp, a `basic,peaks,v2` request at the full 1825-wide
  layout would compute every family the width reaches (the coordinator's
  measured `basic+peaks+v2+append+append2+csfw+…` feature-set id). After the
  clamp the compute set — and therefore `populated_slots`, `emit`,
  `compute_parts`, the producer `feature_set_id` — names only the three.

  `emit` for the pipeline request is exactly the 576 requested slots;
  `feature_set_id` is `basic+peaks+v2`. Unrequested positions stay
  structural zeros (`provenance.populated == false`; the assembler turns
  them into NaN).

Tests (all in a `ZENSIM_FORMULA_REV=5` child via `run_at_revision`):

- `research::tests::rev5_basic_peaks_v2_request_computes_only_the_three_families`
  — `extract` succeeds at the full width; `emitted()` is exactly the 576
  slots; the producer id's compute parts are `{Basic, Peaks, V2}` only;
  provenance `populated` flags equal the request; one slot from each
  unsupported family (masked 228, iw 300, append 720, append2 924, csfw
  944, bank 1100) refuses with the Rev5 scope message. PASS.
- `feature_plan::tests::rev5_for_bake_refuses_reads_outside_the_supported_families`
  — `Plan::for_bake` directly: supported ids plan, unsupported refuse as
  `UnsupportedAtRev5` naming the slot. PASS.
- `metric::bake::revision_contract_tests::rev5_bake_serves_supported_reads_and_refuses_unsupported`
  — a Rev5-declaring bake reading basic/peaks/v2 serves through
  `BakeScorer`; an unsupported read refuses at `check_servable`'s eager
  `plan()`; a Rev4-declaring bake in the Rev5 process is the refused mix.
  PASS.

Per the brief's step-1 allowance, Rev5 still computes the supported
families with Rev4 canonical arithmetic (`featcanon::mode(Rev5) =
Canon64`); the local-window arithmetic is step 2.

**Not changed, deliberately:** `zensim-validate`'s
`ADMITTED_FORMULA_REVISIONS = [1,2,3]` mirrors the committed registry
JSON's era declarations, which top out at Rev3 — a Rev5 era entry updates
it with its registration, as Rev4's never landed there. Recorded in
`REV5_decisions.md`.

Checks: `cargo check` clean; the three new tests pass; the 78 tests in the
touched modules pass; `cargo fmt` applied; `just api-doc` regenerated the
snapshots (`Rev5` added to the public-enum list — the brief's single
approved public item); CHANGELOG entry written.

## 2026-10-04 — Codex takeover; Step 2 A1

Reviewed inherited `pskvvvlo` including the stamping edits and parity test.
The inherited local plain blur bodies were uncalled: corrected H/V and
activity dispatch to select local windows at Rev5 while preserving earlier
product dispatch. Removed the unused duplicate one-pass wrapper. Added the
required token-testing lock to the permutation test.

Gate: release `cargo test -p zensim --features custom-profiles,feature-regime-v2,threads,training rev5 -- --nocapture`
passes (log `/var/tmp/rev5/a1.log`, wrapper elapsed 103 s). Direct window
checks cover 1x1, 3x2, 17x9, 97x63, four fused H planes and plain H/V;
f64 window error bounded by 8 f32 eps relative. Perturbation test proves
unchanged outputs outside the five-pixel support. Full-vector tier test
covers 64x64, 97x63, 131x65, 255x129. Rev5 differs from Rev4 on the fixture.
No speed claim yet: inherited local kernels are scalar.
