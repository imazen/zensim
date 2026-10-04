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

## 2026-10-04 — Step 3 A2/A3 initial arithmetic

Rev5 basic fused pools, v2 dense and gradient pools use sixteen virtual f64
lanes with one adjacent-pair tree. Earlier revisions retain eight lanes.
Existing fused expressions remain; no speculative FMA reassociation is made.
Initial f64 choice preserves the Rev4 per-element error floor; FEATACC
measurement and SIMD speed work remain pending. Extended the existing
`tier_audit_features` owner to request only Rev5's supported families.
Gate: `featcanon_rev5_parity`, release, 3/3 tests pass (0.29 s; a2.log).

### FEATACC initial A1/A2 measurement (before F1)

12 registered pairs, v3, one thread, existing exact/fresh ruler; max relative
error (floor 1e-9), Rev3 -> Rev5: basic 7.95158e-4 -> 2.28536e-4;
peaks 1.15493e-3 -> 2.05528e-4; v2 0.306175 -> 0.0736313.
All three families improve. Worst v2 remains f664, 17x9 crop:
Rev5 3.485591236e-4 vs exact 3.762639432e-4. This is BEFORE stable moments.
Data `/var/tmp/rev5/a2-accuracy-valid/summary.json`; prior `a2-accuracy/`
is INVALID (systemd expanded inline shell variables). No training data mixed.

## 2026-10-04 — Step 4 F1

Extended the existing OnlineMoments owner with two-pass 16-element blocks
and fixed-order Chan/Pébay merges. Rev5 dense folds carry these central
moments through strip accumulation; feature and prepared-map finalizers use
them. Historical revisions keep the original raw-power arithmetic. The exact
Rev5 ruler also uses stable moments. Release constant-plus-small-noise and
constant tests pass at relative 1e-6 (bases 0, .5, 1, 10000; n 1,16,153,1025),
and Rev5 tier parity passes 3/3. Log `/var/tmp/rev5/f1.log`.

## 2026-10-04 — Step 5 F2/F4

Rev5 zero-residue identity invariant passes all 12 existing geometries under
all available token permutations (2.64 s). Computed formulas already cancel
exactly after A1; the initial failure was a registry label: v2 `ssim_mean`
emits mean dissimilarity, so corrected Similarity/HigherIsBetter to
Difference/HigherIsWorse without changing numbers. F15's contradictory
"should be 0" prose removed; PJND_FRAGILITY is reference-only.

Rev5 BakeScorer now calls the same fold with its persistent pixel_scratch;
identity still scores 100, but returns computed features. Release bake test
passes equality with research extraction, including nonzero f393. This also
removes the scorer's fresh per-call V2Scratch allocation (W3). Earlier revision
scorer routing is unchanged. Logs `/var/tmp/rev5/f2.log`.

## 2026-10-04 — Step 6 work removal

W1: Rev5 basic bands read the v2 vertical planes through a radius-zero
canonical fold; each moment plane is blurred only in phase A. W6: serving
no longer promotes Off to Peaks at Rev5, and the canonical basic element
skips peak powers/maxima independently from needed HF ratios. Identity and
tier tests pass after these changes (`a5.log`). Diagnostic 1MP score timings:
by_v2fy 259.5 -> 180.7 ms, v2basic 520.8 -> 353.8 ms. These are NOT qualified
speedups: zenbench marked noisy/resource-gated runs (5/10 rounds).

W5: unread full-resolution X/B now uses two transient rows, downscaled
immediately into scale one; full Y remains. A direct gate compares all read
planes against full conversion under all token permutations, including odd
97x63 and multi-strip 257x289: PASS (`producer-test2.log`). Cached-reference
producer still requires extending this optimization.

W7: Rev5 global XYB mean-offset metadata is defined as [0,0,0]; supported
feature slots do not consume it. No global mean pass or row scratch remains
on this path. Feature accuracy is unchanged by definition; historical
metadata is unchanged. W3 scorer scratch reuse landed with F2. Work counters
are added to existing fold_timing for actual V-plane calls, activity chains,
peak bands, scale-0 X/B consumers, scratch initialization, and stored X/B rows.

## 2026-10-04 — gpt-6.1-sol continuation review

Verified jj history and reviewed inherited arithmetic changes. Inherited full
release suite passes: 589 library tests, all integration binaries and doc tests
(log `/var/tmp/rev5/full-release.log`, wrapper 310 s). Targeted Rev5 gates pass
with vector H/V blur (103 s), vector dense/gradient pools (105 s), and vector
basic fused pools (110 s); logs `vector-gates.log`, `pools-gates.log`,
`basic-gates.log`. These preserve the sixteen-lane tree and local-window values.

Native perf after vector H/V + dense/gradient, before vector basic (1 MP,
20 calls, 5,468 samples): basic fused owner 27.95%, its scalar window closure
24.20%, dense 10.39%, V blur 8.69%, H blur 8.55%. Data
`/var/tmp/rev5/perf-vector-pools.data`. This identifies the radius-zero basic
consumer as the next bottleneck; extended the existing vector owner with
sixteen-lane storage and direct shared-plane reads, without recurrence scratch.

W3: producer buffers now recycle in reverse construction order, making each
next pop match its prior geometry. Previous LIFO order repeatedly paired small
coarse buffers with full-resolution planes. H/V production taps and scalar
fallback taps now use stack buffers. W5 cached-reference walks also skip copying
scale-zero X/B, feeding cached coarse chroma directly. No historical arithmetic
expressions changed; historical byte/panel gates remain required.

### 2026-10-04 — sol continuation: vector kernels, shared map walk, gates

The inherited scalar local-window and pool implementations were functionally
correct but scalarized hot x86 loops. Radius-five H/V windows now evaluate the
same f32 tree in vector bodies, dense/gradient/basic canon64 kernels use sixteen
virtual f64 lanes at Rev5, and Rev1–Rev4 retain their eight-lane arithmetic.
The first 1 MP single-thread profile (5,468 samples) attributed 27.95% to the
basic fold and 24.20% to its scalar local-window closure, motivating the basic
vector body. The subsequent 3,000-sample score profile attributed 22.65% to
plain local V blur, 22.06% to dense v2, 20.48% to fused H blur, 8.54% to XYB,
7.09% to basic fold, and 6.43% to gradients. Raw profiles are under
`/var/tmp/rev5/perf-{vector-pools,external-base-score}.data`.

Rev5 prepared maps now retain the basic SD/mu planes and basic/HF reductions
from the v2 fold, and consume them without a second basic blur/activity walk.
Retention allocates only requested channel-scales, omits unused bs2, and the
map pass skips append arithmetic when bs2 is absent. Scratch recycling visits
strips in reverse release order so the same geometry reuses the same buffers.
The no-scale-0-X/B producer also handles cached reference feeds; a behavioral
comparison against full XYB conversion passes on 64², 97×63 and 257×289 pairs.

The actual work census caught the original duplicate prepared-map walk
(609 V-plane visits and 116 peak bands at 1 MP), then caught an unused bs2
read after selective retention. Both were fixed. The corrected score/map
walks visit 29 strip/channel-scale cells, 145 V planes, 29 activity chains,
zero peak bands, zero scale-0 X/B cells, and zero stored scale-0 X/B rows.
Prepared-map runs at one and eight threads both show zero overwritten-scratch
clears after warmup. Logs: `/var/tmp/rev5/census-map-v2-{1,8}.log` and
`census-score.log`. Cold map allocation clears 7,939,072 elements at one
thread and 8,700,928 at eight threads; cold allocation is not a reuse claim.

Accuracy on the twelve registered FEATACC pairs, production v3 versus exact:

| family | Rev3 maximum relative / absolute | Rev5 maximum relative / absolute |
|---|---:|---:|
| basic | 7.951577e-4 / 5.862508e-5 | 2.804724e-4 / 2.074988e-5 |
| peaks | 1.154929e-3 / 7.483363e-4 | 2.558571e-4 / 1.893044e-4 |
| v2 | 3.061749e-1 / 2.413061e-4 | 1.094758e-3 / 2.989946e-5 |

All three families improve. The 17×9 crop f664 changes from Rev3 production
0.00049146652924 (exact 0.000376263943186) to Rev5 production
0.0003762477981094654 (exact 0.00037626394189433005). Element/blur error is
separate from the central-moment algorithm gate: the oracle-only two-pass
checker reconstructs the SAME f32 leaves, tests all 264 strip moment pairs
across the registered panel, and has maximum relative error 8.665432e-14,
passing the 1e-6 bar. Logs: `/var/tmp/rev5/vector-accuracy/` and
`moment-accuracy-complete.log`. The first moment-auditor invocation failed to
write because its dump directory was absent; the complete rerun passed.

Native full-vector parity passes on 64², 97×63, 131×65 and 255×129 pairs;
WASM SIMD128, i686 scalar, and reachable aarch64/NEON produce the identical
serialized native vectors. WASM uses Rust 1.98.1 (the CI pin); stable 1.99.0
fails in upstream zenflate. i686 uses the native multilib kernel with an env
runner because configured qemu-i386-static is absent. aarch64 uses clang's
cross target and `qemu-aarch64 -L /usr/aarch64-linux-gnu`. Logs:
`/var/tmp/rev5/{native-parity,wasm-rev5-rerun,i686-rev5-rerun,aarch64-rev5}.log`.

Historical bytes: Rev1/2/3 compare all four x86 tiers × twelve registered
pairs against the compatible preexisting review probe: 144 vectors, zero
differences. That probe predates canonical Rev4, so its Rev4 comparison is
not the correct baseline. Using `/var/tmp/rev4canon/target-prod/release/tier_audit_features`
for Rev4 gives 48 more vectors, zero differences. Owner steering identity
has 48 cases per revision and broad identity has 384 rows per revision,
both Rev3 and Rev4: every comparison against saved NEIGHSTEER outputs is
clean. `just rev4serve-gate` passes its real held-out bake/corpus test.
Artifacts: `/var/tmp/rev5/historical-{tiers,steering,r4-canon}/`.

Full release suite passes after shared/selective map retention
(`/var/tmp/rev5/final-full-release.log`, 324 s). All-features library passed
635 tests, 13 ignored before that change; CI-exact clippy, 810 lint-script
checks, and API snapshot check passed. The 27 CI entries contain 26 unique
feature permutations. Initial permutation failures exposed missing cfg
on census calls and feature-specific dead-code annotations; corrected
cells pass except the strengthened bake test initially stamped the enum's
zero-based discriminant. That helper is corrected to discriminant + 1;
final feature/gate reruns follow optimization completion.

Performance qualification caveat: another owner's ignored exhaustive f32
SIMD test (PID 760429) has continuously used one CPU throughout the lane.
It is not this lane's process and has not been interrupted. Strict zenbench
resource-gate flags will be retained; a quiet-box certification cannot be
claimed while this interference exists. Revision comparisons now use
persistent isolated revision workers so rounds interleave without changing
the process-wide formula switch. Parent timings include the pipe round-trip;
setup and warmup are untimed. Fixed/per-pixel fits must disclose this fixed
IPC overhead rather than presenting the intercept as pure extraction cost.

The widened AVX-512 local H/V windows keep the same tree and pass the seven
Rev5 library tests. In 30 interleaved rounds, 1/4 MP single-thread score
changes 26.59/101.23 ms → 24.55/91.76 ms; maps change
54.16/213.72 ms → 52.01/200.55 ms. This is provisional (every round marked
noisy), but the change is retained. An explicit vector update of basic f64
pool lanes regresses v4x 4 MP score by 6.80%; reverted. A two-vector unroll
of plain V blur regresses v4x 4 MP score by 11.34%; reverted. Correct v3
cap is `ZENSIM_MAX_TIER=v3`; the first pool trial's v3-labelled rows had no
cap and are native repeats. `/var/tmp/rev5/trials/poolvec-v3/` is the actual
v3 rerun. Unchanged v3 code in the AVX-512-only unroll trial still varies
by as much as 3.59% in one map comparison, illustrating the noise limit.

Prepared-map profiling (`perf-wide-map.data`, 3,000 samples) identifies
13.05% self time in retention and another 7.80% in fmaf, chiefly retention.
The retained basic SD plane now uses the existing vector SSIM expression,
with correctly rounded scalar tails. The complete production-scope bake
test compares features and map densities across every token permutation,
and passes. Thirty-round provisional v4x maps improve 50.07/192.99 ms →
41.48/163.29 ms (17.17%/15.39% at 1/4 MP); score is effectively unchanged.

The vector map-SD path also improves v3 maps 57.56/220.39 ms →
46.51/183.45 ms (19.19%/16.76%). It is retained. The next score profile
puts dense v2 at 26.51%, fused H at 18.54%, V at 14.98%, XYB at 10.63%,
basic fold at 8.10%, and gradients at 7.71% (3,000 samples).
Computing paired H tap groups before advancing to the next group reduces
live intermediate vectors while preserving every leaf and tree operation.
Provisional v4x scores improve 23.55/91.57 ms → 22.71/82.99 ms;
maps improve 41.60/160.63 ms → 38.34/153.21 ms. v3 is unchanged in source.
Full-vector parity remains byte-identical to the frozen native capture,
and F2 identity passes on every native token permutation
(`/var/tmp/rev5/tapstream-parity.log`, 14 s).
