# Reachable targets and codec steering — September 8, 2026

User ruling: establish each image's codec range before judging steering; fit
codec-specific seeds/heuristics on imazen/imazen-26 **training** content; compare
1, 2 and 3 shots. JXL is the first diffmap integration, with JPEG and WebP also
in scope, followed by deeper AVIF work. The user still controls one score,
including negative scores. Do not rescale that dial per image or codec.

This supersedes the July 18 G-TARGET screen as a qualification procedure. The
September 7 midpoint/bisection measurements remain reproducible observations,
but their mixed feasible/infeasible median does not establish a model failure.
The old median-error <=2 bar was a research screen, not a perceptually justified
product tolerance. Report error distributions and hit rates at explicit bands;
no new universal perceptual error bar is established here.

## Preregistered instrument change

### Complete candidate sensitivities — September 8 shipping goal

The active goal includes a qualified model, Rust scalar/spatial serving and
actual JXL/AVIF/JPEG/WebP targeting, with supported HDR/color/alpha behavior.
The reuse audit and earlier scalar experiments do not complete that goal.

Concrete current caller: `zensim/examples/diffmap_block_coherence.rs`;
subsequent native caller: JXL's existing attribution loop. Before implementation,
register this additive signature on the existing `BakeScorer` owner:

```rust
pub fn score_features_fd_gradient(
    &mut self, features: &[f64], width: u32, height: u32,
    codec_hint: Option<&str>,
) -> Result<Vec<f64>, ZensimError>;
```

It differentiates the complete `score_features` operation, including output
disposition, ensembles and corruption companions, in identity feature layout.
Use the existing central-difference step `max(abs(f[k])*1e-3, 1e-5)`. Reject
nonfinite inputs, scores or probe results; never convert a failed model into a
zero gradient. At nonsmooth heads/gates this is a finite local secant, not a
claim of differentiability or of finite-block accuracy.

The coherence caller must extract through `BakeScorer::compute`, preserve the
full declared-ID row, and score finite pixel interventions through that same
pixel surface (including its identity rule). Remove input-count routing and
manual structural-zero assumptions there. Existing M1/M3 signal-fold controls
remain historical comparators; the signed attribution owner remains M3a.
No quality/qualification threshold changes are authorized by this instrument
repair. Validate linear known derivatives, dense layout, head/spline/affine
composition, active gates, negative scores and sequential-surface parity.

### Candidate score/map binding — registered before implementation

The coherence example and the recovered native JXL loop are concrete callers.
Add the following on `BakeScorer`, gated by `custom-profiles` and
`feature-regime-v2`, using existing reference and retention owners:

```rust
pub fn precompute_reference(&self, source: &impl ImageSource)
    -> Result<PrecomputedReference, ZensimError>;
pub fn compute_with_ref_and_attribution(
    &mut self, source: &impl ImageSource, precomputed: &PrecomputedReference,
    distorted: &impl ImageSource, codec_hint: Option<&str>,
    session: &mut Fused944Session, bin: usize,
) -> Result<ScoredAttribution, ZensimError>;
```

`ScoredAttribution` is an opaque returned object, exported under those same
features. Its exact read-only accessors are `result() -> &ZensimResult`,
`attribution() -> &AttributionResult`, `sensitivities() -> &[f64]`,
`unsupported_feature_ids() -> &[usize]` and `has_corruption_gate() -> bool`.
The last two make approximation limits visible at the call boundary; zero
local sensitivity is never proof that a finite intervention is inert.

Use the existing planned fold with retention hooks, complete `BakeScorer`
sensitivities and existing binned attribution assembly. The returned scalar
and consumed feature values must match ordinary `compute` exactly. Keep
negative scores and pixel identity. Reuse sessions and reference caches;
no process-global model or first-image gradient. Reference caches must be
built by this surface from the same source image. Spatial coverage must
explicitly distinguish unimplemented integrands from reference-only and
SDR structural-zero terms. Unsupported extraction variants must refuse or
report their unsupported terms, never silently use default feature semantics.
SDR attribution does not imply an HDR attribution qualification.

Before native evidence: test plan/score/consumed-feature parity on shipped
bakes and composed fixtures, binned/per-pixel consistency, session reuse,
identity, tiny/odd sizes, negative outputs, unsupported slots and declared
formula/activity variants. Existing attribution tests protect legacy callers.

The first C smoke timing exposed roughly 250 ms in the complete
score/sensitivity/map call at 256 pixels. Before native use, allow only exact
finite-probe acceleration: skip IDs absent from every active member/head,
validate nonfinite probes before that shortcut, and parallelize independent
columns with private predictor state when threading is available. Preserve
per-forward arithmetic and composition order. Gate against an explicit
sequential complete-surface oracle, including ensembles; record measured
timing separately from correctness and avoid a new scoring implementation.

### Native complete-encode calibration reuse — registered September 8

Concrete caller: JXL's existing `zensim_diffmap_rd` example. Reuse the
`zensim-target` search and median/envelope calibration owner for actual
1/2/3 full encodes; do not copy that controller into each codec harness.
The native adapter uses JXL's existing two-update map loop at each proposed
distance, independently decodes through jxl-rs and records native work.
No evaluation bounds or image-specific ladder seeds reach the controller.

Add to the unpublished `zensim-target` tool library:

```rust
pub fn target_search_with_backend_and_bake(
    rgb: &[u8], width: u32, height: u32, codec: CodecKind,
    spec: TargetSpec, backend: &dyn codec::CodecBackend,
    scorer: &mut zensim::BakeScorer<'_>,
) -> anyhow::Result<TargetResult>;
```

The pre-existing packed-sRGB contract remains explicit; codec identity selects
the model's codec hint and result label, while the supplied backend owns native
configuration/range. Built-in wrappers retain enabled-feature checks. Add an
opaque serializable `SeedCurve` with `fit(rows: &[Vec<(f32, f32)>], inverted:
bool) -> anyhow::Result<Self>`, `estimate(target: f32) -> anyhow::Result<SeedEstimate>`,
`points() -> &[(f32, f32)]` and `adjusted_points() -> usize`. Move the existing
median/monotone-envelope/inverse-segment implementation there, preserving
its arithmetic and artifact fields. Reject empty, nonfinite, misaligned or
unordered input curves and unusable seed slopes. Existing source/split/model
identity validation remains at the experiment boundary; `SeedCurve` alone
cannot certify training provenance.

The first native experiment reuses the already recorded 12 training and
8 validation families. Freeze codec/model/arm calibration before validation;
report fixed requests and five ladder-witnessed targets in original units,
with error bands 0.25/0.5/1/2 and full tails/coverage. Compare budgets 1/2/3,
midpoint versus frozen training seed, and neutral versus active H3. Include
a scalar-only zero-update arm for the actual latency baseline; it does not
pay for unused native maps. Regenerate the ladders with the final driver
before fitting/evaluation. Keep
full encodes, internal reconstructions, scores, maps, terminal verification,
bytes, total time and memory separate. This extends the existing instrument;
it does not establish a new universal perceptual tolerance or waive RD gates.

### Earlier scalar bounds/calibration instrument

Owner: `zensim-target`, extending its existing search and `demo_matrix`.
Concrete caller: the imazen-26 calibration/steering experiment. Unpublished API
delta, registered before implementation: `TargetSpec` gains an optional
`SeedEstimate { knob, score_per_knob }`. The existing search uses the estimate
for shot 1, its training slope for shot 2, and a bracket-safeguarded measured
secant for later shots. An absent seed preserves midpoint behavior. Estimates
must be finite, in the codec range and have the correct slope sign. No new
metric inference API or second controller is introduced.

`demo_matrix` gains an explicit bound-sweep/calibration/evaluation mode. Fit
and evaluation execute in Rust; candidate scores always call `BakeScorer`,
named scores call `Zensim`. Calibration is a codec/model-specific curve table,
not a change to the zensim score scale. Freeze it before evaluating other
source families. A calibration artifact includes scorer, codec configuration,
binary, source/split and probe identities; mismatches must fail.

## Bounds before steering

1. Pin original bytes, rendered RGB interpretation, dimensions, encoder and
   decoder commits, codec configuration (including chroma, effort, resampling,
   filters, lossless mode), metric bytes/formula revision and map mode.
2. Encode both ends and an interior ladder over the **declared configuration's**
   full supported knob range. Score the actual reconstructions. Save encoded
   and decoded hashes, bytes, scores and measurement cost. Deduplicate
   saturation by encoded hash, not assumed q intervals. Inspect reversals and
   discontinuities. Lossless and alternative chroma/resampling are separate
   configurations, not unreported extensions of the same range.
3. The minimum/maximum observed scores are **attained bounds on the sampled
   family**, not a proof of extrema over every possible codec mode or every
   floating-point knob. Endpoint-only monotonicity is not assumed. Targets
   with a measured reconstruction in their tolerance band are witnessed
   feasible. Other fixed requests are outside the measured envelope or
   unwitnessed inside it; neither category is silently called impossible.
4. Evaluate the steering model on witnessed targets, including targets drawn
   from the ladder itself to exercise each image's usable range and gaps.
   Keep the original score coordinates. Report excluded/unwitnessed requests
   and coverage beside errors. A sparse grid cannot qualify range coverage.
   Separate G-ADDR still prevents a collapsed model dial from earning an easy
   targeting pass.
5. Bounds are an evaluation oracle only. The timed controller gets source
   pixels, target, frozen training calibration and its own prior probes. It
   must never receive evaluation q/score pairs, endpoint scores or an
   image-specific seed chosen from the bound sweep. Report oracle cost
   separately; do not conceal it inside a claimed one-shot production method.

## Calibration and measurements

- Canonical corpus moved from `imazen/codec-corpus` to `imazen/imazen-26` on
  August 23. Its August 27 `split_map_family.tsv` adds shared-content families;
  the old local README/digit-only tables alone are insufficient. Use both
  canonical origin membership and family separation; exclude disagreements
  for the initial experiment. Pin the latest manifest bytes. Read measured
  render URLs from the August 30 index; do not derive them by extension swap.
- First baseline: median training score at each native knob, followed by an
  explicitly recorded monotone running envelope; invert that curve per codec
  and scorer, with its local slope. Compare the same images/targets with midpoint search.
  This bounded baseline is not a replacement for the existing codec-specific
  content predictors. Fit future predictor/heuristic choices on train only;
  use source-family-disjoint validation for development. Reserve terminal
  test for a frozen, adequately covered candidate.
- Run budgets 1, 2 and 3 separately, preserving early stopping and actual work.
  Report median, p90/p95, worst and signed error, undershoot, hit rates at
  explicit tolerances, bytes, total latency, score/map latency and memory.
  Group by codec, content family, size and target band; uncertainty is by
  source family, never by treating repeated requests as independent images.
- An outer shot is a full encode/decode/score. Native JXL reconstructs inside
  one encode: report full encodes, internal reconstructions, scalar compares,
  map evaluations, terminal decode/verification and time **separately**.
  Do not rename K updates plus an initial compare as K complete shots.
- Diffmap trials must prove the candidate map was consumed and changed the
  intended quantization/RDO state. Compare active and neutral maps at the
  same target/budget, including an engagement trace. Use independent SSIM2
  and Butteraugli judges at matched quality/bytes; scalar accuracy alone is
  not evidence of better allocation. Re-score the emitted bitstream, not
  only the encoder's internal reconstruction.
- No default model or codec setting changes on a smoke result. Preserve failed
  experiments. Expand source/size coverage before a product qualification.

## Branch and implementation audit

Later September 8 correction: the [fleet and attribution reuse audit](../benchmarks/diffmap_reuse_audit_2026-09-08.md)
extends this initial inventory. JXL already has fused/binned/stale attribution
and H3 steering. JPEG's `zenjpeg--zensim-diffmap-rd` is a `jj` workspace at
`46a6ff30`, absent from a Git-only worktree listing and not merged into main.
AVIF's later `examples/zensim_cq_rd.rs` already implements folded-944/H3,
split-role maps and a hint-engagement probe; its August 7 report contains
August 29 negative results and must be read through its final sections.
The candidate surface is now implemented and JXL consumes it in commit
`9f038d4dd7f4`. Its 840-ladder D/H3 screen proves native map engagement but
not a broad independent-judge RD benefit. The [native record](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md)
retains the negative findings. Reuse the other codec owners in the same way.

Remote refs were refreshed September 8 without changing sibling checkouts.
The exact inventory and local dirty-state records are retained in the private
target-audit directory. Main branches were newer than several local HEADs.

| Codec | Existing owner and mechanism | Follow-through |
|---|---|---|
| JXL | `jxl-encoder/src/vardct/zensim_loop.rs`, `examples/zensim_diffmap_rd.rs`: global controller plus map-driven quant-field redistribution, map/model experiments and seed head | First native integration. August 27 independent-judge results and August 30 secant study precede September 7 reconstruction corrections. Rejected `issue103-{coordinate-control,full-distance-feedback,targeting-research,upsample-phase-order}` branches retain negative research. Do not repeat or merge them blindly. |
| JPEG | `zenjpeg/src/target_quality.rs`: float-native outer secant; `encode/zq.rs` behind `target-zq`: global correction plus per-block AQ feedback; `zq_seed.rs` and bucket anchors | Include real diffmap engagement; reconcile existing profile/seed identity and pass accounting before candidate comparisons. `feat/quality-zq-target`, `feat/zq-bucket-calibration`, `feat/zq-linear-f32`, `explore/perceptual-loops` preserve earlier work. |
| WebP | `src/encoder/zensim_target.rs` behind `target-zensim`: outer q correction plus per-segment map correction near the band | Map path is conditional, not guaranteed by enabling the feature. One-pass metrics can be NaN/optimistic, so independently measure final bytes. `recovered/zensim-target-extended-eval-a8516` and `spike/zenpicker-knobs` are historical experiments. |
| AVIF | `src/target_quality.rs`, `two_pass_zensim.rs` and `sb_pool.rs`; `cooptloop` branch connects rav1e research | Inventory now; deeper map integration after JXL, with JPEG/WebP included. Preserve July 12 negative two-pass results and the dirty rav1e trace file. |

Existing native map integrations often construct `ZensimProfile::Custom` or
older named profiles. `BakeScorer` currently exposes complete scalar serving,
not a complete composed-model map surface. Do not describe old per-bake
gradient mounting as support for every multi-head/ensemble/corruption model.
That serving gap must be closed and parity-tested before those candidates can
qualify in native map loops. Keep this distinct from the scalar baseline.

First execution and remaining native integration details:
[September 8 measurement record](../benchmarks/target_steering_bounds_2026-09-08.md).
The existing alternate JXL `zensim_backend.rs` bridge also clamps
`100 - score` to 0..100; the dedicated `zensim_loop.rs` uses another loss
conversion. Audit the actual selected route before claiming negative-target
support. The generic scalar controller measured here never clamps scores.

### Native JXL implementation result

The preregistered native adapter and shared `SeedCurve` now run in the JXL RD
example. Its new `--native-fit`/`--native-eval` modes regenerate train and
validation ladders for the pinned driver/configuration; the historical target
mode remains a separate experiment. `rd_probe_analyze_2026-07-18.py` validates
its full matrix, emitted bytes and native work counts, then applies the existing
independent-judge interpolation owner. The [native result](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_native_targeting_2026-09-08.md)
retains narrow witnessed coverage and mixed RD as release limitations. A ±1
controller tolerance here is an experimental band, not a perceptual acceptance
threshold. No new product tolerance has been established.

### Shared native probe owner and AVIF actual-encode study (September 8)

Before implementation: move reusable source-manifest, calibration, bounds,
coverage and emitted-byte verification orchestration from the existing JXL
research instrument into `zensim-target::native_probe`, behind the additive
`native-probe` tool feature. This unpublished tooling surface has a concrete
caller in `zenavif/examples/zensim_cq_rd/targeting.rs`; the JXL driver remains a
reproduction reference until ported. Scoring and search remain existing owners.

Exact new public signatures (tooling only):

```rust
pub trait NativeProbeBackend: codec::CodecBackend {
    fn take_work(&self) -> Vec<NativeProbeWork>;
}
pub trait NativeProbeCodec {
    fn codec(&self) -> CodecKind;
    fn configuration(&self) -> &str;
    fn quality_knots(&self) -> &[f32];
    fn bound_encodes(&self, arm: &str) -> usize;
    fn backend<'a>(&'a self, arm: &str, model: &'a zenpredict::Model,
        scratch: &Path, bake: &Path) -> Result<Box<dyn NativeProbeBackend + 'a>>;
    fn decode(&self, encoded: &[u8], width: u32, height: u32) -> Result<Vec<u8>>;
}
pub fn fit(codec: &impl NativeProbeCodec, sources: &Path,
    bake: &Path, output: &Path) -> Result<()>;
pub fn evaluate(codec: &impl NativeProbeCodec, sources: &Path,
    calibration: &Path, bake: &Path, output: &Path) -> Result<()>;
```

`decode` returns exactly width*height*3 tightly packed opaque sRGB8 bytes.
`NativeProbeWork` is the serialized per-complete-encode accounting record:
public fields `knob: f32`, `encode_seconds: f64`, `decode_seconds: f64`,
`internal_reconstructions: usize`, `native_pixel_comparisons: usize`,
`map_evaluations: usize`, `consumed_maps: usize`, `native_loop_ms: f64`,
`encoded_sha256: String`, `decoded_sha256: String`. Each backend call must perform
exactly one full encode and decode; model maps may use that decoded image but
must report their extra scalar/map work. Fixed arms are scalar/neutral/active.

AVIF uses 17 integer CQ knots spanning 1..255, speed 6, Zenravif, 4:4:4,
8-bit opaque RGB, one encoder thread, formula revision 1, bin 8 and the existing
zerosum gain-10/factor-1.15 rule. Scalar has no map work; neutral computes maps
with gain 0; active uses the preceding reconstruction's map. One-shot active
therefore has no consumed spatial map. State resets for every target case and
every bound CQ. Bounds record one encode for scalar and three for neutral/active;
every encode and map is counted, and the final emitted state is measured.
Calibration always uses each bound sequence's FIRST decode score, since the
first actual encode has no preceding reconstruction. Train-only median curves
and slopes are from the existing Rust `SeedCurve`; the existing shared search
compares midpoint and calibrated seeds at actual budgets 1/2/3.

Use the existing 12 train / 8 validation imazen-26 canonical family manifests.
Finish all per-image/arm ladders before target evaluation; pass no attained
bound, image optimum or validation-fitted seed to the controller. Preserve the
existing fixed requests [-10,30,70,90,99] plus five neutral-ladder score quantiles;
report all coverage and only count jointly witnessed ±1 requests in the error
screen. This conservative sparse-ladder rule does not prove gaps impossible.
Report wider error bands, signed undershoot, tail error, selected emitted bytes,
complete cost and matched SSIMULACRA2/Butteraugli outcomes. This is development
validation, not terminal qualification or a newly justified perceptual tolerance.

## Native finite-block analysis extension — September 8 (before outcome analysis)

The existing `rd_probe_analyze_2026-07-18.py` owner will accept
`--interventions <directory>` for the codec-owned `native-jxl-interventions-v1`
instrument. Codec registration: sibling
`jxl-encoder/docs/zensim-native-interventions-2026-09-08.md`.
No scoring/controller/statistics implementation is replaced. Use SciPy's
Spearman implementation for rank summaries; preserve constant/insufficient
samples explicitly rather than returning a fabricated correlation.

Before deriving results, verify completion and expected cell/probe coverage,
source and emitted-byte/pixel/quantizer hashes, actual global quantizer constancy,
neutral-repeat identity, source-clipped transform regions, deterministic raster
sampling, requested local changes and observed local/nonlocal quantizer changes.
Both independent judges must cover every exact reference/distorted pair without
duplicates/nonfinite values. Include rejection controls for altered counts,
quantizer records, image hashes and judge identities. Record native scalar and
judge deltas for both directions. Rank associations use central local differences
per actual log-quantizer change and report map mass, density and byte cost
separately within each image/distance. Flat quantizers and nonpositive rate
changes stay visible. This mechanism evidence is not a targeting or RD release
gate; no acceptance threshold is fitted after inspecting outcomes.


## Native IO and coarse JXL follow-up

Registered 2026-09-08T20:09:41.967730+00:00, before implementation and new encodes. Extends the existing
`zensim_diffmap_rd --native-interventions` owner and its existing analyzer.
Previous per-transform +/-10% results remain immutable and mixed.

Freeze the same D artifact, formula 1, source manifests, four training origins
2010/6068/7066/8206, distances 1 and 3, effort 8 Reference, normal transform
strategies/CfL/gaborish, exact decoder transfer and cached precomputed fields.
No validation or terminal examples, no fit or policy selection.

First replace this private instrument's PNG IO with zenpng 0.1.4 and record
native decoded RGB hashes. Reproduce the old 272 transform probes using the
new IO; require all emitted JXL bytes, raw RGB, quantizers, scores and map
integrals to match the retained screen-final packet. PNG compression may
change. Historical PIL checks remain only in the v1 analyzer branch.
Source admission requires static, opaque, eight-bit sRGB-compatible PNG;
unsupported metadata/depth/alpha fail explicitly, not silent conversion.

Coarse mode: assign each complete transform to one of 4x4 grid cells by its
anchor block: gx=floor(4*x/xsize_blocks), gy=floor(4*y/ysize_blocks).
Keep all members whole; groups are unions of transforms, not assumed
rectangles. Enumerate nonempty cells in raster order. Every source pixel and
padded block belongs to exactly one group. Sum clipped areas and signed map
integrals, with density=mass/area. Large transforms may cross grid boundaries.
For each group change all covered raw quantizers by factors 0.8 and 1.2,
rounding with minimum step one and clamp [1,255]. Baseline and exact neutral
repeat plus two probes per nonempty group; at most 34 full encodes per cell.
No response-based group sampling, threshold tuning or omitted inert probes.

Measure actual captured quantizer changes, raw-pixel changes inside/outside
the union, complete D score, bytes, CPU SSIMULACRA2 and Butteraugli. Retain
nonpositive central byte differences; compute signed central derivatives
normalized by actual mean-log-q span and rank associations with mass/density
and quality gained per byte where the byte derivative is positive. Report
each image/distance, not only pooled correlation. Count all encodes, native
JXL decodes, comparisons, maps, source/PNG roundtrip decodes, compatibility
decodes, times and RSS. libjxl v0.12 remains port compatibility only.

Run the same coarse configuration on the existing 512-long-edge variants
of these four origins as separate multigroup/scale coverage, not new families.
Require whole-transform coverage, neutral byte/pixel/q/score equality, native
PNG readback parity, complete independent judges and content hashes. Add
negative controls for group membership, quantizer requests, PNG hashes and
coverage. Reproduce final-source output if implementation changes after run.

This bounded screen ends after these fixed experiments and integrity checks.
It cannot establish held-out RD gains, 1/2/3-shot target accuracy, or release
qualification. Do not turn descriptive correlation into a retrospective gate.
No runtime oracle probing or new allocation policy is authorized by these
measurements alone; preregister the next intervention separately.


# Coarse JXL allocation policy — registered 2026-09-08T20:35:32.658483+00:00

Question: can one complete-model coarse map improve an actual emitted JXL
against all locally reachable global quantizer fields, not just correlate with
single-region quality responses? No fitted model or seed, no new public API.

Reuse the current native precomputed e8 Reference/CfL/gaborish/pixel-loss path,
D bake cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6,
formula 1, exact decoder transfer, native PNG IO and complete Rust surface.
First canonical train origin per class: 2010/6068/7066/8206, distances 1 and 3,
existing 256-long-edge variants. No validation or terminal sources in this
first bounded policy screen. No new feature/model training or source admission
exception. Frozen AC strategies are shared by all fields in each cell.

Build all scalar control fields BEFORE applying the policy. Domain: multiply
all initial integer raw q by one common factor in [2/3,3/2], half-up round,
clamp [1,255]. Enumerate exact rational breakpoints (2n+1)/(2q), n=1..254,
plus both endpoints; sort by integer cross-products and deduplicate resulting
full raw fields. Every distinct state of this declared scalar operation is
represented. Do not resample only convenient factors. Refuse more than 4096
states in a cell rather than silently truncate. Encode/decode/score each
unique field, retain counts/order and attained D/byte bounds. Baseline and an
independent exact neutral repeat remain explicit. This is an exhaustive local
raw-field comparator, not a claim to exhaust the codec's full distance range.

Policy inputs are ONLY baseline complete attribution, existing whole-transform
4x4 groups, and the original raw field. No control outcome, judge, bound or
intervention derivative is available to the policy. For group density d and
pixel area A, compute area-weighted center mu and mean absolute deviation m.
If m <= 1e-20, factors are one. Otherwise set f=1+0.2*clamp((d-mu)/m,-1,1).
Expand each factor to every block in its complete transforms. Preserve the
initial sum of requested q before rounding by multiplying all factors by
sum(raw)/sum(raw*f); then half-up round and clamp [1,255]. This preserves a
quantizer-sum proxy, not actual bytes. Record unnormalized/normalized factors,
center, dispersion, raw-sum normalization and actual requested/captured fields.
Apply the same function to zero densities and require raw-field/byte/pixel/
score identity with the neutral repeat. Active is one actual full encode after
one baseline full encode and one map: two encodes and one map, with all extra
control encodes separately labeled engineering cost. No runtime oracle probes.

Every emitted control and active output receives native JXL decoding, complete
D scoring and independent CPU SSIMULACRA2/Butteraugli. Bounded libjxl v0.12
compatibility checks remain port-only. Report exact measured scalar frontiers:
for each active byte budget, best scalar D/SSIM2/-BA within budget; for each
active quality, minimum measured scalar bytes meeting/exceeding that quality.
Keep no-match/coverage failures explicit. No interpolation or extrapolation.
Also report a single scalar output chosen by best D within the active budget,
including its independent judges, and any scalar output that dominates active
on bytes and all three qualities. Only compare in the attained scalar D/byte
ranges; do not count uncovered cells as wins.

Advance this fixed policy to a separately registered broader evaluation only
if every covered training cell is noninferior to best scalar at budget in D
(>= -0.05), SSIM2 (>= -0.1), and -BA (>= -0.005), every content class has positive
median D gain, and at least half the cells improve D by >= 0.05. These are
screening bars, not new release gates. Predefine floating comparison slack
1e-5 for D, 1e-6 for independent metrics, zero bytes. Report all counts regardless
of verdict. If it fails, preserve the failure and revise the hypothesis before
spending validation families. No opportunistic alternate arm/gain sweep.

Require complete native hash/region/scalar-state/policy coverage, exact neutral
identity, independently recomputed policy fields and rational state enumeration,
negative controls, final-source reproduction, scoped Rustfmt and local CI-exact
Clippy. Preserve any prototypes. Count all full encodes, JXL/PNG decodes, scalar
comparisons, maps, independent judges, preparation/IO time and RSS separately.
This screen does not establish general target attainment or shippability. The
full goal still requires train-calibrated 1/2/3-shot targeting and matched-RD
validation on separate families, model qualification and all four codecs.


## Engineering multigroup coverage, registered 2026-09-08T20:47:36.951101+00:00

The fixed 256 policy failed the independent-judge screen; that verdict is
unchanged and no separate validation family will be spent on it. JXL's repo
requires a multigroup roundtrip for changed encoding paths. Run the same final
policy/control instrument on the existing 512 variants of the SAME four
training origins from the previous multigroup manifest. This is software
coverage and a descriptive scale check, not promotion or a larger independent
validation set. Preserve all outcomes, direct scalar-state comparisons, native
PNG/JXL hashes, exact neutral controls and independent judges. Keep results
separate; they cannot overturn the registered 256 screening failure.
