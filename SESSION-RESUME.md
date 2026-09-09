# Start here — one target score, one development path

Reviewed September 8, 2026. [CLAUDE.md](CLAUDE.md) contains current working rules;
[WAVE_PLAYBOOK](docs/WAVE_PLAYBOOK.md) contains the tool map and experiment cycle.
Older operational notes are in [docs/history](docs/history/).

Latest continuation [implements candidate L8 attribution](benchmarks/l8_attribution_2026-09-08.md).
All 36 terms pass canonical reconstruction across ten SIMD configurations.
All 792 pixel/cache/spatial audits preserve exact features/scores; D maps stay
exact and F/blend maps gain L8. Max coverage remains incomplete (36 IDs on
241 nonidentity rows, 35 on eleven, per F/blend arm). Tests/Clippy/API/lint
pass; initial fixture, dispatch-isolation and replay-count corrections are
retained. A max-removal reference matches brute force on 7,365 rectangles:
row/column prefix/suffix maxima give exact signal-rectangle effects in O(1)
with O(width+height) storage. Next implement this non-additive rectangle
estimate in the existing Rust owner, validate source/coarse/reflected/padded
geometry and actual pixel interventions, then qualify native JXL spatial RD.
Do not treat L8 coverage as complete maps or product qualification.

Latest continuation [fits and Rust-serves a cheaper complete model](benchmarks/cheap_model_blend_2026-09-08.md):
equal F_nonneg32 seeds 4004/4005/4006 at 0.11333333333333333 each, plus D at 0.66.
All 1,647 FIT and 824 CAL training consensus pairs resolve; identities/upper bound
and 792 pixel/cache/spatial audits pass. Cost screen passes at 23.77 ms (1024²) and
84.38 ms (2048²), essentially D cost. The first 4 MP run has only 29 clean paired
samples and stays rejected; its preregistered 60-round repeat has 55 and passes.
No p95/HDR/cache/map/RSS or model qualification. All 252 nonidentity maps still
omit 72 peak terms. Next implement/check L8 and max contributions in the
existing Rust attribution owner, with ties/finite-removal semantics explicit,
then qualify native spatial RD. Keep historical recipe limitations, corruption
honest-protection failures and the full release contract intact.

The preceding continuation [reduces full-pool activity blur cost](benchmarks/padded_activity_rows_2026-09-08.md).
Separate existing producer profiles locate about 63% of full-pool incremental
sampled cycles in horizontal activity box blur. A second padded-row kernel
preserves all 1,320 pixel and 792 spatial audits exactly and passes 482 tests
(six ignored), Clippy, formatting and script lint. Complete blend observed
means improve 42.10→35.33 ms at 1024² and 141.81→133.88 ms at 2048²; both
registered gains and the 576/1152 non-trigger controls pass. The blend still
costs 1.49/1.53 times D; no quiet/p95/HDR/memory qualification. Next assess
existing cheaper training recipes and data coverage before another kernel
repair or D-regime candidate. Missing pooled attribution, corruption honest
protection and useful native spatial targeting remain required.

The preceding continuation [reduces horizontal SSIM row cost](benchmarks/padded_ssim_rows_2026-09-08.md)
without changing any feature or score bit. Native profiling locates 69% of the
mixed benchmark's sampled cycles in that kernel. Padded sixteen-row storage
reduces observed blend means 88.40→42.20 ms at 1024² and 194.94→141.44 ms at
2048²; D becomes 25.34/89.77 ms. All 1,320 pixel audits and 792 spatial audits
are byte-identical, including the blend's existing unsupported map terms.
Golden/fold/SIMD/allocation/serving tests pass. Non-trigger widths stay within
the registered 5% normalized regression tolerance; maximum observed increase
is 4.50%. Formal quiet/p95/HDR/memory qualification remains incomplete, and the
blend still exceeds 1.25 times current D. Next profile remaining full-pool
cost before choosing further compute repair or a cheaper competitive model;
missing pooled attribution, corruption protection and codec RD remain required.
Post-change native profile: fused horizontal SSIM 31%, fused vertical SSIM 15%,
single-plane horizontal blur 12% of the mixed workload. Inspect those owners
next, separating common work from A's pool increment. The initial noisy 576²
control was replaced by a passing preregistered batched control; no threshold
was relaxed and no latency percentile was inferred from batching.

The preceding continuation [measures actual A/D inference cost](benchmarks/model_blend_speed_2026-09-08.md)
and fixes zenbench waiting on its own Linux lock-heartbeat thread. Root bench
now pins pushed zenbench `1bf8a6509fce`. Corrected 40-round, single-worker means:
blend 88.05/194.45 ms at 1024²/2048²; D 76.41/136.04 ms; SSIM2 67.73/289.15 ms.
Blend and A alone cost essentially the same. Observed extrema miss the 1 MP
absolute/SSIM2 and 4 MP relative-to-D bars. Strict quiet admission still fails
(one advisory/background activity), so these are diagnostic results, not a
performance qualification. Next profile shared extraction and the full-pool
increment before choosing a compute repair or cheaper competitive regime.
Do not invest in all missing A spatial integrands before resolving cost.
No model/inference arithmetic changed; full product goal remains incomplete.

The preceding continuation fits and Rust-serves the registered A/D blend: exact
weights `[0.5900000000000001, 0.41]`. It resolves all 1,647 fit and 824 separate
training-calibration consensus pairs, preserves twelve exact identities, and
matches independent arithmetic exactly. The actual attribution check then
finds 169 active unsupported feature IDs across f156..371; all 252 nonidentity
blend maps are incomplete. D-only maps have complete coverage. Do not advance
this candidate into native steering with incomplete maps. Next price the
actual blend through `ssim2_speed_bar` against D/SSIM2 before committing to
full-pool attribution work: the shared extraction must serve A's expensive
pools too. If cost passes, implement/check those integrands in Rust; otherwise
repair cost or train a competitive cheaper model. Preserve scalar arithmetic, explicit coverage,
max/clamp/gate caveats and the existing performance/release bars.
[Blend recipe, results and spatial gap](benchmarks/model_blend_2026-09-08.md).
The existing pixel-audit owner now supports explicit ensembles and weights;
legacy single scores and both endpoint controls reproduce. No new public API.

The preceding continuation compares nine existing 372-class base models on 264
canonical training pairs, with full Rust pixel/cache scoring and independent
judges. D has zero disagreements among 2,471 resolved same-source pairs;
generation A and B each have three. The newer A_plain seeds have 170–202,
and H_anchorlad seeds 48–69, including substantial near-lossless errors.
All models pass twelve exact identities and never exceed 100. None advances
under the registered strict-improvement rule; a zero-error D baseline makes
that rule unable to resolve equal-perfect candidates. This separates strong
bulk scalar ordering from the already failed D spatial policy.
[Results and next registered A/D blend](benchmarks/model_preferences_2026-09-08.md).
That registered fit and Rust audit now completed as recorded above. Preserve
all release gates and the head's two honest false positives; the base blend
does not repair those automatically.

The preceding continuation repairs the native JXL delivered-pixel contract. Controlled
dither on/off decoding explains nearly all of the earlier drift: 8,646,810
samples change by at most one code; only 178 samples still differ between the
undithered canonical and historical decoders. The native loop now uses the
canonical zenjxl-decoder U8 path. All 756 newly encoded training ladder outputs
match independent decoded hashes and f32-reported D scores exactly; the 336-row
historical diagnostic reproduces exactly. Old decoder-era calibration is
refused before encoding. This closes the measured decoder discrepancy, not
model accuracy or spatial qualification. Continue with honest protection and
base-model preferences; do not rerun the already failed fixed D allocation
policy as though decoder parity qualified it.
[Decoder repair and evidence](../jxl-encoder/benchmarks/zensim_decode_contract_2026-09-08.md).

The preceding continuation adds 608 retained honest JXL/AVIF map-arm attempts on the
eight admitted fit origins (402 additional unique pixel pairs), then fits the
unchanged v2 cost-4 recipe once. Calibration false positives fall from three
to two, still FAIL. No validation/terminal scoring. More importantly, the
native JXL target loop's decoded pixels differ from canonical extraction on
all 336 added JXL attempts: median absolute D drift 0.185, p95 1.807, maximum
2.425 points. AVIF pixels agree exactly. Source inspection identifies a likely
cause: native JXL uses upstream jxl f32 output plus plain rounding; canonical
zenjxl-decoder U8 defaults to blue-noise dithering. The later controlled test
and repair above supersede that hypothesis. Preserve the old results as their
named decoder era. The head still fails honest calibration protection.
[Honest coverage and decoder evidence](benchmarks/corruption_honest_map_2026-09-08.md).

Latest September 8 continuation establishes the missing numerical release
contract at the top of [MODEL_SELECTION_SCORECARD](docs/MODEL_SELECTION_SCORECARD.md).
It repairs the head's precision boundary with explicit ZCTH v2 f32 inputs;
v1 behavior and all fitted numeric sections remain unchanged. The cost-4
training-only replay now has exact raw/probability/composed-score equality
on all 9,036 pixel comparisons and exact sklearn/Rust parity on 8,213 unique
rows. Accuracy still fails: three honest calibration JXL outputs are lowered.
No validation was evaluated and no model qualifies. See
[precision record](benchmarks/corruption_input_precision_2026-09-08.md).
Next corruption work is broader admitted honest coverage, especially legitimate
near-lossless JXL across independent source families; repeating cost fits or
retuning validation thresholds cannot close that gap.

Latest user direction: **keep the corruption head in D's existing feature
regime**. A new f0..227 HGB head is fully Rust-servable without expanding image
extraction. All three seeds detect every tested non-inert RGB swap and 99.57%
of validation corruptions, but lower 20/304 honest native codec outputs. It
fails the fixed development bars; no model qualifies or replaces a default.
Three bounded honest-cost follow-ups also fail on training calibration or
serving precision and never advance to validation. [Exact results, artifacts
and remaining precision issue](benchmarks/canonical_corruption_d228_2026-09-08.md).
The earlier full content-admission audit is complete; its all-372 head and
serving refusal remain preserved. [Admission record](benchmarks/canonical_corruption_refit_2026-09-08.md).

Remaining work, in order:

1. Improve the D-regime corruption head's honest-output protection with broader
   admitted honest codec/chroma/tone examples and source coverage. Resolve the
   tree's training/stored-f32 versus pixel-feature precision boundary (now
   resolved for newly exported v2 heads; preserve v1 artifacts). Preserve
   fixed safety bars and full Rust evaluation. The completed cost sweep chose
   no arm; do not tune its failed arms on validation or expand D's extraction.
2. Repair base-model identity, near-lossless and codec-floor preferences using
   the strongest existing candidates and native-codec evidence. A corruption
   gate or monotone calibration cannot repair the base model's rank inversions.
3. Prove JXL spatial benefit against a strong ordinary scalar controller at
   matched quality/bytes, with the complete frozen model and independent judges.
   Separate model-preference failures from allocation-rule failures; retain the
   already failed fixed-D policy rather than retuning it on observed outcomes.
4. Finish train-family calibrated, separate-validation 1/2/3-shot scalar/spatial
   targeting for JXL, then AVIF, JPEG and WebP. Measure attained ranges before
   judging steering, keep bounds out of runtime, and count every encode/map.
5. Freeze and qualify the full Rust artifact: perceptual/dial/tail behavior,
   supported SDR/HDR/color/alpha, latency/memory, spatial RD and terminal tests;
   deliver the reproducible recipe, examples, manifests and gauntlet report.

The immediate HDR task was a protected-reference overlap check, not a new HDR
training pipeline. HDR development PNGs already exist. That admission dependency
is now closed; the next work belongs to model serving, accuracy and JXL steering.

Latest user direction: EXR belongs in **`zenextras/zenexr`, wrapping the Rust
`exr` crate**. This explicitly authorizes that dependency and supersedes the
earlier pending exception question. The custom zenbitmaps port is retired from
the active checkout; its verified source checkpoint and results remain archived.
The replacement reproduces all 98 saved fixtures and all 30 UPIQ HDR references
bit-for-bit (124,609,944 f32 samples). [Wrapper contract and validation](../zenextras/benchmarks/zenexr_validation_2026-09-08.md).
The later fingerprinting/contextual review is complete as recorded above;
decoder parity alone does not qualify a model or authorize holdout training.

The preceding user correction still applies: HDR development uses the existing imazen-26
`variant/png-v3` inputs: **76 HDR PNGs**, all locally present and byte-verified
against branch LFS payload OIDs, with **38 train / 20 validate / 18 terminal**
origins under the family manifest. The active 1,140-image HDR scale set is also
local. [Source binding and chronology](docs/TARGET_STEERING_PROTOCOL_2026-09-08.md#existing-hdr-png-inputs--user-correction-september-8).
The separate UPIQ EXRs are holdout-overlap references, not a missing HDR
training corpus. The earlier port checkpoint is preserved at
`/mnt/v/output/zensim/native-exr-port-2026-09-08/`; active EXR implementation
now uses zenexr. One new corruption head has been fit; no model is qualified.

Previous September 8 continuation: [actual coarse JXL allocation policy](benchmarks/zensim_coarse_allocation_2026-09-08.md)
**fails its preregistered independent-judge screen**. All eight 256 cells improve
D versus every locally reachable globally rescaled raw field, but photo 2010/d3
loses 0.021502 Butteraugli and screen 8206/d1 loses 0.143189 SSIM2. The separately
registered 512 software check reveals further losses. This fixed D+policy does
not advance; no validation or terminal families were spent, and no model is
qualified. The comparator exhausts a declared local integer-field domain, not
ordinary global-scale/distance control or the full codec optimum. Policy inputs
exclude measured bounds/judges. All 249 primary encodes and 498 judge values
reproduce exactly; 36 analyzer / 11 CLI refusals and local CI-closure checks pass.
Artifacts: `/mnt/v/output/zensim/jxl-coarse-allocation-2026-09-08/`, mirrored to
`~/work/zensim-validation-2026-09-08/jxl-coarse-allocation/` with an HTML report
and original/active/failed-judge scalar comparator gallery. Next separate model
preferences from allocation-rule effects using supported existing candidate maps
under a registered fixed rule, and strengthen the ordinary scalar comparator
before any product RD claim. Do not simply repeat this failed D policy or sweep
its gain against these outcomes. The EXR reader question was subsequently
resolved by the user direction above; fingerprinting/contextual admission
remain unfinished.

Previous September 8 continuation: [coarse JXL intervention screen](benchmarks/zensim_coarse_interventions_2026-09-08.md)
finishes native PNG IO and whole-transform 4×4 grouping in the existing codec
instrument. Coarse / ±20% gives expected D direction in 244/256 probes at 256
and 245/256 at 512; independent judges mostly agree. Map mass predicts response
much better, but density versus gain per byte remains weak/negative in some
cells. Area and amplitude both changed, so their separate effects are not
identified. **No matched-RD win or model qualification.** All 272 historical
transform outputs are byte/pixel/score/map-exact under native IO, and 816 final
probes plus 1,632 judge values reproduce. Thirty-one analyzer and eleven CLI
refusals pass; final/local CI-closure checks pass. All evidence is retained at
`/mnt/v/output/zensim/jxl-coarse-interventions-2026-09-08/`.
Next spatial work: preregister an actual coarse allocation policy in the same
owner and test matched quality/bytes before separate-family 1/2/3-shot targeting.
Do not repeat the completed intervention screens or treat oracle-probe cost as
runtime cost. The earlier corruption-refit EXR question is resolved above.

Previous September 8 continuation: [canonical refit preparation](benchmarks/canonical_corruption_refit_2026-09-08.md)
adds native content admission and a trainer mode that exports and evaluates the
same single fit. All 12 training origins have zero strict flags against 182 SDR
holdout reference entries; four looser matches were reviewed as distinct content.
The source/pixel-deduplicated fit / calibration / evaluation views contain
5,504 / 2,709 / 5,679 rows and pass explicit full-key Parquet checks. An invented
800-row numeric fixture has exact weighted-HGB export parity in Rust. **No
canonical image-data fit yet:** 30 UPIQ HDR EXR references remain unaudited. The
reader question at that stage was pending. The user later explicitly selected
`zenextras/zenexr` over the Rust `exr` crate. Use that owner to finish the
remaining reference admission; do not repeat the completed SDR audit.

Previous September 8 continuation: [canonical corruption serving screen](benchmarks/canonical_corruption_serving_2026-09-08.md)
is complete on all 15,060 canonical/native rows, with exact pixel, cached and
stored-f32 feature/score parity. New `BakeScorer::score_features_with_identity`
carries proven pixel identity; raw zero features remain insufficient evidence.
Frozen D+HGB detects 93.82% of unique validation corruptions and orders 98.64%
below q20, but incorrectly sends eight near-lossless JXL outputs to zero and
detects only 64.35% of the newer real-bug family. **No new fit or qualification.**
Before fitting, finish T0 content admission, construct source/pixel-deduplicated
views (raw catalogs still fail C10), and register exact family-level fit /
calibration / evaluation identities. Extend the existing corruption trainer:
its legacy CV-ensemble report describes a different model than its exported
single fit. Evaluate the exact exported artifact through Rust. Corruption-gate
discontinuity and native adapters' rejection of companions remain product work.
Prior native intervention and RD findings remain negative/mixed; corruption
detection or restoration coherence does not prove useful spatial allocation.

The end user controls one target score. The codec chooses parameters and must
reach useful quality across codecs/content, near-lossless settings and codec
floors with few passes, small output, low latency and low memory. Negative
scores are valid. A rank correlation or research composite cannot establish
that behavior. B remains `codec_target()`; D is the explicit fast profile.
B/C/D may be replaced: there are no consumers calibrated to their current
scores. [The integration guide](docs/CODEC_TARGET_METRIC.md) owns the mapping.

Every new model must execute and serve entirely in Rust through a zensim API,
and evaluation must call that API, including heads, corruption and splines.
`BakeScorer` now provides that dynamic surface. Python remains useful for
invention and independent numerical references.

| Question | Current answer / evidence |
|---|---|
| Can we train after cleanup? | Three A_plain seeds exactly reproduce all measured verdict and ladder fields. Three H_anchorlad seeds also reproduce every earlier verdict and ladder field; three paired rav1e additions complete the floor-data control. [Reproduction](benchmarks/cleanup_training_reproduction_2026-09-07.md); [paired controls](benchmarks/cleanup_scientific_controls_2026-09-07.md). These historical recipes have documented split limits. |
| How does target search behave? | [2,730 witnessed-target cells](benchmarks/target_steering_bounds_2026-09-08.md) compare train-calibrated 1/2/3-shot policies, with exact final-code reproduction. The [360 older loops](benchmarks/cleanup_target_loop_2026-09-07.md) mixed feasible/unwitnessed targets; their median cannot establish model failure. [Native JXL targeting](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_native_targeting_2026-09-08.md) now measures three arms with shared Rust calibration/search: 810 cases, 45/80 jointly witnessed targets, all arms hit ±1 in three shots on that subset. Coverage and independent RD still prohibit qualification. |
| Does native attribution already exist? | Yes: JXL fused/binned/stale H3, a recovered JPEG `jj` experiment, WebP segments and the later AVIF CQ/H3 experiment. [Fleet/source/chronology audit](benchmarks/diffmap_reuse_audit_2026-09-08.md): reuse them. The binding now exists: `BakeScorer::compute_with_ref_and_attribution` uses complete scoring and declared extraction, with explicit spatial coverage. JXL now consumes it in [commit 9f038d4d](https://github.com/imazen/jxl-encoder/commit/9f038d4dd7f4); 840 ladders and decoded map engagement pass, but D/H3 has no broad RD win. [Native record](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md). [Serving/coherence evidence](benchmarks/candidate_attribution_serving_2026-09-08.md): exact pixel parity, 15 serving tests and 54 coherence cells. D is the first native baseline; native engagement and product qualification remain incomplete. |
| Is good rank enough? | No. A_plain retains identity and all-codec floor failures. Qualification reports failure/incomplete evidence separately from research selection. No new common-dial winner is established. |
| Can candidates serve their whole model? | Yes, through `BakeScorer`: declared IDs, validated metadata, heads, pin, spline, codec affine, ensemble and corruption composition. [Execution plan](docs/PLAN_CRUFT_PURGE_2026-09-06.md). Pixel luminance revision must still match the process. |
| What happened to C/CHdr? | Both use explicit IDs and canonical activity semantics. The old training/serving mismatch is fixed; two alternate planners are retired. Pixel scores intentionally change. [984-case feature census and HDR/matrix evidence](benchmarks/feature_plan_cleanup_2026-09-07.md). |
| Where do I train and evaluate? | `zensim_mlp_train` → `bake_dial_refit` → `run_full_eval.sh` → `freeze_check`. Admission now checks actual headers, IDs and declarations before work. [Trainer cleanup](benchmarks/trainer_admission_cleanup_2026-09-07.md). |
| What owns calibration? | Rust `bake_dial_refit`, with pre-calibration coordinates and final full-surface evaluation. Four historical writers are retired with recipe/boundary evidence. [Spline record](benchmarks/spline_owner_cleanup_2026-09-07.md). |
| Which data are valid? | [DATA_SPLITS](docs/DATA_SPLITS.md), later [DATASET_HISTORY](docs/DATASET_HISTORY.md) entries, actual manifests and the [shared index](../DATA_PROVENANCE.md). Width is not an era. TID is train-only; historical KonJND/KADID overlaps are guards, not holdouts. |
| Where is the board? | `scripts/v_next/gauntlet.py` renders stored owner verdicts to `/mnt/v/output/zensim/reports/summer_gauntlet{_fair}.html`. Codec-q score charts include negative scores; product qualification appears before composite. |
| Why keep buffered/full features? | Remaining callers and numerical differences prohibit blanket retirement. Free/shared features are not removed on coefficient counts. The contended extraction benchmark was excluded, not presented as a speedup. |

The authorized cleanup is complete: [execution and validation record](benchmarks/cleanup_completion_2026-09-07.md); [original checklist](docs/PLAN_CRUFT_PURGE_2026-09-06.md).
The [transcript/memory audit](benchmarks/science_workflow_audit_2026-09-07.md)
records chronology and bounded confidence in Rust/Python equivalence. Scientific
controls and actual-loop measurements have their own result records; a passing
code migration cannot inherit an unmeasured product gate.

Use the existing capped `run-heavy` owner for heavy work and
`scripts/safe_push.sh` for every push. Data/bake evidence is preserved under
`/mnt/v/output/zensim/cleanup-*-2026-09-07/`; private audit and retired sources
are under `~/tmp/zensim-science-audit-2026-09-07/`. Choose fresh output paths
for reproductions. Do not overwrite historical results or infer completion
from a filename without its content identity.

September 8 repository sync + AVIF binding continuation: all 75 immediate zen
repos plus five external dependency/corpus repos fetched; 52 checkouts advanced
or rebased, local changes rescued and five conflict sets resolved. Exact private
recovery/status: `~/work/zensim-recovery-2026-09-08/sync-all/README.md`. Keep
syncing before new work and use tight local checks; do not wait on CI.

AVIF complete-candidate research binding pushed as `4777a30aa54aaab6534719f6026e4a18e91d60e3`. Exact-bake
`BakeScorer` owns scalar and current spatial scores; first map feeds encode 2,
unsupported terms fail, terminal bytes are decoded/scored independently. On the
one training-family control, legacy gain 10 is inert (all blocks saturate then
normalize to 1); existing zerosum engages and repeats byte/pixel/map-exactly.
Neutral 79.368/18,355 B; active 80.287/19,573 B; three full encodes, three maps,
four scalar comparisons including terminal. Six CLI rejection controls pass.
See sibling `benchmarks/zensim_avif_loop_2026-08-07.md` September 8 addendum and
`zensim_avif_candidate_binding_2026-09-08.json`. Artifacts at
`/mnt/v/output/zensim/avif-candidate-binding-2026-09-08/`; Windows-readable copy
under `~/work/zensim-validation-2026-09-08/avif-candidate-binding/`.
This is engagement only, not RD or competitive-model qualification. Next: AVIF
attained bounds and train-only calibrated actual 1/2/3-encode loops through the
shared Rust search, then matched-quality judging. No new model training or
terminal holdout was performed in this continuation.

Shared native probe owner added after the AVIF binding: unpublished tool feature
`zensim-target/native-probe`, with codec-owned adapter for AVIF. See
`benchmarks/native_probe_instrument_2026-09-08.md`: 51-bound/119-encode train and
validation pilots, 54 target cases, source/model/driver/family rejection checks
and independent decoded-pixel integrity checks pass. AVIF's stateful map uses
the previous complete decode; its first shot has no consumed map. Full 12/8
family matrix is next, then model qualification and native JPEG/WebP work.

Later September 8: the full AVIF 12-train/8-validation matrix and independent
judges are COMPLETE. [Instrument record](benchmarks/native_probe_instrument_2026-09-08.md)
links the codec report: 504 cases, 28/80 joint witnesses; three-shot calibrated
scalar/neutral 28/28 hits ±1 versus active 24/28. Fixed-CQ active maps modestly
help SSIMULACRA2 but worsen Butteraugli in every content class; no spatial
benefit or model qualification. Artifact `RESULT_COMPLETE.json` supersedes
the launch-time pending text in `SOURCE_PINNED.json`. Judge-pair identity,
duplicate/missing controls pass, with JXL/AVIF numerical results preserved.
The September 7 H+rav1e floor-data control already completed and failed all five
floor gates; do not repeat the older September 6 proposal as new work. Continue
with useful native allocation/model qualification and JPEG/WebP integration.

Later September 8 JPEG continuation: complete-candidate binding now runs in the
existing Zq loop (`__zensim-research`, recovered `zq_rd_probe`). The first trace
exposed old AQ defects: unit scales clamped real strengths to 0.20 and the last
strip skipped the controller. Both are repaired, along with a fractional-q
clamp panic near 100. [JPEG record](https://github.com/imazen/zenjpeg/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md):
one train family, exact D, 444/q80; scalar and neutral 77.438789 / 16,864 B,
active repeat 77.314751 / 16,795 B. All 832 blocks controlled; current maps,
JPEGs and pixels repeat exactly. SSIMULACRA2 and Butteraugli both show the small
quality loss, so no RD benefit is claimed. Eighteen Zq tests pass before/after,
22 AQ tests, two Clippy routes, API snapshot check and ten rejection controls
pass. Artifacts: `/mnt/v/output/zensim/jpeg-candidate-binding-2026-09-08/final/`;
Windows bundle under `~/work/zensim-validation-2026-09-08/jpeg-candidate-binding/`.
This is fixed-seed engagement, not calibrated targeting: two corrections cost
three full encodes, and its existing `targets_met` flag checks only the score
floor. Reuse the shared Rust native owner for actual JPEG 1/2/3-shot bounds and
train calibration; WebP binding and competitive model/spatial qualification
remain incomplete. No training or terminal holdout was performed in this step.


Later September 8 WebP continuation: [complete-candidate binding and accounting
repairs](https://github.com/imazen/zenwebp/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md)
are pushed in `8aa8a7858b97`. The existing segment loop now accepts the exact Rust
bake through `__zensim-research`; registry 0.2 and the separate recompress A
calibration remain intact. One-pass targeting is measured and strict, encode
counts are actual, ship-band flags are truthful, and finite negatives are valid.
All five old failures were observed in regression tests before repair; hard-floor
selection has its own observed failing control. Local 354 library, 15 targeting,
33 validation tests, two Clippy routes, release/API/format checks pass.

One canonical training family, D/q80/m4: scalar and neutral select 73.673485 /
16,040 B; active selects 72.955315 / 15,728 B. The fixed-q neutral correction is
byte/pixel/map exact; active quantizer 11→14 is consumed and repeats exactly.
All 208 macroblocks, including the partial edge, have explicit map/segment traces.
Both independent judges see quality loss with the smaller active file. Thirteen
full encodes, nine maps, two non-neutral map uses across active+repeat and five
terminal comparisons are counted; ten CLI rejection controls pass. Full result:
`/mnt/v/output/zensim/webp-candidate-binding-2026-09-08/`; Windows copy under
`~/work/zensim-validation-2026-09-08/webp-candidate-binding/`.

All four codec owners now have an exact complete-candidate binding and native
engagement evidence. Useful spatial RD, a competitive qualified model, JPEG/WebP
train-calibrated witnessed bounds and actual 1/2/3-shot validation remain required.
Do not repeat binding screens as model qualification. No training or terminal
holdout was performed in this continuation.

Later September 8 native JXL finite-block continuation:
[intervention record](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_native_interventions_2026-09-08.md).
Codec instrument/report pushed as `feab1d7fd734c89212b449a1945cf1e6e7bf81e4`.
Four train families, distances 1/3, 16 native transform regions with ±10%
quantizer changes: 272 complete probes reproduce bytes/pixels/scores/maps and
both independent judges exactly. Separate 272-probe 512-long-edge multi-group
coverage passes; 48 total libjxl 0.12 compatibility checks complete. The existing
RD analyzer's new `--interventions` mode passes 19 rejection controls, preserves
older analysis functions, and verifies all source/byte/pixel/quantizer/judge
identities. Six CLI controls and local release/Clippy/format/script checks pass.

Restoration coherence does not establish useful native marginal allocation:
primary map-mass/D-response rank associations range from −0.259 to 0.676.
222/256 interventions affect pixels outside their selected region even though
captured quantizer changes stay local. Seven interventions preserve the final
quantizer field but change pixels; native thresholds also depend on the incoming
field. CfL is frozen here. Do not treat final quantizer identity as a no-op or
these response secants as isolated quantizer gradients. No model/policy change
or qualification. Artifacts: `/mnt/v/output/zensim/jxl-native-interventions-2026-09-08/`;
Windows copy under `~/work/zensim-validation-2026-09-08/jxl-native-interventions/`.
Next spatial hypothesis: preregister a coarse-region/rate-cost intervention
before tuning allocation. Model qualification and JPEG/WebP actual targeting
remain open; no new training or terminal holdout in this packet.
