# Frozen Rev3 public human assessment — September 14, 2026

Registered before this batch's human test reads, following the user's explicit
clarification in DATA_SPLITS. Assess the same nine frozen MT913 candidates and
matched B/D controls. CID22's 201-reference oracle training population is
distinct from its 49-reference human test population. No model, seed, feature
set, output spline, head, threshold, metric or acceptance gate changes.

Admit CID22 gold (4,292 pairs /49 references), AIC-3 CTC (600/10) and AIC-4
public sample (300/5). Preserve their public-test role; no separate EVAL split
exists for these panels. Do not access secret holdouts. AIC-4 includes the
SDR25 subset; do not count that subset as independent evidence.

Reuse existing source-pair loaders/manifests and the original human targets.
Verify counts, reference identity, source file hashes and all frozen model
hashes before scoring. Extract fresh Rev3 full944 and matched Rev1 features
through the existing Rust extractor. Use the explicitly labelled legacy RGB8
SDR input contract for comparability with the six existing SDR panels; this
does not establish native-depth, ICC or HDR qualification. Keep the independent
same-buffer fast-ssim2 peer. Never reuse numerically incompatible old features.

Use BakeScorer/bake_verdict and the existing panel/scatter owners for all
compositions, human rank/Mohammadi panels, outliers, tails, clumping, saturation
and the unchanged six-population composite. Preserve previous EVAL predictions.
Publish every candidate, baseline and failure; do not select a winner by this
batch or adapt a recipe to its results. Record public-test exposure permanently.
Future iterations must disclose this exposure; secret holdouts remain the
independent final assessment. Full product qualification still requires the
separate dial, spatial, targeting, corruption and runtime measurements.

Artifacts: `~/work/zensim-validation-2026-09-14/rev3-public-human-eval/`.

## Results

All 5,192 newly admitted human pairs were extracted successfully in both
revisions. All eleven candidates/baselines now have nine populated panels and
29,969 panel memberships each. The six prior EVAL panels and every corresponding
prediction are unchanged. The three codec populations overlap; memberships are
not independent observations. SSIMULACRA2 has six matched human panels, without
misleading teacher-self panels on its own metric-labelled populations.

The canonical composite is now calculated with **6/6 coverage**. All nine
recent candidates fail the necessary CID22-band ranking requirement against
matched B, in every one of the three populated bands. They also fail the
registered p5<=25 dial requirement. These are measured failures, not a claim
that all remaining qualification gates have run.

| Frozen composition | Composite | Ladder p5 | Ladder p95 | Monotonicity % | M3a |
|---|---:|---:|---:|---:|---:|
| y40/H32 | .7708 | 35.5 | 84.1 | 99.20 | .9582 |
| y40/H128 | .7346 | 35.5 | 82.5 | 99.01 | .9673 |
| y60/H32 | .7576 | 35.4 | 84.8 | 99.42 | .9110 |
| y60/H128 | .7496 | 34.8 | 85.9 | 99.23 | .9156 |
| local120/H128 | .6997 | 31.8 | 88.2 | 98.75 | .9425 |
| selected619/H128 | .5031 | 27.2 | 94.8 | 94.22 | .7190 |
| full944/H128 | .5490 | 28.0 | 94.7 | 94.72 | .6477 |
| full944/H256 | .5574 | 26.5 | 96.4 | 94.70 | .6434 |
| linear60 | .7765 | 39.7 | 87.4 | 99.31 | .9456 |
| Matched B | .8382 | 8.5 | 93.0 | 97.76 | .6205 |
| Matched D | .8308 | -17.8 | 93.9 | 99.31 | .9641 |

These are observations of frozen models, not a new selection round. Improved
map consistency alone does not establish a useful target-score model. This
batch does not establish the ceiling of any feature set or justify adapting
models to these public-test failures.

## Ladder and spatial measurement scope

All 9,593 distinct ladder cells complete in each extraction revision. Their
keys preserve JPEG, WebP, AVIF-rav1e, AVIF-SVT and JXL, including the original
13 truncated JXL ladder exclusions. Those historical exclusions remain an
explicit coverage limitation. The instrument uses preserved decoded PNGs;
it does not qualify current native codec decoding or perform target search.
Fresh same-buffer SSIMULACRA2 supplies the reference bars through the existing
Rust owner. No candidate supplies a bar. Rev1 feature/key columns reproduce
the original registered grid; a new metadata-bound grid records the fresh
reference assessment without replacing the historical reference bars.

Every complete composition also runs all 27 registered JPEG repair cells:
297/297 successful instrument calls, with bin8 attribution maps and block32
repairs. Full944 models retain 144 unsupported refinement feature IDs; B
retains 23. Their reported M3a values are partial-map diagnostics, not complete
steering support. The narrower models and D have no unsupported refinement IDs
on these fixtures. Selected619's successful 27-cell sweep does not erase its
previous failures on other images.

M3 is the historical sensitivity fold; M3a is the current attribution map.
The first private qualification assembly incorrectly treated the historical
M3 threshold as a failure of the current map. That assembly is preserved as
`assessed/`; the published `assessed-current/` removes that unsupported
inference. No numerical result changes. Native integration controls are still
required before G-STEER can pass; unsupported refinement remains a real blocker.

## Qualification and remaining work

The existing `freeze_check --qualify` owner now reports explicit failures and
missing checks for every candidate. Content-bound artifacts prove the necessary
CID22-band and dial failures; ladder floor states come from the existing G-ADDR
owner. Full qualification is not inferred from composite or M3a.

Remaining evidence includes native bounded 1/2/3-shot targeting, native spatial
RD and intervention controls, corruption specificity, complete identity/tail
contracts, and controlled latency/memory qualification. This comparison remains
legacy RGB8 SDR. It does not repair earlier ICC limitations or qualify native
HBD/HDR. Some provenance declarations remain incomplete, particularly historical
B/D training-era metadata. Actual artifact hashes and matching arithmetic are
retained; incomplete provenance does not become a pass.

The latest user clarification is recorded in DATA_SPLITS. These public TEST
reads are frozen assessments where no EVAL split exists; they are not TRAIN.
The exact models, exposure, original human targets, all failures and producing
commands are preserved. No fit, calibration, feature search or checkpoint
selection used these results. Secret holdouts remain untouched.
