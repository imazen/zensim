# E36 draft — native TRAIN ladders for core ordering

Revision 5 budget amendment, before any E36 fit. Preparation permits native data generation and
non-training checks only. No optimizer step, first-epoch smoke, fit-cell claim or
model qualification is authorized. The coordinator releases fitting separately.

The existing 97 inverted edges survived same-final-epoch f32 and output-curve
controls. E36 changes the core ordering loss. Those inspected G-ADDR panels
remain exploratory; untouched confirmation belongs to the later ship study.

## Fixed model, arms and objective

Primary family C is diff+frag-820, N head, H128, Rev5, exact E34 base recipe
`sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1__N`: 120 epochs, 50,000 rank-pair
draws/epoch, final epoch119. Preserve group targets/weights, feature/fx1
identities, shared initialization/sample streams and quantize-before-TRAIN-
calibration. Apply any required TRAIN-selection family exclusions identically
to treatment and fresh control. No diff-410 arm is included.

Three fixed arms: fresh TV0 control, lambda1 and lambda10; raw margin0.05.
Do not search other weights/margins after generation or assessment. Registered
loss is `LE34 + lambda * sum(w * max(0,s*(better-worse)+0.05))/sum(w)`, where
s=+1 for distance and -1 for score. Pair weight2 for rav1e/JPEG,1 otherwise;
all q-band weights1. Report effective mass by codec/content/size/band.

Use the existing polarity-aware TV owner. Exact CLI reduction, independent TV
RNG seed, batch size and application frequency must be admitted before fits.
Legacy band weights replace the flat TV weight; they are not assumed to
multiply it. A flag ignored on the N-head path does not implement a treatment.

Synthetic feature rows enter only TV, never human-rank sampling, scaler fits,
target normalization, calibration or base-group weights. Base-rank RNG draws
must remain identical across matched cells. Native features use the frozen E34
consumed-ID/fx1 declaration; the1853-wide archive layout contains the1825-wide
native prefix and28 explicitly unconsumed structural zeros. No missing consumed
feature is filled. Local synthetic row ordinals are not legacy global group
indices. Key/layout/polarity/zero-dropped-pair and controlled raw-loss checks
must pass without an optimizer step before fitting.

## References and admission

Use canonical imazen-26 TRAIN under origin and family owners, supplemented by
the current family-v2 burst/crop map. Recheck current bytes, canonical split
bindings, native dHash audit and every G-ADDR/G-STEER/evaluation reference key.
Close exclusions through origin IDs, source SHA, lineage and native dHash<=10;
quarantine unresolved relevant lineage and missing current origins. Reuse E34
metadata/embeddings only with exact hash bindings. Never read protected labels.

Also exclude previously used development families and their native dHash
neighbors. TRAIN-selection families must be absent from every optimization,
model preprocessing and calibration input, including full-data pilots. Native
metadata links to existing base groups require matched removal before fitting;
reference-only sampling/clustering is separate from model preprocessing.

Reference-only collection strata: origins1000–4999 photo,5000–5999 mixed,
6000–6599/6800–7999 line art,8000–8999 screen. The6600–6799 illustrated color plates and other collections are outside the primary
matrix. No palette classes. Select>=50 distinct references per semantic class.
Tiny is an overlapping presented-size tag: both dimensions<=64, from>=50 TRAIN
source families, without upscaling; report original dimensions separately.
Do not claim>=50 originally tiny sources.

Freeze native metadata admission before final selection. The shared native
`srgb-opaque-v1` owner interprets unprofiled sources without conflicting gamma/
chromaticities as sRGB; converts named P3 through zenpixels-convert; admits ICC
only when native recognition authorizes sRGB matrix/TRC substitution. Unknown
profiles, conflicting metadata and malformed/unresolved EXIF refuse admission.
Valid EXIF absence means identity; native nonidentity orientation is baked once
through the existing orientation owner. Alpha-bearing sources are excluded:
the existing native matte conversion has an open one-channel rounding finding
and is not qualified here. No matte fix is claimed. Resize opaque native RGB
with Lanczos. Keep original metadata
and original/normalized pixel hashes. Refused sources quarantine their families;
they are not silently relabeled or repaired by third-party writers.

Keep every rendition of a family in one split. Per class, order families by
SHA256(`e36:family:<family>`), reserve ceil(0.2*N) families for selection,
increasing only until>=10 candidate sources are available. Unexpected
cross-class families refuse the independent class split. Cluster native
reference embeddings separately within each class/split: population z-score,
zero-variance scale1, Lloyd k-means, n_init4, seed36, K40 fit/K10 selection,
centroid-nearest representative with lexical origin-ID tie-break and singleton
retention. Conservatively quarantine both families of any cross-internal-split native dHash<=10 flag and repeat this reference-only selection until the screen closes; no label or teacher/model values enter this procedure. Require>=20% selected families and references in selection. Hash
normalization, cluster sizes and final source/family manifests before bulk.

## Native generation and teachers

Five native encoder paths only: zenjpeg, zenavif/zenravif/zenrav1e, the imazen
SVT port, zenwebp and native JXL. No third-party encoder or metric produces a
training variant/target. Quality grid is0..20 step1 and22..100 step2 (61).
Longest-edge sizes:32,40,48,64,80,96,128,160,192,256,320,384,512,640,768,1024,
1280,1536,2048,4096. Preserve aspect ratio, never upscale. Record reachability
and every structurally unsupported or failed cell. Keep actual q-to-setting
identities and aliases; nominal q alone does not define a physical setting.

The budgeted matrix in the prospective six-hour amendment below is current.
Dense 20-size ladders use frozen native baselines. The separate 64/256 mode panel
uses native `modes_full` plans for JPEG,rav1e,WebP,JXL and `svt_doe_main` for SVT,
with `--max-deviations 1`, without thinning its registered q axis. Native
physical aliases retain requested-setting provenance; inactive values do not
prove active coverage. Every required active knob/value needs a resolved
coverage table; conditional values need named compatible baselines/submatrices.
Missing coverage remains explicit. Nonbaseline large-size effects and systematic
multi-knob interactions are untested. Shipping-gate configurations are unchanged.
The superseded full-cross/all-source main-effects census exports are retained.

Active teachers are zenmetrics' own butteraugli pnorm3 and CVVDP standard_fhd,
with native maps from its diffmap owner. SSIMULACRA2 and served B do not generate
targets. This prospective choice follows the owner's any-metric grant and
independent Astra agreement; shipping judges remain fixed. Pin builds, scalar
precision, color ingress and CVVDP display/resampling. Bulk targets use full
native JSON f64 values from ScoreFile, with exact decoded pixel hashes. The
initial TSV-rounded pilot targets remain diagnostic-only and are not mixed
into the bulk TRAIN view. Both teachers receive
identical normalized native RGB8 reference/distorted pixels.

Orient CVVDP positively and butteraugli pnorm3 negatively. For endpoints a,b,
define delta_t=oriented_score_t(b)-oriented_score_t(a). Admit b preferred to a
only if BOTH deltas>0.01 native units; reverse only if BOTH deltas<−0.01.
Generate each unordered same-reference/codec/mode/size pair once at ladder-index
lags{1,2,4,8,16}. Exclude/count disagreements, ties and insufficient margins;
missing/nonfinite scores refuse cells. Higher q is never truth. Report each
teacher's forward/reverse margin passes, vetoes, exclusions and reversed
preferences by codec/content/size/q-band/split. Bands0–20,22–60,62–100 are
assigned by the lower nominal-q endpoint, independent of preferred direction.
Zero support in a required stratum blocks fits, without redistributing mass.

Retain encoded bytes, native decoded pixels, all exposed teacher scalars and
genuine float maps by hash. Map keys bind raw reference/distorted pixels and
metric/build/display identity. Manifests bind source/family/split, color,
encoder/decoder/metric closure, resolved settings and ordered feature keys.
Verify first-cell bytes/maps before scaling. Independent decoders inspect
outputs only; their pixels/scores never become targets. Use native zenmetrics/
zenfleet plan/execution/artifact owners, with no parallel claim/retry protocol.

## Fixed future selection and assessment

129 possible C fits: three arms x(four withheld sources x10 shared seeds +three
full-data seeds). Freeze fold/source, initialization/sample streams, all data
hashes and control bindings before release. Retries preserve seeds/configuration;
versioned implementation repairs apply consistently to affected matched cells.

Choose one lambda globally on held-out TRAIN families: maximize weighted
fraction meeting raw margin0.05, then strict-order fraction, then smaller
lambda, aggregated over registered strata and three full-data seeds. Identity and
finite guards use unchanged existing tests/thresholds. TRAIN selection has no
human ranks. Cross-reference ranking is protected at assessment by the
registered noninferiority test and its dataset and W2 guards.
If neither arm passes, select neither. No assessment dataset/fold/seed chooses
checkpoint, feature set, margin or another lambda rule.

Full-data pilots include human training sources, so their TRAIN selection is
not assumed independent of human assessment. Test BOTH fixed lambda arms,
including the unselected/failed/unavailable arm, in one Holm family of2 at
one-sided familywise alpha0.05. H0(lambda):Delta_lambda<=−0.0065. The smaller
p must pass0.025 before the larger may be tested at0.05. Never shrink the
family. The globally selected arm must reject its own adjusted null and pass
all unchanged guards; if it fails, select no qualifying model, without
substituting the other arm.

Delta is E34's equal-weight mean across KADID,TID2013,KonFiG,CID22-A25 of the
arm-minus-control signed SROCC on corresponding source-withheld rows, paired
by seed. Primary assessment uses ONLY source-withheld models/controls; full-data
pilots contribute no primary assessment predictions. Exclude each withheld
source's labels/families from optimization, calibration, preprocessing and
model-specific selection. Use the existing source-reference x shared-seed
statistical owner,10000 resamples,type7,Student and bootstrap (larger p decides),
with frozen seed offsets. Guards: mean point>=−0.002, each dataset>=−0.005,
W2 delta>−2SE. Dataset/W2/engineering checks are conjunctive eligibility
conditions, not separately established inferential claims. Report both arms,
raw/adjusted p values, finite draws, selection and every guard. The E36 two-slot
adapter/manifest must be admitted before fits; existing fixed inventories
must not be silently repurposed.

Keep every engineering gate, adopted butteraugli steering/CVVDP scoring
reference, matched ranking and complete-map cost/RSS gate. Bind endpoint,
threshold, implementation and qualification status to immutable adopted
manifests; do not reinterpret RSS limits from a review gloss. Inspected G-ADDR
and E36 pilot results remain exploratory. Untouched ladder confirmation comes
later; missing shipping evidence stays unqualified.

## Preparation boundary and review history

The coordinator requested registration/review, START data generation on dev,
measured data-cell smokes and E36_READY, then STOP before fits. All data runs use
the shared heavy flock/run-heavy and20GiB free-space floor on both /home and the
data backing filesystem. Store bulk data under the E36 root and mirror by
manifest to tower. Smoke receipts record actual time/RSS/bytes; do not present
planning products as completed cells or extrapolated timing as measurement.

The bounded primary smoke is five native producers xq{0,1,20,60,100} x longest
edges32,256,4096 from one admitted source reaching4096:75 cells. Supplemental
native full61q pilot generation may establish consensus/export mechanics; it
is not full source/size/mode coverage and cannot qualify fitting. Bulk remains
blocked on resolved conditional-knob coverage and an admitted storage/staging
budget. Feature-only TV/RNG/loss and two-slot evaluation admissions remain
pre-fit requirements. E36_READY reports these as missing, not passes.

Revision1/2 and the pre-encode bounded-smoke amendment are retained in history
(`b8ef9586`,`0e60afff`). Independent Astra reviewed the draft, explicitly agreed
to the two-teacher/tiny clarifications, conditionally accepted main effects,
and accepted the prospective two-hypothesis statistical correction. Review
receipts/source pins are in the private artifact manifest. No E36 fit occurred
before any of these amendments. The coordinator alone releases fitting; this
draft and preparation receipt do not authorize it.


## Prospective six-hour bulk amendment (2026-10-17)

The coordinator now requests budgeted bulk generation, followed by trainer
integration, before the fit go. The generation wall budget is about six hours
on dev; i265 is permitted only after a live idle admission. Other workers
remain assigned elsewhere. No fit is authorized. This amendment replaces the
rev3 all-source/all-size mode cross-product; it preserves all 61 qualities,
five native codecs and 20 registered sizes. It changes no shipping gate.

The dense-size panel uses frozen native baseline configurations. The separate
eight-source configuration panel covers resolved active native alternatives
and compatible baselines at 64 and 256, with all61q. It includes one fit and
one selection source per class, chosen from reference features only. Deduplicate
overlapping logical baseline configurations without counting their preferences
twice. Nonbaseline knob effects at large sizes, as well as multi-knob
interactions, are untested; small-size coverage cannot qualify either domain.

Baseline source counts below are requested/selected before teacher scoring.
At sizes through768, keep all200 requested sources and every reachable source.
The only omission there is structural:11 sources cannot reach768 without
upscaling. At larger sizes, use stratified centroid-nearest sampling within
the frozen content/split strata. The fast-codec column applies independently
to JPEG, WebP, JXL and the native SVT AVIF port.

|Longest edge|Reachable of200|rav1e full|Other codec full|rav1e fallback|Other codec fallback|Measured cost basis before bulk|
|---:|---:|---:|---:|---:|---:|---|
|32|200|200|200|200|200|encode medians 0.004561/0.013575/0.005916/0.003535/0.008481; unique-cell medians 0.096915/0.064877/0.061886/0.028470/0.047296|
|40|200|200|200|200|200|encode medians 0.003013/0.019645/0.005797/0.003194/0.008842; unique-cell medians 0.038882/0.060958/0.059290/0.026258/0.043661|
|48|200|200|200|200|200|encode medians 0.008365/0.034666/0.007863/0.003087/0.008657; unique-cell medians 0.069768/0.083693/0.091291/0.023798/0.045065|
|64|200|200|200|200|200|encode medians 0.003014/0.050174/0.006867/0.003600/0.009752; unique-cell medians 0.024809/0.095112/0.047290/0.030485/0.048474|
|80|200|200|200|200|200|encode medians 0.004675/0.070003/0.012614/0.003677/0.010488; unique-cell medians 0.040252/0.120458/0.079655/0.036854/0.049460|
|96|200|200|200|200|200|encode medians 0.004607/0.108335/0.011908/0.004230/0.011059; unique-cell medians 0.040950/0.168301/0.063605/0.030019/0.050975|
|128|200|200|200|200|200|encode medians 0.004315/0.157374/0.021288/0.004637/0.014353; unique-cell medians 0.057579/0.222593/0.084960/0.037277/0.065169|
|160|200|200|200|200|200|encode medians 0.004087/0.277186/0.028007/0.005878/0.018390; unique-cell medians 0.041591/0.383357/0.092301/0.058473/0.077346|
|192|200|200|200|200|200|encode medians 0.005533/0.334103/0.032555/0.007122/0.018429; unique-cell medians 0.166131/0.425012/0.097249/0.077149/0.095658|
|256|200|200|200|200|200|encode medians 0.006184/0.504239/0.064404/0.007677/0.024871; unique-cell medians 0.078795/0.664088/0.182640/0.069504/0.098467|
|320|200|200|200|200|200|encode medians 0.006174/0.924121/0.093953/0.012941/0.039760; unique-cell medians 0.088030/1.183969/0.213518/0.135859/0.176062|
|384|200|200|200|200|200|encode medians 0.008037/1.241351/0.120784/0.012398/0.046233; unique-cell medians 0.123344/1.528536/0.263969/0.133217/0.211946|
|512|200|200|200|200|200|encode medians 0.011303/2.084153/0.205835/0.020849/0.077573; unique-cell medians 0.177394/2.575877/0.441237/0.219730/0.274000|
|640|200|200|200|200|200|encode medians 0.018317/3.016449/0.292578/0.027653/0.112142; unique-cell medians 0.390412/3.854734/0.587637/0.321149/0.427055|
|768|189|189|189|189|189|encode medians 0.021528/4.348998/0.383989/0.054767/0.155276; unique-cell medians 0.393386/5.441512/0.859281/0.653485/0.542643|
|1024|186|40|80|8|20|encode medians 0.036413/7.458310/0.516240/0.059912/0.275367; unique-cell medians 0.686423/9.735817/1.450641/0.803038/0.990319|
|1280|146|20|40|8|20|encode medians 0.049936/11.683835/0.802853/0.089413/0.408583; unique-cell medians 0.989209/14.455641/2.193507/1.351896/1.396144|
|1536|129|8|20|8|12|encode medians 0.068078/15.634002/1.098092/0.135175/0.577065; unique-cell medians 1.572187/20.000195/2.831455/2.018131/1.985772|
|2048|121|8|20|8|12|encode medians 0.125601/26.937547/1.583742/0.208981/0.834037; unique-cell medians 2.931062/33.754656/4.809152/3.013305/3.750966|
|4096|33|5|16|5|6|encode medians 0.367785/92.559851/3.129097/0.754497/3.163048; unique-cell medians 10.128724/138.106744/13.493125/12.859013/14.112359|

For1024, full rav1e quotas per class are8fit/2selection, other codecs16/4.
For1280 they are4/1 and8/2. For1536/2048 they are1/1 and4/1.
Fallback1024/1280 quotas are1/1 for rav1e and4/1 for other codecs;
fallback1536/2048 other codecs use2/1. At4096 the full fast allocation is
photo6fit/1selection, lineart2fit, mixed5fit/2selection; its fallback is
photo1fit/1selection, lineart1fit, mixed1fit/2selection. Rav1e uses one
reference per available content/split stratum. There are no naturally reachable
screen sources or lineart-selection sources at4096. These are unavailable,
not passes or replacements. Retain the three available selection references
in the fast panel. Report references and unique families; reduced panels do
not inherit the full cohort's precision or per-class holdout claims.

Freeze exact IDs, centroid feature/layout hashes, tie breaks, native baseline
and active-knob coverage, structural availability mask, normalized reference
hashes and both source-budget manifests before bulk. Cluster on reference
embeddings only (same standardization/Lloyd/n_init4/seed36 owner), retain
singletons, and choose fallback centroids within each full subset so panels
are nested. No teacher or model value chooses a reference.

The mandatory fallback matrix runs first, in complete whole61q ladders; full
source additions run only after it completes. At the six-hour boundary, use
only a wholly completed predeclared matrix. If additions are incomplete, the
TRAIN view uses exactly the frozen fallback matrix, without cherry-picking
completed additions; all partial artifacts remain retained and reported. An
incomplete mandatory matrix cannot qualify READY. No q thinning, missing-size
substitution or selection of fastest-finished references is allowed. The
required-stratum positive-support guard uses the frozen reachable mask; a
reachable required stratum with no consensus preferences remains blocking.

The table records observed seconds in JPEG/rav1e/SVT/WebP/JXL order. Encode
medians use five actual q0/1/20/60/100 commands on source1520 at every size;
unique complete-cell medians exclude encoded-byte aliases and include native
encode/decode/scalars/maps/features, inspection and storage. The source is
3072x4096 at longest edge4096; it is not a square. Retain command-level samples
and sample counts in MEASURED_STAGE_COSTS-v2.json. No full-grid forecast is
claimed. V4 alias cell-total fields inherited the original cell timing and are
invalid for summing ladders; these fields are excluded. The bulk producer
records actual alias encode/conversion cost and complete ladder wall time. Do not extrapolate
a fit cost. Preflight/build time is recorded separately from the six-hour bulk
clock; integration/READY follows generation. The initial2,440-cell subset and
all earlier pilot renders are diagnostic, never a substitute for bulk coverage.

Storage admission uses a bounded local staging area with the20GiB free-space
floor. Native bytes/maps/pixels live in a content-addressed bundle store written
from a capped Docker storage process directly to a NAS array disk; local
metadata binds each encoded SHA/member, both map SHAs/pixel-metric keys and
producer build. Zenfleet owns declaration, scheduling, gap/ledger and retries.
The storage process does not claim compute work. Verify the first persisted
producer bundle end-to-end before scaling. No artifact is discarded to meet
the budget; missing/partial work remains explicit.

The E34 N binding is the plain --nonneg-distance K=1 owner, not the separate
alpha head. The private E36 build/schema leaves normal builds and TV0 behavior
unchanged. Every50 loss-bearing primary pairs within an epoch, sample32 pairs
uniformly with replacement using an independent TV RNG. With frozen
mean_pool_weight=sum_pool(w)/N_pool, accumulate
lambda*sum_batch(w*hinge)/(32*mean_pool_weight) into the existing pre-update
primary gradients. Include inactive hinges in normalization. Keep rawmargin.05,
codec weights2/1 and all q-band weights1. Add no TV-only Adam step, penalty or
projection. Admit actual fused/unfused gradient consumption with non-training
oracles; do not change control optimizations. Strict external-pool loading,
base-scaler isolation and zero dropped pairs remain mandatory. No new public
Rust type/field/signature is needed.


### Cost-driven mandatory 4096 amendment, before bulk

The mandatory fast-codec subset at4096 is reduced from10 to6 references;
full remains16 and rav1e remains5. The six contain one fit and one selection
photo, one fit line-art source, one fit and both available selection mixed
sources. Choose fit centroids inside the previously frozen ten-source subset,
using the same reference-only owner, seed and lexical tie break. Retain all
three available selection sources and every available content/split stratum.
No teacher/model value chooses IDs. All other source counts, all61q and all20
sizes are unchanged. Archive the ten-source proposal and its normalization;
write a new budget/reference-index version, never replace pinned inputs.

Measured motivation is the five-q native stage census above, plus the actual
4096x4096 source2018 witness (q0/60/100). Cached-alias totals are excluded;
the earlier quoted SVT/JXL five-cell sums were inherited metadata and are
withdrawn. The true-square unique complete-cell medians are JPEG15.099442s,
rav1e106.481567s, SVT18.445593s, WebP17.630070s and JXL18.578682s.
These measurements motivate the prospective six-source minimum; they do not
establish full-grid feasibility. A six-hour actual bulk deadline remains
binding; an incomplete mandatory matrix cannot qualify READY.

Deduplicate each unordered physical reference/encoded-endpoint pair globally.
If codec/mode aliases propose the same direction, retain the larger registered
codec weight, then lexical codec/mode/q metadata as the deterministic tie break;
never teacher magnitude. Opposite directions for one physical edge refuse
admission. Report raw proposals and unique effective support separately.
Private TV scaling can be computed on sampled compact rows using the frozen
base scaler, with the identical f64 arithmetic, avoiding a second full-pool
standardized copy. This changes no objective, scaler fit or sampler stream.

### Storage transport repair within the original bulk window

The first v5 runtime passed its five-codec artifact readback, then concurrent
cold SSH starts caused connection resets. Failed warm clients retained dead
pipes. The allocation was stopped after110seconds; its outputs, native failure
ledgers and partial artifacts are retained as diagnostics and excluded from
both qualifying pools. No native pixel, score, feature or codec defect was
observed in that failure.

The repaired client paces connection launches at most four per second and
permits three sends of the same immutable hash-bound object after transport
failure. This is storage transport, not a second job-claim system. Bad hashes
or semantic acknowledgements still refuse. Worker error frames use its native
classified protocol. The storage server, numerical producers, source budgets,
qualities, configurations and resources remain pinned. Admit28 concurrent
readback connections and the five-codec first producer before the repaired
runtime's complete matrix. Both native job attempts and all failed outputs
remain visible; no failed ledger row is rewritten or removed.

Version6 uses new runtime-bound native IDs consistently for the complete
predeclared matrix. Its versioned clock inherits the original v5 start and
six-hour deadline exactly. The controller verifies unchanged scientific
views after removing only their runtime hash, and unchanged native numerical,
source, resource and budget bindings. Initial v5 files remain immutable. This
repair grants no extra generation time and changes no qualifying subset.


### Prospective integration clarification (before fits)

TRAIN selection pools unique physical edges across the registered strata,
using the frozen codec weights, with equal weight for each of the three
full-data seeds. For plain N, raw(x)=pin-g(x) is higher-is-better. Use delta=raw(better)-raw(worse).
The raw-margin fraction uses delta>=0.05; strict order uses delta>0. Ties then choose the smaller lambda. These definitions are fixed
before any E36 model exists. An unavailable full-data slot needs terminal native
ledger evidence; pending/gap slots are not failures. A treatment needs all three
pilots and the fresh matched control. Both original human-assessment hypotheses
remain in the fixed Holm family, including unavailable arms.

The native E36 program package binds actual trainer/protocol hashes and the
exact same frozen pool and family mask for all three arms. Its private harvest
adapter verifies the metadata-only projected child keys/declaration and zero
changed targets, then reuses the existing checkpoint identity, final epoch,
seed, row-count and unchanged numeric-weight gates. The inherited86400s cell
wall cap is a conservative limit, not a measured E36 runtime. No E36 training
smoke is admitted. The code is preparation; actual pool/build/package admission
must pass before READY, and READY does not release fits.


Coordinator clarification (2026-10-17 UTC): the no-human-ranks branch applies to
TRAIN selection. No cross-reference rank guard or threshold is added to that
selector. Existing exact-identity and native finite-forward checks remain;
the selector invokes the frozen identity owner and derives pass from its
complete bit-level evidence, rather than trusting a supplied aggregate boolean.
Cross-reference ranking remains protected at human assessment by the registered
noninferiority test, each-dataset point delta>=-0.005 and W2 delta>-2SE.

### Prospective assessment provenance closure (2026-10-17, before fits)

The paired assessment artifact is admitted only after the global TRAIN-only
selection locks and all source-withheld checkpoints pass native qualification.
The original E34 metadata owner freezes source-table, manifest and ordered-key
hashes and reference/codec IDs without decoding targets. Inside the bound
assessment executor, the existing E34 score-unit forward is replayed with the
actual qualified checkpoint: `BakeScorer::ensemble` and
`score_features(row, 0, 0, None)`. Predictions must match every native f64 bit in
source order; target, reference, codec and orientation must match the frozen
original source. Forward receipts bind checkpoint, source/key and output hashes
to the pinned assessment binary. Swapped models or rows refuse before a
statistical decision. This changes no statistic, threshold or selection rule.

### Coordinator-authorized frozen-time budget exclusion (2026-10-17)

Before any E36 fit, the coordinator authorized excluding the second cooperative
freeze from the generation budget. The original deadline is
2026-10-17T11:48:19.9591377Z. The scope journal records the exact freeze start
as2026-10-17T07:31:48.642139Z; the progress log's later window-open entry is not
the start clock. The exclusion ends at the journal-confirmed actual generation
thaw. Record both timestamps, integer nanoseconds of excluded time, and the
resulting exact deadline in the immutable amendment before fitting. Preserve
the original clocks and transport-equivalence evidence byte for byte; the new
clock references them and the freeze/thaw evidence by hash. No additional time
is granted for other intervals by this amendment.

This is a budget-only amendment, authorized before any model exists. It changes
no source or family membership, quality/size/codec grid, native numerical
producer, teacher margin, pair deduplication, loss, lambda family, selector,
assessment test or guard. Extend the live outer and per-ladder deadlines by the
same measured frozen interval; retaining an expired inner timeout would defeat
the approved exclusion. All predeclared mandatory matrix and support requirements
remain required. Freeze-spanning performance timings remain contaminated and
cannot replace the prebulk cost census. This amendment grants no fit execution.

Exact budget-only amendment: journal freeze 2026-10-17T07:31:48.642139+00:00, actual native-generation thaw 2026-10-17T10:13:45.193356+00:00; excluded 9716551217000ns (9716.551217s). Original deadline 2026-10-17T11:48:19.959138+00:00; effective deadline 2026-10-17T14:30:16.510355+00:00. Amendment SHA256 a888cc9133ae409a932d36a66c047efc1deb37a206ab0a9b429d87a1c6aa4f9c. No scientific change; no fit has run.

Nanosecond-precise clock notation (supersedes rounded datetime display above): original deadline 2026-10-17T11:48:19.9591377Z; freeze 2026-10-17T07:31:48.642139Z; actual native-generation thaw 2026-10-17T10:13:45.193356Z; excluded 9716551217000ns =9716.551217s; effective deadline 2026-10-17T14:30:16.5103547Z. Integer-nanosecond receipt is authoritative.

### Optional whole-phase tower transport (before any E36 fit)

The already-declared468-view additions phase may run as one exclusive native
zenfleet-worker pass on tower, independently of dev's mandatory15342-view pass.
No manifest splitting, custom claims/retries or partial-row selection is allowed.
Both phases retain the same amended deadline and original full/fallback rule.
Complete timely evidence may be harvested after the deadline; qualification
checks native completion times and restored original output-file timestamps.
It must not depend on the wall time at which the controller returns.

The prospective resource cap is16GiB Docker memory,12GiB native packing budget
and4slots; a first-producer diagnostic uses8GiB/1slot. Pin the bulk container to
cpus0-7,16-23 (observed eight physical cores), cpu-shares256. Admit only after
actual competing heavy allocations drain and at least12GiB available memory
remains beyond the cap. The shared storage container is a separate allocation.
Every native descendant stays inside the bound Docker limits and run-heavy.
These operational limits change no source count, size, q, mode or teacher.

Use the ready image digest965b72e456d18e54915f25f124a4620ec7bd7702c329d38a5105fb4145761beb,
with the exact frozen native executables, oracle inspection tools, ELF libraries,
Pillow wheel files, references and runtime. Its Python3.14.4 differs from dev's
3.14.7. Require the original frozen five first-producer views to match encoded
and decoded hashes, consumed-feature f32 bits, every native scalar f64 bit and
map hashes before scaling. That bounded check does not establish universal
cross-host parity. No tower numerical parity or generation result is claimed yet.

Before tower additions start, hold the dev controller at a verified pre-tail
frame, seal the exact container/image/invocation/resources/input/clock/parity
ownership binding, then start that container before releasing the controller.
An unsuccessful start keeps the ownership intent and cannot start a dev copy.
The wait adapter releases the dev paper allocation while only tower computes,
then reacquires it before copying and qualifying evidence. Final matrix and
data-packet admission retain the complete transfer proof. No fit is authorized.

The actual200-source by27-reference current-panel native exclusion replay passed
on2026-10-17 with zero strict flags at the existing threshold10 and native hash
era. Cohort SHA7504bfb8f16ffd0b6f9567b831d39aa65d01819402c7857e082449286f1cee11,
panel manifest SHAe9b21e153a7d22c59e5e6341f9e7f35d0050a5b55882558f62c808c4dbae56df,
native helper SHA63323e29a4b840b17883415b49d9ae96da8bb887defa0ba30e1a08efec36d7e4.
This is the registered namespace/family plus pixel-screen admission, with its
existing lineage limits; it does not recover unknown cross-corpus parentage.

## Budget-only amendment, 2026-10-17T12:33:01.546807+00:00

The coordinator authorized extending generation to **2026-10-17T17:00:00Z** before any E36 fit, based on `finish-projection-v1.json` (SHA256 b360bfe831b1c27426ed280c4d8f8bd67cec1814cd7a49c310e6519a59d150bc), whose resource planning model projected16:13:22.788314Z. This supersedes14:30:16.5103547Z; all original clocks and the exact earlier frozen-time amendment remain preserved. Matrix, registered source counts,61qualities,20sizes, five native codecs, teachers, selection and assessment remain unchanged. No additional frozen interval is excluded. Wider native scheduling retains24 concurrent views within a38GiB run-heavy scope and measured size-specific reservations, checking live RAM minus8GiB. The operational disk floor is raised to40GiB; completed views are mirrored progressively as needed. This grants generation resources only; E36 fits remain prohibited.

Private effective-clock SHA256 c77e5e094b780115e86469642e070cca6925db6123cf3005cb36817efbfcc55b; budget amendment SHA256 2434fa5575a35584f0060ec062b760e5f4595db0601fae19ce03dea6cff6cc0d. The single-host scheduling mechanism retains native job identities and publishers; complete native ledger and streamed census admission remain mandatory.
