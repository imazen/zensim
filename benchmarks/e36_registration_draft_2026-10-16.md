# E36 draft — native TRAIN ladders for core ordering

Revision 4 budget amendment, before any E36 fit. Preparation permits native data generation and
non-training checks only. No optimizer step, first-epoch smoke, fleet claim or
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
lambda, aggregated over registered strata and three full-data seeds. Identity,
finite and cross-reference guards use unchanged existing tests/thresholds.
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
|32|200|200|200|200|200|five-q native source1520 receipt|
|40|200|200|200|200|200|per-size preflight pending|
|48|200|200|200|200|200|per-size preflight pending|
|64|200|200|200|200|200|per-size preflight pending|
|80|200|200|200|200|200|per-size preflight pending|
|96|200|200|200|200|200|per-size preflight pending|
|128|200|200|200|200|200|per-size preflight pending|
|160|200|200|200|200|200|per-size preflight pending|
|192|200|200|200|200|200|per-size preflight pending|
|256|200|200|200|200|200|five-q native source1520 receipt|
|320|200|200|200|200|200|per-size preflight pending|
|384|200|200|200|200|200|per-size preflight pending|
|512|200|200|200|200|200|per-size preflight pending|
|640|200|200|200|200|200|per-size preflight pending|
|768|189|189|189|189|189|per-size preflight pending|
|1024|186|40|80|8|20|per-size preflight pending|
|1280|146|20|40|8|20|per-size preflight pending|
|1536|129|8|20|8|12|per-size preflight pending|
|2048|121|8|20|8|12|per-size preflight pending|
|4096|33|5|16|5|10|five-q native source1520 receipt; square witness pending|

For1024, full rav1e quotas per class are8fit/2selection, other codecs16/4.
For1280 they are4/1 and8/2. For1536/2048 they are1/1 and4/1.
Fallback1024/1280 quotas are1/1 for rav1e and4/1 for other codecs;
fallback1536/2048 other codecs use2/1. At4096 the full fast allocation is
photo6fit/1selection, lineart2fit, mixed5fit/2selection; its fallback is
photo3fit/1selection, lineart1fit, mixed3fit/2selection. Rav1e uses one
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

Before bulk, fill every pending table cost basis from bounded native preflight
receipts, retain per-size/codec/configuration timings and actual byte/RSS
observations, and distinguish forecasts from measurements. Do not extrapolate
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
