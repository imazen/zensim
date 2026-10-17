# E36 draft — native TRAIN ladders for core ordering

Revision 3, before any E36 fit. Preparation permits native data generation and
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

The candidate finite mode matrix uses native `modes_full` plans for JPEG,
rav1e, WebP and JXL and `svt_doe_main` for SVT, with `--max-deviations 1` and
no budget/compute/q coarsening. Every active knob value needs a resolved
coverage table. Conditional values requiring companion modes need named
compatible baselines/submatrices before bulk, or remain explicit blockers.
Flags, inactive overrides and aliased values do not prove active coverage.
This main-effects design excludes systematic multi-knob interactions and makes
no claim across the full production configuration cross-product. Keep all
shipping-gate configurations unchanged. Full-cross census exports are retained.

Active teachers are zenmetrics' own butteraugli pnorm3 and CVVDP standard_fhd,
with native maps from its diffmap owner. SSIMULACRA2 and served B do not generate
targets. This prospective choice follows the owner's any-metric grant and
independent Astra agreement; shipping judges remain fixed. Pin builds, scalar
precision, color ingress and CVVDP display/resampling. Both teachers receive
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
