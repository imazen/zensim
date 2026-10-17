# E36 draft — native TRAIN ladders for core ordering

Draft revision 2 (review amendments frozen before generation). No E36 fit, development selection or assessment is authorized
by this preparation. The coordinator requires an independent Astra review,
data generation on dev, measured smoke receipts and E36_READY before any fit.
The prior 97-edge diagnosis informed this design; G-ADDR replay remains
exploratory and cannot qualify a model. All adopted gate numbers stay fixed.

## 1. Fixed hypothesis and controls

H1: teacher-resolved native low-quality ladder preferences improve ordering
without degrading the four-source human ranking or engineering behavior.
Same-final-epoch f32 and curve controls preserved all 97 inversions, so change
the core loss, not K. Primary family C is diff+frag-820, N head, H128, Rev5,
exact E34 base recipe (`sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1__N`), 120
epochs, 50,000 rank-pair draws per epoch, final epoch 119 selected. Preserve
original rank groups, group weights, scaling, feature and fx1 declarations,
initialization/sample seeds and quantize-before-TRAIN-calibration pipeline.
Use fresh matched controls with identical data access/order and TV disabled.
Do not run any first-epoch training smoke in this preparation.

The primary study has TV weights lambda={1,10} and raw-output margin 0.05,
chosen here before generation or fits. No search outside that set. Freeze the
exact CLI translation after native-owner admission; no ignored option counts
as a treatment. Positive margin is an anti-collapse term, not a JOD margin.
Teacher-side improvements are defined below in each teacher's own units.

## 2. References, disjointness and axes

Canonical imazen-26 TRAIN under both origin and family split owners only.
Reuse the E34 canonical inventory/admission as a starting index, not a blanket
permission: recheck source SHA, current corpus revision, family closure and
all G-ADDR, G-STEER, human evaluation and previously used development image
exclusions. Build exclusions only from admitted label-free key/reference
manifests. Never open sealed/protected labels or discover terminal tables.
Apply the existing native dHash owner and conservative family closure at the
unchanged distance <=10. Quarantine unresolved parent/crop lineage relevant to an excluded family. Hash the exclusion closure, admitted source hashes and all key manifests before selection. TRAIN-selection families must be absent from optimization, preprocessing and calibration inputs, including full-data fits.

Select >=50 distinct sources in each semantic class. Freeze the reference-only
canonical collection rubric: origin IDs 1000–4999 photo, 5000–5999 mixed,
6000–7999 line art, 8000–8999 screen; other collections are outside this
primary matrix. These are collection strata, not human perceptual labels.
Tiny is an overlapping presented-size tag: both dimensions <=64 pixels,
from >=50 eligible TRAIN source families, downsampled without upscaling.
Report original dimensions separately; do not claim >=50 originally tiny
sources. Astra explicitly accepted this interpretation and the two-teacher
amendment below before generation.

Keep every rendition of a family in one split. Reserve >=20% of selected
families and references in every class for TRAIN-internal selection. Use the
existing native reference-feature embedding, population z-score within each
class/split (zero-variance scale 1), Lloyd k-means, n_init=4, random_state=36,
K=40 fit and K=10 selection, centroid-nearest representative, lexical origin
ID as tie-break, retaining singleton clusters. Assign families by SHA256 of
`e36:family:<family>` in lexical digest order; reserve the first ceil(0.2*N)
families per class, increasing the reservation until >=10 candidate sources
are available. Cross-class families use their global assignment. Recheck the
>=20% selected-family condition and block generation on any shortfall. Freeze
all manifests and cluster sizes/normalization before encoding.
Use native imazen encoder paths only: zenavif/zenravif (zenrav1e), the imazen
SVT port, zenjpeg, zenwebp and native JXL. No third-party encoder or teacher
binary writes a training variant or target. Q={0..20 step1,22..100 step2},
61 values, every supported production mode/CPU knob via the native plan-cell
bridge. Codec quality and JXL distance orientation must be explicit; retain
actual resolved settings, not just a nominal q. Dense longest-edge sizes are
32,40,48,64,80,96,128,160,192,256,320,384,512,640,768,1024,1280,1536,2048,4096.
Preserve aspect ratio, use pinned imazen Lanczos3 resampling, never upscale.
Record reachability for every source/size; every refused q/mode/size is visible.
Do not silently truncate size, quality, codec or content axes for budget.

## 3. Teachers, native artifacts and preference rule

Active target teachers are zenmetrics' own butteraugli pnorm3 and CVVDP
standard_fhd, with native maps from its existing diffmap executor. SSIMULACRA2
and served B are excluded from target generation. This prospective choice
uses the owner's any-metric grant and Astra's explicit agreement; it neither
alters shipping judges nor substitutes a proxy for the absent SSIM2 map API.
Freeze source/dependency builds, scalar precision and CVVDP display/resampling
contract. Both teachers receive identical native RGB8 SDR reference/distorted
pixels; non-sRGB or unresolved color metadata blocks admission until the native
color owner converts it explicitly.

For endpoints a,b orient scores as higher-is-better: CVVDP JOD+, butteraugli
pnorm3−. Define delta_t = oriented_score_t(b) - oriented_score_t(a). Admit
b preferred to a only when BOTH deltas exceed 0.01 native units; admit the
reverse only when BOTH negated deltas exceed 0.01. Generate unordered pairs
once at ladder-index separations {1,2,4,8,16}. Exclude/count disagreements,
ties and insufficient margins; nonfinite/missing scores refuse the cell.
Report proposals, admissions, reversed direction and each teacher's veto by
codec/content/size/q band. Bands are 0–20,22–60,62–100, assigned by the lower
nominal-q endpoint independent of preference direction. A required stratum
with no admitted preferences blocks fitting. Higher q alone is never truth.
Give rav1e and JPEG pair weight 2 and each other codec pair weight 1. All band weights are 1. Normalize by total contributing pair weight; report effective mass by codec/content/size/band. Use
the existing TV owner with explicit polarity and raw separation margin. Bind
all feature-row indices and exact concatenated group order by SHA, reject any
out-of-range index and require zero trainer-dropped pairs. Feature-only TV
input integration must preserve base-rank targets; do not synthesize human
labels to make a loader accept the new rows. Native consumed features/fx1
must match serving and the frozen E34 layout; no silent zero fill. Synthetic rows enter only the TV objective, never human-rank sampling, scaler fits, target normalization or calibration. Use an independent TV RNG so matched base-rank draws are identical. Before fitting, a non-training loader/loss check must establish keyed row retention, polarity and decreasing loss when a preferred raw gap increases. No optimizer step is allowed.

Registered loss is LE34 + lambda * sum(w * max(0, s*(better-worse)+0.05))/sum(w), s=+1 for distance and -1 for score. Native-owner CLI reduction/frequency/batch bindings remain an admission blocker until verified; the legacy band-weight option replaces the flat TV weight and cannot be assumed multiplicative. Freeze pairs/step, TV step frequency, independent RNG seed and exact flags in a prospective implementation amendment before any fit.

Encoded bytes, native decoded pixels, maps and metric variants are keyed by
hash. Manifests bind original bytes, encoder/decoder builds, metric IDs/display,
feature formula/layout, source family/split, resolved settings and ordered row
keys. Verify the first cell's actual bytes/maps/receipt before scaling. Use
zenmetrics/zenfleet's native plan, execution and artifact owners; no parallel
bespoke job/claim/retry protocol. Data generation runs only on dev under the
shared heavy flock and run-heavy. Stop below 20 GiB available on /home.
Bulk data lives under the E36 artifact root and is mirrored to tower by
manifest with random SHA verification. Capacity is measured, not extrapolated
as a completed data claim; oversized plans must expose a storage blocker or
an explicitly reviewed retention/staging strategy before execution.

## 4. Frozen future fit/selection and assessment template

No fit starts now. The prospective C manifest contains 129 possible cells:
3 arms (fresh TV0 control, lambda1, lambda10) x (four withheld human sources x
10 shared seeds + three full-data seeds). Each cell binds source/fold,
initialization seed, sampling seed, data hashes and matched-control ID before
release. These are planned counts, not completed fits. No diff-410 arm is
included without a prospective amendment and measured incremental cost.

Select one lambda globally using held-out TRAIN families: maximize the weighted
fraction satisfying raw margin 0.05, then strict-order fraction, then smaller
lambda. Aggregate all registered strata and three full-data seeds using the
same frozen pair weights. Identity/finite/cross-reference guards bind existing
unchanged tests in the gate manifest; if neither arm passes, select neither.
No assessment dataset, fold or seed chooses lambda or checkpoint. Report failed
and unselected arms; retries preserve seeds/configuration and versioned repairs
apply consistently to affected matched cells.
Assessment follows the existing E34 paired source-reference × shared-seed
statistical owner: eligibility margin 0.0065, mean-point guard -0.002, each
dataset guard -0.005 and W2 delta > -2 SE; existing signed orientation, native
W2, 10,000 resamples/type7 and Student/bootstrap decisions. Fix the complete
Holm family before any future assessment; with one selected C treatment there
is one primary one-sided alpha=0.05 noninferiority hypothesis versus its fresh control (Holm family size 1); dataset/W2 and engineering checks are conjunctive eligibility guards, and peer comparisons descriptive. Report both lambda
arms descriptively and make no confirmatory selection from those reports.
Retain the engineering gates, adopted butter steering/CVVDP scoring references,
matched ranking and matched complete-map cost/RSS. Bind each gate endpoint, threshold, implementation revision and qualification status to the unchanged adopted gate manifests; do not silently reinterpret RSS thresholds from a review gloss. Original G-ADDR panels are
exploratory regression instruments; untouched ladder confirmation belongs to
the later ship registration. Protected sources remain untouched.

## 5. Preparation exit and fleet plan

READY requires: independent review resolved, concrete native tool/source pins,
complete source/class/family admission, declared full generation grid and
coverage, first-cell artifact audit, measured encode/score/feature smoke times,
TV row-layout/polarity admission without fitting, mirrored manifests and a
fit-cell specification with measured caps and host admission still unreleased.
Report missing items as NOT READY, not a conditional pass. No fit, fleet worker
claim, worker restart or action on the protected worker host occurs here.
The coordinator alone releases fitting after E36_READY.

## Review and unresolved admission

Independent Astra review and its focused teacher/tiny agreement are archived with the private run manifest. Revision 2 resolves design choices; source/color admission, finite native grid export, build closure, measured artifact smokes, storage staging and feature-only TV mechanics still require evidence. Preparation completion does not authorize fits.

## Generation admission amendment (before any E36 encode)

The current corpus family-v2 map supplements the original E34 family owner;
require TRAIN membership in both and close all older rejections/development
families through the newer burst/crop groups. Quarantine origins missing from
the current corpus. Native source dimensions/dHash are reused only with exact
current-byte SHA checks. No missing lineage is inferred from an absent hash.

Freeze the finite native mode plan names to `modes_full` for zenjpeg, zenwebp,
zenjxl and rav1e/zenavif, and `svt_doe_main` for the SVT port. Use all 61 q
values, no budget, no compute cap and no deviation cap. These are the native
owners' finite documented mode/CPU-knob sets, not every real-valued setting.
Export resolved identities, aliases, unsupported states and source-size
reachability before bulk generation. Preserve resolved q/setting aliases.

First bounded smoke: five native codec producers at q{0,1,20,60,100} and
presented longest edges32,256,4096 from a selected TRAIN source reaching4096;
75 cells if all are admitted. Freeze actual source/dimensions after native
color probing and before encoding. A source-color refusal cannot be waived
by this amendment. Retain all bytes/pixels, native scalar variants and genuine
maps; first-cell verification precedes the remaining cells. This is a data
artifact/capacity smoke, never a fit or a source-informing calibration sweep.
Bulk execution remains blocked until the exported full-grid/staging budget
and admission evidence close. Feature-only TV integration remains a pre-fit
requirement and is not required to generate admitted data.
