# Feature/scale capability study — preregistered September 13, 2026

Status: completed, 162 fits; no model qualified. This follows the September
13 sampling and coarse-v1-pool studies. Their 264 JXL proxy pairs cannot establish
feature capability. Later August 29 split corrections override July recipes.

The existing LAN gallery service serves the chart, comparison table, full
report and downloadable measurements at
`/zensim/reports/feature-ceiling-2026-09-13/index.html` (port 3300).

## Questions and fixed interventions

Can fine luma plus cheap multiscale features recover the information in expensive
features at all scales? Does retaining expensive features only at 1/4 and 1/8
resolution lose human quality, modern-codec response, or corruption detection?
Compare nine frozen read sets, using native box pyramids and fresh formula Rev3:
basic228; fine-Y190; full372; basic228+coarse-v2; fine-Y190+coarse-v2;
fine-Y190+half/coarse-v2; basic228+all-v2; full944; full944 without scale0.
Here v2 includes append/append2. The JSON records exact canonical IDs. No pooled
feature is assumed to save compute merely because it was removed from a model.
No new feature arithmetic or fractional-plane contract is introduced in this study.

## Data and separation

* Human panel: all 5,000 admitted KADID train rows (40 references, all 25
  distortions and five severities), all 3,000 TID rows as train-only supplemental
  human supervision, and all 3,125 KADID selection rows (25 distinct references).
  Eight KADID training references, chosen by a fixed hash, supply checkpoint dev;
  the other 32 plus TID train. KADID terminal and all T0 remain untouched.
  KADID quality=(DMOS-1)*25; TID quality=MOS*100/9. Dataset affine mappings do not
  establish a perceptually uniform product dial. Report TID training residuals
  separately; never use TID as an evaluation/ranking surface.
* Codec panel: admitted September 5 32-source training anchor, JPEG/WebP/AVIF-SVT/
  JXL. Five approximately equally spaced distinct *knob* settings per source and
  codec, including both endpoints, plus identities. Original dimensions and
  signed SSIMULACRA2 labels preserved. Current admitted source families hashed
  18 fit/6 dev/5 test (18/6/6 reference origins);
  all are inner development splits of T2. This is proxy recoverability, not
  independent human validation or evidence of useful codec targeting.
  Pre-extraction inventory correction: existing JXL decode pairs omit origins
  6602 and 6604, so that codec has 30 sources/150 sampled outputs. Other codecs
  initially retained 32/160. Before fitting, the September 8 family manifest
  additionally excludes validation-family origins 8414/8434 from every codec.
  Origins 7004/7058 are one plot family and must remain together. Final codec
  panel: 30 origins/29 families, 150 JPEG, 150 WebP, 150 AVIF, 140 JXL and 30
  identities = 620 rows. This is an available-output study, not a codec
  reachability claim. The older digit-based compliance check alone missed this.
* Corruption panel: all deduplicated September 8 train-origin records across
  13 families, including noise, aliasing, channels, blocks, geometry, composites,
  real bugs, and honest codec outputs. Preserve fixed eight fit origins and four
  calibration origins; divide the latter into two dev/two inner test origins.
  The eight validation origins remain untouched. Duplicate pixel/source pairs
  collapse; inert corruptions retain their supplied negative labels. Balanced
  honest/corrupt MSE training groups; output is a diagnostic 0/100 class score,
  not a calibrated quality score. Report class/family errors and false alarms.

All feature values are freshly extracted through canonical native decoding and
BakeScorer. Old parquets supply split membership/labels only. Record input byte
hashes, row IDs, reference families, producer and tool hashes, recipe/source diff,
feature identity and revision. Check split compliance before fitting. No partial
extraction is admitted. Pixel-cache audits use final fitted Rust bakes.

## Capacity, sampling and decision protocol

Primary matrix: nine layouts x three independent paired seeds (5101/5103/5107)
x three tasks at H128, 160 epochs, 8192 pairs/epoch, fixed regularization and
explicitly disabled early stopping. Feature width is not a hyperparameter proxy.
Basic228 and full944 also run at H32 and H256, and H128 on half the fitting
references (fixed hash, all distortion/quality rows retained per reference).
Initialization seeds equal the recorded seed; sampling seed is seed+10000.
Checkpoint selection uses only dev. Three seeds are development evidence; no
release finalist is selected or qualified here. Frozen bakes are scored through
Rust BakeScorer; canonical Rust panel owns rank/correlation/error statistics.

A capability limit is **not established** when capacity or additional content
still materially improves results. Flag >0.01 signed SROCC or >5% raw MAE change
between controls as unresolved capacity/data dependence. A coarse layout is a
candidate for further work if median signed SROCC is within 0.01 and raw MAE
within 5% of full944, without a >10% relative per-family MAE regression (for
near-zero baselines use a one-point absolute allowance). Report all seeds and
families, not only a pooled winner. These are engineering screening margins,
not significance tests; seed spread is not a confidence interval over sources.

Spatial follow-up uses actual block repairs on the existing manifest, explicitly
reporting unsupported features; RB global-swap spatial correlation is diagnostic,
not a promotion bar. This checks attribution mechanisms, not native codec RD.
Quiet-machine Rust extraction timings follow fitting, with same images, threads,
binary and repeat counts. Unsupported spatial families and unchanged extraction
work remain limitations. No mathematical upper bound, universal feature ceiling,
shippable model, HDR qualification or target-loop improvement may be claimed.

Each cached-table fit/eval should be timed against the five-minute development
objective. Initial corpus admission/extraction and the complete matrix have a
separate preparation/campaign cost; do not disguise that as a five-minute study.

Pre-fit execution corrections: preserve TID filename case; include both `q`
and the supplemental knob tuple in codec joins/deduplication; register the fresh
944 producer before training admission. Early preflights and their failures are
retained outside git. No completed fit used those preliminary tables. Final
inventory is 19,958 pairs: 11,125 human, 620 codec, 8,213 corruption.

## Bounded follow-up registered during execution, before these fits

Source inspection showed `--log-every` also determines checkpoint-selection
cadence. After the main 135 fits, repeat basic228/full944 at H128 for the first
32 epochs, checking every epoch, with the same three seed pairs and all three
tasks (18 fits). The fixed 50-epoch learning-rate cycle preserves the early
training trajectory. This checks missed early optima; it does not silently
replace the main matrix. Record both outcomes and any remaining limitation.

Also run the **previously registered coarse262** layout (fine-Y190 plus v1
masked/IW at scales 2/3) with the full H128 recipe and three seeds on all three
tasks (nine fits). Its earlier ~3% extraction premium deserves a representative
quality check alongside the new v2 candidates. No feature search is performed.
Total planned completed fits including these controls: **162**.

Reporting correction: the legacy Rust batch panel's `mae` is logistic-remapped
on evaluation rows. Preserve that output, but use the explicit new
`--raw-errors`/`mae_raw` output for every raw-error comparison in this study.
The added mode shares the existing Rust MAE calculation; all legacy fields
were checked byte-for-byte against the prior binary on the three panels.

## Results

**We have not established a feature ceiling.** Under the fixed MLP recipe,
more training content still improves errors, and checkpoint selection changes
what a feature set appears capable of. Increasing H128 to H256 did not improve
median raw error by the registered 5% bar on any control. This tests one model
family; it cannot exclude a stronger optimizer, loss, regularization or model.

All 162 final Rust bakes passed **2,322 native pixel/cache audits**, with maximum
consumed-feature absolute difference **0.0**. Training used fresh Rev3 values;
old wide tables were not relabelled. Complete evidence lives at
`~/work/zensim-validation-2026-09-13/ceiling/final/`: `INPUTS.json`, `_MANIFEST.json`,
`RESULT.json`, `audits/RESULT.json`, `SUMMARY.json`, bakes, keyed scores and logs.
The compact tracked summary is `feature_ceiling_2026-09-13.results.json`.
`CODEC_PROVENANCE.json` records the retained native encoder binary and available
historical codec pins; this is not a benchmark of current codec implementations.

### Scalar accuracy and actual cost

Three-seed medians, H128/full-data primary recipe. Errors are raw points on each
panel's own scale. They are **not interchangeable product quality units**.
Timings are warm `BakeScorer::compute`, one deterministic 1024×1024 pair, 15
interleaved samples per arm, pinned CPU 0 or CPUs 0–7, no concurrent training or
build. Same existing inference binary for all arms; no cross-build speedup claim.

| Layout | Human MAE | Codec-proxy MAE | Corruption class MAE | ST ms | 8-thread ms |
|---|---:|---:|---:|---:|---:|
| basic228 | 7.831 | 15.577 | 16.086 | 26.387 | 5.460 |
| fine-Y190 | 7.813 | 14.101 | 18.757 | 15.393 | 4.137 |
| coarse262 | 7.865 | 13.494 | 19.168 | 15.756 | 4.215 |
| full372 | 7.795 | 15.081 | 16.196 | 34.446 | 6.941 |
| basic + coarse v2, 514 | 7.704 | 14.101 | 11.853 | 51.480 | 17.502 |
| fine-Y + coarse v2, 476 | 7.699 | 13.385 | 13.021 | 51.529 | 17.517 |
| fine-Y + half/coarse v2, 619 | 7.574 | 11.660 | 9.908 | 51.481 | 17.527 |
| basic + all v2, 800 | 7.456 | 13.298 | 8.646 | 51.497 | 17.619 |
| full944 | 7.337 | 13.129 | 9.016 | 57.950 | 18.498 |
| no scale0, 708 | 7.519 | 11.255 | 9.791 | 53.094 | 17.672 |

The v2 subsets cost essentially the same: **restricting their read sets does
not yet skip the corresponding extraction work**. Coarse262 is actually cheap
(2.4% ST / 1.9% MT over fine-Y190), but its corruption behavior is poor. The
800-feature variant is promising as a simpler information reference, not an
accepted replacement: pooled accuracy hides regressions on JPEG/brightening
human distortions and honest-codec class-score calibration. No reduced layout
passes all registered family/error bars across all panels. The 619 variant
has useful codec-proxy results, but weakens several corruption families.

### Data and checkpoint controls

Full944 H128 raw error, half → full fitting content:

| Panel | Half | Full | Change |
|---|---:|---:|---:|
| Human | 8.221 | 7.337 | −10.8% |
| Codec proxy | 13.755 | 13.129 | −4.6% |
| Corruption | 14.434 | 9.016 | −37.5% |

Five of six baseline/full944 controls cross the registered data-dependence bar;
the full944 codec control does not. This is evidence within the tested recipe,
not an extrapolated learning-curve asymptote.

Frequent early checkpoints reduce basic228 human MAE **7.831 → 7.332**, and
full944 **7.337 → 7.199**. Thus the apparent human advantage of the wide feature
set shrinks substantially when the training procedure is checked. Corruption
still benefits strongly from wider features: corresponding early-checkpoint
errors are 17.585 versus 8.936. Keep these controls separate from the 160-epoch
matrix; do not silently select the better test result from each protocol.

At the fixed diagnostic class midpoint 50, full944 has **0–1 false alarms out
of 82 honest test rows**, versus **17–20/82** for basic228 and **20–27/82** for
fine-Y190. Misses are 7–19/1,145 for full944, versus 21–64 for basic228. The
800 variant has 0–1 false alarms and 6–13 misses. These are counts across three
seeds, not calibrated probabilities or confidence intervals. Only two original
sources supply this inner corruption test: broader content is still essential.

### Spatial result: an independent blocker

The 756 checks performed **39,744 actual block repairs**. Excluding the 108
whole-image R/B-swap diagnostics: **131 PASS, 247 FAIL, 270 UNSUPPORTED**.
Unsupported v1 masked/IW refinements are not passes. Among the primary human
bakes, basic228 passes 13/18 checks; the 514/476/619/800 v2 layouts pass only
3/1/1/3 respectively. Coverage metadata saying a feature is implemented does
not establish useful spatial prediction.

A concrete supported failure is
`audits/human-basic_v2_coarse-h128-full-s5101--jxl_2010.json`:
M2 ≈ 0.9968, M3f ≈ 0.0749, no unsupported refinement IDs. The local linear model
can explain actual score changes, while the predicted repair map does not.
Investigate the feature-to-rectangle prediction before using these models for
codec allocation. This study measures repair consistency, not native codec RD.

### What to do next

1. Fix and validate v2 spatial repair prediction against the retained failures.
2. Implement real per-scale v2 dispatch, using the 619/800/full944 controls to
   test values, maps and measured cost. Feature-count ablation alone saves no
   v2 extraction time here.
3. Use frequent-checkpoint, cached-data H32/H128 screens; enlarge codec and
   corruption source coverage before declaring information limits. Keep the
   broad feature reference and per-family panels during that work.
4. Revisit feature removal only when accuracy, family tails, spatial behavior
   and actual work reduction agree. Do not globally drop fine X/B on the basis
   of the human-quality panel alone.

Preparation of the corrected corpus took about two minutes. Main H128 fits
were 83–146 seconds; H32 controls were 23–27 seconds. H256 controls took several
minutes and exceed the five-minute development objective. The initial full
matrix took about 45 minutes of capped parallel fitting, followed by the
248-second bounded follow-up and 45-second pixel/spatial audit. These campaign
costs are separate from an individual cached development fit. No CI was awaited.

Validation: 16 Rust panel tests; 10 registry/compatibility tests; CI-exact clippy;
script lint; explicit native producer and final-bake audits; byte-identical
legacy panel outputs on all three datasets. The registry test also corrected
an older assumption that every 372-wide producer contains fine X/B: the earlier
Y-only sampling producers intentionally do not.

Limits: SDR only; native 1×/½/¼/⅛ box pyramid, no new fractional-plane sweep;
metric-proxy codec labels; 30 codec origins/29 families and 12 corruption
origins; fixed MLP model family and losses; no terminal/T0 scoring, no native
1/2/3-shot targeting trial, no HDR or release qualification. The first completed
baseline fit used the initial eight-thread environment; the parallel campaign
used one thread per fit. All arithmetic/tool/input identities are retained.
