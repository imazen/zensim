# Frozen Rev3 qualification — September 14, 2026

Status: **NO SHIP for these nine frozen candidates; overall qualification work
remains incomplete.** This extends the nine frozen `MT913_*` ensemble/control
models from `minimal_top_2026-09-13.md`; it does not fit or select a new model.
The production requirements in `docs/MODEL_SELECTION_SCORECARD.md` remain the
acceptance contract. Missing populations or measurements cannot pass a gate.

## Admission and chronology

The September 13 user ruling, reiterated in the September 14 assigned goal,
forbids all historical test/terminal segments. DATA_SPLITS §8 (August 29)
places CID22 gold and the T0 estate under TERMINAL despite older descriptions
of them as eval-only. This run excludes those populations. It does not resolve
the conflict by relabelling them. The historical composite requiring those
panels remains incomplete; no replacement composite is introduced.

The first additional populations are KonJND JPEG SELECT (404 references,
404 pairs, `_MANIFEST_splitv2_views_2026-08-29.json`) and KonFiG origin validation
(SRC01/03/31/45, 436 deduplicated pairs, `_MANIFEST_konfig.json`). Only these
admitted views and their original pixels may be read. KADID SELECT retains
its existing 3,125-pair native identity-aware evidence. TID remains train-only.

KonJND uses original mean PJND thresholds and round-half-up selection of JPEG
quality, preserving the corrected September 6 pairing rule. Its registered
rank statistic is magnitude SROCC, with signed correlation also reported.
KonFiG uses quality-oriented `1 - q_jnd / 3.2`; native q_jnd is retained.
Its four validation sources support a narrow JND/distortion instrument, not
an independent SSIMULACRA2-superiority claim: SSIMULACRA2 tuned on KonFiG.

## Frozen measurement contract

- Evaluate all eight uniform five-seed ensembles and the linear60 control;
  retain their exact packed bytes, member order and weights. No fitting,
  calibration, checkpoint, feature, threshold or ensemble changes from eval.
- Re-extract native Rev3 full944 features once, with explicit row identities,
  original file hashes, decoder/tool identity and formula metadata. Preserve
  native decoded identity and audit complete pixel/cache serving parity.
- Score with the existing Rust `BakeScorer`/`bake_verdict` owners. Extend the
  existing corpus registry where necessary; never create a second scorer.
- Retain full rank/panel/scatter diagnostics, raw tails, density and saturation.
  MAE is supplementary. Compare baselines only on the identical admitted rows,
  with each baseline's correct extraction era and contamination caveats.
- Add source-cluster uncertainty and per-content/codec failure analysis through
  existing assessment owners. Four-source intervals cannot imply broad coverage.
- Continue admissible near-lossless/dial, bounded 1/2/3-shot targeting, complete
  spatial and corruption qualification, and controlled p95/memory measurements.
  Prior four-image JPEG repairs do not qualify native codec spatial steering.

Private replay artifacts are under
`~/work/zensim-validation-2026-09-14/rev3-qualification/`. Preserve each command,
input/admission/model/output hash and failure disposition there. Publish actual
verdicts through `promote_fulleval.py` and regenerate/gate both gauntlet views.
No candidate is shippable until its advertised capabilities meet the unchanged
production contract.

## Measured expansion

All eleven bake comparisons now have actual Rust full verdicts on all three
admitted populations, alongside a newly scored CPU SSIMULACRA2 peer. Every
KADID prediction and rank statistic is unchanged from the earlier frozen
evaluation. The two new populations add 840 pairs. The eight successful
candidate pixel audits cover all 840; B and D each pass complete native
pixel/cache audits on all 3,965 pairs. Selected619 fails its spatial/scalar
audit on 836/840 pairs and produces no partial-success audit artifact.

| Frozen model | KADID SROCC | KonJND magnitude SROCC | KonFiG SROCC |
|---|---:|---:|---:|
| y40/H32 | .9091 | .3535 | .8242 |
| y40/H128 | .9135 | .3527 | .8436 |
| y60/H32 | .9290 | .4566 | .7977 |
| y60/H128 | .9323 | .4425 | .8151 |
| local120/H128 | .9328 | .2248 | .7981 |
| selected619/H128 | .9399 | .1329 | .8481 |
| full944/H128 | .9433 | .1776 | .8399 |
| full944/H256 | .9430 | .0823 | .8278 |
| linear60 | .8965 | .4036 | .8020 |
| Matched B, native Rev1 | .8126 | .5706 | .8191 |
| Matched D, native Rev1 | .8107 | .5883 | .7227 |
| Matched CPU SSIMULACRA2 | .8079 | .5533 | .7351 |

These candidates trade improved KADID ranking for weaker visibility-threshold
performance. The widest head is particularly weak on KonJND. These observations
do not establish a feature ceiling or justify fitting to this eval population.
KonFiG has four sources, and SSIMULACRA2 tuned on it; that column cannot support
an independent superiority claim. B/D also have historical training exposure
limitations. Their missing bake-era metadata causes strict provenance admission
to refuse them: the baseline reports retain that limitation, and every reported
prediction is independently checked against their current native Rust surface.

Source-cluster bootstrap (1,000 draws, seed 914, same draws for every model)
gives KonJND y60/H32 a 95% interval [.373, .532], versus D [.518, .650]. The
paired difference interval is [-.184, -.080]. Full944/H128's paired interval
against D is [-.496, -.321]. These are exploratory percentile intervals, not
multiple-comparison-adjusted winner declarations. The existing verdict's
row-resampled intervals remain recorded separately. Full source intervals and
complete population raw/shape diagnostics are in the
[machine-readable results](rev3_qualification_2026-09-14.results.json).

The unchanged G-OUT owner, using matched SSIMULACRA2 peer bars, finds failures
for all nine: KADID normalized raw-residual severity and KonJND outlier rate.
The range clause remains unmeasured without a declared spline range. Independently,
wide models violate the identity ceiling on distorted pairs: full944/H128
reaches 116.22 on KonJND and selected619 reaches approximately 120. This is an
actual serving defect for a target dial, not a reason to clip the report.
OR still uses corpus dispersion where per-stimulus uncertainty is unavailable.

The report previously renormalized its product composite over whichever axes
were available, turning KonJND alone into a purported product composite. The
Rust owner now emits null for an incomplete composite, records the six required
axes and missing coverage, and retains the historical partial arithmetic in a
separately named diagnostic. All eleven regenerated rank/prediction/scatter
blocks are unchanged by this correction. Promotion preserves these fields and
the board displays coverage. Historical artifacts are not rewritten.

Replay: `admit.py`, `evaluate.py`, `audit_models.py`, `evaluate_baselines.py`,
`uncertainty.py`, explicit keyed peer TSVs, and every command/result/log are in
the private artifact directory above. The initial registration is preserved as
`PROTOCOL.md` at its original SHA. Failed invocations are retained: incorrect
audit weight normalization refused before scoring; Selected619's actual map
failure; and B's missing-era strict refusal before its explicitly unqualified
native-parity baseline report. No failures were converted into gate passes.

## Remaining work under the assigned goal

Broaden admissible imazen-26 content, honest low-quality/near-lossless ladders,
and corruption controls; finish source-qualified independent judge provenance.
Complete native bounded targeting, native spatial intervention/RD, and p95/RSS
qualification. Repair feature arithmetic/serving defects using train-side
reproductions, then perform the registered compute-saving feature experiments
and train-only selection before freezing the next candidates. Recover remaining
historical research and codec integrations before replacing implementations.
No named profile/default changed. The full assigned goal remains active.
