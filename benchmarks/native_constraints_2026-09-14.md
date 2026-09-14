# Existing constrained heads: native behavior measured, none advances

September 14, 2026. **All three constrained profiles fail the registered TRAIN advancement screen.** They improve association between maps and their own score responses, but introduce or retain incorrect native preferences and/or lose human rank. No model is promoted.

## What was measured

No new training, encodes, extraction, calibration or source admission. Six frozen, three-member Rev3 ensembles from the existing product study are measured through the complete Rust native-replay public API: y60/H32, y60/H128 and local120/H128, each plain and nonnegative-distance. Full944 retains its existing scalar evidence; unsupported full attribution excludes it from this map comparison. The architecture was already fitted; repeating it would not answer the missing native question.

The primary comparison uses the four source-family-disjoint TRAIN development families from the earlier native-local packet, at 256-long-edge and JXL distances 1/3: eight cells, 272 bitstreams, 79 robust same-buffer peer-consensus comparisons. The eight fitting-role families are reported separately, as are the original three-family 12-cell diagnostic packet, which may overlap fitting content. No EVAL, TEST or TERMINAL segment is accessed.

Models, pixels, decoder, formula revisions and tools are hash-bound. The 812 new-packet and 408 original-packet decoded-pixel/peer rows match the frozen evidence. All 1,628 overlapping plain-model public score objects and their corresponding maps reproduce exactly. Every new model has full additive-density coverage and zero M2 failures below the .99 native diagnostic line on the eight development cells. This does not establish native allocation RD.

## Native development comparison

| Complete model | Conflicts /79 | Own-score map rank median | Cells below .70 /8 | Map vs SSIM2 median | Map vs Butteraugli quality median |
|---|---:|---:|---:|---:|---:|
| PT914U_y60_h32_plain_ens3 | 0 | 0.557 | 5 | 0.454 | 0.575 |
| PT914U_y60_h32_nonneg_ens3 | 1 | 0.581 | 5 | 0.469 | 0.637 |
| PT914U_y60_h128_plain_ens3 | 0 | 0.601 | 5 | 0.465 | 0.550 |
| PT914U_y60_h128_nonneg_ens3 | 5 | 0.709 | 4 | 0.425 | 0.499 |
| PT914U_local120_h128_plain_ens3 | 10 | 0.494 | 5 | 0.376 | 0.300 |
| PT914U_local120_h128_nonneg_ens3 | 6 | 0.743 | 2 | 0.385 | 0.472 |
| D_frozen_revision1 | 0 | 0.638 | 5 | 0.535 | 0.525 |

Own-score correlation measures internal consistency, not perceptual correctness. For y60/H128 the constrained head improves it .601→.709 while SSIM2 association falls .465→.425 and Butteraugli association falls .550→.499. Its five new preference conflicts cover photo, graphic and screen content. The H32 constraint introduces one screen conflict. Local120 removes its photo/screen conflicts but retains all six graphic conflicts.

The .70 line remains a mechanism diagnostic; no allocation gate is substituted. SSIM2 supplies codec training targets, and neither peer is human ground truth. Four development families cannot establish universal class behavior. The current gallery now displays peer-response medians beside explicitly labelled own-score columns; all correlations are existing canonical Rust outputs, aggregated over complete matching cells. Missing cells are refused rather than omitted.

## Existing scalar assessment, reused with verified hashes

The original full human/codec panels and all member scores remain immutable in product-train/assessment-panels. Human development is 1,000 KADID TRAIN rows from eight references; distorted codec development is 1,380 pairs from 23 sources, with 249 identities kept in separate and pooled panels. Below, codec rank excludes identities; the registered screen uses the pooled panel and the full report retains both.

| Ensemble | Human SROCC | Distorted codec SROCC | Human geometric out4 | Human raw residual p99 |
|---|---:|---:|---:|---:|
| PT914U_y60_h32_plain_ens3 | 0.83894 | 0.87798 | 0.30% | 64.04 |
| PT914U_y60_h32_nonneg_ens3 | 0.84885 | 0.87427 | 1.30% | 108.37 |
| PT914U_y60_h128_plain_ens3 | 0.84666 | 0.88043 | 0.20% | 75.86 |
| PT914U_y60_h128_nonneg_ens3 | 0.84031 | 0.87319 | 0.90% | 92.73 |
| PT914U_local120_h128_plain_ens3 | 0.86536 | 0.89823 | 0.70% | 74.64 |
| PT914U_local120_h128_nonneg_ens3 | 0.82918 | 0.89168 | 0.70% | 97.04 |

Y60/H32 gains .00991 human rank but adds a native conflict, raises human out4 .3%→1.3%, and worsens residual p99 64.04→108.37. Y60/H128 loses .00635 human rank and local120 loses .03618, both exceeding the fixed .005 tolerance. All three fail the registered screen without adding a retrospective gate. Full raw/scatter/clumping/saturation and percentile panels remain available; the table does not replace them or a release composite.

These models predate the sampling-window correction. Their frozen numerical measurements and paired architecture comparisons are retained, but nearby sampling seeds cannot be advertised as independent sampling replicas. This replay fits no replacements and makes no new sampling-variance claim.

## Research chronology and consequence

The July additive/MLP correction establishes that a 156-input head is not necessarily additive, and exact feature-level M2 does not prove finite or native steering. Current native results reinforce that distinction: all six pass local linearization diagnostics while several rank actual interventions incorrectly. Later attribution support supersedes the old claim that everything outside basic156 is inherently unspatializable.

The September 6 best-of-all program already implemented the nonnegative-distance architecture and reused the TV hinge owner. Its later cookbook correction limits the structural identity guarantee to zero-preserving transforms; nonnegative distance alone never established codec-floor ordering. Its measured hinge arms improved monotonicity but did not solve the floor gate. The current product study already tested the architecture with fresh Rev3 features; this native replay supplies its missing mechanism evidence. It does not justify repeating the old capacity or constraint sweep.

The July TV investigation also found wrongly ordered training pairs and mixed evaluation provenance. The September 13 TRAIN/EVAL/never-TEST ruling and current reference-level admission supersede its historical dataset-access recommendations. Narrow-band and tail statistics require their actual target support and matched peer context; a low rank alone is neither proof of a broken model nor automatically a reporting artifact.

Twenty-four of the 99 exact-project memory files now have explicit cumulative full-read receipts. The memory audit remains incomplete; large research memories, related roots and transcript chronology still require reconciliation. No claim of a complete Claude audit is made.

Before any new fit, use the retained native feature deltas and existing contribution tools to distinguish feature-response failures from head/normalization tradeoffs on these exact cases. Any proposed clean hinge experiment must demonstrate its actual source-pair order, reached loss path and fixed TRAIN-only budget first. Preserve the unqualified scalar/targeting/native-RD/corruption/runtime requirements; neither self-coherence nor mentor agreement may replace them.

Evidence: `~/work/zensim-validation-2026-09-14/native-constraints/`; [served comparison and A/B gallery](/zensim/reports/native-constraints-2026-09-14/index.html); [compact results](native_constraints_2026-09-14.results.json). Existing scalar context: [product training](product_train_2026-09-14.md); preceding local supervision: [robust-pair study](native_robust_train_2026-09-14.md). The full product goal remains active.
